"""Deviations played in branch copies (cfg["branch_deviations"]); truncated ones valued by a leaf network.

A sampled learning decision deviates in a copy, not on the hand: the
branches hook draws the deviation with boosted_deviations' law (loop.py),
and the framework plays it and its argmax control (branch_control) from the
same fork on the same wall, both stopping at the learner's k-th later
decision. A stopped copy is worth the leaf network's value of the learner's
row there; a copy whose hand ended first is worth its hand score, as a
playout is. The deviation's advantage is the candidate's value less the
control's, each less the draw luck it met. Whatever happens in the hand after
the stop no longer reaches the advantage; the price is the leaf network's
error, which the paired difference partly cancels.

The leaf network sees only the legal observation (the Actor's inputs) and is
trained on the hand's own rows: under cfg["vine"] the hand never deviates, so
the rest of the hand from any row is a sample of the argmax value a leaf
needs. Two outputs, the rest of the hand's score and whether the learner
deals next (each less its draw luck), so the dealer premium and the game's
last hand (no premium) apply per row, as they do to a playout. It values a
rollout's leaves before it trains on that rollout: a control's leaf is the
very row the hand reaches k decisions later, whose realized outcome it would
otherwise have fitted.
"""

import copy

import numpy as np
import torch
from torch import nn

from .batch import make_batch


def weighted_argmax(boost):
    """The sample hook: always the argmax; the logprob slot carries the row's deviation weight.

    Rate and weight are boosted_deviations' (loop.py): a decision deviates
    with probability min(1, boost * (1 - pi(argmax))), and where that is
    capped at 1 the weight boost * (1 - pi(argmax)) keeps the state's share.
    """
    def sample(logits, mask):
        probs = torch.softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        top, best = probs.max(-1)
        leave = (1.0 - top).clamp_min(0.0)
        rate = (boost * leave).clamp(max=1.0)
        weight = (boost * leave / rate.clamp_min(1e-12)).clamp_min(1.0)
        return best, weight.log()

    sample.capturable = True
    return sample


def boosted_candidates(boost):
    """The branches hook: one deviation at the rate above, drawn among the non-argmax actions as the policy would; else -1."""
    def branches(logits, mask, actions):
        probs = torch.softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        top, best = probs.max(-1)
        rate = (boost * (1.0 - top).clamp_min(0.0)).clamp(max=1.0)
        u = 1.0 - torch.rand(len(probs), 2, device=probs.device)  # (0, 1]
        cdf = probs.scatter(1, best[:, None], 0.0).cumsum(-1)
        alt = (cdf < u[:, 1:] * cdf[:, -1:]).sum(-1).clamp(max=probs.shape[1] - 1)
        go = (u[:, 0] <= rate) & (mask.sum(-1) > 1) & (cdf[:, -1] > 0)
        return torch.where(go, alt, torch.full_like(alt, -1))[:, None]

    return branches


class LeafValue(nn.Module):
    """The Actor's encoder (copied) and a two-output head: rest of the hand's score, deals next."""

    def __init__(self, actor):
        super().__init__()
        self.encoder = copy.deepcopy(actor.encoder)
        dim = actor.dim
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, planes, scalars, events):
        state, tiles = self.encoder(planes, scalars, events)
        return self.head(torch.cat([state, tiles.mean(1)], -1))


class Leaves:
    """The leaf network across the run: predict a rollout's leaves, then fit its rows."""

    def __init__(self, actor, cfg, device):
        self.device = device
        self.model = LeafValue(actor).to(device).float()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg.get("leaf_learning_rate", 1e-4)), eps=1e-5)
        self.fit_rows = int(cfg.get("leaf_fit_rows", 200000))
        self.size = 512
        self.mean = self.sd = None
        self.ready = False

    def _out(self, mb):
        cuda = torch.device(self.device).type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            return self.model(mb["planes"], mb["scalars"], mb["events"]).float()

    @torch.no_grad()
    def predict(self, inputs):
        """(n, 2) values of legal-observation inputs (dicts of planes, scalars, events, mask), in target units."""
        self.model.eval()
        out = np.zeros((len(inputs), 2))
        if not inputs:
            return out
        order = np.argsort([len(np.asarray(x["events"]).reshape(-1)) for x in inputs], kind="stable")
        for s in range(0, len(order), 1024):
            b = order[s : s + 1024]
            mb = make_batch([dict(inputs[i], action=0) for i in b], self.device)
            out[b] = self._out(mb).cpu().numpy()
        return out * self.sd + self.mean

    def fit(self, rows, target, rng, *, batch_source=None, epochs=1):
        """Fit (n, 2) targets on up to fit_rows rows; the held-out R² of the rows it had not seen first."""
        stats = {}
        if self.ready:
            # Rows of this rollout, before the network has seen them.
            probe = rng.choice(len(rows), size=min(len(rows), 20000), replace=False)
            pred = self.predict([rows[i] for i in probe])
            for k, name in enumerate(("pay", "deal")):
                err = ((target[probe, k] - pred[:, k]) ** 2).mean()
                stats[f"leaf_r2_{name}"] = 1.0 - float(err / max(target[probe, k].var(), 1e-9))
        else:
            self.mean = target.mean(0)
            self.sd = np.maximum(target.std(0), 1e-6)
        self.model.train()
        idx = rng.permutation(len(rows))[: self.fit_rows]
        losses = []
        for _ in range(epochs):
            order = rng.permutation(idx)
            for s in range(0, len(order), self.size):
                b = order[s : s + self.size]
                mb = (batch_source.batch(b) if batch_source is not None
                      else make_batch([rows[i] for i in b], self.device))
                goal = torch.as_tensor((target[b] - self.mean) / self.sd, device=self.device, dtype=torch.float32)
                loss = ((self._out(mb) - goal) ** 2).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                losses.append(loss.detach())
        self.ready = True
        stats["leaf_loss"] = float(torch.stack(losses[-50:]).mean()) if losses else 0.0
        return stats


def branch_advantages(rows, cfg, leaves):
    """Each deviation's truncated paired advantage, the leaf targets of every row, and stats.

    Rows with a candidate get it as their action (the gradient's term); every
    other row played the argmax and has advantage 0, as under vine. Without a
    ready leaf network (the first rollout) the advantages are all 0.
    Returns (normalized advantages, stats, (n, 2) leaf targets).
    """
    count = len(rows)
    hand = np.array([r["hand"] for r in rows])
    reward = np.array([float(r["reward"]) for r in rows])
    if "game_mark" in rows[0]:
        mark = np.array([r["game_mark"] for r in rows])
        game = np.cumsum(np.r_[1, (np.diff(mark) != 0).astype(int)]) - 1
    else:
        game = np.cumsum(np.r_[1, (np.diff(hand) < 0).astype(int)]) - 1
    new_hand = np.r_[True, (np.diff(hand) != 0) | (np.diff(game) != 0)]
    hand_id = np.cumsum(new_hand) - 1
    first = np.flatnonzero(new_hand)
    outcome = np.bincount(hand_id, weights=reward)
    dealer = np.array([float(r["is_dealer"]) for r in rows])
    whole = (np.bincount(game[first]) > 1)[game[first]]
    dealt = (dealer[first] > 0.5) & whole
    undealt = (dealer[first] <= 0.5) & whole
    premium = (float(outcome[dealt].mean() - outcome[undealt].mean())
               if dealt.any() and undealt.any() else 0.0)
    same_game_next = np.r_[game[first[1:]] == game[first[:-1]], False]
    deals_next = np.where(same_game_next, np.r_[dealer[first[1:]], 0.0], 0.0)
    last_hand = ~same_game_next
    # The rest of the hand from each row (its own reward included), and
    # whether the learner deals next, each less the draw luck still to come.
    rest = np.zeros(count)
    running = 0.0
    for i in range(count - 1, -1, -1):
        if i + 1 == count or new_hand[i + 1]:
            running = 0.0
        running += reward[i]
        rest[i] = running
    target = np.stack([rest, deals_next[hand_id]], -1)
    luck = bool(cfg.get("luck"))
    if luck:
        row_luck = np.array([[float(r.get("luck_pay", 0.0)), float(r.get("luck_deal", 0.0))] for r in rows])
        end = np.zeros((len(first), 2))
        for i, r in enumerate(rows):
            if "end_luck_pay" in r:
                end[hand_id[i]] = (float(r["end_luck_pay"]), float(r["end_luck_deal"]))
        target -= end[hand_id] - row_luck
    adv = np.zeros(count)
    picked = [i for i, r in enumerate(rows) if r.get("branches") and r.get("branch_control") is not None]
    stats = {"premium": premium, "games": int(game.max() + 1), "branch_rate": len(picked) / max(1, count)}
    copies = [(i, r) for i in picked for r in (rows[i]["branches"][0], rows[i]["branch_control"])]
    stopped = [k for k, (_, r) in enumerate(copies) if r.get("truncated")]
    if not picked or (stopped and (leaves is None or not leaves.ready)):
        stats["adv_scale"] = 1.0
        return adv.astype(np.float32), stats, target
    values = np.zeros(len(copies))
    if stopped:
        leaf = leaves.predict([copies[k][1]["leaf_row"] for k in stopped])
        for k, v in zip(stopped, leaf):
            i = copies[k][0]
            values[k] = v[0] + (0.0 if last_hand[hand_id[i]] else premium) * v[1]
    for k, (i, r) in enumerate(copies):
        scale = 0.0 if last_hand[hand_id[i]] else premium
        if not r.get("truncated"):
            values[k] = float(r["playout"]) + scale * float(r["playout_deals_next"])
        if luck:
            # The luck since the fork: the copy's cumulative luck less the fork row's (common to both copies).
            values[k] -= float(r.get("playout_luck_pay", 0.0)) + scale * float(r.get("playout_luck_deal", 0.0))
    gap = values[0::2] - values[1::2]
    weight = np.exp(np.array([float(rows[i].get("logprob", 0.0)) for i in picked]))
    if cfg.get("branch_resample"):
        # Progress-matched importance weights of the re-dealt worlds (worlds.py).
        from .worlds import importance

        real = [r["opp_code"] for r in rows if "opp_code" in r]
        world = importance(real, [rows[i]["branches"][0]["world_weight"] for i in picked])
        gap = gap * world
        stats |= {"world_weight_sd": float(world.std()), "world_weight_max": float(world.max()),
                  "world_ess": float(world.sum() ** 2 / max((world ** 2).sum(), 1e-12) / len(world))}
    for j, i in enumerate(picked):
        adv[i] = gap[j] * weight[j]
        rows[i] = dict(rows[i], action=int(rows[i]["branches"][0]["action"]))
    flags = np.array([bool(r.get("truncated")) for _, r in copies])
    moved = adv[picked]
    stats |= {
        "truncated_candidate": float(flags[0::2].mean()), "truncated_control": float(flags[1::2].mean()),
        "deviation_adv_mean": float(moved.mean()), "deviation_adv_sd": float(moved.std()),
        "deviation_zero": float((moved == 0).mean()),
        "boost_weight_mean": float(weight.mean()), "boost_capped": float((weight > 1.0 + 1e-6).mean()),
    }
    stats["adv_scale"] = float(max(adv.std(), 1e-9))
    return (adv / max(adv.std(), 1e-9)).astype(np.float32), stats, target
