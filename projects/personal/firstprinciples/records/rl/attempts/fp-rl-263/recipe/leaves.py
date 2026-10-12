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

cfg["leaf_zero_sum"] (#12, the four-player zero-sum critic): the network
predicts the hand's score for all four seats (the learner, then the next,
opposite and previous seat) and its output is projected to zero mean across
them, as the platform's scores are (a hand pays only when it ends, so the rest
of the hand is the whole hand's delta). The others' deltas are labels only:
the next hand's starting scores less this hand's (rows' hand_scores, absolute
seats), or end_scores; a game's last hand without them trains only the
learner's channel. The learner's channel (less its draw luck, as before) is
the leaf value; the others are there so its value has to be consistent with
what the three opponents can still make of the hand.
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


def boosted_candidates(boost, k=1):
    """The branches hook: deviations at the rate above, drawn among the non-argmax actions as the policy would; else -1.

    With k > 1 (cfg["branch_candidates"]) a deviating row draws k candidates
    independently from that law and plays each distinct one once: a draw
    equal to an earlier one is -1. With k = 2 a row's estimate is the mean of
    its two draws' advantages, so a row with two copies weighs each 1/2 and a
    row whose draws coincided (one copy) weighs it whole (branch_advantages).
    """
    def branches(logits, mask, actions):
        probs = torch.softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        top, best = probs.max(-1)
        rate = (boost * (1.0 - top).clamp_min(0.0)).clamp(max=1.0)
        u = 1.0 - torch.rand(len(probs), 1 + k, device=probs.device)  # (0, 1]
        cdf = probs.scatter(1, best[:, None], 0.0).cumsum(-1)
        go = (u[:, 0] <= rate) & (mask.sum(-1) > 1) & (cdf[:, -1] > 0)
        none = torch.full_like(best, -1)
        out = []
        for j in range(k):
            alt = (cdf < u[:, 1 + j:2 + j] * cdf[:, -1:]).sum(-1).clamp(max=probs.shape[1] - 1)
            fresh = go
            for earlier in out:
                fresh = fresh & (alt != earlier)
            out.append(torch.where(fresh, alt, none))
        return torch.stack(out, 1)

    return branches


SEATS = [0, 2, 3, 4]  # zero-sum outputs: the learner's pay, then the next, opposite and previous seat's


class LeafValue(nn.Module):
    """The Actor's encoder (copied) and a head: rest of the hand's score, deals next (and the others' scores)."""

    def __init__(self, actor, zero_sum=False):
        super().__init__()
        self.encoder = copy.deepcopy(actor.encoder)
        self.zero_sum = zero_sum
        dim = actor.dim
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
                                  nn.Linear(dim, 5 if zero_sum else 2))

    def forward(self, planes, scalars, events):
        state, tiles = self.encoder(planes, scalars, events)
        out = self.head(torch.cat([state, tiles.mean(1)], -1))
        if self.zero_sum:
            # The four seats' scores share one scale and no offset, so the projection is in score units.
            seats = out[:, SEATS]
            out = out.index_copy(1, torch.as_tensor(SEATS, device=out.device), seats - seats.mean(1, keepdim=True))
        return out


def seat_deltas(rows, first, hand_id, game, hand):
    """(n, 3) each row's next, opposite and previous seat's score in its hand (base units); NaN where unknown."""
    out = np.full((len(rows), 3), np.nan)
    if "hand_scores" not in rows[0] or "seat" not in rows[0]:
        return out
    outcome = np.bincount(hand_id, weights=np.array([float(r["reward"]) for r in rows]))
    for h, f in enumerate(first):
        start = np.asarray(rows[f]["hand_scores"], dtype=np.float64)
        n = first[h + 1] if h + 1 < len(first) else None
        last = (n is None) or game[n] != game[f]
        if not last and hand[n] == hand[f] + 1:
            end = np.asarray(rows[n]["hand_scores"], dtype=np.float64)
        else:
            stop = (n if n is not None else len(rows)) - 1
            if not last or "end_scores" not in rows[stop]:
                continue
            end = np.asarray(rows[stop]["end_scores"], dtype=np.float64)
        delta = end - start
        seat = int(rows[f]["seat"])
        if abs(delta.sum()) > 1e-3 or abs(delta[seat] - outcome[h]) > 1e-3:
            continue  # not this hand's settlement alone (a hand without learner rows, a pre-decision payment)
        stop = n if n is not None else len(rows)
        out[f:stop] = delta[[(seat + j) % 4 for j in (1, 2, 3)]]
    return out


class Leaves:
    """The leaf network across the run: predict a rollout's leaves, then fit its rows."""

    def __init__(self, actor, cfg, device):
        self.device = device
        self.zero_sum = bool(cfg.get("leaf_zero_sum"))
        self.model = LeafValue(actor, self.zero_sum).to(device).float()
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
        out = np.zeros((len(inputs), len(self.mean)))
        if not inputs:
            return out[:, :2]
        order = np.argsort([len(np.asarray(x["events"]).reshape(-1)) for x in inputs], kind="stable")
        for s in range(0, len(order), 1024):
            b = order[s : s + 1024]
            mb = make_batch([dict(inputs[i], action=0) for i in b], self.device)
            out[b] = self._out(mb).cpu().numpy()
        return (out * self.sd + self.mean)[:, :2]

    def fit(self, rows, target, rng, *, batch_source=None, epochs=1):
        """Fit (n, 2) targets (n, 5 under zero_sum, NaN where unknown) on up to fit_rows rows; held-out R² first."""
        stats = {}
        known = ~np.isnan(target)
        target = np.nan_to_num(target)
        if self.zero_sum and self.ready:
            probe = rng.choice(len(rows), size=min(len(rows), 20000), replace=False)
            self.model.eval()
            with torch.no_grad():
                pred = np.concatenate([self._out(make_batch([dict(rows[i], action=0) for i in probe[s:s + 1024]], self.device)).cpu().numpy()
                                       for s in range(0, len(probe), 1024)]) * self.sd + self.mean
            k = known[probe][:, 2:].all(1)
            if k.sum() > 2:
                d = target[probe][k][:, 2:]
                stats["leaf_r2_others"] = 1.0 - float(((d - pred[k][:, 2:]) ** 2).mean() / max(d.var(0).mean(), 1e-9))
            stats["leaf_others_known"] = float(known[:, 2:].all(1).mean())
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
            if self.zero_sum:
                # One scale and no offset for the four seats' scores (the projection's units).
                pooled = np.concatenate([target[known[:, k], k] for k in SEATS])
                self.mean[SEATS] = 0.0
                self.sd[SEATS] = max(float(np.sqrt((pooled ** 2).mean())), 1e-6)
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
                seen = torch.as_tensor(known[b], device=self.device, dtype=torch.float32)
                loss = (((self._out(mb) - goal) ** 2) * seen).sum() / seen.sum().clamp_min(1.0)
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
    ready leaf network (the first rollout) the advantages are all 0. A row
    with a second candidate (cfg["branch_candidates"] = 2) adds a copy of the
    row taking it, after the rows; the two share the row's control and weight,
    each with half of it, and both carry that half as their weight's log in
    the logprob slot (what the control variate reads).
    Returns (normalized advantages of rows + extra, stats, (n, 2) leaf targets of rows, extra rows).
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
    if leaves is not None and leaves.zero_sum:
        target = np.concatenate([target, seat_deltas(rows, first, hand_id, game, hand)], -1)
    luck = bool(cfg.get("luck"))
    if luck:
        row_luck = np.array([[float(r.get("luck_pay", 0.0)), float(r.get("luck_deal", 0.0))] for r in rows])
        end = np.zeros((len(first), 2))
        for i, r in enumerate(rows):
            if "end_luck_pay" in r:
                end[hand_id[i]] = (float(r["end_luck_pay"]), float(r["end_luck_deal"]))
        target[:, :2] -= end[hand_id] - row_luck
    picked = [i for i, r in enumerate(rows) if r.get("branches") and r.get("branch_control") is not None]
    # (row, candidate's copy) per candidate: a row's first candidate stays on the row, a second becomes an extra row.
    pairs = [(i, b) for i in picked for b in rows[i]["branches"][:2]]
    adv = np.zeros(count + len(pairs) - len(picked))
    stats = {"premium": premium, "games": int(game.max() + 1), "branch_rate": len(picked) / max(1, count)}
    copies = [(i, r) for i, b in pairs for r in (b, rows[i]["branch_control"])]
    stopped = [k for k, (_, r) in enumerate(copies) if r.get("truncated")]
    if not picked or (stopped and (leaves is None or not leaves.ready)):
        stats["adv_scale"] = 1.0
        return adv[:count].astype(np.float32), stats, target, []
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
    share = {i: 1.0 / min(2, len(rows[i]["branches"])) for i in picked}
    boosted = np.exp(np.array([float(rows[i].get("logprob", 0.0)) for i, _ in pairs]))
    weight = boosted * np.array([share[i] for i, _ in pairs])
    if cfg.get("branch_resample"):
        # Progress-matched importance weights of the re-dealt worlds (worlds.py).
        from .worlds import importance

        real = [r["opp_code"] for r in rows if "opp_code" in r]
        world = importance(real, [b["world_weight"] for _, b in pairs])
        gap = gap * world
        stats |= {"world_weight_sd": float(world.std()), "world_weight_max": float(world.max()),
                  "world_ess": float(world.sum() ** 2 / max((world ** 2).sum(), 1e-12) / len(world))}
    extra, slots, seen = [], [], set()
    for j, (i, b) in enumerate(pairs):
        taken = dict(rows[i], action=int(b["action"]), logprob=float(np.log(weight[j])))
        if i in seen:
            slots.append(count + len(extra))
            extra.append(taken)
        else:
            seen.add(i)
            slots.append(i)
            rows[i] = taken
        adv[slots[-1]] = gap[j] * weight[j]
    flags = np.array([bool(r.get("truncated")) for _, r in copies])
    moved = adv[slots]
    stats |= {
        "truncated_candidate": float(flags[0::2].mean()), "truncated_control": float(flags[1::2].mean()),
        "deviation_adv_mean": float(moved.mean()), "deviation_adv_sd": float(moved.std()),
        "deviation_zero": float((moved == 0).mean()),
        "boost_weight_mean": float(boosted.mean()), "boost_capped": float((boosted > 1.0 + 1e-6).mean()),
        "branch_second": len(extra) / max(1, len(picked)),
    }
    stats["adv_scale"] = float(max(adv.std(), 1e-9))
    return (adv / max(adv.std(), 1e-9)).astype(np.float32), stats, target, extra
