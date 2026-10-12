"""Policy improvement from a learned action-advantage network (cfg["mpo"]).

The playout PG uses each deviation once: one sampled action at one state, one
wall, its paired difference against the argmax continuation, a single
gradient step, then the sample is gone (183 measured the three rollouts' own
gradients at cosine ~0.01: a step's direction is almost all noise). Here the
deviations are kept for a few steps and fitted by a Q network (the Actor's
shape on public inputs, heads zeroed): q(s, a) - q(s, argmax) regresses the
paired difference D. The fitted network gives an advantage for every legal
action at every state, also the ones never tried there, and the step follows

    sum_s sum_a pi(a|s) A(s, a) grad log pi(a|s),   A = q - E_pi q,

over a sample of the step's states: the first-order MPO step toward
pi* ~ pi exp(q / tau), scaled to the KL goal like the sampled gradient.

Diagnostics: before the Q network trains on a step's new deviations it
predicts them (held out): correlation with D, the calibration slope
cov(pred, D) / var(pred) and R2 against predicting zero, overall and per kind
of decision. The E-step at the KL goal reports its tau, how many argmaxes pi*
would change and the predicted gain of its argmax over the old one.

The opponent-progress probe (cfg["progress_probe"]) is a diagnostic only: how
well each opponent's true shanten (the oracle side of every row) can be
predicted from the learner's legal inputs by a dedicated network, against a
logistic model on the encoder's scalar counters alone.
"""

from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

from selfplay_core.actions import GANG_EXPOSED, HU

from .batch import actor_args, make_batch, masked_log_probs

INPUTS = ("planes", "scalars", "events", "mask")


def _inputs(row, action=0):
    return {"planes": row["planes"], "scalars": row["scalars"], "events": row["events"],
            "mask": row["mask"], "action": action}


def _kind(mask):
    """0 discard (draw phase, no kong or win), 1 response, 2 kong or win window."""
    window = mask[:, GANG_EXPOSED:HU + 1].any(1)
    discard = mask[:, :34].any(1)
    return torch.where(window, 2, torch.where(discard, 0, 1))


class Improver:
    def __init__(self, ctx, actor, cfg, rng):
        self.ctx, self.cfg, self.rng, self.device = ctx, cfg, rng, ctx.device
        self.q = ctx.new_actor()
        self.q.load_state_dict(actor.state_dict())
        with torch.no_grad():
            for head in (self.q.discard_head, self.q.claim_head, self.q.state_head):
                head.weight.zero_()
                head.bias.zero_()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(cfg.get("q_learning_rate", 1e-4)), eps=1e-5)
        self.replay = deque(maxlen=int(cfg.get("q_replay_steps", 4)))
        self.fresh = []  # this step's deviations: (inputs, action, argmax, D)
        self.states = []  # this step's states for the M-step
        self.scale = None  # the D scale the Q network regresses in
        self.probe = Probe(ctx, cfg, rng) if cfg.get("progress_probe") else None

    # ---- collection ----------------------------------------------------------
    def add_chunk(self, actor, rows, raw_adv):
        """Keep the chunk's deviations (with the argmax they deviated from) and a sample of its states."""
        deviated = np.flatnonzero([bool(r.get("deviated")) for r in rows])
        argmax = self.argmax(actor, [rows[i] for i in deviated])
        for i, g in zip(deviated, argmax):
            self.fresh.append((_inputs(rows[i], int(rows[i]["action"])), int(g), float(raw_adv[i])))
        keep = int(self.cfg.get("mpo_states_per_chunk", 150000))
        picked = self.rng.choice(len(rows), size=min(len(rows), keep), replace=False)
        self.states += [_inputs(rows[i]) for i in picked]
        if self.probe is not None:
            self.probe.add_chunk(rows)

    def argmax(self, actor, rows):
        out = []
        actor.eval()
        with torch.no_grad():
            for s in range(0, len(rows), 512):
                mb = make_batch([_inputs(r) for r in rows[s:s + 512]], self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = actor(*actor_args(mb))
                out += masked_log_probs(logits, mb["mask"]).argmax(-1).tolist()
        return out

    # ---- Q network -------------------------------------------------------------
    def _pred(self, mb, action, greedy):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = self.q(*actor_args(mb)).float()
        return q.gather(1, action[:, None]).squeeze(1) - q.gather(1, greedy[:, None]).squeeze(1)

    def _batches(self, items, size, shuffle):
        order = self.rng.permutation(len(items)) if shuffle else np.arange(len(items))
        for s in range(0, len(order), size):
            chunk = [items[i] for i in order[s:s + size]]
            mb = make_batch([c[0] for c in chunk], self.device)
            greedy = torch.as_tensor([c[1] for c in chunk], device=self.device)
            target = torch.as_tensor([c[2] for c in chunk], device=self.device, dtype=torch.float32)
            yield mb, greedy, target

    def fit(self):
        """Predict the new deviations first (held out), then train on the replay including them."""
        fresh, self.fresh = self.fresh, []
        stats = {"q_fresh": len(fresh)}
        if not fresh:
            return stats
        if self.scale is None:
            self.scale = max(float(np.std([d for _, _, d in fresh])), 1e-6)
        self.q.eval()
        preds, targets, kinds = [], [], []
        with torch.no_grad():
            for mb, greedy, target in self._batches(fresh, 512, False):
                preds.append(self._pred(mb, mb["action"], greedy) * self.scale)
                targets.append(target)
                kinds.append(_kind(mb["mask"]))
        pred, target, kind = torch.cat(preds), torch.cat(targets), torch.cat(kinds)
        for name, sel in (("all", torch.ones_like(kind, dtype=torch.bool)), ("discard", kind == 0),
                          ("response", kind == 1), ("window", kind == 2)):
            p, d = pred[sel].double(), target[sel].double()
            stats[f"q_{name}_n"] = int(sel.sum())
            if len(p) > 2 and float(p.std()) > 0:
                cov = float(((p - p.mean()) * (d - d.mean())).mean())
                stats[f"q_{name}_corr"] = cov / max(float(p.std() * d.std()), 1e-12)
                stats[f"q_{name}_slope"] = cov / max(float(p.var(unbiased=False)), 1e-12)
                stats[f"q_{name}_r2"] = 1.0 - float(((d - p) ** 2).mean() / max(float((d ** 2).mean()), 1e-12))
                stats[f"q_{name}_pred_sd"] = float(p.std())
        self.replay.append(fresh)
        items = [x for step in self.replay for x in step]
        self.q.train()
        losses = []
        for _ in range(int(self.cfg.get("q_epochs", 2))):
            for mb, greedy, target in self._batches(items, int(self.cfg.get("q_batch", 256)), True):
                loss = ((self._pred(mb, mb["action"], greedy) - target / self.scale) ** 2).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
                self.opt.step()
                losses.append(float(loss.detach()))
        self.q.eval()
        stats["q_replay"] = len(items)
        stats["q_train_loss"] = float(np.mean(losses[-200:])) if losses else 0.0
        if self.probe is not None:
            stats |= self.probe.fit()
        return stats

    # ---- MPO step -------------------------------------------------------------
    def _q(self, mb):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.q(*actor_args(mb)).float() * self.scale

    def _tau(self, batches, logp_old, goal):
        """tau with mean KL(pi_old || pi*) = goal on the KL rows (bisection in log tau)."""
        qs = [self._q(mb) for mb in batches]

        def kl(tau):
            total, rows = 0.0, 0
            for mb, q, old in zip(batches, qs, logp_old):
                star = masked_log_probs(old + q / tau, mb["mask"])
                total += float((old.exp().masked_fill(~mb["mask"], 0) * (old - star).masked_fill(~mb["mask"], 0)).sum())
                rows += len(old)
            return total / max(1, rows)

        lo, hi = np.log(1e-3 * self.scale), np.log(1e3 * self.scale)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if kl(float(np.exp(mid))) > goal:
                lo = mid
            else:
                hi = mid
        tau = float(np.exp(hi))
        flips, gains, rows = 0, 0.0, 0
        for mb, q, old in zip(batches, qs, logp_old):
            star = masked_log_probs(old + q / tau, mb["mask"])
            a_old, a_new = old.argmax(-1), star.argmax(-1)
            flips += int((a_old != a_new).sum())
            gains += float((q.gather(1, a_new[:, None]) - q.gather(1, a_old[:, None])).sum())
            rows += len(old)
        return tau, {"mpo_tau": tau, "mpo_e_kl": kl(tau), "mpo_planned_flip": flips / max(1, rows),
                     "mpo_planned_gain": gains / max(1, rows)}

    def direction(self, actor, params, batches, goal):
        """The gradient of sum_a pi(a|s) A(s, a) over the step's states, A from the Q network.

        It is the first-order MPO step (the cross entropy toward pi* at the
        old weights) and the policy gradient with every legal action's
        advantage at every state instead of one sampled deviation's; the KL
        step then scales it like the sampled gradient. The E-step at the KL
        goal is reported for what such a step aims at.
        """
        actor.eval()
        with torch.no_grad():
            logp_old = [masked_log_probs(actor(*actor_args(mb)).float(), mb["mask"]) for mb in batches]
        _, stats = self._tau(batches, logp_old, goal)
        states, self.states = self.states, []
        total = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        size = int(self.cfg.get("q_batch", 256))
        spread = []
        for s in range(0, len(states), size):
            mb = make_batch(states[s:s + size], self.device)
            q = self._q(mb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = actor(*actor_args(mb))
            logp = masked_log_probs(logits, mb["mask"])
            p = logp.exp().masked_fill(~mb["mask"], 0)
            with torch.no_grad():
                q = q.masked_fill(~mb["mask"], 0)
                adv = (q - (p.detach() * q).sum(-1, keepdim=True)).masked_fill(~mb["mask"], 0)
                spread.append(float((p.detach() * adv * adv).sum()))
            loss = (p * adv).sum() / len(states)
            for acc, g in zip(total, torch.autograd.grad(loss, params, allow_unused=True)):
                if g is not None:
                    acc += g.float()
        stats["mpo_states"] = len(states)
        stats["mpo_adv_sd"] = (sum(spread) / max(1, len(states))) ** 0.5
        return total, stats


class Probe:
    """Opponents' true shanten from the learner's legal inputs (diagnostic only)."""

    CLASSES = 4  # tenpai, 1, 2, 3+ shanten

    def __init__(self, ctx, cfg, rng):
        from guided.networks import HistoryEncoder
        from selfplay_core.encoding import N_SCALARS

        self.cfg, self.rng, self.device = cfg, rng, ctx.device
        dim = 128
        self.encoder = HistoryEncoder(dim, 2, 4, state_only=True).to(self.device)
        self.head = torch.nn.Linear(dim, 3 * self.CLASSES).to(self.device)
        self.base = torch.nn.Linear(N_SCALARS, 3 * self.CLASSES).to(self.device)
        self.opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.head.parameters()), lr=3e-4, eps=1e-5)
        self.base_opt = torch.optim.Adam(self.base.parameters(), lr=3e-3)
        self.rows, self.held = [], []

    def add_chunk(self, rows):
        keep = int(self.cfg.get("progress_probe", 100000))
        picked = self.rng.choice(len(rows), size=min(len(rows), keep), replace=False)
        taken = []
        for i in picked:
            x = np.asarray(rows[i]["oracle_scalars"][:3], dtype=np.float32)
            if (x > 0).all():
                sh = np.clip(np.rint(8 - 9 * x), 0, self.CLASSES - 1).astype(np.int64)
                taken.append((_inputs(rows[i]), sh))
        self.held = taken[: len(taken) // 10]
        self.rows += taken[len(taken) // 10:]

    def _logits(self, mb):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state, _ = self.encoder(*actor_args(mb))
            net = self.head(state).float()
        base = self.base(mb["scalars"].float())
        return net.view(-1, 3, self.CLASSES), base.view(-1, 3, self.CLASSES)

    def _batches(self, items, size):
        for s in range(0, len(items), size):
            chunk = items[s:s + size]
            yield make_batch([c[0] for c in chunk], self.device), torch.as_tensor(np.stack([c[1] for c in chunk]), device=self.device)

    def fit(self):
        stats = {}
        if self.held:
            nets, bases, labels = [], [], []
            with torch.no_grad():
                for mb, y in self._batches(self.held, 512):
                    net, base = self._logits(mb)
                    nets.append(net.softmax(-1)[..., 0].flatten())
                    bases.append(base.softmax(-1)[..., 0].flatten())
                    labels.append((y == 0).flatten())
            y = torch.cat(labels).cpu().numpy()
            for name, score in (("net", torch.cat(nets)), ("scalars", torch.cat(bases))):
                stats[f"probe_tenpai_auc_{name}"] = _auc(score.cpu().numpy(), y)
            stats["probe_tenpai_rate"] = float(y.mean())
        rows, self.rows = self.rows, []
        order = self.rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        for mb, y in self._batches(rows, 256):
            net, base = self._logits(mb)
            loss = F.cross_entropy(net.reshape(-1, self.CLASSES), y.flatten())
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            base_loss = F.cross_entropy(base.reshape(-1, self.CLASSES), y.flatten())
            self.base_opt.zero_grad(set_to_none=True)
            base_loss.backward()
            self.base_opt.step()
        return stats


def _auc(score, y):
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
