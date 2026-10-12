"""Training the belief head (cfg["belief_aux"]; representation #14, the belief bottleneck).

The start (rl-196-belief) is 196 with a zero-initialized belief head
(guided.networks, "belief_head": true): after the middle block it predicts
each opponent's concealed copies of every tile, and its stop-gradient output
is added back to the tile tokens. The head learns from the hidden hands
(ADR-0009: hidden state only as a training label; rows' oracle_planes, the
recipe's critic_features), by binary cross entropy against
guided.networks.belief_target.

Its loss reaches the blocks below the head, which the policy shares: plain
Adam on it moved the policy by KL 0.65 and flipped 27% of the argmaxes in the
first 200 steps (220), the policy flattened and the boosted deviation rate
went from 0.10 to 0.96. So:
- first, `warm_steps` steps train the head alone (the shared blocks fixed);
- after that, each policy-gradient step is followed by `steps` joint steps on
  BCE + anchor * KL(policy before them || policy), on the same microbatch,
  `anchor` adapted so the belief steps' KL on the step's KL rows stays near
  `kl_target`; steps that overshoot 4 x kl_target are undone.

belief_aux["outcome"] ({"weight": w}; the outcome auxiliary, as phasic policy
gradient's auxiliary phase): the same joint steps also fit how the row's hand
ends, from the Actor's own encoder output (state and the mean tile token), by
a recipe-side head the exported Actor does not carry. Its targets are the leaf
network's (leaves.branch_advantages, normalized as Leaves.fit does): the rest
of the hand's score and whether the learner deals next, each less its draw
luck, and under leaf_zero_sum the three opponents' hand scores (labels only,
ADR-0009). The loss is the masked mean squared error, weighted by w and added
to the BCE; the warm steps train this head too, with the shared blocks fixed.
The policy's movement stays under the same anchor and undo.

belief_aux["discards"] ({"weight": w}; opponent modeling): likewise, a
recipe-side head on the encoder's tile tokens predicts, for each opponent
(next, opposite, previous seat), which tile it discards first after the
row's decision, before the learner's next decision in the hand: a softmax
over the 34 tiles per opponent, cross entropy where that discard is known.
The label is public (the next learner row's events), so this is the policy's
own observation stream, predicted one round ahead.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batch import actor_args, make_batch, masked_log_probs

DISCARD = 2  # selfplay_core.encoding.EVENT_KINDS["discard"]


def _kl(old, new, mask):
    return (old.exp().masked_fill(~mask, 0) * (old - new).masked_fill(~mask, 0)).sum(-1)


def next_discards(rows, picked):
    """(len(picked), 3) the tile each opponent (next, opposite, previous seat) discards first after row i's
    decision, before the learner's next decision in the same hand; -1 where there is none."""
    out = np.full((len(picked), 3), -1, np.int64)
    if len(rows) < 2:
        return out
    marks = "game_mark" in rows[0]
    for k, i in enumerate(picked):
        if i + 1 >= len(rows):
            continue
        a, b = rows[i], rows[i + 1]
        if a["hand"] != b["hand"] or (marks and a["game_mark"] != b["game_mark"]):
            continue
        before = np.asarray(a["events"]).reshape(-1, 5)
        after = np.asarray(b["events"]).reshape(-1, 5)
        start = 0 if (len(before) == 1 and not before.any()) else len(before)  # encode's placeholder: no events
        if len(after) < start or (start and not np.array_equal(after[start - 1], before[start - 1])):
            continue
        rest = after[start:]
        for seat in (1, 2, 3):
            hit = np.flatnonzero((rest[:, 0] == DISCARD) & (rest[:, 1] == seat) & (rest[:, 2] < 34))
            if len(hit):
                out[k, seat - 1] = rest[hit[0], 2]
    return out


class BeliefTrainer:
    def __init__(self, actor, spec):
        from guided.networks import belief_parameters, belief_target, belief_target_v2

        self.version = 2 if getattr(actor.encoder, "belief_version", 1) == 2 else 1
        self.target = belief_target if self.version == 1 else belief_target_v2
        self.steps = int(spec.get("steps", 200))
        self.warm_steps = int(spec.get("warm_steps", 1500))
        self.size = int(spec.get("microbatch", 256))
        self.keep = int(spec.get("keep_rows", 60000))
        self.kl_target = float(spec.get("kl_target", 0.002))
        self.anchor = float(spec.get("anchor", 30.0))
        head = list(belief_parameters(actor))
        self.head_opt = torch.optim.Adam(head, lr=float(spec.get("head_lr", 3e-4)), eps=1e-5)
        self.opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],
                                    lr=float(spec.get("lr", 3e-5)), eps=1e-5)
        self.warmed = False
        self.rows = []
        dim = actor.dim
        device = next(actor.parameters()).device
        self.outcome = self.discards = None
        aux = []
        if spec.get("outcome"):
            self.outcome = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
                                         nn.Linear(dim, 5)).to(device).float()
            self.outcome_weight = float(spec["outcome"].get("weight", 1.0))
            self.goals = []
            aux.append(self.outcome)
        if spec.get("discards"):
            self.discards = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3)).to(device).float()
            self.discard_weight = float(spec["discards"].get("weight", 1.0))
            self.next_tiles = []
            aux.append(self.discards)
        self.aux = aux
        self.aux_opt = (torch.optim.Adam([p for m in aux for p in m.parameters()],
                                         lr=float(spec.get("head_lr", 3e-4)), eps=1e-5) if aux else None)

    def add(self, rows, rng, goals=None):
        """Keep a sample of a rollout's rows (and their auxiliary targets) for the next belief steps."""
        pick = rng.choice(len(rows), size=min(len(rows), self.keep // 3), replace=False)
        self.rows += [rows[i] for i in pick]
        if self.outcome is not None:
            width = 5
            if goals is None:
                goals = np.full((len(rows), width), np.nan)
            goals = np.asarray(goals, dtype=np.float64)
            if goals.shape[1] < width:
                goals = np.concatenate([goals, np.full((len(goals), width - goals.shape[1]), np.nan)], 1)
            self.goals.append(goals[pick, :width])
        if self.discards is not None:
            self.next_tiles.append(next_discards(rows, pick))

    def _measure(self, actor, kl_batches, before):
        with torch.no_grad():
            after = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in kl_batches]
        n = sum(len(o) for o in before)
        kl = sum(float(_kl(old, new, mb["mask"]).sum()) for mb, old, new in zip(kl_batches, before, after)) / n
        flip = sum(int((old.argmax(-1) != new.argmax(-1)).sum()) for old, new in zip(before, after)) / n
        return kl, flip

    def step(self, actor, device, rng, kl_batches):
        rows, self.rows = self.rows, []
        goals = tiles_next = None
        if self.outcome is not None:
            goals = np.concatenate(self.goals) if self.goals else np.full((len(rows), 5), np.nan)
            self.goals = []
        if self.discards is not None:
            tiles_next = np.concatenate(self.next_tiles) if self.next_tiles else np.full((len(rows), 3), -1)
            self.next_tiles = []
        if not rows:
            return {}
        with torch.no_grad():
            before = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in kl_batches]
        saved = [p.detach().clone() for p in actor.parameters()]
        actor.train(False)
        warm = not self.warmed
        reference = None if warm else copy.deepcopy(actor).eval()
        trainable = [p for p in actor.parameters() if p.requires_grad]
        # The encoder's (state, tiles) for the auxiliary heads, hooked on the trained Actor only.
        encoded = {}
        hook = (actor.encoder.register_forward_hook(lambda module, args, out: encoded.__setitem__("out", out))
                if self.aux else None)
        for module in self.aux:
            module.to(device).train()
        losses, anchors = [], []
        aux_losses, aux_r2, disc_losses, disc_hits = [], [], [], []
        for _ in range(self.warm_steps if warm else self.steps):
            idx = rng.choice(len(rows), size=min(len(rows), self.size), replace=False)
            mb = make_batch([rows[i] for i in idx], device, oracle=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                logits, belief = actor(*actor_args(mb), return_belief=True)
            if self.version == 2:
                target = self.target(mb["oracle_planes"], mb["oracle_scalars"], mb["opp_waits"])
                loss = sum(F.binary_cross_entropy_with_logits(p.float(), t) for p, t in zip(belief, target))
            else:
                loss = F.binary_cross_entropy_with_logits(belief.float(), self.target(mb["oracle_planes"]))
            losses.append(loss.detach())
            total = loss
            if self.outcome is not None:
                state, tiles = encoded["out"][0], encoded["out"][1]
                pred = self.outcome(torch.cat([state.float(), tiles.float().mean(1)], -1))
                goal = goals[idx]
                known = torch.as_tensor(~np.isnan(goal), device=pred.device, dtype=torch.float32)
                goal = torch.as_tensor(np.nan_to_num(goal), device=pred.device, dtype=torch.float32)
                err = (pred - goal) ** 2 * known
                aux = err.sum() / known.sum().clamp_min(1.0)
                aux_losses.append(aux.detach())
                if known[:, 0].sum() > 1:
                    pay = goal[:, 0][known[:, 0] > 0]
                    aux_r2.append((1.0 - err[:, 0].sum() / known[:, 0].sum() / pay.var().clamp_min(1e-9)).detach())
                total = total + self.outcome_weight * aux
            if self.discards is not None:
                tiles = encoded["out"][1]
                scores = self.discards(tiles[:, :34].float()).transpose(1, 2)  # [b, 3 opponents, 34 tiles]
                label = torch.as_tensor(tiles_next[idx], device=scores.device, dtype=torch.long)
                seen = label >= 0
                if bool(seen.any()):
                    ce = F.cross_entropy(scores[seen], label[seen])
                    disc_losses.append(ce.detach())
                    disc_hits.append((scores[seen].argmax(-1) == label[seen]).float().mean().detach())
                    total = total + self.discard_weight * ce
            if warm:
                self.head_opt.zero_grad(set_to_none=True)
                if self.aux_opt is not None:
                    self.aux_opt.zero_grad(set_to_none=True)
                total.backward()
                self.head_opt.step()
                if self.aux_opt is not None:
                    self.aux_opt.step()
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.device(device).type == "cuda"):
                old = masked_log_probs(reference(*actor_args(mb)), mb["mask"])
            drift = _kl(old, masked_log_probs(logits, mb["mask"]), mb["mask"]).mean()
            anchors.append(drift.detach())
            self.opt.zero_grad(set_to_none=True)
            if self.aux_opt is not None:
                self.aux_opt.zero_grad(set_to_none=True)
            (total + self.anchor * drift).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            self.opt.step()
            if self.aux_opt is not None:
                self.aux_opt.step()
        if hook is not None:
            hook.remove()
        encoded.clear()
        self.opt.zero_grad(set_to_none=True)
        self.head_opt.zero_grad(set_to_none=True)
        kl, flip = self._measure(actor, kl_batches, before)
        undone = 0.0
        if not warm and kl > 4 * self.kl_target:
            with torch.no_grad():
                for p, s in zip(actor.parameters(), saved):
                    p.copy_(s)
            undone = 1.0
        stats = {"belief_loss_first": float(torch.stack(losses[:20]).mean()),
                 "belief_loss_last": float(torch.stack(losses[-20:]).mean()),
                 "belief_kl": kl, "belief_flip": flip, "belief_warm": float(warm),
                 "belief_anchor": self.anchor, "belief_undone": undone}
        if anchors:
            stats["belief_drift"] = float(torch.stack(anchors).mean())
        if aux_losses:
            stats["outcome_loss_first"] = float(torch.stack(aux_losses[:20]).mean())
            stats["outcome_loss_last"] = float(torch.stack(aux_losses[-20:]).mean())
            stats["outcome_known"] = float((~np.isnan(goals)).all(1).mean())
        if aux_r2:
            stats["outcome_r2_pay"] = float(torch.stack(aux_r2[-50:]).mean())
        if disc_losses:
            stats["discard_loss_first"] = float(torch.stack(disc_losses[:20]).mean())
            stats["discard_loss_last"] = float(torch.stack(disc_losses[-20:]).mean())
            stats["discard_acc"] = float(torch.stack(disc_hits[-50:]).mean())
        if tiles_next is not None:
            stats["discard_known"] = float((tiles_next >= 0).mean())
        if not warm:
            # Keep the belief steps' policy movement near the target.
            if kl > self.kl_target:
                self.anchor = min(self.anchor * 2.0, 1e4)
            elif kl < self.kl_target / 4:
                self.anchor = max(self.anchor / 2.0, 1.0)
        self.warmed = True
        del reference, saved
        return stats
