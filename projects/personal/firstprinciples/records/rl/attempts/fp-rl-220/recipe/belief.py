"""Training the belief head (cfg["belief_aux"]; representation #14, the belief bottleneck).

The start (rl-196-belief) is 196 with a zero-initialized belief head
(guided.networks, "belief_head": true): after the middle block it predicts
each opponent's concealed copies of every tile, and its stop-gradient output
is added back to the tile tokens. The head learns from the hidden hands
(ADR-0009: hidden state only as a training label; rows' oracle_planes, the
recipe's critic_features), by binary cross entropy against
guided.networks.belief_target, through the head and the blocks before it.
After each policy-gradient step a few Adam steps on this loss run over a
sample of the step's rows; the policy's KL from them is measured on the
step's KL rows, since the lower blocks the policy shares move too.
"""

import numpy as np
import torch
from torch.nn import functional as F

from .batch import actor_args, make_batch, masked_log_probs


class BeliefTrainer:
    def __init__(self, actor, spec):
        from guided.networks import belief_target

        self.target = belief_target
        self.steps = int(spec.get("steps", 200))
        self.size = int(spec.get("microbatch", 256))
        self.keep = int(spec.get("keep_rows", 60000))
        self.opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],
                                    lr=float(spec.get("lr", 3e-5)), eps=1e-5)
        self.rows = []

    def add(self, rows, rng):
        """Keep a sample of a rollout's rows for the next belief steps."""
        pick = rng.choice(len(rows), size=min(len(rows), self.keep // 3), replace=False)
        self.rows += [rows[i] for i in pick]

    def step(self, actor, device, rng, kl_batches):
        rows, self.rows = self.rows, []
        if not rows:
            return {}
        with torch.no_grad():
            before = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in kl_batches]
        losses = []
        actor.train(False)
        for _ in range(self.steps):
            idx = rng.choice(len(rows), size=min(len(rows), self.size), replace=False)
            mb = make_batch([rows[i] for i in idx], device, oracle=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, belief = actor(*actor_args(mb), return_belief=True)
            loss = F.binary_cross_entropy_with_logits(belief.float(), self.target(mb["oracle_planes"]))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], 1.0)
            self.opt.step()
            losses.append(float(loss.detach()))
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            after = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in kl_batches]
        kl = sum(float((old.exp().masked_fill(~mb["mask"], 0) * (old - new).masked_fill(~mb["mask"], 0)).sum())
                 for mb, old, new in zip(kl_batches, before, after)) / sum(len(o) for o in before)
        flip = sum(int((old.argmax(-1) != new.argmax(-1)).sum()) for old, new in zip(before, after)) / sum(
            len(o) for o in before)
        k = max(1, len(losses) // 10)
        return {"belief_loss_first": float(np.mean(losses[:k])), "belief_loss_last": float(np.mean(losses[-k:])),
                "belief_kl": kl, "belief_flip": flip}
