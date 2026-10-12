"""Auxiliary predictions on the encoder's last layer (cfg["aux"]; representation, after 224 and 227).

The two measured gains of the line were representation changes (224: suit
ranks shared; 227: a belief head at the middle block trained by its own loss
under a KL anchor), while every change to the optimizer, the advantage's
noise and the opponents stayed flat. This head reads the encoder's output
(the 34 tile tokens and the state token, what the policy heads read) and
predicts, from the hidden state (ADR-0009: labels only, rows' oracle_planes
and oracle_scalars from the recipe's critic_features):

    each opponent's concealed copies of every tile (counts >= 1..4: 12 x 34),
    the wall's copies of every tile (>= 1..4: 4 x 34),
    each opponent's progress (8 - shanten) / 9 and the wall's size (4 scalars).

Its gradient reaches every block, so the policy moves with it: first
`warm_steps` steps train the head alone on frozen features, then each
policy-gradient step is followed by `steps` joint steps on
loss + anchor * KL(policy before them || policy), the anchor adapted so the
aux steps' own KL stays near `kl_target`, undone past 4 x kl_target (as
belief.py, which 227 measured at a KL ~1e-4 per step). The head is not part
of the Actor: exports carry only the policy.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batch import actor_args, make_batch, masked_log_probs


class AuxHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tiles = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 16))
        self.state = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))

    def forward(self, state, tiles):
        return self.tiles(tiles).transpose(1, 2), self.state(state)  # [n, 16, 34], [n, 4]


def _kl(old, new, mask):
    return (old.exp().masked_fill(~mask, 0) * (old - new).masked_fill(~mask, 0)).sum(-1)


class AuxTrainer:
    def __init__(self, actor, spec, device):
        self.device = device
        self.head = AuxHead(actor.dim).to(device).float()
        self.steps = int(spec.get("steps", 200))
        self.warm_steps = int(spec.get("warm_steps", 1500))
        self.size = int(spec.get("microbatch", 256))
        self.keep = int(spec.get("keep_rows", 60000))
        self.kl_target = float(spec.get("kl_target", 0.002))
        self.anchor = float(spec.get("anchor", 30.0))
        self.head_opt = torch.optim.Adam(self.head.parameters(), lr=float(spec.get("head_lr", 3e-4)), eps=1e-5)
        trainable = [p for p in actor.parameters() if p.requires_grad] + list(self.head.parameters())
        self.opt = torch.optim.Adam(trainable, lr=float(spec.get("lr", 3e-5)), eps=1e-5)
        self.warmed = False
        self.rows = []

    def add(self, rows, rng):
        """Keep a sample of a rollout's rows for the next aux steps."""
        pick = rng.choice(len(rows), size=min(len(rows), self.keep // 3), replace=False)
        self.rows += [rows[i] for i in pick]

    def _loss(self, state, tiles, mb):
        planes, scalars = self.head(state.float(), tiles.float())
        target = mb["oracle_planes"].float()
        bce = F.binary_cross_entropy_with_logits(planes, target)
        mse = F.mse_loss(scalars, mb["oracle_scalars"].float())
        return bce + mse, bce, mse

    def _measure(self, actor, kl_batches, before):
        with torch.no_grad():
            after = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in kl_batches]
        n = sum(len(o) for o in before)
        kl = sum(float(_kl(old, new, mb["mask"]).sum()) for mb, old, new in zip(kl_batches, before, after)) / n
        flip = sum(int((old.argmax(-1) != new.argmax(-1)).sum()) for old, new in zip(before, after)) / n
        return kl, flip

    def step(self, actor, device, rng, kl_batches):
        rows, self.rows = self.rows, []
        if not rows:
            return {}
        cuda = torch.device(device).type == "cuda"
        with torch.no_grad():
            before = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in kl_batches]
        saved = [p.detach().clone() for p in actor.parameters()]
        actor.train(False)
        warm = not self.warmed
        reference = None if warm else copy.deepcopy(actor).eval()
        trainable = [p for p in actor.parameters() if p.requires_grad] + list(self.head.parameters())
        bces, mses, drifts = [], [], []
        for _ in range(self.warm_steps if warm else self.steps):
            idx = rng.choice(len(rows), size=min(len(rows), self.size), replace=False)
            mb = make_batch([rows[i] for i in idx], device, oracle=True)
            args = actor_args(mb)
            if warm:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    state, tiles = actor.encoder(*args[:3])
                loss, bce, mse = self._loss(state, tiles, mb)
                self.head_opt.zero_grad(set_to_none=True)
                loss.backward()
                self.head_opt.step()
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    state, tiles = actor.encoder(*args[:3])
                    logits = actor(*args)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    old = masked_log_probs(reference(*args), mb["mask"])
                loss, bce, mse = self._loss(state, tiles, mb)
                drift = _kl(old, masked_log_probs(logits, mb["mask"]), mb["mask"]).mean()
                drifts.append(drift.detach())
                self.opt.zero_grad(set_to_none=True)
                (loss + self.anchor * drift).backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                self.opt.step()
            bces.append(bce.detach())
            mses.append(mse.detach())
        self.opt.zero_grad(set_to_none=True)
        self.head_opt.zero_grad(set_to_none=True)
        kl, flip = self._measure(actor, kl_batches, before)
        undone = 0.0
        if not warm and kl > 4 * self.kl_target:
            with torch.no_grad():
                for p, s in zip(actor.parameters(), saved):
                    p.copy_(s)
            undone = 1.0
        stats = {"aux_bce_first": float(torch.stack(bces[:20]).mean()), "aux_bce_last": float(torch.stack(bces[-20:]).mean()),
                 "aux_mse_last": float(torch.stack(mses[-20:]).mean()), "aux_kl": kl, "aux_flip": flip,
                 "aux_warm": float(warm), "aux_anchor": self.anchor, "aux_undone": undone}
        if drifts:
            stats["aux_drift"] = float(torch.stack(drifts).mean())
        if not warm:
            if kl > self.kl_target:
                self.anchor = min(self.anchor * 2.0, 1e4)
            elif kl < self.kl_target / 4:
                self.anchor = max(self.anchor / 2.0, 1.0)
        self.warmed = True
        del reference, saved
        return stats
