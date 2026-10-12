"""Distillation of an ensemble of the line's versions (cfg["ensemble"]).

Each policy-gradient version ends where its last KL steps left it: around
the same optimum, plus that run's own noise (each step moves the policy as
far along its noise as along its signal). Versions from different runs
(224, 227, 233: suit-shared, belief head, continuation portfolio; within a
standard error of each other) carry independent parts of that noise. Their
mean policy, each member's softmax over the legal actions averaged, keeps
what they share and averages the rest. The learner, starting from one of
them, fits it on its own states (DAgger, no teacher labels): KL(mean || pi),
under bf16 autocast for every network alike.
"""

import numpy as np
import torch

from .batch import actor_args, make_batch, masked_log_probs


def load_members(ctx, versions):
    from guided.deployment import load_actor

    return [load_actor(ctx.plan["catalog"][v]["path"], ctx.device)[0].eval() for v in versions]


def distill_ensemble(actor, members, rows, opt, size, device, rng, epochs):
    actor.eval()
    cuda = torch.device(device).type == "cuda"
    total, kl_sum, start_kl, agree, n = 0.0, 0.0, 0.0, 0.0, 0
    for epoch in range(epochs):
        order = rng.permutation(len(rows))
        for s in range(0, len(order), size):
            idx = order[s : s + size]
            mb = make_batch([rows[i] for i in idx], device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                probs = torch.stack([masked_log_probs(m(*actor_args(mb)), mb["mask"]).exp() for m in members]).mean(0)
            target_logp = probs.clamp_min(1e-30).log()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                logp = masked_log_probs(actor(*actor_args(mb)), mb["mask"])
            kl = (probs * (target_logp - logp)).masked_fill(~mb["mask"], 0.0).sum(1)
            loss = kl.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            if epoch == 0:
                kl_sum += float(kl.detach().sum())
                agree += float((logp.argmax(-1) == probs.argmax(-1)).sum())
                n += len(idx)
    opt.zero_grad(set_to_none=True)
    return {"ensemble_kl": kl_sum / max(1, n), "ensemble_agreement": agree / max(1, n)}
