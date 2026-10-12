"""REBEL's pairwise regression in place of the policy-gradient step (cfg["rebel"]).

A playout deviation gives, at state s, the paired difference A of its action
a against the argmax g on one wall (targets.playout_advantages, luck taken
off). The policy gradient uses it once, weighted by pi(a) through grad log
pi(a), so a rare but better action barely moves. REBEL (Gao et al. 2024)
instead regresses it on the policy's own log-ratios against the step's start
pi_t:

    min sum_i w_i ( (1/eta) [ln pi(a_i|s_i)/pi_t(a_i|s_i) - ln pi(g_i|s_i)/pi_t(g_i|s_i)] - A_i )^2

whose exact solution is the mirror-descent step pi_t exp(eta Q). w_i is the
deviation's boost weight (loop.boosted_deviations), the share the policy's own
sampling would give it. The fit runs a few epochs over the step's deviations
with a held-out tenth: the epoch with the lowest held-out loss is kept, and
its R2 against predicting zero says how much of the signal is learnable at
all. The kept parameters are then pulled back towards pi_t until
KL(pi_t || pi) is at most the step's KL goal, so its size compares with the
policy-gradient step's.
"""

import copy

import numpy as np
import torch

from .batch import actor_args, make_batch, masked_log_probs


def _pair_logps(model, rows, actions, device, size=512):
    """log pi(a) and log pi(argmax), and the argmax, under `model` (autocast as in training)."""
    out_a, out_g, best = [], [], []
    with torch.no_grad():
        for s in range(0, len(rows), size):
            mb = make_batch(rows[s : s + size], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*actor_args(mb))
            logp = masked_log_probs(logits, mb["mask"])
            g = logp.argmax(-1)
            a = torch.as_tensor(actions[s : s + size], device=device)
            out_a.append(logp.gather(1, a[:, None]).squeeze(1).float().cpu())
            out_g.append(logp.gather(1, g[:, None]).squeeze(1).float().cpu())
            best.append(g.cpu())
    return torch.cat(out_a), torch.cat(out_g), torch.cat(best)


def fit(actor, rows, actions, advantages, weights, cfg, device, rng, kl_batches, goal):
    spec = cfg["rebel"]
    eta = float(spec.get("eta", 0.3))
    epochs = int(spec.get("epochs", 4))
    size = int(spec.get("microbatch", 256))
    frozen = copy.deepcopy(actor).eval().requires_grad_(False)
    actions = np.asarray(actions, dtype=np.int64)
    base_a, base_g, best = _pair_logps(frozen, rows, actions, device)
    target = torch.as_tensor(np.asarray(advantages, dtype=np.float32))
    weight = torch.as_tensor(np.asarray(weights, dtype=np.float32))
    weight = weight / weight.mean().clamp_min(1e-12)
    order = rng.permutation(len(rows))
    held = order[: max(1, len(rows) // 10)]
    train = order[len(held):]
    params = [p for p in actor.parameters() if p.requires_grad]
    start = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=float(spec.get("lr", 5e-5)), eps=1e-5)

    def loss_of(idx, grad):
        mb = make_batch([rows[i] for i in idx], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = actor(*actor_args(mb))
        logp = masked_log_probs(logits, mb["mask"])
        a = torch.as_tensor(actions[idx], device=device)
        g = best[idx].to(device)
        ratio = ((logp.gather(1, a[:, None]).squeeze(1) - base_a[idx].to(device))
                 - (logp.gather(1, g[:, None]).squeeze(1) - base_g[idx].to(device)))
        err = ratio / eta - target[idx].to(device)
        loss = (weight[idx].to(device) * err * err).sum()
        if grad:
            opt.zero_grad(set_to_none=True)
            (loss / len(idx)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        return float(loss.detach())

    def held_loss():
        with torch.no_grad():
            return sum(loss_of(held[s : s + 1024], False) for s in range(0, len(held), 1024)) / len(held)

    zero = float((weight[held] * target[held] ** 2).sum()) / len(held)
    best_loss, best_params, best_epoch = held_loss(), [p.detach().clone() for p in params], 0
    curve = [best_loss]
    actor.train(False)
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(train)
        for s in range(0, len(perm), size):
            loss_of(perm[s : s + size], True)
        current = held_loss()
        curve.append(current)
        if current < best_loss:
            best_loss, best_params, best_epoch = current, [p.detach().clone() for p in params], epoch
    opt.zero_grad(set_to_none=True)
    fitted = best_params

    # Pull back towards pi_t until KL(pi_t || pi) <= goal (bisection on the
    # interpolation weight; 1 when the fitted step is already within it).
    with torch.no_grad():
        reference = [masked_log_probs(frozen(*actor_args(mb)), mb["mask"]) for mb in kl_batches]

    def at(alpha):
        with torch.no_grad():
            for p, p0, p1 in zip(params, start, fitted):
                p.copy_(p0 + alpha * (p1 - p0))
            latest = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in kl_batches]
        kl = sum(float((old.exp().masked_fill(~mb["mask"], 0) * (old - new).masked_fill(~mb["mask"], 0)).sum())
                 for mb, old, new in zip(kl_batches, reference, latest)) / sum(len(o) for o in reference)
        flip = sum(int((old.argmax(-1) != new.argmax(-1)).sum()) for old, new in zip(reference, latest)) / sum(
            len(o) for o in reference)
        return kl, flip

    alpha, (kl, flip) = 1.0, at(1.0)
    if kl > goal:
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            k, f = at(mid)
            if k > goal:
                hi = mid
            else:
                lo, kl, flip = mid, k, f
        alpha = lo
        at(alpha)
    return {"rebel_rows": len(rows), "rebel_r2": 1.0 - best_loss / max(zero, 1e-12), "rebel_best_epoch": best_epoch,
            "rebel_curve_first": curve[0], "rebel_curve_last": curve[-1], "rebel_alpha": alpha,
            "kl": kl, "flip": flip}
