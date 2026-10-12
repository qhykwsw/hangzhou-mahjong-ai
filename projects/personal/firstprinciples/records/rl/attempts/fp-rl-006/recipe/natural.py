"""Natural-gradient direction: solve (F + damping I) x = g by conjugate gradient.

The raw gradient spends the KL budget mostly on the few directions the policy
is most sensitive to; F^-1 g is the direction that gains the most per unit of
KL (TRPO). F is the Fisher of the Actor's action distribution on the KL subset.
A Fisher-vector product takes a central difference of the logits along v (two
fp32 forwards) and one backward of the logits against the softmax Fisher of
that difference, so it needs no double backward through attention.
"""

import torch

from .batch import actor_args, masked_log_probs


def _dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def natural_direction(actor, params, gradient, batches, *, iterations, damping, perturbation, chunk=128):
    start = [p.detach().clone() for p in params]
    rows = sum(len(mb["mask"]) for mb in batches)
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    # Differences of logits need full fp32 precision.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    def logits_at(shift):
        with torch.no_grad():
            for p, p0, x in zip(params, start, shift):
                p.copy_(p0 + x)
            return [actor(*actor_args(mb)).float() for mb in batches]

    def fisher(v):
        h = perturbation / max(_dot(v, v) ** 0.5, 1e-30)
        plus = logits_at([h * x for x in v])
        minus = logits_at([-h * x for x in v])
        with torch.no_grad():
            for p, p0 in zip(params, start):
                p.copy_(p0)
        total = [torch.zeros_like(p) for p in params]
        for mb, prob, lp, lm in zip(batches, probabilities, plus, minus):
            u = ((lp - lm) / (2 * h)).masked_fill(~mb["mask"], 0)
            coefficient = prob * u - prob * (prob * u).sum(-1, keepdim=True)
            for s in range(0, len(u), chunk):
                part = {key: value[s : s + chunk] for key, value in mb.items()}
                logits = actor(*actor_args(part)).float()
                loss = (logits * coefficient[s : s + chunk]).sum() / rows
                for acc, g in zip(total, torch.autograd.grad(loss, params, allow_unused=True)):
                    if g is not None:
                        acc += g
        return total

    try:
        with torch.no_grad():
            probabilities = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]).exp() for mb in batches]
        x = [torch.zeros_like(g) for g in gradient]
        r = [g.detach().clone() for g in gradient]
        d = [g.detach().clone() for g in gradient]
        rr = first = _dot(r, r)
        # Damping relative to the curvature along the gradient keeps it scale-free.
        lam, curvature, used = None, 0.0, 0
        for used in range(1, iterations + 1):
            fd = fisher(d)
            if lam is None:
                curvature = _dot(d, fd) / max(rr, 1e-30)
                lam = damping * curvature
            ad = [f + lam * v for f, v in zip(fd, d)]
            alpha = rr / max(_dot(d, ad), 1e-30)
            x = [xi + alpha * v for xi, v in zip(x, d)]
            r = [ri - alpha * a for ri, a in zip(r, ad)]
            new = _dot(r, r)
            if new < 1e-6 * first:
                break
            d = [ri + (new / rr) * v for ri, v in zip(r, d)]
            rr = new
        cosine = _dot(x, gradient) / max((_dot(x, x) * first) ** 0.5, 1e-30)
        return x, {"cg_iterations": used, "cg_residual": (_dot(r, r) / first) ** 0.5,
                   "fisher_curvature": curvature, "natural_cos": cosine}
    finally:
        with torch.no_grad():
            for p, p0 in zip(params, start):
                p.copy_(p0)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
