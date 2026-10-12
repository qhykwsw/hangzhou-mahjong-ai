"""An all-action control variate for the sampled deviations (cfg["cv"]; optimizer item 2, shrinkage).

The playout PG estimates, per state, boost * sum_{a != argmax} pi(a) D(a)
grad log pi(a) from one sampled deviation's paired difference D (weight w,
see loop.boosted_deviations). With a Q network's prediction Q(s, a) of D
(mpo.Improver, fitted on earlier steps' deviations only), the estimator

    w (D - c Q(s, a)) grad log pi(a)            on the deviation rows
  + c boost sum_a pi(a) Q(s, a) grad log pi(a)  over the step's states

has the same expectation for any c (the second term is the first's
expectation for Q), and less variance when Q tracks D: c* = cov(D, Q) / var(Q),
the held-out calibration slope the Improver measures on each step's new
deviations before it trains on them, clipped to [0, 1] (shrinkage: c = 0 is
the plain PG). Q never sees the deviations it corrects: their predictions and
the dense term use the network as it was before the step's fit.
"""

import numpy as np
import torch

from .batch import make_batch


def predicted_gaps(improver, actor, rows, index):
    """Q(s, a) - Q(s, argmax) for the rows at `index` (their sampled actions), in D units."""
    if not len(index):
        return np.zeros(0)
    picked = [rows[i] for i in index]
    greedy = improver.argmax(actor, picked)
    out = []
    improver.q.eval()
    with torch.no_grad():
        for s in range(0, len(picked), 512):
            mb = make_batch(picked[s:s + 512], improver.device)
            g = torch.as_tensor(greedy[s:s + 512], device=improver.device)
            out.append((improver._pred(mb, mb["action"], g) * improver.scale).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def coefficient(stats, spec):
    """c from the Improver's held-out fit of the newest deviations: the slope, clipped, 0 without evidence."""
    slope, corr = stats.get("q_all_slope"), stats.get("q_all_corr")
    if slope is None or corr is None or corr <= 0:
        return 0.0
    return float(np.clip(slope, 0.0, float(spec.get("max", 1.0))))
