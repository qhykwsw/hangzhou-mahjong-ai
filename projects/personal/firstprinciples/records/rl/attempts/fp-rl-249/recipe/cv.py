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


def rule_gaps(rows, cfg, stats, index):
    """The 131-continuation paired difference of each deviation at `index` (cfg["cv"]["label"] == "rule").

    The branch copy of the deviation and its control both continue with
    fp-rule-131 (portfolio.py's branches), so the difference does not depend
    on the current Actor: a stationary label the Q network can keep learning
    across steps (the label cache). Luck since the fork is subtracted as in
    the playout advantages; no boost weight (the Q network fits the gap).
    """
    hand = np.array([r["hand"] for r in rows])
    if "game_mark" in rows[0]:
        mark = np.array([r["game_mark"] for r in rows])
        game = np.cumsum(np.r_[1, (np.diff(mark) != 0).astype(int)]) - 1
    else:
        game = np.cumsum(np.r_[1, (np.diff(hand) < 0).astype(int)]) - 1
    new_hand = np.r_[True, (np.diff(hand) != 0) | (np.diff(game) != 0)]
    hand_id = np.cumsum(new_hand) - 1
    first = np.flatnonzero(new_hand)
    last_hand = ~np.r_[game[first[1:]] == game[first[:-1]], False]
    premium, luck = stats["premium"], bool(cfg.get("luck"))

    def value(r, i):
        scale = 0.0 if last_hand[hand_id[i]] else premium
        v = float(r["playout"]) + scale * float(r["playout_deals_next"])
        if luck:
            v -= float(r.get("playout_luck_pay", 0.0)) + scale * float(r.get("playout_luck_deal", 0.0))
        return v

    out = np.full(len(index), np.nan)
    for j, i in enumerate(index):
        r = rows[i]
        b, c = (r.get("branches") or [None])[0], r.get("branch_control")
        if b is None or c is None or b.get("truncated") or c.get("truncated"):
            continue
        out[j] = value(b, i) - value(c, i)
    return out
