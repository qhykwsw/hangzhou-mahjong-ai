"""Stage value for rank objectives: the chance of advancing from a stage state.

A semifinal advances the top two of four on the stage's total score, a final
crowns one. From the current totals s (the learner's first, the others after,
any order) with n hands left, each player's remaining gain is taken as an
independent normal with sd per_hand_sd * sqrt(n) (the pool's games: about 70
per 16-hand game); given the learner's final y, another player ends above it
with p_j(y) = Phi((s_j - y) / sigma), and

  P(top two) = E_y[ prod_j (1 - p_j) + sum_j p_j prod_{k != j} (1 - p_k) ]
  P(first)   = E_y[ prod_j (1 - p_j) ]

by Gauss-Hermite quadrature over y. With no hands left the places are exact
(a tie straddling the line counts half, as harness.stage scores it).
"""
import math

import numpy as np

NODES, WEIGHTS = np.polynomial.hermite.hermgauss(32)
WEIGHTS = WEIGHTS / math.sqrt(math.pi)


def _phi(x):
    """Standard normal CDF (Abramowitz-Stegun 7.1.26 for erf, error below 1.5e-7)."""
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def advance(scores, hands_left, places, per_hand_sd=17.5):
    """P(the learner, column 0 of `scores` (n x 4), finishes within `places`), per row."""
    s = np.asarray(scores, dtype=np.float64)
    left = np.broadcast_to(np.asarray(hands_left, dtype=np.float64), s.shape[:1])
    out = np.empty(len(s))
    done = left <= 0
    if done.any():
        mine, rest = s[done, :1], s[done, 1:]
        above = (rest > mine).sum(1)
        tied = (rest == mine).sum(1)
        # midrank of the learner among the four; a tie straddling the line counts half
        lo, hi = above + 1, above + 1 + tied
        out[done] = np.where(hi <= places, 1.0, np.where(lo > places, 0.0, (places - lo + 1) / (hi - lo + 1)))
    live = ~done
    if live.any():
        sigma = per_hand_sd * np.sqrt(left[live])[:, None, None]
        y = s[live, :1, None] + math.sqrt(2.0) * sigma * NODES[None, None, :]
        p = _phi((s[live, 1:, None] - y) / sigma)  # rows x 3 x nodes: other j ends above
        q = 1.0 - p
        none = q.prod(1)
        if places >= 2:
            one = sum(p[:, j] * np.prod(np.delete(q, j, axis=1), axis=1) for j in range(3))
            val = none + one
        else:
            val = none
        out[live] = (val * WEIGHTS[None, :]).sum(1)
    return out
