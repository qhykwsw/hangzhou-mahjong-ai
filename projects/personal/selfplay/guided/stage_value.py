"""Stage value for rank objectives: the chance of advancing from a stage state, and its risk coefficient.

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

The risk coefficient beta (guided-current-hand-v4's last scalar, risk_input)
is the curvature of that value in the learner's own score. With
V(s) = advance([s, the other three's stage totals], hands left, places),

  lambda = V'' / (2 V'),   beta = -2 lambda = -V'' / V'   (per base score),

so a small gain X of the learner's is worth about E[X] - (beta / 2) Var[X]
points: beta > 0 where V is concave (ahead of the line: the steadier of two
plays of equal mean is worth more), beta < 0 where it is convex (behind it:
the wider one is). Choices:
  - scores in base-score units (stage_scores / base_score), the unit of
    per_hand_sd; V' and V'' by central differences with a step of one base
    score, the scores' own resolution;
  - hands left = stage_rounds_total - stage_round_no + 1, this hand included,
    and at least 1: StageContext.effective_horizon, the encoder's horizon
    (the game's own when an observation gives no stage horizon);
  - beta = 0 where V' < MIN_SLOPE (1e-6 per base score: the stage is as good
    as decided, and -V''/V' is then a ratio of two vanishing tails);
  - clipped to +-BETA_MAX = 0.05 per base score; the input is beta / BETA_MAX
    in [-1, 1];
  - no beta (None, an input of 0) outside a semifinal or a final or without
    stage scores: free play and the development set hold 0.

The same rows carry their ``stage_state`` (stage_state: the places that
count, the hands left with this one and the four totals from the learner in
turn order), which the stage re-rank of an Actor with the outcome head reads
(guided.outcome); it is not an Actor input.
"""
from functools import lru_cache
import math

import numpy as np

NODES, WEIGHTS = np.polynomial.hermite.hermgauss(32)
WEIGHTS = WEIGHTS / math.sqrt(math.pi)

# Places that count by stage kind (StageKind is a StrEnum: its members look up as these strings).
PLACES = {"advancement": 2, "final": 1}
BETA_MAX = 0.05  # per base score
STEP = 1.0  # base scores
MIN_SLOPE = 1e-6  # P(advance) per base score


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


def beta(scores, seat, hands_left, places):
    """The risk coefficient -V''/V' of ``seat`` (per base score, clipped), 0 once V' < MIN_SLOPE.

    ``scores``: the four stage totals by seat, in base scores.
    """
    own = float(scores[seat])
    others = [float(scores[(seat + k) % 4]) for k in (1, 2, 3)]
    up, mid, down = advance([[own + d, *others] for d in (STEP, 0.0, -STEP)], hands_left, places)
    slope = (up - down) / (2.0 * STEP)
    if not slope >= MIN_SLOPE:
        return 0.0
    curvature = (up - 2.0 * mid + down) / (STEP * STEP)
    return min(BETA_MAX, max(-BETA_MAX, -curvature / slope))


@lru_cache(maxsize=4096)
def risk_input(kind, scores, seat, current, total):
    """beta / BETA_MAX of ``seat`` (in [-1, 1]), or None outside a semifinal or a final.

    ``scores``: the four stage totals by seat in base scores, a tuple;
    ``current``, ``total``: the stage horizon (this hand's number, the hands in
    the stage). The same for a seat throughout a hand, hence cached.
    """
    places = PLACES.get(kind)
    if places is None or scores is None or len(scores) != 4:
        return None
    return beta(scores, seat, max(1, total - current + 1), places) / BETA_MAX


def risk_of(ctx, base_score=1):
    """risk_input of an observation's StageContext (obs.stage, obs.base_score)."""
    if PLACES.get(ctx.kind) is None or ctx.stage_scores is None:
        return None
    current, total = ctx.effective_horizon
    return risk_input(ctx.kind, tuple(s / base_score for s in ctx.stage_scores), ctx.seat, current, total)


def stage_state(kind, scores, seat, current, total):
    """A row's ``stage_state``: (places, hands left, the four totals from ``seat`` in turn order), or None.

    ``scores``: the four stage totals by seat in base scores; ``current``,
    ``total``: the stage horizon. None outside a semifinal or a final (as
    risk_input). Hands left count this hand, at least 1.
    """
    places = PLACES.get(kind)
    if places is None or scores is None or len(scores) != 4:
        return None
    return (places, max(1, total - current + 1), *(float(scores[(seat + k) % 4]) for k in range(4)))


def stage_state_of(ctx, base_score=1):
    """stage_state of an observation's StageContext (obs.stage, obs.base_score)."""
    if PLACES.get(ctx.kind) is None or ctx.stage_scores is None:
        return None
    current, total = ctx.effective_horizon
    return stage_state(ctx.kind, tuple(s / base_score for s in ctx.stage_scores), ctx.seat, current, total)
