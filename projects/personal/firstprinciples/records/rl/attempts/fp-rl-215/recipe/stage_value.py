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


def _place_value(totals, places):
    """The learner's (column 0) share of the first `places` places, per row of `totals` (..., 4); ties split."""
    mine, rest = totals[..., :1], totals[..., 1:]
    above = (rest > mine).sum(-1)
    tied = (rest == mine).sum(-1)
    lo, hi = above + 1, above + 1 + tied
    return np.where(hi <= places, 1.0, np.where(lo > places, 0.0, (places - lo + 1) / np.maximum(hi - lo + 1, 1)))


class ScoreLibrary:
    """The stage's remaining scores as this rollout's games played them (Suphx's GRP, by simulation).

    advance() above takes each player's remaining gain as an independent normal.
    The games say otherwise: the four scores of a hand sum to zero, the dealer
    pays and takes eight times, and wins are heavy-tailed. Here a stage's rest
    is drawn from the rollout itself: the current game's remaining hands as the
    remainder of a recorded game from the same hand on, and each later game as
    a whole recorded game, the three opponents' columns shuffled per game. All
    by the learner's seat (column 0). A deviation's copy and its hand are
    valued on the same draws, so their difference carries little of the
    simulation's noise.
    """

    def __init__(self, finals, suffixes, rounds):
        self.finals, self.suffixes, self.rounds = finals, suffixes, rounds

    @classmethod
    def from_rows(cls, rows, first, last, game, rounds):
        cumulative = {}  # game -> {hand: learner-relative game scores after it}
        for k, end in enumerate(last):
            r = rows[end]
            if "end_scores" not in r:
                continue
            seat = int(r["seat"])
            scores = r["end_scores"]
            cumulative.setdefault(int(game[first[k]]), {})[int(r["hand"])] = np.array(
                [float(scores[(seat + j) % 4]) for j in range(4)])
        finals, suffixes = [], [[] for _ in range(rounds)]
        for hands in cumulative.values():
            if rounds not in hands:
                continue  # an opening's one hand, or a game cut short
            final = hands[rounds]
            finals.append(final)
            suffixes[0].append(final)
            for h, c in hands.items():
                if h < rounds:
                    suffixes[h].append(final - c)
        finals = np.array(finals)
        suffixes = [np.array(s) if s else None for s in suffixes]
        return cls(finals, suffixes, rounds)

    def draws(self, hand, full_games, count, rng):
        """`count` draws (count x 4) of the stage's remaining scores after `hand` of this game and `full_games` more."""
        h = int(hand)
        while 0 < h < self.rounds and self.suffixes[h] is None:
            h -= 1  # a hand no recorded game ended a row on: the nearest earlier one
        if h >= self.rounds:
            out = np.zeros((count, 4))
        else:
            pool = self.suffixes[h]
            out = pool[rng.integers(len(pool), size=count)].copy()
        for _ in range(int(full_games)):
            g = self.finals[rng.integers(len(self.finals), size=count)].copy()
            order = np.argsort(rng.random((count, 3)), axis=1) + 1
            g[:, 1:] = np.take_along_axis(g, order, axis=1)
            out += g
        return out

    def advance_pair(self, copy, main, hand, full_games, places, rng, count=1024):
        """advance()'s chance for two sets of current totals (n x 4) sharing their rows' draws."""
        hand, full_games = np.asarray(hand), np.asarray(full_games)
        v_copy, v_main = np.empty(len(copy)), np.empty(len(main))
        for key in sorted(set(zip(hand.tolist(), full_games.tolist()))):
            sel = np.flatnonzero((hand == key[0]) & (full_games == key[1]))
            rest = self.draws(key[0], key[1], count, rng)  # count x 4
            v_copy[sel] = _place_value(copy[sel, None, :] + rest[None], places).mean(1)
            v_main[sel] = _place_value(main[sel, None, :] + rest[None], places).mean(1)
        return v_copy, v_main
