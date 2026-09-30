"""Path B, the self-draw simulation: a hand is worth what its own future draws make of it.

Only self-draws win, so a kept hand is worth the chance I draw its win before
an opponent does, times what that win pays. Instead of deriving that chance
from formulas, each candidate is played forward: the tiles I cannot see are
shuffled and I draw them in turn, discarding by a quick greedy rule, until the
hand wins or my draws run out (native.c). A win at my t-th draw counts its fan
plus the payment it saves, weighted by the chance no opponent has won by then
(the measured per-round hazard). All candidates meet the same shuffles, so
their differences are measured, not their noise. Discards keep only the hands
closest to a win and let the simulation choose among them; a call is taken
when the best hand it leaves simulates better than passing. Every win is
taken, and no kong is made.
"""
COMBINATION = "firstprinciples-rule"

from mahjong_ai import _fast
from mahjong_ai.sdk.shape import remaining_tiles

native = _fast.load_native(__file__)

LAST_WALL = 20  # the hand is drawn once the wall is down to this
SAVED = 0.467  # my win cancels the payment to another winner, in fan units
SIMS = 64


def my_draws(wall):
    return max(1, -(-(wall - LAST_WALL) // 4))


def closest(options, opened):
    """The options (counts, action) whose kept hand is closest to a win."""
    dist = [_fast.shanten_raw(c, opened) for c, _ in options]
    best = min(dist)
    return [o for o, d in zip(options, dist) if d == best]


class Sim:
    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass

    def values(self, hands, opened, unseen, draws, seed):
        return native.sim_values(hands, opened, unseen, draws, SIMS, seed, SAVED)

    def best_discard(self, counts, opened, unseen, draws, seed, actions):
        options = []
        for a in actions:
            rest = list(counts)
            rest[a.tile] -= 1
            options.append((rest, a))
        options = closest(options, opened)
        if len(options) == 1:
            return options[0][1], None
        vals = self.values([c for c, _ in options], opened, unseen, draws, seed)
        i = max(range(len(vals)), key=vals.__getitem__)
        return options[i][1], vals[i]

    def act(self, obs):
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        for a in legal:
            if a.action == "hu":
                return a
        seat = obs.stage.seat
        opened = len(obs.melds[seat])
        unseen = list(remaining_tiles(obs))
        draws = my_draws(obs.wall_remaining)
        counts = list(_fast.counts(tuple(obs.hand)))
        seed = hash((tuple(sorted(obs.hand)), obs.wall_remaining, seat)) & 0xFFFFFFFF
        if obs.phase == "draw":
            discards = [a for a in legal if a.action == "discard"]
            return self.best_discard(counts, opened, unseen, draws, seed, discards)[0]
        choice = next(a for a in legal if a.action == "pass")
        best = self.values([counts], opened, unseen, draws, seed)[0]
        for a in legal:
            if a.action not in ("chi", "peng"):
                continue
            kept = list(counts)
            for t in (a.tiles if a.action == "chi" else (a.tile, a.tile)):
                kept[t] -= 1
            discards = [type(a)("discard", t) for t in range(34) if kept[t]]
            options = closest([(self._minus(kept, d.tile), d) for d in discards], opened + 1)
            vals = self.values([c for c, _ in options], opened + 1, unseen, draws, seed)
            if max(vals) > best:
                best, choice = max(vals), a
        return choice

    @staticmethod
    def _minus(counts, tile):
        rest = list(counts)
        rest[tile] -= 1
        return rest


def build(model_dir):
    return Sim()
