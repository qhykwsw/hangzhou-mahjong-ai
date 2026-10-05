"""Training-only hands: a stable wall and the learner's draw luck (cfg["stable_wall"], cfg["luck"]).

A playout's paired difference (deviation versus argmax, same wall) stops being
paired as soon as one branch calls: every later draw shifts to another seat.
With a stable wall each seat's normal draws come from its own interleaved
positions of the wall, counted from the back as of the conversion (seat
``next`` takes the back position, the seat after it the one before, and so
on round the table, four positions apart), so a call no longer changes what
anyone draws; kong replacements still come from the front. Without calls the
draws are exactly the ordinary wall's, and since the wall is a uniform
shuffle every draw is still uniform over the unseen tiles: the game is the
same in law. When a seat's next position is gone (drawn as a kong
replacement, or past the front) it draws the back-most remaining tile.

The learner's draw luck is an AIVAT control variate. At every draw (after
the dealer's first) by a ready seat, f(y) is the learner's payoff, in base
units, if the drawer won at once with tile y, and d(y) whether the learner
would then deal next; the luck accumulates f(drawn) - E f and d(drawn) - E d,
the means over the remaining wall (the next tile is uniform over it whatever
the wall). Both have mean zero given the history, so a value minus its luck
since a decision keeps its expectation and loses who-hits-the-wait chance:
measured on 2548 deviations of fp-rl-172 against the fixed pool, the stable
wall and the luck together cut the paired difference's variance about
threefold (within-point variance 304 -> 98).

``native.Hand`` (native.c) implements both bit for bit; ``TrainingHand`` is
the Python engine's, for the games that stay on it (openings, searches, rule
teachers) and for replays of C games.
"""

from mahjong_ai import _fast
from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.rules import WHITE


class TrainingHand(HandState):
    """A HandState with an optional stable wall and draw-luck tracking (see the module docstring)."""

    def _draw(self, seat, kong=False, initial=False):
        if len(self.wall) <= 20:
            self._finish(None)
            return
        expect = None
        if not initial and self.luck_seat >= 0:
            expect = self._luck_expect(seat, kong)
        self.turn = seat
        self.drawn = self._take(seat, kong)
        self.hands[seat].append(self.drawn)
        if self.drawn == WHITE:
            self.white_acquired[seat] += 1
        self.kong_draw = kong
        self.phase = "draw"
        if not initial:
            self._event("draw", seat, self.drawn, private=seat)
        if expect is not None:
            counts = list(_fast.counts(tuple(self.hands[seat])))
            fan = _fast.fan_value(counts, self.drawn, len(self.melds[seat]), self.chain[seat], self.piao[seat],
                                  int(self.config.you_cai_bi_kao), int(kong))
            got = self._luck_payoff(seat, fan) if fan > 0 else 0.0
            dealt = 1.0 if fan > 0 and seat == self.luck_seat else 0.0
            self.luck = (self.luck[0] + (got - expect[0]), self.luck[1] + (dealt - expect[1]))

    def _take(self, seat, kong):
        if self._positions is None:
            return self.wall.pop(0) if kong else self.wall.pop()
        if kong:
            k = 0
        else:
            target = self._back - (4 * self._next[seat] + self._offset[seat])
            self._next[seat] += 1
            k = self._index.get(target, len(self.wall) - 1)
        position = self._positions.pop(k)
        del self._index[position]
        if k < len(self._positions):
            for i in range(k, len(self._positions)):
                self._index[self._positions[i]] = i
        return self.wall.pop(k)

    def _luck_payoff(self, seat, fan):
        learner = self.luck_seat
        if seat == learner:
            return sum(float(fan * (8 if self.dealer in (t, seat) else 1)) for t in range(4) if t != seat)
        return -float(fan * (8 if self.dealer in (learner, seat) else 1))

    def _luck_expect(self, seat, kong):
        opened = len(self.melds[seat])
        counts = list(_fast.counts(tuple(self.hands[seat])))
        if _fast.shanten_raw(counts, opened) > 0:
            return None
        remaining = [0] * 34
        for tile in self.wall:
            remaining[tile] += 1
        pay = deal = 0.0
        ycbk = int(self.config.you_cai_bi_kao)
        for y in range(34):
            if not remaining[y]:
                continue
            counts[y] += 1
            fan = _fast.fan_value(counts, y, opened, self.chain[seat], self.piao[seat], ycbk, int(kong))
            counts[y] -= 1
            if fan > 0:
                pay += float(remaining[y]) * self._luck_payoff(seat, fan)
                if seat == self.luck_seat:
                    deal += float(remaining[y])
        return pay / float(len(self.wall)), deal / float(len(self.wall))


def train_hand(state, *, stable=False, luck_seat=None):
    """Make a HandState a TrainingHand in place: a stable wall from here on, and/or a seat's draw luck."""
    if not isinstance(state, TrainingHand):
        state.__class__ = TrainingHand
        state._positions = state._index = None
        state.luck_seat, state.luck = -1, (0.0, 0.0)
    if luck_seat is not None:
        state.luck_seat = luck_seat
    if stable and state._positions is None:
        following = state.turn if state.phase == "draw" else state.discarder
        nxt = (following + 1) % 4
        state._positions = list(range(len(state.wall)))
        state._index = {p: i for i, p in enumerate(state._positions)}
        state._back = len(state.wall) - 1
        state._next = [0, 0, 0, 0]
        state._offset = [(s - nxt) % 4 for s in range(4)]
    return state
