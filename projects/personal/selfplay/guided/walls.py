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
threefold (within-point variance 304 -> 98). This is cfg["luck_kind"]
"immediate", the default.

The race luck (cfg["luck_kind"] = "race", kept apart from the luck above,
which is still tracked) is a fuller control variate over the same draws:
every draw, the four seats' and kong replacements alike, also moves who is
going to win the hand, by taking a tile somebody waits on or leaving it in
the wall. Before a draw by seat s, with W the remaining wall (n tiles, w_y
copies of tile y), for each tile y

    v(y) = f(y)                when y wins for s at once (f as above, the kong flag of this draw),
    v(y) = R(W - y, s + 1)     otherwise,

and the race luck accumulates v(drawn) - sum_y w_y v(y) / n, in a pay part
(base units) and a deal part. With u(y) = v(y) - R(W, s), the race value's
change over the draw, this is u(drawn) - E u (R(W, s) does not depend on y).
v reads only the state before the draw (every hand, the wall's tiles, whose
turn it is) and the tile, so given the history the increment has mean zero
whatever R is, as the luck above does. R is the race value of the hand as it
stands: each ready seat p (shanten_raw <= 0) wins on its tiles y with
fan_value > 0 (no kong flag), worth the learner's payoff for p self-drawing
y; h_p is the number of such copies in W and S_p the sum of the payoffs
over them; seats draw in turn from ``next`` until the wall is down to 20
tiles, D = n - 20 draws, p's own at positions o_p + 4i (o_p = (p - next)
mod 4), m_p of them. p's own draws are a uniform sample of W without
replacement, so it misses its first i with miss_p(i) = prod_{j<i}
(n - h_p - j) / (n - j); taking the seats as independent,

    P_first(p) = sum_{i<m_p} miss_p(i) h_p / (n - i) * prod_{q != p} miss_q(c_q),
    R_pay      = sum_p P_first(p) S_p / h_p,
    R_deal     = P_first(learner) + [learner deals] prod_p miss_p(m_p),

c_q the draws q makes before p's draw i (the dealer keeps the deal when
nobody wins). The approximations (they cost variance reduction, never bias):
no call or kong reorders the draws; the waits stay as they are (a seat not
ready stays so, a ready one keeps its wait, chain and piao, s lets go what it
drew); the seats' misses are independent (exact with one ready seat, and
overlapping waits count for each); a seat that can win does. Every y that
no seat waits on gives the same R. The arithmetic runs in the same order in
Python and C.

``native.Hand`` (native.c) implements all of this bit for bit; ``TrainingHand``
is the Python engine's, for the games that stay on it (openings, searches,
rule teachers) and for replays of C games.
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
        expect = race = None
        if not initial and self.luck_seat >= 0:
            expect = self._luck_expect(seat, kong)
            if self.race_tracked:
                race = self._race_values(seat, kong)
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
        if race is not None:
            values, mean = race
            pay, deal = values[self.drawn]
            self.race_luck = (self.race_luck[0] + (pay - mean[0]), self.race_luck[1] + (deal - mean[1]))

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

    def _race_values(self, seat, kong):
        """Before ``seat`` draws: v(y) (pay, deal) per tile of the wall and their means, or None when nobody can win."""
        remaining = [0] * 34
        for tile in self.wall:
            remaining[tile] += 1
        n = len(self.wall)
        ycbk = int(self.config.you_cai_bi_kao)
        racers = []  # (seat, copies that win, sum of the payoffs over them, wins per tile, payoff per tile)
        now, now_pay, anything_now = [False] * 34, [0.0] * 34, False
        for p in range(4):
            opened = len(self.melds[p])
            counts = list(_fast.counts(tuple(self.hands[p])))
            if _fast.shanten_raw(counts, opened) > 0:
                continue
            hits, total = 0, 0.0
            wins, pays = [False] * 34, [0.0] * 34
            for y in range(34):
                if not remaining[y]:
                    continue
                counts[y] += 1
                fan = _fast.fan_value(counts, y, opened, self.chain[p], self.piao[p], ycbk, 0)
                at_once = fan
                if p == seat and kong:
                    at_once = _fast.fan_value(counts, y, opened, self.chain[p], self.piao[p], ycbk, 1)
                counts[y] -= 1
                if p == seat and at_once > 0:
                    now[y], now_pay[y], anything_now = True, self._luck_payoff(p, at_once), True
                if fan > 0:
                    wins[y], pays[y] = True, self._luck_payoff(p, fan)
                    hits += remaining[y]
                    total += float(remaining[y]) * pays[y]
            if hits > 0:
                racers.append((p, hits, total, wins, pays))
        if not racers and not anything_now:
            return None
        learner, nxt = self.luck_seat, (seat + 1) % 4
        deals = self.dealer == learner
        missed = race_value(n - 1, nxt, [(p, h, t) for p, h, t, _, _ in racers], learner, deals)
        values = [None] * 34
        pay = deal = 0.0
        for y in range(34):
            if not remaining[y]:
                continue
            if now[y]:
                v = (now_pay[y], 1.0 if seat == learner else 0.0)
            elif not any(r[3][y] for r in racers):
                v = missed
            else:
                left = []
                for p, h, t, wins, pays in racers:
                    if not wins[y]:
                        left.append((p, h, t))
                    elif h > 1:
                        left.append((p, h - 1, t - pays[y]))
                v = race_value(n - 1, nxt, left, learner, deals)
            values[y] = v
            pay += float(remaining[y]) * v[0]
            deal += float(remaining[y]) * v[1]
        return values, (pay / float(n), deal / float(n))


def race_value(n, first, racers, learner, learner_deals):
    """R (pay, deal) of a wall of n tiles, seats drawing in turn from ``first``.

    ``racers``: (seat, h, S) of each seat that can win, in seat order, h > 0
    (see the module docstring); native.c's g_race runs the same arithmetic.
    """
    draws = n - 20
    plan = []
    for p, hits, total in racers:
        off = (p - first) % 4
        m = (draws - off + 3) // 4 if draws > off else 0
        miss = [1.0]
        for j in range(m):
            keep = n - hits - j
            miss.append(miss[j] * ((keep if keep > 0 else 0) / (n - j)))
        plan.append((p, hits, total, off, m, miss))
    pay, none, mine = 0.0, 1.0, 0.0
    for i, (p, hits, total, off, m, miss) in enumerate(plan):
        first_p = 0.0
        for j in range(m):
            k = off + 4 * j
            others = 1.0
            for q, (_, _, _, off_q, _, miss_q) in enumerate(plan):
                if q != i:
                    others *= miss_q[(k - off_q + 3) // 4 if k > off_q else 0]
            first_p += miss[j] * hits / (n - j) * others
        pay += first_p * (total / hits)
        if p == learner:
            mine = first_p
        none *= miss[m]
    return pay, (mine + none if learner_deals else mine)


def train_hand(state, *, stable=False, luck_seat=None, race=False):
    """Make a HandState a TrainingHand in place: a stable wall from here on, and/or a seat's draw luck
    (with ``race``, its race luck too)."""
    if not isinstance(state, TrainingHand):
        state.__class__ = TrainingHand
        state._positions = state._index = None
        state.luck_seat, state.luck = -1, (0.0, 0.0)
        state.race_tracked, state.race_luck = False, (0.0, 0.0)
    if luck_seat is not None:
        state.luck_seat = luck_seat
        state.race_tracked = state.race_tracked or bool(race)
    if stable and state._positions is None:
        following = state.turn if state.phase == "draw" else state.discarder
        nxt = (following + 1) % 4
        state._positions = list(range(len(state.wall)))
        state._index = {p: i for i, p in enumerate(state._positions)}
        state._back = len(state.wall) - 1
        state._next = [0, 0, 0, 0]
        state._offset = [(s - nxt) % 4 for s in range(4)]
    return state
