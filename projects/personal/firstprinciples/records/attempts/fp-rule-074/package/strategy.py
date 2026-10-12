"""Path B, the self-draw simulation: a hand is worth what its own future draws make of it.

Only self-draws win, so a kept hand is worth the chance I draw its win before
an opponent does, times what that win pays. Instead of deriving that chance
from formulas, each candidate is played forward: the tiles I cannot see are
shuffled and dealt four per round as the wall is: three thrown by the
opponents, which I call when that brings the hand closer to a win, and one I
draw, discarding by a quick greedy rule, until the hand wins or my draws run
out (native.c). A win at my t-th draw counts its fan
plus the payment it saves, weighted by the chance no opponent has won by then
(the measured per-round hazard). All candidates meet the same shuffles, so
their differences are measured, not their noise. Discards keep only the hands
closest to a win and let the simulation choose among them; a call is taken
when the best hand it leaves simulates better than passing. A win is taken
unless going on simulates better: its fan plus the saved payment against the
best discard's simulated value, where a white thrown from a baotou wait that
stays baotou scores piao and doubles what the wait then wins; there the
opponents' hazard is scaled by how fast they look from their melds and
discards (a logistic model measured against the pool). A kong is made
when the hand it leaves, drawing its replacement at once (a win on it
doubles), simulates better than the best discard or reply.
"""
COMBINATION = "firstprinciples-rule"

import math

from mahjong_ai import _fast
from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.rules import WHITE
from mahjong_ai.sdk.shape import remaining_tiles

native = _fast.load_native(__file__)

LAST_WALL = 20  # the hand is drawn once the wall is down to this
SAVED = 0.467  # my win cancels the payment to another winner, in fan units
SIMS = 64
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}


def my_draws(wall):
    return max(1, -(-(wall - LAST_WALL) // 4))


# Measured against the fixed pool (logistic fit over 200k opponent-rounds): the
# log-odds that one opponent self-draws before my next draw, weighting in order
# 1, my draws left d, d*d/16, its melds being 1 / 2 / 3+, melds*d, middle suit
# tiles among its last three discards, whether it has three yet, middle tiles
# among its last six, honors and terminals among its last three, its discards
# /10, its discards since its last call /9 (9 without a call), whether it
# deals, and whether it has discarded a white (a piao: it waits on baotou).
OPPONENT_LOGIT = (-0.0571, -0.1131, -0.2408, -0.5687, -0.2777, 0.1483, 0.0804, 0.0021, 1.0988, 0.1779, -0.4002, -0.1582, -1.9073, -0.6204, 0.1573, 5.1224)


def my_draws(wall):
    return max(1, -(-(wall - LAST_WALL) // 4))


def round_hazard(draws):
    return min(0.30, max(0.10, 0.35 - 0.018 * draws))


def opponent_speed(obs, draws):
    """The opponents' chance to win before my next draw, from their melds and
    discards, as a multiple of the generic per-round hazard (0.1-3)."""
    hand_no = obs.public_events[-1].hand_no if obs.public_events else 0
    since = {}
    for event in obs.public_events:
        if event.hand_no != hand_no:
            continue
        if event.kind in ("chi", "peng", "gang"):
            since[event.seat] = 0
        elif event.kind == "discard" and event.seat in since:
            since[event.seat] += 1
    d = min(draws, 16)
    stay = 1.0
    for seat in range(4):
        if seat == obs.stage.seat:
            continue
        discards = obs.discards[seat]
        last3, last6 = discards[-3:], discards[-6:]
        melds = min(3, len(obs.melds[seat]))
        features = (
            1, d, d * d / 16, melds == 1, melds == 2, melds == 3, (melds >= 1) * d,
            sum(1 for t in last3 if t < 27 and 2 <= t % 9 <= 6), len(last3) == 3,
            sum(1 for t in last6 if t < 27 and 2 <= t % 9 <= 6),
            sum(1 for t in last3 if t >= 27), sum(1 for t in last3 if t < 27 and t % 9 in (0, 8)),
            len(discards) / 10, min(since.get(seat, 9), 9) / 9, obs.stage.dealer == seat, WHITE in discards,
        )
        logit = sum(w * x for w, x in zip(OPPONENT_LOGIT, features))
        stay *= 1 - 1 / (1 + math.exp(-logit))
    return min(3.0, max(0.1, (1 - stay) / round_hazard(draws)))


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

    def values(self, hands, opened, unseen, draws, seed, chis=0, chain=0, piao=0, kong=0, speed=1.0):
        return native.sim_values(hands, opened, chis, unseen, draws, SIMS, seed, SAVED, chain, piao, kong, speed)

    def kong_value(self, counts, a, opened, unseen, draws, seed, chis, obs):
        """The simulated value of making kong a: the rest of the hand, one more
        meld, drawing its replacement at once."""
        rest = list(counts)
        rest[a.tile] -= KONG_TILES[a.kind]
        after = opened if a.kind == "added" else opened + 1
        return self.values([rest], after, unseen, draws, seed, chis, obs.chain_count, obs.piao_count, 1)[0]

    def at_win(self, obs, counts, opened, unseen, draws, seed, chis, hu):
        """Win now, or go on when a discard (or a piao) simulates better."""
        now = obs.hu_fan + SAVED
        speed = opponent_speed(obs, draws)
        discards = [a for a in obs.legal_actions if a.action == "discard"]
        options = []
        for a in discards:
            rest = list(counts)
            rest[a.tile] -= 1
            options.append((rest, a))
        options = closest(options, opened)
        best, choice = now, hu
        vals = self.values([c for c, _ in options], opened, unseen, draws, seed, chis, obs.chain_count, obs.piao_count, 0, speed)
        for (c, a), v in zip(options, vals):
            if v > best:
                best, choice = v, a
        piao = next((a for a in discards if a.tile == WHITE), None)
        if piao is not None and can_discard_piao(
            obs.hand, obs.drawn_tile, opened, obs.public_events, obs.stage.seat, obs.stage.round_no,
            catch_exempt=obs.catch_play and not obs.catch_restricted,
        ):
            rest = list(counts)
            rest[WHITE] -= 1
            v = self.values([rest], opened, unseen, draws, seed, chis, obs.chain_count + 1, obs.piao_count + 1, 0, speed)[0]
            if v > best:
                best, choice = v, piao
        return choice

    def best_discard(self, counts, opened, unseen, draws, seed, actions, chis=0):
        options = []
        for a in actions:
            rest = list(counts)
            rest[a.tile] -= 1
            options.append((rest, a))
        options = closest(options, opened)
        if len(options) == 1:
            return options[0][1], None
        vals = self.values([c for c, _ in options], opened, unseen, draws, seed, chis)
        i = max(range(len(vals)), key=vals.__getitem__)
        return options[i][1], vals[i]

    def act(self, obs):
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        seat = obs.stage.seat
        opened = len(obs.melds[seat])
        unseen = list(remaining_tiles(obs))
        draws = my_draws(obs.wall_remaining)
        counts = list(_fast.counts(tuple(obs.hand)))
        seed = hash((tuple(sorted(obs.hand)), obs.wall_remaining, seat)) & 0xFFFFFFFF
        chis = sum(1 for m in obs.melds[seat] if m.kind == "chi")
        hu = next((a for a in legal if a.action == "hu"), None)
        if hu is not None:
            return self.at_win(obs, counts, opened, unseen, draws, seed, chis, hu)
        if obs.phase == "draw":
            discards = [a for a in legal if a.action == "discard"]
            choice, value = self.best_discard(counts, opened, unseen, draws, seed, discards, chis)
            kongs = [a for a in legal if a.action == "gang"]
            if kongs:
                if value is None:
                    rest = list(counts)
                    rest[choice.tile] -= 1
                    value = self.values([rest], opened, unseen, draws, seed, chis)[0]
                for a in kongs:
                    v = self.kong_value(counts, a, opened, unseen, draws, seed, chis, obs)
                    if v > value:
                        value, choice = v, a
            return choice
        choice = next(a for a in legal if a.action == "pass")
        best = self.values([counts], opened, unseen, draws, seed, chis)[0]
        for a in legal:
            if a.action == "gang":
                v = self.kong_value(counts, a, opened, unseen, draws, seed, chis, obs)
                if v > best:
                    best, choice = v, a
                continue
            if a.action not in ("chi", "peng"):
                continue
            kept = list(counts)
            for t in (a.tiles if a.action == "chi" else (a.tile, a.tile)):
                kept[t] -= 1
            discards = [type(a)("discard", t) for t in range(34) if kept[t]]
            options = closest([(self._minus(kept, d.tile), d) for d in discards], opened + 1)
            vals = self.values([c for c, _ in options], opened + 1, unseen, draws, seed, chis + (a.action == "chi"))
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
