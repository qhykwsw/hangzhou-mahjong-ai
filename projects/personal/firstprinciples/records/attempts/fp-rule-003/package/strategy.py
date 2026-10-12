"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Each opponent self-draws more often as
the hand goes on, so the three draws between two of mine end the hand with a
chance that rises draw by draw, and the wall allows only so many draws of mine.
A waiting hand that wins on a fraction p of draws, collecting e fan per draw on
average, is valued backwards from my last draw. One step from waiting, each
improving draw leads to the best waiting hand it allows; further away, typical
hands stand in for the steps between. A win also cancels the payment I would
owe to someone else's win, SAVED fan. When the hand is already baotou (every
draw wins), a win now is traded for a doubling that keeps the certain win: a
kong whose replacement draw still wins, else a white discard that scores piao.
"""
COMBINATION = "firstprinciples-rule"

from functools import lru_cache

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.rules import WHITE, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles

SAVED = 0.4  # my win cancels the payment to another winner, in fan units
DEALT_WALL = 83  # wall after the deal and the dealer's first draw
LAST_WALL = 20  # the hand is drawn once the wall is down to this
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}
# Far hands step down through typical ones: (improving share, next level).
TYPICAL_TENPAI = (0.06, 1.2)  # winning share of unseen tiles, fan
TYPICAL_STEP = (0.15, 0.3, 0.4)  # improving share one, two, three+ steps away


def opponent_hazard(n):
    """One opponent's chance to self-draw on their n-th draw of the hand."""
    return min(0.14, max(0.0, 0.012 * (n - 2.5)))


def segments(wall):
    """Chance an opponent wins before each of my remaining draws."""
    done = (DEALT_WALL - wall) / 4
    draws = max(1, -(-(wall - LAST_WALL) // 4))
    return tuple(1 - (1 - opponent_hazard(done + 1 + j)) ** 3 for j in range(draws))


def chase(qs, share, target):
    """Curve of a state that moves to ``target`` with chance ``share`` per draw."""
    curve = [0.0] * len(qs)
    for j in range(len(qs) - 2, -1, -1):
        live = 1 - qs[j + 1]
        curve[j] = live * (share * target[j + 1] + (1 - share) * curve[j + 1])
    return curve


def tenpai_curve(p, e, qs):
    curve = [0.0] * len(qs)
    after = 0.0
    for j in range(len(qs) - 1, -1, -1):
        curve[j] = e + (1 - p) * after
        after = (1 - qs[j]) * curve[j]
    return curve


@lru_cache(maxsize=4096)
def typical_curve(steps, qs):
    share, fan = TYPICAL_TENPAI
    curve = tenpai_curve(share, share * (fan + SAVED), qs)
    for level in range(steps):
        curve = chase(qs, TYPICAL_STEP[min(level, 2)], curve)
    return tuple(curve)


@lru_cache(maxsize=200000)
def waiting_curve(hand, opened, available, qs, chain=0, piao=0):
    """Expected fan of a 13-tile hand just before each of my remaining draws."""
    unseen = max(1, sum(available))
    row = analyze(hand, opened, available, chain, piao)
    if row["shanten"] <= 0:
        copies = row["winning_copies"]
        e = (row["fan_value"] + SAVED * copies) / unseen
        return tuple(tenpai_curve(copies / unseen, e, qs))
    u = row["ukeire"] / unseen
    if row["shanten"] > 1:
        return tuple(chase(qs, u, typical_curve(row["shanten"] - 1, qs)))
    # Each improving draw moves to its best waiting hand; others are thrown back.
    target = [0.0] * len(qs)
    for tile in row["improving"]:
        drawn = list(hand)
        drawn.append(tile)
        best = [0.0] * len(qs)
        for drop in set(drawn):
            rest = tuple(sorted(without(drawn, drop)))
            if analyze(rest, opened, available)["shanten"] > 0:
                continue
            curve = waiting_curve(rest, opened, available, qs)
            best = [max(a, b) for a, b in zip(best, curve)]
        weight = available[tile] / unseen
        target = [t + weight * b for t, b in zip(target, best)]
    curve = [0.0] * len(qs)
    for j in range(len(qs) - 2, -1, -1):
        live = 1 - qs[j + 1]
        curve[j] = live * (target[j + 1] + (1 - u) * curve[j + 1])
    return tuple(curve)


def waiting_value(hand, opened, available, qs, chain=0, piao=0):
    return waiting_curve(hand, opened, available, qs, chain, piao)[0]


def best_discard_value(hand, opened, available, qs):
    return max(
        waiting_value(tuple(sorted(without(hand, tile))), opened, available, qs)
        for tile in set(hand)
    )


def without(hand, tile):
    rest = list(hand)
    rest.remove(tile)
    return rest


class Race:
    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass

    def piao(self, obs, opened):
        return can_discard_piao(
            obs.hand,
            obs.drawn_tile,
            opened,
            obs.public_events,
            obs.stage.seat,
            obs.stage.round_no,
            catch_exempt=obs.catch_play and not obs.catch_restricted,
        )

    def value(self, obs, action, opened, available, qs):
        hand = list(obs.hand)
        if action.action == "hu":
            return obs.hu_fan + SAVED
        if action.action == "pass":
            return waiting_value(
                tuple(hand), opened, available, qs, obs.chain_count, obs.piao_count
            )
        if action.action == "discard":
            rest = tuple(sorted(without(hand, action.tile)))
            if action.tile == WHITE and self.piao(obs, opened):
                chain, piao = obs.chain_count + 1, obs.piao_count + 1
            else:
                chain, piao = 0, 0
            return (1 - qs[0]) * waiting_value(rest, opened, available, qs, chain, piao)
        if action.action in ("peng", "chi"):
            used = action.tiles if action.action == "chi" else (action.tile,) * 2
            for tile in used:
                hand.remove(tile)
            return (1 - qs[0]) * best_discard_value(hand, opened + 1, available, qs)
        if action.action == "gang":
            for _ in range(KONG_TILES[action.kind]):
                hand.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            rest = tuple(sorted(hand))
            row = analyze(rest, after, available, obs.chain_count + 1, obs.piao_count)
            if row["shanten"] > 0:
                return waiting_value(rest, after, available, qs)
            # The replacement draw comes at once and carries the kong's doubling.
            unseen = max(1, sum(available))
            copies = row["winning_copies"]
            now = (row["fan_value"] + SAVED * copies) / unseen
            miss = 1 - copies / unseen
            return now + miss * (1 - qs[0]) * waiting_value(rest, after, available, qs)
        return float("-inf")

    def certain_double(self, obs, opened):
        """A chain-extending action that keeps every next draw a win."""
        hand = list(obs.hand)
        for action in obs.legal_actions:
            if action.action != "gang":
                continue
            rest = list(hand)
            for _ in range(KONG_TILES[action.kind]):
                rest.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            if is_baotou(tuple(sorted(rest)), after):
                return action
        for action in obs.legal_actions:
            if action.action == "discard" and action.tile == WHITE:
                if self.piao(obs, opened):
                    return action
        return None

    def act(self, obs):
        available = remaining_tiles(obs)
        opened = len(obs.melds[obs.stage.seat])
        if any(a.action == "hu" for a in obs.legal_actions):
            double = self.certain_double(obs, opened)
            if double is not None:
                return double
        qs = segments(obs.wall_remaining)
        return max(
            obs.legal_actions, key=lambda a: self.value(obs, a, opened, available, qs)
        )


def build(model_dir):
    return Race()
