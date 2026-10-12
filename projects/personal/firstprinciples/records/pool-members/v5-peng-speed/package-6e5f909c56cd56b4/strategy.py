"""固定对手池成员（维护者，2026-09-27）：以规则线 fp-rule-033 的竞速估值为底的风格版本。

温和猛碰：能让向听减少、或向听不变而进张增加的碰一律碰，其余吃碰按估值；估值与翻倍（爆头升级、必胜时杠或财飘）与 fp-rule-033 相同。
"""

COMBINATION = "firstprinciples-rule"

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai import _fast
from mahjong_ai.domain.rules import WHITE, counts, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles, shanten


MISSING = object()


def memo(key, limit):
    """Memoize a pure function of tile tuples and small counts in a plain dict.

    Keys are bytes and values numbers, so the dict stays out of the garbage
    collector's walks; an lru_cache is walked entry by entry on every full
    pass, and seven members share an evaluation process. Emptied when full.
    """

    def wrap(fn):
        table = {}

        def cached(*args, **kwargs):
            try:
                k = key(*args, **kwargs)
            except ValueError:  # a count outside a byte: not worth a key
                return fn(*args, **kwargs)
            value = table.get(k, MISSING)
            if value is MISSING:
                if len(table) >= limit:
                    table.clear()
                value = table[k] = fn(*args, **kwargs)
            return value

        cached.table = table
        return cached

    return wrap


def hand_key(hand, opened, available, chain=0, piao=0, draws=16):
    return bytes(hand) + bytes(available) + bytes((opened, chain, piao, draws % 256))


def draws_key(hand, opened, available, draws=16):
    return bytes(hand) + bytes(available) + bytes((opened, draws % 256))


def held_key(hand, opened):
    return bytes(hand) + bytes((opened,))

Q = 0.15  # an opponent self-draws during the three draws between two of mine
DRAW_RISK = 1 - (1 - Q) ** (1 / 3)  # the same, for one opponent's draw
SAVED = 0.467  # my win cancels the payment to another winner, in fan units
ONE_STEP = 0.22  # typical value of a hand one step from waiting
FAR_STEP = 0.45  # discount for each further step
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}
LAST_WALL = 20  # the hand is drawn once the wall is down to this
Q_EARLY = 0.10
Q_LATE = 0.30
TIE = 1e-7  # weight of shape potential, which only separates equal values
UPGRADE = True  # a wait holding a white counts the draws that turn it into baotou


LONG = 16  # more draws of mine than any hand holds


def repeat(stay, draws):
    """Expected number of my draws while a state lasts, within the draws left."""
    return (1 - stay**draws) / (1 - stay)


def tenpai_hazard(draws):
    """Opponents close in as the wall runs down: the race gets harder late."""
    return min(Q_LATE, max(Q_EARLY, 0.35 - 0.018 * draws))


def race(p, e, draws=LONG):
    return e * repeat((1 - p) * (1 - tenpai_hazard(draws)), draws)


# The race valuation runs in the runtime's C port of the Python functions this
# strategy was written with (the same arithmetic in the same order); the SDK's
# analysis of a white-joker boundary hand comes from examined below.
PROFILE = _fast.race_profile(Q, SAVED, ONE_STEP, FAR_STEP, Q_EARLY, Q_LATE, 1.0, UPGRADE)


def waiting_value(hand, opened, available, chain=0, piao=0, draws=LONG):
    """Expected fan of a 13-tile hand just before my next draw."""
    return _fast.race_waiting(hand, opened, available, chain, piao, draws, PROFILE, examined)


# Tile weights whose sums spell out, bit by bit, which draws allow a baotou
# discard: one C call for each half of the tiles.
LOW_BITS = tuple(1 << t if t < 17 else 0 for t in range(34))
HIGH_BITS = tuple(1 << (t - 17) if t >= 17 else 0 for t in range(34))


@memo(held_key, 50_000)
def baotou_draws(hand, opened):
    """The tiles whose draw lets a discard leave baotou."""
    held = counts(hand)
    low = _fast.baotou_outs(held, opened, LOW_BITS)
    high = _fast.baotou_outs(held, opened, HIGH_BITS) << 17
    return low | high  # a bit per tile


@memo(hand_key, 50_000)
def baotou_draw_fans(hand, opened, available, chain, piao):
    """Copies and fan-weighted copies of the draws a baotou wait wins on, as copies << 32 | fan."""
    held = list(counts(hand))
    copies = fan = 0
    for tile in range(34):
        if available[tile] <= 0 or held[tile] >= 4:
            continue
        held[tile] += 1
        value = _fast.fan_value(held, tile, opened, chain, piao, False, False)
        held[tile] -= 1
        if value:
            copies += available[tile]
            fan += available[tile] * value
    return copies << 32 | fan


def examined(hand, opened, available, chain=0, piao=0):
    """The SDK analysis of a hand (its keys rarely repeat, so it is not cached)."""
    return analyze(hand, opened, available, chain, piao)


def reshaped_value(hand, opened, available, draws):
    """A waiting hand with the option to trade a non-winning draw for a better wait."""
    return _fast.race_reshaped(hand, opened, available, draws, PROFILE, examined)


def potential(hand, available):
    """Live tiles that pair with or sit within two of each kept tile."""
    total = 0
    for tile in hand:
        if tile >= 27:
            total += available[tile]
            continue
        low, high = tile - tile % 9, tile - tile % 9 + 8
        total += sum(available[t] for t in range(max(low, tile - 2), min(high, tile + 2) + 1))
    return total


def best_discard_value(hand, opened, available, draws=LONG):
    return max(
        waiting_value(tuple(sorted(rest)), opened, available, draws=draws)
        for rest in (without(hand, tile) for tile in set(hand))
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

    def value(self, obs, action, opened, available):
        hand = list(obs.hand)
        # My draws left before the wall is down to its last 20 tiles.
        draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
        if action.action == "hu":
            return obs.hu_fan + SAVED
        if action.action == "pass":
            # Passing lets everyone between the discarder and me draw first.
            discarder = next(
                e.seat for e in reversed(obs.public_events) if e.kind == "discard"
            )
            ahead = (obs.stage.seat - discarder - 1) % 4
            return (1 - DRAW_RISK) ** ahead * waiting_value(
                tuple(hand), opened, available, obs.chain_count, obs.piao_count, draws
            )
        if action.action == "discard":
            rest = tuple(sorted(without(hand, action.tile)))
            if action.tile == WHITE and self.piao(obs, opened):
                chain, piao = obs.chain_count + 1, obs.piao_count + 1
            else:
                chain, piao = 0, 0
            if chain:
                value = waiting_value(rest, opened, available, chain, piao, draws)
            else:
                value = reshaped_value(rest, opened, available, draws)
            return (1 - Q) * value + TIE * potential(rest, available)
        if action.action in ("peng", "chi"):
            used = action.tiles if action.action == "chi" else (action.tile,) * 2
            for tile in used:
                hand.remove(tile)
            value = best_discard_value(hand, opened + 1, available, draws)
            # Called from a baotou wait, a white discard that keeps baotou
            # scores piao without waiting for my own draw.
            if WHITE in hand and is_baotou(obs.hand, opened):
                rest = tuple(sorted(without(hand, WHITE)))
                if is_baotou(rest, opened + 1):
                    owner = obs.god_discarder_seat == obs.stage.seat
                    chain = (obs.chain_count if owner else 0) + 1
                    piao = (obs.piao_count if owner else 0) + 1
                    value = max(
                        value,
                        waiting_value(rest, opened + 1, available, chain, piao, draws),
                    )
            return (1 - Q) * value
        if action.action == "gang":
            for _ in range(KONG_TILES[action.kind]):
                hand.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            rest = tuple(sorted(hand))
            row = analyze(rest, after, available, obs.chain_count + 1, obs.piao_count)
            if row["shanten"] > 0:
                return waiting_value(rest, after, available, draws=draws)
            # The replacement draw comes at once and carries the kong's doubling.
            unseen = max(1, sum(available))
            copies = row["winning_copies"]
            now = (row["fan_value"] + SAVED * copies) / unseen
            miss = 1 - copies / unseen
            return now + miss * (1 - Q) * waiting_value(
                rest, after, available, draws=draws
            )
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
        return max(obs.legal_actions, key=lambda a: self.value(obs, a, opened, available))




def after_call_shanten(obs, action, opened):
    """Distance of the best hand a call leaves (after its discard)."""
    hand = list(obs.hand)
    used = action.tiles if action.action == "chi" else (action.tile,) * 2
    for tile in used:
        hand.remove(tile)
    return min(shanten(tuple(sorted(without(hand, t))), opened + 1) for t in set(hand))


def kong_tenpai(obs, action, opened):
    hand = list(obs.hand)
    for _ in range(KONG_TILES[action.kind]):
        hand.remove(action.tile)
    after = opened if action.kind == "added" else opened + 1
    return shanten(tuple(sorted(hand)), after) <= 0


def after_call_ukeire(obs, action, opened, available):
    """Live draws that improve the best hand a call leaves."""
    hand = list(obs.hand)
    used = action.tiles if action.action == "chi" else (action.tile,) * 2
    for tile in used:
        hand.remove(tile)
    rows = [analyze(tuple(sorted(without(hand, t))), opened + 1, available) for t in set(hand)]
    best = min(r["shanten"] for r in rows)
    return max(r["ukeire"] for r in rows if r["shanten"] == best)


def own_ukeire(obs, opened, available):
    return analyze(tuple(sorted(obs.hand)), opened, available)["ukeire"]


class Style(Race):
    def act(self, obs):
        available = remaining_tiles(obs)
        opened = len(obs.melds[obs.stage.seat])
        legal = obs.legal_actions
        if any(a.action == "hu" for a in legal):
            double = self.certain_double(obs, opened)
            if double is not None:
                return double
        now = shanten(tuple(sorted(obs.hand)), opened) if len(obs.hand) % 3 == 1 else None
        for a in legal:
            if a.action == "peng" and now is not None:
                after = after_call_shanten(obs, a, opened)
                if after < now or (after == now and after_call_ukeire(obs, a, opened, available) > own_ukeire(obs, opened, available)):
                    return a
        return max(legal, key=lambda a: self.value(obs, a, opened, available))


def build(model_dir):
    return Style()
