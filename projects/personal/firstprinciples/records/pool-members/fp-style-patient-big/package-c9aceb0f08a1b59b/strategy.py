"""固定对手池成员（维护者，2026-09-27）：以规则线 fp-rule-033 的竞速估值为底的风格版本。

慢打大牌：把对手的胡牌速度估得更慢（每圈 Q 0.10，听牌期 0.06→0.22），更愿意为爆头、翻倍和更好的听牌多等几轮；估值结构与翻倍与 fp-rule-033 相同。
"""

COMBINATION = "firstprinciples-rule"

from functools import lru_cache

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai import _fast
from mahjong_ai.domain.rules import WHITE, counts, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles, shanten

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
HUNT = 1.0
Q = 0.10
DRAW_RISK = 1 - (1 - Q) ** (1 / 3)
Q_EARLY = 0.06
Q_LATE = 0.22


def repeat(stay, draws):
    """Expected number of my draws while a state lasts, within the draws left."""
    return (1 - stay**draws) / (1 - stay)


def tenpai_hazard(draws):
    """Opponents close in as the wall runs down: the race gets harder late."""
    return min(Q_LATE, max(Q_EARLY, 0.35 - 0.018 * draws))


def race(p, e, draws=LONG):
    return e * repeat((1 - p) * (1 - tenpai_hazard(draws)), draws)


@lru_cache(maxsize=200000)
def waiting_value(hand, opened, available, chain=0, piao=0, draws=LONG):
    """Expected fan of a 13-tile hand just before my next draw."""
    unseen = max(1, sum(available))
    if is_baotou(hand, opened):
        copies, fan = baotou_draw_fans(hand, opened, available, chain, piao)
        return race(copies / unseen, (fan + SAVED * copies) / unseen, draws)
    row = examined(hand, opened, available, chain, piao)
    if row["shanten"] <= 0:
        copies = row["winning_copies"]
        gain = row["fan_value"] + SAVED * copies
        if WHITE not in hand or not UPGRADE:
            return race(copies / unseen, gain / unseen, draws)
        # A draw that frees a white lets a discard turn the wait into baotou,
        # which may beat winning on it at once.
        fans = row["fans"]
        gain, reach = 0.0, 0
        outs = baotou_draws(hand, opened)
        held = list(counts(hand))
        kinds = [t for t in range(34) if held[t]]
        for tile in range(34):
            if available[tile] <= 0 or held[tile] >= 4:
                continue
            best = 0.0
            if tile in outs:
                held[tile] += 1
                for drop in kinds:
                    if drop == tile:
                        continue
                    held[drop] -= 1
                    if _fast.baotou(held, opened):
                        rest = sorted((*hand, tile))
                        rest.remove(drop)
                        best = max(
                            best,
                            waiting_value(tuple(rest), opened, available, draws=draws - 1),
                        )
                    held[drop] += 1
                held[tile] -= 1
            if tile in fans:
                gain += available[tile] * max(fans[tile] + SAVED, (1 - Q) * best * HUNT)
            elif best:
                gain += available[tile] * (1 - Q) * best * HUNT
                reach += available[tile]
        stay = (1 - tenpai_hazard(draws)) * (1 - (copies + reach) / unseen)
        return gain / unseen * repeat(stay, draws)
    u = row["ukeire"] / unseen
    if row["shanten"] == 1:
        weighted = 0.0
        for tile in row["improving"]:
            drawn = list(hand)
            drawn.append(tile)
            best = 0.0
            for drop in set(drawn):
                rest = list(drawn)
                rest.remove(drop)
                rest = tuple(sorted(rest))
                best = max(best, tenpai_value(rest, opened, available, draws - 1))
            weighted += available[tile] * best
        # Improving draws move to a waiting hand; the others are thrown back.
        stay = (1 - u) * (1 - Q)
        return (1 - Q) * (weighted / unseen) * repeat(stay, draws - 1)
    step = (1 - Q) * u * repeat((1 - u) * (1 - Q), draws - row["shanten"])
    return step * ONE_STEP * FAR_STEP ** (row["shanten"] - 2)


# Tile weights whose sums spell out, bit by bit, which draws allow a baotou
# discard: one C call for each half of the tiles.
LOW_BITS = tuple(1 << t if t < 17 else 0 for t in range(34))
HIGH_BITS = tuple(1 << (t - 17) if t >= 17 else 0 for t in range(34))


@lru_cache(maxsize=200000)
def baotou_draws(hand, opened):
    """The tiles whose draw lets a discard leave baotou."""
    held = counts(hand)
    low = _fast.baotou_outs(held, opened, LOW_BITS)
    high = _fast.baotou_outs(held, opened, HIGH_BITS) << 17
    return frozenset(t for t in range(34) if (low | high) >> t & 1)


@lru_cache(maxsize=200000)
def baotou_draw_fans(hand, opened, available, chain, piao):
    """Copies and fan-weighted copies of the draws a baotou wait wins on."""
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
    return copies, fan


@lru_cache(maxsize=100000)
def examined(hand, opened, available, chain=0, piao=0):
    """The SDK analysis of a hand, computed once (callers only read it)."""
    return analyze(hand, opened, available, chain, piao)


@lru_cache(maxsize=200000)
def tenpai_value(hand, opened, available, draws=LONG):
    row = examined(hand, opened, available)
    if row["shanten"] > 0 or draws < 1:
        return 0.0
    return waiting_value(hand, opened, available, draws=draws)


@lru_cache(maxsize=20000)
def reshaped_value(hand, opened, available, draws):
    """A waiting hand with the option to trade a non-winning draw for a better wait."""
    value = waiting_value(hand, opened, available, draws=draws)
    if draws < 2 or shanten(hand, opened) > 0:
        return value
    unseen = max(1, sum(available))
    row = analyze(hand, opened, available)
    wins = set(row["winning_tiles"])
    stay = waiting_value(hand, opened, available, draws=draws - 1)
    gain, reach = 0.0, 0
    held = list(counts(hand))
    kinds = [t for t in range(34) if held[t]]
    for tile in range(34):
        if tile in wins or available[tile] <= 0 or held[tile] >= 4:
            continue
        held[tile] += 1
        best = stay
        for drop in kinds:
            if drop == tile:
                continue
            held[drop] -= 1
            if _fast.shanten_raw(held, opened) <= 0:
                rest = sorted((*hand, tile))
                rest.remove(drop)
                best = max(
                    best, waiting_value(tuple(rest), opened, available, draws=draws - 1)
                )
            held[drop] += 1
        held[tile] -= 1
        if best > stay:
            gain += available[tile] * (best - stay)
            reach += available[tile]
    survive = (1 - Q) * (1 - (row["winning_copies"] + reach) / unseen)
    return value + (1 - Q) * gain / unseen * repeat(survive, draws - 1)


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


class Style(Race):
    pass


def build(model_dir):
    return Style()
