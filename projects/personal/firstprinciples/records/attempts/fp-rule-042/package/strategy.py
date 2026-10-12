"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Between two of my draws the three
opponents win with chance Q. A waiting hand that wins on a fraction p of draws,
collecting e fan per draw on average, is worth e / (1 - (1 - p)(1 - Q)) just
before my draw, where Q for a waiting hand rises as the wall runs down (the
others close in; measured waits are worth more early and less late than a
fixed Q says); a wait holding a white also counts the draws that let a discard
turn it into baotou, when that beats winning on them at once, and the
opponents' discards of those tiles that a pung or chow turns into baotou. One step from waiting, each improving draw leads to the best
waiting hand it allows; further away a generic discount per step stands in.
A win also cancels the payment I would owe to someone else's win, SAVED fan.
All of this counts only the draws I have left before the wall runs out.
Passing on a discard lets the opponents between its discarder and me draw
before my turn, so a call or kong also takes their draws away. A wait my
discard leaves also counts its non-winning draws that a later discard turns
into a better wait; a discard leaving a hand one step away counts the same
option for the waits it leads to, pricing each wait there as a plain race
(only when discarding, to keep call and pass replies fast). Discards of
equal value keep the hand whose tiles connect to more live tiles. When the hand is already baotou (every draw wins), a win now is traded for a
doubling that keeps the certain win: a kong whose replacement draw still wins,
else a white discard that scores piao. A call from a baotou wait is priced with
the piao its white discard scores when the wait stays baotou.
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
OPTION_REACH = 1.35  # the largest measured ratio was 1.30
TIE = 1e-7  # weight of shape potential, which only separates equal values
# An opponent discards a given unseen copy about as often per round as I draw
# it; any of three can feed a pung, only the player before me a chow (measured
# on waits holding a white: 3.3 and 1.0 times my own draw).
PENG_FEED = 3.3
CHI_FEED = 1.0


LONG = 16  # more draws of mine than any hand holds


def repeat(stay, draws):
    """Expected number of my draws while a state lasts, within the draws left."""
    if draws <= 0:
        return 0.0
    return (1 - stay**draws) / (1 - stay)


def tenpai_hazard(draws):
    """Opponents close in as the wall runs down: the race gets harder late."""
    return min(Q_LATE, max(Q_EARLY, 0.35 - 0.018 * draws))


def race(p, e, draws=LONG):
    return e * repeat((1 - p) * (1 - tenpai_hazard(draws)), draws)


@lru_cache(maxsize=2000)
def waiting_value(hand, opened, available, chain=0, piao=0, draws=LONG, options=False):
    """Expected fan of a 13-tile hand just before my next draw."""
    unseen = max(1, sum(available))
    if is_baotou(hand, opened):
        copies, fan = baotou_draw_fans(hand, opened, available, chain, piao)
        return race(copies / unseen, (fan + SAVED * copies) / unseen, draws)
    row = examined(hand, opened, available, chain, piao)
    if row[SHANTEN] <= 0:
        copies = row[COPIES]
        gain = row[FAN_VALUE] + SAVED * copies
        if WHITE not in hand:
            return race(copies / unseen, gain / unseen, draws)
        # A draw that frees a white lets a discard turn the wait into baotou,
        # which may beat winning on it at once.
        fans = dict(row[FANS])
        gain, reach = 0.0, 0
        fed, feed = 0.0, 0.0
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
                gain += available[tile] * max(fans[tile] + SAVED, (1 - Q) * best)
            elif best:
                gain += available[tile] * (1 - Q) * best
                reach += available[tile]
            if best:
                # The same tile discarded by an opponent may be called into
                # baotou; I cannot win on it, but every draw after that wins.
                rate, called = fed_baotou(hand, opened, tile)
                if called:
                    value = max(
                        waiting_value(rest, opened + 1, available, draws=draws - 1)
                        for rest in called
                    )
                    fed += rate * available[tile] * value
                    feed += rate * available[tile]
        # Each round my draw comes first; if the wait goes on, the opponents
        # may feed a call into baotou before they can win.
        going = 1 - (copies + reach) / unseen
        gain = gain + going * (1 - Q) * fed
        stay = (1 - tenpai_hazard(draws)) * going * (1 - feed / unseen)
        return gain / unseen * repeat(stay, draws)
    u = row[UKEIRE] / unseen
    if row[SHANTEN] == 1:
        weighted = 0.0
        held = list(counts(hand))
        kinds = [t for t in range(34) if held[t]]
        for tile in row[IMPROVING]:
            held[tile] += 1
            best, target = 0.0, None
            for drop in kinds:
                held[drop] -= 1
                # Only a discard that leaves a wait has a value; discarding
                # the drawn tile goes back to this hand, one step away.
                if _fast.shanten_raw(held, opened) <= 0:
                    rest = sorted((*hand, tile))
                    rest.remove(drop)
                    value = tenpai_value(tuple(rest), opened, available, draws - 1)
                    if value > best:
                        best, target = value, tuple(rest)
                held[drop] += 1
            held[tile] -= 1
            if options and target is not None and chain == 0:
                best += reshape_gain(target, opened, available, draws - 1)
            weighted += available[tile] * best
        # Improving draws move to a waiting hand; the others are thrown back.
        stay = (1 - u) * (1 - Q)
        return (1 - Q) * (weighted / unseen) * repeat(stay, draws - 1)
    step = (1 - Q) * u * repeat((1 - u) * (1 - Q), draws - row[SHANTEN])
    return step * ONE_STEP * FAR_STEP ** (row[SHANTEN] - 2)


# Tile weights whose sums spell out, bit by bit, which draws allow a baotou
# discard: one C call for each half of the tiles.
LOW_BITS = tuple(1 << t if t < 17 else 0 for t in range(34))
HIGH_BITS = tuple(1 << (t - 17) if t >= 17 else 0 for t in range(34))


@lru_cache(maxsize=2000)
def baotou_draws(hand, opened):
    """The tiles whose draw lets a discard leave baotou."""
    held = counts(hand)
    low = _fast.baotou_outs(held, opened, LOW_BITS)
    high = _fast.baotou_outs(held, opened, HIGH_BITS) << 17
    return tuple(t for t in range(34) if (low | high) >> t & 1)


@lru_cache(maxsize=2000)
def baotou_fans(hand, opened, chain, piao):
    """The fan a baotou wait scores on each tile it can draw (0 if it cannot)."""
    held = list(counts(hand))
    fans = []
    for tile in range(34):
        if held[tile] >= 4:
            fans.append(0)
            continue
        held[tile] += 1
        fans.append(_fast.fan_value(held, tile, opened, chain, piao, False, False))
        held[tile] -= 1
    return tuple(fans)


@lru_cache(maxsize=2000)
def baotou_draw_fans(hand, opened, available, chain, piao):
    """Copies and fan-weighted copies of the draws a baotou wait wins on."""
    fans = baotou_fans(hand, opened, chain, piao)
    copies = fan = 0
    for tile in range(34):
        if available[tile] > 0 and fans[tile]:
            copies += available[tile]
            fan += available[tile] * fans[tile]
    return copies, fan


# The SDK analysis kept as a tuple of numbers and tuples, which the garbage
# collector stops tracking; cached dicts and lists are walked by every full
# collection.
SHANTEN, IMPROVING, UKEIRE, COPIES, FANS, FAN_VALUE = range(6)


@lru_cache(maxsize=10000)
def examined(hand, opened, available, chain=0, piao=0):
    """The SDK analysis of a hand, computed once, indexed as above."""
    row = analyze(hand, opened, available, chain, piao)
    return (
        row["shanten"],
        tuple(row["improving"]),
        row["ukeire"],
        row["winning_copies"],
        tuple(row["fans"].items()),
        row["fan_value"],
    )


@lru_cache(maxsize=2000)
def fed_baotou(hand, opened, tile):
    """How often this tile is fed for a call after which a discard leaves
    baotou, and the baotou hands such calls can leave."""
    uses = []
    if hand.count(tile) >= 2:
        uses.append((PENG_FEED, (tile, tile)))
    if tile < 27:
        suit = tile // 9
        for a, b in ((tile - 2, tile - 1), (tile - 1, tile + 1), (tile + 1, tile + 2)):
            if 0 <= a and b < 27 and a // 9 == suit == b // 9 and a in hand and b in hand:
                uses.append((CHI_FEED, (a, b)))
    rate, called = 0.0, []
    for feed, used in uses:
        rest = list(hand)
        for t in used:
            rest.remove(t)
        for drop in set(rest):
            left = tuple(sorted(without(rest, drop)))
            if left not in called and is_baotou(left, opened + 1):
                called.append(left)
                rate = max(rate, feed)
    return rate, tuple(called)


@lru_cache(maxsize=2000)
def tenpai_value(hand, opened, available, draws=LONG):
    row = examined(hand, opened, available)
    if row[SHANTEN] > 0 or draws < 1:
        return 0.0
    return waiting_value(hand, opened, available, draws=draws)


@lru_cache(maxsize=2000)
def reshaped_value(hand, opened, available, draws, options=True):
    """A waiting hand with the option to trade a non-winning draw for a better wait."""
    distance = shanten(hand, opened)
    if distance > 0:
        return waiting_value(
            hand, opened, available, draws=draws, options=options and distance == 1
        )
    value = waiting_value(hand, opened, available, draws=draws)
    if draws < 2:
        return value
    unseen = max(1, sum(available))
    row = examined(hand, opened, available)
    wins = {tile for tile, _ in row[FANS]}
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
    survive = (1 - Q) * (1 - (row[COPIES] + reach) / unseen)
    return value + (1 - Q) * gain / unseen * repeat(survive, draws - 1)


@lru_cache(maxsize=2000)
def plain_value(hand, opened, available, draws):
    """A waiting hand's race, without the white upgrade option."""
    row = examined(hand, opened, available)
    unseen = max(1, sum(available))
    return race(row[COPIES] / unseen, (row[FAN_VALUE] + SAVED * row[COPIES]) / unseen, draws)


@lru_cache(maxsize=2000)
def reshape_gain(hand, opened, available, draws):
    """The trade option of reshaped_value for a wait one step ahead, pricing
    every wait there as a plain race."""
    if draws < 2:
        return 0.0
    unseen = max(1, sum(available))
    row = examined(hand, opened, available)
    wins = {tile for tile, _ in row[FANS]}
    stay = plain_value(hand, opened, available, draws - 1)
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
                best = max(best, plain_value(tuple(rest), opened, available, draws - 1))
            held[drop] += 1
        held[tile] -= 1
        if best > stay:
            gain += available[tile] * (best - stay)
            reach += available[tile]
    survive = (1 - Q) * (1 - (row[COPIES] + reach) / unseen)
    return (1 - Q) * gain / unseen * repeat(survive, draws - 1)


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

    def value(self, obs, action, opened, available, options=True):
        hand = list(obs.hand)
        # My draws left before the wall is down to its last 20 tiles.
        draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
        if action.action == "hu":
            return obs.hu_fan + SAVED
        if action.action == "pass":
            # Passing lets everyone between the discarder and me draw first.
            discarder = next(
                (e.seat for e in reversed(obs.public_events) if e.kind == "discard"),
                None,
            )
            ahead = 0 if discarder is None else (obs.stage.seat - discarder - 1) % 4
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
                value = reshaped_value(rest, opened, available, draws, options)
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
        # The trade option one step away adds at most about a quarter to a
        # hand's value, so only discards that close to the best need it.
        values = [self.value(obs, a, opened, available, False) for a in obs.legal_actions]
        top = max(values)
        for i, a in enumerate(obs.legal_actions):
            if a.action == "discard" and values[i] * OPTION_REACH >= top:
                values[i] = self.value(obs, a, opened, available)
        return obs.legal_actions[max(range(len(values)), key=values.__getitem__)]


def build(model_dir):
    return Race()
