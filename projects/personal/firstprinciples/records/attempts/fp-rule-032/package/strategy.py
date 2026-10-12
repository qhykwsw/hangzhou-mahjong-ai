"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Between two of my draws the three
opponents win with chance Q. A waiting hand that wins on a fraction p of draws,
collecting e fan per draw on average, is worth e / (1 - (1 - p)(1 - Q)) just
before my draw, where Q for a waiting hand rises as the wall runs down (the
others close in; measured waits are worth more early and less late than a
fixed Q says); a wait also counts the draws that let a discard turn it into
baotou, when that beats winning on them at once (for a wait without a white,
drawing one), and, holding a white, the opponents' discards of those tiles
that a pung or chow turns into baotou. One step from waiting, each improving draw leads to the best
waiting hand it allows; further away a generic discount per step stands in.
A win also cancels the payment I would owe to someone else's win, SAVED fan.
All of this counts only the draws I have left before the wall runs out.
Passing on a discard lets the opponents between its discarder and me draw
before my turn, so a call or kong also takes their draws away. A wait my
discard leaves also counts its non-winning draws that a later discard turns
into a better wait. Discards of
equal value keep the hand whose tiles connect to more live tiles. When the hand is already baotou (every draw wins), a win now is traded for a
doubling that keeps the certain win: a kong whose replacement draw still wins,
else a white discard that scores piao. A call from a baotou wait is priced with
the piao its white discard scores when the wait stays baotou.
"""
COMBINATION = "firstprinciples-rule"

from functools import lru_cache

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.rules import WHITE, is_baotou
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
# An opponent discards a given unseen copy about as often per round as I draw
# it; any of three can feed a pung, only the player before me a chow (measured
# on waits holding a white: 3.4 and 0.75 times my own draw).
PENG_FEED = 3.4
CHI_FEED = 0.75


LONG = 16  # more draws of mine than any hand holds


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
    row = analyze(hand, opened, available, chain, piao)
    if row["shanten"] <= 0:
        copies = row["winning_copies"]
        gain = row["fan_value"] + SAVED * copies
        if is_baotou(hand, opened):
            return race(copies / unseen, gain / unseen, draws)
        if WHITE not in hand:
            # Drawing a white wins, or a discard may turn it into baotou.
            fans = row["fans"]
            best = 0.0
            if available[WHITE] > 0 and WHITE in fans:
                for rest in baotou_after(hand, opened, WHITE):
                    best = max(best, waiting_value(rest, opened, available, draws=draws - 1))
            if (1 - Q) * best > fans.get(WHITE, 0) + SAVED:
                gain += available[WHITE] * ((1 - Q) * best - fans[WHITE] - SAVED)
            return race(copies / unseen, gain / unseen, draws)
        # A draw that frees a white lets a discard turn the wait into baotou,
        # which may beat winning on it at once.
        fans = row["fans"]
        gain, reach = 0.0, 0
        fed, feed = 0.0, 0.0
        for tile in range(34):
            if available[tile] <= 0 or hand.count(tile) >= 4:
                continue
            drawn = list(hand)
            drawn.append(tile)
            best = 0.0
            for drop in set(drawn):
                rest = tuple(sorted(without(drawn, drop)))
                if is_baotou(rest, opened):
                    best = max(
                        best, waiting_value(rest, opened, available, draws=draws - 1)
                    )
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


@lru_cache(maxsize=50000)
def baotou_after(hand, opened, tile):
    """The baotou hands a discard leaves after drawing this tile."""
    drawn = list(hand)
    drawn.append(tile)
    rests = {tuple(sorted(without(drawn, drop))) for drop in set(drawn)}
    return tuple(rest for rest in rests if is_baotou(rest, opened))


@lru_cache(maxsize=50000)
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
    rate, called = 0.0, set()
    for feed, used in uses:
        rest = list(hand)
        for t in used:
            rest.remove(t)
        for drop in set(rest):
            left = tuple(sorted(without(rest, drop)))
            if is_baotou(left, opened + 1):
                called.add(left)
                rate = max(rate, feed)
    return rate, tuple(called)


@lru_cache(maxsize=200000)
def tenpai_value(hand, opened, available, draws=LONG):
    row = analyze(hand, opened, available)
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
    for tile in range(34):
        if tile in wins or available[tile] <= 0 or hand.count(tile) >= 4:
            continue
        drawn = sorted((*hand, tile))
        best = stay
        for drop in set(drawn):
            rest = list(drawn)
            rest.remove(drop)
            rest = tuple(rest)
            if rest != hand and shanten(rest, opened) <= 0:
                best = max(best, waiting_value(rest, opened, available, draws=draws - 1))
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


def build(model_dir):
    return Race()
