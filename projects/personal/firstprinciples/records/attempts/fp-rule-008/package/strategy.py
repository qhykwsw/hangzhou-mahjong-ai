"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Between two of my draws the three
opponents win with chance Q. A waiting hand that wins on a fraction p of draws,
collecting e fan per draw on average, is worth e / (1 - (1 - p)(1 - Q)) just
before my draw; a wait holding a white also counts the draws that let a discard
turn it into baotou, when that beats winning on them at once. One step from waiting, each improving draw leads to the best
waiting hand it allows; further away a generic discount per step stands in.
A win also cancels the payment I would owe to someone else's win, SAVED fan.
Passing on a discard lets the opponents between its discarder and me draw
before my turn, so a call or kong also takes their draws away. Discards of
equal value keep the hand whose tiles connect to more live tiles. When the hand is already baotou (every draw wins), a win now is traded for a
doubling that keeps the certain win: a kong whose replacement draw still wins,
else a white discard that scores piao.
"""
COMBINATION = "firstprinciples-rule"

from bisect import insort
from functools import lru_cache

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.rules import WHITE, is_baotou
from mahjong_ai.sdk.lookahead import baotou_outs
from mahjong_ai.sdk.shape import analyze, remaining_tiles, shanten

Q = 0.15  # an opponent self-draws during the three draws between two of mine
DRAW_RISK = 1 - (1 - Q) ** (1 / 3)  # the same, for one opponent's draw
SAVED = 0.4  # my win cancels the payment to another winner, in fan units
ONE_STEP = 0.22  # typical value of a hand one step from waiting
FAR_STEP = 0.45  # discount for each further step
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}
TIE = 1e-7  # weight of shape potential, which only separates equal values


def race(p, e):
    return e / (1 - (1 - p) * (1 - Q))


@lru_cache(maxsize=200000)
def row_of(hand, opened, available, chain=0, piao=0):
    return analyze(hand, opened, available, chain, piao)


def discards(drawn):
    """Each distinct sorted hand one discard leaves from a sorted hand."""
    seen = None
    for i, tile in enumerate(drawn):
        if tile != seen:
            seen = tile
            yield tuple(drawn[:i] + drawn[i + 1 :])


def with_draw(hand, tile):
    drawn = list(hand)
    insort(drawn, tile)
    return drawn


EVERY = (1,) * 34
ONLY = tuple(tuple(int(t == tile) for t in range(34)) for tile in range(34))


@lru_cache(maxsize=100000)
def baotou_turns(hand, opened):
    """For each drawn tile, the waiting hands one discard turns into baotou."""
    # The native count finds the draws that have such a discard at all.
    if not baotou_outs(hand, opened, EVERY):
        return ((),) * 34
    return tuple(
        tuple(r for r in discards(with_draw(hand, tile)) if is_baotou(r, opened))
        if hand.count(tile) < 4 and baotou_outs(hand, opened, ONLY[tile])
        else ()
        for tile in range(34)
    )


@lru_cache(maxsize=200000)
def waiting_value(hand, opened, available, chain=0, piao=0):
    """Expected fan of a 13-tile hand just before my next draw."""
    unseen = max(1, sum(available))
    row = row_of(hand, opened, available, chain, piao)
    if row["shanten"] <= 0:
        copies = row["winning_copies"]
        gain = row["fan_value"] + SAVED * copies
        if WHITE not in hand or is_baotou(hand, opened):
            return race(copies / unseen, gain / unseen)
        # A draw that frees a white lets a discard turn the wait into baotou,
        # which may beat winning on it at once.
        fans = row["fans"]
        turns = baotou_turns(hand, opened)
        gain, reach = 0.0, 0
        for tile in range(34):
            if available[tile] <= 0 or hand.count(tile) >= 4:
                continue
            best = 0.0
            for rest in turns[tile]:
                best = max(best, waiting_value(rest, opened, available))
            if tile in fans:
                gain += available[tile] * max(fans[tile] + SAVED, (1 - Q) * best)
            elif best:
                gain += available[tile] * (1 - Q) * best
                reach += available[tile]
        stay = (1 - Q) * (1 - (copies + reach) / unseen)
        return gain / unseen / (1 - stay)
    u = row["ukeire"] / unseen
    if row["shanten"] == 1:
        weighted = 0.0
        for tile in row["improving"]:
            best = 0.0
            for rest in discards(with_draw(hand, tile)):
                best = max(best, tenpai_value(rest, opened, available))
            weighted += available[tile] * best
        # Improving draws move to a waiting hand; the others are thrown back.
        return (1 - Q) * (weighted / unseen) / (1 - (1 - u) * (1 - Q))
    step = (1 - Q) * u / (1 - (1 - u) * (1 - Q))
    return step * ONE_STEP * FAR_STEP ** (row["shanten"] - 2)


@lru_cache(maxsize=200000)
def tenpai_value(hand, opened, available):
    if shanten(hand, opened) > 0:
        return 0.0
    return waiting_value(hand, opened, available)


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


def best_discard_value(hand, opened, available):
    return max(
        waiting_value(tuple(sorted(rest)), opened, available)
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
        if action.action == "hu":
            return obs.hu_fan + SAVED
        if action.action == "pass":
            # Passing lets everyone between the discarder and me draw first.
            discarder = next(
                e.seat for e in reversed(obs.public_events) if e.kind == "discard"
            )
            ahead = (obs.stage.seat - discarder - 1) % 4
            return (1 - DRAW_RISK) ** ahead * waiting_value(
                tuple(hand), opened, available, obs.chain_count, obs.piao_count
            )
        if action.action == "discard":
            rest = tuple(sorted(without(hand, action.tile)))
            if action.tile == WHITE and self.piao(obs, opened):
                chain, piao = obs.chain_count + 1, obs.piao_count + 1
            else:
                chain, piao = 0, 0
            return (1 - Q) * waiting_value(
                rest, opened, available, chain, piao
            ) + TIE * potential(rest, available)
        if action.action in ("peng", "chi"):
            used = action.tiles if action.action == "chi" else (action.tile,) * 2
            for tile in used:
                hand.remove(tile)
            return (1 - Q) * best_discard_value(hand, opened + 1, available)
        if action.action == "gang":
            for _ in range(KONG_TILES[action.kind]):
                hand.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            rest = tuple(sorted(hand))
            row = analyze(rest, after, available, obs.chain_count + 1, obs.piao_count)
            if row["shanten"] > 0:
                return waiting_value(rest, after, available)
            # The replacement draw comes at once and carries the kong's doubling.
            unseen = max(1, sum(available))
            copies = row["winning_copies"]
            now = (row["fan_value"] + SAVED * copies) / unseen
            miss = 1 - copies / unseen
            return now + miss * (1 - Q) * waiting_value(rest, after, available)
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
