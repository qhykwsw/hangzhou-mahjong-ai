"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Between two of my draws the three
opponents win with chance Q. A waiting hand that wins on a fraction p of draws,
collecting e fan per draw on average, is worth e / (1 - (1 - p)(1 - Q)) just
before my draw. One step from waiting, each improving draw leads to the best
waiting hand it allows; further away a generic discount per step stands in.
A win also cancels the payment I would owe to someone else's win, SAVED fan.
When the hand is already baotou (every draw wins), a win now is traded for a
doubling that keeps the certain win: a kong whose replacement draw still wins,
else a white discard that scores piao.
"""
COMBINATION = "firstprinciples-rule"

from functools import lru_cache

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.rules import WHITE, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles

Q = 0.15  # an opponent self-draws during the three draws between two of mine
SAVED = 0.4  # my win cancels the payment to another winner, in fan units
ONE_STEP = 0.22  # typical value of a hand one step from waiting
FAR_STEP = 0.6  # discount for each further step
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}


def race(p, e):
    return e / (1 - (1 - p) * (1 - Q))


@lru_cache(maxsize=200000)
def waiting_value(hand, opened, available, chain=0, piao=0):
    """Expected fan of a 13-tile hand just before my next draw."""
    unseen = max(1, sum(available))
    row = analyze(hand, opened, available, chain, piao)
    if row["shanten"] <= 0:
        copies = row["winning_copies"]
        return race(copies / unseen, (row["fan_value"] + SAVED * copies) / unseen)
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
                best = max(best, tenpai_value(tuple(sorted(rest)), opened, available))
            weighted += available[tile] * best
        # Improving draws move to a waiting hand; the others are thrown back.
        return (1 - Q) * (weighted / unseen) / (1 - (1 - u) * (1 - Q))
    step = (1 - Q) * u / (1 - (1 - u) * (1 - Q))
    return step * ONE_STEP * FAR_STEP ** (row["shanten"] - 2)


@lru_cache(maxsize=200000)
def tenpai_value(hand, opened, available):
    row = analyze(hand, opened, available)
    if row["shanten"] > 0:
        return 0.0
    return waiting_value(hand, opened, available)


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
            return waiting_value(
                tuple(hand), opened, available, obs.chain_count, obs.piao_count
            )
        if action.action == "discard":
            rest = tuple(sorted(without(hand, action.tile)))
            if action.tile == WHITE and self.piao(obs, opened):
                chain, piao = obs.chain_count + 1, obs.piao_count + 1
            else:
                chain, piao = 0, 0
            return (1 - Q) * waiting_value(rest, opened, available, chain, piao)
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
