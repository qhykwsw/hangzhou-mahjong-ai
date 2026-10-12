"""Rule line: plain tile efficiency, doubling when the win is certain.

Every action is priced by the 13-tile hand it leaves waiting for the next draw:
-12 per step of shanten plus 0.12 per live improving tile. When the hand is
already baotou (every draw wins), a win now is traded for a doubling that keeps
the certain win: a kong whose replacement draw still wins, else a white
discard that scores piao.
"""
COMBINATION = "firstprinciples-rule"

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.rules import WHITE, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles

SPEED = 12.0
UKEIRE = 0.12
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}


def waiting_value(hand, opened, available):
    row = analyze(tuple(sorted(hand)), opened, available)
    return -SPEED * row["shanten"] + UKEIRE * row["ukeire"]


def best_discard_value(hand, opened, available):
    best = None
    for tile in set(hand):
        rest = list(hand)
        rest.remove(tile)
        value = waiting_value(rest, opened, available)
        if best is None or value > best:
            best = value
    return best


class Efficiency:
    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass

    def value(self, obs, action, opened, available):
        hand = list(obs.hand)
        if action.action == "hu":
            return float("inf")
        if action.action == "pass":
            return waiting_value(hand, opened, available)
        if action.action == "discard":
            hand.remove(action.tile)
            return waiting_value(hand, opened, available)
        if action.action == "peng":
            hand.remove(action.tile)
            hand.remove(action.tile)
            return best_discard_value(hand, opened + 1, available)
        if action.action == "chi":
            for tile in action.tiles:
                hand.remove(tile)
            return best_discard_value(hand, opened + 1, available)
        if action.action == "gang":
            for _ in range(KONG_TILES[action.kind]):
                hand.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            return waiting_value(hand, after, available)
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
                if can_discard_piao(
                    obs.hand,
                    obs.drawn_tile,
                    opened,
                    obs.public_events,
                    obs.stage.seat,
                    obs.stage.round_no,
                    catch_exempt=obs.catch_play and not obs.catch_restricted,
                ):
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
    return Efficiency()
