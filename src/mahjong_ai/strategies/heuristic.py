"""Exact shape policy with explicit, configurable risk preferences."""

import numpy as np
import math
from mahjong_ai.strategies.common import RulePolicy
from mahjong_ai.sdk.shape import analyze, remaining_tiles
from mahjong_ai.domain.rules import WHITE

STYLES = {"balanced": (1.0, 1.0), "adventurous": (2.2, 0.4), "conservative": (0.4, 2.2)}


class HeuristicPolicy(RulePolicy):
    def __init__(self, style="balanced", speed=12.0, ukeire=0.12, fan=0.12):
        if style not in STYLES or any(
            not math.isfinite(v) or v < 0 for v in (speed, ukeire, fan)
        ):
            raise ValueError("invalid heuristic parameters")
        super().__init__()
        self.style = style
        self.speed = speed
        self.ukeire = ukeire
        self.fan_weight = fan
        self.reward_preference, self.risk_aversion = STYLES[style]

    def utilities(self, obs):
        available = remaining_tiles(obs)
        values = []
        own = obs.stage.seat
        open_count = len(obs.melds[own])
        lead = obs.stage.own_score - max(
            v for i, v in enumerate(obs.stage.effective_scores) if i != own
        )
        protect = self.risk_aversion * (1 + obs.stage.progress if lead > 0 else 1)
        for action in obs.legal_actions:
            if action.action == "hu":
                values.append(100000.0)
                continue
            hand = list(obs.hand)
            opened = open_count
            if action.action == "discard":
                hand.remove(action.tile)
            elif action.action == "peng":
                hand.remove(action.tile)
                hand.remove(action.tile)
                opened += 1
            elif action.action == "chi":
                for t in action.tiles:
                    hand.remove(t)
                opened += 1
            elif action.action == "gang":
                # Kong draw distribution is represented by current shape prior.
                values.append(
                    float(super().utilities(obs)[obs.legal_actions.index(action)])
                )
                continue
            if len(hand) == 14 - 3 * opened:
                alternatives = []
                for tile in set(hand):
                    h = list(hand)
                    h.remove(tile)
                    alternatives.append(self._value(h, opened, available, obs, protect))
                value = max(alternatives)
            elif len(hand) == 13 - 3 * opened:
                value = self._value(hand, opened, available, obs, protect)
            else:
                value = -1000.0
            if action.action == "discard" and action.tile == WHITE:
                value += self.reward_preference * (obs.chain_count + 1) - protect * 1.5
            values.append(value)
        return np.asarray(values)

    def _value(self, hand, opened, available, obs, protect):
        row = analyze(
            tuple(hand),
            opened,
            available,
            obs.chain_count,
            obs.piao_count,
            obs.you_cai_bi_kao,
        )
        return (
            -self.speed * row["shanten"]
            + self.ukeire * row["ukeire"]
            + self.fan_weight * self.reward_preference * row["fan_value"]
            - protect * max(0, row["shanten"]) * (0.5 + obs.stage.progress)
        )
