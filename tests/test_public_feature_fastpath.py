"""Public feature acceleration agrees with uncached Python calculations."""
from collections import Counter
from dataclasses import replace

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import Meld, RoundConfig
from mahjong_ai.sdk.shape import remaining_tiles
from mahjong_ai.strategies.heuristic import HeuristicPolicy


def reference_remaining(obs):
    visible = Counter(obs.hand)
    for row in obs.discards:
        visible.update(row)
    for row in obs.melds:
        for meld in row:
            visible.update(meld.tiles)
    return tuple(max(0, 4 - visible[t]) for t in range(34))


def test_remaining_tiles_counts_public_melds_and_clamps_overexposure():
    obs = HandState(RoundConfig(), 991).observation()
    cases = [obs, replace(obs, hand=(1,) * 5 + (33, 33),
                         discards=((1, 2, 33), (), (3,), ()),
                         melds=((Meld('chi', (4, 5, 6)),),
                                (Meld('gang', (7,) * 4, concealed=True),),
                                (Meld('gang', (), concealed=True),), ()))]
    for case in cases:
        assert remaining_tiles(case) == reference_remaining(case)


def test_preferences_cache_tracks_scores_progress_and_policy_settings():
    obs = HandState(RoundConfig(), 991).observation()
    for style in ('adaptive', 'efficiency', 'fan'):
        p = HeuristicPolicy(style)
        cases = [obs, replace(obs, stage=replace(obs.stage, round_no=7))]
        for case in cases:
            assert p._preferences(case) == p._calculate_preferences(case)
            assert p._preferences(case) == p._calculate_preferences(case)
        p.risk_aversion += 3
        p.reward_preference += 2
        assert p._preferences(cases[-1]) == p._calculate_preferences(cases[-1])
