import random

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import RoundConfig
from selfplay_core.actions import N_ACTIONS, action_index, describe_index, legal_mask


def test_index_is_a_bijection_over_the_full_space():
    seen = set()
    for i in range(N_ACTIONS):
        action = describe_index(i)
        assert action_index(action) == i
        assert action not in seen
        seen.add(action)


def test_every_engine_legal_action_round_trips():
    rng = random.Random(3)
    kinds = set()
    for seed in range(40):
        hand = HandState(RoundConfig(you_cai_bi_kao=seed % 2 == 0), seed)
        while not hand.settled:
            legal = hand.legal_actions()
            for a in legal:
                i = action_index(a)
                assert describe_index(i) == a
                kinds.add((a.action, a.kind))
            assert legal_mask(legal).sum() == len(set(legal))
            special = [a for a in legal if a.action in ("chi", "peng", "gang")]
            hand.step(rng.choice(special) if special and rng.random() < 0.6 else rng.choice(legal))
    assert {("chi", ""), ("peng", ""), ("gang", "exposed"), ("discard", ""), ("pass", "")} <= kinds
