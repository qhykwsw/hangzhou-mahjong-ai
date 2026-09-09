import random
import numpy as np
import pytest
from mahjong_ai.domain.engine import HandState, MahjongEngine
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.domain.rules import can_hu
from mahjong_ai.sdk.shape import shanten, analyze
from mahjong_ai.sdk.simulation import sample_world
from mahjong_ai.strategies.common import RulePolicy
from mahjong_ai.strategies.learning import PPOPolicy
from mahjong_ai.strategies.search import SearchPolicy


def test_shanten_against_exhaustive_one_meld_one_pair_targets():
    # Independent minimum missing-tile oracle; includes whites, honors and a suit.
    melds = [(i, i, i) for i in range(33)] + [
        (i, i + 1, i + 2) for i in range(27) if i % 9 < 7
    ]
    targets = [
        tuple(sorted((*meld, pair, pair))) for meld in melds for pair in range(33)
    ]
    from collections import Counter

    targets = [Counter(t) for t in targets]
    rng = random.Random(123)
    for _ in range(100):
        wall = [x for x in (0, 1, 2, 3, 27, 28, 33) for _ in range(4)]
        rng.shuffle(wall)
        hand = tuple(sorted(wall[: rng.choice([4, 5])]))
        natural = Counter(t for t in hand if t != 33)
        white = hand.count(33)
        expected = max(
            -1,
            min(
                sum(max(0, n - natural[t]) for t, n in target.items())
                for target in targets
            )
            - white
            - 1,
        )
        assert shanten(hand, 3) == expected, (hand, expected)


def test_legal_waits_use_remaining_copies_and_fan():
    hand = (0, 0, 0, 1, 2, 3, 9, 10, 11, 18, 19, 20, 27)
    result = analyze(hand)
    assert (
        result["shanten"] == 0
        and 27 in result["winning_tiles"]
        and 33 in result["winning_tiles"]
    )
    assert all(can_hu((*hand, t)) for t in result["winning_tiles"])
    empty = [0] * 34
    assert analyze(hand, remaining=empty)["winning_copies"] == 0


def test_hypothetical_world_reconstructs_responses_and_conservation():
    policy = RulePolicy()
    for seed in range(8):
        hand = HandState(RoundConfig(rounds=1), seed)
        while not hand.settled:
            obs = hand.observation()
            world = sample_world(obs, seed + hand.steps)
            world.assert_conservation()
            assert world.observation().hand == obs.hand
            assert set(world.legal_actions()) == set(obs.legal_actions)
            hand.step(policy.act(obs))


def test_ppo_real_episode_updates_actor_and_value(tmp_path):
    learner = PPOPolicy("ppo", tmp_path, training=True, epochs=2)
    result = MahjongEngine().run_round(
        [learner, RulePolicy(), RulePolicy(), RulePolicy()], RoundConfig(rounds=2), 28
    )
    assert result.completed and learner.buffer
    old = {k: v.copy() for k, v in learner.params.items()}
    for row in learner.buffer:
        assert row.source == "executed-terminal-return"
        assert len(row.prior) == len(row.features)
    learner.update()
    learner.save(tmp_path)
    assert learner.updates == 1
    assert any(not np.array_equal(old[k], v) for k, v in learner.params.items())
    loaded = PPOPolicy("ppo", tmp_path)
    assert (
        loaded.act(HandState(RoundConfig(), 1).observation())
        in HandState(RoundConfig(), 1).legal_actions()
    )


@pytest.mark.parametrize("algorithm", ["rollout", "mcts"])
def test_search_completes_real_simulations(algorithm):
    obs = HandState(RoundConfig(), 3).observation()
    policy = SearchPolicy(algorithm=algorithm, simulations=4, seconds=0.1)
    policy.reset(5)
    assert policy.act(obs) in obs.legal_actions
    assert policy.last_search["simulations"] > 0
    if algorithm == "mcts":
        assert policy.last_search["nodes"] > len(obs.legal_actions)
