"""The incremental per-seat event views must equal the legacy two-pass masking."""

import random
from dataclasses import replace

import pytest

from mahjong_ai.domain.engine import HandState, derive_seed
from mahjong_ai.domain.observations import policy_events
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.sdk.simulation import sample_world
from mahjong_ai.strategies.common import RulePolicy


def legacy_public_events(state, seat):
    """The observation() projection before views were maintained incrementally."""
    public = []
    for e, private in state.events:
        if private is None or private == seat:
            public.append(e)
        else:
            public.append(replace(e, tile=None, tiles=()))
    return policy_events(public, seat)


def mixed_chooser(seed):
    rng = random.Random(seed)
    rule = RulePolicy()

    def choose(obs, legal):
        # Prefer claims and kongs often enough to exercise every event kind.
        special = [a for a in legal if a.action in ("chi", "peng", "gang")]
        roll = rng.random()
        if special and roll < 0.5:
            return rng.choice(special)
        if roll < 0.75:
            return rule.act(obs)
        return rng.choice(legal)

    return choose


def play_game(config, seed, chooser, carry_views=True):
    history, views, scores, dealer = [], None, [0] * 4, seed % 4
    decisions = 0
    for hand_no in range(1, config.rounds + 1):
        hand = HandState(
            config,
            derive_seed(seed, "wall", hand_no),
            dealer,
            hand_no,
            scores,
            history,
            previous_views=views if carry_views else None,
        )
        while not hand.settled:
            legal = hand.legal_actions()
            obs = hand.observation()
            assert obs.public_events == legacy_public_events(hand, hand.turn)
            decisions += 1
            hand.step(legal[0] if len(legal) == 1 else chooser(obs, legal))
        for seat in range(4):
            assert tuple(hand.public_views()[seat]) == legacy_public_events(hand, seat)
        scores = [s + d for s, d in zip(scores, hand.delta)]
        history, views = hand.events, hand.public_views()
        if hand.winner is not None:
            dealer = hand.winner
    return decisions


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("you_cai_bi_kao", [False, True])
def test_views_match_legacy_projection_across_hands(seed, you_cai_bi_kao):
    config = RoundConfig(rounds=8, you_cai_bi_kao=you_cai_bi_kao)
    assert play_game(config, seed, mixed_chooser(seed)) > 8


def test_views_rebuild_from_history_without_seed():
    config = RoundConfig(rounds=4)
    play_game(config, 11, mixed_chooser(11), carry_views=False)


def test_views_handle_externally_built_states():
    rng = random.Random(5)
    hand = HandState(RoundConfig(rounds=2), 5)
    for _ in range(40):
        if hand.settled:
            break
        legal = hand.legal_actions()
        hand.step(rng.choice(legal))
    if not hand.settled:
        obs = hand.observation()
        world = sample_world(obs, 9)
        # sample_world bypasses __init__; the views must be rebuilt lazily.
        assert world.observation().public_events == legacy_public_events(
            world, world.turn
        )
        # Replacing the event list is detected and re-projected.
        world.events = list(world.events)[: len(world.events) // 2]
        assert world.observation().public_events == legacy_public_events(
            world, world.turn
        )
