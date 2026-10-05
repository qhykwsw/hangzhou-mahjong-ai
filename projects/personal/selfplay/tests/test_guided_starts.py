"""Openings: a one-hand training game from a recorded position (guided.starts)."""

import random

import numpy as np
import pytest

from guided.environment import GREEDY, Game
from guided.starts import check, prefix, replay
from test_guided_playout import CFG, Stubs


def record(seed, dealer, hand_no, stop, rng):
    """Play a hand with random legal moves; return the opening before the ``stop``-th choice and that state."""
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig

    state = HandState(RoundConfig(rounds=CFG["rounds"], allow_tiebreak=False), seed, dealer, hand_no, (5, -5, 0, 0))
    steps, choices = [], 0
    while not state.settled:
        legal = state.legal_actions()
        if len(legal) > 1:
            if choices == stop:
                opening = {"seed": seed, "dealer": dealer, "hand_no": hand_no, "scores": [5, -5, 0, 0],
                           "prefix": prefix(steps), "focal": state.turn}
                return opening, state
            choices += 1
        action = rng.choice(legal)
        steps.append(action)
        state.step(action, legal)
    return None, None


def openings(count):
    rng, found = random.Random(7), []
    for seed in range(200):
        opening, state = record(seed, seed % 4, 1 + seed % CFG["rounds"], rng.randrange(1, 40), rng)
        if opening:
            found.append((opening, state))
        if len(found) == count:
            return found
    raise AssertionError("no openings")


def drive(game, rng):
    while not game.settled():
        for shadow in game.shadows:
            if not shadow.done:
                request = shadow.advance()
                if request is not None:
                    assert request["slot"] == GREEDY
                    shadow.apply((int(np.flatnonzero(request["row"]["mask"])[0]), 0.0, 0.0, 0))
        request = game.advance()
        if request is None:
            continue
        legal = np.flatnonzero(request["row"]["mask"])
        greedy = int(legal[0])
        chosen = int(rng.choice(legal)) if rng.random() < 0.3 else greedy
        game.apply((chosen, 0.0, 0.0, greedy))


def test_an_opening_reaches_the_recorded_position_and_plays_one_hand():
    for opening, original in openings(6):
        seats = ["stub"] * 4
        seats[opening["focal"]] = "model:current"
        game = Game(CFG, None, {}, seats, 11, focal=opening["focal"], train=True, external=Stubs(), opening=opening)
        state = game.state
        assert state.hands == original.hands and state.wall == original.wall
        assert (state.turn, state.phase, state.melds, state.discards) == (
            original.turn, original.phase, original.melds, original.discards)
        assert game.hand_no == opening["hand_no"] and game.dealer == opening["dealer"]
        assert [e for e, _ in state.events] == [e for e, _ in original.events]
        drive(game, random.Random(3))
        assert game.complete and {r["hand"] for r in game.rows} == {opening["hand_no"]}
        assert game.rows and game.rows[0]["mask"].sum() >= 2  # the learner decides first
        assert all("playout" in r for r in game.rows if r["deviated"])


def test_an_opening_must_stop_at_its_learners_choice():
    opening, original = openings(1)[0]
    with pytest.raises(ValueError):
        Game(CFG, None, {}, ["model:current", "stub", "stub", "stub"], 11, focal=(opening["focal"] + 1) % 4,
             train=True, external=Stubs(), opening=dict(opening, focal=(opening["focal"] + 1) % 4))
    from selfplay_core.actions import action_index

    legal = {action_index(a) for a in original.legal_actions()}
    illegal = next(i for i in range(231) if i not in legal)
    with pytest.raises(ValueError):
        replay(original, bytes([illegal]).hex(), opening["focal"])
    with pytest.raises(ValueError):
        check(dict(opening, prefix=[0, 1]))  # the prefix is the hex of action indices


def test_a_reshuffled_opening_keeps_the_position_and_draws_a_fresh_wall():
    cfg = dict(CFG, starts_reshuffle=True, stable_wall=True, luck=True)
    differed = 0
    for opening, original in openings(6):
        seats = ["stub"] * 4
        seats[opening["focal"]] = "model:current"
        game = Game(cfg, None, {}, seats, 11, focal=opening["focal"], train=True, external=Stubs(), opening=opening)
        state = game.state
        assert state.hands == original.hands and sorted(state.wall) == sorted(original.wall)
        assert (state.turn, state.phase, state.melds, state.discards) == (
            original.turn, original.phase, original.melds, original.discards)
        differed += state.wall != original.wall
        drive(game, random.Random(3))
        assert game.complete and all("luck_pay" in r for r in game.rows)
    assert differed == 6
