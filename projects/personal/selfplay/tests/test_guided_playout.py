"""The playout baseline: a learner deviation's greedy continuation of its hand."""

import random

import numpy as np

from guided.environment import GREEDY, Game


class Stub:
    """A deterministic opponent: the legal action with the smallest index."""

    def reset(self, seed):
        pass

    def act(self, obs):
        from selfplay_core.actions import action_index

        return min(obs.legal_actions, key=action_index)

    def finish_round(self, result, seat):
        pass


class Stubs:
    specs = {"stub": {}}

    def acquire(self, spec):
        return Stub()


CFG = {"rounds": 2, "lookahead": False, "gae_lambda": 0.9, "playout_baseline": True,
       "finish_game": "test_guided_playout:finish"}


def finish(rows, cfg):
    """The finish_game hook runs after the continuations: every deviation has its playout."""
    for row in rows:
        assert not row.get("deviated") or "playout" in row
        row["adv"] = row["ret"] = 0.0


def play(seed, deviation, rng):
    """Drive one game like the collection worker; the learner's argmax is its smallest legal index."""
    game = Game(CFG, None, {}, ["model:current", "stub", "stub", "stub"], seed, train=True, external=Stubs())
    while not game.settled():
        for shadow in game.shadows:
            if not shadow.done:
                request = shadow.advance()
                if request is not None:
                    assert request["slot"] == GREEDY and not request["train"]
                    shadow.apply((int(np.flatnonzero(request["row"]["mask"])[0]), 0.0, 0.0, 0))
        request = game.advance()
        if request is None:
            continue
        legal = np.flatnonzero(request["row"]["mask"])
        greedy = int(legal[0])
        chosen = int(rng.choice(legal)) if len(legal) > 1 and rng.random() < deviation else greedy
        game.apply((chosen, 0.0, 0.0, greedy))
    return game


def test_until_its_first_deviation_a_hand_follows_the_greedy_path():
    checked = 0
    for seed in range(12):
        game = play(seed, 0.3, random.Random(seed))
        first = next((r for r in game.rows if r["hand"] == 1 and r["deviated"]), None)
        if first is None:
            continue
        greedy = play(seed, 0.0, random.Random(seed))
        assert not greedy.shadows and not any(r["deviated"] for r in greedy.rows)
        outcome = sum(r["reward"] for r in greedy.rows if r["hand"] == 1)
        assert first["playout"] == outcome  # the deviation's continuation is the all-greedy hand
        assert first["playout_deals_next"] in (0.0, 1.0)
        assert all("playout" in r for r in game.rows if r["deviated"])
        assert all("adv" in r for r in game.rows)  # finished after the continuations
        checked += 1
    assert checked >= 3


def test_without_the_playout_baseline_nothing_is_played_out():
    game = Game(dict(CFG, playout_baseline=False), None, {}, ["model:current", "stub", "stub", "stub"], 5,
                train=True, external=Stubs())
    rng = random.Random(5)
    while not game.settled():
        request = game.advance()
        if request is None:
            continue
        legal = np.flatnonzero(request["row"]["mask"])
        game.apply((int(rng.choice(legal)), 0.0, 0.0))  # evaluation-style answer without the argmax
    assert not game.shadows and not any("deviated" in r for r in game.rows)
