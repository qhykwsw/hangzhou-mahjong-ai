"""Search at key windows (guided.search): candidates played out in sampled worlds."""

import random

import numpy as np

from guided.environment import GREEDY, Game
from test_guided_playout import CFG, Stubs
from test_guided_starts import drive, record

SEARCH_CFG = dict(CFG, search={"kinds": ["kong"], "worlds": 3, "max_windows": 1},
                  finish_game="test_guided_search:finish")


def finish(rows, cfg):
    """Runs after every search of the game: its rows already carry their results."""
    for row in rows:
        assert "search_kind" not in row or len(row["search_values"]) == row["search_worlds"]
        row["adv"] = row["ret"] = 0.0


def kong_openings(count):
    """Openings whose learner may kong first (random play rarely gets there, so scan)."""
    found, rng = [], random.Random(5)
    for seed in range(3000):
        for stop in range(0, 60, 3):
            opening, state = record(seed, seed % 4, 1, stop, rng)
            if opening is None:
                break
            if any(a.action == "gang" for a in state.legal_actions()):
                found.append(opening)
                break
        if len(found) == count:
            return found
    raise AssertionError("no kong window found")


def test_a_kong_window_is_searched_over_its_alternatives_in_shared_worlds():
    from selfplay_core.actions import action_index

    for opening in kong_openings(3):
        seats = ["stub"] * 4
        seats[opening["focal"]] = "model:current"
        game = Game(SEARCH_CFG, None, {}, seats, 21, focal=opening["focal"], train=True, external=Stubs(),
                    opening=opening)
        gangs = {action_index(a) for a in game.state.legal_actions() if a.action == "gang"}
        drive(game, random.Random(4))
        row = game.rows[0]
        assert row["search_kind"] == "kong"
        actions = row["search_actions"]
        assert gangs <= set(actions) and len(set(actions)) == len(actions)
        assert any(a not in gangs for a in actions)  # the best non-kong move is a candidate too
        assert 1 <= row["search_worlds"] <= 3
        assert all(len(values) == len(actions) for values in row["search_values"])
        assert len(row["search_q"]) == len(actions)
        assert all("search_kind" not in r for r in game.rows[1:])  # max_windows


def test_without_search_a_game_is_unchanged():
    opening = kong_openings(1)[0]
    seats = ["stub"] * 4
    seats[opening["focal"]] = "model:current"
    plain = Game(CFG, None, {}, seats, 21, focal=opening["focal"], train=True, external=Stubs(), opening=opening)
    drive(plain, random.Random(4))
    assert all("search_kind" not in r for r in plain.rows)


def test_a_kong_argmax_asks_the_actor_for_its_best_non_kong_move():
    from selfplay_core.actions import action_index

    opening = kong_openings(1)[0]
    seats = ["stub"] * 4
    seats[opening["focal"]] = "model:current"
    game = Game(SEARCH_CFG, None, {}, seats, 21, focal=opening["focal"], train=True, external=Stubs(), opening=opening)
    gangs = {action_index(a) for a in game.state.legal_actions() if a.action == "gang"}
    probes = []
    first = True
    while not game.settled():
        for shadow in game.shadows:
            if not shadow.done:
                request = shadow.advance()
                if request is not None:
                    assert request["slot"] == GREEDY
                    legal = np.flatnonzero(request["row"]["mask"])
                    if not set(legal) & gangs and len(legal) < len(request["row"]["mask"]):
                        probes.append(set(legal))
                    shadow.apply((int(legal[0]), 0.0, 0.0, 0))
        request = game.advance()
        if request is None:
            continue
        legal = np.flatnonzero(request["row"]["mask"])
        greedy = max(int(i) for i in legal if i in gangs) if first else int(legal[0])  # the argmax kongs at the window
        first = False
        game.apply((greedy, 0.0, 0.0, greedy))
    row = game.rows[0]
    assert row["search_actions"][0] in gangs
    assert probes and not probes[0] & gangs  # the question excludes every kong
    assert any(a not in gangs for a in row["search_actions"])


def test_collection_searches_openings_through_the_parents_inference(monkeypatch):
    """End to end: workers ask the parent for masked argmaxes and play the worlds out."""
    import multiprocessing as mp
    import time

    import torch

    from guided import artifacts, collection
    from guided.config import configuration
    from guided.networks import Actor
    from mahjong_ai.strategies.heuristic import HeuristicPolicy

    monkeypatch.setattr(artifacts, "read_bundle", lambda _: {"teacher": "rule", "entries": [{"id": "rule"}]})
    monkeypatch.setattr(artifacts, "bind_runtime", lambda _: None)
    monkeypatch.setattr(artifacts, "build_policy", lambda b, cid: HeuristicPolicy(style="balanced"))
    fork = mp.get_context("fork")
    monkeypatch.setattr(collection.mp, "get_context", lambda _: fork)
    cfg = configuration(rounds=2, tables=2, workers=1, samples=6, lookahead=False) | {
        "cuda_graphs": False, "playout_baseline": True,
        "search": {"kinds": ["kong"], "worlds": 2, "max_windows": 1}}
    command = dict(kind="collect", iteration=1, deadline=time.time() + 120, labels=False, pool=["rule"],
                   samples=6, starts=kong_openings(2))
    torch.manual_seed(3)
    rows, records, _ = collection.collect(Actor(16, 1, 1), None, cfg, "fixture", "fixture", command, {})
    searched = [r for r in rows if "search_kind" in r]
    assert records and all(r["hands"] == 1 for r in records)
    assert searched and all(r["search_worlds"] >= 1 and len(r["search_q"]) == len(r["search_actions"])
                            for r in searched)
