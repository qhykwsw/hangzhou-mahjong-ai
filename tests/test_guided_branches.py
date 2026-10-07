"""Branches of a learning decision (guided.branches): the copies play the same on both engines, bit for bit."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame  # noqa: E402  (fastgame first: members register through it)
from guided.config import configuration  # noqa: E402
from guided.environment import GREEDY, Game  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from test_guided_fastgame import learner, members, rule_table  # noqa: E402,F401


def candidates(row, legal):
    """Scripted candidates: none at some rows, else up to three legal actions picked from the row's bytes."""
    digest = int(np.frombuffer(row["planes"].tobytes(), dtype=np.uint16).sum()) + 3 * int(row["events"].sum())
    if digest % 4 == 0 or len(legal) < 2:
        return ()
    return tuple(legal[(digest // 5 + j * (1 + digest % 3)) % len(legal)] for j in range(min(3, len(legal) - 1)))


def play(game, slots=None):
    """Drive a game and its copies as the worker does, the scripted learner answering (with candidates)."""
    requests = 0
    while True:
        for shadow in game.shadows:
            while not shadow.done:
                request = shadow.advance()
                if request is None:
                    break
                if slots is not None:
                    slots.append(request["slot"])
                legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
                shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
                requests += 1
        request = game.advance()
        if request is None:
            if game.settled():
                return requests
            continue
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        choice, greedy = learner(request["row"], legal)
        game.apply((choice, -0.5, 0.125, greedy, candidates(request["row"], legal)))
        requests += 1


def plain(value):
    """A row (with nested results) as comparable values, arrays by dtype, shape and bytes."""
    if hasattr(value, "tobytes"):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(v) for v in value)
    return value


def both(cfg, members, monkeypatch, seats, seed, *, entries=None):
    """The same game on the Python and the C engine, played out; (games, request counts)."""
    if entries is None:
        by_id = {f"m{i}": m for i, m in enumerate(members)}
        monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
        entries = {k: None for k in by_id}
    focal = seats.index("model:current")
    games = [cls(cfg, {"entries": []}, entries, seats, seed, focal=focal, train=True)
             for cls in (Game, fastgame.FastGame)]
    assert games[1].seats is not None
    counts = [play(game) for game in games]
    assert counts[0] == counts[1] > 50
    assert games[0].scores == games[1].scores
    assert plain(games[0].rows) == plain(games[1].rows)
    return games, counts


BASE = {"playout_baseline": True, "branches": 3}


@pytest.mark.parametrize("extra", [
    {},
    {"vine": True, "stable_wall": True, "luck": True},
    {"stage": {"kind": "advancement", "games": 10, "per_game_sd": 70}, "encoding": "guided-current-hand-v4"},
])
def test_branch_copies_play_the_same_on_both_engines(members, monkeypatch, extra):
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra
    games, _ = both(cfg, members, monkeypatch, ["m0", "model:current", "m2", "m5"],
                    seed_for(61, "branches", str(sorted(extra))))
    rows = games[1].rows
    branched = [row for row in rows if "branches" in row]
    assert len(branched) > 20 and len(branched) < len(rows)
    for row in branched:
        legal = np.flatnonzero(row["mask"])
        assert [b["action"] for b in row["branches"]] == list(candidates(row, list(legal)))
        for result in row["branches"]:
            assert result["truncated"] is False and result["playout_dealer"] in range(4)
            assert len(result["playout_scores"]) == 4 and result["playout_deals_next"] in (0.0, 1.0)
            assert ("playout_luck_pay" in result) == bool(extra.get("luck"))
    assert not any("branches" in row for row in rows if not candidates(row, list(np.flatnonzero(row["mask"]))))


@pytest.mark.parametrize("extra", [{}, {"vine": True, "stable_wall": True, "luck": True}])
def test_branch_copies_with_python_rule_seats_play_the_same_on_both_engines(members, monkeypatch, extra):
    entries, _ = rule_table(members, monkeypatch, {"py": 2})
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra
    games, _ = both(cfg, members, monkeypatch, ["model:current", "py", "m1", "m3"],
                    seed_for(67, "branches-python", str(sorted(extra))), entries=entries)
    assert games[1].python_seats
    assert sum("branches" in row for row in games[1].rows) > 20


def test_a_candidate_copy_is_the_hand_it_continues(members, monkeypatch):
    """Under vine the hand plays the argmax and the deviation's copy the sampled move, each continuing
    greedily: a candidate equal to the argmax replays the hand itself, one equal to the sampled move
    its deviation copy (same wall, same seats, the same answers)."""
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "vine": True, "branches": 2, "stable_wall": True, "luck": True}
    seats = ["m1", "model:current", "m3", "m4"]
    for cls in (Game, fastgame.FastGame):
        game = cls(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed_for(71, "branch-identity"),
                   focal=1, train=True)
        slots = []
        while True:
            for shadow in game.shadows:
                while not shadow.done:
                    request = shadow.advance()
                    if request is None:
                        break
                    slots.append(request["slot"])
                    legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
                    shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
            request = game.advance()
            if request is None:
                if game.settled():
                    break
                continue
            legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
            choice, greedy = learner(request["row"], legal)
            game.apply((choice, -0.5, 0.125, greedy, (greedy, choice)))
        assert set(slots) == {GREEDY}
        outcome = {}
        for row in game.rows:
            outcome[row["hand"]] = outcome.get(row["hand"], 0.0) + row["reward"]
        checked = 0
        for row in game.rows:
            same, deviation = row["branches"]
            # The hand after this decision is the argmax path: the first branch is its outcome.
            assert same["playout"] == outcome[row["hand"]]
            if row["deviated"]:
                assert deviation["playout"] == row["playout"]
                assert deviation["playout_luck_pay"] == row["playout_luck_pay"]
                checked += 1
            else:
                assert deviation == same
        assert checked > 5


def test_without_branches_no_row_carries_them_and_stray_candidates_fail(members, monkeypatch):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True}
    game = fastgame.FastGame(cfg, {"entries": []}, {k: None for k in by_id}, ["model:current", "m1", "m2", "m3"],
                             seed_for(3, "no-branches"), focal=0, train=True)
    assert game.branching is None
    request = game.advance()
    legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
    with pytest.raises(ValueError, match="need cfg"):
        game.apply((legal[0], 0.0, 0.0, legal[0], (legal[0],)))
    with pytest.raises(ValueError, match="playout_baseline"):
        Game(configuration(rounds=16, lookahead=True) | {"branches": 2}, {"entries": []}, {k: None for k in by_id},
             ["model:current", "m1", "m2", "m3"], 1, train=True)
    with pytest.raises(ValueError, match="positive integer"):
        Game(cfg | {"branches": True}, {"entries": []}, {k: None for k in by_id},
             ["model:current", "m1", "m2", "m3"], 1, train=True)
    # Evaluation games never branch.
    assert Game(cfg | {"branches": 2}, {"entries": []}, {k: None for k in by_id},
                ["model:current", "m1", "m2", "m3"], 1).branching is None
