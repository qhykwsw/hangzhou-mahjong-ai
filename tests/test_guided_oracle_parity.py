"""Oracle inputs (guided.oracle, ADR-0018): both engines give the learner the same masked hidden state, bit for bit."""

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
from guided.oracle import PLANES, SCALARS  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from test_guided_branches import candidates, plain  # noqa: E402
from test_guided_fastgame import learner, members, rule_table  # noqa: E402,F401


def play(game, answer_candidates):
    """Drive a game and its copies as the worker does; the copies' requests, by copy, in order."""
    asked = {}
    while True:
        for shadow in game.shadows:
            while not shadow.done:
                request = shadow.advance()
                if request is None:
                    break
                asked.setdefault(id(shadow), (shadow, []))[1].append(request)
                legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
                shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
        request = game.advance()
        if request is None:
            if game.settled():
                return list(asked.values())
            continue
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        choice, greedy = learner(request["row"], legal)
        game.apply((choice, -0.5, 0.125, greedy, *answer_candidates(request["row"], legal, choice, greedy)))


def oracle_of(row):
    return (row[PLANES].tobytes(), row[SCALARS].tobytes()) if PLANES in row else None


def both(cfg, members, monkeypatch, seats, seed, answer_candidates, entries=None):
    if entries is None:
        by_id = {f"m{i}": m for i, m in enumerate(members)}
        monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
        entries = {k: None for k in by_id}
    focal = seats.index("model:current")
    games = [cls(cfg, {"entries": []}, entries, seats, seed, focal=focal, train=True)
             for cls in (Game, fastgame.FastGame)]
    assert games[1].seats is not None
    asked = [play(game, answer_candidates) for game in games]
    assert games[0].scores == games[1].scores
    assert plain(games[0].rows) == plain(games[1].rows)
    # The copies ask the same questions, with the same oracle inputs.
    seen = [[(request["slot"], oracle_of(request["row"])) for _, requests in copies for request in requests]
            for copies in asked]
    assert seen[0] == seen[1] and seen[1]
    return games[1], asked[1]


def check_rows(rows, keep):
    kept = total = 0
    for row in rows:
        planes, scalars = row[PLANES], row[SCALARS]
        assert planes.dtype == scalars.dtype == np.float16 and planes.shape == (16, 34) and scalars.shape == (4,)
        # Without a critic_features hook the row's oracle_planes is the full oracle: each value is kept or 0.
        full = np.concatenate([row["oracle_planes"].ravel(), row["oracle_scalars"]]).astype(np.float16)
        masked = np.concatenate([planes.ravel(), scalars])
        assert np.all((masked == full) | (masked == 0))
        kept += int(((masked == full) & (full != 0)).sum())
        total += int((full != 0).sum())
    assert total > 1000 and abs(kept / total - keep) < 0.05


def no_candidates(row, legal, choice, greedy):
    return ()


@pytest.mark.parametrize("extra", [
    {},
    {"vine": True, "stable_wall": True, "luck": True},
    {"critic_features": "guided.encoding:oracle", "stable_wall": True},
    {"encoding": "guided-current-hand-v3", "oracle_inputs": {"keep": 0.5, "continuations": False}},
])
def test_both_engines_mask_the_same_oracle(members, monkeypatch, extra):
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "oracle_inputs": {"keep": 0.5}} | extra
    game, copies = both(cfg, members, monkeypatch, ["m0", "model:current", "m2", "m5"],
                        seed_for(83, "oracle", str(sorted(extra))), no_candidates)
    check_rows(game.rows, 0.5)
    greedy = [request for _, requests in copies for request in requests if request["slot"] == GREEDY]
    assert greedy
    continuations = cfg["oracle_inputs"].get("continuations", True)
    assert all((PLANES in request["row"]) == continuations for request in greedy)


def test_python_rule_seats_and_branches_mask_the_same_oracle(members, monkeypatch):
    entries, _ = rule_table(members, monkeypatch, {"py": 2})
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "vine": True, "branches": 3, "oracle_inputs": {"keep": 0.5}}
    game, copies = both(cfg, members, monkeypatch, ["model:current", "py", "m1", "m3"],
                        seed_for(89, "oracle-python"), lambda row, legal, choice, greedy: (candidates(row, legal),),
                        entries=entries)
    assert game.python_seats
    check_rows(game.rows, 0.5)
    assert sum("branches" in row for row in game.rows) > 20


def test_a_copy_on_the_games_own_path_draws_the_games_masks(members, monkeypatch):
    """Under vine the hand plays the argmax; a branch copy of the argmax follows it, and the oracle
    masks of its learner decisions are the ones the game's next rows of the hand drew (same streams)."""
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "vine": True, "branches": 2, "oracle_inputs": {"keep": 0.5}}
    game, copies = both(cfg, members, monkeypatch, ["m1", "model:current", "m3", "m4"],
                        seed_for(97, "oracle-crn"), lambda row, legal, choice, greedy: ((greedy, choice),))
    by_result = {id(copy.row): requests for copy, requests in copies}
    compared = 0
    for j, row in enumerate(game.rows):
        same = by_result.get(id(row["branches"][0]), [])
        later = [r for r in game.rows[j + 1:] if r["hand"] == row["hand"]]
        assert len(same) == len(later)
        for request, mine in zip(same, later):
            assert request["slot"] == GREEDY
            assert oracle_of(request["row"]) == oracle_of(mine)
            assert request["row"]["planes"].tobytes() == mine["planes"].tobytes()
            compared += 1
    assert compared > 50


def test_keep_one_is_the_whole_oracle_and_keep_zero_is_off(members, monkeypatch):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    seats = ["m0", "m3", "model:current", "m5"]
    base = configuration(rounds=4, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True}
    rows = {}
    for name, extra in (("off", {}), ("none", {"oracle_inputs": None}), ("zero", {"oracle_inputs": {"keep": 0}}),
                        ("one", {"oracle_inputs": {"keep": 1.0}})):
        game = fastgame.FastGame(base | extra, {"entries": []}, {k: None for k in by_id}, seats,
                                 seed_for(101, "oracle-keep"), focal=2, train=True)
        play(game, no_candidates)
        rows[name] = game.rows
    assert plain(rows["off"]) == plain(rows["none"]) == plain(rows["zero"])
    assert not any(PLANES in row for row in rows["off"])
    for row, off in zip(rows["one"], rows["off"]):
        assert np.array_equal(row[PLANES], row["oracle_planes"].astype(np.float16))
        assert np.array_equal(row[SCALARS], row["oracle_scalars"].astype(np.float16))
        assert plain({k: v for k, v in row.items() if k not in (PLANES, SCALARS)}) == plain(off)
