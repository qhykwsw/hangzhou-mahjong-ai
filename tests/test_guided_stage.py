"""Training games inside a tournament stage (guided.stage): what every seat sees, on both engines."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame, stage  # noqa: E402
from guided.config import configuration  # noqa: E402
from guided.encoding import ENCODING_V2, encoder  # noqa: E402
from guided.environment import Game, fork  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from mahjong_ai.domain.settlement import score_ranks  # noqa: E402
from mahjong_ai.domain.types import StageKind  # noqa: E402
from test_guided_fastgame import learner, members, rows_of  # noqa: E402,F401

STAGE = {"kind": "advancement", "games": 10, "per_game_sd": 70}


def staged_seed(tag, want=lambda g: g > 0):
    """A game seed whose stage start satisfies ``want`` (a later game, by default)."""
    for n in range(1000):
        seed = seed_for(43, tag, n)
        if want(stage.start({"stage": STAGE}, seed)["game"]):
            return seed
    raise AssertionError("no such seed")


class Recording:
    """A rule seat that records the stage of every observation it is handed."""

    def __init__(self, policy, seen):
        self.policy, self.seen = policy, seen

    def reset(self, seed):
        self.policy.reset(seed)

    def finish_round(self, result, seat):
        self.policy.finish_round(result, seat)

    def act(self, obs):
        self.seen.append(obs.stage)
        return self.policy.act(obs)


def play_all(game):
    """Drive a game and its continuations with the scripted learner; every request's row, in order."""
    rows = []
    while True:
        for shadow in game.shadows:
            while not shadow.done:
                request = shadow.advance()
                if request is None:
                    break
                rows.append(request["row"])
                legal = [i for i in np.flatnonzero(request["row"]["mask"])]
                shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
        request = game.advance()
        if request is None:
            if game.settled():
                return rows
            continue
        rows.append(request["row"])
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        choice, greedy = learner(request["row"], legal)
        game.apply((choice, -0.5, 0.125, greedy))


def test_the_carried_scores_sum_to_zero_with_the_set_spread():
    starts = [stage.start({"stage": STAGE}, seed_for(5, "stage-prior", n)) for n in range(6000)]
    assert all(sum(s["prior"]) == 0 and all(type(x) is int for x in s["prior"]) for s in starts)
    assert all(s["prior"] == (0, 0, 0, 0) for s in starts if s["game"] == 0)
    counts = np.bincount([s["game"] for s in starts], minlength=10)
    assert counts.min() > 480 and counts.max() < 720
    scaled = np.array([np.array(s["prior"]) / np.sqrt(s["game"]) for s in starts if s["game"] > 0])
    assert abs(scaled.std() - 70) < 2.5
    assert stage.start({}, 1) is None and stage.start({"stage": None}, 1) is None


def test_the_stage_draws_from_its_own_stream():
    seed = seed_for(7, "own-stream")
    assert stage.start({"stage": STAGE}, seed) == stage.start({"stage": STAGE}, seed)
    assert stage.start({"stage": dict(STAGE, kind="final")}, seed)["kind"] is StageKind.FINAL


def test_every_seat_sees_the_stage_as_the_settlement_writes_it(members, monkeypatch):
    seen = []
    by_id = {f"m{i}": Recording(m, seen) for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True, "stage": STAGE}
    seed = staged_seed("observe")
    game = Game(cfg, {"entries": []}, {k: None for k in by_id}, ["m0", "model:current", "m2", "m3"], seed,
                focal=1, train=True)
    info = game.stage
    by_hand, scores = {}, [0] * 4
    original = Game._next_hand

    def next_hand(self):  # the game's scores before each hand
        original(self)
        by_hand[self.hand_no] = tuple(self.scores)

    monkeypatch.setattr(Game, "_next_hand", next_hand)
    by_hand[1] = (0, 0, 0, 0)
    play_all(game)
    assert seen
    for ctx in seen:
        before = by_hand[ctx.round_no]
        want = tuple(p + s for p, s in zip(info["prior"], before))
        assert ctx.kind is StageKind.ADVANCEMENT and ctx.stage_scores == want
        assert ctx.rank_by_seat == score_ranks(want)
        assert ctx.stage_round_no == 16 * info["game"] + ctx.round_no and ctx.stage_rounds_total == 160
        assert tuple(ctx.score_by_seat) == before and not ctx.in_tiebreak
    rows = [r for r in game.rows]
    assert rows and all(r["stage_game"] == info["game"] for r in rows)
    assert all(r["stage_prior"] == [info["prior"][(1 + k) % 4] for k in range(4)] for r in rows)
    assert game.state.stage_prior == info["prior"] and game.state.stage_game == info["game"]
    copy = fork(game.state)
    assert (copy.stage_prior, copy.stage_game, copy.stage_kind) == (info["prior"], info["game"], info["kind"])


@pytest.mark.parametrize("encoding", [None, ENCODING_V2])
def test_the_encoders_read_the_stage_and_the_c_rewrite_matches(members, monkeypatch, encoding):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {"stage": STAGE}
    if encoding:
        cfg["encoding"] = encoding
    encode = encoder(cfg.get("encoding") or "guided-current-hand-v1")
    game = Game(cfg, {"entries": []}, {k: None for k in by_id}, ["model:current", "m1", "m2", "m3"],
                staged_seed("rewrite"), train=True)
    state, checked, differs = game.state, 0, 0
    while not state.settled and checked < 40:
        legal = state.legal_actions()
        if len(legal) > 1:
            plain = state.observation(legal)
            row = encode(plain, True)
            want = encode(stage.observe(plain, game.stage, state.scores, state.hand_no, 16), True)
            got = stage.rewrite(dict(row), game.stage, state.scores, state.hand_no, 16, state.turn)
            assert got["scalars"].dtype == want["scalars"].dtype
            assert got["scalars"].tobytes() == want["scalars"].tobytes()
            assert got["planes"].tobytes() == want["planes"].tobytes()
            differs += row["scalars"].tobytes() != want["scalars"].tobytes()
            checked += 1
        state.step(legal[0], legal)
    assert checked > 10 and differs == checked


@pytest.mark.parametrize("kind,encoding", [("advancement", None), ("final", None), ("advancement", ENCODING_V2)])
def test_a_fast_game_inside_a_stage_plays_the_python_game(members, monkeypatch, kind, encoding):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "stable_wall": True, "luck": True,
        "critic_features": "guided.encoding:oracle", "stage": dict(STAGE, kind=kind)}
    if encoding:
        cfg["encoding"] = encoding
    seed = staged_seed(f"fast-{kind}")
    seats = ["m0", "m2", "model:current", "m5"]
    games = [cls(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed, focal=2, train=True)
             for cls in (Game, fastgame.FastGame)]
    assert games[1].seats is not None
    requests = [play_all(game) for game in games]
    assert len(requests[0]) == len(requests[1]) > 50
    for key in ("scalars", "planes"):
        assert [r[key].tobytes() for r in requests[0]] == [r[key].tobytes() for r in requests[1]]
    assert rows_of(games[0]) == rows_of(games[1])
    assert any(r.get("deviated") for r in games[1].rows)
    # The same game without the stage plays the same moves, only the scalars differ.
    plain = fastgame.FastGame(dict(cfg, stage=None), {"entries": []}, {k: None for k in by_id}, seats, seed,
                              focal=2, train=True)
    plain_requests = play_all(plain)
    assert plain.scores == games[1].scores
    assert [r["planes"].tobytes() for r in plain_requests] == [r["planes"].tobytes() for r in requests[1]]
    assert all(a["scalars"].tobytes() != b["scalars"].tobytes() for a, b in zip(plain_requests, requests[1]))
    assert not any(k.startswith("stage_") for row in plain.rows for k in row)


def test_a_critic_feature_hook_reads_the_stage_on_both_engines(members, monkeypatch):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "critic_features": "test_guided_stage:stage_features", "stage": STAGE}
    seed = staged_seed("features")
    games = [cls(cfg, {"entries": []}, {k: None for k in by_id}, ["model:current", "m1", "m2", "m3"], seed,
                 train=True) for cls in (Game, fastgame.FastGame)]
    for game in games:
        play_all(game)
    assert rows_of(games[0]) == rows_of(games[1])
    assert all(r["read_game"] == games[1].stage["game"] for r in games[1].rows)


def stage_features(state, seat):
    return {"read_prior": list(state.stage_prior), "read_game": state.stage_game, "read_kind": str(state.stage_kind)}
