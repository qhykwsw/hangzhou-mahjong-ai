"""The C training game (guided/native.c) plays the Python engine's games bit for bit."""

from pathlib import Path
import importlib.util
import random
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame  # noqa: E402  (fastgame first: members register through it)
from guided.config import configuration  # noqa: E402
from guided.encoding import encode, oracle  # noqa: E402
from guided.environment import Game  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from mahjong_ai.domain.engine import HandState  # noqa: E402
from mahjong_ai.domain.types import RoundConfig  # noqa: E402
from selfplay_core.actions import action_index  # noqa: E402

CONFIG = RoundConfig(rounds=16, allow_tiebreak=False)

# A fixed pool member's shape: the whole decision in the runtime's race_act.
MEMBER = """
from mahjong_ai import _fast
from mahjong_ai.sdk.shape import analyze

Q = 0.15
PROFILE = _fast.race_profile(Q, 0.467, 0.22, 0.45, 0.10, 0.30, {hunt}, {upgrade}, 1 - (1 - Q) ** (1 / 3), 1e-7, 20, {style})


def examined(hand, opened, available, chain=0, piao=0):
    return analyze(hand, opened, available, chain, piao)


class Race:
    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass


class Native(Race):
    def act(self, obs):
        return obs.legal_actions[_fast.race_act(obs, PROFILE, examined)]
"""


@pytest.fixture(scope="module")
def members(tmp_path_factory):
    folder = tmp_path_factory.mktemp("members")
    built = []
    for i, (style, hunt, upgrade) in enumerate([(0, 1.0, True), (1, 1.0, True), (2, 1.0, False), (3, 1.0, True),
                                                (4, 0.8, True), (5, 1.2, True)]):
        path = folder / f"member{i}.py"
        path.write_text(MEMBER.format(style=style, hunt=hunt, upgrade=upgrade))
        spec = importlib.util.spec_from_file_location(f"fastgame_member{i}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        built.append(module.Native())
    return built


def snapshot_of(state):
    events = [
        (e.kind, e.seat, e.tile, tuple(e.tiles), tuple(e.scores), e.concealed, private)
        for e, private in state.events[state.hand_start:]
        if e.kind != "deal"
    ]
    return {
        "dealer": state.dealer, "hand_no": state.hand_no, "rounds_total": state.rounds_total,
        "scores": tuple(state.scores), "wall": tuple(state.wall),
        "hands": tuple(tuple(h) for h in state.hands), "initial_hands": tuple(tuple(h) for h in state.initial_hands),
        "melds": tuple(tuple((m.kind, tuple(m.tiles), m.from_seat, m.concealed) for m in row) for row in state.melds),
        "discards": tuple(tuple(r) for r in state.discards), "white_acquired": tuple(state.white_acquired),
        "events": tuple(events), "phase": state.phase, "turn": state.turn, "drawn": state.drawn,
        "kong_draw": state.kong_draw, "chain": tuple(state.chain), "piao": tuple(state.piao),
        "catch_owner": state.catch_owner, "response_queue": tuple(state.response_queue),
        "discarder": state.discarder, "discarded": state.discarded, "winner": state.winner,
        "fan": state.fan.fan if state.fan else None, "delta": tuple(state.delta), "settled": state.settled,
        "steps": state.steps,
    }


def test_the_c_hand_is_the_python_hand_at_every_step():
    rng = random.Random(7)
    for _ in range(40):
        seed, dealer, hand_no = rng.getrandbits(64), rng.randrange(4), rng.randrange(1, 17)
        scores = tuple(rng.randrange(-50, 50) for _ in range(4))
        state = HandState(CONFIG, seed, dealer, hand_no, scores)
        state.rounds_total = 16
        hand = fastgame.native.Hand(seed, dealer, hand_no, scores, 16)
        applied = []
        while True:
            got = hand.snapshot()
            assert got.pop("log") == tuple(applied)
            assert got == snapshot_of(state)
            if state.settled:
                break
            legal = state.legal_actions()
            assert hand.legal() == tuple(action_index(a) for a in legal)
            choice = rng.randrange(len(legal))
            state.step(legal[choice], legal)
            hand.step(action_index(legal[choice]))
            applied.append(action_index(legal[choice]))


def test_members_encoding_and_oracle_match_at_every_decision(members):
    rng = random.Random(11)
    plans = [fastgame.race_member(m) for m in members]
    assert all(plans)
    fallbacks = decisions = 0
    for _ in range(12):
        seed, dealer, hand_no = rng.getrandbits(64), rng.randrange(4), rng.randrange(1, 17)
        scores = tuple(rng.randrange(-60, 60) for _ in range(4))
        seats = [rng.randrange(len(members)) for _ in range(4)]
        state = HandState(CONFIG, seed, dealer, hand_no, scores)
        state.rounds_total = 16
        hand = fastgame.native.Hand(seed, dealer, hand_no, scores, 16)
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) == 1:
                state.step(legal[0], legal)
                hand.step(action_index(legal[0]))
                continue
            seat = state.turn
            obs = state.observation(legal)
            choice = members[seats[seat]].act(obs)
            assert hand.act(*plans[seats[seat]]) == action_index(choice)
            for lookahead in (True, False):
                row, native = encode(obs, lookahead), hand.encode(lookahead)
                if native is None:
                    fallbacks += 1
                    continue
                assert list(native) == list(row)
                for key in row:
                    assert native[key].dtype == row[key].dtype and native[key].shape == row[key].shape
                    assert native[key].tobytes() == row[key].tobytes(), key
            want, got = oracle(state, seat), hand.oracle(seat)
            for key in want:
                assert got[key].dtype == want[key].dtype and got[key].tobytes() == want[key].tobytes()
            decisions += 1
            state.step(choice, legal)
            hand.step(action_index(choice))
    assert decisions > 300 and fallbacks < decisions


def learner(row, legal):
    """A scripted learner: its choice, and a different argmax now and then (deviations)."""
    digest = int(np.frombuffer(row["planes"].tobytes(), dtype=np.uint16).sum()) + int(row["events"].sum())
    choice = legal[digest % len(legal)]
    greedy = legal[(digest // 7) % len(legal)]
    return choice, greedy


def play(game):
    """Drive a game (with its greedy continuations) as the worker does, answering with the scripted learner."""
    requests = 0
    while True:
        for shadow in game.shadows:
            while not shadow.done:
                request = shadow.advance()
                if request is None:
                    break
                legal = [i for i in np.flatnonzero(request["row"]["mask"])]
                shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
                requests += 1
        request = game.advance()
        if request is None:
            if game.settled():
                return requests
            continue
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        choice, greedy = learner(request["row"], legal)
        game.apply((choice, -0.5, 0.125, greedy))
        requests += 1


def rows_of(game):
    return [
        {k: (v.dtype.str, v.shape, v.tobytes()) if hasattr(v, "tobytes") else v for k, v in row.items()}
        for row in game.rows
    ]


@pytest.mark.parametrize("features", [None, "guided.encoding:oracle", "test_guided_fastgame:reads_events"])
def test_fast_game_plays_the_python_game(members, monkeypatch, features):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True}
    if features:
        cfg["critic_features"] = features
    for n in range(2):
        seed = seed_for(29, "fast-game", n)
        seats = ["model:current", "m1", "m3", "m5"] if n == 0 else ["m0", "m2", "model:current", "m4"]
        focal = seats.index("model:current")
        games = [cls(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed, focal=focal, train=True)
                 for cls in (Game, fastgame.FastGame)]
        assert games[1].seats is not None
        counts = [play(game) for game in games]
        assert counts[0] == counts[1] > 50
        assert games[0].scores == games[1].scores and games[0].hand_no == games[1].hand_no == 16
        assert games[0].predecision_reward == games[1].predecision_reward
        assert rows_of(games[0]) == rows_of(games[1])
        assert any(row.get("deviated") for row in games[1].rows)


# The third fixed pool's members (fp-rule-068 whole in C): styles 10 to 16.
MEMBER68 = MEMBER.replace("{style})", "{style}, True, True)")


@pytest.fixture(scope="module")
def members68(tmp_path_factory):
    folder = tmp_path_factory.mktemp("members68")
    built = []
    for k in range(7):
        q, early, late = (0.10, 0.06, 0.22) if k == 6 else (0.15, 0.10, 0.30)
        path = folder / f"member68_{k}.py"
        text = MEMBER68.format(hunt=1.0, upgrade=k != 1, style=10 + k)
        text = text.replace("Q = 0.15", f"Q = {q}").replace("0.10, 0.30,", f"{early}, {late},")
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(f"fastgame_member68_{k}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        built.append(module.Native())
    return built


def test_third_pool_members_play_the_python_game(members68, monkeypatch):
    by_id = {f"n{i}": m for i, m in enumerate(members68)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True}
    for n, seats in enumerate([["model:current", "n0", "n1", "n2"], ["n3", "n4", "model:current", "n5"],
                               ["n6", "model:current", "n0", "n3"]]):
        seed = seed_for(31, "fast-game-68", n)
        focal = seats.index("model:current")
        games = [cls(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed, focal=focal, train=True)
                 for cls in (Game, fastgame.FastGame)]
        assert games[1].seats is not None
        counts = [play(game) for game in games]
        assert counts[0] == counts[1] > 50
        assert games[0].scores == games[1].scores and games[0].hand_no == games[1].hand_no == 16
        assert rows_of(games[0]) == rows_of(games[1])


def reads_events(state, seat):
    """A critic_features hook reading the Python state beyond its plain fields."""
    features = oracle(state, seat)
    features["events_seen"] = np.float32(len(state.events))
    features["legal_seen"] = np.float32(len(state.legal_actions()))
    return features


def test_other_rule_seats_stay_on_the_python_engine(members, monkeypatch):
    class Other:
        def reset(self, seed):
            pass

        def act(self, obs):
            return obs.legal_actions[-1]

    by_id = {"m0": members[0], "other": Other()}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=False)
    game = fastgame.FastGame(cfg, {"entries": []}, {k: None for k in by_id}, ["model:current", "m0", "other", "m0"],
                             seed_for(3, "other"), focal=0, train=True)
    assert game.seats is None and isinstance(game.state, HandState)


def test_encode_into_writes_the_same_bytes_into_row_store_columns(members):
    rows, arena = 8, 4096
    planes = np.zeros((rows, 57, 34), np.float16)
    scalars = np.zeros((rows, 79), np.float16)
    mask = np.zeros((rows, 231), bool)
    events = np.zeros((arena, 5), np.int16)
    oracle_planes = np.zeros((rows, 16, 34), np.float32)
    oracle_scalars = np.zeros((rows, 4), np.float32)
    plans = [fastgame.race_member(m) for m in members]
    rng = random.Random(5)
    offset = checked = 0
    for _ in range(4):
        hand = fastgame.native.Hand(rng.getrandbits(64), rng.randrange(4), rng.randrange(1, 17), (3, -1, -2, 0), 16)
        while not hand.settled:
            legal = hand.legal()
            if len(legal) > 1:
                row, seat = rng.randrange(rows), hand.turn
                want = hand.encode(True)
                count = hand.encoded_event_count()
                if offset + count > arena:
                    offset = 0
                written = hand.encode_into(planes, scalars, events, mask, row, offset, count, True)
                if want is None:
                    assert written is None
                else:
                    assert written == count == len(want["events"])
                    assert planes[row].tobytes() == want["planes"].tobytes()
                    assert scalars[row].tobytes() == want["scalars"].tobytes()
                    assert mask[row].tobytes() == want["mask"].tobytes()
                    assert events[offset:offset + count].tobytes() == want["events"].tobytes()
                    checked += 1
                offset += count
                hand.oracle_into(oracle_planes, oracle_scalars, row, seat)
                reference = hand.oracle(seat)
                assert oracle_planes[row].tobytes() == reference["oracle_planes"].tobytes()
                assert oracle_scalars[row].tobytes() == reference["oracle_scalars"].tobytes()
            hand.step(legal[0] if len(legal) == 1 else hand.act(*plans[rng.randrange(len(plans))]))
    assert checked > 50
    with pytest.raises(ValueError):
        hand = fastgame.native.Hand(1, 0, 1, (0, 0, 0, 0), 16)
        hand.encode_into(planes, scalars, events, mask, rows, 0, 64, True)  # no such row
    with pytest.raises(ValueError):
        hand.encode_into(planes, scalars, events, mask, 0, arena - 1, 2, True)  # past the arena
    with pytest.raises(ValueError):
        hand.encode_into(planes, scalars, events, mask, 0, 0, 0, True)  # an empty history still writes one row


def test_encoding_first_in_a_fresh_worker_process():
    """A worker whose first decision is the learner's encodes before any member has run."""
    import subprocess

    code = (
        "from guided import fastgame\n"
        "row = fastgame.native.Hand(11502053608047823656, 0, 1, (0, 0, 0, 0), 16).encode(True)\n"
        "assert row['planes'][50].any()\n"  # a winning draw after some discard: the race tables' fans
    )
    import os

    paths = [str(PROJECT), str(PROJECT / "src"), str(PROJECT.parents[2] / "src"), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_a_hand_without_a_learner_decision_credits_no_earlier_row(members, monkeypatch):
    """A later hand that ends before the learner's first choice keeps its score out
    of the previous hand's last row: rows carry their own hand's score (the
    recipes build per-hand targets); the score stays in the game ledger."""
    from selfplay_core.actions import HU

    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    seed = 14590  # a naturally shuffled deal whose dealer (seat 1) can win on its opening draw
    assert HU in fastgame.native.Hand(seed, 1, 2, (0, 0, 0, 0), 2).legal()
    cfg = configuration(rounds=2, lookahead=False)
    for cls in (Game, fastgame.FastGame):
        game = cls(cfg, {"entries": []}, {k: None for k in by_id}, ["model:current", "m1", "m2", "m3"], 1,
                   focal=0, train=True)
        game.rows = [dict(hand=1, reward=-8.0, value=0.0, adv=0.0, ret=0.0)]
        game.scores, game.hand_no, game.dealer = [-8, 24, -8, -8], 2, 1
        if cls is Game:
            game.state = HandState(game.config, seed, 1, 2, game.scores)
            game.state.rounds_total = 2
            game.state.step(next(a for a in game.state.legal_actions() if a.action == "hu"))
        else:
            game.start = (1, tuple(game.scores))
            game.state = fastgame.native.Hand(seed, 1, 2, game.scores, 2)
            game.state.step(HU)
        assert game.advance() is None and game.complete
        assert game.rows[-1]["reward"] == -8.0, cls.__name__
        assert game.predecision_reward == game.scores[0] + 8 == -8, cls.__name__


def test_a_model_teacher_keeps_the_c_game_and_labels_from_the_answers(members, monkeypatch):
    """A model teacher labels in the collection parent: the game runs no teacher,
    stays on the C engine, and takes each label from the learner's answer."""
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=False)
    seats = ["m0", "model:current", "m2", "m4"]
    game = fastgame.FastGame(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed_for(31, "model-teacher"),
                             focal=1, teacher="model:teacher:frozen", train=True)
    assert game.teacher is None and game.seats is not None
    labels = []
    while (request := game.advance()) is not None:
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        choice, greedy = learner(request["row"], legal)
        labels.append(legal[-1])
        game.apply((choice, -0.5, 0.0, greedy, legal[-1]))
    assert game.complete and len(labels) == len(game.rows) > 50
    assert [row["teacher"] for row in game.rows] == labels
    assert all(row["label_version"] == "model:teacher:frozen" for row in game.rows)
    # A rule teacher acts in the game, which then stays on the Python engine.
    ruled = fastgame.FastGame(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed_for(31, "model-teacher"),
                              focal=1, teacher="m1", train=True)
    assert ruled.teacher is by_id["m1"] and ruled.seats is None


# guided.walls: the stable wall and the learner's draw luck, on both engines.
from guided.walls import train_hand  # noqa: E402


def test_a_stable_c_hand_with_luck_is_the_python_hand_at_every_step():
    rng = random.Random(23)
    for case in range(60):
        seed, dealer, hand_no = rng.getrandbits(64), rng.randrange(4), rng.randrange(1, 17)
        scores = tuple(rng.randrange(-50, 50) for _ in range(4))
        state = HandState(CONFIG, seed, dealer, hand_no, scores)
        state.rounds_total = 16
        hand = fastgame.native.Hand(seed, dealer, hand_no, scores, 16)
        learner = rng.randrange(4)
        # Converted at the deal, or partway through the hand (as an opening is).
        convert_at = 0 if case % 2 == 0 else rng.randrange(1, 30)
        steps = 0
        while True:
            if steps == convert_at:
                train_hand(state, stable=True, luck_seat=learner)
                hand.stabilize()
                hand.track_luck(learner)
            got = hand.snapshot()
            got.pop("log")
            assert got == snapshot_of(state)
            if steps >= convert_at:
                assert hand.luck == state.luck
            if state.settled:
                break
            legal = state.legal_actions()
            assert hand.legal() == tuple(action_index(a) for a in legal)
            # Calls are favoured, so that branches of a hand would shift the ordinary wall.
            calls = [i for i, a in enumerate(legal) if a.action in ("chi", "peng", "gang")]
            choice = rng.choice(calls) if calls and rng.random() < 0.7 else rng.randrange(len(legal))
            state.step(legal[choice], legal)
            hand.step(action_index(legal[choice]))
            steps += 1


def test_without_calls_a_stable_wall_draws_the_ordinary_walls_tiles():
    rng = random.Random(5)
    for _ in range(20):
        seed, dealer = rng.getrandbits(64), rng.randrange(4)
        plain = HandState(CONFIG, seed, dealer, 1, (0, 0, 0, 0))
        stable = train_hand(HandState(CONFIG, seed, dealer, 1, (0, 0, 0, 0)), stable=True)
        while not plain.settled:
            legal = plain.legal_actions()
            quiet = [a for a in legal if a.action not in ("chi", "peng", "gang", "hu")] or legal
            move = quiet[rng.randrange(len(quiet))]
            plain.step(move, legal)
            stable.step(move, stable.legal_actions())
        assert [e for e, _ in plain.events] == [e for e, _ in stable.events]


def test_the_draw_luck_has_mean_zero():
    rng = random.Random(41)
    totals = []
    for _ in range(3000):
        hand = fastgame.native.Hand(rng.getrandbits(64), rng.randrange(4), 1, (0, 0, 0, 0), 16)
        hand.stabilize()
        hand.track_luck(rng.randrange(4))
        while not hand.settled:
            legal = hand.legal()
            hand.step(legal[rng.randrange(len(legal))] if rng.random() < 0.5 else legal[0])
        totals.append(hand.luck)
    for k in range(2):
        values = np.array([t[k] for t in totals])
        assert values.std() > 0
        assert abs(values.mean()) < 4 * values.std() / np.sqrt(len(values))


@pytest.mark.parametrize("stable", [False, True])
def test_a_fast_game_with_walls_and_luck_plays_the_python_game(members, monkeypatch, stable):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "stable_wall": stable, "luck": True,
        "critic_features": "guided.encoding:oracle"}
    seed = seed_for(31, "fast-game-walls")
    seats = ["m0", "model:current", "m2", "m5"]
    games = [cls(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed, focal=1, train=True)
             for cls in (Game, fastgame.FastGame)]
    assert games[1].seats is not None
    counts = [play(game) for game in games]
    assert counts[0] == counts[1] > 50
    assert games[0].scores == games[1].scores
    assert rows_of(games[0]) == rows_of(games[1])
    rows = games[1].rows
    assert all("luck_pay" in row for row in rows)
    deviated = [row for row in rows if row.get("deviated")]
    assert deviated and all("playout_luck_pay" in row for row in deviated)
    assert any("end_luck_pay" in row for row in rows)


def test_without_the_keys_rows_carry_no_luck(members, monkeypatch):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True}
    game = fastgame.FastGame(cfg, {"entries": []}, {k: None for k in by_id}, ["model:current", "m1", "m2", "m3"],
                             seed_for(3, "no-luck"), focal=0, train=True)
    play(game)
    assert not any(key.startswith(("luck", "end_luck", "playout_luck")) for row in game.rows for key in row)
