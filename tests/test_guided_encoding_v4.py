"""Encoding v4 is v3 and the stage's risk coefficient: both engines write the same rows, free play holds 0,
match workers' requests carry it to a v4 model only, and an extended Actor keeps every decision."""

import base64
import math
from pathlib import Path
import random
import struct
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame, stage, stage_value  # noqa: E402
from guided.config import configuration  # noqa: E402
from guided.encoding import (  # noqa: E402
    ENCODING,
    ENCODING_V2,
    ENCODING_V3,
    ENCODING_V4,
    EVENT_TAILS,
    INPUTS,
    RISK,
    augment,
    encode,
    encode_v3,
    encode_v4,
    encoder,
    widest,
)
from guided.environment import Game  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from mahjong_ai.domain.types import StageKind  # noqa: E402
from test_guided_fastgame import learner, members, rows_of, rule_table  # noqa: E402,F401
from test_guided_stage import STAGE, play_all, staged_seed  # noqa: E402


def observations(hands=8, seed=20261007, keep=0.3):
    """Decision observations of random play: (plain, in a semifinal, in a final, in a qualifier) at drawn stage states."""
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig

    rng, out = random.Random(seed), []
    for i in range(hands):
        hand_no = 1 + i % 16
        state = HandState(RoundConfig(rounds=16), seed + i, i % 4, hand_no, [0] * 4, [], False)
        state.rounds_total = 16
        game = rng.randrange(4)
        prior = stage.zero_sum(rng, 70 * math.sqrt(game) + 40)
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) > 1 and rng.random() < keep:
                obs = state.observation(legal)
                staged = [stage.observe(obs, {"kind": kind, "games": 4, "game": game, "prior": prior}, state.scores,
                                        hand_no, 16) for kind in (StageKind.ADVANCEMENT, StageKind.FINAL,
                                                                  StageKind.QUALIFIER)]
                out.append((obs, *staged))
            state.step(legal[rng.randrange(len(legal))], legal)
    return out


@pytest.fixture(scope="module")
def sample():
    return observations()


def test_v4_is_v3_with_the_risk_coefficient_last(sample):
    assert list(INPUTS)[-2:] == [ENCODING_V3, ENCODING_V4]
    assert INPUTS[ENCODING_V4] == (INPUTS[ENCODING_V3][0], INPUTS[ENCODING_V3][1] + 1) and RISK == INPUTS[ENCODING_V3][1]
    assert EVENT_TAILS[ENCODING_V4] == EVENT_TAILS[ENCODING_V3]
    assert encoder(ENCODING_V4) is encode_v4
    assert widest([ENCODING_V2, ENCODING_V4, ENCODING, ENCODING_V3]) == ENCODING_V4
    nonzero = 0
    for plain, semifinal, final, qualifier in sample:
        for obs in (plain, semifinal, final, qualifier):
            v1, v3, v4 = encode(obs), encode_v3(obs), encode_v4(obs)
            assert v4["planes"].shape == (INPUTS[ENCODING_V4][0], 34) and v4["planes"].dtype == np.float16
            assert v4["scalars"].shape == (INPUTS[ENCODING_V4][1],) and v4["scalars"].dtype == np.float16
            assert v4["planes"].tobytes() == v3["planes"].tobytes()
            assert v4["scalars"][:RISK].tobytes() == v3["scalars"].tobytes()
            assert v4["events"].tobytes() == v3["events"].tobytes() and np.array_equal(v4["mask"], v1["mask"])
            # A v3 row (which keeps the observation's risk_beta) extends to the same v4 row.
            again = augment(v3, ENCODING_V4)
            assert again["scalars"].tobytes() == v4["scalars"].tobytes()
            assert augment(v4, ENCODING_V4) is v4 and augment(v4, ENCODING_V3) is v4 and augment(v4, ENCODING) is v4
            risk = float(v4["scalars"][RISK])
            if obs is plain or obs is qualifier:
                # No semifinal or final: no coefficient, an input of 0.
                assert "risk_beta" not in v1 and risk == 0.0
            else:
                ctx = obs.stage
                want = stage_value.risk_input(ctx.kind, ctx.stage_scores, ctx.seat, ctx.stage_round_no,
                                              ctx.stage_rounds_total)
                assert v1["risk_beta"] == want and risk == float(np.float16(want)) and -1.0 <= risk <= 1.0
                nonzero += risk != 0.0
    assert nonzero > len(sample)


def test_the_coefficient_is_the_curvature_of_the_chance_to_advance():
    """-V''/V' per base score: its sign, a two-player closed form, the line, and where it stops."""
    final, semifinal = StageKind.FINAL, StageKind.ADVANCEMENT
    cap = stage_value.BETA_MAX

    def coefficient(kind, scores, left, seat=0, total=160):
        value = stage_value.risk_input(kind, tuple(scores), seat, total - left + 1, total)
        return None if value is None else value * cap

    # With the two others far below, a final is a race against one rival:
    # V = Phi((s - r) / tau), tau^2 = 2 * 17.5^2 * n, so -V''/V' = (s - r) / tau^2.
    for gap, left in ((-100, 16), (60, 16), (-200, 40), (150, 80)):
        tau2 = 2 * 17.5 ** 2 * left
        assert coefficient(final, (gap, 0, -10000, -10000), left) == pytest.approx(gap / tau2, rel=0.01)
    # Behind in a final, variance is worth having (beta < 0); well ahead, it costs (beta > 0). A small lead
    # with much left to play is still convex: first of four is above the middle of the field.
    for left in (4, 16, 64):
        assert coefficient(final, (-60, 60, 0, 0), left) < 0 < coefficient(final, (150, -50, -50, -50), left)
    assert coefficient(final, (60, -60, 0, 0), 4) > 0 > coefficient(final, (60, -60, 0, 0), 64)
    # A semifinal's line is between second and third: third just below it wants variance, second above it does not.
    assert coefficient(semifinal, (-30, 100, 10, -80), 16) < 0 < coefficient(semifinal, (30, 100, -10, -120), 16)
    # Level with everyone in a semifinal V(s) + V(-s) = 1, so V'' = 0.
    assert abs(coefficient(semifinal, (5, 5, 5, 5), 16)) < 1e-9
    # Seat order: the coefficient is the seat's own.
    scores = (40, -120, 100, -20)
    assert coefficient(final, scores, 16, seat=1) == coefficient(final, scores[1:] + scores[:1], 16)
    # Clipped to +-BETA_MAX; 0 once the stage is decided either way.
    assert coefficient(final, (-40, 0, -300, -300), 1) == -cap and coefficient(final, (40, 0, -300, -300), 1) == cap
    assert coefficient(final, (2000, 0, 0, 0), 1) == 0.0 and coefficient(final, (-2000, 0, 0, 0), 1) == 0.0
    # The stage's last hand still has itself to play: one hand of variance, not the standings as they are.
    last = stage_value.risk_input(final, (-20, 0, -300, -300), 0, 8, 8) * cap
    assert last == pytest.approx(-20 / (2 * 17.5 ** 2), rel=0.01)
    assert stage_value.risk_input(final, (0, 0, 0, 0), 0, 9, 8) == stage_value.risk_input(final, (0, 0, 0, 0), 0, 8, 8)
    # Outside a semifinal or a final there is none.
    assert coefficient(StageKind.QUALIFIER, (-120, 120, 0, 0), 16) is None


@pytest.mark.parametrize("kind", ["advancement", "final"])
def test_both_engines_write_the_same_v4_rows_inside_a_stage(members, monkeypatch, kind):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "stable_wall": True, "luck": True, "encoding": ENCODING_V4,
        "critic_features": "guided.encoding:oracle", "stage": dict(STAGE, kind=kind)}
    seed = staged_seed(f"v4-{kind}")
    # Two model seats: the learner and another seat, whose rows hold its own coefficient.
    seats = ["m0", "model:other", "model:current", "m5"]
    games = [cls(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed, focal=2, train=True)
             for cls in (Game, fastgame.FastGame)]
    assert games[1].seats is not None
    requests = [play_all(game) for game in games]
    assert len(requests[0]) == len(requests[1]) > 50
    for key in ("scalars", "planes", "events"):
        assert [r[key].tobytes() for r in requests[0]] == [r[key].tobytes() for r in requests[1]]
    assert [r["risk_beta"] for r in requests[0]] == [r["risk_beta"] for r in requests[1]]
    assert rows_of(games[0]) == rows_of(games[1])
    assert any(r.get("deviated") for r in games[1].rows)
    risks = [float(r["scalars"][RISK]) for r in requests[1]]
    assert all(risk == float(np.float16(r["risk_beta"])) for risk, r in zip(risks, requests[1]))
    assert len(set(risks)) > 3
    # The same game outside a stage plays the same moves with a coefficient of 0.
    plain = fastgame.FastGame(dict(cfg, stage=None), {"entries": []}, {k: None for k in by_id}, seats, seed,
                              focal=2, train=True)
    plain_requests = play_all(plain)
    assert [r["planes"].tobytes() for r in plain_requests] == [r["planes"].tobytes() for r in requests[1]]
    assert all(r["scalars"][RISK] == 0 and "risk_beta" not in r for r in plain_requests)
    python = Game(dict(cfg, stage=None), {"entries": []}, {k: None for k in by_id}, seats, seed, focal=2, train=True)
    assert [r["scalars"].tobytes() for r in play_all(python)] == [r["scalars"].tobytes() for r in plain_requests]


def test_a_python_teacher_and_rule_seat_inside_a_final_on_both_engines(members, monkeypatch):
    entries, built = rule_table(members, monkeypatch, {"teach": 1, "py": 3})
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "vine": True, "encoding": ENCODING_V4, "stage": dict(STAGE, kind="final")}
    seats = ["m0", "model:current", "py", "m3"]
    games = [cls(cfg, {"entries": []}, entries, seats, staged_seed("v4-teacher"), focal=1, teacher="teach",
                 train=True) for cls in (Game, fastgame.FastGame)]
    assert games[1].mirrored and games[1].python_seats
    requests = [play_all(game) for game in games]
    assert len(requests[0]) == len(requests[1]) > 50
    assert [r["scalars"].tobytes() for r in requests[0]] == [r["scalars"].tobytes() for r in requests[1]]
    assert rows_of(games[0]) == rows_of(games[1])
    assert all(row["teacher"] >= 0 for row in games[1].rows)
    assert any(row["scalars"][RISK] != 0 for row in games[1].rows)


def test_extended_actor_keeps_every_decision(sample, tmp_path):
    torch = pytest.importorskip("torch")
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.inference import policy_batch
    from guided.networks import Actor
    from guided.trial import export_actor

    torch.manual_seed(0)
    source = Actor(32, 2, 4)
    with torch.no_grad():
        source.encoder.event_embeddings[4].weight.normal_()  # rows a v1 Actor never uses
    starts = {ENCODING: tmp_path / "v1"}
    export_actor(source.state_dict(), {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}, starts[ENCODING])
    starts[ENCODING_V2] = expand(starts[ENCODING], tmp_path / "models-v2", ENCODING_V2)
    starts[ENCODING_V3] = expand(starts[ENCODING_V2], tmp_path / "models-v3", ENCODING_V3)
    # Rows inside a stage, so the new column holds values.
    rows = [encode(obs) for _, semifinal, final, _ in sample[:120] for obs in (semifinal, final)]
    assert sum(augment(r, ENCODING_V4)["scalars"][RISK] != 0 for r in rows) > 100
    for narrow_encoding, start in starts.items():
        wide_dir = expand(start, tmp_path / f"models-v4-from-{narrow_encoding}", ENCODING_V4)
        narrow, _ = load_actor(start, "cpu")
        wide, meta = load_actor(wide_dir, "cpu")
        assert meta["encoding"] == ENCODING_V4
        assert wide.encoder.state_in.in_features == INPUTS[ENCODING_V4][1]
        assert not wide.encoder.state_in.weight[:, RISK].any()
        with torch.no_grad():
            a = policy_batch([augment(r, narrow_encoding) for r in rows], "cpu")
            b = policy_batch([augment(r, ENCODING_V4) for r in rows], "cpu")
            assert torch.equal(narrow(a["planes"], a["scalars"], a["events"]), wide(b["planes"], b["scalars"], b["events"]))
            # The narrow Actor reads a v4 row's prefixes.
            assert torch.equal(narrow(a["planes"], a["scalars"], a["events"]), narrow(b["planes"], b["scalars"], b["events"]))
    with pytest.raises(ValueError):
        expand(wide_dir, tmp_path / "again", ENCODING_V3)


def test_model_service_extends_v1_requests_with_their_coefficient(sample):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from guided.inference import policy_batch
    from guided.model_server import Service, decide
    from guided.networks import actor_of
    from selfplay_core.wire import pack, unpack

    torch.set_num_threads(1)
    torch.manual_seed(79)
    actor = actor_of(ENCODING_V4, 16, 1, 4).eval()
    with torch.no_grad():
        for p in actor.parameters():
            p.normal_()
        actor.encoder.state_in.weight[:, RISK] *= 50  # the coefficient alone moves decisions
    meta = {"config": {"lookahead": True}, "encoding": ENCODING_V4}
    service = Service(SimpleNamespace(device="cpu", get=lambda name: (actor, meta)))
    rows = [encode(obs) for plain, semifinal, final, _ in sample[:40] for obs in (plain, semifinal, final)]
    batch = policy_batch([augment(r, ENCODING_V4) for r in rows], "cpu")
    with torch.inference_mode():
        expected, valid = decide(actor, *(batch[k] for k in ("planes", "scalars", "events", "mask")))
        blind = policy_batch([augment({k: v for k, v in r.items() if k != "risk_beta"}, ENCODING_V4) for r in rows],
                             "cpu")
        without, _ = decide(actor, *(blind[k] for k in ("planes", "scalars", "events", "mask")))
    assert valid and not torch.equal(expected, without)
    # Match workers' binary requests to a model that reads it: a semifinal or final row's coefficient after the
    # plain body (SocketClient appends Body.risk), a plain row the plain body.
    bodies = [pack("0" * 64, r, lookahead=True) for r in rows]
    assert len(bodies[1].risk) == 2 and bodies[0].risk == b"" and "risk_beta" not in rows[0]
    binary = [unpack(body + body.risk) for body in bodies]
    assert [a["index"] for a in service.decide("wide", binary)] == expected.tolist()
    extended = bodies[1] + bodies[1].risk
    for bad in (extended + b"x", bodies[1] + np.float16(np.nan).tobytes(), bodies[1] + np.float16(1.5).tobytes()):
        with pytest.raises(ValueError):
            unpack(bad)
    # The JSON endpoint's base64 rows likewise.
    payloads = [{"lookahead": True, "encoded": {
        k: base64.b64encode(np.ascontiguousarray(v if k != "risk_beta" else np.float16(v)).tobytes()).decode()
        for k, v in r.items()}} for r in rows]
    assert [a["index"] for a in service.decide("wide", payloads)] == expected.tolist()


def main_body(sha, row):
    """A match worker's request as selfplay_core.wire packed it before v4."""
    arrays = [np.asarray(row[k], dtype=d) for k, d in (
        ("planes", "<f2"), ("scalars", "<f2"), ("events", "<i2"), ("mask", "?"))]
    return struct.pack("<4s64sBI", b"MJE1", sha.encode("ascii"), True, len(arrays[2])) + b"".join(
        a.tobytes() for a in arrays)


def test_only_a_v4_model_gets_the_coefficient_on_the_wire(sample, monkeypatch):
    """Match seats (mahjong_ai's ModelPolicy, unchanged) against the service: inside a stage a v1, v2 or v3
    model's requests are byte for byte those from before v4, which a service from before v4 accepts; only a v4
    model's carry the coefficient."""
    import socket
    import threading
    import time
    import urllib.request
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from guided.model_server import Service, serve
    from guided.networks import actor_of
    from mahjong_ai.execution import model_seat
    from selfplay_core import wire

    torch.set_num_threads(1)
    torch.manual_seed(80)
    shas = {encoding: str(n) * 64 for n, encoding in enumerate((ENCODING, ENCODING_V2, ENCODING_V3, ENCODING_V4), 1)}
    models = {sha: (actor_of(encoding, 16, 1, 4).eval(), {"config": {"lookahead": True}, "encoding": encoding})
              for encoding, sha in shas.items()}
    service = Service(SimpleNamespace(device="cpu", get=lambda name: models[name]))
    asked = []
    service_inputs = service.inputs
    service.inputs = lambda model: (asked.append(model), service_inputs(model))[1]
    received = []
    unpack = wire.unpack
    monkeypatch.setattr(wire, "unpack", lambda body: (received.append(bytes(body)), unpack(body))[1])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=serve, args=(service, port), daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            with urllib.request.urlopen(url + "/health", timeout=1):
                break
        except OSError:
            time.sleep(0.02)
    monkeypatch.setenv(model_seat.SERVER_ENV, url)
    monkeypatch.setenv(model_seat.ENCODER_ENV, str(PROJECT))
    monkeypatch.setattr(model_seat, "_ENCODER", [])

    def play(sha, observations):
        received.clear()
        policy = model_seat.ModelPolicy(sha, "v")
        for obs in observations:
            assert policy.act(obs) in obs.legal_actions
        policy.close()
        assert policy.stats["errors"] == policy.stats["fallback_actions"] == 0
        return list(received)

    staged = [obs for plain, semifinal, final, _ in sample[:20] for obs in (plain, semifinal, final)]
    rows = [encode(obs) for obs in staged]
    assert sum("risk_beta" in r for r in rows) == 40
    for encoding in (ENCODING, ENCODING_V2, ENCODING_V3):
        sha = shas[encoding]
        assert play(sha, staged) == [main_body(sha, r) for r in rows]
    assert asked == [shas[ENCODING], shas[ENCODING_V2], shas[ENCODING_V3]]  # once per seat, on its first staged row
    sha = shas[ENCODING_V4]
    want = [main_body(sha, r) + (np.float16(r["risk_beta"]).tobytes() if "risk_beta" in r else b"") for r in rows]
    assert play(sha, staged) == want and want != [main_body(sha, r) for r in rows]
    # Outside a stage no seat asks: the traffic is what it was before v4.
    asked.clear()
    plain = [obs for obs, *_ in sample[20:30]]
    assert play(shas[ENCODING_V3], plain) == [main_body(shas[ENCODING_V3], encode(obs)) for obs in plain] and not asked
    # A service that cannot say what a model reads (one from before v4 answers 404) gets the plain body.
    service.inputs = lambda model: (_ for _ in ()).throw(KeyError(model))
    assert play(sha, staged) == [main_body(sha, r) for r in rows]
