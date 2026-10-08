"""A model seat that switches models by the stage's score (model_seat.Switch)."""
from dataclasses import replace
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from mahjong_ai.domain.types import StageContext, StageKind
from mahjong_ai.execution import model_seat
from mahjong_ai.strategies.common import PolicySpec

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
MAIN, ALT = "a" * 64, "b" * 64
FINAL, SEMI, QUALIFIER = StageKind.FINAL, StageKind.ADVANCEMENT, StageKind.QUALIFIER


def seen(kind, scores, current=150, total=160, seat=0, base=1):
    """An observation as a switch reads it: the stage (the engine's stage fields) and the base score."""
    ctx = StageContext(kind, seat, (0, 0, 0, 0), 1, 16, 0, stage_scores=scores,
                       stage_round_no=current, stage_rounds_total=total)
    return SimpleNamespace(stage=ctx, base_score=base)


def fires(rule, obs, **parameters):
    return model_seat.Switch(dict(rule=rule, weights=ALT, **parameters)).fires(obs)


@pytest.fixture
def selfplay(monkeypatch):
    """The self-play project on the path, as a match worker has it (MAHJONG_SELFPLAY)."""
    pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")
    monkeypatch.setenv(model_seat.ENCODER_ENV, str(PROJECT))
    monkeypatch.setattr(model_seat, "_ENCODER", [])
    assert model_seat.encoder()
    from guided import stage_value

    return stage_value


def test_outside_a_semifinal_or_a_final_every_rule_plays_the_own_model(selfplay):
    rules = [("always", {}), ("tail", {"hands_left": 1000}), ("beta", {"threshold": 0}),
             ("contest", {"hands_left": 1000, "margin": 1e9})]
    for rule, parameters in rules:
        assert not fires(rule, seen(QUALIFIER, (10, 0, 0, -10), current=160), **parameters)
        assert not fires(rule, seen(FINAL, None, current=160), **parameters)  # no stage totals
        assert not fires(rule, seen(SEMI, None, current=160), **parameters)
    assert fires("always", seen(FINAL, (0, 0, 0, 0), current=1)) and fires("always", seen(SEMI, (0, 0, 0, 0), current=1))


def test_tail_counts_the_hands_left_with_this_one():
    assert fires("tail", seen(FINAL, (0,) * 4, current=157))  # 157..160: four left
    assert not fires("tail", seen(FINAL, (0,) * 4, current=156))
    assert fires("tail", seen(SEMI, (0,) * 4, current=160), hands_left=1)
    assert not fires("tail", seen(SEMI, (0,) * 4, current=159), hands_left=1)
    # A final's tiebreak game extends the horizon by a game.
    assert not fires("tail", seen(FINAL, (5, 5, 0, 0), current=161, total=176))
    assert fires("tail", seen(FINAL, (5, 5, 0, 0), current=173, total=176))


@pytest.mark.parametrize("kind, scores, seat, base, gap", [
    # A final: the line is the best other total (the leader's, or the second's for the leader).
    (FINAL, (100, 60, 50, 40), 0, 1, 40),       # leading
    (FINAL, (101, 60, 50, 40), 0, 1, 41),
    (FINAL, (60, 100, 50, 40), 0, 1, -40),      # behind the leader
    (FINAL, (59, 100, 99, 40), 0, 1, -41),      # the second's total does not count
    (FINAL, (70, 70, 0, 0), 1, 1, 0),           # tied for the lead
    (FINAL, (100, 0, 60, 50), 2, 1, -40),
    (FINAL, (200, 120, 100, 80), 0, 2, 40),     # in base scores
    (FINAL, (202, 120, 100, 80), 0, 2, 41),
    # A semifinal: the line is the second best other total (between second and third place).
    (SEMI, (100, 120, 60, 10), 0, 1, 40),       # second, above third
    (SEMI, (100, 120, 59, 10), 0, 1, 41),
    (SEMI, (160, 120, 100, 10), 0, 1, 60),      # first, above third
    (SEMI, (140, 120, 100, 10), 0, 1, 40),
    (SEMI, (60, 120, 100, 10), 0, 1, -40),      # third, below second
    (SEMI, (59, 120, 100, 10), 0, 1, -41),
    (SEMI, (0, 120, 100, 80), 0, 1, -100),      # fourth
    (SEMI, (120, 100, 10, 60), 3, 1, -40),      # seat 3, third
    (SEMI, (240, 200, 20, 120), 3, 2, -40),
])
def test_contest_measures_the_gap_to_the_line_in_base_scores(kind, scores, seat, base, gap):
    six, seven = (seen(kind, scores, current=current, seat=seat, base=base) for current in (155, 154))
    assert fires("contest", six) is (abs(gap) <= 40) and not fires("contest", seven)
    for margin in {abs(gap), max(0, abs(gap) - 1), 10}:
        assert fires("contest", six, margin=margin) is (abs(gap) <= margin)
        assert fires("contest", seven, hands_left=7, margin=margin) is (abs(gap) <= margin)
        assert not fires("contest", seven, margin=margin)


def test_beta_is_the_v4_encodings_coefficient(selfplay):
    """|beta| >= threshold, beta from guided.stage_value as the v4 encoding computes it."""
    stage_value = selfplay
    rng = random.Random(7)
    checked = 0
    for _ in range(200):
        kind = rng.choice((FINAL, SEMI))
        seat, current, base = rng.randrange(4), rng.randrange(100, 161), rng.choice((1, 2))
        scores = tuple(base * rng.randrange(-150, 150) for _ in range(4))
        obs = seen(kind, scores, current=current, seat=seat, base=base)
        want = float(stage_value.beta(tuple(s / base for s in scores), seat, 160 - current + 1,
                                      stage_value.PLACES[kind]))
        assert stage_value.risk_of(obs.stage, base) * stage_value.BETA_MAX == pytest.approx(want)
        size = abs(want)
        if size:
            checked += 1
            assert fires("beta", obs, threshold=size) and not fires("beta", obs, threshold=math.nextafter(size, 1))
        assert fires("beta", obs) is (size >= 0.01)
    assert checked > 150
    # A stage as good as decided has no coefficient: the own model plays at any threshold.
    assert not fires("beta", seen(FINAL, (500, 0, 0, 0), current=160), threshold=1e-12)


def test_beta_needs_the_self_play_project(monkeypatch):
    monkeypatch.delenv(model_seat.ENCODER_ENV, raising=False)
    monkeypatch.setattr(model_seat, "_ENCODER", [])
    with pytest.raises(RuntimeError):
        model_seat.Switch({"rule": "beta", "weights": ALT})
    model_seat.Switch({"rule": "contest", "weights": ALT})


@pytest.mark.parametrize("switch", [
    "always", {"weights": ALT}, {"rule": "sometimes", "weights": ALT}, {"rule": "always"},
    {"rule": "always", "weights": ALT.upper()}, {"rule": "always", "weights": ALT[:63]},
    {"rule": "tail", "weights": ALT, "margin": 40}, {"rule": "always", "weights": ALT, "hands_left": 4},
    {"rule": "tail", "weights": ALT, "hands_left": 0}, {"rule": "tail", "weights": ALT, "hands_left": 4.0},
    {"rule": "tail", "weights": ALT, "hands_left": True}, {"rule": "contest", "weights": ALT, "margin": -1},
    {"rule": "beta", "weights": ALT, "threshold": float("nan")}, {"rule": "beta", "weights": ALT, "threshold": "0.01"},
])
def test_a_malformed_switch_is_refused(switch):
    with pytest.raises(ValueError):
        PolicySpec(family="model", version="v", parameters={"weights": MAIN, "switch": switch})


def test_a_switch_is_part_of_the_policy_and_its_served_weights():
    plain = PolicySpec(family="model", version="fp-rl-227", parameters={
        "weights": "4e9023dc07fa97b86825ce2b1a00192ab290f4f9e874d67b4a6639fc8be2ca25"})
    # The identity of a model policy without a switch is what it was before switching existed.
    assert plain.identity == "4b05f05cea56cb065595638a6f65c45b58c659f1759e114ecbc63955f8d52b49"
    with pytest.raises(ValueError):
        PolicySpec(family="model", version="v", parameters={"weights": MAIN, "other": 1})
    spec = {"family": "model", "version": "s", "parameters": {
        "weights": MAIN, "switch": {"rule": "contest", "weights": ALT, "hands_left": 6, "margin": 40}}}
    switching = PolicySpec(**spec)
    assert switching.identity != PolicySpec(**dict(spec, parameters={"weights": MAIN})).identity
    assert model_seat.served_weights(spec) == {MAIN, ALT}
    assert model_seat.served_weights(plain.as_dict()) == {plain.parameters["weights"]}
    from mahjong_ai.execution.remote import model_weights

    task = {"policies": {"s": spec, "p": plain.as_dict(), "r": {"family": "code", "parameters": {}}}}
    assert model_weights(task) == {MAIN, ALT, plain.parameters["weights"]}


def test_the_job_readout_counts_a_switching_seats_decisions_only():
    from mahjong_ai.evaluation.schedule import standings

    receipt = {"players": ["s", "p", "x", "y"], "ranks": [1, 2, 3, 4], "scores": [3, 1, -1, -3],
               "execution": [{"actions": 10, "alt_actions": 4, "errors": 0}, {"actions": 12, "errors": 1}, {}, {}]}
    rows = {r["candidate_id"]: r for r in standings(["s", "p"], [receipt, receipt])}
    assert rows["s"]["actions"] == 20 and rows["s"]["alt_actions"] == 8
    assert "actions" not in rows["p"] and "alt_actions" not in rows["p"] and rows["p"]["errors"] == 2


def observations(hands=6, seed=20261008):
    """Decision observations of random play, each also inside a semifinal and a final near their end."""
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig

    rng, out = random.Random(seed), []
    for i in range(hands):
        state = HandState(RoundConfig(rounds=16), seed + i, i % 4, 1 + i, [0] * 4, [], False)
        state.rounds_total = 16
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) > 1 and rng.random() < 0.3:
                obs = state.observation(legal)
                scores = tuple(rng.randrange(-60, 60) for _ in range(4))
                out.append(obs)
                for kind in (SEMI, FINAL):
                    out.append(replace(obs, stage=replace(obs.stage, kind=kind, stage_scores=scores,
                                                          stage_round_no=155 + i, stage_rounds_total=160)))
            state.step(legal[rng.randrange(len(legal))], legal)
    return out


class Service:
    """The inference service as a seat sees it: every request's model and body; the first legal action back."""

    def __init__(self):
        self.requests = []

    def client(self, host, port, timeout):
        from selfplay_core import wire

        service = self

        class Client:
            def decide(self, body):
                service.requests.append((body[4:68].decode("ascii"), bytes(body), body.risk))
                return {"index": int(np.flatnonzero(wire.unpack(bytes(body))["encoded"]["mask"])[0])}

            def close(self):
                pass

        return Client()


@pytest.fixture
def service(selfplay, monkeypatch):
    from selfplay_core import wire

    served = Service()
    monkeypatch.setattr(wire, "SocketClient", served.client)
    monkeypatch.setenv(model_seat.SERVER_ENV, "http://127.0.0.1:9")
    return served


def test_without_a_switch_a_seat_sends_and_counts_what_it_did_before(service):
    from guided.encoding import encode
    from selfplay_core.wire import pack

    sample = observations()
    for policy in (model_seat.ModelPolicy(MAIN, "v"),
                   PolicySpec(family="model", version="v", parameters={"weights": MAIN}).build()):
        service.requests.clear()
        for obs in sample:
            assert policy.act(obs) in obs.legal_actions
        sent = [pack(MAIN, encode(obs, lookahead=True), lookahead=True) for obs in sample]
        assert service.requests == [(MAIN, bytes(b), b.risk) for b in sent]
        assert sum(bool(b.risk) for b in sent) == 2 * len(sample) // 3  # the staged rows carry beta beside the body
        assert list(policy.stats) == ["actions", "timeouts", "errors", "elapsed_seconds", "compute_seconds",
                                      "ipc_seconds", "fallback_actions"]
        assert policy.stats["actions"] == len(sample) and policy.stats["errors"] == policy.stats["fallback_actions"] == 0
        assert policy.switch is None


@pytest.mark.parametrize("switch", [{"rule": "always"}, {"rule": "tail", "hands_left": 4},
                                    {"rule": "beta", "threshold": 0.005}, {"rule": "contest"}])
def test_a_switching_seat_asks_the_second_model_where_its_rule_fires(service, switch):
    from guided.encoding import encode
    from selfplay_core.wire import pack

    sample = observations()
    rule = model_seat.Switch(dict(switch, weights=ALT))
    policy = PolicySpec(family="model", version="v", parameters={"weights": MAIN, "switch": dict(switch, weights=ALT)}).build()
    for obs in sample:
        assert policy.act(obs) in obs.legal_actions
    models = [ALT if rule.fires(obs) else MAIN for obs in sample]
    assert [r[0] for r in service.requests] == models
    # The request is the one-model seat's, addressed to the model the rule chose.
    assert [r[1] for r in service.requests] == [bytes(pack(m, encode(obs), lookahead=True)) for m, obs in zip(models, sample)]
    assert policy.stats["alt_actions"] == models.count(ALT) and policy.stats["actions"] == len(sample)
    assert 0 < models.count(ALT) < len(sample) or switch["rule"] == "always"
    assert models.count(ALT) == 2 * len(sample) // 3 or switch["rule"] != "always"
