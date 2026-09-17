"""Production v34 counterexamples and strategy-observation boundary regressions."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.rules import WHITE
from mahjong_ai.domain.types import Action
from mahjong_ai.execution.candidate_worker import observation as decode_observation
from mahjong_ai.platform.official import OfficialAdapter
from mahjong_ai.platform.replay_audit import audit_replay
from mahjong_ai.sdk.simulation import sample_world
from mahjong_ai.strategies.common import state_features
from test_catch_play import circle_state, official_document
from test_kong_decisions import kong_state


FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "official-v34-parity.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_real_official_round_replays_through_engine(case, monkeypatch):
    calls = []
    original = HandState.step

    def step(self, action):
        calls.append(action)
        return original(self, action)

    monkeypatch.setattr(HandState, "step", step)
    report = audit_replay(case["replay"])
    assert calls, "The audit must execute the real transition implementation"
    assert report["status"] == "passed", report


def test_actual_server_accepted_post_peng_kong_is_in_projected_actions():
    document = json.loads(
        (FIXTURES / "official-v34-post-peng-kong-snapshot.json").read_text()
    )
    obs = OfficialAdapter()(document)
    assert document["accepted"] and obs.drawn_tile is None
    assert any(
        a.as_dict() | {"kind": ""} == document["action"] | {"kind": ""}
        for a in obs.legal_actions
    )
    assert Action("hu") not in obs.legal_actions


def test_official_audit_rejects_tampered_score_even_when_fan_is_valid():
    replay = deepcopy(CASES[0]["replay"])
    for block in replay["blocks"]:
        for event in block["events"]:
            if event["type"] == "round_ended":
                event["data"]["scores"] = [123456, 0, 0, 0]
    assert audit_replay(replay)["status"] == "failed"


def test_sampling_keeps_nondefault_base_across_wire_and_terminal_payment():
    from dataclasses import asdict

    state, action = kong_state("concealed")
    state.config = replace(state.config, base_score=7)
    state.step(action)
    obs = decode_observation(json.loads(json.dumps(asdict(state.observation()))))
    world = sample_world(obs, 71)
    state.step(Action("hu"))
    world.step(Action("hu"))
    assert world.delta == state.delta == (672, -224, -224, -224)


def test_sampled_opponent_does_not_see_observers_concealed_kong():
    state, action = kong_state("concealed")
    state.step(action)
    world = sample_world(state.observation(), 72)
    state.step(Action("discard", WHITE))
    world.step(Action("discard", WHITE))
    assert state.turn == world.turn == 1
    for obs in (state.observation(), world.observation()):
        gang = next(e for e in obs.public_events if e.kind == "gang")
        assert gang.tile is None and gang.tiles == ()


def test_platform_reconstructs_meld_origin_from_public_claim_history():
    state, target = circle_state()
    state.step(Action("peng", target))
    doc = official_document(state)
    obs = OfficialAdapter()(doc)
    assert obs.melds == state.observation().melds


def test_platform_and_engine_provide_same_events_and_features():
    state, _ = circle_state(claim="none")
    doc = official_document(state)
    doc["wire_events"].append({"type": "pass", "seat": 2, "data": None, "_round_no": 1})
    doc["wire_events"].append(
        {"type": "timeout", "seat": 2, "data": None, "_round_no": 1}
    )
    projected = OfficialAdapter()(doc)
    obs = state.observation()
    assert projected.public_events == obs.public_events
    np.testing.assert_array_equal(state_features(projected), state_features(obs))


def test_post_call_kongs_still_respect_wall_and_white_restrictions():
    state, target = circle_state()
    state.step(Action("peng", target))
    # Isolate legal-action boundaries; this synthetic hand is not a conservation test.
    state.hands[0].append(target)
    state.hands[0].extend([WHITE] * 4)
    assert Action("gang", target, kind="added") in state.legal_actions()
    assert not any(
        a.action == "gang" and a.tile == WHITE for a in state.legal_actions()
    )
    state.wall = state.wall[:20]
    assert not any(a.action == "gang" for a in state.legal_actions())
