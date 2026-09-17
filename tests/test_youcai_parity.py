"""Negative official action evidence, including the C and simulation paths."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mahjong_ai.domain.rules import calculate_fan, counts, tile_id
from mahjong_ai.domain.types import Action
from mahjong_ai.domain.engine import HandState
from mahjong_ai.platform.official import OfficialAdapter
from mahjong_ai.platform.replay_audit import audit_replay
from mahjong_ai.sdk.simulation import sample_world
from test_kong_decisions import KINDS, kong_state, official_document


CASES = json.loads(
    (
        Path(__file__).parent / "fixtures/official-v34-youcai-rejected-hu.json"
    ).read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_official_rejected_hu_is_absent_and_timeout_discards(case):
    assert case["response"] == {
        "accepted": False,
        "http_status": 409,
        "error_code": "INVALID_ACTION",
    }
    obs = OfficialAdapter()(case["document"])
    assert obs.hu_fan == 0
    assert Action("hu") not in obs.legal_actions
    world = sample_world(obs, 778)
    assert world.kong_draw
    assert world.current_fan() is None
    assert world.timeout_action() == Action(
        "discard", tile_id(case["expected_discard"])
    )
    with pytest.raises(ValueError):
        world.step(Action("hu"))
    world.step(world.timeout_action())
    world.assert_conservation()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
@pytest.mark.parametrize("enabled", [False, True])
def test_c_and_python_fan_on_official_rejected_shapes(case, enabled):
    fast = pytest.importorskip("mahjong_ai._fast")
    snap = case["document"]["snapshot"]
    hand = [tile_id(t) for t in snap["my_hand"]]
    drawn = tile_id(snap["drawn_tile"])
    args = (drawn, len(snap["melds"][snap["seat"]]), 1, 0, enabled, True)
    fan = calculate_fan(hand, *args)
    expected = 0 if enabled else 2
    assert (fan.fan if fan else 0) == expected
    assert fast.fan_value(counts(hand), *args) == expected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("baotou,draw", [(True, "白"), (False, "白"), (False, "西")])
def test_kong_gate_matrix_keeps_baotou_and_no_white_wins(kind, enabled, baotou, draw):
    # Synthetic coverage: the captured rejection cases cover ming/bu, not an.
    state, action = kong_state(kind, draw, baotou)
    state.step(action)
    document = official_document(state, kind)
    state.config = replace(state.config, you_cai_bi_kao=enabled)
    document["tournament"]["config"]["YouCaiBiKao"] = enabled
    obs = OfficialAdapter()(document)
    expected = not enabled or baotou or draw != "白"
    assert (Action("hu") in state.legal_actions()) is expected
    assert obs.legal_actions == state.legal_actions()
    world = sample_world(obs, 779)
    assert world.timeout_action() == state.timeout_action()
    for current in (state, world):
        current.step(current.timeout_action())
        assert current.settled is expected
        current.assert_conservation()


def test_replay_auditor_catches_extra_local_hu_at_official_timeout(monkeypatch):
    case = json.loads(
        (
            Path(__file__).parent / "fixtures/official-v34-youcai-timeout-round.json"
        ).read_text()
    )
    args = (case["replay"], case["base_score"], case["you_cai_bi_kao"])
    report = audit_replay(*args)
    assert report["status"] == "passed", report
    assert report["timeout_checks"] > 0
    original = HandState.timeout_action

    def old_fallback(state):
        if state.kong_draw and state.config.you_cai_bi_kao:
            fan = calculate_fan(
                state.hands[state.turn],
                state.drawn,
                len(state.melds[state.turn]),
                state.chain[state.turn],
            )
            if fan and not fan.baotou and 33 in state.hands[state.turn]:
                return Action("hu")
        return original(state)

    monkeypatch.setattr(HandState, "timeout_action", old_fallback)
    report = audit_replay(*args)
    assert report["status"] == "failed"
    assert any(
        f["check"] == "official discard timeout matches local fallback"
        for f in report["failures"]
    )
