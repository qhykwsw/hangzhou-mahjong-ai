import importlib.util
from pathlib import Path

from mahjong_ai.persistence.artifacts import atomic_json, read_json


def test_actual_multihand_score_deltas_sum_to_final_not_difference_of_deltas(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "platform_audit_script",
        Path(__file__).parents[1] / "scripts/audit_platform_run.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = read_json(
        Path(__file__).parent / "fixtures/official-v25-eight-hand-replay.json"
    )
    atomic_json(
        tmp_path / "workflow.json",
        {
            "protocol": {
                "config": {"M": 1, "Rounds": 8, "BaseScore": 1, "YouCaiBiKao": False}
            }
        },
    )
    atomic_json(tmp_path / "replays" / "game.json", replay)
    final = next(
        e["data"]["final_scores"]
        for b in replay["blocks"]
        for e in b["events"]
        if e["type"] == "game_ended"
    )
    for i in range(4):
        atomic_json(
            tmp_path / f"player-{i}" / "game-fixture.json",
            {"status": "finished", "final": {"snapshot": {"seat": i, "scores": final}}},
        )
    result = module.audit(tmp_path)
    assert result["status"] == "passed"
    assert result["games"][0]["final_score_consistent"]
    assert all(c["ok"] for c in result["games"][0]["score_checks"])


def test_delayed_pass_requires_own_timeout_and_accepted_pass_evidence():
    spec = importlib.util.spec_from_file_location(
        "platform_audit_script",
        Path(__file__).parents[1] / "scripts/audit_platform_run.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snap = {"phase": "response_chi", "seat": 2, "round_no": 2, "last_discard": "发"}
    row = {
        "game_id": "fixture",
        "seq": 313,
        "action": {"action": "pass"},
        "snapshot": snap,
    }
    previous = dict(
        row, seq=309, accepted=True, snapshot=dict(snap, phase="response_peng")
    )
    events = [
        {"seq": 310, "type": "timeout", "seat": 2, "data": {"window": "peng"}},
        {"seq": 313, "type": "pass", "seat": 2},
    ]
    assert module.crossed_pass_window(row, [previous], events) == [310, 313]
    assert module.crossed_pass_window(row, [previous], events[1:]) == []
    assert module.crossed_pass_window(row, [], events) == []
    # v34 also rejects a legal chi after the delayed peng-pass consumed chi.
    chi = dict(row, action={"action": "chi", "tile": "9w", "tiles": ["7w", "8w"]})
    assert module.crossed_pass_window(chi, [previous], events) == [310, 313]
    assert module.crossed_pass_window(chi, [previous], events[1:]) == []
    assert module.crossed_pass_window(chi, [], events) == []
    assert (
        module.crossed_pass_window(
            dict(row, action={"action": "hu"}), [previous], events
        )
        == []
    )
