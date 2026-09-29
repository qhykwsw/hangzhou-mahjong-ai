import json
import importlib.util
from pathlib import Path

from mahjong_ai.persistence.artifacts import atomic_json, digest, read_json


def module():
    spec = importlib.util.spec_from_file_location(
        "free_audit", Path(__file__).parents[1] / "scripts/audit_free_play_run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_completed_replay_accounts_for_only_our_timeouts(tmp_path, monkeypatch):
    mod = module()
    replay = read_json(
        Path(__file__).parent / "fixtures/official-v25-two-hand-replay.json"
    )
    for b in replay["blocks"]:
        b["events"] = [e for e in b["events"] if e["type"] != "timeout"]
    replay["blocks"][0]["events"].append(
        {"seq": -1, "type": "timeout", "seat": 1, "data": {"kind": "discard"}}
    )
    gid = replay["game_id"]
    atomic_json(
        tmp_path / "match.json",
        {"room_id": "fixture", "match": {"config": {"M": 1, "Rounds": 2}}},
    )
    atomic_json(tmp_path / "summary.json", {"status": "completed"})
    path = tmp_path / "player" / ("game-" + digest(gid)[:24] + ".json")
    atomic_json(path, {"final": {"snapshot": {"seat": 0, "scores": [-8, -1, -1, 10]}}})
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        mod,
        "_request",
        lambda base, path, **kw: (
            replay
            if path.endswith("/events")
            else [{"batch": 0, "game_id": gid, "status": "finished"}]
        ),
    )
    result = mod.audit(tmp_path, "unused")
    assert result["status"] == "passed"
    assert result["other_discard_timeouts"] == 1 and result["own_discard_timeouts"] == 0
    cached = tmp_path / "replays" / (digest(gid)[:24] + ".json")
    replay["blocks"][0]["events"][-1]["seat"] = 0
    atomic_json(cached, replay)
    result = mod.audit(tmp_path, "unused")
    assert (
        result["status"] == "issues_or_incomplete"
        and result["own_discard_timeouts"] == 1
    )
    assert not result["feedback_to_iteration"]


def test_a_claim_lost_to_its_window_fails_the_audit(tmp_path, monkeypatch):
    """Our peng refused because its window closed (our own response timeout
    follows): explained, but the claim was never played, so not a clean run."""
    mod = module()
    replay = read_json(
        Path(__file__).parent / "fixtures/official-v25-two-hand-replay.json"
    )
    for b in replay["blocks"]:
        b["events"] = [e for e in b["events"] if e["type"] != "timeout"]
    # Seat 1 discards 1w at seq 8; our peng there lands after the window.
    replay["blocks"][0]["events"].append(
        {"seq": 8.5, "type": "timeout", "seat": 0, "data": {"kind": "response", "window": "peng"}}
    )
    gid = replay["game_id"]
    atomic_json(
        tmp_path / "match.json",
        {"room_id": "fixture", "match": {"config": {"M": 1, "Rounds": 2}}},
    )
    atomic_json(tmp_path / "summary.json", {"status": "completed"})
    path = tmp_path / "player" / ("game-" + digest(gid)[:24] + ".json")
    atomic_json(path, {"final": {"snapshot": {"seat": 0, "scores": [-8, -1, -1, 10]}}})
    path.with_suffix(".jsonl").write_text(json.dumps(
        {"game_id": gid, "seq": 8, "action": {"action": "peng", "tile": "1w"}, "accepted": False,
         "snapshot": {"last_discard": "1w"}}) + "\n")
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        mod,
        "_request",
        lambda base, path, **kw: (
            replay
            if path.endswith("/events")
            else [{"batch": 0, "game_id": gid, "status": "finished"}]
        ),
    )
    result = mod.audit(tmp_path, "unused")
    assert result["failure_counts"] == {"claim_window_expired": 1}
    assert result["own_response_timeouts"] == 1
    assert result["status"] == "issues_or_incomplete"
