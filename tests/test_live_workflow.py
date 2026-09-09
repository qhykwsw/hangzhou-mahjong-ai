import json
from collections import Counter

import pytest

from mahjong_ai.platform.live_workflow import load_credentials, summarize_player
from mahjong_ai.platform.replay_audit import audit_replay
from mahjong_ai.domain.rules import calculate_fan, tile_id, tile_code
from mahjong_ai.persistence.artifacts import atomic_json


def test_credentials_accept_four_token_export_and_never_accept_duplicates(tmp_path):
    path = tmp_path / "credentials.json"
    atomic_json(
        path,
        {
            "room_id": "test-example",
            "players": [
                {"name": str(i), "token": f"fixture-only-{i}"} for i in range(4)
            ],
        },
    )
    room, tokens = load_credentials(path)
    assert room == "test-example" and len(tokens) == 4
    atomic_json(path, {"tokens": ["duplicate"] * 4})
    with pytest.raises(ValueError, match="distinct"):
        load_credentials(path)


def test_summarize_real_runner_record_format_includes_failures_and_latency(tmp_path):
    atomic_json(
        tmp_path / "game-fixture.json",
        {
            "status": "finished",
            "final": {"snapshot": {"seat": 1, "scores": [-1, 3, -1, -1]}},
        },
    )
    records = [
        {
            "game_id": "fixture",
            "seq": 1,
            "action": {"action": "pass"},
            "accepted": True,
            "decision_ms": 1,
            "request_ms": 2,
        },
        {
            "game_id": "fixture",
            "seq": 2,
            "action": {"action": "hu"},
            "accepted": False,
            "http_status": 409,
            "error_code": "INVALID_ACTION",
            "decision_ms": 2,
            "request_ms": 4,
        },
    ]
    (tmp_path / "game-fixture.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )
    out = summarize_player(tmp_path)
    assert out["score_sum"] == 3 and out["unsuccessful_actions"] == 1
    assert out["request_latency"]["p50_ms"] == 3


def winning_replay():
    # Normalized official replay-contract fixture, not a captured platform game.
    own = [tile_id(t) for t in "1w 1w 1w 2w 3w 4w 5w 6w 7w 8w 9w 东 东 东".split()]
    deck = list((Counter({i: 4 for i in range(34)}) - Counter(own)).elements())
    hands = [own, deck[:13], deck[13:26], deck[26:39]]
    fan = calculate_fan(own, own[-1]).fan
    return {
        "blocks": [
            {
                "round_no": 1,
                "seq_start": 1,
                "start_hands": [[tile_code(t) for t in h] for h in hands],
                "events": [
                    {
                        "seq": 1,
                        "type": "round_ended",
                        "seat": 0,
                        "data": {"draw": False, "dealer": 0, "fan": fan},
                    }
                ],
            }
        ]
    }


def test_replay_audit_checks_local_eligibility_fan_and_missing_blocks():
    replay = winning_replay()
    assert audit_replay(replay)["status"] == "passed"
    replay["blocks"][0]["events"][0]["data"]["fan"] *= 2
    result = audit_replay(replay)
    assert result["status"] == "failed"
    assert any(
        x["check"] == "official fan matches local calculator"
        for x in result["failures"]
    )
    replay["blocks"][0]["events"] = []
    assert audit_replay(replay)["status"] == "incomplete"


def test_room_workflow_finishes_one_readiness_wave_then_resumes_without_new_games(
    tmp_path, monkeypatch
):
    import mahjong_ai.platform.live_workflow as live
    from mahjong_ai.strategies.common import PolicySpec

    token_file = tmp_path / "tokens.json"
    atomic_json(
        token_file,
        {"room_id": "fixture-room", "tokens": [f"fixture-{i}" for i in range(4)]},
    )
    ready = []
    config = {"M": 1, "Rounds": 1, "BaseScore": 1, "YouCaiBiKao": False}

    class Client:
        def __init__(self, base, token, insecure):
            self.token = token

        def discover(self, expected):
            return expected, True, {"active_games": [], "user_id": self.token}

        def request(self, method, path, body=None):
            if path.endswith("/rules"):
                return {
                    "status": "registering" if not ready else "finished",
                    "config": config,
                }
            if path.endswith("/ready"):
                ready.append(self.token)
                return {}
            if "/state?" in path:
                return {
                    "finished": True,
                    "snapshot": {
                        "seat": int(self.token[-1]),
                        "scores": [3, -1, -1, -1],
                    },
                }
            return {"status": "finished", "my_games": ["fixture-game"]}

    class Runner:
        def __init__(self, client, policy, adapter, output, **kw):
            import threading

            self.stop = threading.Event()

        def run(self, *args):
            return {"status": "stopped"}

    monkeypatch.setattr(live, "PlatformClient", Client)
    monkeypatch.setattr(live, "TournamentRunner", Runner)
    monkeypatch.setattr(live, "SnapshotAdapter", lambda mapping: None)

    def request(base, path, **kw):
        if path.endswith("/events"):
            return winning_replay() | {"game_id": "fixture-game"}
        return (
            []
            if not ready
            else [{"game_id": "fixture-game", "batch": 0, "status": "finished"}]
        )

    monkeypatch.setattr(live, "_request", request)
    args = (
        token_file,
        [PolicySpec()] * 4,
        {},
        tmp_path / "out",
        "https://fixture.invalid",
    )
    first = live.run_room(*args)
    assert first["status"] == "completed" and len(ready) == 4
    assert first["replays"][0]["status"] == "passed"
    assert sum(p["score_sum"] for p in first["players"]) == 0
    again = live.run_room(*args)
    assert again == first and len(ready) == 4


def test_actual_official_replay_matches_local_rules_and_is_physically_valid():
    from pathlib import Path
    from mahjong_ai.persistence.artifacts import read_json

    replay = read_json(
        Path(__file__).parent / "fixtures/official-v25-two-hand-replay.json"
    )
    report = audit_replay(replay)
    assert report["status"] == "passed" and report["hands"] == 2
    assert report["event_types"]["gang"] == 1
    assert report["outcomes"][-1]["local_payments"] == (-8, -1, -1, 10)


def test_actual_eight_hand_replay_uses_previous_winner_as_next_dealer():
    from pathlib import Path
    from mahjong_ai.persistence.artifacts import read_json

    replay = read_json(
        Path(__file__).parent / "fixtures/official-v25-eight-hand-replay.json"
    )
    report = audit_replay(replay)
    assert report["status"] == "passed", report["failures"]
    assert report["hands"] == 8
