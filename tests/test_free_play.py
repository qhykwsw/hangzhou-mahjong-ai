"""Public-session lifecycle regressions; no network or training."""

import pytest

from mahjong_ai.platform.free_play import run_match
from mahjong_ai.strategies.common import PolicySpec
from mahjong_ai.persistence.artifacts import atomic_json, read_json


class Client:
    def __init__(self):
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path))
        if path == "/api/me":
            return {"user_id": "me", "active_games": []}
        if path == "/api/match":
            return {"room_id": "room", "config": {"M": 2, "Rounds": 3}, "round_no": 1}
        if path == "/api/tournaments/room":
            return {
                "status": "finished",
                "my_games": ["a", "b"],
                "ranking": [{"user_id": "me", "total_score": 4}],
            }
        if path.startswith("/api/games/"):
            return {"finished": True, "snapshot": {"seat": 0, "scores": [2, -1, -1, 0]}}
        raise AssertionError(path)


class Runner:
    def __init__(self, client, policy, adapter, output, workers, join):
        assert workers >= 10 and not join
        self.root, self.client = output, client

    def run(self, tid, seconds):
        atomic_json(
            self.root / "tournament.json",
            self.client.request("GET", "/api/tournaments/" + tid),
        )


def credentials(tmp_path):
    path = tmp_path / "credentials.json"
    atomic_json(path, {"token": "fake-local-test-token"})
    return path


def test_single_match_freezes_and_collects_finals_without_ready_or_rematch(tmp_path):
    cred, client = credentials(tmp_path), Client()
    root = tmp_path / "competition"
    result = run_match(
        cred,
        PolicySpec(),
        root,
        "https://example.invalid",
        client=client,
        runner_class=Runner,
    )
    assert result["status"] == "completed"
    assert result["player"]["score_sum"] == 4
    assert result["feedback_to_iteration"] is False
    calls = list(client.calls)
    assert (
        run_match(
            cred,
            PolicySpec(),
            root,
            "https://example.invalid",
            client=client,
            runner_class=Runner,
        )
        == result
    )
    assert client.calls == calls
    assert [path for method, path in calls if method == "POST"] == ["/api/match"]
    assert read_json(root / "policy.json")["family"] == "rule"


def test_lost_match_response_never_rejoins_blindly(tmp_path):
    class Lost(Client):
        def request(self, method, path, body=None):
            if method == "POST":
                self.calls.append((method, path))
                raise ConnectionError("unknown outcome")
            return super().request(method, path, body)

    cred, client = credentials(tmp_path), Lost()
    root = tmp_path / "competition"
    with pytest.raises(ConnectionError):
        run_match(cred, PolicySpec(), root, "https://example.invalid", client=client)
    with pytest.raises(RuntimeError, match="unknown"):
        run_match(cred, PolicySpec(), root, "https://example.invalid", client=client)
    assert sum(method == "POST" for method, _ in client.calls) == 1


def test_resume_existing_room_and_reject_changed_policy(tmp_path):
    cred, client = credentials(tmp_path), Client()
    root = tmp_path / "competition"
    run_match(
        cred,
        PolicySpec(),
        root,
        "https://example.invalid",
        client=client,
        runner_class=Runner,
    )
    state = read_json(root / "match.json")
    state["status"] = "incomplete"
    atomic_json(root / "match.json", state)
    run_match(
        cred,
        PolicySpec(),
        root,
        "https://example.invalid",
        client=client,
        runner_class=Runner,
    )
    assert sum(method == "POST" for method, _ in client.calls) == 1
    with pytest.raises(ValueError, match="protocol changed"):
        run_match(
            cred,
            PolicySpec(version="changed"),
            root,
            "https://example.invalid",
            client=client,
        )


def test_closed_room_with_missing_finals_is_incomplete(tmp_path):
    class Closed(Client):
        def request(self, method, path, body=None):
            from mahjong_ai.platform.online import ApiError

            if path.startswith("/api/tournaments/") or path.startswith("/api/games/"):
                raise ApiError(404)
            return super().request(method, path, body)

    cred, client = credentials(tmp_path), Closed()
    result = run_match(
        cred,
        PolicySpec(),
        tmp_path / "competition",
        "https://example.invalid",
        client=client,
        runner_class=Runner,
    )
    assert result["status"] == "incomplete"
    assert result["valid_final_snapshots"] == 0


def test_match_busy_is_retried_and_other_rejections_are_not(monkeypatch):
    import mahjong_ai.platform.free_play as free_play
    from mahjong_ai.platform.free_play import request_match
    from mahjong_ai.platform.online import ApiError

    monkeypatch.setattr(free_play.time, "sleep", lambda _: None)

    class Busy:
        def __init__(self, errors):
            self.errors, self.calls = list(errors), 0

        def request(self, method, path, body=None):
            self.calls += 1
            if self.errors:
                raise self.errors.pop(0)
            return {"room_id": "room"}

    busy = Busy([ApiError(409, "MATCH_BUSY"), ApiError(409, "MATCH_BUSY")])
    assert request_match(busy) == {"room_id": "room"} and busy.calls == 3
    limit = Busy([ApiError(409, "MATCH_LIMIT_REACHED")])
    with pytest.raises(ApiError):
        request_match(limit)
    assert limit.calls == 1
