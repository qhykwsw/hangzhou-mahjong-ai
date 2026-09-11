"""Normalized contract fixtures; these are not captured official player snapshots."""

from dataclasses import asdict

import pytest

from mahjong_ai.domain.engine import HandState
from mahjong_ai.platform.online import (
    ApiError,
    MAPPING_TEMPLATE,
    PlatformClient,
    RateLimiter,
    SnapshotAdapter,
    SnapshotError,
    TournamentRunner,
    active_games,
    wire_action,
)
from mahjong_ai.strategies.common import PolicySpec
from mahjong_ai.domain.rules import tile_code
from mahjong_ai.domain.types import Action, RoundConfig


def fixture(state):
    obs = state.observation()
    d = dict(
        seat=state.turn,
        phase=state.phase,
        turn=state.turn,
        responding_seats=[state.turn],
        hand=[tile_code(t) for t in obs.hand],
        drawn_tile=tile_code(obs.drawn_tile) if obs.drawn_tile is not None else None,
        melds=[
            [asdict(m) | {"tiles": [tile_code(t) for t in m.tiles]} for m in row]
            for row in obs.melds
        ],
        scores=list(obs.stage.score_by_seat),
        round_no=state.hand_no,
        catch_play=obs.catch_play,
        god_discarder_seat=obs.god_discarder_seat,
        chain_count=obs.chain_count,
        dealer=state.dealer,
        rounds_total=state.config.rounds,
        base_score=state.config.base_score,
        you_cai_bi_kao=state.config.you_cai_bi_kao,
        discards=[[tile_code(t) for t in row] for row in obs.discards],
        hand_sizes=list(obs.hand_sizes),
        wall_remaining=len(state.wall),
        discarded_tile=tile_code(state.discarded)
        if state.discarded is not None
        else None,
        kong_draw=state.kong_draw,
        piao_count=obs.piao_count,
        public_events=[],
        stage_kind=state.config.stage,
        stage_no=1,
        stage_scores=None,
        stage_round_no=None,
        stage_rounds_total=None,
        in_tiebreak=False,
    )
    return {"snapshot": d}


def adapter():
    return SnapshotAdapter({key: "snapshot." + key for key in MAPPING_TEMPLATE})


def test_mapping_requires_verified_fields_and_fails_closed():
    with pytest.raises(SnapshotError, match="verify every"):
        SnapshotAdapter(MAPPING_TEMPLATE)
    snap = fixture(HandState(RoundConfig(rounds=11), 45))
    obs = adapter()(snap)
    assert obs.stage.rounds_total == 11
    del snap["snapshot"]["wall_remaining"]
    with pytest.raises(SnapshotError, match="missing mapped field"):
        adapter()(snap)


def test_local_and_mapped_actions_match_across_all_phases():
    state = HandState(RoundConfig(rounds=3), 41)
    phases = set()
    while not state.settled:
        obs = adapter()(fixture(state))
        phases.add(obs.phase)
        assert obs.legal_actions == state.legal_actions()
        discards = [a for a in obs.legal_actions if a.action == "discard"]
        state.step(discards[0] if discards else obs.legal_actions[0])
    assert phases == {"draw", "response_peng", "response_chi"}


def test_rate_limit_shared_reservations():
    waits = []
    limit = RateLimiter(rate=4, clock=lambda: 0, sleep=waits.append)
    for _ in range(4):
        limit.acquire()
    assert waits == [0, 0.25, 0.5, 0.75]


def test_global_token_only_plays_tournament_intersection():
    me = {"active_games": [{"game_id": "a"}, {"game_id": "b"}]}
    assert active_games(me, {"my_games": ["a", "old"]}, False) == ["a"]
    assert active_games(me, {}, True) == ["a", "b"]
    assert wire_action(Action("gang", 0, kind="concealed")) == {
        "action": "gang",
        "tile": "1w",
    }


def test_409_resync_does_not_submit_duplicate_action(tmp_path):
    data = fixture(HandState(RoundConfig(), 4))
    calls = []

    class Client:
        reads = 0

        def request(self, method, path, body=None):
            calls.append((method, path))
            if method == "POST":
                raise ApiError(409)
            self.reads += 1
            return {"finished": True} if self.reads == 3 else data | {"seq": 10}

    runner = TournamentRunner(Client(), PolicySpec(), adapter(), tmp_path)
    runner.tournament = {}
    runner._play("test")
    assert sum(m == "POST" for m, _ in calls) == 1
    post_index = next(i for i, (method, _) in enumerate(calls) if method == "POST")
    assert calls[post_index + 1][1].endswith("seq=0")


def test_http_retries_429_but_not_unknown_post_outcome(monkeypatch):
    import io
    import urllib.error
    import mahjong_ai.platform.online as online

    calls = 0

    def request(req, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "limited", {}, io.BytesIO())
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(online.urllib.request, "urlopen", request)
    monkeypatch.setattr(online.time, "sleep", lambda _: None)
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    assert client.request("POST", "/action", {}) == {"ok": True} and calls == 2

    def failed(*a, **kw):
        raise urllib.error.URLError("may contain sensitive provider information")

    monkeypatch.setattr(online.urllib.request, "urlopen", failed)
    with pytest.raises(ConnectionError, match="outcome may be unknown"):
        client.request("POST", "/action", {})


def test_stage_loop_registers_before_detail_and_runs_games_concurrently(
    tmp_path, monkeypatch
):
    import threading
    import time

    calls = []

    class Client:
        stages = 0

        def discover(self, tid=None):
            return "t", True, {}

        def request(self, method, path, body=None):
            calls.append((method, path))
            if method == "POST":
                return {}
            if path == "/api/me":
                return {"active_games": [{"game_id": "a"}, {"game_id": "b"}]}
            self.stages += 1
            status = ("stage_done", "stage_open", "running", "running", "finished")[
                min(self.stages - 1, 4)
            ]
            return {
                "status": status,
                "qualified": True,
                "ranking": [{"total_score": 30}],
            }

    runner = TournamentRunner(
        Client(), PolicySpec(), adapter(), tmp_path, workers=2, join=True
    )
    barrier = threading.Barrier(2, timeout=2)
    played = []

    def play(gid):
        barrier.wait()
        played.append(gid)

    monkeypatch.setattr(runner, "_play", play)
    monkeypatch.setattr(runner.stop, "wait", lambda timeout: time.sleep(0.005))
    runner.run(seconds=3)
    assert sorted(played) == ["a", "b"]
    assert calls[:3] == [
        ("POST", "/api/tournaments/t/register"),
        ("POST", "/api/tournaments/t/ready"),
        ("GET", "/api/tournaments/t"),
    ]
    assert sum(path.endswith("/ready") for _, path in calls) == 2
    from mahjong_ai.persistence.artifacts import read_json

    assert read_json(tmp_path / "tournament.json")["ranking"][0]["total_score"] == 30


def test_official_chi_window_can_open_without_a_new_event(tmp_path):
    from mahjong_ai.platform.official import OfficialAdapter

    # Actual v25 field contract: phase changes from peng to chi at the same seq.
    snap = {
        "game_id": "fixture",
        "seat": 0,
        "phase": "response_peng",
        "round_no": 1,
        "dealer": 0,
        "turn": 3,
        "responding_seats": [1, 2],
        "wall_remaining": 60,
        "window_deadline_ms": 9999999999999,
        "discards": [[], [], [], ["3w"]],
        "melds": [[], [], [], []],
        "hand_counts": [13, 13, 13, 13],
        "my_hand": [
            "1w",
            "2w",
            "东",
            "东",
            "南",
            "南",
            "西",
            "西",
            "北",
            "北",
            "中",
            "中",
            "白",
        ],
        "last_discard": "3w",
        "drawn_tile": "",
        "scores": [0, 0, 0, 0],
        "god": {"chain_count": 0, "baotou": False, "catch_play": False},
    }

    class Client:
        snapshots = 0
        submitted = None

        def request(self, method, path, body=None):
            if method == "POST":
                self.submitted = body
                return {"ok": True}
            if self.submitted:
                return {"finished": True, "snapshot": snap}
            if path.endswith("seq=1"):
                return {"seq": 1, "pending": True}
            self.snapshots += 1
            assert self.snapshots <= 3
            if self.snapshots == 2:
                snap["phase"] = "response_chi"
                snap["responding_seats"] = [0]
                snap["window_deadline_ms"] += 1000
            return {"seq": 1, "snapshot": snap.copy()}

    client = Client()
    runner = TournamentRunner(client, PolicySpec(), OfficialAdapter(), tmp_path)
    runner.tournament = {
        "config": {"Kind": "test", "Rounds": 2, "BaseScore": 1, "YouCaiBiKao": False}
    }
    runner._play("fixture")
    assert client.submitted["action"] in ("chi", "pass")
    assert client.snapshots == 2


def test_v31_settled_window_snapshot_is_never_played(tmp_path):
    from mahjong_ai.platform.official import OfficialAdapter

    # v31: the 5-second inter-round window reports the previous round's own
    # furniture (phase="settled"); acting on it would use stale tiles.
    snap = {
        "seat": 0,
        "phase": "settled",
        "round_no": 2,
        "dealer": 0,
        "turn": 0,
        "responding_seats": [],
        "wall_remaining": 0,
        "discards": [["3w"], [], [], []],
        "melds": [[], [], [], []],
        "hand_counts": [13, 13, 13, 13],
        "my_hand": ["1w"] * 13,
        "last_discard": "3w",
        "drawn_tile": "",
        "scores": [8, -8, 0, 0],
        "god": {"chain_count": 0, "baotou": False, "catch_play": False},
    }

    class Client:
        def __init__(self):
            self.posts = 0
            self.polls = 0

        def request(self, method, path, body=None):
            if method == "POST":
                self.posts += 1
                return {"ok": True}
            self.polls += 1
            if self.polls > 3:
                return {"finished": True, "snapshot": snap}
            return {"seq": 1, "snapshot": snap}

    client = Client()
    runner = TournamentRunner(client, PolicySpec(), OfficialAdapter(), tmp_path)
    runner.tournament = {
        "config": {"Kind": "test", "Rounds": 8, "BaseScore": 1, "YouCaiBiKao": False}
    }
    runner._play("fixture")
    assert client.posts == 0
    assert client.polls == 4


def test_actual_v25_dealer_snapshot_supports_empty_sentinel_and_missing_response_field():
    from mahjong_ai.platform.official import OfficialAdapter
    from mahjong_ai.persistence.artifacts import read_json
    from pathlib import Path

    payload = read_json(
        Path(__file__).parent / "fixtures/official-v25-dealer-snapshot.json"
    )
    obs = OfficialAdapter()(
        {
            "snapshot": payload["snapshot"],
            "wire_events": [],
            "tournament": {
                "config": {
                    "Kind": "test",
                    "Rounds": 2,
                    "BaseScore": 1,
                    "YouCaiBiKao": False,
                }
            },
        }
    )
    assert obs.drawn_tile is not None and len(obs.hand) == 14
    assert len(obs.legal_actions) > 1 and obs.stage.rounds_total == 2


def test_official_piao_and_kong_replenishment_recover_from_own_events():
    from mahjong_ai.platform.official import OfficialAdapter
    from mahjong_ai.persistence.artifacts import read_json
    from pathlib import Path

    payload = read_json(
        Path(__file__).parent / "fixtures/official-v25-dealer-snapshot.json"
    )
    snap = payload["snapshot"]
    snap["god"]["chain_count"] = 2
    events = [
        {
            "seq": 1,
            "type": "tile_discarded",
            "seat": 0,
            "tile": "白",
            "data": None,
            "_round_no": 1,
        },
        {
            "seq": 2,
            "type": "gang",
            "seat": 0,
            "tile": "2w",
            "data": {"kind": "bu"},
            "_round_no": 1,
        },
        {
            "seq": 3,
            "type": "tile_drawn",
            "seat": 0,
            "tile": snap["drawn_tile"],
            "data": None,
            "_round_no": 1,
        },
    ]
    obs = OfficialAdapter()(
        {
            "snapshot": snap,
            "wire_events": events,
            "tournament": {
                "config": {
                    "Kind": "test",
                    "Rounds": 2,
                    "BaseScore": 1,
                    "YouCaiBiKao": False,
                }
            },
        }
    )
    assert obs.piao_count == 1 and obs.chain_count == 2
