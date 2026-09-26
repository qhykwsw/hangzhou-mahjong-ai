"""Normalized contract fixtures; these are not captured official player snapshots."""

import json
import threading
import time
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


def test_history_refresh_keeps_snapshot_sequence_and_round_together(tmp_path):
    initial = fixture(HandState(RoundConfig(), 4)) | {"seq": 10}
    refreshed = fixture(HandState(RoundConfig(), 5, hand_no=2)) | {"seq": 14}

    class Client:
        reads = 0
        posts = 0

        def request(self, method, path, body=None):
            if method == "POST":
                self.posts += 1
                return {"ok": True}
            self.reads += 1
            if self.reads == 1:
                return initial
            if self.reads == 2:
                assert path.endswith("seq=1")
                return refreshed
            return {"finished": True, "seq": 15}

    client = Client()
    runner = TournamentRunner(client, PolicySpec(), adapter(), tmp_path)
    runner.tournament = {}
    runner._play("refresh-fixture")
    assert client.posts == 1
    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert record["snapshot"]["round_no"] == 2
    assert record["seq"] == 14
    state = json.loads(next(tmp_path.glob("game-*.json")).read_text())
    assert state["round_no"] == 2


def fake_connection(answers, fail=0):
    """A stand-in HTTPSConnection: `answers` are statuses, `fail` first attempts raise."""

    class Response:
        def __init__(self, answer):
            # An answer is a status, or (status, JSON body) for error bodies.
            status, self.body = answer if isinstance(answer, tuple) else (answer, b'{"ok":true}')
            self.status, self.reason, self.headers = status, "", {}

        def read(self):
            return self.body

    class Connection:
        opened = 0
        attempts = 0

        def __init__(self, *args, **kwargs):
            type(self).opened += 1

        def request(self, *args, **kwargs):
            type(self).attempts += 1
            if type(self).attempts <= fail:
                raise OSError("may contain sensitive provider information")

        def getresponse(self):
            return Response(answers.pop(0))

        def close(self):
            pass

    return Connection


def test_http_retries_429_but_not_unknown_post_outcome(monkeypatch):
    import mahjong_ai.platform.online as online

    monkeypatch.setattr(online.time, "sleep", lambda _: None)
    monkeypatch.setattr(online.http.client, "HTTPSConnection", fake_connection([429, 200]))
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    assert client.request("POST", "/action", {}) == {"ok": True}

    monkeypatch.setattr(online.http.client, "HTTPSConnection", fake_connection([200], fail=9))
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    with pytest.raises(ConnectionError, match="outcome may be unknown"):
        client.request("POST", "/action", {})


def test_tournament_gone_is_retried_but_not_found_is_final(monkeypatch):
    import mahjong_ai.platform.online as online

    monkeypatch.setattr(online.time, "sleep", lambda _: None)
    gone = (404, b'{"error":"TOURNAMENT_GONE"}')
    monkeypatch.setattr(online.http.client, "HTTPSConnection", fake_connection([gone, gone, 200]))
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    assert client.request("POST", "/api/tournaments/t1/ready") == {"ok": True}

    missing = (404, b'{"error":"TOURNAMENT_NOT_FOUND"}')
    monkeypatch.setattr(online.http.client, "HTTPSConnection", fake_connection([missing, 200]))
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    with pytest.raises(ApiError) as error:
        client.request("GET", "/api/tournaments/t1")
    assert error.value.code == "TOURNAMENT_NOT_FOUND"


def test_a_recycled_connection_is_reopened_for_the_next_poll(monkeypatch):
    import mahjong_ai.platform.online as online

    connection = fake_connection([200, 200], fail=0)
    monkeypatch.setattr(online.http.client, "HTTPSConnection", connection)
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    client.limiter.acquire = lambda priority=0: None
    client.request("GET", "/api/games/g/state?seq=0")
    client.request("GET", "/api/games/g/state?seq=1")
    assert connection.opened == 1  # 同一线程复用，不再每次握手

    recycled = fake_connection([200], fail=1)
    monkeypatch.setattr(online.http.client, "HTTPSConnection", recycled)
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    client.limiter.acquire = lambda priority=0: None
    client.local.connection = recycled()  # 上一次留下的连接已被对端回收
    assert client.request("GET", "/api/games/g/state?seq=2") == {"ok": True}
    assert recycled.opened == 2  # 重连一次，GET 幂等可以重试


def test_only_state_polling_spends_the_shared_budget(monkeypatch):
    import mahjong_ai.platform.online as online

    monkeypatch.setattr(online.http.client, "HTTPSConnection", fake_connection([200, 200, 200]))
    client = PlatformClient("https://example.invalid", "test-not-a-real-token")
    taken = []
    client.limiter.acquire = taken.append
    client.request("GET", "/api/games/g/state?seq=0", priority=1)
    client.request("POST", "/api/games/g/action", {"action": "pass"})
    client.request("GET", "/api/me")
    # 动作提交与杂项 GET 不排队：2026-09-20 实测把提交并入预算后，窗口错过率从
    # 0.3% 涨到 7%，拉取被提交系统性推迟，得不偿失。
    assert taken == [1]


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


def test_connections_warm_up_only_before_the_room_starts(tmp_path, monkeypatch):
    class Client:
        def __init__(self, status):
            self.status = status

        def discover(self, tid=None):
            return "t", True, {}

        def request(self, method, path, body=None):
            if path == "/api/me":
                return {"active_games": []}
            return {"status": self.status, "qualified": True}

    for status, expected in (("running", 0), ("registering", 1)):
        runner = TournamentRunner(
            Client(status), PolicySpec(), adapter(), tmp_path / status, workers=2
        )
        warmed = []
        monkeypatch.setattr(runner, "_warm_workers", lambda pool: warmed.append(pool))
        monkeypatch.setattr(runner.stop, "wait", lambda timeout: time.sleep(0.01))
        runner.run(seconds=0.3)
        # 预热要占住整个线程池：房间已经在打还去预热，等于把每桌的首次拉取推迟到
        # 第一批窗口之后，2026-09-20 实测开局两分钟错过率因此从 0.9% 涨到 2.4%。
        assert len(warmed) == expected


def test_the_deal_is_seen_within_a_quarter_second_even_through_tournament_gone(
    tmp_path, monkeypatch
):
    # 2026-09-26 02:42 的房：十桌同时发牌，各桌首次拉取晚于发牌 8.8-10.2 s，九桌的
    # 开局出牌超时。发牌只能靠下一次房间轮询发现，而发牌那一刻房间最常答
    # TOURNAMENT_GONE；开局前改为每 0.25 s 看一次，GONE 也立即重看。
    polls, started = [], {}

    class Client:
        def discover(self, tid=None):
            return "t", False, {}

        def request(self, method, path, body=None, priority=0, retries=None):
            if path == "/api/me":
                return {"active_games": [{"game_id": "g0"}, {"game_id": "g1"}]}
            polls.append(time.monotonic())
            room = {"config": {"StartAt": time.time() - 30}, "my_games": ["g0", "g1"]}
            if len(polls) <= 2:
                return room | {"status": "registering", "my_games": []}
            if len(polls) == 3:
                raise ApiError(404, "TOURNAMENT_GONE")  # the room is dealing
            return room | {"status": "finished" if len(started) == 2 else "running"}

    runner = TournamentRunner(Client(), PolicySpec(), adapter(), tmp_path, workers=2)
    monkeypatch.setattr(runner, "_warm_workers", lambda pool: None)
    monkeypatch.setattr(runner, "_play", lambda gid: started.setdefault(gid, time.monotonic()))
    runner.run(seconds=5)
    assert sorted(started) == ["g0", "g1"]
    assert polls[1] - polls[0] < 0.5
    assert max(started.values()) - polls[2] < 0.6  # was a 1 s poll plus a 2 s pause

    # A scheduled tournament is not polled at that pace an hour before its start.
    polls.clear()

    class Early(Client):
        def request(self, method, path, body=None, priority=0, retries=None):
            polls.append(time.monotonic())
            return {"status": "registering", "config": {"StartAt": time.time() + 3600}}

    runner = TournamentRunner(Early(), PolicySpec(), adapter(), tmp_path / "early", workers=2)
    monkeypatch.setattr(runner, "_warm_workers", lambda pool: None)
    runner.run(seconds=1.2)
    assert len(polls) <= 2


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


def test_fetch_plan_spends_requests_only_when_a_turn_can_be_pending():
    from mahjong_ai.platform.online import TournamentRunner

    plan = TournamentRunner._fetch_plan
    assert plan(None, 0, False, False, False, None) == "high"
    draw_other = {"phase": "draw", "seat": 1, "turn": 3, "responding_seats": []}
    assert plan(draw_other, 0, False, False, False, None) is None
    assert plan(draw_other, 1, False, False, False, None) == "medium"
    assert plan(draw_other, 0, True, False, False, None) == "low"
    # the seat right before us is about to discard: chi window and our draw follow
    draw_prev = {"phase": "draw", "seat": 1, "turn": 0, "responding_seats": []}
    assert plan(draw_prev, 1, False, False, False, None) == "high"
    # frames inside a predicted quiet stretch after our own action are skipped
    assert plan(draw_other, 2, False, False, True, None) is None
    # others' response window: a peng resolves it at once and the claimer's
    # discard is the second event, so one event can never need us
    others = {"phase": "response_peng", "seat": 1, "turn": 2, "responding_seats": [0, 3]}
    assert plan(others, 1, False, False, False, None) is None
    assert plan(others, 2, False, False, False, None) == "medium"
    others_prev = {"phase": "response_peng", "seat": 1, "turn": 0, "responding_seats": [2, 3]}
    assert plan(others_prev, 2, False, False, False, None) == "high"
    chi_other = {"phase": "response_chi", "seat": 1, "turn": 2, "responding_seats": [3]}
    assert plan(chi_other, 1, False, False, False, None) is None
    assert plan(chi_other, 2, False, False, False, None) == "medium"
    mine = {"phase": "response_peng", "seat": 1, "turn": 2, "responding_seats": [1, 3]}
    assert plan(mine, 1, False, False, False, None) == "medium"
    mine_chi = {"phase": "response_chi", "seat": 1, "turn": 0, "responding_seats": [1]}
    assert plan(mine_chi, 1, False, False, False, None) == "high"
    # states we already answered: our own draw follows a chi-window pass,
    # otherwise only a claimer's discard can need us
    assert plan(mine, 1, False, False, False, None, acted=True) == "medium"
    assert plan(mine_chi, 1, False, False, False, None, acted=True) == "high"



def test_rate_limiter_lets_high_priority_callers_go_first():
    from mahjong_ai.platform.online import RateLimiter

    clock = [0.0]
    order = []
    limiter = RateLimiter(rate=10, clock=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    limiter.acquire(priority=2)
    order.append(("high", clock[0]))
    limiter.acquire(priority=0)
    order.append(("low", clock[0]))
    limiter.acquire(priority=1)
    order.append(("medium", clock[0]))
    assert [round(t, 2) for _, t in order] == [0.0, 0.1, 0.2]
    assert limiter.waiting == [0, 0, 0]


def test_rate_limiter_promotes_a_starved_medium_caller():
    from mahjong_ai.platform.online import RateLimiter

    limiter = RateLimiter(rate=100)
    limiter.waiting[2] = 1  # a high caller that never stops waiting
    started = time.monotonic()
    limiter.acquire(priority=1)
    elapsed = time.monotonic() - started
    assert 0.2 <= elapsed < 0.6  # yields for a while, then is promoted past the high caller
    assert limiter.waiting == [0, 0, 1]


def test_needs_snapshot_only_for_events_that_can_require_our_action():
    from mahjong_ai.platform.online import TournamentRunner

    need = TournamentRunner._needs_snapshot
    snap = {"seat": 1, "turn": 2, "phase": "response_peng", "responding_seats": [0, 3]}
    assert need([{"type": "pass", "seat": 0}], snap) is False
    assert need([{"type": "timeout", "seat": 3, "data": {"kind": "response"}}], snap) is False
    assert need([{"type": "tile_discarded", "seat": 3}], snap) is True
    assert need([{"type": "tile_drawn", "seat": 1}], snap) is True
    assert need([{"type": "tile_drawn", "seat": 3}], snap) is False
    assert need([{"type": "peng", "seat": 0}], snap) is False
    assert need([{"type": "round_ended", "seat": None}], snap) is True
    assert need([{"type": "timeout", "seat": 0, "data": {"kind": "discard"}}], snap) is True
    # we are the next seat and already answered the peng window: its resolution
    # opens our chi window without any event
    chi_next = {"seat": 3, "turn": 2, "phase": "response_peng", "responding_seats": [0, 1]}
    assert need([{"type": "pass", "seat": 0}], chi_next) is True
    assert need([{"type": "mystery", "seat": 0}], snap) is True


def _peng_window(now_ms, turn, last_discard, responders=(0, 1, 2), seq=1, melds=None, counts=None):
    return {
        "game_id": "fixture",
        "seat": 0,
        "phase": "response_peng",
        "round_no": 1,
        "dealer": 0,
        "turn": turn,
        "waited_seat": 0,
        "responding_seats": list(responders),
        "wall_remaining": 60,
        "window_deadline_ms": now_ms,
        "discards": [[], ["7t"] if turn == 1 else [], [], [last_discard] if turn == 3 else []],
        "melds": melds or [[], [], [], []],
        "hand_counts": counts or [13, 13, 13, 13],
        "my_hand": ["1w", "2w", "东", "东", "南", "南", "西", "西", "北", "北", "中", "中", "白"],
        "last_discard": last_discard,
        "drawn_tile": "",
        "scores": [0, 0, 0, 0],
        "god": {"chain_count": 0, "baotou": False, "catch_play": False},
    }


class _NotifyClient:
    """Fake server with an SSE notify stream driven by the test script."""

    def __init__(self, states):
        self.states = states  # callable(client) -> response for a snapshot GET
        self.gets = 0
        self.posts = []
        self.frames = []
        self.posted = threading.Event()
        self.on_frame = None

    def request(self, method, path, body=None):
        if method == "POST":
            self.posts.append(body)
            self.posted.set()
            return {"ok": True}
        if "seq=0" not in path:
            return {"seq": 1, "pending": True}
        self.gets += 1
        return self.states(self)

    def notify(self, path, on_frame, stop):
        self.on_frame = on_frame
        on_frame(1, False)
        for frame in self.frames:
            frame(self, stop)
        stop.wait()
        return True


def _run_fixture(tmp_path, client):
    from mahjong_ai.platform.official import OfficialAdapter

    runner = TournamentRunner(client, PolicySpec(), OfficialAdapter(), tmp_path)
    runner.tournament = {
        "config": {"Kind": "test", "Rounds": 2, "BaseScore": 1, "YouCaiBiKao": False}
    }
    runner._play("fixture")
    state = json.loads(next(tmp_path.glob("game-*.json")).read_text())
    return state["client"]


def test_silent_chi_window_is_looked_for_at_the_peng_deadline(tmp_path):
    # Live behaviour (2026-09-17 二测): the peng window lasts exactly 1 s and,
    # when we are the next seat, our chi window opens at that deadline with no
    # event on the notify stream.  Fast opponents answer within milliseconds,
    # so no frame arrives in between: the client must look on its own timer.
    deadline = time.time() * 1000 + 300

    def states(client):
        if len(client.posts) >= 2:
            return {"finished": True, "seq": 3, "snapshot": None}
        snap = _peng_window(deadline, 3, "3w")
        if client.posts and time.time() * 1000 >= deadline:
            snap.update(phase="response_chi", responding_seats=[0], window_deadline_ms=deadline + 1000)
            return {"seq": 2, "snapshot": snap}
        return {"seq": 1 if not client.posts else 2, "snapshot": snap}

    def after_first_post(client, stop):
        client.posted.wait(5)
        client.posted.clear()
        client.on_frame(2, False)  # our own pass; nothing else follows
        client.posted.wait(5)
        client.on_frame(3, True)

    client = _NotifyClient(states)
    client.frames = [after_first_post]
    started = time.monotonic()
    diag = _run_fixture(tmp_path, client)
    assert [p["action"] for p in client.posts][0] == "pass"
    assert client.posts[1]["action"] in ("chi", "pass")
    assert time.monotonic() - started < 1.5
    assert diag["deadline_checks"] == 1
    assert client.gets == 3  # peng window, deadline check, final state


def test_claimers_discard_right_after_our_pass_is_not_skipped(tmp_path):
    # Another responder may peng at once; its discard is then only two events
    # after our pass and opens a window for us.  The old prediction skipped
    # the responders plus two events and missed a quarter of such windows.
    now = time.time() * 1000 + 1000

    def states(client):
        if len(client.posts) >= 2:
            return {"finished": True, "seq": 14, "snapshot": None}
        if client.posts:
            melds = [[], [], [], [{"kind": "peng", "tiles": ["5t", "5t", "5t"], "from_seat": 1}]]
            snap = _peng_window(now + 60, 3, "3w", melds=melds, counts=[13, 13, 13, 10])
            snap["discards"][1] = ["7t"]
            return {"seq": 13, "snapshot": snap}
        return {"seq": 10, "snapshot": _peng_window(now, 1, "7t", responders=(0, 2, 3))}

    def after_first_post(client, stop):
        client.posted.wait(5)
        client.posted.clear()
        for seq in (11, 12):  # our pass, seat 3 pengs
            client.on_frame(seq, False)
            time.sleep(0.03)
        client.on_frame(13, False)  # seat 3 discards: a window for us
        client.posted.wait(5)
        client.on_frame(14, True)

    client = _NotifyClient(states)
    client.frames = [after_first_post]
    started = time.monotonic()
    diag = _run_fixture(tmp_path, client)
    assert len(client.posts) == 2
    assert client.gets == 3  # first window, the claimer's window, final state
    assert diag["count_skips"] >= 1
    assert time.monotonic() - started < 1.0  # answered on the discard frame, not on the idle timeout


def test_action_unavailable_is_retried_while_the_window_has_time(tmp_path):
    from mahjong_ai.execution.candidate_runtime import ActionUnavailable
    from mahjong_ai.platform.official import OfficialAdapter

    class Stub:
        calls = 0

        def reset(self, seed):
            pass

        def act(self, obs):
            self.calls += 1
            if self.calls == 1:
                raise ActionUnavailable("slow candidate")
            return next(a for a in obs.legal_actions if a.action == "discard")

    class Spec:
        identity = "stub"
        action_seconds = 3.0
        response_seconds = 1.0

        def build(self):
            return stub

    stub = Stub()
    deadline = time.time() * 1000 + 3000

    class Client:
        gets = 0
        posts = []

        def request(self, method, path, body=None):
            if method == "POST":
                self.posts.append(body)
                return {"ok": True}
            if "seq=0" not in path:
                return {"seq": 1, "pending": True}
            if self.posts:
                return {"finished": True, "seq": 2, "snapshot": None}
            self.gets += 1
            return {
                "seq": 1,
                "snapshot": {
                    "game_id": "fixture",
                    "seat": 0,
                    "phase": "draw",
                    "round_no": 1,
                    "dealer": 0,
                    "turn": 0,
                    "responding_seats": [],
                    "wall_remaining": 60,
                    "window_deadline_ms": deadline,
                    "discards": [[], [], [], []],
                    "melds": [[], [], [], []],
                    "hand_counts": [14, 13, 13, 13],
                    "my_hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "东", "南"],
                    "last_discard": "",
                    "drawn_tile": "南",
                    "scores": [0, 0, 0, 0],
                    "god": {"chain_count": 0, "baotou": False, "catch_play": False},
                },
            }

    client = Client()
    runner = TournamentRunner(client, Spec(), OfficialAdapter(), tmp_path)
    runner.tournament = {
        "config": {"Kind": "test", "Rounds": 2, "BaseScore": 1, "YouCaiBiKao": False}
    }
    runner._play("fixture")
    assert stub.calls == 2 and client.posts[0]["action"] == "discard"
    diag = json.loads(next(tmp_path.glob("game-*.json")).read_text())["client"]
    assert diag["action_unavailable"] == 1 and diag["action_retries"] == 1


def test_action_timeline_tells_claims_from_answers_and_the_draw():
    from mahjong_ai.platform.online import ActionTimeline

    end = 100.0  # local time the peng window closes; we acted on seq 10 at 99.2
    # (our own pass, seq 11, is skipped by count in the loop and never reaches the timeline)
    # Frames that may be a claim are still skipped, but whatever follows one is
    # looked at: a claim is followed by the claimer's discard, which opens a
    # window for us, and missing it costs a peng we could have called.
    # 1. fast bots, nobody claims: both answers were already in our snapshot
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(100.3, 11, 11) == "quiet"  # deadline passed, no burst: nothing to fetch
    assert tl.decide(100.4, 12, 11) == "quiet"  # the next seat's chi-window answer
    assert tl.decide(101.0, 13, 12) == "fetch"  # its draw, or a chi's discard: look
    # 2. slow table: nobody answered before the deadline, two timeouts arrive as the burst
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(100.0, 13, 11) == ("defer", end + tl.BURST_END)
    assert tl.decide(100.25, 13, 13) == "quiet"  # burst of 2 = the 2 expected timeouts
    assert tl.decide(101.0, 15, 13) == "quiet"  # chi-window timeout together with the draw
    assert tl.decide(101.3, 16, 15) == "fetch"  # the discard
    # 2b. the next seat answers early and its draw arrives alone
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(100.0, 13, 11) == ("defer", end + tl.BURST_END)
    assert tl.decide(100.25, 13, 13) == "quiet"
    assert tl.decide(100.6, 14, 13) == "quiet"  # the next seat's answer
    assert tl.decide(101.0, 15, 14) == "fetch"  # its draw, or a chi's discard: look
    # 3. a bot chis right at the deadline and discards at once: the frames that
    #    follow the answer we skipped before the deadline are looked at
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(99.5, 12, 11) == "followup"  # one answer after ours
    tl.pre_answer_skipped = True
    assert tl.decide(100.03, 15, 12) == "fetch"
    # 3b. the next seat chis a little later and discards before the timed draw
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(100.3, 11, 11) == "quiet"
    assert tl.decide(100.5, 12, 11) == "quiet"  # the chi, taken for an answer
    assert tl.decide(100.7, 13, 12) == "fetch"  # its discard, too early for the draw
    # 3c. seen live 2026-09-20: the chi arrives after the burst and its discard
    #     lands where the timed draw would be, so it must not be taken for one
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(100.0, 13, 11) == ("defer", end + tl.BURST_END)
    assert tl.decide(100.25, 13, 13) == "quiet"  # the two expected timeouts
    assert tl.decide(100.6, 14, 13) == "quiet"  # the chi, taken for an answer
    assert tl.decide(101.02, 15, 14) == "fetch"  # its discard opens our window
    # 3d. seen live 2026-09-20: a peng before the deadline looks like an answer
    #     and its discard lands one second later, on the timed draw
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(99.6, 12, 11) == "followup"  # the peng
    tl.pre_answer_skipped = True  # the follow-up saw nothing yet
    assert tl.decide(100.9, 13, 12) == "fetch"  # its discard, whenever it lands
    # 4. a peng before the deadline: the lone claim frame asks for a follow-up look
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(99.4, 12, 11) == "followup"
    tl.pre_answer_skipped = True  # the follow-up saw nothing: taken for an answer
    assert tl.decide(99.6, 13, 12) == "fetch"  # a second lone frame is looked at
    # 4b. a peng resolves the window before any timeout: fewer burst frames than
    #     silent responders, and the claimer's discard follows
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(100.05, 12, 11) == ("defer", end + tl.BURST_END)
    assert tl.decide(100.25, 12, 12) == "fetch"  # burst of 1 < 2 expected timeouts
    # 5. two frames at once before the deadline: a peng and its discard
    tl = ActionTimeline(10, end, 2, 0, 1)
    assert tl.decide(99.4, 13, 11) == "fetch"
    # a claim seen in a fetched state ends the timeline
    assert tl.stale({"round_no": 1, "melds": [[], [{"kind": "peng"}], [], []]}) is True
    assert tl.stale({"round_no": 1, "melds": [[], [], [], []]}) is False
    # 6. a look taken meanwhile showed the next seat already holding its draw:
    #    the following frame is its discard, not the draw
    tl = ActionTimeline(10, end, 2, 0, 1)
    tl.sync({"phase": "draw", "turn": 2, "hand_counts": [13, 13, 14, 13]}, 2)
    assert tl.decide(101.02, 16, 15) == "fetch"
    tl = ActionTimeline(10, end, 2, 0, 1)
    tl.sync({"phase": "response_chi", "turn": 1, "hand_counts": [13, 13, 13, 13]}, 2)
    assert tl.decide(100.5, 14, 13) == "quiet"  # the next seat's answer
    assert tl.decide(101.0, 15, 14) == "fetch"  # its draw, or a chi's discard: look


def test_timeout_burst_pass_and_draw_cost_no_request_until_the_discard(tmp_path):
    # After our pass in a peng window (we are not the next seat): two timeouts
    # at the deadline, the next seat's chi-window pass, its timed draw and
    # its discard.  Only the discard can open a window for us, so the client
    # must look exactly once more, on that frame.
    end = time.time() * 1000 + 500

    def states(client):
        if len(client.posts) >= 2:
            return {"finished": True, "seq": 17, "snapshot": None}
        if client.posts:
            snap = _peng_window(time.time() * 1000 + 1000, 3, "3w")
            snap["discards"][1] = ["7t"]
            return {"seq": 16, "snapshot": snap}
        return {"seq": 10, "snapshot": _peng_window(end, 1, "7t", responders=(0, 2, 3))}

    def script(client, stop):
        client.posted.wait(5)
        client.posted.clear()
        client.on_frame(11, False)  # our pass
        time.sleep(max(0.0, end / 1000 - time.time()))
        client.on_frame(12, False)  # timeout
        client.on_frame(13, False)  # timeout
        time.sleep(0.3)
        client.on_frame(14, False)  # the next seat passes its chi window
        time.sleep(0.7)
        client.on_frame(15, False)  # its draw, one second after the deadline
        time.sleep(0.05)
        client.on_frame(16, False)  # its discard: a window for us
        client.posted.wait(5)
        client.on_frame(17, True)

    client = _NotifyClient(states)
    client.frames = [script]
    diag = _run_fixture(tmp_path, client)
    assert len(client.posts) == 2
    assert client.gets == 3  # our window, the next seat's discard, final state
    assert diag["timeline_fetches"] == 1


def test_next_seat_without_a_chi_window_still_plays_its_own_draw(tmp_path):
    # We pass a peng window as the next seat, but no chi window follows (the
    # server gives us none): our draw one second after the deadline must be
    # fetched and discarded, not taken for the next seat's timed draw.
    end = time.time() * 1000 + 400

    def states(client):
        if len(client.posts) >= 2:
            return {"finished": True, "seq": 15, "snapshot": None}
        if client.posts:
            snap = _peng_window(end, 3, "3w")
            snap.update(phase="draw", turn=0, responding_seats=[], drawn_tile="5w", window_deadline_ms=end + 4000)
            snap["my_hand"] = snap["my_hand"] + ["5w"]
            snap["hand_counts"] = [14, 13, 13, 13]
            return {"seq": 14, "snapshot": snap}
        return {"seq": 10, "snapshot": _peng_window(end, 3, "3w")}

    def script(client, stop):
        client.posted.wait(5)
        client.posted.clear()
        client.on_frame(11, False)  # our pass
        time.sleep(max(0.0, end / 1000 - time.time()))
        client.on_frame(13, False)  # two timeouts at the deadline
        time.sleep(1.0)
        client.on_frame(14, False)  # our own draw
        client.posted.wait(5)
        client.on_frame(15, True)

    client = _NotifyClient(states)
    client.frames = [script]
    started = time.monotonic()
    _run_fixture(tmp_path, client)
    assert [p["action"] for p in client.posts] == ["pass", "discard"]
    assert time.monotonic() - started < 2.5


def test_a_newly_dealt_table_is_looked_at_before_its_stream_and_policy_are_up(tmp_path):
    # The dealer's first discard window runs 3 s from the deal.  The first look
    # must not wait for the notify stream's first frame (a fresh connection per
    # table, ten at once), and a sandboxed policy's process start runs beside
    # that look instead of before it.
    from mahjong_ai.platform.official import OfficialAdapter

    class Stub:
        def reset(self, seed):
            pass

        def act(self, obs):
            return next(a for a in obs.legal_actions if a.action == "discard")

    built = []

    class SlowSpec:
        identity = "stub"
        action_seconds = 3.0
        response_seconds = 1.0

        def build(self):
            time.sleep(0.3)  # a sandbox process start with ten tables dealt together
            built.append(time.monotonic())
            return Stub()

    dealer = {
        "game_id": "fixture",
        "seat": 0,
        "phase": "draw",
        "round_no": 1,
        "dealer": 0,
        "turn": 0,
        "responding_seats": [],
        "wall_remaining": 60,
        "window_deadline_ms": time.time() * 1000 + 3000,
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "hand_counts": [14, 13, 13, 13],
        "my_hand": ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1t", "2t", "3t", "东", "南"],
        "last_discard": "",
        "drawn_tile": "南",
        "scores": [0, 0, 0, 0],
        "god": {"chain_count": 0, "baotou": False, "catch_play": False},
    }

    class Client:
        def __init__(self):
            self.gets, self.posts = [], []
            self.posted = threading.Event()

        def request(self, method, path, body=None):
            if method == "POST":
                self.posts.append(time.monotonic())
                self.posted.set()
                return {"ok": True}
            self.gets.append(time.monotonic())
            if self.posts:
                return {"finished": True, "seq": 2, "snapshot": None}
            return {"seq": 0, "snapshot": dealer}

        def notify(self, path, on_frame, stop):
            # The stream is still connecting until after our discard.
            while not self.posted.wait(0.01):
                if stop.is_set():
                    return True
            on_frame(2, True)
            return True

    client = Client()
    runner = TournamentRunner(client, SlowSpec(), OfficialAdapter(), tmp_path)
    runner.tournament = {
        "config": {"Kind": "test", "Rounds": 2, "BaseScore": 1, "YouCaiBiKao": False}
    }
    begin = time.monotonic()
    runner._play("fixture")
    assert client.gets[0] < built[0]  # looked before the policy was even up
    assert client.posts[0] - begin < 0.9  # was 0.3 s start + 1 s frame wait + fetch
