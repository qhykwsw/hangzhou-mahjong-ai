"""Official HTTP transport and strict, explicitly mapped player snapshots.

The guide omits several snapshot fields. A mapping verified against an actual
player snapshot is required for play; the probe needs no mapping and never acts.
"""

from __future__ import annotations

import inspect
import http.client
import io
import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from mahjong_ai.domain.engine import HandState, derive_seed
from mahjong_ai.domain.rules import tile_id
from mahjong_ai.domain.observations import meld_origins, policy_events
from mahjong_ai.persistence.artifacts import (
    atomic_json,
    digest,
    exclusive_lock,
    read_json,
)
from mahjong_ai.domain.types import (
    Meld,
    Observation,
    PublicEvent,
    RoundConfig,
    StageContext,
    StageKind,
)


class ApiError(RuntimeError):
    def __init__(self, status, code=""):
        self.status, self.code = status, code
        super().__init__(
            f"platform HTTP {status}" + (f" {code}" if code else "")
        )  # never echo credentials or HTTP bodies


class SnapshotError(ValueError):
    pass


# Seconds a low (index 0) or medium (index 1) limiter caller yields to higher
# levels before it is promoted one level.
PROMOTE_AFTER = (0.5, 0.25)


class RateLimiter:
    """Shared across all game workers for one credential/user process.

    Three priority levels: a caller waits while any caller of a higher level is
    waiting, so the shared /state budget goes first to tables where our own
    draw or discard may be pending (2), then to possible claim windows (1),
    then to housekeeping polls (0).
    """

    def __init__(self, rate=14, clock=time.monotonic, sleep=time.sleep):
        self.interval, self.clock, self.sleep = 1 / rate, clock, sleep
        self.next = 0.0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.waiting = [0, 0, 0]

    @property
    def high_waiting(self):
        return self.waiting[2]

    def acquire(self, priority=0):
        priority = 2 if priority is True else 0 if priority is False else int(priority)
        level = max(0, min(2, priority))
        with self.cond:
            self.waiting[level] += 1
            started = self.clock()
            try:
                while any(self.waiting[above] for above in range(level + 1, 3)):
                    self.cond.wait(timeout=0.05)
                    # A caller that has yielded long enough is promoted, so a
                    # claim window is never starved by a stream of own-turn
                    # requests from the other tables.
                    waited = self.clock() - started
                    promoted = level
                    while promoted < 2 and waited >= PROMOTE_AFTER[promoted]:
                        promoted += 1
                    if promoted > level:
                        self.waiting[level] -= 1
                        level = promoted
                        self.waiting[level] += 1
                        self.cond.notify_all()
                now = self.clock()
                scheduled = max(now, self.next)
                self.next = scheduled + self.interval
            finally:
                self.waiting[level] -= 1
                self.cond.notify_all()
        self.sleep(max(0.0, scheduled - now))


class PlatformClient:
    def __init__(self, base, token, insecure=False, retries=3):
        if not token:
            raise ValueError("token environment variable is empty")
        self.base, self._token, self.retries = base.rstrip("/"), token, retries
        self.context = (
            ssl._create_unverified_context()
            if insecure
            else ssl.create_default_context()
        )
        self.limiter = RateLimiter()
        self.local = threading.local()  # per-thread connection, limiter wait and 429 count
        split = urllib.parse.urlsplit(self.base)
        self.host, self.port = split.hostname, split.port

    def _send(self, method, path, body, headers):
        """One request over this thread's kept-alive connection.

        Every new connection costs a TLS handshake, and a burst of them stalls
        several tables at once: 2026-09-20 实测新建连接 p50 17 ms、p95 109 ms、
        最慢 2.0 s，复用后 4.7 / 10.5 / 17 ms。连接被对端回收时 GET 重连重试一次，
        提交不重放——结果未知一律由调用方重新同步。
        """
        payload = json.dumps(body).encode() if body is not None else None
        for attempt in (0, 1):
            connection = getattr(self.local, "connection", None)
            reused = connection is not None
            if connection is None:
                connection = http.client.HTTPSConnection(
                    self.host, self.port, timeout=35, context=self.context
                )
                self.local.connection = connection
            try:
                connection.request(method, path, body=payload, headers=headers)
                response = connection.getresponse()
                data = response.read()
            except (OSError, http.client.HTTPException) as exc:
                connection.close()
                self.local.connection = None
                if attempt or not reused or method != "GET":
                    raise urllib.error.URLError(type(exc).__name__) from None
                continue
            if response.status >= 400:
                raise urllib.error.HTTPError(
                    self.base + path, response.status, response.reason,
                    response.headers, io.BytesIO(data),
                )
            return json.loads(data) if data else {}

    def request(self, method, path, body=None, priority=0, retries=None):
        """``retries`` overrides the client's own backoff for one call; 0 hands
        every failure straight back to a caller that polls on its own clock."""
        retries = self.retries if retries is None else retries
        self.local.throttled = 0
        for attempt in range(retries + 1):
            # 只有 /state 轮询计入本地预算：官方额度也是按轮询算的，动作提交笔数少、
            # 卡着窗口，排进同一队列会让每次拉取都慢一档——实测错过率反而涨到 7%。
            if "/state?" in path:
                began = time.perf_counter()
                self.limiter.acquire(priority)
                self.local.wait_ms = (time.perf_counter() - began) * 1000
            try:
                return self._send(
                    method,
                    path,
                    body,
                    {
                        "Authorization": "Bearer " + self._token,
                        "Content-Type": "application/json",
                    },
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < retries:
                    self.local.throttled += 1
                    try:
                        delay = float(exc.headers.get("Retry-After", 2**attempt))
                    except ValueError:
                        delay = 2**attempt
                    time.sleep(min(10, max(0.1, delay)))
                    continue
                try:
                    error = json.load(exc)
                    code = error.get("error", error.get("code", ""))
                    code = (
                        code
                        if isinstance(code, str) and re.fullmatch(r"[A-Z_0-9]+", code)
                        else ""
                    )
                except (ValueError, TypeError, AttributeError):
                    code = ""
                if exc.code == 404 and code == "TOURNAMENT_GONE" and attempt < retries:
                    # Guide v35: the room exists but did not answer in time (busy at
                    # start or settlement); the endpoints concerned are idempotent,
                    # so retry. TOURNAMENT_NOT_FOUND (or no code) is permanent.
                    time.sleep(min(10, 2**attempt))
                    continue
                raise ApiError(exc.code, code) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                # A POST may have been accepted: caller must resync, never replay it blindly.
                if method != "GET" or attempt == retries:
                    raise ConnectionError(
                        "platform connection failed; action outcome may be unknown"
                    ) from None
                time.sleep(min(4, 2**attempt))

    def notify(self, path, on_frame, stop):
        """Follow the server-sent seq stream of one game until closed or stopped.

        Frames carry only a seq watermark (and ``closed`` at the end); the
        stream does not consume the /state request budget.  Reconnects with a
        short backoff; returns False when the endpoint is unavailable so the
        caller can fall back to timed polling.
        """
        failures = 0
        while not stop.is_set():
            req = urllib.request.Request(
                self.base + path,
                headers={"Authorization": "Bearer " + self._token, "Accept": "text/event-stream"},
            )
            try:
                with urllib.request.urlopen(req, timeout=45, context=self.context) as response:
                    failures = 0
                    for raw in response:
                        if stop.is_set():
                            return True
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue  # comments / keepalives / blank separators
                        try:
                            frame = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        if not isinstance(frame, dict):
                            continue
                        on_frame(int(frame.get("seq", 0) or 0), bool(frame.get("closed")))
                        if frame.get("closed"):
                            return True
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 405, 501):
                    return False
                failures += 1
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                failures += 1
            if stop.is_set():
                return True
            if failures >= 8:
                return False
            time.sleep(min(5.0, 0.5 * 2 ** min(failures, 4)))
        return True

    def discover(self, tournament_id=None):
        me = self.request("GET", "/api/me")
        scoped = me.get("tournament_id")
        if scoped and tournament_id and scoped != tournament_id:
            raise ValueError("requested tournament differs from scoped token")
        tid = scoped or tournament_id
        if not tid:
            raise ValueError("global token requires --tournament-id")
        return str(tid), bool(scoped), me


def _id(value):
    return quote(str(value), safe="")


def _redact(value):
    if isinstance(value, dict):
        return {
            k: _redact(v)
            for k, v in value.items()
            if not any(
                x in k.lower() for x in ("token", "password", "secret", "authorization")
            )
        }
    if isinstance(value, list):
        return [_redact(x) for x in value]
    return value


def probe(client, output, tournament_id=None):
    tid, scoped, me = client.discover(tournament_id)
    tournament = client.request("GET", f"/api/tournaments/{_id(tid)}")
    games = active_games(me, tournament, scoped)
    data = {"me": me, "tournament": tournament, "games": {}}
    for gid in games:
        data["games"][gid] = client.request("GET", f"/api/games/{_id(gid)}/state?seq=0")
    root = Path(output).parent
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    atomic_json(output, _redact(data))
    return {"games": len(games), "output": str(output), "actions_submitted": 0}


def active_games(me, tournament, scoped):
    ids = [str(g["game_id"]) for g in me["active_games"]]
    return ids if scoped else [g for g in ids if g in tournament.get("my_games", [])]


# Null entries deliberately prevent play until verified against a player snapshot.
MAPPING_TEMPLATE = {
    "seat": "snapshot.seat",
    "phase": "snapshot.phase",
    "turn": "snapshot.turn",
    "responding_seats": "snapshot.responding_seats",
    "hand": "snapshot.my_hand",
    "drawn_tile": "snapshot.drawn_tile",
    "melds": "snapshot.melds",
    "scores": "snapshot.scores",
    "round_no": "snapshot.round_no",
    "catch_play": "snapshot.god.catch_play",
    "chain_count": "snapshot.god.chain_count",
    "god_discarder_seat": "snapshot.god.god_discarder_seat",
    "dealer": None,
    "rounds_total": None,
    "base_score": None,
    "you_cai_bi_kao": None,
    "discards": None,
    "hand_sizes": None,
    "wall_remaining": None,
    "discarded_tile": None,
    "kong_draw": None,
    "piao_count": None,
    "public_events": None,
    "stage_kind": None,
    "stage_no": None,
    "stage_scores": None,
    "stage_round_no": None,
    "stage_rounds_total": None,
    "in_tiebreak": None,
}


def _mapped(document, locator, key):
    if isinstance(locator, dict) and set(locator) == {"value"}:
        return locator["value"]
    if not isinstance(locator, str):
        raise SnapshotError(f"unverified mapping: {key}")
    value = document
    try:
        for part in locator.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError):
        raise SnapshotError(f"missing mapped field: {key}") from None
    return value


def _tile(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SnapshotError("wire tiles must use official tile strings")
    return tile_id(value)


class SnapshotAdapter:
    """Map only the supplied player's wire data, then use the local action rules.

    Complex upstream representations should be normalized by a custom adapter
    implementing this same callable interface, not inferred from hidden state.
    """

    def __init__(self, mapping):
        if set(mapping) != set(MAPPING_TEMPLATE) or any(
            v is None for v in mapping.values()
        ):
            raise SnapshotError(
                "complete and verify every field in the mapping template before play"
            )
        self.mapping = mapping

    def __call__(self, document):
        s = {k: _mapped(document, locator, k) for k, locator in self.mapping.items()}
        seat = s["seat"]
        if type(seat) is not int or not 0 <= seat < 4:
            raise SnapshotError("snapshot has no player seat")
        phase = s["phase"]
        if not (
            phase == "draw"
            and s["turn"] == seat
            or phase in ("response_peng", "response_chi")
            and seat in s["responding_seats"]
        ):
            return None
        for key in ("melds", "discards", "hand_sizes", "scores"):
            if not isinstance(s[key], list) or len(s[key]) != 4:
                raise SnapshotError(f"{key} must contain four seats")
        if s["stage_scores"] is not None and len(s["stage_scores"]) != 4:
            raise SnapshotError(
                "stage_scores must align with current four player identities"
            )
        for key in ("catch_play", "kong_draw", "you_cai_bi_kao", "in_tiebreak"):
            if type(s[key]) is not bool:
                raise SnapshotError(f"{key} must be boolean")
        owner = s["god_discarder_seat"]
        if (
            type(owner) is not int
            or owner not in range(-1, 4)
            or s["catch_play"] != (owner >= 0)
        ):
            raise SnapshotError("invalid god_discarder_seat for catch_play")
        for key in (
            "dealer",
            "round_no",
            "rounds_total",
            "wall_remaining",
            "chain_count",
            "piao_count",
        ):
            if type(s[key]) is not int or s[key] < 0:
                raise SnapshotError(f"invalid {key}")
        if s["dealer"] > 3 or s["round_no"] < 1:
            raise SnapshotError("invalid dealer/hand number")
        melds = tuple(
            tuple(
                Meld(
                    m["kind"],
                    tuple(_tile(t) for t in m["tiles"]),
                    m.get("from_seat"),
                    m.get("concealed", False),
                )
                for m in row
            )
            for row in s["melds"]
        )
        if any(m.kind not in ("chi", "peng", "gang") for row in melds for m in row):
            raise SnapshotError("unknown meld kind; provide a verified normalizer")
        events = tuple(
            PublicEvent(
                e["kind"],
                e.get("seat"),
                _tile(e.get("tile")),
                tuple(_tile(t) for t in e.get("tiles", [])),
                e["hand_no"],
                tuple(e.get("scores", [])),
                concealed=e.get("concealed", False),
            )
            for e in s["public_events"]
        )
        events = policy_events(events, seat)
        origins = meld_origins(events, s["round_no"])
        from dataclasses import replace

        melds = tuple(
            tuple(
                replace(
                    m,
                    tiles=tuple(sorted(m.tiles)),
                    from_seat=(
                        m.from_seat
                        if m.from_seat is not None
                        else origins.get((player, m.kind, tuple(sorted(m.tiles))))
                    ),
                )
                for m in row
            )
            for player, row in enumerate(melds)
        )
        config = RoundConfig(
            rounds=s["rounds_total"],
            base_score=s["base_score"],
            stage=s["stage_kind"],
            you_cai_bi_kao=s["you_cai_bi_kao"],
        )
        # This is a legal-action projection, never a simulated unknown tile wall.
        view = HandState.__new__(HandState)
        view.config, view.settled, view.turn, view.phase = config, False, seat, phase
        view.hands = [[] for _ in range(4)]
        view.hands[seat] = [_tile(t) for t in s["hand"]]
        view.melds, view.drawn = melds, _tile(s["drawn_tile"])
        view.chain, view.piao = [0] * 4, [0] * 4
        view.chain[seat], view.piao[seat] = s["chain_count"], s["piao_count"]
        view.kong_draw = s["kong_draw"]
        view.catch_owner = owner if s["catch_play"] else None
        view.wall = range(s["wall_remaining"])  # only len() is read by legal_actions
        view.discarded = _tile(s["discarded_tile"])
        if phase in ("response_peng", "response_chi") and view.discarded is None:
            raise SnapshotError("response window requires the actual discarded tile")
        if (
            view.drawn is not None
            and phase == "draw"
            and view.drawn not in view.hands[seat]
        ):
            raise SnapshotError("drawn tile missing from own hand")
        legal = view.legal_actions()
        if not legal:
            raise SnapshotError("no legal action in an actionable state")
        fan = view.current_fan()
        ctx = StageContext(
            StageKind(s["stage_kind"]),
            seat,
            tuple(s["scores"]),
            s["round_no"],
            s["rounds_total"],
            s["dealer"],
            s["stage_no"],
            tuple(s["stage_scores"]) if s["stage_scores"] is not None else None,
            in_tiebreak=s["in_tiebreak"],
            stage_round_no=s["stage_round_no"],
            stage_rounds_total=s["stage_rounds_total"],
        )
        return Observation(
            ctx,
            tuple(sorted(view.hands[seat])),
            events,
            legal,
            melds,
            tuple(tuple(_tile(t) for t in row) for row in s["discards"]),
            tuple(s["hand_sizes"]),
            s["wall_remaining"],
            phase,
            view.drawn,
            s["catch_play"],
            s["chain_count"],
            s["piao_count"],
            fan.fan if fan else 0,
            s["you_cai_bi_kao"],
            god_discarder_seat=owner,
            base_score=config.base_score,
        )


def wire_action(action):
    result = action.as_dict()
    result.pop(
        "kind", None
    )  # official API infers the kong kind; this key is local only
    return result


DIAG_COUNTERS = (
    "frames",
    "fetch_high",
    "fetch_medium",
    "fetch_low",
    "deadline_checks",
    "count_skips",
    "wait_timeouts",
    "key_seen",
    "obs_none",
    "action_unavailable",
    "action_retries",
    "adapter_retries",
    "rejected",
    "timeline_skips",
    "timeline_fetches",
    "throttled",
    "history_fetches",
    "policy_restarts",
)
# A sandboxed candidate closes its process on an error; it is started again this
# many times per game at most, so a candidate failing on every decision still ends.
MAX_POLICY_RESTARTS = 5
# Seconds between two warm-up connections, so the warm-up is not itself a burst.
WARM_SPACING = 0.03
# Seconds between room polls while the deal may come any moment (from START_LEAD
# seconds before the scheduled start; an auto room starts when its fourth seat
# fills).  The deal opens the dealer's 3 s discard window on every table at once
# and the client only learns of it on its next poll.
START_POLL = 0.25
START_LEAD = 10.0
# Seconds after the peng deadline at which the silent chi window is looked for.
CHI_CHECK_DELAYS = (0.12, 0.4)
# Seconds a lone pre-deadline frame waits for the discard that follows a peng.
CLAIM_FOLLOWUP = 0.08


class ActionTimeline:
    """What the server does after our own action, from frame seqs and times only.

    Measured on the live platform (2026-09-17/18): a discard opens a 1 s peng
    window in which only answers are recorded; a peng resolves it at once and
    the claimer discards right after.  Otherwise, at the deadline the server
    records one timeout per responder that stayed silent, opens the next
    seat's chi window (no event) and, one second later, lets the next seat
    draw (an event) and discard.  Between two discards there are therefore
    three peng answers, one chi answer and one draw, or three answers and a
    chi, or up to one answer and a peng.  Knowing our own event's seq and the
    number of frames before the deadline is enough to tell the timeout burst
    from a claim and the next seat's draw from its discard.
    """

    PRE_MARGIN = 0.1  # seconds before the local deadline that still count as "before"
    BURST_END = 0.25  # seconds after the local deadline by which the timeout burst is complete
    DRAW_EARLY = 0.2  # a frame this long before the timed draw cannot be it

    def __init__(self, base_seq, window_end, others, melds, round_no):
        self.base = base_seq  # seq of the snapshot we acted on; our event is base + 1
        self.window_end = window_end  # local time the peng window closes
        self.others = others  # responders other than us
        self.melds = melds  # meld count when we acted; a change means a claim happened
        self.round_no = round_no
        self.pre_watermark = None
        self.burst_decided = False
        self.claim_pending = False  # a skipped frame that may be a claim; its discard follows
        self.pre_answer_skipped = False
        self.answer_skipped = False  # the next seat's chi-window answer was seen
        self.draw_skipped = False
        self.disabled = False

    @property
    def draw_at(self):
        return self.window_end + 1.0

    def stale(self, snap):
        """A fetched state that no longer follows this timeline disables it."""
        melds = sum(len(row) for row in snap.get("melds") or [])
        return snap.get("round_no") != self.round_no or melds != self.melds

    def sync(self, snap, next_seat):
        """Align with a state fetched meanwhile so no step is counted twice."""
        phase, turn = snap.get("phase"), snap.get("turn")
        counts = snap.get("hand_counts") or []
        if phase == "response_chi":
            self.burst_decided = True  # the peng window is over
        elif phase == "draw" and turn == next_seat:
            self.burst_decided = self.answer_skipped = True
            if counts and counts[turn] % 3 == 2:
                self.draw_skipped = True  # already drawn: the discard is next
        elif phase != "response_peng":
            self.disabled = True

    def decide(self, now, seq, prev_seq):
        """Return "fetch", "quiet", "followup" (wait CLAIM_FOLLOWUP, then ask
        again) or ("defer", wake_at) (quiet until then)."""
        jump = seq - prev_seq
        end = self.window_end
        if jump > 0 and self.claim_pending:
            # The frame we skipped may have been a claim rather than an answer,
            # and what follows a claim is the claimer's discard: our window.
            self.claim_pending = False
            return "fetch"
        if now < end - self.PRE_MARGIN:
            # Answers only, unless a peng: its discard follows within a moment.
            if jump <= 0:
                return "quiet"
            if jump == 1 and not self.pre_answer_skipped:
                self.claim_pending = True  # a peng looks like this too
                return "followup"
            if jump == 1 and self.pre_answer_skipped:
                return "fetch"  # a second lone frame may be a slow claimer's discard
            return "fetch"
        if not self.burst_decided:
            if self.pre_watermark is None:
                self.pre_watermark = prev_seq if jump > 0 else seq
            if now < end + self.BURST_END:
                return ("defer", end + self.BURST_END)
            self.burst_decided = True
            answered = max(0, self.pre_watermark - self.base - 1)
            expected_timeouts = max(0, self.others - answered)
            burst = seq - self.pre_watermark
            if burst and burst != expected_timeouts:
                # More frames than silent responders means a claim (and its
                # discard) rode along; fewer, but not none, means a peng
                # resolved the window instead of the timeouts. Either way the
                # claimer's discard is next. No frame at all is the ordinary
                # case of responders that had already answered before us.
                return "fetch"
            return "quiet"
        if jump <= 0:
            return "quiet"
        if now < self.draw_at - self.DRAW_EARLY:
            # The next seat's chi-window answer: a pass needs nothing, a chi is
            # followed by its discard. Skipping costs nothing as long as the
            # frame after it is looked at, which claim_pending arranges.
            if jump == 1 and not self.answer_skipped:
                self.answer_skipped = self.claim_pending = True
                return "quiet"
            return "fetch"
        if not self.draw_skipped and (jump == 1 or (jump == 2 and not self.answer_skipped)):
            # The next seat's draw on its timer, alone or together with the
            # chi-window timeout that precedes it.
            self.draw_skipped = self.answer_skipped = True
            return "quiet"
        return "fetch"  # the discard that opens a window
# Seconds between event-history refreshes when nothing else forces one.
HISTORY_INTERVAL = 8.0


def _policy_stats(policy):
    """Diagnostics a sandboxed policy keeps in memory; empty for local ones."""
    stats = getattr(policy, "stats", None)
    if not isinstance(stats, dict):
        return {}
    result = dict(stats)
    latencies = sorted(getattr(policy, "latencies", []) or [])
    if latencies:
        result["latency_ms"] = {
            "p50": latencies[len(latencies) // 2] * 1000,
            "p99": latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))] * 1000,
            "max": latencies[-1] * 1000,
        }
    error = getattr(policy, "last_error", None)
    if error:
        result["last_error"] = {
            "type": error.get("type"),
            "message": (error.get("message") or "")[:500],
            "stderr": (error.get("stderr") or "")[-1000:],
        }
    return result


class TournamentRunner:
    def __init__(self, client, policy, adapter, output, workers=16, join=False):
        if not 1 <= workers <= 16:
            raise ValueError("workers must be 1..16")
        self.client, self.policy, self.adapter = client, policy, adapter
        # identity hashes the frozen package and runtime trees (35 ms; half a
        # second per table when ten tables compute it at the deal): once per run.
        self.identity = policy.identity
        self.root, self.workers, self.join = Path(output), workers, join
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.tournament = None
        self.discovered = {}  # game id -> time the room poll first listed it

    def _warm_workers(self, pool):
        """Open every worker thread's connection before the deal.

        新连接要一次 TLS 握手，十桌同时开局会把一批握手和首批全量快照挤在一起；
        预热放在开桌之前做，此时没有窗口等着，握手贵一点也无所谓。
        """
        ready = threading.Barrier(self.workers, timeout=30)

        def warm(index):
            self.stop.wait(WARM_SPACING * index)
            try:
                self.client.request("GET", "/api/me", priority=0)
            except (ApiError, ConnectionError, threading.BrokenBarrierError):
                pass  # 预热失败不影响对局，首次拉取会自己建连
            try:
                ready.wait()  # 占住线程，保证每个 worker 各拿到一条连接
            except threading.BrokenBarrierError:
                pass

        for index in range(self.workers):
            pool.submit(warm, index)

    def _play(self, gid):
        # A sandboxed policy starts a process first (0.1-0.5 s when ten tables
        # are dealt together); it starts beside the first fetch, not before it.
        starting = Future()

        def start_policy():
            try:
                built = self.policy.build()
                built.reset(derive_seed(gid, self.identity))
            except BaseException as exc:
                starting.set_exception(exc)
            else:
                starting.set_result(built)

        threading.Thread(target=start_policy, name=f"policy-{gid}", daemon=True).start()
        policy = None

        def ready_policy():
            nonlocal policy
            if policy is None:
                policy = starting.result()  # a failed start fails this table
            return policy

        path = self.root / ("game-" + digest(gid)[:24] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = (
            read_json(path)
            if path.exists()
            else {"responses": [], "status": "running", "events": []}
        )
        if state.get("policy_id", self.identity) != self.identity:
            raise ValueError("cannot switch policy while resuming a game")
        state.update(policy_id=self.identity, game_id=gid)
        seen = set(state["responses"])
        seq = state.get("event_seq", 0)
        events = state.setdefault("events", [])
        event_ids = {e["seq"] for e in events}
        hand_no = state.get("round_no", 1)
        force_snapshot = False
        diag = state.setdefault("client", {})
        for name in DIAG_COUNTERS:
            diag.setdefault(name, 0)
        if gid in self.discovered:
            diag.setdefault("discovered_at", round(self.discovered[gid], 3))

        def ingest(payload):
            nonlocal seq, hand_no
            for event in payload.get("events") or []:
                if event["seq"] not in event_ids:
                    for i in range(len(events) - 1, -1, -1):
                        local = events[i]
                        if (
                            local.get("reconstructed")
                            and local["seq"] != 1
                            and local["type"] == event.get("type")
                            and local.get("seat") == event.get("seat")
                            and local.get("tile", "") == (event.get("tile") or "")
                        ):
                            del events[i]
                            break
                    event = dict(event)
                    event["_round_no"] = (event.get("data") or {}).get(
                        "round_no", hand_no
                    )
                    events.append(event)
                    event_ids.add(event["seq"])
                    if event["type"] == "round_ended":
                        hand_no = event["_round_no"] + 1
                seq = max(seq, event["seq"])

        # Server-sent seq frames wake the loop; every fetch is still a full
        # seq=0 snapshot because a fully answered peng window can become chi
        # without emitting any event.  Frames only tell us *when* to look.
        wake = threading.Event()
        frames = {"seq": 0, "closed": False, "count": 0}
        notify_stop = threading.Event()
        notify_thread = None
        if callable(getattr(self.client, "notify", None)):

            def on_frame(frame_seq, closed):
                frames["seq"] = max(frames["seq"], frame_seq)
                frames["count"] += 1
                diag["frames"] += 1
                if closed:
                    frames["closed"] = True
                wake.set()

            def follow():
                if not self.client.notify(f"/api/games/{_id(gid)}/notify", on_frame, notify_stop):
                    frames["unavailable"] = True
                wake.set()

            notify_thread = threading.Thread(target=follow, name=f"notify-{gid}", daemon=True)
            notify_thread.start()
        accepts_priority = "priority" in inspect.signature(self.client.request).parameters
        levels = {"low": 0, "medium": 1, "high": 2}
        last_snap = None
        last_key = None  # dedupe key of last_snap when it asked us to act
        fetched_seq = 0
        skip_until = 0  # absolute seq below which no frame can need our action
        settled_since = None
        last_history = 0.0
        last_round = None
        adapter_retries = 0
        force_snapshot = True
        # Set by our own accepted action: the key we acted on, the peng deadline
        # at which our chi window can open silently (next seat only), how many
        # deadline checks were spent, and when the next seat's draw is due.
        acted_key = None
        chi_deadline_ms = None
        deadline_checks = 0
        timeline = None  # ActionTimeline after our own pass or discard
        timeline_discarder = 0
        window_end = None  # local end of the response window our action belongs to
        offsets = []  # server clock minus this clock, from our own discards
        own_discard_at = None
        own_discard_round = None
        retries = {}
        woke_at = None
        defer_until = None
        why = "start"  # what triggered the current fetch, for the fetch log
        last_jump = 0

        def to_local(server_seconds):
            if not offsets:
                return server_seconds
            return server_seconds - sorted(offsets)[len(offsets) // 2]
        fetch_log = path.with_suffix(".fetches")  # not *.jsonl: audits read those as actions

        def log_fetch(outcome, priority, res, snap, fetch_ms, now):
            try:
                with fetch_log.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "t": round(now, 3),
                                "seq": res.get("seq"),
                                "why": why,
                                "jump": last_jump,
                                "off": round(sorted(offsets)[len(offsets) // 2], 3) if offsets else None,
                                "prio": priority,
                                "phase": (snap or {}).get("phase"),
                                "turn": (snap or {}).get("turn"),
                                "ms": round(fetch_ms, 1),
                                "wait_ms": getattr(getattr(self.client, "local", None), "wait_ms", None),
                                "rel_end": round(now - window_end, 3) if window_end else None,
                                "throttled": getattr(getattr(self.client, "local", None), "throttled", 0),
                                "outcome": outcome,
                            }
                        )
                        + "\n"
                    )
            except OSError:
                pass
        try:
          while not self.stop.is_set():
            if notify_thread is not None and not frames.get("unavailable"):
                priority = None
                if force_snapshot:
                    # A forced look has nothing to wait for: not the stream's
                    # first frame on a newly dealt table (its connection is still
                    # being set up while the dealer's window runs), nor the next
                    # frame after a claim, a rejection or a retry.
                    wake.clear()
                    woke_at = time.perf_counter()
                    priority, why = "high", "force"
                    last_jump = max(0, frames["seq"] - fetched_seq)
                while priority is None and not self.stop.is_set():
                    acted = last_key is not None and last_key == acted_key
                    settled = last_snap is not None and last_snap.get("phase") == "settled"
                    timeout = 1.0 if settled or last_snap is None else 2.0
                    check_at = None
                    if (
                        acted
                        and chi_deadline_ms
                        and deadline_checks < 2
                        and last_snap.get("phase") == "response_peng"
                    ):
                        # Our chi window opens at the peng deadline without any
                        # event.  Look then, and once more a little later in
                        # case this clock runs ahead of the server's.
                        check_at = to_local(chi_deadline_ms / 1000) + (
                            CHI_CHECK_DELAYS[deadline_checks]
                        )
                        timeout = max(0.0, min(timeout, check_at - time.time()))
                    if defer_until is not None:
                        timeout = max(0.0, min(timeout, defer_until - time.time()))
                    prev_seq = frames["seq"]
                    timed_out = not wake.wait(timeout=timeout)
                    wake.clear()
                    woke_at = time.perf_counter()
                    last_jump = frames["seq"] - prev_seq
                    if frames.get("unavailable"):
                        priority = "high"
                        why = "no-notify"
                        break
                    if check_at is not None and time.time() >= check_at - 0.005:
                        deadline_checks += 1
                        diag["deadline_checks"] += 1
                        priority = "high"
                        why = "chi-check"
                        break
                    if timed_out and check_at is not None and time.time() < check_at - 0.005:
                        if defer_until is None or time.time() < defer_until - 0.005:
                            continue  # woke early for a deadline check not due yet
                    deferred_due = defer_until is not None and time.time() >= defer_until - 0.005
                    defer_until = None
                    if timed_out and not deferred_due:
                        diag["wait_timeouts"] += 1
                    now = time.time()
                    quiet = frames["seq"] < skip_until and not deferred_due
                    verdict = None
                    if quiet and not timed_out:
                        diag["count_skips"] += 1
                    elif quiet:
                        pass
                    elif force_snapshot or frames["closed"] or (timed_out and not deferred_due):
                        pass
                    elif timeline is not None and not timeline.disabled and chi_deadline_ms is None:
                        verdict = timeline.decide(now, frames["seq"], prev_seq)
                        if verdict == "followup":
                            # A peng is followed at once by the claimer's discard:
                            # give the second frame a moment before deciding.
                            before = frames["seq"]
                            wake.wait(timeout=CLAIM_FOLLOWUP)
                            wake.clear()
                            if frames["seq"] == before:
                                timeline.pre_answer_skipped = True
                                verdict = "quiet"
                            else:
                                verdict = "fetch"
                        if isinstance(verdict, tuple):
                            defer_until = verdict[1]
                            verdict = "quiet"
                            quiet = True  # look again when the burst is complete
                        elif verdict == "quiet":
                            diag["timeline_skips"] += 1
                            skip_until = frames["seq"] + 1
                            quiet = True
                        elif verdict == "fetch":
                            diag["timeline_fetches"] += 1
                    elif (
                        timeline is not None
                        and not timeline.disabled
                        and chi_deadline_ms
                        and now < timeline.window_end - ActionTimeline.PRE_MARGIN
                    ):
                        # We are the next seat: before the deadline only a claim
                        # can need us; from the deadline on every frame may be
                        # our chi window or our own draw and is looked at.
                        verdict = timeline.decide(now, frames["seq"], prev_seq)
                        if verdict == "followup":
                            before = frames["seq"]
                            wake.wait(timeout=CLAIM_FOLLOWUP)
                            wake.clear()
                            verdict = "quiet" if frames["seq"] == before else "fetch"
                            if verdict == "quiet":
                                timeline.pre_answer_skipped = True
                        if isinstance(verdict, tuple):
                            defer_until = verdict[1]
                            quiet = True
                        elif verdict == "quiet":
                            skip_until = frames["seq"] + 1
                            quiet = True
                            diag["timeline_skips"] += 1
                    if timed_out and deferred_due:
                        timed_out = False  # a planned look, not an idle poll
                    priority = self._fetch_plan(
                        last_snap,
                        frames["seq"] - fetched_seq,
                        timed_out,
                        force_snapshot or frames["closed"],
                        quiet,
                        settled_since,
                        now_ms=time.time() * 1000,
                        acted=acted,
                    )
                    if verdict == "fetch" and priority is None:
                        priority = "medium"
                    if priority is not None:
                        why = (
                            "force" if force_snapshot or frames["closed"]
                            else "idle" if timed_out
                            else "timeline" if verdict == "fetch"
                            else "plan"
                        )
                if self.stop.is_set():
                    break
            else:
                priority = "high"
                why = "poll"
                woke_at = time.perf_counter()
            cursor = 0
            force_snapshot = False
            fetch_began = time.perf_counter()
            res = self.client.request(
                "GET", f"/api/games/{_id(gid)}/state?seq={cursor}", **({"priority": levels.get(priority, 2)} if accepts_priority else {})
            )
            fetch_ms = (time.perf_counter() - fetch_began) * 1000
            diag["fetch_" + (priority if priority in levels else "high")] += 1
            diag["throttled"] += getattr(getattr(self.client, "local", None), "throttled", 0)
            fetched_seq = max(fetched_seq, int(res.get("seq", 0) or 0), frames["seq"] if res.get("finished") else 0)
            skip_until = 0
            last_key = None
            ingest(res)
            if res.get("finished"):
                log_fetch("finished", priority, res, res.get("snapshot"), fetch_ms, time.time())
                diag["policy"] = _policy_stats(ready_policy())
                diag["clock_offset_s"] = round(sorted(offsets)[len(offsets) // 2], 3) if offsets else None
                state.update(status="finished", final=_redact(res), event_seq=seq)
                atomic_json(path, state)
                return
            if res.get("pending"):
                log_fetch("pending", priority, res, None, fetch_ms, time.time())
                continue
            if res.get("snapshot") is None:
                res = self.client.request("GET", f"/api/games/{_id(gid)}/state?seq=0")
            snap = res.get("snapshot")
            if snap is None:
                raise SnapshotError("full state response lacks snapshot")
            last_snap = snap
            if (
                own_discard_at is not None
                and own_discard_round == snap.get("round_no")
                and snap.get("turn") == snap.get("seat")
                and snap.get("phase") in ("response_peng", "response_chi")
                and snap.get("window_deadline_ms")
            ):
                # The window our discard opened ends 1 s (chi: 2 s) after the
                # discard: its deadline calibrates the server clock offset.
                span = 1.0 if snap["phase"] == "response_peng" else 2.0
                sample = snap["window_deadline_ms"] / 1000 - span - own_discard_at
                if -3.0 < sample < 3.0:
                    offsets.append(sample)
                    del offsets[:-7]
            if timeline is not None and not timeline.disabled:
                if timeline.stale(snap):
                    timeline.disabled = True
                else:
                    timeline.sync(snap, (timeline_discarder + 1) % 4)
            if snap.get("phase") == "settled":
                settled_since = settled_since or time.monotonic()
            else:
                settled_since = None
            hand_no = snap["round_no"]
            # seq=0 is the server's full-snapshot sentinel. Preserve subsequent
            # events instead of repeatedly discarding them after each action.
            if seq == 0 and res.get("seq", 0) > 0:
                seq = 1
                # First event is recoverable as the initial dealer discard when
                # the current river still contains it. Otherwise flag the gap.
                river = snap["discards"][snap["dealer"]]
                if river and 1 not in event_ids:
                    events.append(
                        {
                            "seq": 1,
                            "type": "tile_discarded",
                            "seat": snap["dealer"],
                            "tile": river[0],
                            "data": {},
                            "_round_no": hand_no,
                            "reconstructed": True,
                        }
                    )
                    event_ids.add(1)
                else:
                    state["initial_history_gap"] = True
            # The event history feeds the policy's public chronology; it is
            # refreshed at most every few seconds so most decisions cost one
            # request.  It shares the fetch priority: a low-priority wait here
            # would hold up this table's own decision.
            own_actions = sum(
                1
                for e in events
                if e.get("seat") == snap.get("seat")
                and e.get("_round_no", snap["round_no"]) == snap["round_no"]
                and e.get("type") in ("tile_discarded", "gang")
            )
            # A decision of ours must see every event up to it, as in the engine:
            # the policy reads who discarded, claimed or drew from this history,
            # so a response to a discard without that discard misjudges the pass.
            phase = snap.get("phase") or ""
            deciding = (
                snap.get("seat") in (snap.get("responding_seats") or ())
                if phase.startswith("response")
                else snap.get("turn") == snap.get("seat") and phase not in ("settled", "finished")
            )
            history_due = (
                deciding
                or notify_thread is None
                or frames.get("unavailable")
                or snap.get("phase") in ("settled", "finished")
                or snap["round_no"] != last_round
                or own_actions < int((snap.get("god") or {}).get("chain_count") or 0)
                or time.monotonic() - last_history >= HISTORY_INTERVAL
            )
            last_round = snap["round_no"]
            if res.get("seq", 0) > seq and history_due:
                last_history = time.monotonic()
                diag["history_fetches"] += 1
                delta = self.client.request(
                    "GET", f"/api/games/{_id(gid)}/state?seq={seq}", **({"priority": levels.get(priority, 2)} if accepts_priority else {})
                )
                ingest(delta)
                if delta.get("snapshot"):
                    res = delta
                    snap = delta["snapshot"]
                    seq = max(seq, delta.get("seq", seq))
                elif delta.get("seq", 0) > res.get("seq", 0):
                    res = self.client.request(
                        "GET", f"/api/games/{_id(gid)}/state?seq=0", **({"priority": levels.get(priority, 2)} if accepts_priority else {})
                    )
                    if res.get("snapshot"):
                        snap = res["snapshot"]
            # A history refresh can advance the decision (even the hand).
            # Keep its snapshot, logged sequence and timeline watermark from
            # the same response; an old envelope mislabels otherwise valid play.
            fetched_seq = max(fetched_seq, int(res.get("seq", 0) or 0))
            hand_no = snap["round_no"]
            last_snap = snap
            with self.lock:
                tournament = self.tournament
            try:
                obs = self.adapter(
                    {
                        "snapshot": snap,
                        "tournament": tournament,
                        "response": res,
                        "wire_events": sorted(events, key=lambda e: e["seq"]),
                    }
                )
            except SnapshotError:
                if adapter_retries >= 3:
                    raise
                # Usually a stale event history; refresh everything and look again.
                adapter_retries += 1
                diag["adapter_retries"] += 1
                last_history = 0.0
                force_snapshot = True
                time.sleep(0.05)
                continue
            adapter_retries = 0
            state.update(event_seq=seq, round_no=hand_no)
            if obs is None:
                diag["obs_none"] += 1
                log_fetch("none", priority, res, snap, fetch_ms, time.time())
                continue
            key = digest(
                {
                    "window": snap.get("window_deadline_ms"),
                    "hand": obs.stage.round_no,
                    "phase": obs.phase,
                    "turn": snap["turn"],
                    "melds": snap["melds"],
                    "discards": obs.discards,
                    "own_hand": obs.hand,
                    "drawn": obs.drawn_tile,
                    "wall": obs.wall_remaining,
                }
            )
            last_key = key
            if key in seen:
                diag["key_seen"] += 1
                log_fetch("seen", priority, res, snap, fetch_ms, time.time())
                continue
            log_fetch("act", priority, res, snap, fetch_ms, time.time())
            ready_policy()
            begin = time.perf_counter()
            from mahjong_ai.execution.candidate_runtime import (
                ActionUnavailable,
                RemotePolicy,
            )

            if isinstance(policy, RemotePolicy) and snap.get("window_deadline_ms"):
                remaining = max(0.001, snap["window_deadline_ms"] / 1000 - time.time())
                policy.action_seconds = min(self.policy.action_seconds, remaining)
                policy.response_seconds = min(self.policy.response_seconds, remaining)
            try:
                action = policy.act(obs)
            except ActionUnavailable:
                if getattr(policy, "worker", False) is None and diag["policy_restarts"] < MAX_POLICY_RESTARTS:
                    # Its process was closed on an error: without a new one every
                    # later decision of this game would fall to the server's timeout.
                    diag["policy_restarts"] += 1
                    diag.setdefault("policy_errors", []).append(getattr(policy, "last_error", None))
                    policy = self.policy.build()
                    policy.reset(derive_seed(gid, self.identity))
                # The server owns timeout fallback.  Never submit a stale
                # action, but look again while the window leaves time to retry.
                diag["action_unavailable"] += 1
                left = (snap.get("window_deadline_ms") or 0) / 1000 - time.time()
                if left >= 0.4 and retries.get(key, 0) < 2:
                    retries[key] = retries.get(key, 0) + 1
                    diag["action_retries"] += 1
                else:
                    seen.add(key)
                force_snapshot = True
                continue
            decision_ms = (time.perf_counter() - begin) * 1000
            if action not in obs.legal_actions:
                raise SnapshotError("policy returned an illegal action")
            seen.add(key)
            state["responses"].append(key)
            atomic_json(path, state)
            record = {
                "game_id": gid,
                "seq": res.get("seq", seq),
                "snapshot": snap,
                "action": wire_action(action),
                "decision_ms": decision_ms,
                "request_ms": None,
                "accepted": False,
                "ts": time.time(),
                "priority": priority,
                "fetch_ms": fetch_ms,
                "limiter_wait_ms": getattr(getattr(self.client, "local", None), "wait_ms", None),
                "wake_to_post_ms": (time.perf_counter() - woke_at) * 1000 if woke_at else None,
                "window_left_ms": (
                    snap["window_deadline_ms"] - time.time() * 1000
                    if snap.get("window_deadline_ms")
                    else None
                ),
            }
            begin = time.perf_counter()
            try:
                response = self.client.request(
                    "POST", f"/api/games/{_id(gid)}/action", wire_action(action)
                )
                record.update(accepted=True, response=response)
                diag["throttled"] += getattr(getattr(self.client, "local", None), "throttled", 0)
                # Predict how many events must pass before we can be needed
                # again, so frames in between do not spend /state requests.
                acted_key = key
                chi_deadline_ms = None
                deadline_checks = 0
                timeline = None
                window_end = None
                deadline_ms = snap.get("window_deadline_ms") or 0
                melds_now = sum(len(row) for row in snap.get("melds") or [])
                if action.action == "discard":
                    # Our discard opens a 1 s peng window; see ActionTimeline.
                    skip_until = fetched_seq + 2
                    own_discard_at = time.time()
                    own_discard_round = snap.get("round_no")
                    window_end = own_discard_at + 1.0
                    timeline_discarder = obs.stage.seat
                    timeline = ActionTimeline(fetched_seq, window_end, 3, melds_now, snap.get("round_no"))
                elif action.action == "pass" and obs.phase == "response_peng":
                    skip_until = fetched_seq + 2
                    window_end = to_local(deadline_ms / 1000) if deadline_ms else None
                    if obs.stage.seat == (snap["turn"] + 1) % 4:
                        # Our chi window opens at the peng deadline without any
                        # event; the wait loop looks for it then.
                        chi_deadline_ms = deadline_ms or None
                    if window_end:
                        others = max(0, len(snap.get("responding_seats") or [1, 2, 3]) - 1)
                        timeline_discarder = snap["turn"]
                        timeline = ActionTimeline(fetched_seq, window_end, others, melds_now, snap.get("round_no"))
                elif action.action == "pass" and obs.phase == "response_chi":
                    # We are the next seat: our own pass is one event, our draw
                    # (at the chi deadline) the next.
                    skip_until = fetched_seq + 2
                elif action.action in ("chi", "peng", "gang"):
                    force_snapshot = True  # it is our turn to discard now
                if True:
                    kind = {"discard": "tile_discarded", "hu": "round_ended"}.get(
                        action.action, action.action
                    )
                    events.append(
                        {
                            "seq": 1 if seq == 0 else max(seq, fetched_seq) + 0.5,
                            "type": kind,
                            "seat": obs.stage.seat,
                            "tile": wire_action(action).get("tile", ""),
                            "data": {
                                "kind": {
                                    "concealed": "an",
                                    "exposed": "ming",
                                    "added": "bu",
                                }.get(action.kind, "")
                            },
                            "_round_no": hand_no,
                            "reconstructed": True,
                        }
                    )
                    if seq == 0:
                        event_ids.add(1)
                        seq = 1
            except ApiError as exc:
                record.update(http_status=exc.status, error_code=exc.code)
                if exc.status != 409:
                    raise
                diag["rejected"] += 1
                force_snapshot = True
            except ConnectionError:
                force_snapshot = True
                record["error_code"] = "ACCEPTANCE_UNKNOWN"
            finally:
                record["request_ms"] = (time.perf_counter() - begin) * 1000
                with path.with_suffix(".jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            state["event_seq"] = seq
            atomic_json(path, state)
          diag["policy"] = _policy_stats(ready_policy())
          state["status"] = "paused"
          atomic_json(path, state)
        finally:
            notify_stop.set()
            wake.set()

    @staticmethod
    def _fetch_plan(
        last_snap, delta, timed_out, force, predicted_quiet, settled_since, now_ms=None, acted=False
    ):
        """Decide whether to spend a /state request now and at which priority.

        Returns None (wait for the next frame), "high" (our own draw or
        discard may be pending), "medium" (a claim window may be open for us)
        or "low" (housekeeping poll that must not delay other tables).
        ``acted`` says ``last_snap`` is the state we already answered.
        """
        if force or last_snap is None:
            return "high"
        phase = last_snap.get("phase")
        me = last_snap.get("seat")
        turn = last_snap.get("turn")
        if phase == "settled":
            # Dealing the next hand emits no event; poll during the fixed pause
            # and switch to high priority when the new hand is imminent.
            if not timed_out:
                return None
            waited = time.monotonic() - settled_since if settled_since else 0.0
            return "high" if waited >= 4.0 else "low"
        if timed_out:
            return "low"
        if delta <= 0 or predicted_quiet:
            return None
        if acted:
            # After our chi-window pass our own draw is the next event, and
            # after a peng-window pass as the next seat our chi window opens
            # at the deadline: both are our own turn.  Otherwise any frame
            # past the predicted quiet stretch may be a claimer's discard.
            if phase == "response_chi" or (
                phase == "response_peng" and me == (turn + 1) % 4
            ):
                return "high"
            return "medium"
        responders = last_snap.get("responding_seats") or []
        if phase == "draw":
            if turn == me:
                return "high"  # we should already be acting; resync
            # the seat before us discards next: our chi window, then our draw
            return "high" if turn == (me - 1) % 4 else "medium"
        if phase in ("response_peng", "response_chi"):
            if me in responders:
                return "high" if phase == "response_chi" or me == (turn + 1) % 4 else "medium"
            if phase == "response_chi":
                # The next seat answers; its discard (after a chi) or the
                # following draw is the second event.
                return None if delta < 2 else "medium"
            # A peng resolves the window at once and the claimer's discard is
            # the second event; otherwise nothing before the second event can
            # open a window for us either.
            if delta < 2:
                return None
            return "high" if (turn + 1) % 4 == me else "medium"
        return "high"

    @staticmethod
    def _needs_snapshot(fresh_events, last_snap):
        """Do the newly read events make a full snapshot worth a request now?"""
        me = last_snap.get("seat")
        turn = last_snap.get("turn")
        phase = last_snap.get("phase")
        responders = last_snap.get("responding_seats") or []
        for e in fresh_events:
            kind, seat = e.get("type"), e.get("seat")
            data = e.get("data") or {}
            if kind in ("round_ended", "game_ended", "gap"):
                return True
            if seat == me:
                if kind == "tile_drawn":
                    return True  # our turn to discard
                continue  # echoes of our own actions
            if kind == "tile_discarded":
                return True  # a claim window may be open for us
            if kind == "timeout" and data.get("kind") == "discard":
                return True  # server discarded for them; window may follow
            if kind in ("pass", "timeout") and phase == "response_peng" and me == (turn + 1) % 4 and me not in responders:
                return True  # peng window resolving into our chi window, no event for that
            if kind not in ("pass", "timeout", "chi", "peng", "gang", "tile_drawn"):
                return True  # unknown event: look
        return False

    def run(self, tournament_id=None, seconds=300):
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        deadline = time.monotonic() + seconds
        if seconds <= 0:
            raise ValueError("positive online time budget required")
        tid, scoped, _ = self.client.discover(tournament_id)

        def transient(exc):
            # The tournament endpoint occasionally answers 404 TOURNAMENT_GONE or
            # 5xx while the server reshuffles stages; a live event must outlast
            # that, so only a persistent failure ends the run.
            return (
                isinstance(exc, ConnectionError)
                or exc.status == 429
                or exc.status >= 500
                or (exc.status == 404 and exc.code == "TOURNAMENT_GONE")
            )

        if self.join:
            # Tournament detail is forbidden before registration for global tokens.
            for operation in ("register", "ready"):
                for attempt in range(5):
                    try:
                        self.client.request(
                            "POST", f"/api/tournaments/{_id(tid)}/{operation}"
                        )
                        break
                    except (ApiError, ConnectionError) as exc:
                        if isinstance(exc, ApiError) and exc.status == 409:
                            break
                        if not transient(exc) or attempt == 4:
                            raise
                        time.sleep(2 * (attempt + 1))
        futures, completed = {}, set()
        failing_since = None
        last_join = 0.0
        waiting = True  # no table dealt yet: the next poll is what sees the deal
        accepts_retries = "retries" in inspect.signature(self.client.request).parameters
        with (
            exclusive_lock(self.root / ".online.lock"),
            ThreadPoolExecutor(max_workers=self.workers) as pool,
        ):
            warmed = False
            try:
                while time.monotonic() < deadline:
                    try:
                        t = self.client.request(
                            "GET",
                            f"/api/tournaments/{_id(tid)}",
                            **({"retries": 0} if waiting and accepts_retries else {}),
                        )
                    except (ApiError, ConnectionError) as exc:
                        if not transient(exc):
                            raise
                        failing_since = failing_since or time.monotonic()
                        if time.monotonic() - failing_since > 300:
                            raise
                        # TOURNAMENT_GONE is most common while the room deals: ask
                        # again at once instead of sitting out the 1-2-4 s backoff.
                        gone = getattr(exc, "code", "") == "TOURNAMENT_GONE"
                        self.stop.wait(START_POLL if waiting and gone else 2)
                        continue
                    failing_since = None
                    with self.lock:
                        self.tournament = t
                    # Persist authoritative stage ledger; never sum repeated ranking snapshots.
                    atomic_json(self.root / "tournament.json", _redact(t))
                    status = t["status"]
                    if status in ("finished", "closed", "void"):
                        break
                    if status == "stage_open" and not t.get("qualified"):
                        break
                    if self.join and status in ("registering", "stage_open") and time.monotonic() - last_join >= 30:
                        last_join = time.monotonic()
                        try:
                            if status == "registering":
                                self.client.request(
                                    "POST", f"/api/tournaments/{_id(tid)}/register"
                                )
                            self.client.request(
                                "POST", f"/api/tournaments/{_id(tid)}/ready"
                            )
                        except (ApiError, ConnectionError) as exc:
                            if isinstance(exc, ApiError) and exc.status == 409:
                                pass
                            elif not transient(exc):
                                raise
                    for gid, future in list(futures.items()):
                        if future.done():
                            future.result()  # worker failures stop this run, never disappear
                            completed.add(gid)
                            del futures[gid]
                    if not warmed:
                        # 只在开桌之前预热：此时没有窗口等着，预热占住线程池也无所谓；
                        # 房间已经在打就直接开桌，首次拉取自己建连。
                        warmed = True
                        if status != "running":
                            self._warm_workers(pool)
                    if status == "running":
                        me = self.client.request("GET", "/api/me")
                        games = active_games(me, t, scoped)
                        if len(set(games) - completed) > self.workers:
                            raise RuntimeError(
                                "insufficient independent workers for active games"
                            )
                        for gid in games:
                            if gid not in futures and gid not in completed:
                                self.discovered.setdefault(gid, time.time())
                                futures[gid] = pool.submit(self._play, gid)
                    elif status not in ("registering", "stage_open", "stage_done"):
                        raise SnapshotError("unknown tournament status")
                    start_at = (t.get("config") or {}).get("StartAt")
                    if not isinstance(start_at, (int, float)):
                        start_at = 0
                    waiting = (
                        not futures
                        and not completed
                        and status in ("registering", "running")
                        and time.time() >= start_at - START_LEAD
                    )
                    self.stop.wait(START_POLL if waiting else 1)
            finally:
                self.stop.set()
            for future in futures.values():
                future.result()
        return {
            "status": "stopped",
            "games_finished": len(completed),
            "output": str(self.root),
        }
