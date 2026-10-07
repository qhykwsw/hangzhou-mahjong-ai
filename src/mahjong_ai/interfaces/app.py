"""Local dashboard and one host-managed service per project."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from mahjong_ai.research.service import Research
from mahjong_ai.execution.controller import Controller
from mahjong_ai.persistence.artifacts import atomic_json, exclusive_lock, read_json
from mahjong_ai.interfaces.service_host import open_url, start_service, stop_service
from mahjong_ai.project import ProjectDescriptor
from mahjong_ai.recipes.service import RecipeIteration
from mahjong_ai.recipes.store import PROJECT as RECIPE_PROJECT


def recipe_iteration(project):
    """Recipe iteration shares this service when the workspace has its project."""
    workspace = ProjectDescriptor.load(project).workspace
    if (workspace / RECIPE_PROJECT / "recipes/iteration.toml").exists():
        return RecipeIteration(workspace)
    return None


ONLINE_MIN_ROOMS = 20  # a room's total varies by ~180 points: fewer rooms say little about a version
_online_lock = threading.Lock()


def _free_play_game(path, ours):
    """One online game of ours: [table score, place, hands, an opponent was AFK]."""
    replay = json.loads(path.read_text())
    users = [s["user_id"] for s in replay.get("seats", [])]
    if replay.get("status") != "finished" or ours not in users:
        return None
    me = users.index(ours)
    total, hands = [0] * 4, 0
    for r in replay.get("rounds", []):
        if r.get("scores"):
            hands += 1
            total = [a + b for a, b in zip(total, r["scores"])]
    if not hands:
        return None
    discards, timeouts = [0] * 4, [0] * 4
    for block in replay.get("blocks", []):
        for e in block.get("events", []):
            if e.get("type") == "tile_discarded":
                discards[e["seat"]] += 1
            elif e.get("type") == "timeout" and (e.get("data") or {}).get("kind") == "discard":
                timeouts[e["seat"]] += 1
    # An AFK opponent lets every timeout discard for it: half or more of its discards.
    afk = any(discards[i] and timeouts[i] / discards[i] >= 0.5 for i in range(4) if i != me)
    place = 1 + sum(t > total[me] for t in total) + (sum(t == total[me] for t in total) - 1) / 2
    return [total[me], place, hands, afk]


def free_play_versions(competition):
    """Each version's free-play record, from the strongest-entrant runner's rooms (read-only view).

    The runner's log names each room's version. A finished room's games are read
    from its replays once and kept in ``online-rooms.jsonl``. The readout is given
    without the games where an opponent was AFK and over all games (``all_``),
    each with an SE clustered by room. Versions
    with fewer than ONLINE_MIN_ROOMS rooms are listed but not scored.
    """
    log = competition / "free-play-strongest.log"
    if not log.exists():
        return {"enabled": False}
    rooms, numbered, ranks = {}, {}, {}
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\S+ \S+) 第 (\d+) 房开始：(\S+) → (\S+)", line)
        if m:
            started = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            rooms[m.group(4)], numbered[m.group(2)] = (m.group(3), started), m.group(4)
            continue
        m = re.match(r"\S+ \S+ 第 (\d+) 房结束：completed 名次 (\S+)", line)
        if m and m.group(1) in numbered:
            ranks[numbered[m.group(1)]] = float(m.group(2))
    with _online_lock:
        path = competition / "online-rooms.jsonl"
        cache = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    cache[r["room"]] = r
        new = []
        for room in rooms:
            folder = competition / room
            if room in cache or not (folder / "summary.json").exists():  # playing, or never started
                continue
            match = folder / "match.json"
            ours = json.loads(match.read_text()).get("user_id") if match.exists() else None
            games = [g for p in sorted((folder / "replays").glob("*.json")) if (g := _free_play_game(p, ours))]
            cache[room] = {"room": room, "games": games}
            new.append(cache[room])
        if new:
            with path.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
    versions = {}
    for room, (version, started) in rooms.items():
        games = (cache.get(room) or {}).get("games")
        if games:
            versions.setdefault(version, []).append((room, started, games))
    turns = competition / "entrant.rotation"
    pin = competition / "entrant.pin"
    playing = (turns.read_text().split() if turns.exists() else []) or ([pin.read_text().strip()] if pin.exists() else [])
    rows, omitted = [], []
    for version, played in versions.items():
        if len(played) < ONLINE_MIN_ROOMS:
            omitted.append({"version": version, "rooms": len(played)})
            continue
        clean = [[g for g in games if not g[3]] for _, _, games in played]
        n = sum(len(c) for c in clean)
        if not n:
            continue
        total = sum(g[0] for c in clean for g in c)
        mean = total / n
        spread = sum((sum(g[0] for g in c) - mean * len(c)) ** 2 for c in clean)
        se = (spread * len(clean) / (len(clean) - 1)) ** 0.5 / n
        every = [g for _, _, games in played for g in games]
        all_mean = sum(g[0] for g in every) / len(every)
        all_spread = sum((sum(g[0] for g in games) - all_mean * len(games)) ** 2 for _, _, games in played)
        room_ranks = [ranks[room] for room, _, _ in played if room in ranks]
        rows.append({
            "version": version, "rooms": len(played), "games": len(every), "clean_games": n,
            "mean": mean, "se": se, "per_hand": total / sum(g[2] for c in clean for g in c),
            "place": sum(g[1] for c in clean for g in c) / n,
            "first": sum(g[1] == 1 for c in clean for g in c) / n,
            "all_mean": all_mean, "all_se": (all_spread * len(played) / (len(played) - 1)) ** 0.5 / len(every),
            "room_rank": sum(room_ranks) / len(room_ranks) if room_ranks else None,
            "from": min(s for _, s, _ in played), "to": max(s for _, s, _ in played),
            "playing": version in playing})
    return {"enabled": True, "min_rooms": ONLINE_MIN_ROOMS, "versions": rows,
            "omitted": sorted(omitted, key=lambda r: -r["rooms"])}


def application(project, tailscale_url=None):
    arena = Research(project)
    recipes = recipe_iteration(project)

    @asynccontextmanager
    async def lifespan(app):
        controller = Controller(arena, recipes)
        if recipes is not None:
            await asyncio.to_thread(recipes.recover)
        stopped = asyncio.Event()
        watched = {"stamp": None}

        def apply_configuration():
            """research.toml is the only control surface for branches (ADR-0014)."""
            files = [arena.project / "research.toml", arena.project / ".local/machine.toml"]
            stamp = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
            if stamp != watched["stamp"]:
                watched["stamp"] = stamp
                arena.apply()

        async def loop():
            while not stopped.is_set():
                try:
                    await asyncio.to_thread(apply_configuration)
                except Exception:
                    traceback.print_exc()
                try:
                    await asyncio.to_thread(controller.tick)
                except Exception:
                    traceback.print_exc()
                    for s in arena.states():
                        if s["status"] == "running":
                            arena.control(
                                s["id"], "pause", reason="执行循环异常，已自动暂停"
                            )
                if recipes is not None:
                    try:
                        await asyncio.to_thread(recipes.tick)
                    except Exception:
                        traceback.print_exc()
                        try:
                            await asyncio.to_thread(recipes.control, "pause")
                        except Exception:
                            traceback.print_exc()
                # Jobs run for minutes; each pass rewrites a large state.
                await asyncio.sleep(2.0)

        worker = asyncio.create_task(loop())
        try:
            yield
        finally:
            stopped.set()
            await worker
            await asyncio.to_thread(controller.close)
            if recipes is not None:
                await asyncio.to_thread(recipes.close)

    # 本地控制台不缓存页面与静态资源：改了前端刷新就能看到，不会被浏览器的
    # 启发式缓存挡住（StaticFiles 默认不发 Cache-Control）。
    NO_STORE = {"Cache-Control": "no-store"}

    async def home(request):
        return FileResponse(
            Path(__file__).parent / "web/index.html", headers=NO_STORE
        )

    lines_root = ProjectDescriptor.load(project).workspace / "projects/personal/firstprinciples/.local"
    competition = ProjectDescriptor.load(project).workspace / ".local/competition"

    def iteration_lines():
        """Read-only view of the maintainer's iteration lines (ADR-0016), for every pool.

        Scores of different pools are not comparable, so the page shows one
        pool at a time. A version replayed on a new pool (harness.evaluate
        rebase) carries ``made_at``, when its attempt first finished, so it
        takes its place in the line's order instead of the replay's. Replays
        kept out of the records (``rebase/<pool>-unrecorded``, so they cannot
        become a line's best) are shown too, marked ``shown_only``. Reference
        rows of the pool's own members are marked ``member``.
        """
        pool_path, log_path = lines_root / "pool.json", lines_root / "attempts.jsonl"
        if not pool_path.exists():
            return {"enabled": False}
        pool = json.loads(pool_path.read_text())
        latest = {}  # pool id -> attempt id -> record
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    # a reference played again replaces its earlier record
                    latest.setdefault(r.get("pool"), {})[r["id"]] = r
        shown_only = set()
        for folder in (lines_root / "rebase").glob("*-unrecorded"):
            for path in folder.glob("*.json"):
                r = json.loads(path.read_text())
                if r["id"] not in latest.setdefault(r["pool"], {}):
                    latest[r["pool"]][r["id"]] = r
                    shown_only.add((r["pool"], r["id"]))

        def members(pool_id):
            if pool_id == pool["id"]:
                return sorted(pool["members"])
            archived = lines_root / "pools" / f"{pool_id}.json"  # harness.evaluate.freeze_pool
            return sorted(json.loads(archived.read_text())["members"]) if archived.exists() else []

        def made_at(r):
            for _ in range(len(latest)):  # replayed again on each later pool
                origin = latest.get((r.get("rebased") or {}).get("from_pool"), {}).get(r["id"])
                if origin is None:
                    break
                r = origin
            return r["finished_at"]

        rows, pools = [], []
        for pool_id, records in latest.items():
            names = members(pool_id)
            # Pools rebuilt before any attempt ran on them hold only references and a start.
            if pool_id != pool["id"] and not any(
                    r["line"] in ("rule", "rl") and not r["id"].startswith("fp-rl-start-") for r in records.values()):
                continue
            # One point per version, as the harness shows it (harness.evaluate.distinct):
            # of records with the same weights or package, the one over the most games.
            chosen = {}
            for r in records.values():
                spec = r.get("spec") or {}
                key = (spec.get("parameters") or {}).get("weights") or spec.get("package_hash") or r["id"]
                if key not in chosen or (r["games"], r["finished_at"]) >= (chosen[key]["games"], chosen[key]["finished_at"]):
                    chosen[key] = r
            for r in chosen.values():
                row = {k: r.get(k) for k in (
                    "id", "line", "status", "description", "started_at", "finished_at",
                    "games", "score", "se", "best_before", "diff", "confirm", "pdl", "stats",
                    "design_seconds")}
                row["pool"] = pool_id
                if r["line"] == "reference":
                    row["member"] = r["id"].removeprefix("ref-") in names
                if (pool_id, r["id"]) in shown_only:
                    row["shown_only"] = True
                if r.get("rebased"):
                    row["made_at"] = made_at(r)
                    row["rebased"] = {"old_score": r["rebased"].get("old_score")}
                if r.get("refreshed"):
                    row["refreshed"] = {"old_score": r["refreshed"].get("old_score")}
                if r.get("rejudged"):
                    row["rejudged"] = {k: r["rejudged"].get(k) for k in ("old_status", "old_games", "old_score")}
                rows.append(row)
            pools.append({"id": pool_id, "first_at": min(r["finished_at"] for r in records.values()),
                          "current": pool_id == pool["id"], "members": names})
        if pool["id"] not in latest:  # a new pool before its first record
            pools.append({"id": pool["id"], "first_at": time.time(), "current": True,
                          "members": members(pool["id"])})
        pools.sort(key=lambda p: p["first_at"])
        # "attempts" stays the current pool's, as a page loaded before older pools were served expects.
        return {"enabled": True, "pool": {"id": pool["id"], "members": sorted(pool["members"])},
                "pools": pools, "attempts": [r for r in rows if r["pool"] == pool["id"]],
                "older": [r for r in rows if r["pool"] != pool["id"]],
                "proxy": proxy_tables(latest), "online": free_play_versions(competition)}

    def proxy_tables(latest):
        """Whole-game results against the proxy panel (the tournament crowd's stand-in):
        batches appended to proxy-results.json, and any attempt record carrying ``proxy``."""
        path = lines_root / "proxy-results.json"
        value = json.loads(path.read_text()) if path.exists() else {"panels": {}, "results": []}
        for records in latest.values():
            for r in records.values():
                if (r.get("proxy") or {}).get("score") is not None:
                    value["results"].append(dict(r["proxy"], version=r["id"], at=r.get("finished_at")))
        active = lines_root / "proxy" / "panel.json"
        value["active"] = json.loads(active.read_text())["panel"] if active.exists() else None
        return value

    async def lines(request):
        return JSONResponse(await asyncio.to_thread(iteration_lines))

    async def health(request):
        return JSONResponse(
            {
                "project": str(arena.project),
                "pid": os.getpid(),
                "tailscale_url": tailscale_url,
            }
        )

    # The workbench polls every 2 seconds, and a snapshot costs seconds to
    # build (the whole research state is one JSON document). Polls share one
    # summary: once one exists, a poll returns it at once and, if it is older
    # than a few seconds, starts one background rebuild (stale while
    # revalidating). Only the first poll, or one after a mutation, waits.
    polling = {"key": None, "task": None, "at": 0.0, "value": None}

    def rebuild(experiment):
        task = polling["task"]
        if task is None or task.done() or polling.get("task_key") != experiment:
            task = asyncio.ensure_future(
                asyncio.to_thread(arena.snapshot, experiment, None, 0, True)
            )

            def landed(done):
                if not done.cancelled() and done.exception() is None:
                    polling.update(key=experiment, at=time.monotonic(), value=done.result())

            task.add_done_callback(landed)
            polling.update(task=task, task_key=experiment)
        return task

    async def summary(experiment):
        fresh = polling["key"] == experiment and polling["at"] > 0
        if fresh:
            if time.monotonic() - polling["at"] > 3:
                rebuild(experiment)
            return polling["value"]
        return await rebuild(experiment)

    def slim(snapshot):
        """A mutation's full snapshot, reduced to the polling summary's fields."""
        current = {b.get("representative") for b in snapshot["branches"]}
        long = ("last_result", "hypothesis", "checks", "preflight")
        return snapshot | {
            "iterations": [],
            "research_conclusions": [],
            "candidates": [
                {
                    k: v
                    for k, v in c.items()
                    if k not in long or (k != "last_result" and c["id"] in current)
                }
                for c in snapshot["candidates"]
            ],
        }

    async def state(request):
        experiment = request.query_params.get("experiment")
        branch = request.query_params.get("branch")
        try:
            if branch:
                value = await asyncio.to_thread(arena.snapshot, experiment, branch)
            elif request.query_params.get("summary") == "1":
                value = await summary(experiment)
            else:
                value = await asyncio.to_thread(arena.snapshot, experiment)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(value)

    async def activity(request):
        try:
            return JSONResponse(
                await asyncio.to_thread(
                    arena.activity, request.query_params.get("experiment")
                )
            )
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def mutate(request):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse(
                {"error": "local same-origin request required"}, status_code=403
            )
        try:
            value = await request.json()
            if request.url.path == "/api/experiments":
                identity = await asyncio.to_thread(
                    arena.create,
                    None,
                    None,
                    value.get("opponents"),
                    value.get("candidate_runtime_path"),
                )
                result = await asyncio.to_thread(arena.control, identity, "resume")
            else:
                result = await asyncio.to_thread(
                    arena.control,
                    value["experiment"],
                    value["action"],
                    value.get("rounds", 0),
                    value.get("options") or {},
                )
            polling["at"] = 0.0  # the next poll must see this change
            return JSONResponse(slim(result))
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def recipe_state(request):
        if recipes is None:
            return JSONResponse({"enabled": False})
        return JSONResponse(
            {"enabled": True} | await asyncio.to_thread(recipes.snapshot)
        )

    async def recipe_trial(request):
        try:
            return JSONResponse(
                await asyncio.to_thread(
                    recipes.trial_detail, request.query_params.get("id", "")
                )
            )
        except (AttributeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/", home),
            Route("/api/recipes/state", recipe_state),
            Route("/api/recipes/trial", recipe_trial),
            Route("/api/health", health),
            Route("/api/lines", lines),
            Route("/api/state", state),
            Route("/api/activity", activity),
            Route("/api/experiments", mutate, methods=["POST"]),
            Route("/api/control", mutate, methods=["POST"]),
        ],
    )
    class Assets(StaticFiles):
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "no-store"
            return response

    app.mount("/assets", Assets(directory=Path(__file__).parent / "web"), name="assets")
    return app


def service_paths(project):
    root = Path(project).resolve() / ".local/service"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    label = (
        "com.mahjong."
        + hashlib.sha256(str(Path(project).resolve()).encode()).hexdigest()[:16]
    )
    return root, label


def health_at(url, project):
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
            state = json.load(response)
        return state if state["project"] == str(Path(project).resolve()) else None
    except (OSError, ValueError, KeyError):
        return None


def tailscale_ip():
    executable = shutil.which("tailscale")
    if not executable:
        bundled = Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale")
        executable = str(bundled) if bundled.exists() else None
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "ip", "-4"], capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return None
        address = ipaddress.IPv4Address(result.stdout.strip())
        return (
            str(address) if address in ipaddress.IPv4Network("100.64.0.0/10") else None
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def listener_sockets(port, remote_ip=None):
    """Listen only on loopback and the connected tailnet interface."""
    sockets = []
    try:
        for host in ["127.0.0.1", *([remote_ip] if remote_ip else [])]:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            # Port zero must select the same port on both interfaces.
            port = sock.getsockname()[1]
            sock.listen(128)
        return sockets
    except BaseException:
        for sock in sockets:
            sock.close()
        raise


def start(project, open_browser=True):
    root, label = service_paths(project)
    with exclusive_lock(root / "start.lock", blocking=True):
        record = root / "service.json"
        if record.exists():
            url = read_json(record)["url"]
            if health_at(url, project):
                if open_browser:
                    open_url(url)
                return url
        remote_ip = tailscale_ip()
        try:
            probes = listener_sockets(8765, remote_ip)
        except OSError:
            probes = listener_sockets(0, remote_ip)
        port = probes[0].getsockname()[1]
        for sock in probes:
            sock.close()
        url = f"http://127.0.0.1:{port}"
        env = {
            key: os.environ[key]
            for key in (
                "HOME",
                "PATH",
                "CODEX_HOME",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
            )
            if key in os.environ
        }
        env["PYTHONPATH"] = str(Path(__file__).parents[2])
        # Custom providers sometimes use an environment-backed credential.
        import tomllib

        config_path = (
            Path(env.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
        )
        config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
        for provider in config.get("model_providers", {}).values():
            key = provider.get("env_key")
            if key and key in os.environ:
                env[key] = os.environ[key]
        argv = [
            sys.executable,
            "-B",
            "-m",
            "mahjong_ai.interfaces.app",
            "--project",
            str(Path(project).resolve()),
            "serve",
            "--port",
            str(port),
            *(["--tailscale-ip", remote_ip] if remote_ip else []),
        ]
        start_service(root, label, project, argv, env)
        atomic_json(record, {"url": url, "label": label})
        for _ in range(100):
            if health_at(url, project):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                f"service did not become ready; see {root / 'service.log'}"
            )
        if open_browser:
            open_url(url)
        return url


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地麻将研究控制台")
    default_project = Path.cwd()
    migrated_agent = default_project / "projects/agent-research"
    if (migrated_agent / "project.toml").exists() and (
        migrated_agent / "research.toml"
    ).exists():
        default_project = migrated_agent
    parser.add_argument("--project", default=str(default_project))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start")
    p.add_argument("--no-open", action="store_true")
    sub.add_parser("status")
    sub.add_parser("stop")
    # The workbench is read-only; experiments are operated from here.
    p = sub.add_parser("control", help="pause, resume, stop or close the experiment")
    p.add_argument("action", choices=["pause", "resume", "pause_after", "stop_now",
                                      "clear_stop", "close"])
    p = sub.add_parser("create", help="start a new experiment from research.toml")
    p.add_argument("--opponents", nargs="*", default=None)
    p.add_argument("--candidate-runtime", default=None)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--tailscale-ip", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root, label = service_paths(args.project)
    if args.command == "serve":
        import uvicorn

        with exclusive_lock(root / "server.lock"):
            remote_ip = args.tailscale_ip or tailscale_ip()
            sockets = listener_sockets(args.port, remote_ip)
            remote_url = f"http://{remote_ip}:{args.port}" if remote_ip else None
            try:
                server = uvicorn.Server(
                    uvicorn.Config(
                        application(args.project, remote_url), log_level="warning"
                    )
                )
                server.run(sockets=sockets)
            finally:
                for sock in sockets:
                    sock.close()
    elif args.command == "start":
        through_ssh = bool(
            os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")
        )
        url = start(args.project, not args.no_open and not through_ssh)
        print(url)
        running = health_at(url, args.project) or {}
        if running.get("tailscale_url"):
            print("Tailscale: " + running["tailscale_url"])
    elif args.command in ("control", "create"):
        record = root / "service.json"
        value = read_json(record) if record.exists() else {}
        if not value or not health_at(value["url"], args.project):
            raise SystemExit("服务未运行，先执行 start")
        if args.command == "create":
            path, body = "/api/experiments", {
                "opponents": args.opponents,
                "candidate_runtime_path": args.candidate_runtime,
            }
        else:
            with urllib.request.urlopen(value["url"] + "/api/state?summary=1", timeout=60) as r:
                experiment = (json.load(r).get("experiment") or {}).get("id")
            path, body = "/api/control", {"experiment": experiment, "action": args.action}
        request = urllib.request.Request(
            value["url"] + path, json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as r:
                state = json.load(r)
        except urllib.error.HTTPError as exc:
            raise SystemExit(json.load(exc).get("error", str(exc)))
        current = state.get("experiment") or {}
        print(json.dumps({k: current.get(k) for k in ("id", "status", "phase", "pause_reason")},
                         ensure_ascii=False))
    elif args.command == "stop":
        record = root / "service.json"
        value = read_json(record) if record.exists() else {}
        previous = health_at(value["url"], args.project) if value else None
        stopped = stop_service(root, label, previous)
        print("服务已停止" if stopped else "服务未运行")
    else:
        record = root / "service.json"
        value = read_json(record) if record.exists() else {}
        running = health_at(value["url"], args.project) if value else None
        print(
            json.dumps(
                {
                    "running": bool(running),
                    **value,
                    "tailscale_url": (running or {}).get("tailscale_url"),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
