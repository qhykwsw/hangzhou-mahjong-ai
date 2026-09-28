"""Local dashboard and one host-managed service per project."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
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
                    "games", "score", "se", "best_before", "diff", "confirm", "stats",
                    "design_seconds")}
                row["pool"] = pool_id
                if r["line"] == "reference":
                    row["member"] = r["id"].removeprefix("ref-") in names
                if (pool_id, r["id"]) in shown_only:
                    row["shown_only"] = True
                if r.get("rebased"):
                    row["made_at"] = made_at(r)
                    row["rebased"] = {"old_score": r["rebased"].get("old_score")}
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
                "older": [r for r in rows if r["pool"] != pool["id"]]}

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
