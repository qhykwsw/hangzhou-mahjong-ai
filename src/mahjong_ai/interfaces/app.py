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
import urllib.request

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from mahjong_ai.research.service import Research
from mahjong_ai.research.config import configuration, COMBINATIONS
from mahjong_ai.execution.controller import Controller
from mahjong_ai.persistence.artifacts import atomic_json, exclusive_lock, read_json
from mahjong_ai.interfaces.service_host import open_url, start_service, stop_service


def application(project, tailscale_url=None):
    arena = Research(project)

    @asynccontextmanager
    async def lifespan(app):
        controller = Controller(arena)
        stopped = asyncio.Event()

        async def loop():
            while not stopped.is_set():
                try:
                    await asyncio.to_thread(controller.tick)
                except Exception:
                    traceback.print_exc()
                    for s in arena.states():
                        if s["status"] == "running":
                            arena.control(
                                s["id"], "pause", reason="执行循环异常，已自动暂停"
                            )
                await asyncio.sleep(0.25)

        worker = asyncio.create_task(loop())
        try:
            yield
        finally:
            stopped.set()
            await worker
            await asyncio.to_thread(controller.close)

    # 本地控制台不缓存页面与静态资源：改了前端刷新就能看到，不会被浏览器的
    # 启发式缓存挡住（StaticFiles 默认不发 Cache-Control）。
    NO_STORE = {"Cache-Control": "no-store"}

    async def home(request):
        return FileResponse(
            Path(__file__).parent / "web/index.html", headers=NO_STORE
        )

    async def health(request):
        return JSONResponse(
            {
                "project": str(arena.project),
                "pid": os.getpid(),
                "tailscale_url": tailscale_url,
            }
        )

    async def state(request):
        return JSONResponse(
            await asyncio.to_thread(
                arena.snapshot, request.query_params.get("experiment")
            )
        )

    async def activity(request):
        try:
            return JSONResponse(
                await asyncio.to_thread(
                    arena.activity, request.query_params.get("experiment")
                )
            )
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def directions(request):
        return JSONResponse(
            {
                "defaults": configuration(arena.project),
                "combinations": [
                    {"id": k, "name": v[1], "family": v[0]}
                    for k, v in COMBINATIONS.items()
                ],
            }
        )

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
                    value.get("config"),
                    value.get("targets"),
                    value.get("opponents"),
                    value.get("candidate_runtime_path"),
                )
                result = await asyncio.to_thread(arena.control, identity, "resume")
            elif request.url.path == "/api/config":
                result = await asyncio.to_thread(
                    arena.configure, value["experiment"], value["patch"]
                )
            elif request.url.path == "/api/branches":
                result = await asyncio.to_thread(
                    arena.archive, value["branch"], value["archived"]
                )
            else:
                result = await asyncio.to_thread(
                    arena.control,
                    value["experiment"],
                    value["action"],
                    value.get("rounds", 0),
                    value.get("options") or {},
                )
            return JSONResponse(result)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/", home),
            Route("/api/health", health),
            Route("/api/state", state),
            Route("/api/activity", activity),
            Route("/api/directions", directions),
            Route("/api/experiments", mutate, methods=["POST"]),
            Route("/api/config", mutate, methods=["POST"]),
            Route("/api/branches", mutate, methods=["POST"]),
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
