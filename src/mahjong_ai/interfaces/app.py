"""Read-only web page of the maintainer's iteration lines (ADR-0016): attempts, proxy tables and free play."""

import argparse
import asyncio
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from mahjong_ai.persistence.artifacts import atomic_json, exclusive_lock, read_json
from mahjong_ai.interfaces.service_host import open_url, start_service, stop_service


ONLINE_MIN_ROOMS = 10  # a room's total varies by ~180 points: fewer rooms say little about a version (user 10-07: from 20 to 10)
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
    with fewer than ONLINE_MIN_ROOMS rooms are listed but not scored, except the
    ones playing now, scored from their second room (user 10-07).
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
            summary = folder / "summary.json"
            if (len(games) < (json.loads(summary.read_text()).get("valid_final_snapshots") or 0)
                    and time.time() - summary.stat().st_mtime < 3600):
                continue  # the room's replays are still being written: read it again later
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
        if len(played) < (2 if version in playing else ONLINE_MIN_ROOMS):
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


def application(workspace, tailscale_url=None):
    workspace = Path(workspace).resolve()

    # 本地控制台不缓存页面与静态资源：改了前端刷新就能看到，不会被浏览器的
    # 启发式缓存挡住（StaticFiles 默认不发 Cache-Control）。
    NO_STORE = {"Cache-Control": "no-store"}

    async def home(request):
        return FileResponse(
            Path(__file__).parent / "web/index.html", headers=NO_STORE
        )

    lines_root = workspace / "projects/personal/firstprinciples/.local"
    competition = workspace / ".local/competition"

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
                    records = latest.setdefault(r.get("pool"), {})
                    before = records.get(r["id"])
                    if before and before.get("proxy") and not r.get("proxy"):
                        # A later record (a rejudge) keeps the version's proxy tables: same version, same tables.
                        r = dict(r, proxy=before["proxy"])
                    records[r["id"]] = r
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
                if r.get("rejudged") and not r.get("override"):  # a user's verdict replaces the rejudge note
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
                "workspace": str(workspace),
                "pid": os.getpid(),
                "tailscale_url": tailscale_url,
            }
        )

    app = Starlette(
        routes=[
            Route("/", home),
            Route("/api/health", health),
            Route("/api/lines", lines),
        ],
    )
    class Assets(StaticFiles):
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "no-store"
            return response

    app.mount("/assets", Assets(directory=Path(__file__).parent / "web"), name="assets")
    return app


def service_paths(workspace):
    root = Path(workspace).resolve() / ".local/web"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    label = (
        "com.mahjong."
        + hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
    )
    return root, label


def health_at(url, workspace):
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
            state = json.load(response)
        return state if state["workspace"] == str(Path(workspace).resolve()) else None
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


def start(workspace, open_browser=True):
    root, label = service_paths(workspace)
    with exclusive_lock(root / "start.lock", blocking=True):
        record = root / "service.json"
        if record.exists():
            url = read_json(record)["url"]
            if health_at(url, workspace):
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
        env = {key: os.environ[key] for key in ("HOME", "PATH") if key in os.environ}
        env["PYTHONPATH"] = str(Path(__file__).parents[2])
        argv = [
            sys.executable,
            "-B",
            "-m",
            "mahjong_ai.interfaces.app",
            "--workspace",
            str(Path(workspace).resolve()),
            "serve",
            "--port",
            str(port),
            *(["--tailscale-ip", remote_ip] if remote_ip else []),
        ]
        start_service(root, label, workspace, argv, env)
        atomic_json(record, {"url": url, "label": label})
        for _ in range(100):
            if health_at(url, workspace):
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
    parser = argparse.ArgumentParser(description="迭代线只读网页")
    parser.add_argument("--workspace", default=str(Path.cwd()))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start")
    p.add_argument("--no-open", action="store_true")
    sub.add_parser("status")
    sub.add_parser("stop")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--tailscale-ip", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root, label = service_paths(args.workspace)
    record = root / "service.json"
    if args.command == "serve":
        import uvicorn

        with exclusive_lock(root / "server.lock"):
            remote_ip = args.tailscale_ip or tailscale_ip()
            sockets = listener_sockets(args.port, remote_ip)
            remote_url = f"http://{remote_ip}:{args.port}" if remote_ip else None
            try:
                server = uvicorn.Server(
                    uvicorn.Config(
                        application(args.workspace, remote_url), log_level="warning"
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
        url = start(args.workspace, not args.no_open and not through_ssh)
        print(url)
        running = health_at(url, args.workspace) or {}
        if running.get("tailscale_url"):
            print("Tailscale: " + running["tailscale_url"])
    elif args.command == "stop":
        value = read_json(record) if record.exists() else {}
        previous = health_at(value["url"], args.workspace) if value else None
        stopped = stop_service(root, label, previous)
        print("服务已停止" if stopped else "服务未运行")
    else:
        value = read_json(record) if record.exists() else {}
        running = health_at(value["url"], args.workspace) if value else None
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
