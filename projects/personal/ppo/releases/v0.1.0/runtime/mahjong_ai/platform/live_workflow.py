"""Reusable four-token test-room workflow; secrets stay in a local input file."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from mahjong_ai.platform.calibration import _request
from mahjong_ai.interfaces.cli import spec
from mahjong_ai.platform.online import (
    PlatformClient,
    SnapshotAdapter,
    TournamentRunner,
    _id,
    _redact,
)
from mahjong_ai.persistence.artifacts import (
    atomic_json,
    digest,
    exclusive_lock,
    read_json,
    runtime_fingerprint,
)


def load_credentials(path):
    data = read_json(path)
    players = data.get("players", data.get("tokens"))
    if not isinstance(players, list) or len(players) != 4:
        raise ValueError("credentials file requires exactly four players or tokens")
    tokens = [p if isinstance(p, str) else p["token"] for p in players]
    if len(set(tokens)) != 4 or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("four distinct nonempty scoped tokens are required")
    return data.get("room_id"), tokens


def summarize_player(root):
    actions, failures, decision_ms, request_ms, scores, finished = 0, [], [], [], [], 0
    for file in Path(root).glob("game-*.jsonl"):
        for line in file.read_text().splitlines():
            r = json.loads(line)
            actions += 1
            decision_ms.append(r["decision_ms"])
            request_ms.append(r["request_ms"])
            if not r["accepted"]:
                failures.append(
                    {
                        k: r.get(k)
                        for k in (
                            "game_id",
                            "seq",
                            "action",
                            "http_status",
                            "error_code",
                        )
                    }
                )
    for file in Path(root).glob("game-*.json"):
        data = read_json(file)
        if data["status"] == "finished":
            finished += 1
            final = data.get("final", {}).get("snapshot") or {}
            seat = final.get("seat")
            if isinstance(seat, int) and 0 <= seat < 4 and final.get("scores"):
                scores.append(final["scores"][seat])
    import numpy as np

    def timing(values):
        return {
            "count": len(values),
            "p50_ms": float(np.median(values)) if values else None,
            "p95_ms": float(np.quantile(values, 0.95)) if values else None,
            "max_ms": max(values) if values else None,
        }

    return {
        "actions": actions,
        "unsuccessful_actions": len(failures),
        "failures": failures,
        "finished_games": finished,
        "final_scores": scores,
        "score_sum": sum(scores),
        "decision_latency": timing(decision_ms),
        "request_latency": timing(request_ms),
    }


def game_rows(payload):
    rows = payload if isinstance(payload, list) else payload["games"]
    if not isinstance(rows, list):
        raise ValueError("unknown game-list schema")
    return rows


def run_room(
    credentials,
    policies,
    mapping,
    output,
    base,
    insecure=False,
    seconds=900,
    capture_only=False,
):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    expected_room, tokens = load_credentials(credentials)
    clients = [PlatformClient(base, token, insecure) for token in tokens]
    discoveries = [client.discover(expected_room) for client in clients]
    room_ids = {d[0] for d in discoveries}
    if len(room_ids) != 1 or not all(d[1] for d in discoveries):
        raise ValueError("all four tokens must be scoped to the same test room")
    room_id = room_ids.pop()
    # Public endpoint serves test/auto rooms only; formal tournaments return 404.
    previous_games = _request(
        base, f"/api/test-rooms/{_id(room_id)}/games", insecure=insecure
    )
    rules = clients[0].request("GET", "/api/tournaments/me/rules")
    config = rules["config"]
    metadata = {
        "room_id": room_id,
        "runtime": runtime_fingerprint(),
        "config": config,
        "token_hashes": [digest(t) for t in tokens],
        "policies": [p.as_dict() for p in policies],
        "policy_ids": [p.identity for p in policies],
        "mapping": mapping,
        "capture_only": capture_only,
    }
    state_path = root / "workflow.json"
    with exclusive_lock(root / ".room.lock"):
        if state_path.exists():
            state = read_json(state_path)
            if state["protocol"] != metadata:
                raise ValueError("workflow changed; use a new output directory")
            if state["status"] == "completed":
                return state["summary"]
        else:
            state = {
                "protocol": metadata,
                "status": "running" if rules["status"] == "running" else "readying",
                "previous_games": previous_games,
                "excluded_game_ids": (
                    [r["game_id"] for r in game_rows(previous_games)]
                    if rules["status"] != "running"
                    else [
                        r["game_id"]
                        for r in game_rows(previous_games)
                        if r["status"] != "running"
                    ]
                ),
            }
            atomic_json(state_path, state)
        atomic_json(root / "rules.json", _redact(rules))
        atomic_json(
            root / "player-identities.json", [d[2]["user_id"] for d in discoveries]
        )
        deadline = time.monotonic() + seconds
        # One deliberate readiness wave per workflow. A repeated finished room
        # is a new experiment, not an automatic endless ready/finished loop.
        if state["status"] == "readying":
            for client in clients:
                client.request("POST", "/api/tournaments/me/ready")
            state["status"] = "running"
            atomic_json(state_path, state)
        if capture_only:
            snapshots = []
            while time.monotonic() < deadline and len(snapshots) != 4:
                snapshots = []
                for index, client in enumerate(clients):
                    me = client.request("GET", "/api/me")
                    games = me["active_games"]
                    if not games:
                        break
                    gid = games[0]["game_id"]
                    payload = client.request(
                        "GET", f"/api/games/{_id(gid)}/state?seq=0"
                    )
                    atomic_json(
                        root / f"player-{index}-snapshot.json", _redact(payload)
                    )
                    snapshots.append(index)
                if len(snapshots) != 4:
                    time.sleep(0.2)
            return {
                "status": "captured" if len(snapshots) == 4 else "incomplete",
                "players": len(snapshots),
                "output": str(root),
                "note": "room is running; snapshot capture did not play it",
            }
        if mapping in ({"adapter": "official"}, {"adapter": "official-v25"}):
            from mahjong_ai.platform.official import OfficialAdapter

            adapters = [OfficialAdapter() for _ in range(4)]
        else:
            adapters = [SnapshotAdapter(mapping) for _ in range(4)]
        runners = [
            TournamentRunner(
                client,
                policy,
                adapter,
                root / f"player-{i}",
                workers=min(16, config["M"]),
                join=False,
            )
            for i, (client, policy, adapter) in enumerate(
                zip(clients, policies, adapters)
            )
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(runner.run, room_id, max(0.01, deadline - time.monotonic()))
                for runner in runners
            ]
            try:
                results = [future.result() for future in as_completed(futures)]
            finally:
                for runner in runners:
                    runner.stop.set()
        # Record authoritative ranking and game list before any next readiness wave.
        tournament = clients[0].request("GET", f"/api/tournaments/{_id(room_id)}")
        atomic_json(root / "final-tournament.json", _redact(tournament))
        games = _request(
            base, f"/api/test-rooms/{_id(room_id)}/games", insecure=insecure
        )
        atomic_json(root / "games.json", games)
        # A room can finish between the coordinator poll and worker discovery.
        # Collect authoritative final states for all newly observed games as well.
        if tournament["status"] == "finished":
            for i, client in enumerate(clients):
                mine = client.request("GET", f"/api/tournaments/{_id(room_id)}")
                for gid in mine.get("my_games", []):
                    if gid in state["excluded_game_ids"]:
                        continue
                    final = client.request("GET", f"/api/games/{_id(gid)}/state?seq=0")
                    if final.get("finished"):
                        path = (
                            root
                            / f"player-{i}"
                            / ("game-" + digest(gid)[:24] + ".json")
                        )
                        data = (
                            read_json(path)
                            if path.exists()
                            else {"responses": [], "policy_id": policies[i].identity}
                        )
                        data.update(
                            game_id=gid, status="finished", final=_redact(final)
                        )
                        atomic_json(path, data)
        replay_results = []
        rows = game_rows(games)
        for row in rows:
            if (
                row["game_id"] in state["excluded_game_ids"]
                or row["status"] != "finished"
            ):
                continue
            # The public batch URL is ambiguous after a room is reused.
            if sum(r["batch"] == row["batch"] for r in rows) > 1:
                replay_results.append(
                    {
                        "game_id": row["game_id"],
                        "status": "needs_portal_export",
                        "reason": "batch repeated across room cycles; use game_id portal replay",
                    }
                )
                continue
            payload = _request(
                base,
                f"/api/test-rooms/{_id(room_id)}/games/{row['batch']}/events",
                insecure=insecure,
            )
            if payload.get("game_id", row["game_id"]) != row["game_id"]:
                raise ValueError("replay belongs to a different game")
            directory = root / "replays"
            path = directory / (digest(row["game_id"])[:24] + ".json")
            atomic_json(path, payload)
            from mahjong_ai.platform.replay_audit import audit_replay

            audit = audit_replay(payload, config["BaseScore"], config["YouCaiBiKao"])
            atomic_json(path.with_suffix(".audit.json"), audit)
            replay_results.append(
                {
                    "game_id": row["game_id"],
                    "status": audit["status"],
                    "hands": audit["hands"],
                    "failures": len(audit["failures"]),
                }
            )
        players = [
            summarize_player(root / f"player-{i}")
            | {"family": p.family, "version": p.version}
            for i, p in enumerate(policies)
        ]
        complete = tournament["status"] == "finished" and all(
            p["finished_games"] == config["M"] for p in players
        )
        summary = {
            "status": "completed" if complete else "incomplete",
            "room_status": tournament["status"],
            "players": players,
            "replays": replay_results,
            "rounds_per_game": config["Rounds"],
            "games_per_player": config["M"],
            "output": str(root),
            "scope": "platform execution evidence; mixed table is not a paired promotion test",
        }
        state.update(status=summary["status"], summary=summary)
        atomic_json(state_path, state)
        atomic_json(root / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run four frozen strategies in an existing four-token test room"
    )
    parser.add_argument("--credentials", required=True)
    parser.add_argument(
        "--mapping",
        help="optional custom mapping; default uses the current official adapter",
    )
    parser.add_argument("--policies", nargs=4, default=["rule"] * 4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="https://10.240.169.190:18080")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_room(
        args.credentials,
        [spec(p) for p in args.policies],
        read_json(args.mapping) if args.mapping else {"adapter": "official"},
        args.output,
        args.base_url,
        args.insecure,
        args.seconds,
        args.capture_only,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in ("completed", "captured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
