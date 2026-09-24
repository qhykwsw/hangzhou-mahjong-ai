"""Platform replays as an imitation dataset (ADR-0015).

``scan`` indexes every finished replay under the competition directories:
each game is rebuilt by ``platform_replays`` once, and the index keeps its
file hash, room configuration, players, scores and decision counts, or why it
failed to rebuild. ``players`` ranks the external players by table score, the
basis for choosing whom to imitate. ``freeze`` fixes the games a training run
uses (by file hash), and ``rows`` rebuilds and encodes their decisions at
training time, so only replays travel between machines.

Replays exist only for free-play (auto) and test rooms; the platform serves
none for formal tournaments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .platform_replays import ReplayMismatch, decisions

INDEX = "games.jsonl"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _room(replay_path):
    """The room record next to a replay: our user id and the room configuration."""
    match = Path(replay_path).parent.parent / "match.json"
    if not match.exists():
        return None, None
    data = json.loads(match.read_text())
    return data.get("user_id"), (data.get("match") or {}).get("config")


def _game_record(path, root):
    replay = json.loads(Path(path).read_text())
    ours, config = _room(path)
    record = {
        "game_id": replay.get("game_id"),
        "room_id": replay.get("room_id"),
        "path": str(Path(path).resolve().relative_to(root)),
        "sha256": _sha256(path),
        "config": config,
        "ours": ours,
        "users": [s["user_id"] for s in replay.get("seats", [])],
        "names": [s.get("name", "") for s in replay.get("seats", [])],
    }
    if replay.get("status") != "finished":
        return dict(record, status="unfinished")
    try:
        found = decisions(replay, config)
    except ReplayMismatch as exc:
        return dict(record, status="mismatch", error=str(exc)[:500])
    rounds = replay.get("rounds", [])
    record.update(
        status="ok",
        hands=len(rounds),
        scores=[sum(r["scores"][i] for r in rounds) for i in range(4)],
        wins=[sum(r["winner"] == i and not r["is_draw"] for r in rounds) for i in range(4)],
        decisions=[sum(d.seat == i for d in found) for i in range(4)],
        timeouts=[sum(d.seat == i and d.timeout for d in found) for i in range(4)],
    )
    return record


def load_index(data_dir):
    path = Path(data_dir) / INDEX
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scan(sources, data_dir, root):
    """Index replays not yet indexed; returns the records added."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    known = {r["sha256"] for r in load_index(data_dir)}
    added = []
    with (data_dir / INDEX).open("a") as out:
        for source in sources:
            for path in sorted(Path(source).rglob("replays/*.json")):
                if _sha256(path) in known:
                    continue
                record = _game_record(path, root)
                if record["status"] == "unfinished":
                    continue  # may finish later; index it then
                known.add(record["sha256"])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                added.append(record)
    return added


def players(records):
    """Per external player: games, hands, win rate and mean table score per hand."""
    stats = {}
    for r in records:
        if r["status"] != "ok":
            continue
        for seat, user in enumerate(r["users"]):
            if user == r.get("ours"):
                continue
            s = stats.setdefault(user, dict(user_id=user, name="", games=0, hands=0, wins=0,
                                            decisions=0, timeouts=0, per_hand=[]))
            s["name"] = r["names"][seat] or s["name"]
            s["games"] += 1
            s["hands"] += r["hands"]
            s["wins"] += r["wins"][seat]
            s["decisions"] += r["decisions"][seat]
            s["timeouts"] += r["timeouts"][seat]
            s["per_hand"].append(r["scores"][seat] / max(1, r["hands"]))
    table = []
    for s in stats.values():
        values = s.pop("per_hand")
        mean = sum(values) / len(values)
        spread = (
            math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values))
            if len(values) > 1 else None
        )
        table.append(dict(s, win_rate=s["wins"] / max(1, s["hands"]),
                          score_per_hand=mean, score_per_hand_se=spread))
    return sorted(table, key=lambda s: -s["score_per_hand"])


def freeze(records, users=None, output=None):
    """Fix the games (by hash) and players a training run imitates."""
    games = [
        {k: r[k] for k in ("game_id", "path", "sha256", "config", "ours")}
        for r in records
        if r["status"] == "ok" and (users is None or set(users) & set(r["users"]))
    ]
    manifest = {"games": games, "users": sorted(users) if users else None}
    manifest["id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if output:
        Path(output).write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def rows(manifest, root, *, lookahead=True, include_timeouts=False, include_ours=False):
    """Encoded decisions of the frozen games, labelled with the players' actions."""
    from selfplay_core.actions import action_index

    from .encoding import encode

    users = set(manifest["users"]) if manifest.get("users") else None
    for game in manifest["games"]:
        path = Path(root) / game["path"]
        if _sha256(path) != game["sha256"]:
            raise ValueError(f"replay changed since freezing: {game['path']}")
        for d in decisions(json.loads(path.read_text()), game["config"]):
            if d.timeout and not include_timeouts:
                continue
            if d.user_id == game.get("ours") and not include_ours:
                continue
            if users is not None and d.user_id not in users:
                continue
            row = encode(d.observation, lookahead)
            row.update(teacher=action_index(d.action), user=d.user_id, game=d.game_id, hand=d.hand)
            yield row


def main():
    repo = Path(__file__).resolve().parents[4]
    default_data = repo / "projects/personal/selfplay/.local/platform-data"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="index new replays and refresh players.json")
    s.add_argument("--sources", nargs="+", default=[str(repo / ".local/competition")])
    s.add_argument("--data", default=str(default_data))
    f = sub.add_parser("freeze", help="write a training manifest")
    f.add_argument("--data", default=str(default_data))
    f.add_argument("--users", nargs="*", help="imitate only these players (default: all external)")
    f.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "scan":
        added = scan(args.sources, args.data, repo)
        records = load_index(args.data)
        table = players(records)
        (Path(args.data) / "players.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
        ok = [r for r in records if r["status"] == "ok"]
        external = sum(
            n - t for r in ok for u, n, t in zip(r["users"], r["decisions"], r["timeouts"])
            if u != r.get("ours")
        )
        print(json.dumps(dict(
            added=len(added), games=len(records), ok=len(ok),
            mismatched=sum(r["status"] == "mismatch" for r in records),
            external_decisions=external, external_players=len(table),
        ), ensure_ascii=False))
    else:
        manifest = freeze(load_index(args.data), args.users, args.output)
        print(json.dumps({"id": manifest["id"], "games": len(manifest["games"])}))


if __name__ == "__main__":
    main()
