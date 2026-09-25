#!/usr/bin/env python3
"""Check every archived platform room against the current local engine.

Run after changing the engine or the rules: it replays each room's official
games (replays/*.json) through the real engine and compares our decisions
(player/game-*.jsonl) with the engine state at the same point.

1. Replay audit (platform.replay_audit): official actions, draws,
   discard-timeout fallbacks, fan, score deltas, dealer, tile conservation.
2. Decision parity: at each of our decisions, the platform snapshot's hand,
   discards, melds, drawn tile, catch-play owner and piao chain against the
   engine state reached by replaying the same game.
3. Rejected actions: whether the engine held each refused action legal, and
   what the replay shows happened next (another seat's claim, a timeout).

Only paths the games took are checked; unplayed actions and room settings
absent from the archive stay unverified. Exits 1 on any failed audit,
unmatched snapshot, field difference, or refused action the engine rejects too.

    python scripts/audit_archive_parity.py [room-glob ...] [--output report.json]
"""

import argparse
import bisect
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
FIELDS = (
    "my_hand",
    "discards",
    "my_melds",
    "catch_play",
    "god_discarder_seat",
    "chain_count",
    "drawn",
)


def games(root, patterns):
    jobs = []
    for pattern in patterns:
        for room in sorted(root.glob(pattern)):
            match = room / "match.json"
            if not match.exists() or not (room / "replays").is_dir():
                continue
            config = (json.loads(match.read_text()).get("match") or {}).get(
                "config"
            ) or {}
            for replay in sorted((room / "replays").glob("*.json")):
                decisions = room / "player" / f"game-{replay.stem}.jsonl"
                jobs.append(
                    (
                        str(room),
                        str(replay),
                        str(decisions) if decisions.exists() else None,
                        config.get("BaseScore", 1),
                        bool(config.get("YouCaiBiKao")),
                    )
                )
    return jobs


def snapshot_key(s):
    return (
        s["round_no"],
        s["wall_remaining"],
        s["phase"],
        tuple(len(d) for d in s["discards"]),
        tuple(len(m) for m in s["melds"]),
        tuple(s["hand_counts"]),
    )


def state_key(state):
    return (
        state.hand_no,
        len(state.wall),
        state.phase,
        tuple(len(d) for d in state.discards),
        tuple(len(m) for m in state.melds),
        tuple(len(h) for h in state.hands),
    )


def check_game(job):
    """One game: replay audit, decision parity and rejected actions."""
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.rules import tile_id
    from mahjong_ai.platform.replay_audit import audit_replay

    room, replay_path, decisions_path, base, youcai = job
    out = {"room": Path(room).name, "game": Path(replay_path).stem}
    try:
        rows = (
            [json.loads(line) for line in open(decisions_path)]
            if decisions_path
            else []
        )
        rows = [r for r in rows if r.get("snapshot")]
        seat = rows[0]["snapshot"]["seat"] if rows else None
        wanted = {snapshot_key(r["snapshot"]) for r in rows}
        seen = defaultdict(list)

        def capture(state):
            key = state_key(state)
            if key in wanted:
                seen[key].append(
                    {
                        "my_hand": sorted(state.hands[seat]),
                        "discards": [list(d) for d in state.discards],
                        "my_melds": sorted(sorted(m.tiles) for m in state.melds[seat]),
                        "catch_play": state.catch_owner is not None,
                        "god_discarder_seat": -1
                        if state.catch_owner is None
                        else state.catch_owner,
                        "chain_count": state.chain[seat],
                        "drawn": state.drawn
                        if state.phase == "draw" and state.turn == seat
                        else None,
                        "legal": set(state.legal_actions())
                        if state.turn == seat
                        else None,
                    }
                )

        original = HandState.step

        def step(self, action):
            capture(self)
            result = original(self, action)
            capture(self)
            return result

        replay = json.loads(Path(replay_path).read_text())
        HandState.step = step
        try:
            report = audit_replay(replay, base, youcai)
        finally:
            HandState.step = original
        out.update(
            audit=report["status"],
            checks=report["checks"],
            failures=report["failures"][:5],
        )

        differences, unmatched, examples = Counter(), 0, []
        for row in rows:
            s = row["snapshot"]
            candidates = seen.get(snapshot_key(s))
            if not candidates:
                unmatched += 1
                continue
            want = {
                "my_hand": sorted(tile_id(t) for t in s["my_hand"]),
                "discards": [[tile_id(t) for t in d] for d in s["discards"]],
                "my_melds": sorted(
                    sorted(tile_id(t) for t in m["tiles"]) for m in s["melds"][seat]
                ),
                "catch_play": bool(s["god"]["catch_play"]),
                "god_discarder_seat": s["god"]["god_discarder_seat"],
                "chain_count": s["god"]["chain_count"],
            }
            if s.get("drawn_tile") and s["phase"] == "draw" and s["turn"] == seat:
                want["drawn"] = tile_id(s["drawn_tile"])
            diffs = [[k for k, v in want.items() if got[k] != v] for got in candidates]
            closest = min(diffs, key=len)
            if closest:
                differences.update(closest)
                if len(examples) < 3:
                    got = candidates[diffs.index(closest)]
                    examples.append(
                        {
                            "seq": row["seq"],
                            "round": s["round_no"],
                            "phase": s["phase"],
                            "fields": {k: [want[k], got[k]] for k in closest},
                        }
                    )
        out.update(
            decisions=len(rows),
            unmatched=unmatched,
            differences=dict(differences),
            examples=examples,
        )

        events = {
            e["seq"]: e for block in replay["blocks"] for e in block.get("events", [])
        }
        order = sorted(events)
        refused = Counter()
        for row in rows:
            if row.get("accepted") is not False or not row.get("action"):
                continue
            i = bisect.bisect_right(order, row["seq"])
            nxt = events[order[i]] if i < len(order) else None
            if (
                nxt
                and nxt["type"] in ("chi", "peng", "gang", "hu")
                and nxt.get("seat") != seat
            ):
                cause = "claimed-by-other"
            elif (row.get("window_left_ms") or 0) < 200 or (
                nxt and nxt["type"] == "timeout"
            ):
                cause = "late"
            else:
                cause = "stale-or-duplicate"
            action = row["action"]
            candidates = [
                c
                for c in seen.get(snapshot_key(row["snapshot"])) or []
                if c["legal"] is not None
            ]
            tile = tile_id(action["tile"]) if action.get("tile") else None
            if not candidates:
                engine = "not-reached"  # the engine never gave us this window (someone claimed first)
            elif any(
                a.action == action["action"] and (tile is None or a.tile == tile)
                for c in candidates
                for a in c["legal"]
            ):
                engine = "engine-legal"
            else:
                engine = "engine-illegal"
            refused[f"{engine}|{action['action']}|{cause}"] += 1
        out["refused"] = dict(refused)
    except Exception:
        out["error"] = traceback.format_exc()[-800:]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "patterns", nargs="*", default=["*"], help="room directory globs under --root"
    )
    parser.add_argument("--root", type=Path, default=REPO / ".local/competition")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, help="write every game's result as JSON")
    args = parser.parse_args(argv)
    jobs = games(args.root, args.patterns)
    if not jobs:
        print("no archived rooms with replays found")
        return 1
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(check_game, jobs, chunksize=4))
    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False))

    total, refused = Counter(), Counter()
    for r in results:
        total["games"] += 1
        total[f"audit {r.get('audit', 'error')}"] += 1
        total["checks"] += r.get("checks", 0)
        total["decisions"] += r.get("decisions", 0)
        total["unmatched snapshots"] += r.get("unmatched", 0)
        total["field differences"] += sum(r.get("differences", {}).values())
        refused.update(r.get("refused", {}))
    total["rooms"] = len({r["room"] for r in results})
    print(json.dumps(dict(total), ensure_ascii=False))
    for key, count in refused.most_common():
        print(f"refused {count:6d}  {key}")
    bad = [
        r
        for r in results
        if "error" in r
        or r.get("audit") != "passed"
        or r.get("unmatched")
        or r.get("differences")
    ]
    bad_refused = sum(v for k, v in refused.items() if k.startswith("engine-illegal"))
    for r in bad[:10]:
        print(
            "problem",
            r["room"],
            r["game"],
            r.get("audit"),
            r.get("failures") or r.get("examples") or r.get("error"),
        )
    return 1 if bad or bad_refused else 0


if __name__ == "__main__":
    sys.exit(main())
