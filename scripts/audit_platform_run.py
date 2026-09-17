#!/usr/bin/env python3
"""Audit finished platform games, ledger consistency, timeouts and 409 causes."""

import argparse
from collections import Counter
import json
from pathlib import Path

from mahjong_ai.platform.replay_audit import audit_replay
from mahjong_ai.persistence.artifacts import atomic_json, read_json
from mahjong_ai.domain.rules import payments
from mahjong_ai.platform.live_workflow import summarize_player


def crossed_pass_window(row, earlier, events):
    """Identify a delayed peng-pass consuming chi before the next local response."""
    snap = row["snapshot"]
    if (
        row["action"]["action"] not in ("pass", "chi")
        or snap["phase"] != "response_chi"
    ):
        return []
    previous = next(
        (
            r
            for r in reversed(earlier)
            if r["game_id"] == row["game_id"]
            and r["accepted"]
            and r["action"]["action"] == "pass"
        ),
        None,
    )
    if previous is None:
        return []
    before = previous["snapshot"]
    if (
        before["phase"] != "response_peng"
        or before["seat"] != snap["seat"]
        or before["round_no"] != snap["round_no"]
        or before["last_discard"] != snap["last_discard"]
    ):
        return []
    window = [e for e in events if previous["seq"] < e["seq"] <= row["seq"]]
    if not window or any(e["type"] not in ("pass", "timeout") for e in window):
        return []
    passed = window[-1]
    timeout = next(
        (
            e
            for e in window
            if e["type"] == "timeout"
            and e.get("seat") == snap["seat"]
            and (e.get("data") or {}).get("window") == "peng"
        ),
        None,
    )
    if (
        timeout
        and passed["type"] == "pass"
        and passed.get("seat") == snap["seat"]
        and passed["seq"] == row["seq"]
    ):
        return [timeout["seq"], passed["seq"]]
    return []


def audit(root):
    workflow = read_json(root / "workflow.json")
    cfg = workflow["protocol"]["config"]
    games, reports, errors = {}, [], []
    for path in sorted((root / "replays").glob("*.json")):
        if path.name.endswith(".audit.json"):
            continue
        replay = read_json(path)
        report = audit_replay(replay, cfg["BaseScore"], cfg["YouCaiBiKao"])
        ordered = sorted(
            [e for b in replay["blocks"] for e in b["events"]], key=lambda e: e["seq"]
        )
        previous = [0] * 4
        score_checks = []
        for e in ordered:
            if e["type"] != "round_ended":
                continue
            d = e["data"]
            expected = (
                [0] * 4
                if d["draw"]
                else list(payments(e["seat"], d["dealer"], d["fan"], cfg["BaseScore"]))
            )
            actual = d["scores"]  # actual API: per-hand delta, not cumulative scores
            score_checks.append(
                {
                    "round_no": d["round_no"],
                    "ok": actual == expected,
                    "expected": expected,
                    "actual": actual,
                }
            )
            previous = [previous[i] + actual[i] for i in range(4)]
        final = [e for e in ordered if e["type"] == "game_ended"]
        final_ok = (
            bool(final)
            and final[-1]["data"]["final_scores"] == previous
            and sum(previous) == 0
        )
        timeouts = [
            e
            for e in ordered
            if e["type"] == "timeout" and (e.get("data") or {}).get("kind") == "discard"
        ]
        games[replay["game_id"]] = {"events": ordered, "final": previous}
        reports.append(
            {
                "game_id": replay["game_id"],
                "replay_status": report["status"],
                "hands": report["hands"],
                "checks": report["checks"],
                "rule_failures": report["failures"],
                "score_checks": score_checks,
                "final_score_consistent": final_ok,
                "discard_timeouts": len(timeouts),
                "event_types": report["event_types"],
            }
        )
    players = []
    for directory in sorted(p for p in root.glob("player-*") if p.is_dir()):
        summary = summarize_player(directory)
        for path in directory.glob("game-*.jsonl"):
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            for index, row in enumerate(rows):
                if row["accepted"]:
                    continue
                events = games.get(row["game_id"], {}).get("events", [])
                following = [e for e in events if e["seq"] > row["seq"]]
                cause = "unexplained"
                witnesses = []
                for e in following[:8]:
                    if e["type"] in ("chi", "peng", "gang"):
                        if e.get("tile") == row["snapshot"]["last_discard"]:
                            cause = "another_player_claimed_before_pass"
                            witnesses = [e["seq"], e["type"]]
                        break
                    if (
                        e["type"] == "timeout"
                        and (e.get("data") or {}).get("kind") == "response"
                    ):
                        cause = "response_window_expired"
                        witnesses = [e["seq"], "timeout"]
                        break
                    if e["type"] in ("tile_discarded", "tile_drawn", "round_ended"):
                        break
                crossed = crossed_pass_window(row, rows[:index], events)
                if crossed:
                    cause, witnesses = "pass_accepted_in_next_window", crossed
                elif row["action"]["action"] != "pass":
                    cause = "unexplained_nonpass"
                errors.append(
                    {
                        "player": directory.name,
                        "game_id": row["game_id"],
                        "seq": row["seq"],
                        "action": row["action"],
                        "cause": cause,
                        "witnesses": witnesses,
                    }
                )
        players.append({"player": directory.name, **summary})
    all_ok = (
        len(reports) == cfg["M"]
        and all(
            r["hands"] == cfg["Rounds"]
            and r["replay_status"] == "passed"
            and r["final_score_consistent"]
            and all(c["ok"] for c in r["score_checks"])
            and r["discard_timeouts"] == 0
            for r in reports
        )
        and not any(e["cause"].startswith("unexplained") for e in errors)
        and all(p["finished_games"] == cfg["M"] for p in players)
        and sum(p["score_sum"] for p in players) == 0
    )
    ledger = None
    ids = root / "player-identities.json"
    final = root / "final-tournament.json"
    if ids.exists() and final.exists():
        ranking = {r["user_id"]: r["total_score"] for r in read_json(final)["ranking"]}
        identities = read_json(ids)
        ledger = [
            {
                "player": i,
                "snapshot_total": p["score_sum"],
                "official_total": ranking[identities[i]],
                "ok": p["score_sum"] == ranking[identities[i]],
            }
            for i, p in enumerate(players)
        ]
        all_ok = all_ok and all(x["ok"] for x in ledger)
    return {
        "status": "passed" if all_ok else "failed_or_incomplete",
        "config": cfg,
        "games": reports,
        "players": players,
        "ledger": ledger,
        "action_races": errors,
        "race_counts": dict(Counter(e["cause"] for e in errors)),
        "scope": "observed official games and paths only; this is compatibility evidence, not paired strategy improvement",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(Path(args.run))
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "games": len(result["games"]),
                "scores": [p["score_sum"] for p in result["players"]],
                "race_counts": result["race_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
