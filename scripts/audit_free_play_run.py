#!/usr/bin/env python3
"""Archive a public match and audit execution only; never export research feedback."""

import argparse
from collections import Counter
from pathlib import Path
import json
import time

from mahjong_ai.platform.calibration import _request
from mahjong_ai.platform.live_workflow import game_rows
from mahjong_ai.persistence.artifacts import atomic_json, digest, read_json


def audit(root, base, insecure=False):
    state = read_json(root / "match.json")
    rows = game_rows(
        _request(base, f"/api/test-rooms/{state['room_id']}/games", insecure=insecure)
    )
    reports, failures = [], []
    for row in rows:
        if row["status"] != "finished":
            continue
        if sum(r["batch"] == row["batch"] for r in rows) != 1:
            raise ValueError("ambiguous batch; export by game_id through portal")
        target = root / "replays" / (digest(row["game_id"])[:24] + ".json")
        if target.exists():
            replay = read_json(target)
        else:
            time.sleep(0.25)
            replay = _request(
                base,
                f"/api/test-rooms/{state['room_id']}/games/{row['batch']}/events",
                insecure=insecure,
            )
            atomic_json(target, replay)
        if replay["game_id"] != row["game_id"]:
            raise ValueError("replay game_id mismatch")
        events = sorted(
            [e for b in replay["blocks"] for e in b["events"]], key=lambda e: e["seq"]
        )
        path = root / "player" / ("game-" + digest(row["game_id"])[:24] + ".json")
        saved = read_json(path)
        snap = saved.get("final", {}).get("snapshot") or {}
        seat = snap.get("seat")
        end = [e for e in events if e["type"] == "game_ended"]
        score_ok = bool(end) and end[-1]["data"]["final_scores"] == snap.get("scores")
        timeouts = [
            e
            for e in events
            if e["type"] == "timeout" and (e.get("data") or {}).get("kind") == "discard"
        ]
        reports.append(
            {
                "game_id": row["game_id"],
                "hands": len({b["round_no"] for b in replay["blocks"]}),
                "complete_blocks": all(
                    not b.get("truncated") for b in replay["blocks"]
                ),
                "score_matches_final": score_ok,
                "seat": seat,
                "own_discard_timeouts": sum(e["seat"] == seat for e in timeouts),
                "other_discard_timeouts": sum(e["seat"] != seat for e in timeouts),
            }
        )
        records = (
            [
                json.loads(line)
                for line in path.with_suffix(".jsonl").read_text().splitlines()
            ]
            if path.with_suffix(".jsonl").exists()
            else []
        )
        timeout_context = []
        for e in timeouts:
            if e["seat"] != seat:
                continue
            preceding = [
                x
                for x in events
                if x["seq"] < e["seq"]
                and x.get("seat") == seat
                and x["type"] in ("tile_drawn", "chi", "peng", "gang")
            ]
            previous = preceding[-1] if preceding else None
            following = [
                x
                for x in events
                if x["seq"] > e["seq"] and x["type"] in ("tile_drawn", "tile_discarded")
            ]
            boundary = following[0]["seq"] if following else e["seq"] + 1
            accepted = [
                r
                for r in records
                if e["seq"] <= r["seq"] < boundary
                and r["accepted"]
                and r["action"]["action"] == "discard"
            ]
            timeout_context.append(
                {
                    "seq": e["seq"],
                    "ts": e.get("ts"),
                    "previous_turn_event": previous,
                    "seconds_since_turn_event": e.get("ts", 0) - previous.get("ts", 0)
                    if previous
                    else None,
                    "accepted_discard_seqs_after_timeout": [r["seq"] for r in accepted],
                }
            )
        reports[-1]["own_timeout_context"] = timeout_context
        for r in records:
            if r["accepted"]:
                continue
            cause, witness = "unexplained", None
            for e in [e for e in events if e["seq"] > r["seq"]][:8]:
                if e["type"] in ("chi", "peng", "gang"):
                    if (
                        e.get("tile") == r["snapshot"].get("last_discard")
                        and e.get("seat") != seat
                    ):
                        cause, witness = "another_player_claimed", e["seq"]
                    break
                if (
                    e["type"] == "timeout"
                    and (e.get("data") or {}).get("kind") == "response"
                ):
                    cause, witness = "response_window_expired", e["seq"]
                    break
                if e["type"] in ("tile_discarded", "tile_drawn", "round_ended"):
                    break
            if r["action"]["action"] != "pass" and cause == "unexplained":
                cause = "unexplained_nonpass"
            failures.append(
                {
                    "game_id": r["game_id"],
                    "seq": r["seq"],
                    "action": r["action"],
                    "cause": cause,
                    "witness": witness,
                }
            )
    complete = len(reports) == state["match"]["config"]["M"] and all(
        r["hands"] == state["match"]["config"]["Rounds"]
        and r["complete_blocks"]
        and r["score_matches_final"]
        for r in reports
    )
    clean = (
        complete
        and all(r["own_discard_timeouts"] == 0 for r in reports)
        and not any(r["cause"].startswith("unexplained") for r in failures)
    )
    summary_path = root / "summary.json"
    result = {
        "status": "passed" if clean else "issues_or_incomplete",
        "archival_complete": complete,
        "match_completed": summary_path.exists()
        and read_json(summary_path)["status"] == "completed",
        "mode": "competition",
        "feedback_to_iteration": False,
        "games": reports,
        "action_failures": failures,
        "failure_counts": dict(Counter(r["cause"] for r in failures)),
        "own_discard_timeouts": sum(r["own_discard_timeouts"] for r in reports),
        "other_discard_timeouts": sum(r["other_discard_timeouts"] for r in reports),
    }
    if not result["match_completed"]:
        result["status"] = "issues_or_incomplete"
    atomic_json(root / "execution-audit.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--base-url", default="https://10.240.169.190:18080")
    p.add_argument("--insecure", action="store_true")
    args = p.parse_args()
    result = audit(Path(args.run).resolve(), args.base_url, args.insecure)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("games", "action_failures")},
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
