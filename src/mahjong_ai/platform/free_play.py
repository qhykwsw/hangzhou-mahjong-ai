"""One frozen public matchmaking session. No research feedback or training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

from mahjong_ai.interfaces.cli import spec
from mahjong_ai.platform.live_workflow import summarize_player
from mahjong_ai.platform.official import OfficialAdapter
from mahjong_ai.platform.online import (
    ApiError,
    PlatformClient,
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


def run_match(
    credentials,
    policy,
    output,
    base,
    insecure=False,
    seconds=1800,
    client=None,
    runner_class=TournamentRunner,
):
    if seconds <= 0:
        raise ValueError("positive time budget required")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    token = read_json(credentials)["token"]
    client = client or PlatformClient(base, token, insecure)
    protocol = {
        "mode": "competition",
        "feedback_to_iteration": False,
        "token_hash": digest(token),
        "policy_id": policy.identity,
        "policy_version": policy.version,
        "runtime": runtime_fingerprint(),
        "base": base.rstrip("/"),
    }
    state_path = root / "match.json"
    # Serialize this identity across separate output directories, too.
    with (
        exclusive_lock(
            Path(credentials).resolve().parent
            / (".free-" + digest(token)[:24] + ".lock")
        ),
        exclusive_lock(root / ".match.lock"),
    ):
        if state_path.exists():
            state = read_json(state_path)
            if state["protocol"] != protocol:
                raise ValueError(
                    "match protocol changed; restore the original policy/runtime"
                )
            if state["status"] == "completed":
                return read_json(root / "summary.json")
            if state["status"] == "matching_unknown":
                raise RuntimeError(
                    "match result unknown; inspect /api/me and recover room_id before retrying"
                )
        else:
            me = client.request("GET", "/api/me")
            if me.get("tournament_id"):
                raise ValueError("free play requires a global token")
            if me.get("active_games"):
                raise ValueError(
                    "identity already has active games; resume their existing workflow"
                )
            # Validate and copy the artifact before joining: fourth seat starts immediately.
            policy.build()
            frozen = policy
            if policy.model_path:
                shutil.copy2(policy.model_path, root / "model.npz")
                frozen = replace(policy, model_path=str(root / "model.npz"))
            atomic_json(root / "policy.json", frozen.as_dict())
            state = {
                "protocol": protocol,
                "status": "matching_unknown",
                "user_id": me["user_id"],
            }
            atomic_json(state_path, state)
            try:
                match = client.request("POST", "/api/match", {"M": 16})
            except ApiError as exc:
                # A definitive rejection is safe to retry explicitly. Never retry
                # a lost response: a room may already be running by then.
                state.update(
                    status="match_rejected",
                    error={"status": exc.status, "code": exc.code},
                )
                atomic_json(state_path, state)
                raise
            state.update(
                status="running", match=_redact(match), room_id=match["room_id"]
            )
            atomic_json(state_path, state)
        if "room_id" not in state:
            raise RuntimeError(
                "match was rejected; inspect the recorded reason and use a new output directory"
            )
        frozen = spec(str(root / "policy.json"))
        if frozen.identity != protocol["policy_id"]:
            raise ValueError("frozen policy artifact changed")
        tid = state["room_id"]
        deadline = time.monotonic() + seconds
        errors = state.setdefault("runtime_errors", [])
        while time.monotonic() < deadline:
            try:
                runner = runner_class(
                    client,
                    frozen,
                    OfficialAdapter(),
                    root / "player",
                    workers=16,
                    join=False,
                )
                runner.run(tid, max(0.01, deadline - time.monotonic()))
                break
            except (ConnectionError, ApiError) as exc:
                errors.append(
                    {
                        "type": type(exc).__name__,
                        "status": getattr(exc, "status", None),
                        "code": getattr(exc, "code", None),
                    }
                )
                atomic_json(state_path, state)
                if isinstance(exc, ApiError) and exc.status not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    break
                time.sleep(min(2, max(0, deadline - time.monotonic())))
        tournament_path = root / "player" / "tournament.json"
        tournament = read_json(tournament_path) if tournament_path.exists() else {}
        try:
            tournament = client.request("GET", f"/api/tournaments/{_id(tid)}")
            atomic_json(tournament_path, _redact(tournament))
        except ApiError as exc:
            if exc.status != 404:
                raise
            state["room_unavailable"] = True
        # Coordinator may observe finished before a worker captures its final snapshot.
        for gid in tournament.get("my_games", []):
            path = root / "player" / ("game-" + digest(gid)[:24] + ".json")
            saved = (
                read_json(path)
                if path.exists()
                else {"responses": [], "policy_id": frozen.identity}
            )
            if saved.get("status") == "finished":
                continue
            try:
                final = client.request("GET", f"/api/games/{_id(gid)}/state?seq=0")
            except ApiError as exc:
                if exc.status == 404:
                    continue
                raise
            if final.get("finished"):
                saved.update(game_id=gid, status="finished", final=_redact(final))
                atomic_json(path, saved)
        player = summarize_player(root / "player")
        expected = state["match"]["config"]["M"]
        valid_finals, own_timeouts = 0, 0
        for path in (root / "player").glob("game-*.json"):
            saved = read_json(path)
            snap = saved.get("final", {}).get("snapshot") or {}
            scores, seat = snap.get("scores"), snap.get("seat")
            if (
                saved.get("status") == "finished"
                and isinstance(seat, int)
                and 0 <= seat < 4
                and isinstance(scores, list)
                and len(scores) == 4
                and sum(scores) == 0
            ):
                valid_finals += 1
            own_timeouts += sum(
                e.get("type") == "timeout"
                and e.get("seat") == seat
                and (e.get("data") or {}).get("kind") == "discard"
                for e in saved.get("events", [])
            )
        ranking = next(
            (
                r
                for r in tournament.get("ranking", [])
                if r.get("user_id") == state["user_id"]
            ),
            None,
        )
        score_matches = (
            ranking is not None and ranking.get("total_score") == player["score_sum"]
        )
        complete = (
            tournament.get("status") == "finished"
            and valid_finals == expected
            and score_matches
        )
        summary = {
            "status": "completed" if complete else "incomplete",
            "mode": "competition",
            "feedback_to_iteration": False,
            "policy_id": frozen.identity,
            "family": frozen.family,
            "expected_games": expected,
            "valid_final_snapshots": valid_finals,
            "rounds_per_game": state["match"]["config"]["Rounds"],
            "room_status": tournament.get("status"),
            "score_matches_ranking": score_matches,
            "ranking": ranking,
            "player": player,
            "observed_own_discard_timeouts": own_timeouts,
            "runtime_errors": errors,
            "output": str(root),
            "note": "Live event history may have gaps; zero observed timeouts is not a full replay audit.",
        }
        atomic_json(root / "summary.json", summary)
        state.update(status=summary["status"])
        atomic_json(state_path, state)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--policy", default="rule")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="https://10.240.169.190:18080")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    result = run_match(
        args.credentials,
        spec(args.policy),
        args.output,
        args.base_url,
        args.insecure,
        args.seconds,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
