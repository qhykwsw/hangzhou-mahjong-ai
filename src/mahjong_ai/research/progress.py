"""Read-only dashboard progress from registry state and durable worker receipts."""

from collections import Counter, defaultdict
from pathlib import Path
import time

from mahjong_ai.persistence.artifacts import read_json


def branch_progress(state):
    cycles = state["cycles"]
    if not cycles or cycles[-1]["experiment"] != state["id"]:
        return {}
    cycle = cycles[-1]
    root = Path(state["root"]) / "jobs"
    jobs = defaultdict(list)
    competition_done, competition_total = Counter(), Counter()
    for job in state["jobs"].values():
        if job["experiment"] != state["id"]:
            continue
        if job.get("iteration"):
            jobs[job["iteration"]].append(job)
        if job.get("scope") != "competition" or job["cycle"] != cycle["id"]:
            continue
        for assignment in job["assignments"]:
            competition_total.update(assignment["players"])
        if job["status"] == "completed" and not job.get("cancelled"):
            for row in job.get("summary", []):
                competition_done[row["candidate_id"]] += row["appearances"]
        else:
            for path in (root / job["id"] / "matches").glob("*.json"):
                competition_done.update(read_json(path)["players"])
    iterations = {
        i["branch"]: i
        for i in state["iterations"].values()
        if i["experiment"] == state["id"] and i["cycle"] == cycle["id"]
    }
    output = {}
    for bid, branch in state["branches"].items():
        if branch["archived"]:
            continue
        iteration = iterations.get(bid)
        item = dict(
            iteration=iteration["number"] if iteration else None,
            research="waiting" if bid in cycle["targets"] else "not_selected",
            preparation=[],
            evaluation=[],
            outcome=None,
            error=None,
            competition=None,
        )
        roster = cycle.get("roster", {}).get(bid)
        if roster:
            cid = roster["candidate_id"]
            item["competition"] = dict(
                completed_games=competition_done[cid],
                total_games=competition_total[cid],
                status="completed" if cycle["status"] == "completed" else "running",
            )
        if not iteration:
            output[bid] = item
            continue
        owned_jobs = jobs[iteration["id"]]
        item["research"] = (
            "submitted"
            if iteration["candidates"]
            else ("failed" if iteration["status"] == "completed" else "running")
        )
        item["error"] = iteration.get("error")
        if iteration["status"] == "completed":
            item["outcome"] = (
                "improved"
                if iteration.get("improved")
                else "retained"
                if iteration["results"]
                else "cancelled"
                if iteration.get("error") == "experiment closed"
                else "failed"
            )
        for cid in iteration["candidates"]:
            candidate = state["candidates"][cid]
            job = next(
                (
                    j
                    for j in owned_jobs
                    if j["kind"] == "prepare" and cid in j["candidates"]
                ),
                None,
            )
            record = candidate.get("training")
            ready = bool(candidate.get("spec"))
            status = (
                "failed"
                if candidate["status"] == "failed"
                else ("completed" if ready else job["status"] if job else "waiting")
            )
            completed = len(record["episodes"]) if record else None
            source = (
                "partial"
                if record and record.get("interrupted")
                else "completed"
                if record
                else None
            )
            seconds = None
            if job:
                checkpoint = root / job["id"] / cid / "training-checkpoint.json"
                if not record and checkpoint.exists():
                    completed = len(read_json(checkpoint).get("episodes", []))
                    source = "checkpoint"
                seconds = job.get("spent", 0) + job.get("usage", {}).get(
                    "wall_seconds", 0
                )
                if job["status"] == "running" and job.get("started_at"):
                    seconds += max(0, time.time() - job["started_at"])
            item["preparation"].append(
                dict(
                    candidate_id=cid,
                    status="not_required" if ready and record is None else status,
                    completed_games=completed,
                    total_games=iteration.get("training_games"),
                    source=source,
                    elapsed_seconds=seconds,
                    error=candidate.get("error"),
                )
            )
        for role, cid in [
            ("incumbent", iteration["incumbent"]),
            *[("candidate", c) for c in iteration["candidates"]],
        ]:
            shards = [
                j
                for j in owned_jobs
                if j["kind"] == "match" and j.get("candidate") == cid
            ]
            # 只有全部分片完成才算这个候选的评估结果；进行中的半程均值既不
            # 属于选优口径，也不应该在卡片上显示成「完成 · 均分」。
            settled = bool(shards) and all(
                job["status"] == "completed" for job in shards
            )
            result = iteration["results"].get(cid) if settled else None
            if result:
                status = "completed"
            elif any(job.get("cancelled") for job in shards):
                status = "cancelled"
            elif any(job["status"] == "paused" for job in shards):
                status = "paused"
            elif any(job["status"] == "running" for job in shards):
                status = "running"
            elif any(job["status"] in ("queued", "admitted") for job in shards):
                status = "queued"
            elif shards and all(job["status"] == "completed" for job in shards):
                status = "completed"
            else:
                status = shards[-1]["status"] if shards else "waiting"
            completed = 0
            hand = None
            if result:
                completed = result["appearances"]
            else:
                for job in shards:
                    folder = root / job["id"]
                    completed += sum(1 for _ in (folder / "matches").glob("*.json"))
                    progress = folder / "progress.json"
                    if (
                        hand is None
                        and job["status"] == "running"
                        and progress.exists()
                    ):
                        hand = read_json(progress).get("hand")
            item["evaluation"].append(
                dict(
                    role=role,
                    candidate_id=cid,
                    status=status,
                    completed_games=completed,
                    total_games=cycle["config"]["evaluation_games"],
                    mean_utility=result["mean_utility"] if result else None,
                    hand_mean_utility=(
                        result.get("hand_mean_utility") if result else None
                    ),
                    hand_appearances=(
                        result.get("hand_appearances") if result else None
                    ),
                    paired=(
                        result.get("paired") if result and role == "candidate" else None
                    ),
                    hand=hand,
                )
            )
        output[bid] = item
    return output
