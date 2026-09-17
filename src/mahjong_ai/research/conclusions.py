"""Cumulative branch evidence, separate from the agent's interpretation."""

from copy import deepcopy


STATUSES = {"verified", "refuted", "inconclusive", "untested"}
FAILURE_KINDS = {
    "none",
    "implementation",
    "resource_limit",
    "infrastructure",
    "unknown",
}
COUNTERS = ("errors", "timeouts", "fallback_actions")


def ledger(state, branch=None):
    """Project all completed attempts; never infer hypothesis validity from a loss."""
    records = []
    experiments = {e["id"]: e for e in state.get("experiments", [])}
    by_iteration = {}
    jobs_by_iteration = {}
    for job in state.get("jobs", {}).values():
        if job.get("iteration"):
            jobs_by_iteration.setdefault(job["iteration"], []).append(job)
    for candidate in state["candidates"].values():
        if candidate.get("iteration"):
            by_iteration.setdefault(candidate["iteration"], []).append(candidate)
    for iteration in state["iterations"].values():
        if iteration["status"] != "completed" or (
            branch and iteration["branch"] != branch
        ):
            continue
        cycle = state["cycles"][iteration["cycle"] - 1]
        config = cycle["config"]
        expected = config["evaluation_games"]
        reference = iteration["results"].get(iteration["incumbent"]) or {}
        candidates = by_iteration.get(iteration["id"], [])
        # Agent failures before draft creation still carry a durable lesson.
        for candidate in candidates or [None]:
            cid = candidate["id"] if candidate else None
            record_id = "candidate:" + cid if cid else "iteration:" + iteration["id"]
            result = iteration["results"].get(cid) or {}
            errors = []
            if candidate:
                for check in candidate.get("checks", []):
                    if not check["ok"]:
                        errors.append({"stage": "preflight", "reason": check["error"]})
                if candidate.get("error"):
                    errors.append(
                        {"stage": "preparation", "reason": candidate["error"]}
                    )
                training = candidate.get("training") or {}
                if training.get("interrupted"):
                    errors.append(
                        {"stage": "training", "reason": training["interrupted"]}
                    )
            else:
                training = {}
            if iteration.get("error"):
                errors.append({"stage": "iteration", "reason": iteration["error"]})
            errors.extend(
                deepcopy(failure)
                for failure in iteration.get("execution_failures", [])
                if failure.get("candidate_id") in (None, cid, iteration["incumbent"])
            )
            diagnostics_truncated = 0
            for job in jobs_by_iteration.get(iteration["id"], []):
                if job.get("candidate") not in (cid, iteration["incumbent"]):
                    continue
                diagnostics_truncated += job.get("diagnostics_truncated", 0)
                for diagnostic in job.get("diagnostics", []):
                    failure = {
                        "stage": "evaluation",
                        "candidate_id": job.get("candidate"),
                        "reason": diagnostic.get("message"),
                        "type": diagnostic.get("type"),
                    }
                    if failure not in errors:
                        errors.append(failure)
            counters = {
                role: {key: row.get(key, 0) for key in COUNTERS}
                for role, row in (("candidate", result), ("incumbent", reference))
            }
            complete = all(
                row.get("appearances") == expected
                and row.get("mean_utility") is not None
                for row in (result, reference)
            )
            contaminated = any(v for row in counters.values() for v in row.values())
            outcome = (
                "evaluated_with_errors"
                if complete and contaminated
                else "evaluated"
                if complete
                else "not_evaluated"
            )
            history = iteration.get("conclusions", {}).get(record_id, [])
            records.append(
                {
                    "record_id": record_id,
                    "branch": iteration["branch"],
                    "iteration_id": iteration["id"],
                    "iteration": iteration["number"],
                    "candidate_id": cid,
                    "hypothesis": candidate.get("hypothesis", "")
                    if candidate
                    else (iteration.get("plan") or {}).get("direction", "未提交候选"),
                    "outcome": outcome,
                    "assessment": deepcopy(history[-1]) if history else None,
                    "revision_count": len(history),
                    "evidence": {
                        "experiment": iteration["experiment"],
                        "runtime": experiments.get(iteration["experiment"], {}).get(
                            "runtime"
                        ),
                        "candidate_package_hash": (candidate.get("spec") or {}).get(
                            "package_hash"
                        )
                        if candidate
                        else None,
                        "cycle": iteration["cycle"],
                        "incumbent": iteration["incumbent"],
                        "parent": candidate.get("parent") if candidate else None,
                        "opponents": list(cycle["reference"]),
                        "game": deepcopy(config["game"]),
                        "timeouts": deepcopy(config["timeouts"]),
                        "resources": deepcopy(config["resources"]),
                        "evaluation_games": expected,
                        "appearances": result.get("appearances", 0),
                        "reference_appearances": reference.get("appearances", 0),
                        "mean_utility": result.get("mean_utility"),
                        "reference_mean_utility": reference.get("mean_utility"),
                        "mean_delta": result["mean_utility"] - reference["mean_utility"]
                        if complete
                        else None,
                        "paired": deepcopy(result.get("paired")),
                        "promotion": deepcopy(result.get("promotion")),
                        "runtime_counters": counters,
                        "failures": errors,
                        "diagnostics_truncated": diagnostics_truncated,
                        "training_status": training.get("status"),
                        "training_games_completed": len(training["episodes"])
                        if "episodes" in training
                        else None,
                    },
                }
            )
    return records


def validate_assessment(
    record, status, conclusion, conditions, failure_kind, failure_reason
):
    if status not in STATUSES or failure_kind not in FAILURE_KINDS:
        raise ValueError("未知的结论状态或失败类型")
    for name, value in (("conclusion", conclusion), ("conditions", conditions)):
        if not isinstance(value, str) or not value.strip() or len(value) > 3000:
            raise ValueError(f"{name} 必须为 1..3000 字的非空描述")
    if not isinstance(failure_reason, str) or len(failure_reason) > 3000:
        raise ValueError("failure_reason 必须为不超过 3000 字的描述")
    if status in ("verified", "refuted") and record["outcome"] != "evaluated":
        raise ValueError("未完成无运行错误的完整配对评估，不能验证或否定策略假设")
    if status == "untested" and record["outcome"] == "evaluated":
        raise ValueError("已有完整有效评估；证据不足请标记 inconclusive")
    if (record["outcome"] != "evaluated" or record["evidence"]["failures"]) and (
        failure_kind == "none" or not failure_reason.strip()
    ):
        raise ValueError("未有效验证的尝试必须记录失败类型与原因；未知原因使用 unknown")
    if (failure_kind == "none") != (not failure_reason.strip()):
        raise ValueError("失败类型与失败原因必须同时填写，或同时为空")
    return {
        "status": status,
        "conclusion": conclusion.strip(),
        "conditions": conditions.strip(),
        "failure_kind": failure_kind,
        "failure_reason": failure_reason.strip(),
    }
