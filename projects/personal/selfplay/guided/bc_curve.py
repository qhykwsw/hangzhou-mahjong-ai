"""Finite pure-imitation learning curve with immutable timed Actor snapshots."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from . import bootstrap
from .overnight import run_child
from .run import atomic_json
from .statistics import interval


def snapshot_names():
    return [
        (0, "initial"),
        *[(m, f"seconds-{m * 60:06d}") for m in range(10, 60, 10)],
        (60, "final"),
    ]


def validate_records(records, jobs):
    def key(row):
        return row["group"], row["seed"], tuple(row["seats"]), row["focal"]

    if Counter(map(key, records)) != Counter(map(key, jobs)):
        raise ValueError(
            "evaluation receipts do not match the complete paired schedule"
        )
    for row in records:
        if (
            len(row["scores"]) != 4
            or sum(row["scores"]) != 0
            or row["policy"] != row["seats"][row["focal"]]
            or row["score"] != row["scores"][row["focal"]]
        ):
            raise ValueError("invalid focal settlement")


def curve_summary(records, targets, teacher, games):
    from .evaluation import summarize

    results = summarize(records, targets, games)
    differences = {}
    for index, target in enumerate(targets[:-1]):
        groups = results[target]["group_scores"]
        differences[target] = {
            "minus_teacher": interval(
                [
                    a - b
                    for a, b in zip(
                        groups, results[teacher]["group_scores"], strict=True
                    )
                ]
            )
        }
        if index:
            differences[target]["minus_previous"] = interval(
                [
                    a - b
                    for a, b in zip(
                        groups, results[targets[index - 1]]["group_scores"], strict=True
                    )
                ]
            )
    return {"results": results, "differences": differences}


def evaluate(root, plan):
    import torch
    from .artifacts import read_bundle
    from .collection import collect
    from .config import configuration
    from .evaluation import schedule
    from .networks import Actor, Critic
    from .resources import evaluation_resources

    output = root / "evaluation"
    output.mkdir(exist_ok=False)
    logs = [
        json.loads(line) for line in (root / "train/log.jsonl").read_text().splitlines()
    ]
    updates = [row for row in logs if row["kind"] == "update"]
    if not updates or any(
        row["phase"] != "bc"
        or row["stats"]["policy_loss"] != 0
        or row["stats"]["value_loss"] != 0
        for row in updates
    ):
        raise ValueError("not a pure imitation run")
    initial = torch.load(plan["initial"], map_location="cpu", weights_only=False)
    final = torch.load(root / "train/latest.pt", map_location="cpu", weights_only=False)
    if final["critic_optimizer"]["state"] or not final["actor_optimizer"]["state"]:
        raise ValueError("unexpected optimizer state")
    if any(
        not torch.equal(value, final["critic"][key])
        for key, value in initial["critic"].items()
    ):
        raise ValueError("critic changed during imitation")
    del initial, final
    atomic_json(
        root / "pure-bc-verification.json",
        {"verified": True, "updates": len(updates), "critic_unchanged": True},
    )
    torch.set_num_threads(2)
    bundle = read_bundle(plan["bundle"])
    teacher = bundle["teacher"]
    models, snapshots = {}, {}
    for minute, name in snapshot_names():
        path = root / "train/snapshots" / (name + ".pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["code_id"] != plan["source_id"] or state["config"] != plan["config"]:
            raise ValueError("snapshot identity mismatch")
        cfg = state["config"]
        actor = Actor(cfg["dim"], cfg["layers"], cfg["heads"])
        actor.load_state_dict(state.pop("actor"))
        target = f"model:minute-{minute:02d}"
        models[target] = actor.to("cuda").eval()
        snapshots[target] = state | {
            "minute": minute,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ordered = list(snapshots.values())
    for before, after in zip(ordered, ordered[1:]):
        if any(
            after[k] < before[k]
            for k in ["elapsed_seconds", "updates", "sgd_steps", "optimized_samples"]
        ):
            raise ValueError("snapshot progress decreased")
    targets = list(models) + [teacher]
    jobs = schedule(
        [e["id"] for e in bundle["entries"]],
        targets,
        plan["evaluation_games"],
        plan["evaluation_seed"],
        plan["evaluation_namespace"],
    )
    atomic_json(output / "schedule.json", jobs)
    atomic_json(output / "snapshots.json", snapshots)
    resources = evaluation_resources()
    resources.update(plan.get("evaluation_execution", {}))
    cfg = configuration(
        **(
            plan["config"]
            | {"workers": resources["workers"], "tables": resources["tables"]}
        )
    )
    cfg["inference_batch_limit"] = resources["inference_batch_limit"]
    atomic_json(output / "execution.json", resources)
    actor = next(iter(models.values()))
    critic = Critic().to("cuda").eval()
    deadline = time.time() + plan["evaluation_timeout_seconds"]
    records = []
    for first in range(0, plan["evaluation_games"] // 4, 64):
        stop = min(first + 64, plan["evaluation_games"] // 4)
        chunk = [j for j in jobs if first <= j["group"] < stop]
        _, finished, stats = collect(
            actor,
            critic,
            cfg,
            plan["bundle"],
            plan["bundle"],
            {
                "kind": "evaluate",
                "iteration": first,
                "jobs": chunk,
                "deadline": deadline,
                "labels": False,
            },
            models,
        )
        validate_records(finished, chunk)
        atomic_json(
            output / f"chunk-{first:04d}.json",
            {"records": finished, "resources": stats},
        )
        records.extend(finished)
        result = curve_summary(records, targets, teacher, stop * 4)
        result.update(
            snapshots=snapshots,
            teacher=teacher,
            source_id=plan["source_id"],
            games_per_target=stop * 4,
        )
        atomic_json(output / "results.json", result)
        atomic_json(
            output / "status.json",
            {
                "stage": "complete"
                if stop * 4 == plan["evaluation_games"]
                else "running",
                "games_per_target": stop * 4,
                "total_games": len(records),
                "updated": time.time(),
            },
        )


def suite(root):
    plan = json.loads((root / "plan.json").read_text())
    if bootstrap.verify_source() != plan["source_id"]:
        raise ValueError("source mismatch")
    if (root / "suite.json").exists():
        raise FileExistsError("study already started")
    state = {
        "stage": "training",
        "pid": os.getpid(),
        "started": time.time(),
        "tasks": {},
    }

    def save():
        atomic_json(root / "suite.json", state | {"updated": time.time()})

    save()
    try:
        deadline = time.time() + 3600
        commands = [
            (
                "training",
                [
                    sys.executable,
                    "-m",
                    "guided.run",
                    "--run",
                    str(root / "train"),
                    "--config",
                    str(root / "config.json"),
                    "--bundle",
                    plan["bundle"],
                    "--panel",
                    plan["bundle"],
                    "--initial",
                    plan["initial"],
                    "--deadline",
                    str(deadline),
                    "--bc-seconds",
                    "7200",
                    "--no-eval",
                    "--snapshot-seconds",
                    "600",
                ],
                3900,
            ),
            (
                "evaluation",
                [
                    sys.executable,
                    "-m",
                    "guided.bc_curve",
                    "evaluate",
                    "--root",
                    str(root),
                ],
                plan["evaluation_timeout_seconds"] + 300,
            ),
        ]
        for name, argv, timeout in commands:
            state["stage"] = name
            task = {"state": "running", "started": time.time(), "argv": argv}
            state["tasks"][name] = task
            save()

            def launched(pid):
                task["pid"] = pid
                save()

            run_child(argv, root / (name + ".log"), timeout, launched)
            task.update(state="complete", finished=time.time())
            save()
        state["stage"] = "complete"
    except BaseException as exc:
        state.update(stage="failed", error=str(exc))
        raise
    finally:
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["suite", "evaluate"])
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.action == "suite":
        suite(root)
    else:
        plan = json.loads((root / "plan.json").read_text())
        if bootstrap.verify_source() != plan["source_id"]:
            raise ValueError("source mismatch")
        evaluate(root, plan)


if __name__ == "__main__":
    main()
