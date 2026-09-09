"""One admitted compute job. All strategy code lives in isolated child processes."""

import os
from pathlib import Path
import random
import resource
import shutil
import signal
import sys
import threading
import time

from mahjong_ai.execution.candidate_runtime import (
    CandidateProcess,
    JobCancelled,
    check_cancelled,
    train_package,
    tree_hash,
)
from mahjong_ai.execution.dependencies import prepare_environment
from mahjong_ai.domain.engine import MahjongEngine, derive_seed
from mahjong_ai.strategies.common import PolicySpec
from mahjong_ai.persistence.artifacts import (
    digest,
    atomic_json,
    read_json,
    runtime_fingerprint,
)
from mahjong_ai.domain.types import RoundConfig


def prepare(task, root, deadline):
    rows = {}
    for cid, item in task["packages"].items():
        check_cancelled()
        out = root / cid
        out.mkdir(exist_ok=True)
        artifact = out / "artifact.json"
        if artifact.exists():
            row = read_json(artifact)
            PolicySpec(**row["spec"]).identity
            rows[cid] = row
            continue
        try:
            source = Path(item["input"])
            if tree_hash(source) != item["hash"]:
                raise ValueError("submitted artifact changed")
            environment = out / "environment"
            if task.get("dependency_root"):
                import numpy as np

                environment = Path(task["dependency_root"]) / digest(
                    [sorted(item["requirements"]), sys.version, np.__version__]
                )
            python, wheels = prepare_environment(
                item["requirements"], environment, deadline
            )
            probe = CandidateProcess(source, python)
            try:
                capabilities = probe.call(
                    {"op": "describe"}, min(10, max(0.01, deadline - time.monotonic()))
                )
            finally:
                probe.close()
            model = out / "training-model"
            training = None
            if capabilities["train"]:

                def opponents(index):
                    rng = random.Random(
                        derive_seed(task["seed"], "training-opponents", cid, index)
                    )
                    return (
                        [PolicySpec(**p) for p in rng.sample(task["opponents"], 3)]
                        if task["opponents"]
                        else [PolicySpec() for _ in range(3)]
                    )

                training = train_package(
                    source,
                    model,
                    RoundConfig(**task["config"]),
                    opponents,
                    task["training_games"],
                    derive_seed(task["seed"], "training", cid),
                    deadline,
                    python,
                    policy_seed=derive_seed(task["seed"], "policy", cid),
                )
            package = out / "frozen"
            if package.exists():
                shutil.rmtree(package)
            shutil.copytree(source, package)
            if training is not None:
                if (package / "model").exists():
                    shutil.rmtree(package / "model")
                shutil.copytree(model, package / "model")
            if wheels:
                atomic_json(package / "dependencies.json", wheels)
            probe = CandidateProcess(package, python)
            try:
                probe.call(
                    {
                        "op": "build",
                        "combination": task["combination"],
                        "definition": task.get("definition"),
                    },
                    min(10, max(0.01, deadline - time.monotonic())),
                )
            finally:
                probe.close()
            spec = PolicySpec(
                family="code",
                version=cid,
                package_path=str(package),
                package_hash=tree_hash(package),
                python_path=python,
                action_seconds=task["timeouts"]["discard"],
                response_seconds=task["timeouts"]["response"],
            ).as_dict()
            rows[cid] = dict(status="ready", spec=spec, training=training)
            atomic_json(artifact, rows[cid])
        except JobCancelled:
            raise
        except Exception as exc:
            rows[cid] = dict(
                status="failed", error=f"{type(exc).__name__}: {exc}"[:4000]
            )
        atomic_json(
            root / "progress.json",
            {"prepared": len(rows), "total": len(task["packages"])},
        )
    return {"status": "completed", "candidates": rows}


def play(task, root):
    from mahjong_ai.evaluation.schedule import standings

    receipts = []
    for assignment in task["assignments"]:
        check_cancelled()
        path = root / "matches" / f"{assignment['index']:06d}.json"
        if path.exists():
            receipts.append(read_json(path))
            continue
        players = [
            PolicySpec(**task["policies"][cid]).build() for cid in assignment["players"]
        ]
        engine = MahjongEngine(record_events=False)
        engine.progress = lambda progress: atomic_json(
            root / "progress.json",
            dict(
                completed_matches=len(receipts),
                total_matches=len(task["assignments"]),
                hand=progress,
            ),
        )
        try:
            seed = derive_seed(task["seed"], "wall", assignment["block"])
            seeds = [
                derive_seed(task["seed"], "policy", assignment["block"], seat)
                for seat in range(4)
            ]
            result = engine.run_round(
                players, RoundConfig(**task["config"]), seed, policy_seeds=seeds
            )
            if not result.completed:
                raise ValueError("incomplete evaluation settlement")
            receipt = dict(
                players=assignment["players"],
                ranks=result.rank_by_seat,
                scores=result.total_score_by_seat,
                hands=result.rounds_played,
                execution=[dict(p.stats) for p in players],
                diagnostics=[
                    dict(candidate_id=cid, error=p.last_error)
                    for cid, p in zip(assignment["players"], players)
                    if p.last_error
                ],
            )
            atomic_json(path, receipt)
            receipts.append(receipt)
        finally:
            for p in players:
                p.close()
    roster = list(task["policies"])
    summary = standings(roster, receipts)
    result = {"status": "completed", "summary": summary, "matches": len(receipts)}
    if task.get("candidate"):
        result["focal"] = next(
            r for r in summary if r["candidate_id"] == task["candidate"]
        )
    return result


def main():
    root = Path(sys.argv[1])
    task = read_json(root / "task.json")
    if task["runtime"] != runtime_fingerprint():
        raise ValueError("job requires its frozen runtime")
    cancel = root / "cancel.flag"
    os.environ["MAHJONG_JOB_CANCEL"] = str(cancel)

    def stop(signum, frame):
        cancel.touch()
        raise JobCancelled("operator or budget interrupted job")

    signal.signal(signal.SIGTERM, stop)

    def watch_owner():
        while True:
            time.sleep(0.2)
            if os.getppid() != task["owner_pid"] or time.time() >= task["deadline"]:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=watch_owner, daemon=True).start()
    started = time.monotonic()
    deadline = started + max(0, task["deadline"] - time.time())
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        check_cancelled()
        value = (
            prepare(task, root, deadline)
            if task["kind"] == "prepare"
            else play(task, root)
        )
    except JobCancelled:
        if time.time() >= task["deadline"]:
            value = {"status": "failed", "error": "iteration execution limit reached"}
            if task["kind"] == "prepare":
                value["candidates"] = {
                    cid: read_json(root / cid / "artifact.json")
                    for cid in task["packages"]
                    if (root / cid / "artifact.json").exists()
                }
        else:
            value = {"status": "paused", "reason": "operator interruption"}
    except Exception as exc:
        value = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:4000]}
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    value["usage"] = dict(
        wall_seconds=time.monotonic() - started,
        host_cpu_seconds=time.process_time(),
        child_cpu_seconds=after.ru_utime
        + after.ru_stime
        - before.ru_utime
        - before.ru_stime,
    )
    atomic_json(root / "result.json", value)


if __name__ == "__main__":
    main()
