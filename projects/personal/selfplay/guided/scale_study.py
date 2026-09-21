"""Longer, finite scale comparisons with learning curves and paired game scores.

This is a supervised capacity study, not the formal PPO training schedule.
Each host runs all scales with its own seed. Only within-host model comparisons
are used; hardware and initialization are never conflated with model size.
"""

import argparse
import fcntl
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

from . import bootstrap
from .config import configuration
from .run import atomic_json
from .scale_probe import PROBE_SCALES, validation

MICROBATCHES = {
    "4090": {"small": 256, "medium": 256, "large": 128, "xlarge": 64},
    "4070": {"small": 128, "medium": 64, "large": 64, "xlarge": 32},
}


def milestones(seconds):
    if seconds <= 0:
        raise ValueError("positive update budget required")
    return sorted({t for t in (90, 300, 900, 1800, seconds) if t <= seconds})


def train(args):
    import numpy as np
    import torch
    from .artifacts import read_bundle
    from .deployment import benchmark, export_actor
    from .evaluation import evaluate
    from .learning import update
    from .networks import Actor, Critic

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.data, "rb") as stream:
        data = pickle.load(stream)  # Explicitly transferred trusted local artifact.
    bundle = read_bundle(args.bundle)
    if (
        data["bundle_id"] != bundle["id"]
        or data["runtime_hash"] != bundle["runtime_hash"]
    ):
        raise ValueError("dataset and teacher runtime differ")
    dim, layers, heads = PROBE_SCALES[args.scale]
    cfg = configuration(
        dim=dim,
        layers=layers,
        heads=heads,
        seed=args.seed,
        minibatch=MICROBATCHES[args.host][args.scale],
        effective_minibatch=512,
        epochs=1,
        evaluation_games=args.games,
    )
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    actor, critic = Actor(dim, layers, heads).cuda(), Critic().cuda()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=cfg["critic_learning_rate"], eps=1e-5
    )
    rng = np.random.default_rng(cfg["seed"])
    # A fixed train subset reveals a widening train/validation generalization gap.
    indices = np.random.default_rng(1941).permutation(len(data["train"]))[:4096]
    train_probe = [data["train"][int(i)] for i in indices]
    report = {
        "scale": args.scale,
        "host": args.host,
        "seed": args.seed,
        "config": cfg,
        "code_id": bootstrap.verify_source(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": "float32",
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "bundle_id": bundle["id"],
        "parameters": sum(p.numel() for p in actor.parameters()),
        "train_samples": len(data["train"]),
        "validation_samples": len(data["validation"]),
        "requested_update_seconds": args.seconds,
        "started": time.time(),
        "formal_training_started": False,
        "curve": [],
        "stage": "initial_validation",
    }

    def persist(stage):
        report["stage"] = stage
        report["updated"] = time.time()
        atomic_json(root / "result.json", report)

    def checkpoint(name, model):
        target = root / name
        temporary = target.with_suffix(".tmp")
        torch.save(
            {
                "format": "guided-v1",
                "config": cfg,
                "actor": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "code_id": report["code_id"],
                "data_sha256": report["data_sha256"],
                "update_seconds": elapsed,
                "optimized_samples": accepted,
            },
            temporary,
        )
        temporary.replace(target)
        return target

    elapsed = accepted = steps = 0
    persist("initial_validation")
    initial = validation(actor, data["validation"])
    report["curve"].append(
        {
            "target_update_seconds": 0,
            "update_seconds": 0,
            "validation": initial,
            "optimized_samples": 0,
        }
    )
    persist("supervised_updates")
    print(
        json.dumps({"stage": "started", "scale": args.scale, "initial": initial}),
        flush=True,
    )
    for target in milestones(args.seconds):
        while elapsed < target:
            segment_start = time.perf_counter()
            deadline = time.time() + (target - elapsed)
            metrics = update(
                actor,
                critic,
                actor_opt,
                critic_opt,
                data["train"],
                cfg | {"bc_only": True, "deadline": deadline},
                1.0,
                [],
                rng,
            )
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - segment_start
            accepted += metrics["optimized_samples"]
            steps += metrics["sgd_steps"]
            report.update(
                update_seconds=elapsed, optimized_samples=accepted, steps=steps
            )
            persist("supervised_updates")
        point = {
            "target_update_seconds": target,
            "update_seconds": elapsed,
            "optimized_samples": accepted,
            "steps": steps,
            "validation": validation(actor, data["validation"]),
            "train_subset": validation(actor, train_probe),
            "wall_seconds": time.time() - report["started"],
        }
        point["checkpoint"] = str(checkpoint(f"actor-{target:g}s.pt", actor))
        report["curve"].append(point)
        persist("supervised_updates")
        print(
            json.dumps({"stage": "milestone", "scale": args.scale, **point}), flush=True
        )
    # Final time point is predeclared; do not cherry-pick by validation or game score.
    final = Path(report["curve"][-1]["checkpoint"])
    observations = data["observations"]
    del actor_opt, critic_opt, train_probe, data
    gc.collect()
    torch.cuda.empty_cache()
    persist("paired_games")
    evaluation = evaluate(
        actor,
        critic,
        cfg | {"seed": args.eval_seed},
        args.bundle,
        args.bundle,
        0,
        time.time() + args.evaluation_timeout,
        namespace="extended-scale-common-final-v1",
    )
    atomic_json(root / "evaluation.json", evaluation)
    report["evaluation"] = {
        "path": str(root / "evaluation.json"),
        "seed": args.eval_seed,
        "games_per_target": args.games,
        "results": evaluation["results"],
        "teacher_minus_student": evaluation["teacher_minus_student"],
    }
    persist("http_benchmark")
    del actor, critic
    gc.collect()
    torch.cuda.empty_cache()
    artifact = export_actor(final, root / "actor")
    report["deployment"] = benchmark(artifact, observations, count=600, concurrency=10)
    if not report["deployment"]["passed_legality"]:
        raise RuntimeError("deployment legality/HTTP check failed")
    report["finished"] = time.time()
    persist("complete")
    print(
        json.dumps(
            {
                "stage": "complete",
                "scale": args.scale,
                "final_validation": report["curve"][-1]["validation"],
                "games": evaluation["results"]["model:current"],
            }
        ),
        flush=True,
    )


def suite(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "suite.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root / "suite.json"
    if path.exists():
        raise FileExistsError("suite already launched; preserve partial results")
    order = list(PROBE_SCALES)
    if args.host == "4070":
        order.reverse()
    state = {
        "stage": "running",
        "host": args.host,
        "seed": args.seed,
        "seconds_per_scale": args.seconds,
        "order": order,
        "tasks": {},
        "formal_training_started": False,
        "started": time.time(),
    }
    atomic_json(path, state)

    def stop_requested(_signum, _frame):
        raise InterruptedError("suite stop requested")

    signal.signal(signal.SIGTERM, stop_requested)
    for scale in order:
        argv = [
            sys.executable,
            "-m",
            "guided.scale_study",
            "train",
            "--output",
            str(root / scale),
            "--data",
            args.data,
            "--bundle",
            args.bundle,
            "--host",
            args.host,
            "--scale",
            scale,
            "--seconds",
            str(args.seconds),
            "--seed",
            str(args.seed),
            "--games",
            str(args.games),
            "--eval-seed",
            str(args.eval_seed),
            "--evaluation-timeout",
            str(args.evaluation_timeout),
        ]
        with (root / f"{scale}.log").open("ab") as log:
            child = subprocess.Popen(
                argv, stdout=log, stderr=log, start_new_session=True
            )
            state["tasks"][scale] = {
                "state": "running",
                "pid": child.pid,
                "started": time.time(),
                "argv": argv,
            }
            atomic_json(path, state)
            try:
                code = child.wait(
                    timeout=args.seconds * 1.5 + args.evaluation_timeout + 1200
                )
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = 124
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                state["tasks"][scale].update(state="interrupted", finished=time.time())
                state["stage"] = "interrupted"
                atomic_json(path, state)
                raise
        state["tasks"][scale].update(
            state="complete" if code == 0 else "failed",
            exit_code=code,
            finished=time.time(),
        )
        atomic_json(path, state)
    state.update(
        stage="complete"
        if all(t["exit_code"] == 0 for t in state["tasks"].values())
        else "finished_with_failures",
        finished=time.time(),
    )
    atomic_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "suite"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--host", required=True, choices=MICROBATCHES)
    parser.add_argument("--scale", choices=PROBE_SCALES)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=2026092301)
    parser.add_argument("--evaluation-timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.host == "4070":
        parser.error("4070 scale training is disabled by user instruction; use 4090")
    milestones(args.seconds)
    if args.games < 8 or args.games % 4 or args.evaluation_timeout <= 0:
        parser.error(
            "positive timeout and at least 8 games in complete four-seat groups required"
        )
    if args.command == "train" and not args.scale:
        parser.error("train requires --scale")
    bootstrap.verify_source()
    (train if args.command == "train" else suite)(args)


if __name__ == "__main__":
    main()
