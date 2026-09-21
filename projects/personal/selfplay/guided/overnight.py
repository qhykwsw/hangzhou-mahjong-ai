"""Finite same-host startup A/B; no teacher updates or subsequent main training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bootstrap
from .config import configuration
from .run import atomic_json


def host_config(host):
    dim, layers, micro = (384, 8, 128) if host == "4090" else (256, 6, 64)
    return configuration(
        dim=dim,
        layers=layers,
        heads=8,
        minibatch=micro,
        effective_minibatch=512,
        seed=2026092101,
    )


def initialization(root, cfg, bundle):
    from .run import train
    import torch

    train(root / "initial", cfg, bundle, bundle, deadline=0, eval_enabled=False)
    path = root / "initial" / "initial.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    atomic_json(
        root / "initialization.json",
        {
            "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "actor_parameters": sum(v.numel() for v in state["actor"].values()),
            "source_id": bootstrap.verify_source(),
        },
    )
    return path


def preflight(root, host, bundle):
    import torch
    from .run import train

    root.mkdir(parents=True, exist_ok=False)
    cfg = host_config(host) | {"samples": 1024, "workers": 2, "tables": 4}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for mode in ("joint", "bc"):
        target = root / mode
        deadline = time.time() + 900
        train(
            target,
            cfg,
            bundle,
            bundle,
            deadline=deadline,
            bc_seconds=900 if mode == "bc" else 0,
            eval_enabled=False,
            max_updates=1,
        )
        state = torch.load(target / "latest.pt", map_location="cpu", weights_only=False)
        if not state["actor_optimizer"]["state"]:
            raise RuntimeError(f"{mode}: no optimizer update completed")
        for key in ("actor", "critic"):
            if not all(torch.isfinite(v).all() for v in state[key].values()):
                raise FloatingPointError(f"nonfinite {mode} {key}")
        if mode == "joint":
            if not state["critic_optimizer"]["state"]:
                raise RuntimeError("critic was not updated")
            # Reload the entire optimizer/RNG/cache state and execute another update.
            train(
                target,
                cfg,
                bundle,
                bundle,
                deadline=deadline,
                resume=True,
                eval_enabled=False,
                max_updates=1,
            )
            restored = torch.load(
                target / "latest.pt", map_location="cpu", weights_only=False
            )
            if restored["iteration"] <= state["iteration"]:
                raise RuntimeError("resume did not complete another update")
        elif state["critic_optimizer"]["state"]:
            raise RuntimeError("BC unexpectedly updated Critic")
    atomic_json(
        root / "receipt.json",
        {
            "state": "passed",
            "host": host,
            "seconds": time.time() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "source_id": bootstrap.verify_source(),
            "config": cfg,
            "checks": [
                "joint actor/critic update",
                "full-state resume update",
                "BC actor-only update",
            ],
        },
    )


def evaluate_checkpoint(root, route, bundle, cfg, plan):
    import torch
    from .evaluation import evaluate
    from .networks import Actor, Critic

    torch.set_num_threads(cfg["threads"])
    checkpoint = root / route / "latest.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"] != cfg:
        raise ValueError("evaluation configuration differs from training")
    actor = Actor(cfg["dim"], cfg["layers"], cfg["heads"]).to(cfg["device"])
    critic = Critic().to(cfg["device"])
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    result = evaluate(
        actor,
        critic,
        cfg | {"seed": plan["eval_seed"], "evaluation_games": plan["evaluation_games"]},
        bundle,
        bundle,
        0,
        time.time() + plan["evaluation_timeout_seconds"],
        namespace=plan["evaluation_namespace"],
    )
    result.update(
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_id=bootstrap.verify_source(),
        host=plan["host"],
        route=route,
    )
    atomic_json(root / "evaluation" / f"{route}.json", result)


def run_child(argv, log, timeout, on_start):
    with Path(log).open("ab") as stream:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
        on_start(process.pid)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"child exited {code}; see {log}")


def suite(root):
    plan = json.loads((root / "plan.json").read_text())
    cfg, bundle = plan["config"], plan["bundle"]
    if (root / "suite.json").exists():
        raise FileExistsError(
            "suite already started; no automatic restart or budget extension"
        )
    receipt = json.loads((root / "preflight" / "receipt.json").read_text())
    if (
        receipt["state"] != "passed"
        or receipt["source_id"] != bootstrap.verify_source()
    ):
        raise ValueError("missing preflight for this exact source")
    state = {
        "stage": "initializing",
        "pid": os.getpid(),
        "started": time.time(),
        "host": plan["host"],
        "tasks": {},
        "source_id": bootstrap.verify_source(),
    }

    def save():
        state["updated"] = time.time()
        atomic_json(root / "suite.json", state)

    def invoke(name, argv, timeout, **details):
        state["stage"] = name
        task = {"state": "running", "argv": argv, "started": time.time(), **details}
        state["tasks"][name] = task
        save()

        def launched(pid):
            task["pid"] = pid
            save()

        run_child(argv, root / f"{name}.log", timeout, launched)
        task.update(state="complete", finished=time.time(), exit_code=0)
        save()

    save()
    try:
        initial = initialization(root, cfg, bundle)
        atomic_json(root / "config.json", cfg)
        for route in ("A", "B"):
            deadline = time.time() + plan["training_seconds_per_route"]
            argv = [
                sys.executable,
                "-m",
                "guided.run",
                "--run",
                str(root / route),
                "--config",
                str(root / "config.json"),
                "--bundle",
                bundle,
                "--panel",
                bundle,
                "--initial",
                str(initial),
                "--deadline",
                str(deadline),
                "--bc-seconds",
                str(plan["bc_seconds"] if route == "B" else 0),
            ]
            invoke(
                "train-" + route,
                argv,
                plan["training_seconds_per_route"] + 300,
                deadline=deadline,
                initial=str(initial),
            )
        for route in ("A", "B"):
            argv = [
                sys.executable,
                "-m",
                "guided.overnight",
                "evaluate",
                "--root",
                str(root),
                "--route",
                route,
            ]
            invoke("evaluate-" + route, argv, plan["evaluation_timeout_seconds"] + 60)
        from .statistics import interval

        evaluations = {
            r: json.loads((root / "evaluation" / f"{r}.json").read_text())
            for r in ("A", "B")
        }
        a, b = [
            evaluations[r]["results"]["model:current"]["group_scores"]
            for r in ("A", "B")
        ]
        difference = interval([y - x for x, y in zip(a, b, strict=True)])
        atomic_json(
            root / "selection.json",
            {
                "B_minus_A": difference,
                "route": "B" if difference["lower"] > 0 else "A",
                "reason": "B clearly stronger"
                if difference["lower"] > 0
                else "A wins or simpler tie-break",
                "further_training_started": False,
            },
        )
        state["stage"] = "complete"
    except BaseException as exc:
        task = state["tasks"].get(state["stage"])
        if task and task["state"] == "running":
            task.update(state="failed", error=str(exc))
        state.update(stage="failed", error=str(exc))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "suite", "evaluate"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--route", choices=("A", "B"))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    if args.mode == "preflight":
        preflight(root / "preflight", plan["host"], plan["bundle"])
    elif args.mode == "suite":
        suite(root)
    else:
        evaluate_checkpoint(root, args.route, plan["bundle"], plan["config"], plan)


if __name__ == "__main__":
    main()
