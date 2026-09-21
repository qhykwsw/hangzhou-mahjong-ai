"""Calibrate portable evaluation concurrency on a frozen, unscored schedule."""

import argparse
import json
from pathlib import Path
import time

from . import bootstrap
from .benchmark_collection import signature
from .brawl import schedule, validate_records
from .run import atomic_json


def calibrate(plan, output):
    import torch
    from .collection import collect
    from .config import configuration
    from .deployment import load_actor
    from .networks import Critic
    from .resources import evaluation_resources

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    models = {
        name: load_actor(row["path"], "cuda")[0] for name, row in plan["models"].items()
    }
    actor = next(iter(models.values()))
    critic = Critic().to("cuda").eval()
    resources = evaluation_resources()
    cpus = resources["cpu_capacity"]
    base = resources["workers"]
    entrants = list(models) + plan["rules"]
    jobs = schedule(entrants, 32, plan["seed"] + 971)
    atomic_json(output / "schedule.json", jobs)
    # Calibrate available CPU capacity and batch supply independently. No host names.
    workers = sorted({base, min(32, cpus)})
    tables = sorted({min(resources["tables"], v) for v in [16, 32, 64]})
    candidates = [(w, t) for w in workers for t in tables]
    trials = []
    reference = None

    def trial(w, t, name):
        nonlocal reference
        cfg = configuration(workers=w, tables=t) | {
            "inference_batch_limit": resources["inference_batch_limit"]
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            _, records, stats = collect(
                actor,
                critic,
                cfg,
                plan["bundle"],
                plan["bundle"],
                {
                    "kind": "evaluate",
                    "iteration": 0,
                    "jobs": jobs,
                    "deadline": time.time() + 900,
                    "labels": False,
                },
                models,
            )
            validate_records(records, jobs, entrants)
            identity = signature(records)
            if reference is None:
                reference = identity
            seconds = time.monotonic() - started
            result = {
                "name": name,
                "workers": w,
                "tables": t,
                "seconds": seconds,
                "games_per_second": len(records) / seconds,
                "same_game_scores": identity == reference,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "resources": stats,
            }
            atomic_json(
                output / (name + ".json"), {"measurement": result, "records": records}
            )
        except torch.cuda.OutOfMemoryError as exc:
            result = {
                "name": name,
                "workers": w,
                "tables": t,
                "same_game_scores": False,
                "error": "cuda_out_of_memory",
            }
            torch.cuda.empty_cache()
        trials.append(result)
        atomic_json(output / "progress.json", {"detected": resources, "trials": trials})
        print(json.dumps(result), flush=True)
        return result

    for w, t in candidates:
        trial(w, t, f"{w}x{t}")
    eligible = [r for r in trials if r["same_game_scores"]]
    if not eligible:
        raise RuntimeError("no valid concurrency profile")
    best = min(eligible, key=lambda r: r["seconds"])
    confirm = trial(best["workers"], best["tables"], "confirm")
    if not confirm["same_game_scores"]:
        raise ValueError("winning profile failed game-score confirmation")
    selected = resources | {"workers": best["workers"], "tables": best["tables"]}
    atomic_json(
        output / "result.json",
        {
            "source_id": bootstrap.verify_source(),
            "execution": selected,
            "trials": trials,
            "all_selected_game_scores_identical": True,
        },
    )
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    plan = json.loads((root / "plan.json").read_text())
    if bootstrap.verify_source() != plan["source_id"]:
        raise ValueError("source mismatch")
    execution = calibrate(plan, root / "calibration")
    # Inputs are already frozen; this records execution-only settings measured
    # before any tournament game is counted.
    plan["execution"] = execution
    atomic_json(root / "plan.json", plan)


if __name__ == "__main__":
    main()
