"""Finite fixed-schedule collector benchmark; never writes tournament results."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

from . import bootstrap
from .brawl import validate_records
from .config import configuration
from .run import atomic_json


def signature(records):
    rows = sorted(
        (r["group"], r["seed"], tuple(r["seats"]), tuple(r["scores"])) for r in records
    )
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def main():
    import torch
    from .collection import collect
    from .deployment import load_actor
    from .networks import Critic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    old_code = Path(plan["code"])
    manifest = json.loads((old_code / "source-manifest.json").read_text())
    file = "projects/personal/selfplay/guided/collection.py"
    assert (
        hashlib.sha256((old_code / file).read_bytes()).hexdigest()
        == manifest["files"][file]
    )
    spec = importlib.util.spec_from_file_location(
        "guided._reference_collection", old_code / file
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.set_num_threads(2)
    models = {
        name: load_actor(item["path"], "cuda")[0]
        for name, item in plan["models"].items()
    }
    actor = next(iter(models.values()))
    critic = Critic().to("cuda").eval()
    entrants = list(models) + plan["rules"]
    jobs = json.loads(
        (Path(plan["remote_root"]) / "tournament/schedule.json").read_text()
    )[:256]
    reference = json.loads(
        (Path(plan["remote_root"]) / "tournament/chunk-0000.json").read_text()
    )["records"]
    expected = signature(reference)
    results = []
    variants = [
        ("baseline", baseline.collect, 8, 4),
        ("lean-8x4", collect, 8, 4),
        ("lean-8x16", collect, 8, 16),
        ("lean-8x32", collect, 8, 32),
        ("lean-16x16", collect, 16, 16),
    ]

    def trial(name, fn, workers, tables):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = configuration(workers=workers, tables=tables)
        started = time.monotonic()
        _, records, resources = fn(
            actor,
            critic,
            cfg,
            plan["bundle"],
            plan["bundle"],
            {
                "kind": "evaluate",
                "iteration": 0,
                "jobs": jobs,
                "deadline": time.time() + 600,
                "labels": False,
            },
            models,
        )
        seconds = time.monotonic() - started
        validate_records(records, jobs, entrants)
        identity = signature(records)
        result = {
            "name": name,
            "workers": workers,
            "tables": tables,
            "seconds": seconds,
            "games_per_second": len(records) / seconds,
            "same_game_scores": identity == expected,
            "receipt_sha256": identity,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "resources": resources,
        }
        atomic_json(
            output / (name + ".json"), {"measurement": result, "records": records}
        )
        results.append(result)
        atomic_json(output / "progress.json", {"trials": results})
        print(json.dumps(result), flush=True)
        if identity != expected:
            raise AssertionError("game scores differ from the frozen tournament")
        return result

    for variant in variants:
        trial(*variant)
    winner = min(results[1:], key=lambda r: r["seconds"])
    trial("winner-repeat", collect, winner["workers"], winner["tables"])
    trial("baseline-repeat", baseline.collect, 8, 4)
    base = statistics.mean(
        r["seconds"] for r in results if r["name"].startswith("baseline")
    )
    fast = statistics.mean(
        r["seconds"] for r in results if r["name"] in (winner["name"], "winner-repeat")
    )
    result = {
        "source_id": bootstrap.verify_source(),
        "baseline_source_id": manifest["id"],
        "baseline_seconds_mean": base,
        "winner_seconds_mean": fast,
        "speedup": base / fast,
        "winner": winner["name"],
        "workers": winner["workers"],
        "tables": winner["tables"],
        "all_game_scores_identical": all(r["same_game_scores"] for r in results),
        "trials": results,
    }
    atomic_json(output / "result.json", result)
    if base / fast < 1.2:
        raise AssertionError("fixed-schedule throughput gain below 20%")


if __name__ == "__main__":
    main()
