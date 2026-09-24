"""RL line (ADR-0016): train rl/recipe for one hour on the 4090, then evaluate it.

The training harness is the recipe framework's own (guided.trial via
guided.recipe_jobs.trial_train): the recipe package is uploaded, trained from the
fixed start Actor for BUDGET seconds against the fixed opponent pool, and the
trained Actor is fetched back and scored by harness.evaluate like any attempt.

Usage (from projects/personal/firstprinciples):
  python -m harness.rl setup <remote checkpoint> --name <start name>   # once: export the start Actor
  python -m harness.rl attempt "<what changed>" --design-start <epoch>
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

from . import evaluate as E

sys.path.insert(0, str(E.REPO / "projects/personal/selfplay"))
from guided import recipe_jobs, remote  # noqa: E402
from mahjong_ai.execution.remote import RemoteHost, opponent_specs  # noqa: E402

HOST = "4090"
BUDGET = int(__import__("os").environ.get("FP_RL_BUDGET", "3600"))  # seconds of training per attempt
RL = E.LOCAL / "rl"
BUNDLE = remote.ROOT + "/.local/guided-teachers/b492a752d115c52c2c9607aa98ec6cd949846a0644d7741dda658f0d753b2199/manifest.json"
REMOTE = remote.ROOT + "/.local/firstprinciples/rl"
LIMITS = {"gpu_memory_fraction": 0.75, "ram_fraction": 0.75}


def setup(checkpoint, name):
    """Deploy the frozen training code and export the start Actor from a checkpoint on the host."""
    RL.mkdir(parents=True, exist_ok=True)
    manifest = recipe_jobs.freeze_source(RL / "source")
    code = remote.ROOT + "/.local/guided-code/" + manifest["id"]
    remote.upload(HOST, RL / "source", [*manifest["files"], "source-manifest.json"], code)
    remote.command(HOST, [remote.PYTHON, "-c", recipe_jobs.NATIVE_BUILD], cwd=code,
                   environment=recipe_jobs.env(code), timeout=600)
    start = f"{REMOTE}/start/actor"
    out = remote.command(HOST, [remote.PYTHON, "-c", recipe_jobs.START_EXPORT, checkpoint, start],
                         cwd=code, environment=recipe_jobs.env(code), timeout=300)
    meta = json.loads(out.strip().splitlines()[-1])
    local = recipe_jobs.fetch_directory(HOST, start, RL / "start" / "actor")
    spec = E.freeze_model(name, local)
    value = {"code": code, "source_id": manifest["id"], "start_actor": start, "start_meta": meta,
             "checkpoint": checkpoint, "name": name, "weights": spec["parameters"]["weights"]}
    E.atomic_json(RL / "setup.json", value)
    return value


def training_opponents():
    """Host-side specs of the fixed pool, the RL line's training opponents."""
    pool = E.load_pool()
    host = RemoteHost(HOST, E.research_state()["config"]["hosts"][HOST], E.REPO)
    return opponent_specs(host, {cid: {"spec": spec} for cid, spec in pool["members"].items()},
                          E.MODELS, pool["timeouts"])


def attempt(description, design_start=None):
    setup_ = E.read_json(RL / "setup.json")
    pool = E.load_pool()
    attempt_id = f"fp-rl-{len(E.attempts(pool['id'], 'rl')):03d}"
    design = time.time() - design_start if design_start else None
    job = RL / "attempts" / attempt_id
    if job.exists():
        shutil.rmtree(job)
    job.mkdir(parents=True)
    shutil.copytree(E.HERE / "rl" / "recipe", job / "recipe", ignore=shutil.ignore_patterns("__pycache__"))
    task = dict(host=HOST, remote_dir=f"{REMOTE}/{attempt_id}", job_dir=str(job), code=setup_["code"],
                source_id=setup_["source_id"], recipe=str(job / "recipe"), bundle=BUNDLE,
                start_actor=setup_["start_actor"], budget_seconds=BUDGET,
                seed=random.SystemRandom().getrandbits(31), line="rl", limits=LIMITS,
                opponents=training_opponents(), fetch_actor=True)
    E.log(f"{attempt_id}: training {BUDGET // 60} min on {HOST}")
    result = recipe_jobs.trial_train(task, lambda message: None)
    E.atomic_json(job / "training.json", {k: v for k, v in result.items() if k != "metrics"})
    if not result["ok"]:
        record = dict(id=attempt_id, line="rl", pool=pool["id"], status="crash", description=description,
                      started_at=time.time(), finished_at=time.time(), games=0, score=None, se=None,
                      best_before=None, diff=None, confirm=None, stats={}, spec=None, dev_blocks={},
                      design_seconds=design, error=str(result["status"])[:2000])
        with E.records_path().open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
    spec = E.freeze_model(attempt_id, result["local_actor"])
    return E.evaluate("rl", spec, description, attempt_id, design)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup")
    s.add_argument("checkpoint")
    s.add_argument("--name", required=True)
    a = sub.add_parser("attempt")
    a.add_argument("description")
    a.add_argument("--design-start", type=float)
    args = parser.parse_args()
    if args.command == "setup":
        print(json.dumps(setup(args.checkpoint, args.name), ensure_ascii=False))
    else:
        record = attempt(args.description, args.design_start)
        print(E.report(record) if record["status"] != "crash" else f"【RL 线 {record['id']}】训练失败：{record['error'][:300]}")


if __name__ == "__main__":
    main()
