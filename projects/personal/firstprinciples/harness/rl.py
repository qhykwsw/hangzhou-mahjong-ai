"""RL line (ADR-0016): train rl/recipe on a GPU host for a budget of hands, then evaluate it there.

The training harness is the recipe framework's own (guided.trial via
guided.recipe_jobs.trial_train): the recipe package is uploaded, trained from the
fixed start Actor for the budget against the fixed opponent pool, and the
trained Actor is scored on the same host by harness.evaluate like any attempt.

GPU hosts train and evaluate independently, so the line's session can keep
one attempt under way on each; the records and the best version are shared
(harness.evaluate reserves attempt ids and re-reads the best version).

Usage (from projects/personal/firstprinciples):
  python -m harness.rl setup <remote checkpoint | weights sha> --name <start name> --host 4070|4090|5070ti   # once per host
  python -m harness.rl attempt "<what changed>" --design-start <epoch> --host 4070|4090|5070ti
The loop around it (history, one change, restore, report) is handbooks/common.md.
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time

from . import evaluate as E

sys.path.insert(0, str(E.REPO / "projects/personal/selfplay"))
from guided import recipe_jobs, remote  # noqa: E402
from mahjong_ai.execution.remote import opponent_specs  # noqa: E402

# The budget is training hands (16 per game), so an attempt trains the same
# amount on each host; it shows the trend of a change, not a finished model.
# The seconds only cap a run that stalls. FP_RL_HANDS=0 falls back to the time budget.
HANDS = int(os.environ.get("FP_RL_HANDS", "400000")) or None
BUDGET = int(os.environ.get("FP_RL_BUDGET", "14400"))
RL = E.LOCAL / "rl"
BUNDLE = "/.local/guided-teachers/b492a752d115c52c2c9607aa98ec6cd949846a0644d7741dda658f0d753b2199/manifest.json"
REMOTE = "/.local/firstprinciples/rl"
# What each host keeps for itself: the 4070 is an office desktop (GPU memory,
# RAM and cores for it), the 4090 only needs room for its own desktop.
LIMITS = {
    "4070": {"gpu_memory_fraction": 0.6, "ram_fraction": 0.6, "cpu_capacity": 20},
    "4090": {"gpu_memory_fraction": 0.85, "ram_fraction": 0.7, "cpu_capacity": 20},
    "5070ti": {"gpu_memory_fraction": 0.65, "ram_fraction": 0.6, "cpu_capacity": 16},
}


def setup_path(host):
    legacy = RL / "setup.json"  # the 4070's setup from before the line had two hosts
    return legacy if host == "4070" and legacy.exists() and not (RL / f"setup-{host}.json").exists() else RL / f"setup-{host}.json"


def place_on_host(actor_dir, host_name):
    """Put an Actor exported on the host into the host's model store, as if synced.

    The training host evaluates its own copy. The controller also retains the
    small Actor so another host can replay this version for paired confirmation.
    """
    host = E.remote_host(host_name)
    sha = remote.execute(host_name, "import json\nprint(json.load(open(%r))['weights_sha256'])" % (actor_dir + "/meta.json"),
                         timeout=60).strip().splitlines()[-1]
    local = E.MODELS / sha
    script = f"""import json,pathlib,shutil
src,dst,marker=pathlib.Path({actor_dir!r}),pathlib.Path({host.remote_path(local)!r}),pathlib.Path({host._marker(local)!r})
if not dst.exists():
    tmp=dst.with_name(dst.name+'.tmp');shutil.rmtree(tmp,ignore_errors=True);shutil.copytree(src,tmp);tmp.rename(dst)
marker.parent.mkdir(parents=True,exist_ok=True);marker.touch()
print(json.dumps(True))
"""
    remote.execute(host_name, script, timeout=300)
    if not local.exists():
        E.MODELS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=E.MODELS, prefix=".incoming-actor-") as folder:
            fetched = recipe_jobs.fetch_directory(host_name, actor_dir, folder)
            if E.read_json(fetched / "meta.json")["weights_sha256"] != sha:
                raise ValueError("exported Actor changed during transfer")
            if not local.exists():
                fetched.rename(local)
    return {"family": "model", "version": sha[:12], "parameters": {"weights": sha}}


def deploy_code(host):
    """Upload the frozen training framework to ``host`` and compile its native extension."""
    RL.mkdir(parents=True, exist_ok=True)
    manifest = recipe_jobs.freeze_source(RL / "source")
    code = remote.root(host) + "/.local/guided-code/" + manifest["id"]
    remote.upload(host, RL / "source", [*manifest["files"], "source-manifest.json"], code)
    remote.command(host, [remote.python(host), "-c", recipe_jobs.NATIVE_BUILD], cwd=code,
                   environment=recipe_jobs.env(code), timeout=600)
    return code, manifest["id"]


def export(host, checkpoint, name, target):
    """Deploy the frozen training code on ``host`` and export an Actor from a checkpoint there.

    The Actor is also fetched here and frozen as ``name``: the rule line evaluates
    on this machine and needs every pool model.
    """
    code, source_id = deploy_code(host)
    out = remote.command(host, [remote.python(host), "-c", recipe_jobs.START_EXPORT, checkpoint, target],
                         cwd=code, environment=recipe_jobs.env(code), timeout=300)
    meta = json.loads(out.strip().splitlines()[-1])
    local = recipe_jobs.fetch_directory(host, target, RL / "exports" / name)
    spec = E.freeze_model(name, local)
    return {"code": code, "source_id": source_id, "meta": meta, "checkpoint": checkpoint,
            "name": name, "weights": spec["parameters"]["weights"]}


def setup(source, name, host):
    """Put the RL line's start Actor on ``host``: from a checkpoint there, or from the model store."""
    start = f"{remote.root(host)}{REMOTE}/start/actor"
    if re.fullmatch(r"[0-9a-f]{64}", source):
        # An Actor this machine already holds (e.g. the other host's start): the host
        # mirrors the model store, so it is copied there instead of exported again.
        code, source_id = deploy_code(host)
        remote_host = E.remote_host(host)
        remote_host.sync([E.MODELS / source])
        remote.execute(host, f"""import pathlib,shutil,json
src,dst=pathlib.Path({remote_host.remote_path(E.MODELS / source)!r}),pathlib.Path({start!r})
if dst.exists(): shutil.rmtree(dst)
dst.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(src,dst);print(json.dumps(True))
""", timeout=300)
        meta = E.read_json(E.MODELS / source / "meta.json")
        value = {"code": code, "source_id": source_id, "meta": meta, "checkpoint": None, "name": name, "weights": source}
    else:
        value = export(host, source, name, start)
        place_on_host(start, host)  # the RL line trains and evaluates here
    value |= {"start_actor": start, "host": host}
    E.atomic_json(setup_path(host), value)
    return value


def training_opponents(host_name):
    """Host-side specs of the fixed pool (the RL line's training opponents) and its runtime.

    opponent_specs mirrors the pool's runtime on the host and compiles its
    native extension; training games run under it instead of the teacher
    bundle's older copy, and the pool's packages run in the collection workers
    like in evaluation.
    """
    pool = E.load_pool()
    host = E.remote_host(host_name)
    specs = opponent_specs(host, {cid: {"spec": spec} for cid, spec in pool["members"].items()},
                           E.MODELS, pool["timeouts"])
    specs = {cid: (dict(spec, isolation=E.ISOLATION) if spec.get("family") != "model" else spec)
             for cid, spec in specs.items()}
    return specs, host.remote_path(pool["runtime_path"])


def attempt(description, design_start=None, host="4070"):
    setup_ = E.read_json(setup_path(host))
    pool = E.load_pool()
    E.check_harness(pool)  # before an hour of training, not after
    attempt_id = E.reserve_attempt("rl", host, description)
    design = time.time() - design_start if design_start else None
    job = RL / "attempts" / attempt_id
    if job.exists():
        shutil.rmtree(job)
    job.mkdir(parents=True)
    shutil.copytree(E.HERE / "rl" / "recipe", job / "recipe", ignore=shutil.ignore_patterns("__pycache__"))
    opponents, runtime = training_opponents(host)
    remote_job = remote.root(host) + "/" + job.relative_to(E.REPO).as_posix()
    task = dict(host=host, remote_dir=remote_job, job_dir=str(job), code=setup_["code"],
                source_id=setup_["source_id"], recipe=str(job / "recipe"), bundle=remote.root(host) + BUNDLE, runtime=runtime,
                start_actor=setup_["start_actor"], budget_seconds=BUDGET, hand_budget=HANDS,
                seed=random.SystemRandom().getrandbits(31), line="rl", limits=LIMITS[host],
                opponents=opponents, fetch_actor=False)
    E.log(f"{attempt_id}: training {f'{HANDS} hands' if HANDS else f'{BUDGET // 60} min'} on {host}")
    try:
        result = recipe_jobs.trial_train(task, lambda message: None)
    except BaseException:
        E.release_attempt(attempt_id)
        raise
    E.atomic_json(job / "training.json", {k: v for k, v in result.items() if k != "metrics"})
    if not result["ok"]:
        record = dict(id=attempt_id, line="rl", pool=pool["id"], status="crash", description=description, host=host,
                      started_at=time.time(), finished_at=time.time(), games=0, score=None, se=None,
                      best_before=None, diff=None, confirm=None, stats={}, spec=None, dev_blocks={},
                      design_seconds=design, error=str(result["status"])[:2000])
        with E.records_path().open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        E.release_attempt(attempt_id)
        return record
    spec = dict(place_on_host(result["actor"], host), version=attempt_id)
    return E.evaluate("rl", spec, description, attempt_id, design, host=host)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup")
    s.add_argument("source", help="checkpoint path on the host, or the weights sha of an Actor this machine holds")
    s.add_argument("--name", required=True)
    s.add_argument("--host", default="4070", choices=sorted(LIMITS))
    x = sub.add_parser("export", help="export and freeze a model trained elsewhere (e.g. a pool member)")
    x.add_argument("checkpoint")
    x.add_argument("--name", required=True)
    x.add_argument("--host", required=True)
    a = sub.add_parser("attempt")
    a.add_argument("description")
    a.add_argument("--design-start", type=float, required=True)
    a.add_argument("--host", default="4070", choices=sorted(LIMITS))
    args = parser.parse_args()
    if args.command == "setup":
        print(json.dumps(setup(args.source, args.name, args.host), ensure_ascii=False))
    elif args.command == "export":
        target = remote.root(args.host) + "/.local/firstprinciples/exports/" + args.name
        print(json.dumps(export(args.host, args.checkpoint, args.name, target), ensure_ascii=False))
    else:
        print(E.report(attempt(args.description, args.design_start, args.host)))


if __name__ == "__main__":
    main()
