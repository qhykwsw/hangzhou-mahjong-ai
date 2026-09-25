"""RL line (ADR-0016): train a recipe snapshot on a GPU host for a budget of hands, then evaluate it there.

The training harness is the recipe framework's own (guided.trial via
guided.recipe_jobs.trial_train): the attempt's recipe snapshot is uploaded,
trained from the fixed start Actor for the budget against the fixed opponent
pool, and the trained Actor is scored on the same host by harness.evaluate.
harness.line starts attempts and drives them (train() here is one resumable
step of that); the maintainer prepares each host:

  python -m harness.rl setup <remote checkpoint | weights sha> --name <start name> --host 4070|4090|5070ti
  python -m harness.rl export <checkpoint> --name <model name> --host <host>

What the recipe may not decide is fixed here and sealed with the pool: the
hand budget, each host's resource limits and collection execution (workers,
tables, batching), and which modules a recipe may import.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile

from . import evaluate as E

sys.path.insert(0, str(E.REPO / "projects/personal/selfplay"))
from guided import recipe_jobs, remote  # noqa: E402
from mahjong_ai.execution.remote import opponent_specs  # noqa: E402

# The budget is training hands (16 per game), so an attempt trains the same
# amount on each host; it shows the trend of a change, not a finished model.
# The seconds cap the training too: a recipe that spends more per hand trains
# fewer hands within them, which is its own cost.
HANDS = E.override("FP_RL_HANDS", 1_000_000)
BUDGET = E.override("FP_RL_BUDGET", 14400)
# A collection plays out every game it started, so the last one may pass the
# budget a little; past this allowance the attempt is invalid.
HAND_ALLOWANCE = 25_000
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
# Collection execution per host (the recipe's own workers/tables are ignored):
# 80% of the cores the host lends, 32 tables each, 2 ms to gather a batch.
EXECUTION = {
    "4070": {"workers": 16, "tables": 32, "gather_seconds": 0.002},
    "4090": {"workers": 16, "tables": 32, "gather_seconds": 0.002},
    "5070ti": {"workers": 12, "tables": 32, "gather_seconds": 0.002},
}
HOSTS = tuple(LIMITS)
# A recipe trains through ctx; it may not play games or run jobs by itself.
GUIDED_FORBIDDEN = (
    "guided.collection", "guided.environment", "guided.trial", "guided.recipe_jobs", "guided.remote",
    "guided.sync", "guided.run", "guided.bootstrap", "guided.model_server",
)


def check_recipe(package):
    """What in the recipe breaks the line's rules (empty when none)."""
    return E.check_source(package, GUIDED_FORBIDDEN)


def start_id(pool):
    """The RL line's first best version in a pool: the start Actor itself."""
    return f"fp-rl-start-{pool['id'][:8]}"


def baseline(host):
    """Score the start Actor on the whole development ladder as the RL line's first best version.

    Every attempt trains from this start, so it is what an attempt must beat,
    and the three hosts can fork from it at once. Its recipe snapshot is the
    template the first designs change. Once per pool (the maintainer).
    """
    pool = E.load_pool()
    E.check_seal()
    if E.best(pool["id"], "rl"):
        raise E.Refused("RL 线在当前对手池下已经有最好版本")
    setups = [E.read_json(setup_path(h)) for h in HOSTS if setup_path(h).exists()]
    if not setups or len({s["weights"] for s in setups}) != 1:
        raise E.Refused("各主机的 RL 起点不一致或还没有 setup")
    rel = E.SOURCES["rl"].relative_to(E.REPO).as_posix()
    import subprocess

    if subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=E.REPO,
                      capture_output=True, text=True).stdout.strip():
        raise E.Refused(f"{rel} 与 HEAD 不一致：起点的方案快照取已提交的基线方案")
    attempt_id = start_id(pool)
    snapshot(attempt_id)
    spec = {"family": "model", "version": attempt_id, "parameters": {"weights": setups[0]["weights"]}}
    record = E.evaluate("rl", spec, f"起点：{setups[0]['name']}（模仿，未经 RL 训练）", attempt_id, host=host,
                        extra={"why": "RL 线的起点：每次尝试都从它开始训练"})
    E.mark_reported([record])
    return record


def setup_path(host):
    return RL / f"setup-{host}.json"


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


def snapshot(attempt_id):
    """Freeze the working recipe for a new attempt; training only ever sees this copy."""
    job = RL / "attempts" / attempt_id
    if job.exists():
        shutil.rmtree(job)
    job.mkdir(parents=True)
    shutil.copytree(E.SOURCES["rl"], job / "recipe", ignore=shutil.ignore_patterns("__pycache__"))


def train(state):
    """Train the attempt's recipe snapshot on its host; resumable.

    Returns (outcome, spec, training): outcome "ok" with the placed Actor's
    spec, or "crash" / "invalid" with the reason in training["error"].
    Raises recipe_jobs.HostFault when the host lost the work (it restarted)
    and connection errors when the host cannot be reached; harness.line
    requeues or waits on those, since neither is the attempt's fault.
    """
    host, attempt_id = state["host"], state["id"]
    setup_ = E.read_json(setup_path(host))
    job = RL / "attempts" / attempt_id
    remote_job = remote.root(host) + "/" + job.relative_to(E.REPO).as_posix()
    if not (job / "launch-train.json").exists():
        # A fresh launch (the first, or after the work was lost): nothing left from an earlier run.
        remote.execute(host, f"import shutil\nshutil.rmtree({remote_job!r}, ignore_errors=True)", timeout=120)
    opponents, runtime = training_opponents(host)
    task = dict(host=host, remote_dir=remote_job, job_dir=str(job), code=setup_["code"],
                source_id=setup_["source_id"], recipe=str(job / "recipe"), bundle=remote.root(host) + BUNDLE,
                runtime=runtime, start_actor=setup_["start_actor"], budget_seconds=BUDGET, hand_budget=HANDS,
                seed=state["seed"], line="rl", limits=LIMITS[host], execution=EXECUTION[host],
                opponents=opponents, fetch_actor=False)
    E.log(f"{attempt_id}: training {HANDS} hands on {host}")
    result = recipe_jobs.trial_train(task, lambda message: None)
    E.atomic_json(job / "training.json", {k: v for k, v in result.items() if k != "metrics"})
    status = result["status"]
    training = {"host": host, "state": status.get("state"), "hands": status.get("hands"),
                "training_seconds": status.get("training_seconds")}
    if not result["ok"]:
        return "crash", None, training | {"error": str(status.get("error") or status)[-2000:]}
    if HANDS and (status.get("hands") or 0) > HANDS + HAND_ALLOWANCE:
        return "invalid", None, training | {
            "error": f"训练了 {status['hands']:,} 局，超过预算 {HANDS:,} 局加余量 {HAND_ALLOWANCE:,} 局"}
    spec = dict(place_on_host(result["actor"], host), version=attempt_id)
    return "ok", spec, training


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup")
    s.add_argument("source", help="checkpoint path on the host, or the weights sha of an Actor this machine holds")
    s.add_argument("--name", required=True)
    s.add_argument("--host", required=True, choices=HOSTS)
    x = sub.add_parser("export", help="export and freeze a model trained elsewhere (e.g. a pool member)")
    x.add_argument("checkpoint")
    x.add_argument("--name", required=True)
    x.add_argument("--host", required=True)
    b = sub.add_parser("baseline", help="score the start Actor as the RL line's first best version (once per pool)")
    b.add_argument("--host", default="4070", choices=HOSTS)
    args = parser.parse_args()
    if args.command == "setup":
        print(json.dumps(setup(args.source, args.name, args.host), ensure_ascii=False))
    elif args.command == "baseline":
        try:
            print(E.report(baseline(args.host)))
        except E.Refused as exc:
            raise SystemExit(str(exc))
    else:
        target = remote.root(args.host) + "/.local/firstprinciples/exports/" + args.name
        print(json.dumps(export(args.host, args.checkpoint, args.name, target), ensure_ascii=False))


if __name__ == "__main__":
    main()
