"""RL line (ADR-0016): train a recipe snapshot on a GPU host for a budget of hands, then evaluate it there.

The training harness is the recipe framework's own (guided.trial via
guided.recipe_jobs.trial_train): the attempt's recipe snapshot is uploaded,
trained from the fixed start Actor for the budget of hands, and the trained
Actor is scored on the same host by harness.evaluate against the fixed pool.
The recipe trains against the pool by default and may choose other training
opponents and its teacher from the catalog: the pool and every version that
became a line's best.
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
import os
import re
import shutil
import subprocess
import sys
import tempfile

from . import evaluate as E

sys.path.insert(0, str(E.REPO / "projects/personal/selfplay"))
from guided import recipe_jobs, remote  # noqa: E402
from mahjong_ai.execution.remote import opponent_specs  # noqa: E402

# The budget is training hands (16 per game), so an attempt trains the same
# amount on any host. The agent chooses it per attempt up to MAX_HANDS (start
# --hands): more hands per attempt mean fewer results a day, so the cap keeps
# the three slots exploring. A recipe that spends more per hand only takes
# longer. The deadline is not a budget: it only stops a training that hangs
# (the slowest full budget, a slow teacher labelling every step, takes about a day).
HANDS = E.override("FP_RL_HANDS", 1_000_000)
MAX_HANDS = 5_000_000
BUDGET = E.override("FP_RL_BUDGET", 2 * 86400)
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
# Collection execution per host (the recipe's own choices are ignored): 80% of
# the cores the host lends, 32 tables each, 2 ms to gather a batch, and the
# host's inference scheduling (the CPU-bound hosts keep the coarser batches).
EXECUTION = {
    "4070": {"workers": 16, "tables": 32, "gather_seconds": 0.002,
             "inference_history_band": 64, "fine_graph_batches": True},
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


def host_resources():
    """Each host's fixed limits and what its latest training used there, for designing per host."""
    latest = {}
    for r in E.attempts(line="rl"):  # in finishing order
        host = (r.get("training") or {}).get("host")
        if host in LIMITS and (RL / "attempts" / r["id"] / "training.json").exists():
            latest[host] = r
    gib = 2**30
    rows = ["主机资源（上限由维护者固定，方案运行时从 ctx.resources 读取；峰值取该主机最近一次训练）："]
    for host in HOSTS:
        limit, run = LIMITS[host], EXECUTION[host]
        record = latest.get(host)
        status = E.read_json(RL / "attempts" / record["id"] / "training.json")["status"] if record else {}
        caps = status.get("limits") or {}
        if caps:
            row = (f"{host}：显存上限 {caps['gpu_limit_bytes'] / gib:.1f} GiB（整卡的 {limit['gpu_memory_fraction']:.0%}）、"
                   f"内存上限 {caps['ram_limit_bytes'] / gib:.1f} GiB、CPU {limit['cpu_capacity']} 核")
        else:
            row = (f"{host}：显存上限为整卡的 {limit['gpu_memory_fraction']:.0%}、内存上限为本机的 {limit['ram_fraction']:.0%}、"
                   f"CPU {limit['cpu_capacity']} 核")
        row += f"，采集 {run['workers']} 进程 × {run['tables']} 桌"
        if not record:
            rows.append(f"  {row}｜还没有训练回执")
            continue
        used = (f"张量峰值 {status.get('gpu_peak_allocated_bytes', 0) / gib:.1f} GiB、"
                f"分配器保留峰值 {status.get('gpu_peak_reserved_bytes', 0) / gib:.1f} GiB、"
                f"内存峰值 {status.get('ram_peak_bytes', 0) / gib:.1f} GiB")
        if status.get("state") == "complete":
            used += f"，{status['hands'] / 1e4:.1f} 万局用时 {status['training_seconds'] / 60:.0f} 分"
        else:
            used += f"，训练{E.STATUS.get(record['status'], record['status'])}（{status.get('reason') or status.get('state')}）"
        rows.append(f"  {row}｜最近 {record['id']}：{used}")
    return "\n".join(rows)


def setup_path(host):
    return RL / f"setup-{host}.json"


# -- catalog: what a recipe may train against or learn from --------------------------
def catalog(pool=None):
    """Every version that became a line's best under the pool, with its spec and record.

    Rule versions run under the pool's runtime like the pool's own members and
    may also be the teacher; RL versions are Actors.
    """
    pool = pool or E.load_pool()
    offered = {}
    for r in E.distinct(E.attempts(pool["id"])):
        if r["line"] in ("rule", "rl") and r["status"] == "keep" and r.get("spec") and r["id"] not in pool["members"]:
            spec = dict(r["spec"])
            if spec.get("family") != "model":
                spec["runtime_path"] = pool["runtime_path"]
            offered[r["id"]] = {"spec": spec, "record": r}
    return offered


MEASURE = r"""
import json, sys, time
from pathlib import Path
from guided.artifacts import bind_runtime
from guided.config import configuration
from guided.environment import External, Game

task = json.loads(sys.stdin.read())
bind_runtime({"runtime": task["runtime"], "runtime_hash": Path(task["runtime"]).name})
out = {}
for cid, spec in task["specs"].items():
    game = Game(configuration(rounds=task["hands"], lookahead=False), None, {}, [cid] * 4, 17,
                external=External({cid: dict(spec, isolation="none")}))
    times = []
    for _, policy in game.borrowed:
        def timed(obs, act=policy.act):
            start = time.perf_counter()
            action = act(obs)
            times.append(time.perf_counter() - start)
            return action
        policy.act = timed
    game.advance()
    out[cid] = {"mean_ms": 1000 * sum(times) / len(times), "max_ms": 1000 * max(times), "decisions": len(times)}
print(json.dumps(out))
"""


def decision_costs(specs, runtime):
    """Milliseconds per decision of rule versions, measured here once per package and cached."""
    path = RL / "decision-ms.json"
    known = E.read_json(path) if path.exists() else {}
    missing = {cid: s for cid, s in specs.items() if s.get("family") != "model" and s["package_hash"] not in known}
    if missing:
        selfplay = E.REPO / "projects/personal/selfplay"
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
            "PYTHONPATH": os.pathsep.join([str(selfplay), str(selfplay / "src")])}
        # Eight hands with the version in every seat: a few hundred of its decisions.
        out = subprocess.run([sys.executable, "-c", MEASURE], input=json.dumps(
            {"runtime": runtime, "hands": 8, "specs": missing}), env=env, capture_output=True, text=True, check=True)
        measured = json.loads(out.stdout.strip().splitlines()[-1])
        known |= {missing[cid]["package_hash"]: value for cid, value in measured.items()}
        RL.mkdir(parents=True, exist_ok=True)
        E.atomic_json(path, known)
    return {cid: known.get(s.get("package_hash")) for cid, s in specs.items()}


def catalog_facts():
    """What the agent chooses training opponents and the teacher by, one entry per choice."""
    pool = E.load_pool()
    offered = catalog(pool)
    records = {r["id"]: r for r in E.distinct(E.attempts(pool["id"]))}
    teacher = "heuristic-efficiency--01--piao-call-channel"  # the teacher bundle's champion
    specs = dict(pool["members"]) | {cid: e["spec"] for cid, e in offered.items()}
    costs = decision_costs(specs, pool["runtime_path"])
    facts = []
    for cid, spec in specs.items():
        record = offered[cid]["record"] if cid in offered else records.get(f"ref-{cid}")
        rule = spec.get("family") != "model"
        facts.append({
            "id": cid,
            "source": {"rule": "规则线", "rl": "RL 线"}.get(record["line"]) if cid in offered else "固定对手池",
            "kind": "规则" if rule else "模型",
            "teacher": rule,
            "default": {"opponent": cid in pool["members"], "teacher": cid == teacher},
            "score": record and round(record["score"], 2),
            "se": record and round(record["se"], 2),
            "decision_ms": costs.get(cid),
            "description": (f"与 {record['equivalent']['to']} 决策完全一致的等价实现：" if record.get("equivalent") else "")
            + record["description"] if cid in offered else None,
        })
    return facts


def catalog_report():
    rows = ["可选训练对手与教师（方案用 cfg[\"opponents\"] 选对手、cfg[\"teacher\"] 选教师；只有规则可当教师。"
            "成绩是固定对手池上的开发集场均，耗时是 Mac 单进程实测的每个决策）："]
    for f in catalog_facts():
        score = f"{f['score']:+.2f}±{f['se']:.2f}" if f["score"] is not None else "未单独评估"
        cost = (f"每决策 {f['decision_ms']['mean_ms']:.2g} ms（最长 {f['decision_ms']['max_ms']:.0f} ms）"
                if f["decision_ms"] else "GPU 推理" if f["kind"] == "模型" else "未测")
        tags = "、".join(t for t, on in (("默认对手", f["default"]["opponent"]), ("默认教师", f["default"]["teacher"]),
                                         ("可当教师", f["teacher"] and not f["default"]["teacher"])) if on)
        rows.append(f"  {f['id']}｜{f['source']}·{f['kind']}｜{score}｜{cost}" + (f"｜{tags}" if tags else "")
                    + (f"｜{f['description'][:60]}" if f["description"] else ""))
    return "\n".join(rows)


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
    """Host-side specs of the fixed pool (the default training opponents), of the catalog, and the pool's runtime.

    opponent_specs mirrors the versions and the pool's runtime on the host and
    compiles its native extension; training games run under it instead of the
    teacher bundle's older copy, and rule versions run in the collection
    workers like in evaluation.
    """
    pool = E.load_pool()
    host = E.remote_host(host_name)
    offered = catalog(pool)
    specs = opponent_specs(host, {cid: {"spec": spec} for cid, spec in pool["members"].items()}
                           | {cid: {"spec": e["spec"]} for cid, e in offered.items()}, E.MODELS, pool["timeouts"])
    specs = {cid: (dict(spec, isolation=E.ISOLATION) if spec.get("family") != "model" else spec)
             for cid, spec in specs.items()}
    return ({cid: specs[cid] for cid in pool["members"]}, {cid: specs[cid] for cid in offered},
            host.remote_path(pool["runtime_path"]))


def start_actor(state, host_name, setup_):
    """Where the attempt's training starts on the host, and the training hands already behind it.

    By default the fixed start (setup); with ``from: best`` the Actor of the
    best version the attempt was designed on, so training adds up along the
    line (every step of it still had to beat the best version).
    """
    pool = E.load_pool()
    if state.get("from") != "best":
        return setup_["start_actor"], 0
    base = next(r for r in reversed(E.attempts(pool["id"], "rl")) if r["id"] == state["base"])
    weights = E.MODELS / base["spec"]["parameters"]["weights"]
    host = E.remote_host(host_name)
    host.sync([weights])
    training = base.get("training") or {}
    return host.remote_path(weights), training.get("lineage_hands") or training.get("hands") or 0


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
    opponents, offered, runtime = training_opponents(host)
    # What this attempt could choose from, as the agent saw it.
    E.atomic_json(job / "catalog.json", catalog_facts())
    hands = state.get("hands") or HANDS
    start, behind = start_actor(state, host, setup_)
    task = dict(host=host, remote_dir=remote_job, job_dir=str(job), code=setup_["code"],
                source_id=setup_["source_id"], recipe=str(job / "recipe"), bundle=remote.root(host) + BUNDLE,
                runtime=runtime, start_actor=start, budget_seconds=BUDGET, hand_budget=hands,
                seed=state["seed"], line="rl", limits=LIMITS[host], execution=EXECUTION[host],
                opponents=opponents, catalog=offered, fetch_actor=False)
    E.log(f"{attempt_id}: training {hands} hands on {host} from {start}")
    result = recipe_jobs.trial_train(task, lambda message: None)
    E.atomic_json(job / "training.json", {k: v for k, v in result.items() if k != "metrics"})
    status = result["status"]
    training = {"host": host, "state": status.get("state"), "hands": status.get("hands"),
                "training_seconds": status.get("training_seconds"), "budget": hands,
                "from": state["base"] if state.get("from") == "best" else None,
                "lineage_hands": behind + (status.get("hands") or 0)}
    if not result["ok"]:
        return "crash", None, training | {"error": str(status.get("error") or status)[-2000:]}
    if (status.get("hands") or 0) > hands + HAND_ALLOWANCE:
        return "invalid", None, training | {
            "error": f"训练了 {status['hands']:,} 局，超过预算 {hands:,} 局加余量 {HAND_ALLOWANCE:,} 局"}
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
