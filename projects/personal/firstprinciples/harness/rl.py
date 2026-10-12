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

What the recipe may not decide is fixed here and sealed with the pool: each
host's model size, the hand budgets per size, each host's resource limits and
collection execution (workers, tables, batching), and which modules a recipe
may import. The catalog also offers the platform proxies the maintainer
registered (harness.rl proxies), as training opponents only, and the rule
packages registered as training opponents (harness.rl training-rules), which
run under the pool's runtime but are never evaluated against, and RL versions
that never became a line's best but the maintainer registered (harness.rl
unconfirmed), e.g. to average their weights.
"""
import argparse
import json
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from . import evaluate as E

sys.path.insert(0, str(E.REPO / "projects/personal/selfplay"))
from guided import recipe_jobs, remote  # noqa: E402
from mahjong_ai.execution.remote import SERVICE_PORT, opponent_specs  # noqa: E402

# The budget is training hands (16 per game), so an attempt trains the same
# amount on any host. The agent chooses it per attempt up to MAX_HANDS (start
# --hands): more hands per attempt mean fewer results a day, so the cap keeps
# the three slots exploring. A recipe that spends more per hand only takes
# longer. The deadline is not a budget: it only stops a training that hangs
# (the slowest full budget, a large Actor distilling 5M teacher-labelled hands
# and training 5M more, takes about a day on the 4090).
HANDS = E.override("FP_RL_HANDS", 1_000_000)
BUDGET = E.override("FP_RL_BUDGET", 2 * 86400)
# Model size per host (maintainer's choice): the host's start Actor has it, and
# an attempt there trains only from versions of the same size. The large size
# trains in segments: up to 10M hands an attempt (5M of them teacher-labelled),
# each one evaluated, the next continuing from it (--from); the hands add up
# along the path without a cap, as for the small size. Since 2026-10-04 the 4090
# runs the medium size (about 4.8M parameters, between small 0.84M and large
# 14.3M) from a randomly initialized start, budgeted like the small size; the
# large versions stay comparable on the development set but no host trains them.
# Since 2026-10-05 (user: one RL route, on the 4090) the 4090 runs the small size,
# continuing the best lineage; the medium versions stay comparable, untrained.
SIZES = {"small": {"dim": 128, "layers": 4, "heads": 4}, "medium": {"dim": 256, "layers": 6, "heads": 8},
         "large": {"dim": 384, "layers": 8, "heads": 8}}
HOST_SIZE = {"4070": "small", "4090": "small", "5070ti": "small"}
BUDGETS = {"small": {"max_hands": 5_000_000, "max_labeled": None},
           "medium": {"max_hands": 5_000_000, "max_labeled": None},
           "large": {"max_hands": 10_000_000, "max_labeled": 5_000_000}}
MAX_HANDS = BUDGETS["small"]["max_hands"]
# A collection plays out every game it started, so the last one may pass the
# budget a little; past this allowance the attempt is invalid.
HAND_ALLOWANCE = 25_000
RL = E.LOCAL / "rl"
BUNDLE = "/.local/guided-teachers/b492a752d115c52c2c9607aa98ec6cd949846a0644d7741dda658f0d753b2199/manifest.json"
REMOTE = "/.local/firstprinciples/rl"
# What each host keeps for itself: the 4070 is a desktop in daily use (GPU memory,
# RAM and cores for it), the 4090 only needs room for its own desktop.
LIMITS = {
    "4070": {"gpu_memory_fraction": 0.6, "ram_fraction": 0.6, "cpu_capacity": 20},
    "4090": {"gpu_memory_fraction": 0.85, "ram_fraction": 0.7, "cpu_capacity": 20},
    "5070ti": {"gpu_memory_fraction": 0.65, "ram_fraction": 0.6, "cpu_capacity": 16},
}
# Collection execution per host (the recipe's own choices are ignored): 80% of
# the cores the host lends, 32 tables each, 2 ms to gather a batch, and the
# host's inference scheduling (the CPU-bound hosts keep the coarser batches).
# Openings run on `opening_workers` where set: their short calls pay every
# worker's start-up, and on the 4070 (10-06 benchmark, the 196 recipe's PG
# collection) ordinary games ran 25% faster on 24 workers than on 16 while
# openings ran 15% slower; on the 4090 (all 18 cores WSL sees) 18 workers ran
# ordinary games 4% faster than 16 and openings were fastest on 8 to 12; on the
# 5070 Ti (also 18 in WSL, which keeps the desktop's share) 18 workers ran them 18%
# faster than 12 and openings were fastest on 8.
EXECUTION = {
    "4070": {"workers": 24, "opening_workers": 16, "tables": 32, "gather_seconds": 0.002,
             "inference_history_band": 64, "fine_graph_batches": True},
    "4090": {"workers": 18, "opening_workers": 12, "tables": 32, "gather_seconds": 0.002},
    "5070ti": {"workers": 18, "opening_workers": 8, "tables": 32, "gather_seconds": 0.002},
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
    # The line's first best is the small start; large hosts start from their own.
    setups = [E.read_json(setup_path(h)) for h in HOSTS if setup_path(h).exists() and size(h) == "small"]
    if not setups or len({s["weights"] for s in setups}) != 1:
        raise E.Refused("小规格主机的 RL 起点不一致或还没有 setup")
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
        if run.get("opening_workers"):
            row += f"（开局 {run['opening_workers']} 进程）"
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


def size(host):
    return HOST_SIZE[host]


def size_of(weights):
    """The size name of an Actor in the model store, from its architecture."""
    config = E.read_json(E.MODELS / weights / "meta.json")["config"]
    for name, spec in SIZES.items():
        if all(config.get(k) == v for k, v in spec.items()):
            return name
    raise E.Refused(f"{weights[:12]} 的结构 {config} 不是固定的模型规格")


def check_origin(start_from, host, champion, pool):
    """Refuse training on ``host`` from a version of another model size."""
    if start_from == "start" or (start_from == "best" and champion["id"] == start_id(pool)):
        return
    record = champion if start_from == "best" else version(start_from)
    got = size_of(record["spec"]["parameters"]["weights"])
    if got != size(host):
        raise E.Refused(f"{record['id']} 是 {got} 规格（{SIZES[got]}），{host} 固定跑 {size(host)} 规格（{SIZES[size(host)]}）："
                        f"续训只能在同规格版本之间；在 {host} 上用 --from start（该主机的起点）或同规格的版本编号。")


def max_hands(host):
    return BUDGETS[size(host)]["max_hands"]


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
    return offered | unconfirmed_versions() | training_rules(pool) | proxies()


def unconfirmed_path():
    return RL / "unconfirmed-versions.json"


def unconfirmed_versions():
    """RL versions the maintainer registered although never a line's best (e.g. for weight averaging).

    Like the line's best versions they may be opponents or teachers; the
    evaluation stays on the fixed pool.
    """
    if not unconfirmed_path().exists():
        return {}
    out = {}
    for attempt_id in E.read_json(unconfirmed_path())["versions"]:
        record = version(attempt_id)
        out[attempt_id] = {"spec": dict(record["spec"]), "record": dict(
            record, description=f"未成为最好版本（{E.STATUS.get(record['status'], record['status'])}），维护者登记：{record['description']}")}
    return out


def register_unconfirmed(attempt_ids):
    """Offer these RL versions in the catalog; replaces the previous registration (none clears it)."""
    for attempt_id in attempt_ids:
        record = version(attempt_id)
        if not (E.MODELS / record["spec"]["parameters"]["weights"]).is_dir():
            raise SystemExit(f"{attempt_id}: 模型不在模型库")
    E.atomic_json(unconfirmed_path(), {"versions": list(attempt_ids)})
    return len(attempt_ids)


def training_rules_path():
    return RL / "training-rules.json"


def training_rules(pool=None):
    """Rule packages the maintainer registered as extra training opponents (harness.rl training-rules).

    They run under the pool's runtime like its members but are not in the
    pool: evaluation stays on the fixed pool. As rules they may also teach.
    """
    if not training_rules_path().exists():
        return {}
    pool = pool or E.load_pool()
    out = {}
    for name, m in E.read_json(training_rules_path())["members"].items():
        spec = {"family": "code", "version": name, "parameters": {}, "model_path": None, "program": None,
                "package_path": m["package_path"], "package_hash": m["package_hash"],
                "python_path": str(E.REPO / ".venv/bin/python"), "runtime_path": pool["runtime_path"]}
        record = {"id": name, "line": "training", "score": m.get("score"), "se": m.get("se"),
                  "description": m["description"]}
        out[name] = {"spec": spec, "record": record}
    return out


def register_training_rules(packages, scores=None):
    """Freeze rule packages as the training opponents offered in the catalog; replaces the previous registration."""
    members = {}
    for package in map(Path, packages):
        if not (package / "strategy.py").is_file():
            raise SystemExit(f"{package} 不是规则代码包")
        problems = E.check_source(package)
        if problems:
            raise SystemExit(f"{package.name}：" + "；".join(problems))
        target = E.LOCAL / "training-rules" / package.name / f"package-{E.tree_hash(package)[:16]}"
        if not target.exists():
            shutil.copytree(package, target, ignore=shutil.ignore_patterns("__pycache__"))
        doc = (package / "strategy.py").read_text().split('"""')[1].strip().splitlines()
        members[package.name] = {"package_path": str(target), "package_hash": E.tree_hash(target),
                                 "description": doc[-1].strip()} | ((scores or {}).get(package.name) or {})
    E.atomic_json(training_rules_path(), {"at": time.time(), "members": members})
    return len(members)


def proxies_path():
    return RL / "proxies.json"


def proxies():
    """The platform proxies the maintainer registered: training opponents only, never teachers."""
    if not proxies_path().exists():
        return {}
    out = {}
    for name, m in E.read_json(proxies_path())["members"].items():
        spec = {"family": "model", "version": f"proxy-{name}", "parameters": {"weights": m["weights"]}}
        record = {"id": f"proxy-{name}", "line": "proxy", "score": None, "se": None,
                  "description": f"平台代理：模仿 {m['display']}（线上每手 {m['online_per_hand']:+.2f}，"
                                 f"近 3 天同桌占比 {m['share']:.1%}，代理对其动作的一致率 {m['agreement']:.0%}）"}
        out[f"proxy-{name}"] = {"spec": spec, "record": record}
    return out


def register_proxies(panel_path):
    """Offer a proxy panel's members (argmax Actors) in the catalog; replaces the previous registration."""
    panel = E.read_json(Path(panel_path))
    fidelity = Path(panel_path).parent / "results"
    report = E.read_json(Path(panel["proxy_run"]) / "report.json")
    players = {p["user_id"]: p for p in E.read_json(E.REPO / "projects/personal/selfplay/.local/platform-data/players.json")}
    members = {}
    for name, m in panel["members"].items():
        online = [players[u] for u in m["users"] if u in players]
        hands = sum(p["hands"] for p in online)
        if not (E.MODELS / m["weights"]).is_dir():
            raise SystemExit(f"{name}: {m['weights'][:12]} 不在模型库")
        members[name] = {"weights": m["weights"], "display": m["display"], "share": m["share"],
                         "online_per_hand": sum(p["score_per_hand"] * p["hands"] for p in online) / max(1, hands),
                         "agreement": report["proxies"][name]["finetuned"]["agreement"]}
    E.atomic_json(proxies_path(), {"panel": panel["id"], "members": members})
    return len(members)


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
    specs = E.playing(pool) | {cid: e["spec"] for cid, e in offered.items()}
    costs = decision_costs(specs, pool["runtime_path"])
    facts = []
    for cid, spec in specs.items():
        record = offered[cid]["record"] if cid in offered else records.get(f"ref-{cid}")
        rule = spec.get("family") != "model"
        facts.append({
            "id": cid,
            "source": {"rule": "规则线", "rl": "RL 线", "proxy": "平台代理", "training": "训练对手"}.get(record["line"])
            if cid in offered else "固定对手池",
            "kind": "规则" if rule else "模型",
            # RL versions teach by their argmax; platform proxies are opponents only.
            "teacher": rule or (cid in offered and offered[cid]["record"]["line"] == "rl"),
            "default": {"opponent": cid in pool["members"], "teacher": cid == teacher},
            "score": round(record["score"], 2) if record and record.get("score") is not None else None,
            "se": round(record["se"], 2) if record and record.get("se") is not None else None,
            "decision_ms": costs.get(cid),
            "description": (f"与 {record['equivalent']['to']} 决策完全一致的等价实现：" if record.get("equivalent") else "")
            + record["description"] if cid in offered else None,
        })
    return facts


def catalog_report():
    rows = ["可选训练对手与教师（方案用 cfg[\"opponents\"] 选对手、cfg[\"teacher\"] 选教师；规则与 RL 线版本可当教师，平台代理只作对手。"
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
    config = value["meta"]["config"]
    if any(config.get(k) != v for k, v in SIZES[size(host)].items()):
        raise SystemExit(f"{host} 固定跑 {size(host)} 规格 {SIZES[size(host)]}，起点结构是 {config}")
    value |= {"start_actor": start, "host": host, "size": size(host)}
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
    specs = opponent_specs(host, {cid: {"spec": spec} for cid, spec in E.playing(pool).items()}
                           | {cid: {"spec": e["spec"]} for cid, e in offered.items()}, E.MODELS, pool["timeouts"])
    specs = {cid: (dict(spec, isolation=E.ISOLATION) if spec.get("family") != "model" else spec)
             for cid, spec in specs.items()}
    return ({cid: specs[cid] for cid in pool["members"]}, {cid: specs[cid] for cid in offered},
            host.remote_path(pool["runtime_path"]))


def prebuild_fastgame(host_name, code, runtime):
    """Compile the training framework's C engine (guided/native.c) on the host under the pool's runtime.

    Workers load it on their first training game (_fast.load_native, cached
    per machine by source and runtime); compiled there, the compiler's
    processes would count against the training's process limit, as the rule
    packages' own native.c would (RemoteHost.build_packages). Once built, a
    no-op.
    """
    fastgame = code + "/projects/personal/selfplay/guided/fastgame.py"
    script = ("import subprocess,sys\n"
              "subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,sys.argv[1]);"
              "from mahjong_ai import _fast;_fast.load_native(sys.argv[2])'," + repr(runtime) + "," + repr(fastgame)
              + "],check=True,capture_output=True)\nprint('built')\n")
    remote.execute(host_name, script, timeout=900)


def origin(state):
    """The version an attempt trains from: None for the fixed start."""
    source = state["base"] if state.get("from") == "best" else state.get("from")
    return None if source in (None, "start") else source


def version(attempt_id):
    """An evaluated RL version to train from (any pool): its latest record with an Actor."""
    rows = [r for r in E.attempts(line="rl")
            if r["id"] == attempt_id and ((r.get("spec") or {}).get("parameters") or {}).get("weights")]
    if not rows:
        raise E.Refused(f"{attempt_id} 不是评估过的 RL 版本（记录里没有它的模型），不能从它出发。")
    return rows[-1]


def start_actor(state, host_name, setup_):
    """Where the attempt's training starts on the host, and the training hands already behind it.

    By default the fixed start (setup); with ``from: best`` the Actor of the
    best version the attempt was designed on, so training adds up along the
    line (every step of it still had to beat the best version); with a
    version id, that version's Actor, kept or not (e.g. an imitation to
    try RL on).
    """
    source = origin(state)
    if source is None:
        return setup_["start_actor"], 0
    base = version(source)
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
    # The host's inference service (evaluations; training does not use it) keeps the
    # GPU memory its batches grew to: 5 GB on the 4070 after one evaluation, which with
    # the trainer's cap overran the 12 GB card and stalled 257. The next evaluation starts it again.
    # An evaluation still playing on the host (the previous attempt's verdict or its proxy tables)
    # plays through the service and holds the host's evaluation lock: wait for it, or its seats
    # hang (10-09: 266's proxy tables stalled 4 hours when 268 began training on the 4090).
    import fcntl

    with (E.LOCAL / f"evaluation-{host}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        remote.execute(host, "import subprocess\n"
                       f"subprocess.run(['pkill','-f','guided.model_server .*--port {SERVICE_PORT}'])", timeout=120)
    opponents, offered, runtime = training_opponents(host)
    prebuild_fastgame(host, setup_["code"], runtime)
    # What this attempt could choose from, as the agent saw it.
    E.atomic_json(job / "catalog.json", catalog_facts())
    hands = state.get("hands") or HANDS
    start, behind = start_actor(state, host, setup_)
    task = dict(host=host, remote_dir=remote_job, job_dir=str(job), code=setup_["code"],
                source_id=setup_["source_id"], recipe=str(job / "recipe"), bundle=remote.root(host) + BUNDLE,
                runtime=runtime, start_actor=start, hand_budget=hands,
                budget_seconds=BUDGET,
                label_hand_budget=BUDGETS[size(host)]["max_labeled"],
                seed=state["seed"], line="rl", limits=LIMITS[host], execution=EXECUTION[host],
                opponents=opponents, catalog=offered, opponents_only=sorted(proxies()), fetch_actor=False)
    E.log(f"{attempt_id}: training {hands} hands on {host} from {start}")
    result = recipe_jobs.trial_train(task, lambda message: None)
    E.atomic_json(job / "training.json", {k: v for k, v in result.items() if k != "metrics"})
    status = result["status"]
    training = {"host": host, "state": status.get("state"), "hands": status.get("hands"),
                "labeled_hands": status.get("labeled_hands"), "size": size(host),
                "training_seconds": status.get("training_seconds"), "budget": hands,
                "from": origin(state),
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
    t = sub.add_parser("training-rules", help="offer rule packages in the catalog as training opponents (not evaluated against)")
    t.add_argument("packages", nargs="+")
    t.add_argument("--scores", help="JSON {name: {score, se}} of the packages on the pool's development set")
    q = sub.add_parser("proxies", help="offer a proxy panel's members in the catalog (training opponents only)")
    q.add_argument("panel", help="panel.json of the online calibration")
    u = sub.add_parser("unconfirmed", help="offer RL versions that never became a line's best in the catalog")
    u.add_argument("versions", nargs="*", help="attempt ids; none clears the registration")
    b = sub.add_parser("baseline", help="score the start Actor as the RL line's first best version (once per pool)")
    b.add_argument("--host", default="4070", choices=HOSTS)
    args = parser.parse_args()
    if args.command == "setup":
        print(json.dumps(setup(args.source, args.name, args.host), ensure_ascii=False))
    elif args.command == "training-rules":
        scores = E.read_json(Path(args.scores)) if args.scores else None
        print(f"已登记 {register_training_rules(args.packages, scores)} 个训练对手")
    elif args.command == "proxies":
        print(f"已登记 {register_proxies(args.panel)} 个平台代理")
    elif args.command == "unconfirmed":
        print(f"已登记 {register_unconfirmed(args.versions)} 个未成为最好的 RL 版本")
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
