"""Evaluate one attempt of an iteration line against the fixed opponent pool (ADR-0016).

Every game seats the attempt once and three distinct pool members; a block is
one lineup on one wall with the attempt in each of the four seats. Lineups
cycle through all C(7, 3) = 35 opponent triples. Walls and every policy's seed
depend only on the set's seed and the block number, so the development set is
the same games for every attempt and results do not depend on how the blocks
are split into jobs.

Every version (attempt, reference, the RL start) plays the whole development
set, 4096 blocks (16384 games), so one version has one score. An attempt above
the best there goes to a fresh confirmation set, where it and the best each
play 2048 blocks and the paired difference must have a one-sided 90% lower
bound above zero. A best version that changes during the confirmation is
first compared on the development set, where its scores are on record.

The maintainer seals the evaluation conditions (conditions(): the harness
code, the pool, and each RL host's frozen training framework and start); an
attempt refuses to start or resume under anything else until the maintainer
reseals. Shortened settings (FP_DEV, FP_CONFIRM, FP_RL_HANDS, FP_RL_BUDGET)
are accepted only with FP_LOCAL, a separate record for pipeline checks.

Games run without sandboxes: the pool's packages are the maintainer's own, so
each match process plays them in-process (candidate_runtime.InlinePolicy) and
several tables at once, and model seats share the host's inference service.
None of that changes a game's outcome, so these are per-host execution
settings (EVAL_WORKERS, EVAL_TABLES), not part of the pool's identity.

Maintainer commands (from projects/personal/firstprinciples); iteration agents use harness.line:
  python -m harness.evaluate runtime                                    # freeze src/mahjong_ai as a runtime
  python -m harness.evaluate pool [--model <weights-sha> --name <name>] [--rules <ids or package dirs>] [--runtime <dir>] [--replace]
  python -m harness.evaluate drain on "<why>" | off                    # stop new attempts while the pool is replaced
  python -m harness.evaluate rebase <attempt> ... [--host <host>]       # replay earlier versions on the new pool
  python -m harness.evaluate rebase-finish rule|rl                      # record them; the highest is the line's best
  python -m harness.evaluate seal                                       # after any change to the conditions
  python -m harness.evaluate reference <research candidate | model name> [--host local|4070|4090|5070ti]
  python -m harness.evaluate implement <pool member> <package dir>      # an equivalent, faster implementation
"""
import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from mahjong_ai.execution.candidate_runtime import tree_hash  # noqa: E402
from mahjong_ai.execution.remote import Dispatcher, RemoteHost  # noqa: E402
from mahjong_ai.persistence.artifacts import atomic_json, read_json  # noqa: E402
from mahjong_ai.research.service import Research  # noqa: E402

class Refused(Exception):
    """The harness will not do this; the message says why and who can change it."""


class Unsealed(Refused):
    """The conditions differ from the seal: the maintainer changed them and has not resealed yet."""


# FP_LOCAL runs a separate record for pipeline checks; only there may settings be shortened.
LOCAL = Path(os.environ.get("FP_LOCAL", HERE / ".local"))
TEST = "FP_LOCAL" in os.environ


def override(name, default, cast=int):
    value = os.environ.get(name)
    if value is None:
        return default
    if not TEST:
        raise SystemExit(f"{name} 只能和 FP_LOCAL 一起用于流程检查，正式记录不接受缩短的设置")
    return cast(value)



RESEARCH = REPO / "projects/agent-research"
MODELS = RESEARCH / ".local/research/models"
# Each line evaluates where it already is: rules on this machine, RL on the GPU
# host that trained it (no transfer); an attempt names its host explicitly.
EVAL_HOST = {"rule": "local", "reference": "local", "rl": "4070"}
# Match processes per host and tables (games played at once, one thread each)
# per process. The pool's seats are all rules that cost milliseconds, so a
# process is CPU-bound and a second table only makes seats wait on the
# interpreter lock (timeouts doubled in calibration); one process per core
# the host can spare for its desktop.
EVAL_WORKERS = {"local": 12, "4070": 24, "4090": 14, "5070ti": 12}
EVAL_TABLES = {"local": 1, "4070": 1, "4090": 1, "5070ti": 1}
ISOLATION = "none"  # the pool's packages run in the match process (see module docstring)
POLL_SECONDS = 5  # how often remote job results are checked
DEV_BLOCKS = override("FP_DEV", 4096)  # 4 games each
CONFIRM_BLOCKS = override("FP_CONFIRM", 2048)
JOB_RETRIES = 3  # a job that fails this often (not a host fault) fails the attempt
CONFIRM_Z = 1.2816  # one-sided 90%
TIMEOUTS = {"discard": 2.0, "response": 0.6}  # platform limits minus network and concurrency margin
RULE_OPPONENTS = [
    "heuristic-efficiency--01--piao-call-channel",
    "heuristic-efficiency--01--material-dose-double",
    "heuristic-fan--01--chi-quota-all-v1",
    "heuristic-fan--01--kong-reserve-white8-rep2",
    "heuristic-adaptive--01--far_chi_flat3_rep",
    "heuristic-unrestricted--01--baotou-tenpai-price-009-rep2",
]


def log(message):
    print(time.strftime("%H:%M:%S ") + message, flush=True)


def research_state():
    return Research(RESEARCH).read()


def with_timeouts(spec):
    return dict(spec, action_seconds=TIMEOUTS["discard"], response_seconds=TIMEOUTS["response"])


# -- pool ----------------------------------------------------------------------
def freeze_runtime():
    """Copy src/mahjong_ai into the research runtime store under its fingerprint."""
    import subprocess
    import tempfile

    store = RESEARCH / ".local/research/runtimes"
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=store, prefix=".freezing-") as tmp:
        shutil.copytree(REPO / "src" / "mahjong_ai", Path(tmp) / "mahjong_ai",
                        ignore=shutil.ignore_patterns("__pycache__"))
        fingerprint = subprocess.run(
            [sys.executable, "-c", "from mahjong_ai.persistence.artifacts import runtime_fingerprint;print(runtime_fingerprint())"],
            env=dict(os.environ, PYTHONPATH=tmp, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, check=True).stdout.strip()
        target = store / fingerprint
        if not target.exists():
            shutil.move(tmp, target)
            Path(tmp).mkdir()  # the context manager removes it
    return str(target)


def member_spec(entry, s, runtime):
    """A rule member: a research candidate by id, or a package directory frozen under .local/pool-members."""
    if entry in s["candidates"]:
        return entry, dict(s["candidates"][entry]["spec"], runtime_path=runtime)
    package = Path(entry)
    if not (package / "strategy.py").is_file():
        raise SystemExit(f"{entry} is neither a research candidate nor a package directory")
    name = package.name
    target = LOCAL / "pool-members" / name / f"package-{tree_hash(package)[:16]}"
    if not target.exists():
        shutil.copytree(package, target, ignore=shutil.ignore_patterns("__pycache__"))
    return name, {"family": "code", "version": name, "parameters": {}, "model_path": None, "program": None,
                  "package_path": str(target), "package_hash": tree_hash(target),
                  "python_path": str(REPO / ".venv/bin/python"), "runtime_path": runtime}


def freeze_pool(model_weights, model_name, runtime=None, replace=False, rules=None):
    """Fix the seven opponents and the development seed; refuses to change a pool in use.

    ``replace`` starts a new record on purpose (a new runtime or new members):
    earlier attempts stay in attempts.jsonl under the old pool id and drop out
    of history until rebase() replays the versions worth keeping. A member
    kept from the old pool keeps its registered equivalent implementation.
    """
    path = LOCAL / "pool.json"
    s = research_state()
    old = read_json(path) if path.exists() else {}
    # A new pool keeps the current one's runtime unless told otherwise: the
    # research experiment's may be older than what the pool was built on.
    runtime = runtime or old.get("runtime_path") or s["candidate_runtime_path"]
    members = {}
    for entry in rules or RULE_OPPONENTS:
        cid, spec = member_spec(entry, s, runtime)
        members[cid] = with_timeouts(spec)
    if model_weights:  # the pool may be rules only
        if not (MODELS / model_weights).is_dir():
            raise SystemExit(f"model weights {model_weights} are not under {MODELS}")
        members[model_name] = with_timeouts(
            {"family": "model", "version": model_name, "parameters": {"weights": model_weights}}
        )
    pool = {"members": members, "runtime_path": runtime, "config": s["config"]["game"],
            "resources": s["config"]["resources"], "timeouts": TIMEOUTS}
    pool["id"] = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()[:16]
    pool["dev_seed"] = int(pool["id"], 16)
    if not replace and old and old["id"] != pool["id"] and attempts(old["id"]):
        raise SystemExit("the current pool already has attempts; a new pool starts a new record (--replace)")
    kept = {cid: impl for cid, impl in (old.get("implementations") or {}).items()
            if old["members"].get(cid) == members.get(cid)}
    if kept:
        pool["implementations"] = kept
    atomic_json(path, pool)
    return pool


def load_pool():
    return read_json(LOCAL / "pool.json")


def playing(pool):
    """The pool's members as games run them.

    A member with a registered equivalent implementation (implement(): the
    same decision in every position checked, only faster) runs that package.
    The pool's identity, and so every score under it, stays the frozen
    members'; the registration is part of the sealed conditions.
    """
    implementations = pool.get("implementations") or {}
    return {cid: dict(spec, package_path=implementations[cid]["package_path"],
                      package_hash=implementations[cid]["package_hash"]) if cid in implementations else spec
            for cid, spec in pool["members"].items()}


IMPLEMENTATION_HANDS = override("FP_IMPLEMENTATION_HANDS", 16384)  # it plays in every game from now on


def implement(member, package):
    """Register ``package`` as the equivalent implementation of a rule member of the pool.

    It must choose what the frozen member chooses in every position of fresh
    games (harness/equivalence.py, on more hands than an attempt, since it
    plays in every evaluation and training game from now on). The maintainer
    reseals afterwards.
    """
    import fcntl

    from . import equivalence

    pool = load_pool()
    spec = pool["members"].get(member)
    if spec is None or spec.get("family") == "model":
        raise Refused(f"{member} 不是对手池里的规则成员")
    problems = check_source(package)
    if problems:
        raise Refused("不符合迭代线约定：\n" + "\n".join(problems))
    root = LOCAL / "pool-implementations" / member
    target = root / f"package-{tree_hash(package)[:16]}"
    if not target.exists():
        shutil.copytree(package, target, ignore=shutil.ignore_patterns("__pycache__"))
    seed = random.SystemRandom().getrandbits(62)
    with (LOCAL / "evaluation-local.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # alone on the Mac, like an evaluation
        result = equivalence.check(spec["package_path"], target, seed, root / "check", hands=IMPLEMENTATION_HANDS)
    if result["mismatches"]:
        raise Refused(f"{result['decisions']} 个决策里 {result['mismatches']} 个与 {member} 不同：\n"
                      + "\n".join(result["examples"]))
    summary = {"package_path": str(target), "package_hash": tree_hash(target), "seed": seed,
               "hands": result["hands"], "decisions": result["decisions"],
               "time_ratio": result["seconds"][1] / max(result["seconds"][0], 1e-9), "at": time.time()}
    pool = load_pool()
    pool.setdefault("implementations", {})[member] = summary
    atomic_json(LOCAL / "pool.json", pool)
    return summary


def conditions():
    """Digest of everything an attempt's score depends on besides the attempt itself.

    The harness code (evaluation, training budget and execution, the loop),
    the pool, and each RL host's frozen training framework and start Actor.
    The iteration agent changes none of these; the maintainer reseals after
    changing any of them.
    """
    h = hashlib.sha256()
    for path in sorted((HERE / "harness").glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    pool = {k: v for k, v in load_pool().items() if k != "harness"}
    h.update(json.dumps(pool, sort_keys=True).encode())
    for path in sorted((LOCAL / "rl").glob("setup-*.json")):
        s = read_json(path)
        h.update(json.dumps([path.name, s.get("source_id"), s.get("weights"), s.get("start_actor")]).encode())
    return h.hexdigest()


def seal():
    """Accept the current conditions (the maintainer, after changing any of them)."""
    value = {"conditions": conditions(), "at": time.time()}
    atomic_json(LOCAL / "seal.json", value)
    return value["conditions"]


def check_seal():
    path = LOCAL / "seal.json"
    if not path.exists():
        raise Refused("评估条件还没有封存：维护者确认条件后运行 python -m harness.evaluate seal。")
    if read_json(path)["conditions"] != conditions():
        raise Unsealed("评估条件（harness 代码、对手池或 RL 主机的训练框架与起点）与封存时不一致，拒绝运行。"
                      "这些归维护者管：撤回改动，或由维护者确认后重新 seal。")


def lineups(pool):
    return list(itertools.combinations(sorted(pool["members"]), 3))


def assignments(pool, attempt_id, blocks):
    """Four games per block: the attempt in each seat, opponents in fixed order."""
    triples = lineups(pool)
    out = []
    for block in blocks:
        others = list(triples[block % len(triples)])
        for focal in range(4):
            players = list(others)
            players.insert(focal, attempt_id)
            out.append(dict(index=len(out), block=block, players=players, focal=focal))
    return out


# -- attempts --------------------------------------------------------------------
def records_path():
    return LOCAL / "attempts.jsonl"


def attempts(pool_id=None, line=None):
    if not records_path().exists():
        return []
    rows = [json.loads(l) for l in records_path().read_text().splitlines() if l.strip()]
    return [r for r in rows if (pool_id is None or r["pool"] == pool_id) and (line is None or r["line"] == line)]


def version(record):
    """What makes records the same version: the model's weights or the code package's hash."""
    spec = record.get("spec") or {}
    return (spec.get("parameters") or {}).get("weights") or spec.get("package_hash") or record["id"]


def distinct(rows):
    """One record per version, for showing and ranking (the log itself only grows).

    A later record of an id replaces the earlier one (a reference played
    again); of records of the same version, the one over the most games
    stands (the RL start is also the imitation model's reference).
    """
    latest = {r["id"]: r for r in rows}
    chosen = {}
    for r in latest.values():
        held = chosen.get(version(r))
        if held is None or (r["games"], r["finished_at"]) >= (held["games"], held["finished_at"]):
            chosen[version(r)] = r
    return [r for r in latest.values() if chosen[version(r)] is r]


def best(pool_id, line):
    kept = [r for r in attempts(pool_id, line) if r["status"] == "keep"]
    return kept[-1] if kept else None


def running(line=None):
    """Attempts reserved but not yet recorded (one per GPU host may be under way)."""
    rows = [read_json(p) for p in sorted((LOCAL / "running").glob("*.json"))]
    return [r for r in rows if line is None or r["line"] == line]


@contextlib.contextmanager
def attempts_lock():
    """Serializes attempt ids, host claims and the line leases (harness.line)."""
    import fcntl

    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / "running").mkdir(exist_ok=True)
    with (LOCAL / "attempts.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def next_attempt_id(line):
    """The line's next attempt id; call under attempts_lock().

    Packages, training receipts and job directories are keyed by attempt id,
    so no id is ever reused: not after a pool replacement, not after an
    attempt whose record never got written.
    """
    taken = [r["id"] for r in attempts(line=line)] + [r["id"] for r in running(line)]
    taken += [p.name for d in (LOCAL / "attempts", LOCAL / "rl" / "attempts") if d.is_dir() for p in d.iterdir()]
    prefix = f"fp-{line}-"
    numbers = [int(i[len(prefix):]) for i in taken if i.startswith(prefix) and i[len(prefix):].isdigit()]
    return f"{prefix}{max(numbers, default=-1) + 1:03d}"


def reserve_attempt(line, host, description, **state):
    """Reserve the line's next attempt id with its state file (running/<id>.json)."""
    with attempts_lock():
        attempt_id = next_attempt_id(line)
        atomic_json(LOCAL / "running" / f"{attempt_id}.json",
                    dict(id=attempt_id, line=line, host=host, description=description,
                         started_at=time.time(), **state))
    return attempt_id


def release_attempt(attempt_id):
    (LOCAL / "running" / f"{attempt_id}.json").unlink(missing_ok=True)
    (LOCAL / "running" / f"{attempt_id}.lock").unlink(missing_ok=True)


# A strategy or recipe runs in the process the harness gives it and keeps to
# it: no processes, thread pools or native code of its own. In a game that
# would take cores from the opponents at the table; in training, cores the
# host keeps for itself.
FORBIDDEN_IMPORTS = ("subprocess", "multiprocessing", "torch.multiprocessing", "ctypes", "cffi", "importlib")
FORBIDDEN_NAMES = ("ProcessPoolExecutor", "set_num_threads", "set_num_interop_threads",
                   "set_per_process_memory_fraction")
BUILTINS = ("__import__", "eval", "exec")  # would hide the rest from this check
# os functions that start processes or change scheduling.
FORBIDDEN_OS = ("system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp", "nice",
                "setpriority", "sched_setaffinity")


def check_source(package, imports=()):
    """What in a line's source breaks its rules (empty when none); start refuses on any."""
    import ast

    package = Path(package)
    problems = []
    for path in sorted(package.rglob("*.py")):
        name = path.relative_to(package)
        try:
            tree = ast.parse(path.read_text(), str(path))
        except SyntaxError as exc:
            problems.append(f"{name}:{exc.lineno} 语法错误：{exc.msg}")
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for module in modules:
                if any(module == f or module.startswith(f + ".") for f in (*FORBIDDEN_IMPORTS, *imports)):
                    problems.append(f"{name}:{node.lineno} 导入了 {module}")
            word = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            spawn = (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os"
                     and (node.attr in FORBIDDEN_OS or node.attr.startswith(("exec", "spawn"))))
            if word in FORBIDDEN_NAMES or spawn or (isinstance(node, ast.Name) and word in BUILTINS):
                problems.append(f"{name}:{node.lineno} 用了 {word}")
    return problems


def freeze_rule(attempt_id):
    source = HERE / "rule"
    target = LOCAL / "attempts" / attempt_id / "package"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    return {"family": "code", "version": attempt_id, "parameters": {}, "model_path": None,
            "program": None, "package_path": str(target), "package_hash": tree_hash(target),
            "python_path": str(REPO / ".venv/bin/python")}


# Where each line keeps its editable source and each attempt's frozen copy of it.
SOURCES = {"rule": HERE / "rule", "rl": HERE / "rl" / "recipe"}


def frozen_source(record):
    if record["line"] == "rule":
        return Path(record["spec"]["package_path"])
    return LOCAL / "rl" / "attempts" / record["id"] / "recipe"


STAGE_NAMES = {"queued": "排队等主机", "training": "训练中", "evaluating": "评估中", "equivalence": "等价检查中"}


# -- replacing the pool -------------------------------------------------------------
def drain_reason():
    """Why the maintainer holds new attempts back (drain on), or None."""
    path = LOCAL / "drain.json"
    return read_json(path)["why"] if path.exists() else None


def set_drain(why):
    path = LOCAL / "drain.json"
    if why:
        atomic_json(path, {"why": why, "at": time.time()})
    else:
        path.unlink(missing_ok=True)


def rebased_path(pool):
    return LOCAL / "rebase" / pool["id"]


def rebase(ids, host="local"):
    """Play earlier versions on the current pool's development set; rebase_finish records them.

    For a new pool: each line's earlier best versions (and the free-play
    entrant) keep their ids, so recipes that name a teacher or an opponent
    still find it. Resumable; hosts may run disjoint ids at once.
    """
    pool = load_pool()
    check_seal()
    done = []
    for cid in ids:
        out = rebased_path(pool) / f"{cid}.json"
        if out.exists():
            continue
        rows = [r for r in attempts() if r["id"] == cid and r.get("spec") and r["pool"] != pool["id"]
                and r["status"] in ("keep", "reference")]
        if not rows:
            raise Refused(f"{cid} 在之前的对手池下没有成功或参照记录")
        old = rows[-1]
        started = time.time()
        scores, stats = play(pool, cid, old["spec"], range(DEV_BLOCKS), pool["dev_seed"], "dev", host)
        score, se = mean_se(scores[k] for k in range(DEV_BLOCKS))
        atomic_json(out, dict(old, pool=pool["id"], host=host, started_at=started, finished_at=time.time(),
                              games=DEV_BLOCKS * 4, score=score, se=se, stats=stats, best_before=None, diff=None,
                              confirm=None, reference=None, dev_blocks={str(k): scores[k] for k in sorted(scores)},
                              rebased={"from_pool": old["pool"], "old_score": old["score"], "old_status": old["status"]}))
        done.append(cid)
        log(f"{cid} rebased on {pool['id']}: {score:+.2f} ± {se:.2f}")
    return done


def rebase_finish(line):
    """Record the replayed versions of a line, lowest score first: the highest becomes its best.

    Run after the RL start's baseline, whose record would otherwise come last.
    They are not news for the iteration agents, so they are marked reported.
    """
    pool = load_pool()
    rows = [read_json(p) for p in sorted(rebased_path(pool).glob("*.json"))]
    rows = [r for r in rows if r["line"] == line and r["id"] not in {x["id"] for x in attempts(pool["id"])}]
    rows.sort(key=lambda r: r["score"])
    for r in rows:
        with records_path().open("a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mark_reported(rows)
    return rows


def history(line):
    """Every attempt of the line, its best version and the reference, for designing the next one."""
    pool = load_pool()
    shown = distinct(attempts(pool["id"]))
    rows = [r for r in shown if r["line"] == line]
    champion = best(pool["id"], line)
    out = [f"固定对手池 {pool['id']}；{LINE_NAMES[line]}共 {len(rows)} 条记录，"
           + (f"当前最好 {champion['id']}（{champion['score']:+.2f}）" if champion else "还没有成功的版本")]
    for r in (r for r in shown if r["line"] == "reference"):
        out.append(f"参照 {r['id']}：{r['score']:+.2f} ± {r['se']:.2f}（{r['games']} 场）")
    for r in rows:
        score = "—" if r["score"] is None else f"{r['score']:+.2f}±{r['se']:.2f}"
        diff = "" if r.get("diff") is None else f" 差{r['diff']:+.2f}"
        lower = f" 确认下界{r['confirm']['lower90']:+.2f}" if r.get("confirm") else ""
        design = "" if r.get("design_seconds") is None else f" 设计{r['design_seconds'] / 60:.0f}分"
        host = f" @{r['host']}" if r.get("host") not in (None, "local") else ""
        base = f" 基于{r['base']}" if r.get("base") else ""
        cost = f" 每决策{r['decision_ms']['mine']:.2g}ms" if r.get("decision_ms") else ""
        kind = (f" 等价替换（沿用{r['equivalent']['to']}的成绩）" if r.get("equivalent")
                else f" 换池重算（旧池 {r['rebased']['old_score']:+.2f}）" if r.get("rebased") else "")
        t = r.get("training") or {}
        trained = (f" 训练{t['hands'] / 10000:.0f}万局" + (f"（续训自{t['from']}，累计{t['lineage_hands'] / 10000:.0f}万局）"
                                                         if t.get("from") else "")) if t.get("hands") else ""
        out.append(f"{r['id']} {STATUS[r['status']]}{kind}{host}{trained} {r['games']}场 {score}{diff}{lower}{cost}{design}{base}"
                   f"｜{r['description']}")
        if r.get("why"):
            out.append(f"    假设：{r['why']}")
        if r.get("error"):
            # A mismatch report's head says how many; its positions were in the report.
            reason = r["error"].splitlines()[0] if r["status"] == "inequivalent" else r["error"][-300:]
            out.append(f"    原因：{reason}")
    for r in running(line):
        minutes = (time.time() - r["started_at"]) / 60
        where = f" @{r['host']}" if r.get("host") else ""
        plan = (f" {r['hands'] / 10000:g}万局" + (f" 续训自{r['base']}" if r.get("from") == "best" else "")
                if r.get("hands") else "")
        out.append(f"{r['id']} {STAGE_NAMES.get(r.get('stage'), '进行中')}{where}{plan} {minutes:.0f}分｜{r['description']}")
    return "\n".join(out)


def freeze_model(attempt_id, actor_dir):
    """Register an exported Actor under the research model store by its weights hash."""
    actor = Path(actor_dir)
    weights = hashlib.sha256((actor / "actor.pt").read_bytes()).hexdigest()
    target = MODELS / weights
    if not target.exists():
        shutil.copytree(actor, target)
    return {"family": "model", "version": attempt_id, "parameters": {"weights": weights}}


# -- execution -------------------------------------------------------------------
_dispatchers = {}


def remote_host(name):
    sys.path.insert(0, str(REPO / "projects/personal/selfplay"))
    from guided.remote import PROFILES

    return RemoteHost(name, dict(PROFILES[name], workers=EVAL_WORKERS[name]), REPO)


def dispatcher(name):
    if name not in _dispatchers:
        _dispatchers[name] = Dispatcher(remote_host(name), MODELS, poll_seconds=POLL_SECONDS)
    return _dispatchers[name]


def play(pool, attempt_id, spec, blocks, seed, label, host):
    """Play the blocks on host ("local" or a remote); returns ({block: table score}, stats).

    One evaluation per host at a time: two attempts sharing a machine would
    oversubscribe it, and the resulting decision timeouts would lower both scores.
    """
    import fcntl

    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / f"evaluation-{host}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if host == "local":
            return _play_local(pool, attempt_id, spec, blocks, seed, label)
        return _play(pool, attempt_id, spec, blocks, seed, label, host)


def _tasks(pool, attempt_id, spec, blocks, seed, label, host):
    """Job directories and tasks: one job per worker over contiguous blocks."""
    blocks = list(blocks)
    size = -(-len(blocks) // EVAL_WORKERS[host])
    policies = dict(playing(pool), **{attempt_id: with_timeouts(spec)})
    resources = dict(pool["resources"], isolation=ISOLATION, tables=EVAL_TABLES[host])
    for n in range(0, len(blocks), size):
        # Under the pool id: a finished job is reused on restart, and a job of
        # another pool (other runtime, other walls) must never be.
        root = LOCAL / "jobs" / pool["id"] / attempt_id / f"{label}-{n // size:02d}"
        task = dict(kind="match", project=str(REPO), experiment="firstprinciples",
                    runtime_path=pool["runtime_path"], runtime="", config=pool["config"],
                    timeouts=pool["timeouts"], resources=resources, seed=seed,
                    assignments=assignments(pool, attempt_id, blocks[n:n + size]),
                    scope="paired", candidate=attempt_id, policies=policies,
                    deadline=time.time() + 86400, execution_limit_seconds=None, owner_pid=os.getpid())
        yield root, task


def _collect(value, scores, stats):
    focal = value["focal"]
    scores.update(zip(focal["group_blocks"], focal["field_score_group_means"]))
    for key in stats:
        stats[key] += int(focal.get(key) or 0)


def _job_env(pool):
    return {"MAHJONG_MEMORY_MB": str(pool["resources"]["memory_mb"]),
            **{k: str(pool["resources"]["threads"]) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}


_services = {}


def _play_local(pool, attempt_id, spec, blocks, seed, label):
    """Run the jobs as local processes, the way the research executor does."""
    import subprocess

    from mahjong_ai.execution.model_seat import ENCODER_ENV
    from mahjong_ai.execution.model_service import SERVER_ENV, ModelService

    runtime = pool["runtime_path"]
    env = dict(os.environ, PYTHONPATH=runtime, **_job_env(pool))
    fingerprint = subprocess.run(
        [sys.executable, "-c", "from mahjong_ai.persistence.artifacts import runtime_fingerprint;print(runtime_fingerprint())"],
        env=env, capture_output=True, text=True, check=True).stdout.strip()
    weights = {m["parameters"]["weights"] for m in [*pool["members"].values(), spec] if m.get("family") == "model"}
    if weights:
        service = _services.setdefault("local", ModelService(root=MODELS, log=LOCAL / "model-server.log"))
        service.warm(weights)
        env[SERVER_ENV], env[ENCODER_ENV] = service.url(), str(service.project)
    scores, stats, running = {}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}, []
    for root, task in _tasks(pool, attempt_id, spec, blocks, seed, label, "local"):
        result = root / "result.json"
        if result.exists() and read_json(result).get("status") == "completed":
            _collect(read_json(result), scores, stats)
            continue
        root.mkdir(parents=True, exist_ok=True)
        result.unlink(missing_ok=True)
        atomic_json(root / "task.json", dict(task, runtime=fingerprint))
        with (root / "worker.log").open("ab") as out:
            running.append((root, subprocess.Popen(
                [sys.executable, "-B", "-m", "mahjong_ai.execution.jobs", str(root)], cwd=REPO, env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)))
    for root, process in running:
        process.wait()
        value = read_json(root / "result.json") if (root / "result.json").exists() else {}
        if value.get("status") != "completed":
            raise RuntimeError(f"{root.name}: {value.get('status')} {str(value.get('error'))[:300]}")
        _collect(value, scores, stats)
    return scores, stats


def _play(pool, attempt_id, spec, blocks, seed, label, host):
    env = _job_env(pool)
    roots = []
    for root, task in _tasks(pool, attempt_id, spec, blocks, seed, label, host):
        roots.append(root)
        result = root / "result.json"
        if result.exists() and read_json(result).get("status") == "completed":
            continue
        root.mkdir(parents=True, exist_ok=True)
        result.unlink(missing_ok=True)
        atomic_json(root / "task.json", task)
        job = f"{attempt_id}/{root.name}"
        if not dispatcher(host).adopt(job, root):
            dispatcher(host).submit(job, root, task, pool["runtime_path"], env)
    scores, stats = {}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}
    pending, failures = list(roots), {}
    while pending:
        time.sleep(POLL_SECONDS)
        for root in list(pending):
            path = root / "result.json"
            if not path.exists():
                continue
            value = read_json(path)
            if value.get("status") != "completed":
                if value.get("status") == "host_fault":
                    # The host lost the job or cannot be reached: not the attempt's
                    # fault. Relaunch once the host answers again, however long.
                    log(f"{root.name} host_fault: {str(value.get('error'))[:200]}; waiting for {host}")
                    while not dispatcher(host)._reachable():
                        time.sleep(60)
                else:
                    failures[root] = failures.get(root, 0) + 1
                    if failures[root] >= JOB_RETRIES:
                        raise RuntimeError(f"评估任务 {root.name} 连续 {JOB_RETRIES} 次失败：{str(value.get('error'))[-1500:]}")
                    log(f"{root.name} {value.get('status')}: {str(value.get('error'))[:200]}; relaunching")
                path.unlink()
                dispatcher(host).submit(f"{attempt_id}/{root.name}", root, read_json(root / "task.json"),
                                        pool["runtime_path"], env)
                continue
            _collect(value, scores, stats)
            pending.remove(root)
    return scores, stats


def paired(a, b, blocks):
    diffs = [a[k] - b[k] for k in blocks]
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    return mean, sd / math.sqrt(len(diffs))


def mean_se(values):
    values = list(values)
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return mean, sd / math.sqrt(len(values))


def evaluate(line, spec, description, attempt_id=None, design_seconds=None, host=None,
             confirm_seed=None, extra=None):
    """Play the development set and, above the best, the confirmation set; append the record.

    Attempts of a line may run on several hosts at once, so the best version
    is re-read whenever it is compared against: the development blocks are the
    same games for everyone, and a champion that changed during the
    confirmation set replays those fresh walls before the paired test.

    Calling it again with the same attempt id and confirm_seed after an
    interruption reuses every finished job (harness.line resumes attempts so).
    ``extra`` joins the record (hypothesis, base version, training receipt).
    """
    pool = load_pool()
    check_seal()
    attempt_id = attempt_id or reserve_attempt(line, host or EVAL_HOST[line], description)
    # A reference (the version on the platform) plays a fixed set and is never compared.
    reference = line == "reference"
    host = host or EVAL_HOST[line]
    started = time.time()
    confirm = None
    scores, stats = play(pool, attempt_id, spec, range(DEV_BLOCKS), pool["dev_seed"], "dev", host)
    champion = None if reference else best(pool["id"], line)
    if champion is None:
        verdict = "reference" if reference else "keep"
    else:
        diff, se = paired(scores, {int(k): v for k, v in champion["dev_blocks"].items()}, range(DEV_BLOCKS))
        log(f"{attempt_id} dev {DEV_BLOCKS * 4} games: diff {diff:+.3f} ± {se:.3f} vs {champion['id']}")
        verdict = "discard" if diff <= 0 else None
    if verdict is None:
        seed = confirm_seed or random.SystemRandom().getrandbits(62)
        mine, _ = play(pool, attempt_id, spec, range(CONFIRM_BLOCKS), seed, "confirm", host)
        while True:
            current = best(pool["id"], line)
            if current["id"] != champion["id"]:
                # A new best version appeared meanwhile. Its development scores are
                # on record: the attempt must lead it there before it is worth
                # replaying the fresh walls for.
                champion = current
                diff, se = paired(scores, {int(k): v for k, v in champion["dev_blocks"].items()}, range(DEV_BLOCKS))
                log(f"{attempt_id}: best version is now {champion['id']}; dev diff {diff:+.3f} ± {se:.3f}")
                if diff <= 0:
                    verdict = "discard"
                    break
            # The best version replays the same fresh walls under its own id.
            theirs, _ = play(pool, champion["id"], champion["spec"], range(CONFIRM_BLOCKS), seed,
                             f"confirm-for-{attempt_id}", host)
            if best(pool["id"], line)["id"] == champion["id"]:
                break
        if verdict is None:
            diff, se = paired(mine, theirs, range(CONFIRM_BLOCKS))
            confirm = {"seed": seed, "games": CONFIRM_BLOCKS * 4, "diff": diff, "se": se,
                       "lower90": diff - CONFIRM_Z * se, "against": champion["id"]}
            verdict = "keep" if diff - CONFIRM_Z * se > 0 else "unconfirmed"
    score, se = mean_se(scores[k] for k in range(DEV_BLOCKS))
    record = dict(
        id=attempt_id, line=line, pool=pool["id"], status=verdict, description=description, host=host,
        started_at=started, finished_at=time.time(), games=DEV_BLOCKS * 4, score=score, se=se,
        best_before=champion["score"] if champion else None,
        diff=paired(scores, {int(k): v for k, v in champion["dev_blocks"].items()}, range(DEV_BLOCKS))[0] if champion else None,
        confirm=confirm, stats=stats, spec=spec, dev_blocks={str(k): scores[k] for k in sorted(scores)},
        design_seconds=design_seconds,
    ) | (extra or {})
    if verdict == "keep":
        record["reference"] = against_platform(record)
    append_record(record)
    return record


def append_record(record):
    """The attempt's one record; its reservation ends with it."""
    with records_path().open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    release_attempt(record["id"])


def reported():
    """Records already shown to the user (by harness.line wait)."""
    path = LOCAL / "reported.json"
    return set(read_json(path)) if path.exists() else set()


def mark_reported(records):
    if records:
        atomic_json(LOCAL / "reported.json", sorted(reported() | {r["id"] for r in records}))


def platform_entrant():
    pin = REPO / ".local/competition/entrant.pin"
    return pin.read_text().strip() if pin.exists() else None


def against_platform(record):
    """A new best version against the version in free play, on their common development blocks.

    The free-play version's reference (or its own attempt record) played the
    first development blocks; the paired difference there flags a version the
    user may want to put on the platform. It was selected on these games, so
    the flag is optimistic; the user decides.
    """
    entrant = platform_entrant()
    if not entrant or entrant == record["id"]:
        return None
    rows = [r for r in attempts(record["pool"]) if r["id"] in (entrant, f"ref-{entrant}")
            and r["status"] in ("keep", "reference")]
    if not rows:
        return {"against": entrant, "missing": True}
    theirs = {int(k): v for k, v in rows[-1]["dev_blocks"].items()}
    mine = {int(k): v for k, v in record["dev_blocks"].items()}
    blocks = sorted(set(theirs) & set(mine))
    diff, se = paired(mine, theirs, blocks)
    lower = diff - CONFIRM_Z * se
    return {"against": entrant, "games": len(blocks) * 4, "diff": diff, "se": se,
            "lower90": lower, "above": lower > 0}


STATUS = {"keep": "成功", "discard": "失败", "unconfirmed": "开发集高但未确认",
          "reference": "参照", "crash": "崩溃", "invalid": "无效", "cancelled": "已取消", "inequivalent": "不等价"}
LINE_NAMES = {"rule": "规则线", "rl": "RL 线", "reference": "参照"}


def ranking(record):
    """Place among the best versions of both lines and the references (scores are comparable)."""
    pool_id = record["pool"]
    rows = [r for r in attempts(pool_id, "reference")] + [b for b in (best(pool_id, "rule"), best(pool_id, "rl")) if b]
    rows = distinct([r for r in rows if r["id"] != record["id"]] + [record])
    rows.sort(key=lambda r: -r["score"])
    return "；".join(f"{i + 1}. {r['id']} {r['score']:+.2f}" for i, r in enumerate(rows))


def report(record):
    """The fixed per-attempt message to the user (handbooks/common.md)."""
    status = STATUS[record["status"]]
    name = LINE_NAMES[record["line"]]
    number = record["id"].rsplit("-", 1)[-1]
    head = f"尝试 {int(number)} · {name}（{record['id']}）" if number.isdigit() else f"{name}（{record['id']}）"
    where = f"，{record['host']}" if record.get("host") not in (None, "local") else ""
    training = record.get("training") or {}
    trained = (f"训练：{training['hands']:,} 局，{training['training_seconds'] / 60:.0f} 分钟"
               + (f"，从 {training['from']} 继续（这条路径累计 {training['lineage_hands']:,} 局）" if training.get("from") else "")
               if training.get("hands") and training.get("training_seconds") else None)
    cost = record.get("decision_ms")
    cost = cost and (f"每决策 {cost['mine']:.2g} ms" + (f"（此前最好 {cost['base']:.2g} ms）" if cost.get("base") else "")
                     + "，Mac 单进程实测")
    if record["score"] is None:
        # A mismatch report names positions; keep all of it.
        reason = (record.get("error") or "")[-(1500 if record["status"] == "inequivalent" else 300):]
        return "\n".join([f"{head}：{status}{f'（{where[1:]}）' if where else ''}", f"改动：{record['description']}",
                          *([trained] if trained else []), *([cost] if cost else []), f"原因：{reason}"])
    same = record.get("equivalent")
    if same:
        lines = [f"{head}：{status}（等价替换，没有跑开发集）",
                 f"改动：{record['description']}",
                 f"等价检查：{same['hands']} 局新牌局的 {same['decisions']} 个决策与 {same['to']} 完全一致，"
                 f"沿用它的得分 {record['score']:+.2f} ± {record['se']:.2f}；同一批决策上用时为它的 {same['time_ratio']:.0%}"]
    else:
        lines = [f"{head}：{status}（开发集 {record['games']} 场{where}）",
                 f"改动：{record['description']}",
                 *([trained] if trained else []),
                 f"得分：{record['score']:+.2f} ± {record['se']:.2f}"
                 + (f"（此前最好 {record['best_before']:+.2f}，配对差 {record['diff']:+.2f}）" if record["best_before"] is not None else "（基线）")]
    if record["confirm"]:
        c = record["confirm"]
        lines.append(f"确认集：{c['games']} 场配对差 {c['diff']:+.2f} ± {c['se']:.2f}，90% 下界 {c['lower90']:+.2f}")
    if cost:
        lines.append(cost)
    s = record["stats"]
    if not same:
        lines.append(f"超时 {s['timeouts']}，代打 {s['fallback_actions']}，报错 {s['errors']}")
    ref = record.get("reference")
    if ref and ref.get("missing"):
        lines.append(f"自由对战上场版本 {ref['against']} 在当前对手池下没有参照成绩，无法比较")
    elif ref:
        verdict = "高于参照线，可以上场" if ref["above"] else "未确定高于参照线"
        lines.append(f"对照上场版本 {ref['against']}：{ref['games']} 场配对差 {ref['diff']:+.2f} ± {ref['se']:.2f}，"
                     f"90% 下界 {ref['lower90']:+.2f}，{verdict}")
    lines.append(f"排名：{ranking(record)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime", help="freeze src/mahjong_ai into the research runtime store")
    p = sub.add_parser("pool")
    p.add_argument("--model", help="weights SHA-256 under the research model store (omit for a pool of rules only)")
    p.add_argument("--name", default="pool-model")
    p.add_argument("--runtime", help="frozen runtime directory (default: the current pool's, else the research experiment's)")
    p.add_argument("--replace", action="store_true", help="start a new record even though the current pool has attempts")
    p.add_argument("--rules", help="comma-separated rule members: research candidate ids or package directories")
    dr = sub.add_parser("drain", help="hold new attempts back while the pool is replaced (maintainer)")
    dr.add_argument("state", choices=("on", "off"))
    dr.add_argument("why", nargs="?", default="维护者正在更换固定对手池")
    rb = sub.add_parser("rebase", help="replay earlier versions on the current pool's development set")
    rb.add_argument("attempts", nargs="+")
    rb.add_argument("--host", default="local", choices=tuple(EVAL_WORKERS))
    sub.add_parser("rebase-finish", help="record a line's replayed versions; the highest is its best").add_argument(
        "line", choices=("rule", "rl"))
    ref = sub.add_parser("reference", help="score a research candidate or a pool model as a reference line")
    ref.add_argument("candidate")
    ref.add_argument("--host", default="local", choices=tuple(EVAL_WORKERS))
    ref.add_argument("--weights", help="score an exported model by its weights hash under this name")
    sub.add_parser("seal", help="accept the current evaluation conditions (maintainer)")
    imp = sub.add_parser("implement", help="register an equivalent, faster implementation of a rule member of the pool")
    imp.add_argument("member")
    imp.add_argument("package")
    args = parser.parse_args()
    if args.command == "runtime":
        print(freeze_runtime())
        return
    if args.command == "pool":
        rules = [x.strip() for x in args.rules.split(",")] if args.rules else None
        print(json.dumps(freeze_pool(args.model, args.name, args.runtime, args.replace, rules)["id"]))
        return
    if args.command == "drain":
        set_drain(args.why if args.state == "on" else None)
        print(f"排空：{args.state}")
        return
    if args.command == "rebase":
        try:
            print(rebase(args.attempts, args.host))
        except Refused as exc:
            raise SystemExit(str(exc))
        return
    if args.command == "rebase-finish":
        rows = rebase_finish(args.line)
        print("\n".join(f"{r['id']} {r['score']:+.2f}（旧池 {r['rebased']['old_score']:+.2f}）" for r in rows))
        return
    if args.command == "seal":
        print(seal())
        return
    if args.command == "implement":
        try:
            summary = implement(args.member, args.package)
        except Refused as exc:
            raise SystemExit(str(exc))
        print(f"{args.member}：{summary['hands']} 局新牌局的 {summary['decisions']} 个决策逐个一致，"
              f"用时为原来的 {summary['time_ratio']:.0%}；已登记，确认后运行 seal")
        return
    pool = load_pool()
    if args.weights:
        spec = {"family": "model", "version": args.candidate, "parameters": {"weights": args.weights}}
    elif args.candidate in pool["members"]:
        spec = {k: v for k, v in playing(pool)[args.candidate].items()
                if k not in ("action_seconds", "response_seconds")}
    else:
        s = research_state()
        spec = dict(s["candidates"][args.candidate]["spec"], runtime_path=s["candidate_runtime_path"])
    try:
        record = evaluate("reference", spec, args.candidate, f"ref-{args.candidate}", host=args.host)
    except Refused as exc:
        raise SystemExit(str(exc))
    print(report(record))


if __name__ == "__main__":
    main()
