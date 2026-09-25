"""Evaluate one attempt of an iteration line against the fixed opponent pool (ADR-0016).

Every game seats the attempt once and three distinct pool members; a block is
one lineup on one wall with the attempt in each of the four seats. Lineups
cycle through all C(7, 3) = 35 opponent triples. Walls and every policy's seed
depend only on the set's seed and the block number, so the development set is
the same games for every attempt and results do not depend on how the blocks
are split into jobs.

Ladder on the development set: 1024, 2048 then 4096 blocks (4096, 8192, 16384
games). An attempt more than two standard errors below the best on the blocks
played so far stops; one still above the best after 4096 blocks goes to a fresh
confirmation set, where it and the best each play 2048 blocks and the paired
difference must have a one-sided 90% lower bound above zero.

The pool also records the hash of this file: once frozen, an evaluation
refuses to run under changed evaluation code until the user approves resealing.

Games run without sandboxes: the pool's packages are the maintainer's own, so
each match process plays them in-process (candidate_runtime.InlinePolicy) and
several tables at once, and model seats share the host's inference service.
None of that changes a game's outcome, so these are per-host execution
settings (EVAL_WORKERS, EVAL_TABLES), not part of the pool's identity.

Usage (from projects/personal/firstprinciples; handbooks/common.md has the loop):
  python -m harness.evaluate runtime                                    # freeze src/mahjong_ai as a runtime
  python -m harness.evaluate pool --model <weights-sha> --name <name> [--runtime <dir>]  # freeze the pool once
  python -m harness.evaluate seal                                       # user-approved only
  python -m harness.evaluate history rule|rl
  python -m harness.evaluate restore rule|rl
  python -m harness.evaluate rule "<what changed>" --design-start <epoch>
  python -m harness.evaluate model <actor-dir> "<what changed>" --line rl --design-start <epoch> [--host 4070|4090|5070ti]
  python -m harness.evaluate reference <research candidate | model name> [--host local|4070|4090|5070ti]
"""
import argparse
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

# FP_LOCAL / FP_STAGES / FP_CONFIRM run a separate, shortened record for pipeline checks.
LOCAL = Path(os.environ.get("FP_LOCAL", HERE / ".local"))
RESEARCH = REPO / "projects/agent-research"
MODELS = RESEARCH / ".local/research/models"
# Each line evaluates where it already is: rules on this machine, RL on the GPU
# host that trained it (no transfer); an attempt names its host explicitly.
EVAL_HOST = {"rule": "local", "reference": "local", "rl": "4070"}
# Match processes per host and tables (games played at once, one thread each)
# per process. Measured throughput knees with headroom for the desktop: on the
# Mac the inference service is the limit and tables keep the cores busy while
# model seats wait; too many tables add interpreter contention.
EVAL_WORKERS = {"local": 12, "4070": 24, "4090": 20, "5070ti": 16}
EVAL_TABLES = {"local": 3, "4070": 8, "4090": 2, "5070ti": 4}
ISOLATION = "none"  # the pool's packages run in the match process (see module docstring)
POLL_SECONDS = 5  # how often remote job results are checked; each stage ends on a poll
STAGES = tuple(int(x) for x in os.environ.get("FP_STAGES", "1024,2048,4096").split(","))  # blocks; 4 games each
CONFIRM_BLOCKS = int(os.environ.get("FP_CONFIRM", "2048"))
REFERENCE_BLOCKS = 1024  # a reference only needs a score with its error bar
STOP_SE = 2.0
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


def harness_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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


def freeze_pool(model_weights, model_name, runtime=None, replace=False):
    """Fix the seven opponents and the development seed; refuses to change a pool in use.

    ``replace`` starts a new record on purpose (a new runtime): earlier attempts
    stay in attempts.jsonl under the old pool id and drop out of history.
    """
    path = LOCAL / "pool.json"
    s = research_state()
    runtime = runtime or s["candidate_runtime_path"]
    members = {}
    for cid in RULE_OPPONENTS:
        members[cid] = with_timeouts(dict(s["candidates"][cid]["spec"], runtime_path=runtime))
    if not (MODELS / model_weights).is_dir():
        raise SystemExit(f"model weights {model_weights} are not under {MODELS}")
    members[model_name] = with_timeouts(
        {"family": "model", "version": model_name, "parameters": {"weights": model_weights}}
    )
    pool = {"members": members, "runtime_path": runtime, "config": s["config"]["game"],
            "resources": s["config"]["resources"], "timeouts": TIMEOUTS}
    pool["id"] = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()[:16]
    pool["dev_seed"] = int(pool["id"], 16)
    pool["harness"] = harness_hash()
    if not replace and path.exists() and read_json(path)["id"] != pool["id"] and attempts(read_json(path)["id"]):
        raise SystemExit("the current pool already has attempts; a new pool starts a new record (--replace)")
    atomic_json(path, pool)
    return pool


def load_pool():
    return read_json(LOCAL / "pool.json")


def seal():
    """Record the current evaluation code as the pool's; only on the user's explicit approval."""
    pool = load_pool()
    pool["harness"] = harness_hash()
    atomic_json(LOCAL / "pool.json", pool)
    return pool["harness"]


def check_harness(pool):
    if pool.get("harness") != harness_hash():
        raise SystemExit("评估代码（harness/evaluate.py）与冻结对手池时不一致，拒绝评估。"
                         "撤回改动；只有用户明确同意时才运行 seal 重新冻结。")


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


def best(pool_id, line):
    kept = [r for r in attempts(pool_id, line) if r["status"] == "keep"]
    return kept[-1] if kept else None


def running(line=None):
    """Attempts reserved but not yet recorded (one per GPU host may be under way)."""
    rows = [read_json(p) for p in sorted((LOCAL / "running").glob("*.json"))]
    return [r for r in rows if line is None or r["line"] == line]


def reserve_attempt(line, host, description):
    """Next attempt id of the line, under a lock so concurrent attempts never collide."""
    import fcntl

    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / "running").mkdir(exist_ok=True)
    with (LOCAL / "attempts.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Packages and training receipts are keyed by attempt id, so replacing
        # a pool must not reuse the names of historical attempts.
        taken = [r["id"] for r in attempts(line=line)] + [r["id"] for r in running(line)]
        numbers = [int(i.rsplit("-", 1)[-1]) for i in taken if i.rsplit("-", 1)[-1].isdigit()]
        attempt_id = f"fp-{line}-{max(numbers, default=-1) + 1:03d}"
        atomic_json(LOCAL / "running" / f"{attempt_id}.json",
                    dict(id=attempt_id, line=line, host=host, description=description, started_at=time.time()))
    return attempt_id


def release_attempt(attempt_id):
    (LOCAL / "running" / f"{attempt_id}.json").unlink(missing_ok=True)


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


def restore(line):
    """Put the line's best version back in its source directory (after a failed attempt)."""
    champion = best(load_pool()["id"], line)
    if champion is None:
        return None
    target = SOURCES[line]
    shutil.rmtree(target)
    shutil.copytree(frozen_source(champion), target, ignore=shutil.ignore_patterns("__pycache__"))
    return champion["id"]


def history(line):
    """Every attempt of the line, its best version and the reference, for designing the next one."""
    pool = load_pool()
    rows = attempts(pool["id"], line)
    champion = best(pool["id"], line)
    out = [f"固定对手池 {pool['id']}；{LINE_NAMES[line]}共 {len(rows)} 次尝试，"
           + (f"当前最好 {champion['id']}（{champion['score']:+.2f}）" if champion else "还没有成功的版本")]
    for r in attempts(pool["id"], "reference"):
        out.append(f"参照 {r['id']}：{r['score']:+.2f} ± {r['se']:.2f}（{r['games']} 场）")
    for r in rows:
        score = "—" if r["score"] is None else f"{r['score']:+.2f}±{r['se']:.2f}"
        diff = "" if r.get("diff") is None else f" 差{r['diff']:+.2f}"
        lower = f" 确认下界{r['confirm']['lower90']:+.2f}" if r.get("confirm") else ""
        design = "" if r.get("design_seconds") is None else f" 设计{r['design_seconds'] / 60:.0f}分"
        host = f" @{r['host']}" if r.get("host") not in (None, "local") else ""
        out.append(f"{r['id']} {STATUS[r['status']]}{host} {r['games']}场 {score}{diff}{lower}{design}｜{r['description']}")
    for r in running(line):
        minutes = (time.time() - r["started_at"]) / 60
        out.append(f"{r['id']} 进行中 @{r['host']} {minutes:.0f}分｜{r['description']}")
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
    policies = dict(pool["members"], **{attempt_id: with_timeouts(spec)})
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
    pending = list(roots)
    while pending:
        time.sleep(POLL_SECONDS)
        for root in list(pending):
            path = root / "result.json"
            if not path.exists():
                continue
            value = read_json(path)
            if value.get("status") != "completed":
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


def evaluate(line, spec, description, attempt_id=None, design_seconds=None, host=None):
    """Play the ladder and, on success, the confirmation set; append the record.

    Attempts of a line may run on several hosts at once, so the best version
    is re-read whenever it is compared against: the development blocks are the
    same games for everyone, and a champion that changed during the
    confirmation set replays those fresh walls before the paired test.
    """
    pool = load_pool()
    check_harness(pool)
    attempt_id = attempt_id or reserve_attempt(line, host or EVAL_HOST[line], description)
    # A reference (the version on the platform) plays a fixed set and is never compared.
    reference = line == "reference"
    champion = None if reference else best(pool["id"], line)
    host = host or EVAL_HOST[line]
    started = time.time()
    scores, stats, stage_blocks, verdict, confirm = {}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}, 0, None, None
    for stage in (REFERENCE_BLOCKS,) if reference else STAGES:
        new, s = play(pool, attempt_id, spec, range(stage_blocks, stage), pool["dev_seed"], f"dev{stage}", host)
        scores.update(new)
        for k in stats:
            stats[k] += s[k]
        stage_blocks = stage
        champion = None if reference else best(pool["id"], line)
        if champion is None:
            continue  # the baseline plays the whole ladder
        diff, se = paired(scores, {int(k): v for k, v in champion["dev_blocks"].items()}, range(stage))
        log(f"{attempt_id} dev {stage * 4} games: diff {diff:+.3f} ± {se:.3f} vs {champion['id']}")
        if stage != STAGES[-1] and diff < -STOP_SE * se:
            verdict = "discard"
            break
        if stage == STAGES[-1] and diff <= 0:
            verdict = "discard"
    if champion is None:
        verdict = "reference" if reference else "keep"
    elif verdict is None:
        seed = random.SystemRandom().getrandbits(62)
        mine, _ = play(pool, attempt_id, spec, range(CONFIRM_BLOCKS), seed, "confirm", host)
        while True:
            # The best version replays the same fresh walls under its own id.
            theirs, _ = play(pool, champion["id"], champion["spec"], range(CONFIRM_BLOCKS), seed,
                             f"confirm-for-{attempt_id}", host)
            current = best(pool["id"], line)
            if current["id"] == champion["id"]:
                break
            log(f"{attempt_id}: best version changed to {current['id']} meanwhile; confirming against it")
            champion = current
        diff, se = paired(mine, theirs, range(CONFIRM_BLOCKS))
        confirm = {"seed": seed, "games": CONFIRM_BLOCKS * 4, "diff": diff, "se": se,
                   "lower90": diff - CONFIRM_Z * se, "against": champion["id"]}
        verdict = "keep" if diff - CONFIRM_Z * se > 0 else "unconfirmed"
    score, se = mean_se(scores[k] for k in range(stage_blocks))
    record = dict(
        id=attempt_id, line=line, pool=pool["id"], status=verdict, description=description, host=host,
        started_at=started, finished_at=time.time(), games=stage_blocks * 4, score=score, se=se,
        best_before=champion["score"] if champion else None,
        diff=paired(scores, {int(k): v for k, v in champion["dev_blocks"].items()}, range(stage_blocks))[0] if champion else None,
        confirm=confirm, stats=stats, spec=spec, dev_blocks={str(k): scores[k] for k in sorted(scores)},
        design_seconds=design_seconds,
    )
    with records_path().open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    release_attempt(attempt_id)
    return record


STATUS = {"keep": "成功", "discard": "失败", "unconfirmed": "开发集高但未确认",
          "reference": "参照", "crash": "崩溃"}
LINE_NAMES = {"rule": "规则线", "rl": "RL 线", "reference": "参照"}


def ranking(record):
    """Place among the best versions of both lines and the references (scores are comparable)."""
    pool_id = record["pool"]
    rows = [r for r in attempts(pool_id, "reference")] + [b for b in (best(pool_id, "rule"), best(pool_id, "rl")) if b]
    rows = [r for r in rows if r["id"] != record["id"]] + [record]
    rows.sort(key=lambda r: -r["score"])
    return "；".join(f"{i + 1}. {r['id']} {r['score']:+.2f}" for i, r in enumerate(rows))


def report(record):
    """The fixed per-attempt message to the user (handbooks/common.md)."""
    status = STATUS[record["status"]]
    name = LINE_NAMES[record["line"]]
    number = record["id"].rsplit("-", 1)[-1]
    head = f"尝试 {int(number)} · {name}（{record['id']}）" if number.isdigit() else f"{name}（{record['id']}）"
    if record["score"] is None:
        return "\n".join([f"{head}：{status}", f"改动：{record['description']}",
                          f"原因：{(record.get('error') or '')[:300]}"])
    lines = [f"{head}：{status}（开发集 {record['games']} 场）",
             f"改动：{record['description']}",
             f"得分：{record['score']:+.2f} ± {record['se']:.2f}"
             + (f"（此前最好 {record['best_before']:+.2f}，配对差 {record['diff']:+.2f}）" if record["best_before"] is not None else "（基线）")]
    if record["confirm"]:
        c = record["confirm"]
        lines.append(f"确认集：{c['games']} 场配对差 {c['diff']:+.2f} ± {c['se']:.2f}，90% 下界 {c['lower90']:+.2f}")
    s = record["stats"]
    lines.append(f"超时 {s['timeouts']}，代打 {s['fallback_actions']}，报错 {s['errors']}")
    lines.append(f"排名：{ranking(record)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime", help="freeze src/mahjong_ai into the research runtime store")
    p = sub.add_parser("pool")
    p.add_argument("--model", required=True, help="weights SHA-256 under the research model store")
    p.add_argument("--name", default="pool-model")
    p.add_argument("--runtime", help="frozen runtime directory (default: the research experiment's)")
    p.add_argument("--replace", action="store_true", help="start a new record even though the current pool has attempts")
    r = sub.add_parser("rule")
    r.add_argument("description")
    m = sub.add_parser("model")
    m.add_argument("actor")
    m.add_argument("description")
    m.add_argument("--line", default="rl")
    ref = sub.add_parser("reference", help="score a research candidate or a pool model as a reference line")
    ref.add_argument("candidate")
    ref.add_argument("--host", default="local", choices=tuple(EVAL_WORKERS))
    ref.add_argument("--weights", help="score an exported model by its weights hash under this name")
    for sp in (r, m):
        sp.add_argument("--design-start", type=float, required=True, help="epoch when designing this attempt began")
    m.add_argument("--host", default=None, choices=tuple(h for h in EVAL_WORKERS if h != "local"))
    sub.add_parser("seal", help="accept the current evaluation code (only with the user's approval)")
    for name in ("history", "restore"):
        sub.add_parser(name).add_argument("line", choices=("rule", "rl"))
    args = parser.parse_args()
    if args.command == "runtime":
        print(freeze_runtime())
        return
    if args.command == "pool":
        print(json.dumps(freeze_pool(args.model, args.name, args.runtime, args.replace)["id"]))
        return
    if args.command == "seal":
        print(seal())
        return
    if args.command == "history":
        print(history(args.line))
        return
    if args.command == "restore":
        restored = restore(args.line)
        print(f"已恢复为 {restored}" if restored else "这条线还没有成功的版本，未改动")
        return
    pool = load_pool()
    if args.command == "reference":
        if args.weights:
            spec = {"family": "model", "version": args.candidate, "parameters": {"weights": args.weights}}
        elif args.candidate in pool["members"]:
            spec = {k: v for k, v in pool["members"][args.candidate].items()
                    if k not in ("action_seconds", "response_seconds")}
        else:
            s = research_state()
            spec = dict(s["candidates"][args.candidate]["spec"], runtime_path=s["candidate_runtime_path"])
        record = evaluate("reference", spec, args.candidate, f"ref-{args.candidate}", host=args.host)
        print(report(record))
        return
    line = "rule" if args.command == "rule" else args.line
    host = getattr(args, "host", None) or EVAL_HOST[line]
    attempt_id = reserve_attempt(line, host, args.description)
    design = time.time() - args.design_start
    spec = freeze_rule(attempt_id) if args.command == "rule" else freeze_model(attempt_id, args.actor)
    record = evaluate(line, spec, args.description, attempt_id, design, host=host)
    print(report(record))


if __name__ == "__main__":
    main()
