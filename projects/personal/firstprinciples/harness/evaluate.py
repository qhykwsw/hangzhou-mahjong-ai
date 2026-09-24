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

Usage (from the repository root):
  python -m harness.evaluate pool --model <weights-sha>        # freeze the pool once
  python -m harness.evaluate rule "<what changed>"             # evaluate rule/strategy.py
  python -m harness.evaluate model <actor-dir> "<what changed>" --line rl
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
# Each line evaluates where it already is: rules on this machine, RL on the 4090 that
# trained it (no transfer). The two lines never share a host.
EVAL_HOST = {"rule": "local", "reference": "local", "rl": "4090"}
STAGES = tuple(int(x) for x in os.environ.get("FP_STAGES", "1024,2048,4096").split(","))  # blocks; 4 games each
CONFIRM_BLOCKS = int(os.environ.get("FP_CONFIRM", "2048"))
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


def with_timeouts(spec):
    return dict(spec, action_seconds=TIMEOUTS["discard"], response_seconds=TIMEOUTS["response"])


# -- pool ----------------------------------------------------------------------
def freeze_pool(model_weights, model_name):
    """Fix the seven opponents and the development seed; refuses to change a pool in use."""
    path = LOCAL / "pool.json"
    s = research_state()
    runtime = s["candidate_runtime_path"]
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
    if path.exists() and read_json(path)["id"] != pool["id"] and attempts(read_json(path)["id"]):
        raise SystemExit("the current pool already has attempts; a new pool starts a new record")
    atomic_json(path, pool)
    return pool


def load_pool():
    return read_json(LOCAL / "pool.json")


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


def freeze_rule(attempt_id):
    source = HERE / "rule"
    target = LOCAL / "attempts" / attempt_id / "package"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    return {"family": "code", "version": attempt_id, "parameters": {}, "model_path": None,
            "program": None, "package_path": str(target), "package_hash": tree_hash(target),
            "python_path": str(REPO / ".venv/bin/python")}


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
    return RemoteHost(name, research_state()["config"]["hosts"][name], REPO)


def dispatcher(name):
    if name not in _dispatchers:
        _dispatchers[name] = Dispatcher(remote_host(name), MODELS)
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


def _tasks(pool, attempt_id, spec, blocks, seed, label, workers):
    """Job directories and tasks: one job per worker over contiguous blocks."""
    blocks = list(blocks)
    size = -(-len(blocks) // workers)
    policies = dict(pool["members"], **{attempt_id: with_timeouts(spec)})
    for n in range(0, len(blocks), size):
        root = LOCAL / "jobs" / attempt_id / f"{label}-{n // size:02d}"
        task = dict(kind="match", project=str(REPO), experiment="firstprinciples",
                    runtime_path=pool["runtime_path"], runtime="", config=pool["config"],
                    timeouts=pool["timeouts"], resources=pool["resources"], seed=seed,
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

    workers = research_state()["config"]["match_workers"]
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
    for root, task in _tasks(pool, attempt_id, spec, blocks, seed, label, workers):
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
    workers = research_state()["config"]["hosts"][host]["workers"]
    env = _job_env(pool)
    roots = []
    for root, task in _tasks(pool, attempt_id, spec, blocks, seed, label, workers):
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
        time.sleep(15)
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


def evaluate(line, spec, description, attempt_id=None, design_seconds=None):
    pool = load_pool()
    history = attempts(pool["id"], line)
    attempt_id = attempt_id or f"fp-{line}-{len(history):03d}"
    # A reference (the version on the platform) plays the full ladder and is never compared.
    champion = None if line == "reference" else best(pool["id"], line)
    host = EVAL_HOST[line]
    started = time.time()
    scores, stats, stage_blocks, verdict, confirm = {}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}, 0, None, None
    for stage in STAGES:
        new, s = play(pool, attempt_id, spec, range(stage_blocks, stage), pool["dev_seed"], f"dev{stage}", host)
        scores.update(new)
        for k in stats:
            stats[k] += s[k]
        stage_blocks = stage
        if champion is None:
            continue  # the baseline plays the whole ladder
        diff, se = paired(scores, {int(k): v for k, v in champion["dev_blocks"].items()}, range(stage))
        log(f"{attempt_id} dev {stage * 4} games: diff {diff:+.3f} ± {se:.3f}")
        if stage != STAGES[-1] and diff < -STOP_SE * se:
            verdict = "discard"
            break
        if stage == STAGES[-1] and diff <= 0:
            verdict = "discard"
    if champion is None:
        verdict = "reference" if line == "reference" else "keep"
    elif verdict is None:
        seed = random.SystemRandom().getrandbits(62)
        champion_spec = champion["spec"]
        mine, _ = play(pool, attempt_id, spec, range(CONFIRM_BLOCKS), seed, "confirm", host)
        # The best version replays the same fresh walls under its own id.
        theirs, _ = play(pool, champion["id"], champion_spec, range(CONFIRM_BLOCKS), seed,
                         f"confirm-for-{attempt_id}", host)
        diff, se = paired(mine, theirs, range(CONFIRM_BLOCKS))
        confirm = {"seed": seed, "games": CONFIRM_BLOCKS * 4, "diff": diff, "se": se,
                   "lower90": diff - CONFIRM_Z * se}
        verdict = "keep" if diff - CONFIRM_Z * se > 0 else "unconfirmed"
    score, se = mean_se(scores[k] for k in range(stage_blocks))
    record = dict(
        id=attempt_id, line=line, pool=pool["id"], status=verdict, description=description,
        started_at=started, finished_at=time.time(), games=stage_blocks * 4, score=score, se=se,
        best_before=champion["score"] if champion else None,
        diff=paired(scores, {int(k): v for k, v in champion["dev_blocks"].items()}, range(stage_blocks))[0] if champion else None,
        confirm=confirm, stats=stats, spec=spec, dev_blocks={str(k): scores[k] for k in sorted(scores)},
        design_seconds=design_seconds,
    )
    with records_path().open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def report(record):
    status = {"keep": "成功", "discard": "失败", "unconfirmed": "开发集高但未确认", "reference": "参照"}[record["status"]]
    name = {"rule": "规则线", "rl": "RL 线", "reference": "参照"}[record["line"]]
    lines = [f"【{name} {record['id']}】{status}（开发集 {record['games']} 场）",
             f"改动：{record['description']}",
             f"得分：{record['score']:+.2f} ± {record['se']:.2f}"
             + (f"（此前最好 {record['best_before']:+.2f}，配对差 {record['diff']:+.2f}）" if record["best_before"] is not None else "（基线）")]
    if record["confirm"]:
        c = record["confirm"]
        lines.append(f"确认集：{c['games']} 场配对差 {c['diff']:+.2f} ± {c['se']:.2f}，90% 下界 {c['lower90']:+.2f}")
    s = record["stats"]
    lines.append(f"超时 {s['timeouts']}，代打 {s['fallback_actions']}，报错 {s['errors']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--model", required=True, help="weights SHA-256 under the research model store")
    p.add_argument("--name", default="pool-model")
    r = sub.add_parser("rule")
    r.add_argument("description")
    m = sub.add_parser("model")
    m.add_argument("actor")
    m.add_argument("description")
    m.add_argument("--line", default="rl")
    ref = sub.add_parser("reference", help="score a research candidate as the chart's reference line")
    ref.add_argument("candidate")
    for sp in (r, m):
        sp.add_argument("--design-start", type=float, help="epoch when designing this attempt began")
    args = parser.parse_args()
    if args.command == "pool":
        print(json.dumps(freeze_pool(args.model, args.name)["id"]))
        return
    pool = load_pool()
    if args.command == "reference":
        s = research_state()
        spec = dict(s["candidates"][args.candidate]["spec"], runtime_path=s["candidate_runtime_path"])
        record = evaluate("reference", spec, args.candidate, f"ref-{args.candidate}")
        print(report(record))
        return
    line = "rule" if args.command == "rule" else args.line
    attempt_id = f"fp-{line}-{len(attempts(pool['id'], line)):03d}"
    design = time.time() - args.design_start if args.design_start else None
    spec = freeze_rule(attempt_id) if args.command == "rule" else freeze_model(attempt_id, args.actor)
    record = evaluate(line, spec, args.description, attempt_id, design)
    print(report(record))


if __name__ == "__main__":
    main()
