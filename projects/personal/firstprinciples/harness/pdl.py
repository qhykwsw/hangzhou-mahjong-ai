"""Paired performance-difference measurement (PDL) of two versions on the fixed pool.

measure(a, b) estimates V(b) - V(a), the table score per game of b in a's
place, in the development set's units. b plays games in the development
format (block b: the pool triple b % 35, b at each of the four seats, one
game of the pool's rounds each, with the stage in every observation). At each
of b's decisions where a, reading the same observation, would act otherwise,
the hand is continued from the true state K times for each action, a playing
my seat from there on (harness/pdl_worker.py); the game's sum of these
action-value differences, averaged over games, is V(b) - V(a) by the
performance-difference lemma, with every decision where they agree adding
nothing. Why it is unbiased:

- the remaining wall is a uniform order given the history (every hand
  included), so reshuffling it at the copy keeps the law of the rest of the
  hand; the stable wall only couples the two branches' draws (walls.py);
- the draw luck subtracted from each branch has mean zero given the history;
- a hand's value includes the next hand's start: the winner deals next (the
  dealer again after a draw), and dealing is worth D(h) = A (1 - p^n)/(1 - p)
  for n hands left after this one, with A the dealer's edge in a hand and p
  the draw rate, both read from b's own games (the deal persists through
  draws). This term is the one approximation: it takes b's edge for a's, and
  a constant edge over the game's hands and scores;
- policies decide from the observation alone (true of the pool, the rule
  versions and the model seats), so the copies start from the same policy
  state.

Rule packages run in the worker as the development set runs them (inline,
frozen); model versions through the host's inference service. Decisions have
no time limit here, so a version that times out on the development set is
measured as if it had not. On the Mac (host "local") the workers are local
processes; on a GPU host they run through harness.hostrun, the RL attempts'
measurements on the host that trained them.

  python -m harness.pdl run <a> <b> [--hands 32000] [--host local|4070|4090|5070ti] [--seed S] [--k 16]

Results append to .local/pdl-results.jsonl.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import evaluate as E

HERE = Path(__file__).resolve().parent
HANDS = E.override("FP_PDL_HANDS", 32000)
K = E.override("FP_PDL_K", 16)
GUIDED = E.REPO / "projects/personal/selfplay/guided"


def results_path():
    return E.LOCAL / "pdl-results.jsonl"


def spec_of(version):
    rows = [r for r in E.attempts() if r["id"] == version and r.get("spec")]
    if not rows:
        raise E.Refused(f"没有 {version} 的评估记录")
    return rows[-1]["spec"]


def deal_value(hands, rounds):
    """D(h) per hand number h of the game: the dealer's edge A carried through the draws that keep the deal."""
    as_dealer = [d for dealer, d, _ in hands if dealer]
    others = [d for dealer, d, _ in hands if not dealer]
    edge = sum(as_dealer) / len(as_dealer) - sum(others) / len(others) if as_dealer and others else 0.0
    p = sum(1 for *_, draw in hands if draw) / len(hands) if hands else 0.0
    return edge, p, {h: edge * sum(p ** i for i in range(rounds - h)) for h in range(1, rounds + 1)}


def summarize(results, blocks, rounds):
    """V(b) - V(a) per game with its block-clustered standard error, in total and by decision category."""
    hands = [tuple(x) for r in results for x in r["hands"]]
    edge, p, value = deal_value(hands, rounds)
    per_game = defaultdict(lambda: defaultdict(float))  # (block, focal) -> part -> sum
    for r in results:
        for s in r["snapshots"]:
            vals = s["values"]
            score = sum(v[2] - v[0] for v in vals) / len(vals)
            deal = value[s["hand_no"]] * sum(v[3] - v[1] for v in vals) / len(vals)
            g = per_game[(s["block"], s["focal"])]
            g["total"] += score + deal
            g["deal"] += deal
            g[s["category"]] += score + deal
    parts = sorted({p_ for g in per_game.values() for p_ in g})

    def clustered(part):
        means = [sum(per_game[(b, f)][part] for f in range(4)) / 4 for b in range(blocks)]
        mean = sum(means) / len(means)
        se = math.sqrt(sum((m - mean) ** 2 for m in means) / (len(means) - 1) / len(means)) if len(means) > 1 else 0.0
        return mean, se

    diff, se = clustered("total")
    return dict(diff=diff, se=se, lower90=diff - E.CONFIRM_Z * se,
                parts={p_: dict(zip(("diff", "se"), clustered(p_))) for p_ in parts if p_ != "total"},
                snapshots=sum(len(r["snapshots"]) for r in results), dealer_edge=edge, draw_rate=p)


def task_of(policies, lineups, pool, blocks, w, workers, seed, k, guided):
    return dict(policies=policies, lineups=lineups, config=pool["config"], blocks=list(range(w, blocks, workers)),
                seed=seed, k=k, guided=guided)


def collect(job):
    value = E.read_json(job / "result.json") if (job / "result.json").exists() else {}
    if value.get("status") != "completed":
        raise RuntimeError(f"PDL {job.name}: {value.get('status')} {str(value.get('error'))[-1500:]}")
    return value


def on_mac(root, policies, lineups, pool, blocks, seed, k, weights):
    """The workers as local processes; model seats through the Mac's inference service."""
    env = dict(os.environ, PYTHONPATH=pool["runtime_path"], **E._job_env(pool))
    if weights:
        from mahjong_ai.execution.model_seat import ENCODER_ENV
        from mahjong_ai.execution.model_service import SERVER_ENV, ModelService

        service = E._services.setdefault("local", ModelService(root=E.MODELS, log=E.LOCAL / "model-server.log"))
        service.warm(weights)
        env[SERVER_ENV], env[ENCODER_ENV] = service.url(), str(service.project)
    workers = min(E.EVAL_WORKERS["local"], blocks)
    running = []
    for w in range(workers):
        job = root / f"w{w:02d}"
        job.mkdir(parents=True, exist_ok=True)
        (job / "result.json").unlink(missing_ok=True)
        E.atomic_json(job / "task.json", task_of(policies, lineups, pool, blocks, w, workers, seed, k, str(GUIDED)))
        with (job / "worker.log").open("ab") as out:
            running.append((job, subprocess.Popen([sys.executable, "-B", str(HERE / "pdl_worker.py"), str(job)],
                                                  cwd=E.REPO, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                                  stderr=out, start_new_session=True)))
    for _, process in running:
        process.wait()
    return [collect(job) for job, _ in running]


LAUNCH = """import json, os, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parent
env = dict(os.environ, **json.loads((root / "env.json").read_text()))
running = []
for job in sorted(p for p in root.iterdir() if (p / "task.json").exists()):
    (job / "result.json").unlink(missing_ok=True)
    out = open(job / "worker.log", "ab")
    running.append(subprocess.Popen([sys.executable, "-B", str(root / "pdl_worker.py"), str(job)], cwd=root, env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=out))
for p in running:
    p.wait()
"""


def on_host(host, root, policies, lineups, pool, blocks, seed, k, weights):
    """The workers on a GPU host (harness.hostrun): the pool's runtime and the rule packages mirrored there,
    model seats through the host's inference service, the stable wall from a copy of guided/walls.py."""
    from . import hostrun

    h = E.remote_host(host)
    name = f"pdl-{root.name}"
    where = hostrun.target(h, name)
    stage = root / f"on-{host}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy(HERE / "pdl_worker.py", stage)
    shutil.copy(GUIDED / "walls.py", stage)
    (stage / "launch.py").write_text(LAUNCH)
    E.atomic_json(stage / "env.json", E._job_env(pool))
    packages, mirrored = set(), {}
    for n, spec in policies.items():
        if spec.get("package_path"):
            packages.add(spec["package_path"])
            spec = dict(spec, package_path=h.remote_path(Path(spec["package_path"])))
        mirrored[n] = spec
    workers = min(E.EVAL_WORKERS[host], blocks)
    for w in range(workers):
        (stage / f"w{w:02d}").mkdir()
        E.atomic_json(stage / f"w{w:02d}" / "task.json", task_of(mirrored, lineups, pool, blocks, w, workers, seed, k, where))
    hostrun.push(stage, host, name)
    hostrun.run(name, host, ["python", "launch.py"], sync=sorted(packages), models=sorted(weights))
    alive = f"""import json,os,pathlib
pid=int(pathlib.Path({where!r},'run.pid').read_text())
try:
    os.kill(pid,0); print(json.dumps(True))
except Exception:
    print(json.dumps(False))"""
    while h.python(alive, timeout=120):
        time.sleep(20)
    jobs = [f"w{w:02d}" for w in range(workers)]
    hostrun.fetch(name, host, [f"{j}/result.json" for j in jobs] + [f"{j}/worker.log" for j in jobs], to=stage / "out")
    return [collect(stage / "out" / j) for j in jobs]


def measure(a_id, b_id, *, hands=None, seed=None, host="local", k=None, a_spec=None, b_spec=None, reuse=True,
            field=None):
    """V(b) - V(a), points per game on the fixed pool (see the module docstring); appended to pdl-results.jsonl.

    A version without a record yet (an attempt being judged, harness.evaluate) passes its spec. A
    measurement is a function of the versions, pool, seed, hands, K and field, so one already on record
    with all of these is returned as it is (reuse): an interrupted judgement resumes without measuring again.
    ``field`` ({"id", "members": {name: spec}, "lineups": [[name] * 3, ...]}) seats other opponents in
    place of the pool's, block b with lineup b % len(lineups); the pool's runtime and game format stay.
    """
    import fcntl

    hands, k = hands or HANDS, k or K
    pool = E.load_pool()
    rounds = int(pool["config"]["rounds"])
    blocks = max(1, hands // (4 * rounds))
    seed = random.SystemRandom().getrandbits(62) if seed is None else int(seed)
    specs = {"A": a_spec or spec_of(a_id), "B": b_spec or spec_of(b_id)}
    versions = {a_id: E.version({"id": a_id, "spec": specs["A"]}), b_id: E.version({"id": b_id, "spec": specs["B"]})}
    field_id = field["id"] if field else None
    if reuse and results_path().exists():
        same = (a_id, b_id, pool["id"], seed, blocks * 4, k, versions, field_id)
        for line in results_path().read_text().splitlines():
            r = json.loads(line)
            if (r["a"], r["b"], r["pool"], r["seed"], r["games"], r["k"], r["versions"], r.get("field")) == same:
                return r
    opponents, lineups = (field["members"], field["lineups"]) if field else (E.playing(pool), E.lineups(pool))
    policies = dict(opponents, **{n: E.with_timeouts(s) for n, s in specs.items()})
    root = E.LOCAL / "pdl" / (f"{a_id}--{b_id}--{seed}--{blocks}x{k}" + (f"--{field_id}" if field else ""))
    E.LOCAL.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (E.LOCAL / f"evaluation-{host}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one evaluation per host at a time, as E.play
        weights = {s["parameters"]["weights"] for s in policies.values() if s.get("family") == "model"}
        if host != "local":
            results = on_host(host, root, policies, lineups, pool, blocks, seed, k, weights)
        else:
            results = on_mac(root, policies, lineups, pool, blocks, seed, k, weights)
    entry = dict(a=a_id, b=b_id, pool=pool["id"], field=field_id, host=host, seed=seed, hands=blocks * 4 * rounds, games=blocks * 4, k=k,
                 versions=versions,
                 seconds=time.time() - started, at=time.time(), **summarize(results, blocks, rounds))
    with results_path().open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("a")
    r.add_argument("b")
    r.add_argument("--hands", type=int)
    r.add_argument("--host", default="local")
    r.add_argument("--seed", type=int)
    r.add_argument("--k", type=int)
    args = parser.parse_args()
    try:
        E.check_seal()
        e = measure(args.a, args.b, hands=args.hands, seed=args.seed, host=args.host, k=args.k)
    except E.Refused as exc:
        print(f"拒绝：{exc}")
        raise SystemExit(2)
    print(f"{e['b']} − {e['a']}（PDL，{e['games']} 场，K={e['k']}）：{e['diff']:+.3f} ± {e['se']:.3f} 分/场，"
          f"90% 下界 {e['lower90']:+.3f}；{e['snapshots']} 处分歧，用时 {e['seconds'] / 60:.0f} 分")
    for part, v in sorted(e["parts"].items()):
        print(f"  {part:10s} {v['diff']:+.3f} ± {v['se']:.3f}")
    print(f"  （下一手坐庄：庄家每手优势 {e['dealer_edge']:+.2f} 分、流局率 {e['draw_rate']:.2f}）")


if __name__ == "__main__":
    main()
