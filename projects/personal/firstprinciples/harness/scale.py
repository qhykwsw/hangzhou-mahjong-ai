"""Scale run (ADR-0016, 2026-09-30 amendment): the large Actor's whole training, a chain of RL-line attempts.

The maintainer's run, not the iteration agent's: the fixed recipe in
scale/recipe and its plan (config.json: the segments, the teacher and every
schedule) are frozen when the run begins and never change. Each segment is an
attempt of the RL line on the run's host, trained from the previous segment's
Actor (the first from a random initialisation) and evaluated like any other,
so a segment above the best version is confirmed and may become it. The host
is reserved for the run while it lasts: the RL line's own attempts never
start there.

  python -m harness.scale begin [--host 4090]   # freeze the recipe, start segment 1
  python -m harness.scale chain                  # start each next segment once the last is recorded (detached)
  python -m harness.scale redo                   # after a segment ended without an Actor: train it again
  python -m harness.scale status
"""
import argparse
import random
import shutil
import subprocess
import sys
import time

from . import evaluate as E
from . import rl

RECIPE = E.HERE / "scale" / "recipe"
SCALE = E.LOCAL / "scale"
POLL_SECONDS = 120
PHASES = {"imitation": "模仿", "pg": "RL"}


def run_state():
    path = SCALE / "run.json"
    return E.read_json(path) if path.exists() else None


def reserved_hosts():
    """The host a scale run under way holds (the RL line's own attempts do not start there)."""
    state = run_state()
    return {state["host"]} if state and not state.get("finished_at") else set()


def plan():
    return E.read_json(RECIPE / "config.json")


def describe(index, cfg):
    segments = cfg["segments"]
    segment = segments[index]
    phase = [s for s in segments if s["phase"] == segment["phase"]]
    before = sum(s["hands"] for s in segments[:index] if s["phase"] == segment["phase"])
    order = sum(1 for s in segments[:index] if s["phase"] == segment["phase"]) + 1
    total = sum(s["hands"] for s in phase)
    span = f"{before / 1e4:g}～{(before + segment['hands']) / 1e4:g} 万局"
    if segment["phase"] == "imitation":
        bc = cfg["bc_fraction"] * total
        detail = (f"从随机初始化，前 {bc / 1e4:g} 万局教师代打（BC），之后 DAgger" if before == 0 else "DAgger")
        detail += f"；教师 {cfg['teacher']}，学习率 {cfg['distill_learning_rate']:g} 线性降到 0，每批 {cfg['distill_epochs']} 遍"
    else:
        layers = [n for start, n in cfg["pg_top_layers_schedule"] if before >= start][-1]
        detail = (f"playout 反事实优势 PG，每步 rollout {cfg['chunks_start']}→{cfg['chunks_end']} 个、"
                  f"KL {cfg['step_kl']:g}（{cfg['kl_hold_hands'] / 1e4:g} 万局后线性降到 {cfg['step_kl_end']:g}），尾段平均，"
                  + (f"只训最后 {layers} 层与输出头" if layers else "全部参数"))
    return (f"规模训练第 {index + 1}/{len(segments)} 段（维护者，固定方案）：{PHASES[segment['phase']]} {order}/{len(phase)}，"
            f"{PHASES[segment['phase']]}阶段 {span}；{detail}")


WHY = ("规模训练（ADR-0016 2026-09-30 修订）：大规格从零开始，模仿 500 万局 + RL 1 亿局，"
       "方案与计划开跑前固定、中途不改；每段照常评估，超过最好版本就确认")


def begin(host):
    """Freeze the run's recipe and start its first segment on ``host``."""
    if run_state():
        raise E.Refused(f"规模训练已经开始过（{SCALE / 'run.json'}）。")
    E.check_seal()
    if rl.size(host) != "large":
        raise E.Refused(f"{host} 不是大规格主机。")
    problems = rl.check_recipe(RECIPE)
    if problems:
        raise E.Refused("规模训练方案不符合迭代线约定：\n" + "\n".join(problems))
    cfg = plan()
    budget = rl.BUDGETS["large"]
    for s in cfg["segments"]:
        labeled = s["hands"] if s["phase"] == "imitation" else 0
        if s["hands"] > budget["max_hands"] or labeled > budget["max_labeled"]:
            raise E.Refused(f"一段 {s['hands']:,} 局超出大规格单次预算 {budget}。")
    SCALE.mkdir(parents=True, exist_ok=True)
    E.atomic_json(SCALE / "run.json", {"host": host, "recipe_hash": E.tree_hash(RECIPE), "started_at": time.time(),
                                       "segments": []})
    return start(0)


def start(index):
    """Start segment ``index`` from the previous segment's Actor (the first: from the host's start)."""
    from . import line as L

    state = run_state()
    if E.tree_hash(RECIPE) != state["recipe_hash"]:
        raise E.Refused("规模训练方案在开跑后被改动过：它必须原样用到最后。")
    E.check_seal()
    cfg = plan()
    host = state["host"]
    origin = state["segments"][index - 1] if index else "start"
    if index:
        rl.version(origin)  # evaluated, with its Actor
    with E.attempts_lock():
        if any(s.get("host") == host for s in E.running("rl")):
            raise E.Refused(f"{host} 上还有尝试在跑。")
        attempt_id = E.next_attempt_id("rl")
        now = time.time()
        pool = E.load_pool()
        champion = E.best(pool["id"], "rl")
        record = dict(id=attempt_id, line="rl", host=host, description=describe(index, cfg), why=WHY,
                      base=champion["id"] if champion else None, started_at=now, design_seconds=0.0,
                      stage="training", seed=random.SystemRandom().getrandbits(31),
                      confirm_seed=random.SystemRandom().getrandbits(62),
                      hands=cfg["segments"][index]["hands"], size=rl.size(host), scale={"index": index, "host": host})
        record["from"] = origin
        L.save(record)
        # A segment that crashed is redone in its place.
        state["segments"][index:index + 1] = [attempt_id]
        E.atomic_json(SCALE / "run.json", state)
    job = rl.RL / "attempts" / attempt_id
    if job.exists():
        shutil.rmtree(job)
    job.mkdir(parents=True)
    shutil.copytree(RECIPE, job / "recipe", ignore=shutil.ignore_patterns("__pycache__"))
    E.atomic_json(job / "recipe" / "segment.json", {"index": index})
    L.spawn(attempt_id)
    E.log(f"scale: segment {index + 1} is {attempt_id} on {host} from {origin}")
    return attempt_id


def chain():
    """Start each next segment once the last one is recorded; stop at the end or when one has no Actor."""
    while True:
        state = run_state()
        if state.get("finished_at"):
            return
        last = state["segments"][-1]
        records = [r for r in E.attempts(line="rl") if r["id"] == last]
        if not records:
            time.sleep(POLL_SECONDS)
            continue
        if not ((records[-1].get("spec") or {}).get("parameters") or {}).get("weights"):
            E.log(f"scale: {last} ended {records[-1]['status']} without an Actor; the chain stops")
            return
        done = len(state["segments"])
        if done == len(plan()["segments"]):
            state["finished_at"] = time.time()
            E.atomic_json(SCALE / "run.json", state)
            E.log("scale: finished")
            return
        if E.drain_reason("rl"):
            time.sleep(POLL_SECONDS)
            continue
        try:
            start(done)
        except E.Refused as exc:
            E.log(f"scale: segment {done + 1} not started: {exc}")
            time.sleep(POLL_SECONDS)


def spawn_chain():
    (E.LOCAL / "logs").mkdir(parents=True, exist_ok=True)
    with (E.LOCAL / "logs" / "scale-chain.log").open("ab") as out:
        subprocess.Popen([sys.executable, "-m", "harness.scale", "chain"], cwd=E.HERE, stdin=subprocess.DEVNULL,
                         stdout=out, stderr=out, start_new_session=True)


def status():
    state = run_state()
    if not state:
        return "规模训练还没有开始。"
    cfg = plan()
    records = {r["id"]: r for r in E.attempts(line="rl")}
    running = {s["id"]: s for s in E.running("rl")}
    rows = [f"规模训练 @{state['host']}，{time.strftime('%m-%d %H:%M', time.localtime(state['started_at']))} 开始"
            + ("，已完成" if state.get("finished_at") else "")]
    for index, segment in enumerate(cfg["segments"]):
        attempt_id = state["segments"][index] if index < len(state["segments"]) else None
        head = f"  {index + 1:>2}. {PHASES[segment['phase']]} {segment['hands'] / 1e4:g} 万局"
        if attempt_id in records:
            r = records[attempt_id]
            rows.append(f"{head}｜{attempt_id} {E.STATUS.get(r['status'], r['status'])} {r['score']:+.2f}±{r['se']:.2f}"
                        if r.get("score") is not None else f"{head}｜{attempt_id} {E.STATUS.get(r['status'], r['status'])}")
        elif attempt_id in running:
            rows.append(f"{head}｜{attempt_id} {E.STAGE_NAMES.get(running[attempt_id]['stage'], running[attempt_id]['stage'])}")
        else:
            rows.append(f"{head}｜未开始")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("begin")
    b.add_argument("--host", default="4090", choices=rl.HOSTS)
    sub.add_parser("chain")
    sub.add_parser("redo")
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "begin":
            attempt_id = begin(args.host)
            spawn_chain()
            print(f"规模训练开始：第 1 段是 {attempt_id}，之后每段在上一段记录后自动开跑")
        elif args.command == "chain":
            chain()
        elif args.command == "redo":
            attempt_id = start(len(run_state()["segments"]) - 1)
            spawn_chain()
            print(f"规模训练第 {len(run_state()['segments'])} 段重做：{attempt_id}")
        else:
            print(status())
    except E.Refused as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
