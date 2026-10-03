"""Openings for the RL line (guided.starts): positions recorded from rule self-play (the maintainer's).

The rule line's best version plays all four seats on the Python engine; at
every decision whose kind matches, the hand so far is kept as an opening
(wall seed and every step from the deal), so a recipe can train from rare
windows instead of waiting for them (ctx.collect(..., starts=ctx.starts(name))).

Kinds:
  kong-window   the seat may kong, holds a white and is ready (tenpai, after its best discard when holding 14)
  kept-fourth   at such a window the seat is made to keep the fourth tile (peng instead of an exposed kong,
                no concealed or added kong, and not discarding that tile); later decisions where it may
                kong that tile are recorded (``hu`` says whether it may also win there)
  fish-window   the seat may win by its own draw while holding a white: win now or play on
                (``baotou`` says whether its hand without the drawn tile is baotou)

    python -m harness.starts make --kinds kong-window,kept-fourth --count 4000 --name kong-v1 [--workers 8]
Writes .local/rl/starts/<name>.json.gz and <name>.meta.json; a recipe ships a set in recipe/starts/.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from . import evaluate as E

STARTS = E.LOCAL / "rl" / "starts"
KINDS = ("kong-window", "kept-fourth", "fish-window")
SELFPLAY = E.REPO / "projects/personal/selfplay"


def make(kinds, count, name, workers, minutes):
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise E.Refused(f"未知的局面类型：{sorted(unknown)}（可选 {', '.join(KINDS)}）")
    pool = E.load_pool()
    best = E.best(pool["id"], "rule")
    package = best["spec"]["package_path"]
    deadline = time.time() + minutes * 60
    base = random.SystemRandom().getrandbits(62)
    tasks = [{"runtime": pool["runtime_path"], "package": package, "kinds": list(kinds), "seed": base + w,
              "count": -(-count // workers), "deadline": deadline} for w in range(workers)]
    started = time.time()
    STARTS.mkdir(parents=True, exist_ok=True)
    work = STARTS / f".{name}-work"
    work.mkdir(exist_ok=True)
    # Workers run under the pool's frozen runtime, never this checkout's mahjong_ai.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
        "PYTHONPATH": os.pathsep.join([str(SELFPLAY), str(SELFPLAY / "src")])}
    running = []
    for w, task in enumerate(tasks):
        E.atomic_json(work / f"task-{w}.json", task)
        log = (work / f"worker-{w}.log").open("w")
        running.append(subprocess.Popen(
            [sys.executable, "-c", "import json,sys;from guided.starts import harvest;"
             "json.dump(harvest(json.load(open(sys.argv[1]))),open(sys.argv[2],'w'))",
             str(work / f"task-{w}.json"), str(work / f"out-{w}.json")],
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log))
    results = []
    for w, process in enumerate(running):
        if process.wait():
            raise RuntimeError(f"局面生成进程 {w} 失败：{(work / f'worker-{w}.log').read_text()[-1500:]}")
        results.append(json.loads((work / f"out-{w}.json").read_text()))
    openings = [o for found, _ in results for o in found]
    sys.path[:0] = [str(SELFPLAY), str(SELFPLAY / "src")]
    from guided.starts import save

    STARTS.mkdir(parents=True, exist_ok=True)
    path = STARTS / f"{name}.json.gz"
    save(path, openings)
    meta = {"name": name, "kinds": dict(Counter(o["kind"] for o in openings)), "hands": sum(p for _, p in results),
            "policy": best["id"], "package_hash": best["spec"].get("package_hash"), "pool": pool["id"],
            "runtime": Path(pool["runtime_path"]).name, "seed": base, "seconds": round(time.time() - started),
            "with_hu": sum(1 for o in openings if o.get("hu")), "baotou": sum(1 for o in openings if o.get("baotou")), "whites": dict(Counter(o["whites"] for o in openings))}
    E.atomic_json(STARTS / f"{name}.meta.json", meta)
    import shutil

    shutil.rmtree(work)
    return path, meta


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("--kinds", required=True, help=",".join(KINDS))
    m.add_argument("--count", type=int, required=True)
    m.add_argument("--name", required=True)
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--minutes", type=float, default=60, help="stop after this long with what was found")
    args = parser.parse_args()
    try:
        path, meta = make(args.kinds.split(","), args.count, args.name, args.workers, args.minutes)
    except E.Refused as exc:
        raise SystemExit(str(exc))
    print(f"{path}：{meta}")


if __name__ == "__main__":
    main()
