"""Equivalent replacement of the rule line's best version (ADR-0016).

A rule attempt started with ``--equivalent`` claims to decide exactly as the
best version does and only to run faster or read simpler. Instead of the
development set it faces fresh games: the best version plays one seat against
the pool's rule members, and the new code gets every call the best version
gets (the same seeds, observations and round results) and must choose the
same action every time. When it does, it would play the development set's
games move for move, so it becomes the best version with the best version's
scores; if the fallback played some of the best version's moves there
(timeouts), the new code plays the development set itself. One different
decision fails it; the first ones are kept for the agent to reproduce.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from . import evaluate as E

HANDS = E.override("FP_EQUIVALENCE_HANDS", 4096)
HANDS_PER_GAME = 16  # one match, as on the platform
KEPT = 5  # mismatching positions each worker keeps

CHECK = r"""
import itertools, json, pickle, sys, time
from pathlib import Path
from guided.artifacts import bind_runtime, load_package
from guided.config import configuration
from guided.environment import External, Game

task = json.loads(Path(sys.argv[1]).read_text())
bind_runtime({"runtime": task["runtime"], "runtime_hash": Path(task["runtime"]).name})
from mahjong_ai.domain.rules import tile_code


def tiles(values):
    return " ".join(tile_code(t) for t in values)


def shown(action):
    return json.dumps(action.as_dict(), ensure_ascii=False) if hasattr(action, "as_dict") else str(action)


class Failed:
    # An act that raised; two raise alike when the exception type is the same.
    def __init__(self, exc):
        self.kind, self.text = type(exc).__name__, f"报错 {type(exc).__name__}: {exc}"

    def __eq__(self, other):
        return isinstance(other, Failed) and other.kind == self.kind

    def __str__(self):
        return self.text


class Abandon(Exception):
    # The best version raised: the evaluation would play its fallback, so this game ends here.
    pass


class Twin:
    # The best version plays; the new code answers every call too.
    def __init__(self):
        self.best, self.new = load_package(task["best"]), load_package(task["new"])
        self.seconds, self.decisions, self.mismatches, self.kept = [0.0, 0.0], 0, 0, []

    def reset(self, seed):
        self.best.reset(seed)
        self.new.reset(seed)

    def finish_round(self, result, seat):
        self.best.finish_round(result, seat)
        self.new.finish_round(result, seat)

    def act(self, obs):
        order = (0, 1) if self.decisions % 2 else (1, 0)  # neither always runs first
        out = [None, None]
        for who in order:
            start = time.perf_counter()
            try:
                out[who] = (self.best, self.new)[who].act(obs)
            except Exception as exc:
                out[who] = Failed(exc)
            self.seconds[who] += time.perf_counter() - start
        self.decisions += 1
        if out[0] != out[1]:
            self.mismatches += 1
            if len(self.kept) < task["kept"]:
                seat = obs.stage.seat
                where = (f"第 {obs.stage.round_no} 局 {obs.phase}，手牌 {tiles(obs.hand)}"
                         + (f"，摸到 {tile_code(obs.drawn_tile)}" if obs.drawn_tile is not None else "")
                         + f"，副露 {len(obs.melds[seat])} 组，牌墙 {obs.wall_remaining}")
                self.kept.append({"obs": obs, "best": shown(out[0]) if isinstance(out[0], Failed) else out[0],
                                  "new": shown(out[1]) if isinstance(out[1], Failed) else out[1],
                                  "text": f"{where}：最好版本 {shown(out[0])}，新代码 {shown(out[1])}"})
        if isinstance(out[0], Failed):
            raise Abandon
        return out[0]


twin = Twin()


class Seats(External):
    def acquire(self, spec):
        return twin if spec == "best" else super().acquire(spec)

    def release(self, borrowed):
        super().release([(s, p) for s, p in borrowed if s != "best"])


opponents = sorted(task["opponents"])
lineups = list(itertools.combinations(opponents, 3))
seats = Seats({"best": {}, **{cid: dict(spec, isolation="none") for cid, spec in task["opponents"].items()}})
for g in task["games"]:
    others = list(lineups[g % len(lineups)])
    focal = (g // len(lineups)) % 4
    game = Game(configuration(rounds=task["hands_per_game"], lookahead=False), None, {},
                others[:focal] + ["best"] + others[focal:], task["seed"] + g, focal=focal, external=seats)
    try:
        game.advance()
    except Abandon:
        pass
    seats.release(game.borrowed)
with open(sys.argv[2], "wb") as f:
    pickle.dump([{k: v for k, v in row.items() if k != "text"} for row in twin.kept], f)
Path(sys.argv[3]).write_text(json.dumps({
    "decisions": twin.decisions, "mismatches": twin.mismatches, "seconds": twin.seconds,
    "examples": [row["text"] for row in twin.kept]}, ensure_ascii=False))
"""


def check(best_package, new_package, seed, root, workers=None, hands=None):
    """Both packages on the same fresh positions; returns the counts and the first mismatches.

    Mismatching positions (Observation, both actions) are pickled to
    root/mismatches.pickle so the agent can call both versions' act on them.
    """
    import pickle

    pool = E.load_pool()
    opponents = {cid: spec for cid, spec in E.playing(pool).items() if spec.get("family") != "model"}
    games = list(range(max(1, (hands or HANDS) // HANDS_PER_GAME)))
    workers = min(workers or E.EVAL_WORKERS["local"], len(games))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selfplay = E.REPO / "projects/personal/selfplay"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
        "PYTHONPATH": os.pathsep.join([str(selfplay), str(selfplay / "src")])}
    running = []
    for w in range(workers):
        task = root / f"task-{w}.json"
        E.atomic_json(task, {"runtime": pool["runtime_path"], "best": str(best_package), "new": str(new_package),
                             "opponents": opponents, "games": games[w::workers], "seed": seed,
                             "hands_per_game": HANDS_PER_GAME, "kept": KEPT})
        with (root / f"worker-{w}.log").open("w") as log:
            running.append((w, subprocess.Popen(
                [sys.executable, "-c", CHECK, str(task), str(root / f"mismatches-{w}.pickle"),
                 str(root / f"result-{w}.json")], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)))
    total = {"hands": len(games) * HANDS_PER_GAME, "decisions": 0, "mismatches": 0,
             "seconds": [0.0, 0.0], "examples": []}
    kept = []
    for w, process in running:
        if process.wait():
            error = (root / f"worker-{w}.log").read_text()[-1500:]
            raise RuntimeError(f"等价检查进程 {w} 失败：{error}")
        value = E.read_json(root / f"result-{w}.json")
        total["decisions"] += value["decisions"]
        total["mismatches"] += value["mismatches"]
        total["seconds"] = [a + b for a, b in zip(total["seconds"], value["seconds"])]
        total["examples"] += value["examples"]
        with open(root / f"mismatches-{w}.pickle", "rb") as f:
            kept += pickle.load(f)
        for name in (f"mismatches-{w}.pickle", f"task-{w}.json", f"result-{w}.json", f"worker-{w}.log"):
            (root / name).unlink()
    total["examples"] = total["examples"][:KEPT]
    if kept:
        with open(root / "mismatches.pickle", "wb") as f:
            pickle.dump(kept[:KEPT], f)
    return total
