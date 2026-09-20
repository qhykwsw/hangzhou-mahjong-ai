"""Offline evaluation of exported actors with clustered confidence intervals.

    python train/evaluate.py --policy runs/resnet/export --games 1024
    python train/evaluate.py --policy runs/mlp/export runs/resnet/export runs/transformer/export --games 512

One evaluated policy: it takes one seat against three fixed opponents (rotating
the seat and the opponent draw per block of four games, same wall seeds).
Several policies: the given exports fill the table (padded with fixed
opponents) and every game is replayed with seat rotation.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
for sub in (HERE, PROJECT / "src", REPO / "src"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from mahjong_ai.domain.engine import MahjongEngine  # noqa: E402
from mahjong_ai.domain.types import RoundConfig  # noqa: E402
from mahjong_ai.evaluation.schedule import schedule  # noqa: E402
from opponents import BUILTIN, OpponentFactory, auto_research_representatives, load_entries  # noqa: E402
from selfplay_core.policy import OnnxPolicy  # noqa: E402

_factory = None
_policies = {}


def _init(entries):
    global _factory
    os.environ["OMP_NUM_THREADS"] = "1"
    _factory = OpponentFactory(entries)


def _policy(spec):
    if spec not in _policies:
        _policies[spec] = OnnxPolicy(spec) if os.path.isdir(spec) else None
    return _policies[spec]


def _play(job):
    """job: (seed, seats) where seats is a list of specs; returns scores per spec."""
    seed, seats, rounds, ycbk = job
    players = []
    for spec in seats:
        pol = _policy(spec) if os.path.isdir(spec) else _factory.build(spec)
        players.append(pol)
    started = time.perf_counter()
    result = MahjongEngine(record_events=False).run_round(
        players, RoundConfig(rounds=rounds, you_cai_bi_kao=ycbk), seed
    )
    return {
        "seed": seed,
        "seats": seats,
        "scores": list(result.total_score_by_seat),
        "ranks": list(result.rank_by_seat),
        "seconds": time.perf_counter() - started,
    }


def _label(spec):
    if not os.path.isdir(spec):
        return spec
    path = Path(spec)
    return f"{path.parent.name}/{path.name}" if path.name == "export" else path.name


def summarize(records, target):
    """Mean score of `target` with a block-clustered standard error (4 rotations = 1 block)."""
    by_block = {}
    for r in records:
        for seat, spec in enumerate(r["seats"]):
            if spec == target:
                by_block.setdefault(r["seed"] // 4, []).append((r["scores"][seat], r["ranks"][seat]))
    scores = np.asarray([s for block in by_block.values() for s, _ in block], dtype=float)
    ranks = np.asarray([k for block in by_block.values() for _, k in block], dtype=float)
    block_means = np.asarray([np.mean([s for s, _ in block]) for block in by_block.values()])
    se = float(block_means.std(ddof=1) / np.sqrt(len(block_means))) if len(block_means) > 1 else float("nan")
    return {
        "policy": target,
        "games": int(len(scores)),
        "mean_score": float(scores.mean()),
        "score_se": se,
        "mean_rank": float(ranks.mean()),
        "place_points": float((5 - 2 * ranks).mean()),
        "first_rate": float((ranks == 1).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", nargs="*", default=[], help="export dirs (policy.onnx + meta.json)")
    p.add_argument("--brawl", nargs="*", metavar="ENTRANT",
                   help="round-robin brawl among --policy dirs plus these fixed ids "
                        "('auto' = every auto-research representative, 'builtin' = the four baselines)")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--ycbk", choices=["on", "off", "mixed"], default="off",
                   help="有财必拷响：默认关闭，与线上锦标赛配置一致")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--no-auto-research", action="store_true")
    p.add_argument("--entries", help="JSON list of opponent entries; keeps the opponent set fixed across runs")
    p.add_argument("--output")
    args = p.parse_args()
    if args.entries:
        entries = load_entries(args.entries)
    else:
        entries = list(BUILTIN)
        if not args.no_auto_research:
            entries += auto_research_representatives(REPO / "projects/agent-research")
    fixed_ids = [e["id"] for e in entries]
    policies = [str(Path(x).resolve()) for x in args.policy]
    rng = random.Random(args.seed)
    jobs = []
    games = args.games - args.games % 4
    entrants = None
    if args.brawl is not None:
        entrants = list(policies)
        for token in args.brawl:
            if token == "auto":
                entrants += [e["id"] for e in entries if e["kind"] == "package"]
            elif token == "builtin":
                entrants += [e["id"] for e in BUILTIN]
            else:
                entrants.append(token)
        if len(entrants) < 4:
            raise SystemExit("a brawl needs at least four entrants")
        for row in schedule(entrants, games, args.seed):
            ycbk = {"on": True, "off": False}.get(args.ycbk, bool(row["block"] % 2))
            jobs.append((row["index"], list(row["players"]), args.rounds, ycbk))
    for block in range(games // 4 if entrants is None else 0):
        if len(policies) == 1:
            table = [policies[0]] + rng.sample(fixed_ids, 3)
        else:
            table = list(policies[:4]) + rng.sample(fixed_ids, max(0, 4 - len(policies)))
        ycbk = {"on": True, "off": False}.get(args.ycbk, bool(block % 2))
        for rot in range(4):
            seats = table[rot:] + table[:rot]
            jobs.append((block * 4 + rot, seats, args.rounds, ycbk))
    started = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(entries,)) as pool:
        records = list(pool.imap_unordered(_play, jobs, chunksize=2))
    elapsed = time.time() - started
    summary = {
        "rounds": args.rounds,
        "ycbk": args.ycbk,
        "games": len(records),
        "elapsed_s": round(elapsed, 1),
        "games_per_s": round(len(records) / elapsed, 2),
        "opponents": fixed_ids if entrants is None else None,
        "results": [summarize(records, target) for target in (entrants or policies)],
    }
    if entrants is not None:
        summary["results"].sort(key=lambda r: -r["mean_score"])
        summary["leaderboard"] = [
            f"{i + 1}. {_label(r['policy'])}: "
            f"{r['mean_score']:+.2f} ± {r['score_se']:.2f}  名次分 {r['place_points']:+.2f}  首名率 {r['first_rate']:.2f}  场数 {r['games']}"
            for i, r in enumerate(summary["results"])
        ]
    if len(policies) == 1 and entrants is None:
        summary["by_opponent"] = {}
        for fid in fixed_ids:
            subset = [r for r in records if fid in r["seats"]]
            if subset:
                summary["by_opponent"][fid] = summarize(subset, policies[0])
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
