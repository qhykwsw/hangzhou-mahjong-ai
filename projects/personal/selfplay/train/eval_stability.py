"""How many games does an evaluation need before the ranking stops moving?

    PYTHONPATH=src python3 projects/personal/selfplay/train/eval_stability.py --seeds 11 12 13 14 15
    PYTHONPATH=src python3 projects/personal/selfplay/train/eval_stability.py --analyze

Two evaluation modes are measured: `fixed`, one policy against the frozen
opponent set (used to compare a candidate with its incumbent), and `brawl`, all
entrants in one balanced schedule.  One long run per seed is enough for every
size: games are generated in blocks of four and a block's tables depend only on
the seed and the block index, so the first N/4 blocks of a long run are exactly
the games a run of N games with that seed would have played.  Repeats are the
independent seeds.
"""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUNS = PROJECT / "runs"
DEFAULT_OUT = RUNS / "stability"
SWEEP = RUNS / "transformer-main" / "sweep"
FIXED_POLICIES = [
    RUNS / "transformer-main" / "export",
    SWEEP / "exports" / "ckpt-022-it751",
    SWEEP / "exports" / "ckpt-019-it658",
    SWEEP / "exports" / "ckpt-005-it239",
    SWEEP / "exports" / "best-it45-it45",
]
BRAWL_POLICIES = [
    RUNS / "transformer-main" / "export",
    SWEEP / "exports" / "ckpt-022-it751",
    SWEEP / "exports" / "ckpt-019-it658",
    SWEEP / "exports" / "best-it45-it45",
]


def log(message):
    print(f"[stability {time.strftime('%H:%M:%S')}] {message}", flush=True)


def label(spec):
    path = Path(spec)
    return "final" if path.name == "export" else path.name


def run(extra, output, entries, workers):
    if not output.exists():
        cmd = [sys.executable, str(HERE / "evaluate.py"), *extra, "--entries", str(entries),
               "--output", str(output)]
        if workers:
            cmd += ["--workers", str(workers)]
        started = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        log(f"{output.name} done in {time.time() - started:.0f}s")
    return json.loads(output.read_text())


def collect(out_root, seeds, games, brawl_games, entries, workers):
    for seed in seeds:
        for policy in FIXED_POLICIES:
            run(["--policy", str(policy), "--games", str(games), "--seed", str(seed)],
                out_root / f"fixed-{label(policy)}-s{seed}.json", entries, workers)
        run(["--policy", *[str(p) for p in BRAWL_POLICIES], "--brawl", "auto", "builtin",
             "--games", str(brawl_games), "--seed", str(seed)],
            out_root / f"brawl-s{seed}.json", entries, workers)


def block_scores(records, spec):
    """spec's mean score per four-game block."""
    blocks = {}
    for r in records:
        if spec in r["seats"]:
            blocks.setdefault(r["seed"] // 4, []).append(r["scores"][r["seats"].index(spec)])
    return {b: float(np.mean(v)) for b, v in blocks.items()}


def load(out_root, seeds):
    """{mode: {seed: {name: {block: mean score}}}}"""
    data = {"fixed": {}, "brawl": {}}
    for seed in seeds:
        fixed = {}
        for policy in FIXED_POLICIES:
            path = out_root / f"fixed-{label(policy)}-s{seed}.json"
            if not path.exists():
                fixed = None
                break
            d = json.loads(path.read_text())
            spec = d["summary"]["results"][0]["policy"]
            fixed[label(policy)] = block_scores(d["records"], spec)
        if fixed:
            data["fixed"][seed] = fixed
        path = out_root / f"brawl-s{seed}.json"
        if path.exists():
            d = json.loads(path.read_text())
            data["brawl"][seed] = {label(r["policy"]): block_scores(d["records"], r["policy"])
                                   for r in d["summary"]["results"]}
    return data


def means_at(runs, blocks):
    """{name: mean score over the first `blocks` blocks} per repeat."""
    out = {}
    for name, per_block in runs.items():
        vals = [v for b, v in per_block.items() if b < blocks]
        if vals:
            out[name] = float(np.mean(vals))
    return out


def kendall(a, b, names):
    concordant = discordant = 0
    for x, y in itertools.combinations(names, 2):
        s = (a[x] - a[y]) * (b[x] - b[y])
        if s > 0:
            concordant += 1
        elif s < 0:
            discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def analyse(mode, runs_by_seed, sizes, per_entrant):
    seeds = sorted(runs_by_seed)
    names = sorted(set.intersection(*[set(runs_by_seed[s]) for s in seeds]))
    max_blocks = max(max(r) for s in seeds for r in [runs_by_seed[s][n] for n in names]) + 1
    truth = {n: float(np.mean([means_at(runs_by_seed[s], max_blocks)[n] for s in seeds])) for n in names}
    rows = []
    for size in sizes:
        blocks = size // 4
        if blocks > max_blocks:
            continue
        repeats = [means_at(runs_by_seed[s], blocks) for s in seeds]
        taus = [kendall(a, b, names) for a, b in itertools.combinations(repeats, 2)]
        identical = np.mean([kendall(a, b, names) == 1.0 for a, b in itertools.combinations(repeats, 2)])
        se = float(np.median([np.std([r[n] for r in repeats], ddof=1) / np.sqrt(1) for n in names]))
        # agreement with the pooled ranking, per pair, grouped by true gap
        pairs, se_diffs = [], []
        for x, y in itertools.combinations(names, 2):
            gap = truth[x] - truth[y]
            diffs = [r[x] - r[y] for r in repeats]
            agree = np.mean([np.sign(d) == np.sign(gap) for d in diffs])
            pairs.append((abs(gap), agree))
            se_diffs.append(np.std(diffs, ddof=1))
        se_diff = float(np.median(se_diffs))
        rows.append({
            "games": size,
            "per_entrant": round(per_entrant(size)),
            "se": se,
            "se_diff": se_diff,
            "detectable": 1.65 * se_diff,
            "tau": float(np.mean(taus)),
            "identical": float(identical),
            "pairs": pairs,
        })
    return names, truth, rows


def report(mode, names, truth, rows, out_root):
    lines = [f"## {mode}", "",
             "参与比较的策略（括号内是全部重复合并后的均值）：",
             "  " + "，".join(f"{n} ({truth[n]:+.2f})" for n in sorted(truth, key=lambda k: -truth[k])), "",
             "| 总场数 | 每策略场数 | 单策略标准误 | 两策略差的标准误 | 95% 判对所需差距 | 平均 Kendall tau | 排名完全一致 | 差距 ≥3 分判对率 | 差距 <3 分判对率 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        big = [a for gap, a in r["pairs"] if gap >= 3]
        small = [a for gap, a in r["pairs"] if gap < 3]
        lines.append(
            f"| {r['games']} | {r['per_entrant']} | ±{r['se']:.2f} | ±{r['se_diff']:.2f} | {r['detectable']:.1f} 分 | "
            f"{r['tau']:+.3f} | {r['identical']:.0%} | "
            f"{np.mean(big):.0%}" + (f" ({len(big)} 对)" if big else "") + " | "
            + (f"{np.mean(small):.0%} ({len(small)} 对)" if small else "—") + " |"
        )
    lines.append("")
    const = float(np.median([r["se_diff"] * np.sqrt(r["per_entrant"]) for r in rows]))
    seats_factor = rows[-1]["games"] / rows[-1]["per_entrant"]
    lines += [f"拟合：两策略差的标准误 ≈ {const:.0f} / √(每策略场数)。按 95% 单次判对（1.65 倍标准误）反解：", "",
              "| 想判出的差距 | 每策略场数 | 总场数 |", "|---|---|---|"]
    for gap in (1, 2, 3, 5):
        need = (1.65 * const / gap) ** 2
        lines.append(f"| {gap} 分 | {round(need):,} | {round(need * seats_factor):,} |")
    lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 12, 13, 14, 15])
    p.add_argument("--games", type=int, default=2048, help="fixed-opponent games per policy per seed")
    p.add_argument("--brawl-games", type=int, default=8192, help="brawl games per seed")
    p.add_argument("--entries", default=str(SWEEP / "opponents.json"))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--workers", type=int)
    p.add_argument("--analyze", action="store_true", help="only analyse what is already on disk")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if not args.analyze:
        collect(out_root, args.seeds, args.games, args.brawl_games, Path(args.entries), args.workers)

    data = load(out_root, args.seeds)
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    lines = ["# 评估规模与排名稳定性", "",
             f"重复 = 独立种子 {sorted(data['fixed']) or sorted(data['brawl'])}；"
             "每个规模的结果取长评估的前若干块，等价于该规模单独评估。", ""]
    for mode, per_entrant in (("fixed", lambda n: n), ("brawl", lambda n: n / 3)):
        runs = data[mode]
        if len(runs) < 2:
            continue
        names, truth, rows = analyse(mode, runs, sizes, per_entrant)
        title = "固定对手模式（每个策略单独对 8 个固定对手）" if mode == "fixed" else "大乱斗模式（12 方同一赛程）"
        lines += report(title, names, truth, rows, out_root)
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    log(f"done: {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
