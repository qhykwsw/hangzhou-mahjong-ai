"""Five, eight, ten or eleven frozen policies in balanced four-player tables."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import time

from . import bootstrap
from .run import atomic_json
from .statistics import interval, seed_for


def schedule(entrants, blocks, seed):
    count = len(entrants)
    if count not in (5, 8, 10, 11) or len(set(entrants)) != count or blocks < 2:
        raise ValueError(
            "five, eight, ten or eleven distinct entrants and at least two blocks required"
        )
    # Each partition has one table containing the first entrant and its complement.
    # Cycling over all 35 partitions covers every four-policy lineup equally (+/-1).
    partitions = [(entrants[0], *others) for others in combinations(entrants[1:], 3)]
    if count == 5:
        # All five four-player lineups per block; each entrant sits at four tables.
        patterns = [list(combinations(range(5), 4))]
    if count in (10, 11):
        remaining = set(combinations(range(count), 4))
        patterns, short = [], []
        while remaining:
            first = min(remaining)
            orbit = sorted(
                {
                    tuple(sorted((i + shift) % count for i in first))
                    for shift in range(count)
                }
            )
            remaining.difference_update(orbit)
            (patterns if len(orbit) == count else short).append(orbit)
        if count == 10:
            # Twenty size-10 orbits plus two size-5 orbits. Combine the latter:
            # every block now has ten tables and every entrant appears four times.
            assert len(patterns) == 20 and [len(o) for o in short] == [5, 5]
            patterns.append(short[0] + short[1])
        else:
            # 11 is prime: thirty full orbits of eleven tables, four appearances each.
            assert len(patterns) == 30 and not short
    rng = random.Random(seed_for(seed, "brawl-lineups"))
    jobs = []
    for block in range(blocks):
        if count == 8:
            if block % len(partitions) == 0:
                rng.shuffle(partitions)
            first = partitions[block % len(partitions)]
            tables = [first, tuple(e for e in entrants if e not in first)]
        else:
            if block % len(patterns) == 0:
                rng.shuffle(patterns)
                permutation = rng.sample(entrants, len(entrants))
            tables = [
                tuple(permutation[i] for i in table)
                for table in patterns[block % len(patterns)]
            ]
        wall = seed_for(seed, f"{count}-policy-brawl-wall-v2", block)
        for table in tables:
            seats = list(table)
            rng.shuffle(seats)
            for rot in range(4):
                rotated = seats[rot:] + seats[:rot]
                jobs.append(dict(seats=rotated, seed=wall, focal=0, group=block))
    return jobs


def validate_records(records, jobs, entrants):
    def key(row):
        return row["group"], row["seed"], tuple(row["seats"]), row["focal"]

    if Counter(map(key, records)) != Counter(map(key, jobs)):
        raise ValueError("incomplete, duplicated, or different game schedule")
    for row in records:
        if len(row["scores"]) != 4 or sum(row["scores"]) != 0:
            raise ValueError("invalid zero-sum settlement")
    games_per_block, per_seat = {5: (20, 4), 8: (8, 1), 10: (40, 4), 11: (44, 4)}[
        len(entrants)
    ]
    for block in sorted({r["group"] for r in records}):
        rows = [r for r in records if r["group"] == block]
        counts = Counter((e, seat) for r in rows for seat, e in enumerate(r["seats"]))
        if len(rows) != games_per_block or any(
            counts[e, seat] != per_seat for e in entrants for seat in range(4)
        ):
            raise ValueError("each block must balance every entrant across every seat")


def summarize(records, entrants):
    appearances_per_block = 4 if len(entrants) == 8 else 16
    blocks = sorted({r["group"] for r in records})
    groups = {e: [] for e in entrants}
    wins = {a: {b: [0.0, 0] for b in entrants if b != a} for a in entrants}
    for block in blocks:
        rows = [r for r in records if r["group"] == block]
        scores = {e: [] for e in entrants}
        for r in rows:
            for seat, a in enumerate(r["seats"]):
                scores[a].append(r["scores"][seat])
                for other, b in enumerate(r["seats"]):
                    if seat != other:
                        delta = r["scores"][seat] - r["scores"][other]
                        wins[a][b][0] += (
                            1.0 if delta > 0 else (0.5 if delta == 0 else 0.0)
                        )
                        wins[a][b][1] += 1
        for e in entrants:
            if len(scores[e]) != appearances_per_block:
                raise ValueError("partial block cannot be included in standings")
            groups[e].append(sum(scores[e]) / appearances_per_block)
    result = {
        e: interval(groups[e])
        | {
            "appearances": len(blocks) * appearances_per_block,
            "group_scores": groups[e],
        }
        for e in entrants
    }
    differences = {
        a: {
            b: interval([x - y for x, y in zip(groups[a], groups[b], strict=True)])
            for b in entrants
            if b != a
        }
        for a in entrants
    }
    return {
        "results": result,
        "ranking": sorted(entrants, key=lambda e: result[e]["mean"], reverse=True),
        "paired_differences": differences,
        "same_table_score_win_rates": {
            a: {
                b: {"win_rate_ties_half": value / count, "comparisons": count}
                for b, (value, count) in row.items()
                if count
            }
            for a, row in wins.items()
        },
    }


def render(plan, summary, stage, completed_blocks):
    count = len(plan["models"]) + len(plan["rules"])
    lines = [
        f"# {count} 策略大乱斗",
        "",
        f"状态：{stage}；已完成 {completed_blocks}/{plan['blocks']} 个完整配对组。",
        "",
        f"每组 {8 if count == 8 else 4 * count} 场，阵容与四座轮换均衡，共享牌墙。每支目标出场 {plan.get('appearances_per_entrant', 4096)} 次，每场 16 局。",
        "",
        "| 策略 | 出场数 | 场均桌面分 | 95% 区间 |",
        "|---|---:|---:|---|",
    ]
    if summary:
        for e in summary["ranking"]:
            r = summary["results"][e]
            lines.append(
                f"| {e} | {r['appearances']} | {r['mean']:.3f} | [{r['lower']:.3f}, {r['upper']:.3f}] |"
            )
    lines += [
        "",
        "未完成时排名仅供观察，不按中途成绩提前停止或扩样。区间按完整配对组计算；多项排名比较未做同时置信校正。",
        "所有模型仅推理、不更新参数。此次混合阵容棋力与固定规则面板评估分别记录。",
    ]
    return "\n".join(lines) + "\n"


def run(root, smoke=False):
    import torch
    from .artifacts import read_bundle
    from .collection import collect
    from .config import configuration
    from .deployment import load_actor
    from .networks import Critic

    plan = json.loads((root / "plan.json").read_text())
    code_id = bootstrap.verify_source()
    if code_id != plan["source_id"]:
        raise ValueError("unexpected frozen source")
    output = root / ("smoke" if smoke else "tournament")
    output.mkdir(exist_ok=False)
    state = {"stage": "loading", "source_id": code_id, "started": time.time()}
    atomic_json(output / "status.json", state)
    try:
        # An optional second bundle (e.g. an older roster) seats alongside the
        # first; workers accept both only when their frozen runtimes match.
        panel_path = plan.get("panel", plan["bundle"])
        bundle, panel = read_bundle(plan["bundle"]), read_bundle(panel_path)
        if panel["runtime_hash"] != bundle["runtime_hash"]:
            raise ValueError("both rule rosters must share one frozen runtime")
        rules = [entry["id"] for entry in bundle["entries"]]
        if panel_path != plan["bundle"]:
            rules += [entry["id"] for entry in panel["entries"]]
        if rules != plan["rules"]:
            raise ValueError("frozen rule roster changed")
        torch.set_num_threads(2)
        models = {}
        for name, artifact in plan["models"].items():
            actor, meta = load_actor(artifact["path"], "cuda")
            if meta["source_checkpoint_sha256"] != artifact["checkpoint_sha256"]:
                raise ValueError("not the selected final checkpoint")
            if not meta["config"]["lookahead"]:
                raise ValueError("this tournament requires common lookahead encoding")
            models[name] = actor
        entrants = list(models) + rules
        blocks = 4 if smoke else plan["blocks"]
        jobs = schedule(entrants, blocks, plan["seed"] + (1 if smoke else 0))
        atomic_json(output / "schedule.json", jobs)
        from .resources import evaluation_resources

        resources = evaluation_resources()
        resources.update(plan.get("execution", {}))
        cfg = configuration(
            workers=resources["workers"], tables=resources["tables"], rounds=16
        )
        cfg["inference_batch_limit"] = resources["inference_batch_limit"]
        atomic_json(output / "execution.json", resources)
        critic = Critic().to("cuda").eval()
        actor = next(iter(models.values()))
        deadline = time.time() + (900 if smoke else plan["timeout_seconds"])
        records = []
        state.update(
            stage="running",
            deadline=deadline,
            total_games=len(jobs),
            completed_blocks=0,
        )
        atomic_json(output / "status.json", state)
        chunk_size = 4 if smoke else plan["chunk_blocks"]
        for first in range(0, blocks, chunk_size):
            chunk = [
                j for j in jobs if first <= j["group"] < min(blocks, first + chunk_size)
            ]
            _, finished, resources = collect(
                actor,
                critic,
                cfg,
                plan["bundle"],
                panel_path,
                {
                    "kind": "evaluate",
                    "iteration": first,
                    "jobs": chunk,
                    "deadline": deadline,
                    "labels": False,
                },
                models,
            )
            validate_records(finished, chunk, entrants)
            atomic_json(
                output / f"chunk-{first:04d}.json",
                {"records": finished, "resources": resources},
            )
            records.extend(finished)
            summary = summarize(records, entrants)
            completed = min(blocks, first + chunk_size)
            stage = "complete" if completed == blocks else "running"
            atomic_json(
                output / "results.json",
                summary | {"completed_blocks": completed, "source_id": code_id},
            )
            state.update(
                stage=stage,
                completed_blocks=completed,
                completed_games=len(records),
                updated=time.time(),
            )
            atomic_json(output / "status.json", state)
            (output / "REPORT.md").write_text(render(plan, summary, stage, completed))
            print(json.dumps(state), flush=True)
        if smoke:
            atomic_json(output / "passed.json", state)
    except BaseException as exc:
        state.update(stage="failed", error=str(exc), updated=time.time())
        atomic_json(output / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not args.smoke and not (root / "smoke/passed.json").is_file():
        raise ValueError("complete the mixed-model smoke test first")
    run(root, args.smoke)


if __name__ == "__main__":
    main()
