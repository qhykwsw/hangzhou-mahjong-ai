"""Frozen-policy paired evaluation with independent wall and policy streams."""

import random
import time

import numpy as np

from .artifacts import read_bundle
from .collection import collect
from .statistics import guidance, interval, seed_for


def schedule(panel_ids, targets, games, seed, namespace):
    if games < 8 or games % 4 or not panel_ids:
        raise ValueError("complete four-seat groups and a nonempty panel required")
    jobs = []
    for group in range(games // 4):
        rng = random.Random(seed_for(seed, namespace, "opponents", group))
        others = rng.choices(panel_ids, k=3)
        wall_seed = seed_for(seed, namespace, "wall", group)
        for target in targets:
            table = [target, *others]
            for seat in range(4):
                seats = table[-seat:] + table[:-seat] if seat else list(table)
                jobs.append(
                    {"seats": seats, "seed": wall_seed, "focal": seat, "group": group}
                )
    return jobs


def summarize(records, targets, games):
    output = {}
    conditions = {}
    for target in targets:
        groups = []
        for group in range(games // 4):
            rows = [r for r in records if r["policy"] == target and r["group"] == group]
            if len(rows) != 4 or {r["focal"] for r in rows} != set(range(4)):
                raise ValueError(f"incomplete paired group for {target}: {group}")
            if len({r["seed"] for r in rows}) != 1:
                raise ValueError("four seat rotations must share the actual wall seed")
            for row in rows:
                seats = list(row.get("seats", []))
                if seats:
                    seats[row["focal"]] = "<focal>"
                key = (group, row["focal"])
                condition = (row["seed"], tuple(seats))
                if key in conditions and conditions[key] != condition:
                    raise ValueError(
                        "paired targets must share wall seeds and opponent arrangements"
                    )
                conditions[key] = condition
            groups.append(float(np.mean([r["score"] for r in rows])))
        output[target] = interval(groups) | {"group_scores": groups, "games": games}
    return output


def evaluate(
    actor,
    critic,
    cfg,
    bundle_path,
    panel_path,
    generation,
    deadline,
    *,
    namespace="development",
    targets=None,
    history=None,
):
    bundle, panel = read_bundle(bundle_path), read_bundle(panel_path)
    targets = targets or ["model:current", bundle["teacher"]]
    games = cfg["evaluation_games"]
    jobs = schedule(
        [e["id"] for e in panel["entries"]],
        targets,
        games,
        cfg["seed"],
        f"{namespace}:{generation}",
    )
    command = {
        "kind": "evaluate",
        "iteration": generation,
        "jobs": jobs,
        "deadline": deadline,
        "labels": False,
    }
    _, records, stats = collect(
        actor, critic, cfg, bundle_path, panel_path, command, history
    )
    summary = summarize(records, targets, games)
    result = {
        "generation": generation,
        "namespace": namespace,
        "teacher": bundle["teacher"],
        "panel": panel["id"],
        "results": summary,
        "records": records,
        "resources": stats,
        "time": time.time(),
    }
    if "model:current" in targets and bundle["teacher"] in targets:
        w, difference = guidance(
            summary[bundle["teacher"]]["group_scores"],
            summary["model:current"]["group_scores"],
            cfg["guidance_scale"],
        )
        result.update(w=w, teacher_minus_student=difference)
    return result
