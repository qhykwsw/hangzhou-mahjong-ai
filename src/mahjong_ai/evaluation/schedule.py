"""Frozen, balanced schedules; selection use completed evidence only."""

import random
from math import sqrt
from statistics import variance
from itertools import combinations
from collections import Counter


def schedule(ids, matches, seed):
    ids = sorted(ids)
    if len(ids) < 4 or len(ids) != len(set(ids)) or matches < 4 or matches % 4:
        raise ValueError("four-seat blocks required")
    rng = random.Random(str(seed))
    appearances = Counter()
    together = Counter()
    result = []
    for block in range(matches // 4):
        chosen = []
        for _ in range(4):
            tie = {p: rng.random() for p in ids}
            chosen.append(
                min(
                    (p for p in ids if p not in chosen),
                    key=lambda p: (
                        appearances[p],
                        sum(together[tuple(sorted((p, q)))] for q in chosen),
                        tie[p],
                    ),
                )
            )
        appearances.update(chosen)
        together.update(tuple(sorted(p)) for p in combinations(chosen, 2))
        for seat in range(4):
            result.append(
                dict(
                    index=len(result),
                    block=block,
                    players=chosen[seat:] + chosen[:seat],
                )
            )
    return result


def reference_groups(pool, seed, blocks):
    """Uniform independent draws with replacement, including own historical versions."""
    if not pool:
        raise ValueError("select at least one opponent")
    rng = random.Random(str(seed))
    return [rng.choices(list(pool), k=3) for _ in range(blocks)]


def group_uncertainty(group_means):
    """Estimated SE across four-seat groups, not across correlated games.

    This descriptive sampling estimate is not a confidence bound on rank or
    a selection criterion. The balanced opponent schedule is not IID sampling.
    """
    count = len(group_means)
    return {
        "evaluation_groups": count,
        "standard_error": sqrt(variance(group_means) / count) if count >= 2 else None,
    }


def standings(roster, receipts):
    rows = {
        p: dict(
            candidate_id=p,
            appearances=0,
            utility_sum=0.0,
            errors=0,
            timeouts=0,
            fallback_actions=0,
        )
        for p in roster
    }
    for receipt in receipts:
        for seat, p in enumerate(receipt["players"]):
            if p not in rows:
                continue
            row = rows[p]
            row["appearances"] += 1
            row["utility_sum"] += 4 - receipt["ranks"][seat]
            for key in ("errors", "timeouts", "fallback_actions"):
                row[key] += receipt.get("execution", [{}] * 4)[seat].get(key, 0)
    for row in rows.values():
        row["mean_utility"] = (
            row["utility_sum"] / row["appearances"] if row["appearances"] else None
        )
    return sorted(
        rows.values(),
        key=lambda r: (
            -(r["mean_utility"] if r["mean_utility"] is not None else -1),
            -r["appearances"],
            r["candidate_id"],
        ),
    )
