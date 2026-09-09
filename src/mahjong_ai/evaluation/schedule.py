"""Frozen, balanced schedules; selection and protection use completed evidence only."""

import random
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


def reference_groups(branches, own, seed, blocks):
    rng = random.Random(str(seed))
    available = [b for b in branches if b["id"] != own]
    usage = Counter()
    out = []
    if len(available) < 3:
        raise ValueError("three other branches required")
    for _ in range(blocks):
        picked = []
        for _ in range(3):
            tie = {b["id"]: rng.random() for b in available}
            row = min(
                (b for b in available if b not in picked),
                key=lambda b: (
                    sum(x["combination"] == b["combination"] for x in picked),
                    usage[b["id"]],
                    tie[b["id"]],
                ),
            )
            picked.append(row)
            usage[row["id"]] += 1
        out.append([b["representative"] for b in picked])
    return out


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


def eliminated(table, branches, fraction, minimum):
    remaining = Counter(
        b["combination"] for b in branches.values() if b["status"] != "eliminated"
    )
    count = int(len(table) * fraction)
    removed = []
    for row in list(reversed(table))[:count]:
        bid = row["branch"]
        group = branches[bid]["combination"]
        if remaining[group] > minimum and sum(remaining.values()) > 4:
            removed.append(bid)
            remaining[group] -= 1
    return removed
