"""Frozen, balanced schedules; selection use completed evidence only."""

import random
from math import sqrt
from statistics import variance
from itertools import combinations
from collections import Counter

from mahjong_ai.domain.settlement import score_ranks


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
    return metric_uncertainty(group_means)


def metric_uncertainty(group_means, prefix=""):
    """同一口径的聚类标准误；每 4 场（同副牌、同对手的四座轮换）算一个组。"""
    count = len(group_means)
    return {
        f"{prefix}evaluation_groups": count,
        f"{prefix}standard_error": (
            sqrt(variance(group_means) / count) if count >= 2 else None
        ),
    }


def block_means(values_by_block):
    """按 block 聚合逐局读数，返回 (block 编号, 均值) 的升序序列。"""
    return [
        (block, sum(values) / len(values))
        for block, values in sorted(values_by_block.items())
        if values
    ]


def standings(roster, receipts):
    """场口径读数。主口径是桌面分，与平台第一排序键一致；场名次分为次键。

    名次效用（4−名次）作为对照读数保留，不再参与排序。
    """
    rows = {
        p: dict(
            candidate_id=p,
            appearances=0,
            score_appearances=0,
            score_sum=0.0,
            place_sum=0.0,
            utility_sum=0.0,
            hand_appearances=0,
            hand_utility_sum=0.0,
            hand_score_sum=0.0,
            errors=0,
            timeouts=0,
            fallback_actions=0,
        )
        for p in roster
    }
    for receipt in receipts:
        # 逐局读数：每局的热点席位与四座同场，名次按本局增量分排。
        hands = [tuple(delta) for delta in receipt.get("hand_deltas") or ()]
        hand_ranks = [score_ranks(delta) for delta in hands]
        scores = receipt.get("scores")
        for seat, p in enumerate(receipt["players"]):
            if p not in rows:
                continue
            row = rows[p]
            row["appearances"] += 1
            rank = receipt["ranks"][seat]
            if scores is not None:
                row["score_appearances"] += 1
                row["score_sum"] += scores[seat]
            # 场名次分：+3 / +1 / −1 / −3，同分由中位名次自动取平均。
            row["place_sum"] += 5 - 2 * rank
            row["utility_sum"] += 4 - rank
            for delta, ranks in zip(hands, hand_ranks):
                row["hand_appearances"] += 1
                row["hand_utility_sum"] += 4 - ranks[seat]
                row["hand_score_sum"] += delta[seat]
            for key in ("errors", "timeouts", "fallback_actions"):
                row[key] += receipt.get("execution", [{}] * 4)[seat].get(key, 0)
    for row in rows.values():
        # 只有每一场都带分数时才给出场均分，避免把缺分的回执当成 0 分。
        row["mean_score"] = (
            row["score_sum"] / row["score_appearances"]
            if row["score_appearances"] and row["score_appearances"] == row["appearances"]
            else None
        )
        row["mean_place_points"] = (
            row["place_sum"] / row["appearances"] if row["appearances"] else None
        )
        row["mean_utility"] = (
            row["utility_sum"] / row["appearances"] if row["appearances"] else None
        )
        row["hand_mean_utility"] = (
            row["hand_utility_sum"] / row["hand_appearances"]
            if row["hand_appearances"]
            else None
        )
        row["hand_mean_score"] = (
            row["hand_score_sum"] / row["hand_appearances"]
            if row["hand_appearances"]
            else None
        )
    return sorted(rows.values(), key=standing_key)


def standing_key(row):
    """平台排序：总分优先，场名次分次之。缺分的旧回执退回名次效用口径。"""
    if row.get("mean_score") is not None:
        primary, secondary = row["mean_score"], row["mean_place_points"]
    else:
        primary, secondary = row["mean_utility"], row["hand_mean_utility"]
    return (
        -(primary if primary is not None else -1e18),
        -(secondary if secondary is not None else -1e18),
        -row["appearances"],
        row["candidate_id"],
    )
