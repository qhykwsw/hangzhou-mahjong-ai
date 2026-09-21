"""Paired inference uses games grouped by four seats, never individual hands."""

import hashlib
import math

import numpy as np
from scipy.stats import t


def seed_for(seed, namespace, *parts):
    value = ":".join(map(str, (seed, namespace, *parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 62) - 1)


def interval(group_values):
    values = np.asarray(group_values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite independent groups required")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    half = float(t.ppf(0.975, len(values) - 1) * se)
    return {
        "mean": mean,
        "se": se,
        "half_width": half,
        "lower": mean - half,
        "upper": mean + half,
        "groups": len(values),
    }


def guidance(teacher_groups, student_groups, scale=10.0):
    if (
        not math.isfinite(scale)
        or scale <= 0
        or len(teacher_groups) != len(student_groups)
    ):
        raise ValueError("positive scale and paired groups required")
    result = interval(np.asarray(teacher_groups) - np.asarray(student_groups))
    return min(1.0, max(0.0, result["upper"] / scale)), result


def advantages(rows, lam=0.95):
    """One complete game. gamma=1 and the terminal value is exactly zero."""
    adv = 0.0
    next_value = 0.0
    for row in reversed(rows):
        adv = row["reward"] + next_value - row["value"] + lam * adv
        row["adv"] = adv
        row["ret"] = adv + row["value"]
        next_value = row["value"]
    return rows
