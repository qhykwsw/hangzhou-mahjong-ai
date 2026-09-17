"""Promotion from complete paired evidence, conditional on the frozen opponent pool."""

from functools import lru_cache
from math import isfinite


SELECTION_RULE = "paired_ci95_lower_gt_zero"


@lru_cache(maxsize=128)
def critical_value(groups):
    from scipy.stats import t

    return float(t.ppf(0.975, groups - 1))


def paired_interval(delta, standard_error, groups):
    """Two-sided 95% Student-t interval, with one observation per rotation group."""
    lower = upper = None
    if (
        groups >= 2
        and standard_error is not None
        and standard_error >= 0
        and isfinite(standard_error)
        and isfinite(delta)
    ):
        margin = critical_value(groups) * standard_error
        lower, upper = delta - margin, delta + margin
    return dict(confidence_level=0.95, confidence_lower=lower, confidence_upper=upper)


def promotion_decision(reference, candidate, expected_games):
    """Keep the incumbent unless the full clean comparison has a positive lower bound."""
    rows = (reference, candidate)
    reason = "eligible"
    paired = (candidate or {}).get("paired") or {}
    groups = expected_games // 4
    if not all(r and r.get("appearances") == expected_games for r in rows):
        reason = "incomplete_evaluation"
    elif any(
        r.get(k, 0) for r in rows for k in ("errors", "timeouts", "fallback_actions")
    ):
        reason = "execution_errors"
    elif (
        groups < 2
        or paired.get("blocks") != groups
        or any(
            len(r.get("group_blocks") or []) != groups
            or len(set(r.get("group_blocks") or [])) != groups
            or len(r.get("field_group_means") or []) != groups
            for r in rows
        )
        or set(reference["group_blocks"]) != set(candidate["group_blocks"])
    ):
        reason = "incomplete_pairing"
    else:
        lower = paired.get("confidence_lower")
        if lower is None or not isfinite(lower) or lower <= 0:
            reason = "insufficient_evidence"
    return dict(eligible=reason == "eligible", reason=reason)
