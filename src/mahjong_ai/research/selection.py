"""Promotion from complete paired evidence against frozen evaluation opponents.

Evaluation runs in stages with doubling games per side. Each look at the data
is a chance of a false promotion, so earlier stages decide only at a stricter
confidence level; the last stage decides at the ordinary one.
"""

from functools import lru_cache
from math import isfinite


SELECTION_RULE = "staged_paired_ci"


@lru_cache(maxsize=256)
def critical_value(groups, confidence=0.95):
    from scipy.stats import t

    return float(t.ppf(1 - (1 - confidence) / 2, groups - 1))


def paired_interval(delta, standard_error, groups, confidence=0.95):
    """Two-sided Student-t interval, with one observation per rotation group."""
    lower = upper = None
    if (
        groups >= 2
        and standard_error is not None
        and standard_error >= 0
        and isfinite(standard_error)
        and isfinite(delta)
    ):
        margin = critical_value(groups, confidence) * standard_error
        lower, upper = delta - margin, delta + margin
    return dict(
        confidence_level=confidence, confidence_lower=lower, confidence_upper=upper
    )


def promotion_decision(reference, candidate, expected_games, final=True):
    """Promote on a positive lower bound; stop early on a negative upper bound.

    The candidate's ``paired`` interval must already use this stage's
    confidence level. ``continue`` asks for the next stage.
    """
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
        lower, upper = paired.get("confidence_lower"), paired.get("confidence_upper")
        if lower is None or not isfinite(lower):
            reason = "insufficient_evidence"
        elif lower > 0:
            reason = "eligible"
        elif final:
            reason = "insufficient_evidence"
        elif upper is not None and isfinite(upper) and upper < 0:
            reason = "rejected_early"
        else:
            reason = "continue"
    return dict(eligible=reason == "eligible", reason=reason)
