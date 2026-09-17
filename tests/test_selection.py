import pytest

from mahjong_ai.research.selection import paired_interval, promotion_decision


def test_paired_interval_uses_two_sided_student_t_and_group_count():
    interval = paired_interval(0.37890625, 0.03343949708851926, 256)
    assert interval["confidence_lower"] == pytest.approx(0.3130534949331149)
    assert interval["confidence_upper"] == pytest.approx(0.4447590050668851)
    # Two groups have one degree of freedom, not eight independent games.
    assert paired_interval(0.3, 0.1, 2)["confidence_lower"] == pytest.approx(
        -0.9706204736
    )
    assert paired_interval(0.3, None, 1)["confidence_lower"] is None
    assert paired_interval(0.0, 0.0, 256)["confidence_lower"] == 0.0


def rows():
    reference = dict(
        appearances=16, group_blocks=[0, 1, 2, 3], field_group_means=[1.0] * 4
    )
    candidate = dict(
        appearances=16,
        group_blocks=[0, 1, 2, 3],
        field_group_means=[1.5] * 4,
        paired={"blocks": 4, **paired_interval(0.5, 0.0, 4)},
    )
    return reference, candidate


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("counter", ["errors", "timeouts", "fallback_actions"])
def test_runtime_failures_on_either_side_block_promotion(side, counter):
    reference, candidate = rows()
    assert promotion_decision(reference, candidate, 16)["eligible"] is True
    (reference, candidate)[side][counter] = 1
    assert promotion_decision(reference, candidate, 16) == {
        "eligible": False,
        "reason": "execution_errors",
    }


@pytest.mark.parametrize(
    "problem",
    [
        "partial",
        "missing_group",
        "duplicate_group",
        "different_group",
        "zero",
        "negative",
        "unavailable",
    ],
)
def test_promotion_requires_complete_pairing_and_strictly_positive_lower_bound(problem):
    reference, candidate = rows()
    if problem == "partial":
        candidate["appearances"] = 12
    elif problem == "missing_group":
        candidate["field_group_means"].pop()
    elif problem == "duplicate_group":
        candidate["group_blocks"] = [0, 1, 2, 2]
    elif problem == "different_group":
        candidate["group_blocks"] = [0, 1, 2, 4]
    else:
        candidate["paired"]["confidence_lower"] = {
            "zero": 0.0,
            "negative": -0.01,
            "unavailable": None,
        }[problem]
    assert promotion_decision(reference, candidate, 16)["eligible"] is False
