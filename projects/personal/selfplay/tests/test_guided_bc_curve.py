import pytest

pytest.importorskip("torch")

from guided.bc_curve import curve_summary, validate_records
from guided.evaluation import schedule


def test_curve_pairs_identical_conditions_and_rejects_mismatched_receipts():
    targets = ["model:minute-00", "model:minute-10", "teacher"]
    jobs = schedule(["teacher", "other"], targets, 8, 123, "curve-test")
    scores = dict(zip(targets, [-4, 4, 0]))
    rows = []
    for job in jobs:
        target = job["seats"][job["focal"]]
        value = scores[target]
        settlement = [0] * 4
        settlement[job["focal"]] = value
        settlement[(job["focal"] + 1) % 4] = -value
        rows.append(job | {"policy": target, "score": value, "scores": settlement})
    validate_records(rows, jobs)
    result = curve_summary(rows, targets, "teacher", 8)
    assert result["differences"][targets[1]]["minus_previous"]["mean"] == 8
    assert result["differences"][targets[1]]["minus_teacher"]["mean"] == 4
    with pytest.raises(ValueError, match="receipts"):
        validate_records(rows[:-1], jobs)
    with pytest.raises(ValueError, match="receipts"):
        validate_records([*rows[:-1], rows[0]], jobs)
    with pytest.raises(ValueError, match="settlement"):
        validate_records([rows[0] | {"score": 999}, *rows[1:]], jobs)
    changed = [dict(row) for row in rows]
    for row in changed:
        if row["policy"] == targets[1] and row["group"] == 0:
            row["seed"] += 1
    with pytest.raises(ValueError, match="share wall"):
        curve_summary(changed, targets, "teacher", 8)
