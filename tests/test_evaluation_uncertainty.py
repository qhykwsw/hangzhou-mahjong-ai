import math

import pytest

from mahjong_ai.evaluation.schedule import group_uncertainty
from mahjong_ai.execution.jobs import play
from mahjong_ai.persistence.artifacts import atomic_json
from mahjong_ai.research.service import Research


def test_group_standard_error_does_not_treat_correlated_games_as_independent():
    # Within each rotation group the outcomes are perfectly correlated.
    grouped = group_uncertainty([0.0, 3.0])
    naive = group_uncertainty([0.0] * 4 + [3.0] * 4)
    assert grouped["evaluation_groups"] == 2
    assert grouped["standard_error"] == 1.5
    assert grouped["standard_error"] > naive["standard_error"]
    assert group_uncertainty([0.0, 3.0, 0.0, 3.0])["standard_error"] == pytest.approx(
        math.sqrt(0.75)
    )


def test_insufficient_groups_are_not_reported_as_zero_uncertainty():
    for values in ([], [1.5]):
        assert group_uncertainty(values)["standard_error"] is None
    assert group_uncertainty([1.5, 1.5]) == {
        "evaluation_groups": 2,
        "standard_error": 0.0,
    }


def test_competition_accumulates_blocks_without_changing_mean_ranking():
    def summary(cid, utility):
        return dict(
            candidate_id=cid,
            appearances=4,
            utility_sum=utility * 4,
            errors=0,
            timeouts=0,
            fallback_actions=0,
        )

    cycle = {
        "id": 1,
        "roster": {
            bid: dict(
                candidate_id=bid, combination="test", model="astra", effort="medium"
            )
            for bid in ("variable", "steady")
        },
    }
    state = {
        "jobs": {
            str(i): dict(
                scope="competition",
                cycle=1,
                status="completed",
                summary=[summary("variable", score), summary("steady", 1.75)],
            )
            for i, score in enumerate((0, 3))
        }
    }
    state["jobs"]["pending"] = dict(scope="competition", cycle=1, status="running")
    table = Research.competition_table(state, cycle)
    assert [r["branch"] for r in table] == ["steady", "variable"]
    assert table[0]["standard_error"] == 0.0
    assert table[1]["mean_utility"] == table[1]["standard_error"] == 1.5
    assert all(r["evaluation_groups"] == 2 and r["appearances"] == 8 for r in table)
    state["jobs"]["0"]["summary"][0]["appearances"] = 3
    with pytest.raises(ValueError, match="四座轮换组"):
        Research.competition_table(state, cycle)


def test_resumed_paired_evaluation_uses_focal_seat_and_rotation_groups(tmp_path):
    assignments = []
    for block, rank in enumerate((1, 4)):
        for seat in range(4):
            index = len(assignments)
            assignments.append(
                dict(index=index, block=block, focal=seat, players=["self"] * 4)
            )
            ranks = [2, 2, 2, 2]
            ranks[seat] = rank
            atomic_json(
                tmp_path / "matches" / f"{index:06d}.json",
                {"players": ["self"] * 4, "ranks": ranks},
            )
    result = play(
        {"assignments": assignments, "policies": {"self": {}}, "candidate": "self"},
        tmp_path,
    )
    assert result["focal"]["appearances"] == 8
    assert result["focal"]["evaluation_groups"] == 2
    assert result["focal"]["mean_utility"] == result["focal"]["standard_error"] == 1.5
