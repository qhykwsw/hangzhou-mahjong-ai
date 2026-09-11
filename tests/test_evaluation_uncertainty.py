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


def test_standings_count_every_hand_not_only_the_settlement():
    from mahjong_ai.evaluation.schedule import standings

    receipts = [
        dict(
            players=["self", "b", "c", "d"],
            ranks=[1, 2, 3, 4],
            hand_deltas=[[6, -2, -2, -2], [6, -2, -2, -2]],
        ),
        dict(
            players=["self", "b", "c", "d"],
            ranks=[4, 1, 2, 3],
            hand_deltas=[[-6, 2, 2, 2], [-6, 2, 2, 2]],
        ),
    ]
    row = standings(["self"], receipts)[0]
    assert row["appearances"] == 2
    assert row["mean_utility"] == 1.5  # 场名次：一次第 1、一次第 4
    assert row["hand_appearances"] == 4  # 两场各 2 局
    assert row["hand_mean_utility"] == 1.5  # 每局不是第 1 就是第 4
    assert row["hand_mean_score"] == 0.0  # +6、+6、−6、−6


def test_paired_evaluation_reports_hand_level_readout(tmp_path):
    assignments = []
    for block in range(2):
        for seat in range(4):
            index = len(assignments)
            assignments.append(
                dict(
                    index=index,
                    block=block,
                    focal=seat,
                    players=["self", "a", "b", "c"],
                )
            )
            deltas = [[3, -1, -1, -1], [-3, 1, 1, 1]]
            ranks = [2, 2, 2, 2]
            ranks[seat] = 1 if block == 0 else 4
            atomic_json(
                tmp_path / "matches" / f"{index:06d}.json",
                {
                    "players": ["self", "a", "b", "c"],
                    "ranks": ranks,
                    "hand_deltas": deltas,
                },
            )
    focal = play(
        {"assignments": assignments, "policies": {"self": {}}, "candidate": "self"},
        tmp_path,
    )["focal"]
    # 主口径（场）：两个轮换组各 3.0 与 0.0；辅助口径（局）：每局第 1 或第 4。
    assert focal["appearances"] == 8
    assert focal["mean_utility"] == focal["standard_error"] == 1.5
    assert focal["hand_appearances"] == 16
    assert focal["hand_mean_utility"] == 1.5
    assert focal["hand_standard_error"] == 0.0  # 两组局均分相同，无组间波动
    assert focal["hand_mean_score"] == 0.0
    assert focal["score_standard_error"] == 0.0
    assert focal["group_means"] == [3.0, 0.0]
    assert focal["hand_group_means"] == [1.5, 1.5]
    assert focal["score_group_means"] == [0.0, 0.0]
    assert focal["group_blocks"] == [0, 1]


def test_competition_ranks_by_field_metric_and_keeps_hand_readout():
    def row(cid, utility, hand_utility):
        return dict(
            candidate_id=cid,
            appearances=4,
            utility_sum=utility * 4,
            hand_appearances=32,
            hand_utility_sum=hand_utility * 32,
            hand_score_sum=0.0,
            hand_mean_utility=hand_utility,
            hand_mean_score=0.0,
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
            for bid in ("steady", "variable")
        },
    }
    state = {
        "jobs": {
            str(i): dict(
                scope="competition",
                cycle=1,
                status="completed",
                summary=[row("steady", 2.0, 1.0), row("variable", 1.0, 2.0)],
            )
            for i in range(2)
        }
    }
    table = Research.competition_table(state, cycle)
    # 场口径下 steady 更高；逐局口径只作辅助，不改变排名。
    assert [r["branch"] for r in table] == ["steady", "variable"]
    assert table[0]["mean_utility"] == 2.0
    assert table[0]["hand_mean_utility"] == 1.0
    assert table[1]["hand_mean_utility"] == 2.0
    assert table[0]["evaluation_groups"] == 2
    assert table[0]["standard_error"] == 0.0
    assert table[0]["hand_standard_error"] == 0.0


def test_merge_focal_aligns_blocks_by_id_and_pairs_candidate_with_baseline():
    from mahjong_ai.research.service import Research, paired_readout

    def shard(blocks, field, hands):
        return dict(
            candidate_id="c",
            appearances=len(blocks),
            utility_sum=sum(field),
            hand_appearances=len(blocks) * 8,
            hand_utility_sum=sum(hands) * 8,
            hand_score_sum=0.0,
            errors=0,
            timeouts=0,
            fallback_actions=0,
            group_blocks=blocks,
            group_means=field,
            hand_group_means=hands,
            score_group_means=[0.0] * len(blocks),
        )

    merged = Research.merge_focal(
        [shard([2, 3], [1.0, 3.0], [1.0, 3.0]), shard([0, 1], [0.0, 2.0], [0.0, 2.0])]
    )
    assert merged["group_blocks"] == [0, 1, 2, 3]
    assert merged["hand_group_means"] == [0.0, 2.0, 1.0, 3.0]
    assert merged["hand_mean_utility"] == merged["mean_utility"] == 1.5
    baseline = dict(
        shard([0, 1, 2, 3], [0.0] * 4, [0.0] * 4),
        field_group_means=[0.0] * 4,
        hand_group_means=[0.0] * 4,
    )
    paired = paired_readout(baseline, merged)
    assert paired["blocks"] == 4
    assert paired["delta"] == 1.5
    assert paired["standard_error"] == pytest.approx(math.sqrt(5 / 3) / 2)
    # 主口径是场：把两套读数调开，配对差跟随场口径而不是逐局口径。
    lookup = {"field_group_means": [0.0] * 4, "hand_group_means": [3.0] * 4}
    baseline.update(lookup)
    merged.update({"field_group_means": [1.0] * 4, "hand_group_means": [3.0] * 4})
    assert paired_readout(baseline, merged)["delta"] == 1.0
    assert paired_readout(baseline, merged, metric="hand")["delta"] == 0.0
