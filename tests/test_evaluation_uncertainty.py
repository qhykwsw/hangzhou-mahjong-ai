import math

import pytest

from mahjong_ai.evaluation.schedule import group_uncertainty
from mahjong_ai.execution.jobs import play
from mahjong_ai.persistence.artifacts import atomic_json


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


def test_standings_rank_by_table_score_then_place_points():
    """主口径是桌面分；分数相同才看名次分，缺分的旧回执退回名次效用。"""
    from mahjong_ai.evaluation.schedule import standings

    receipts = [
        # big 赢得少但赢得大，small 赢得频繁但赢得小：名次效用相反于桌面分。
        dict(players=["big", "small", "c", "d"], ranks=[1, 2, 3, 4],
             scores=[180, 10, -80, -110]),
        dict(players=["big", "small", "c", "d"], ranks=[3, 1, 2, 4],
             scores=[-20, 40, 10, -30]),
        dict(players=["big", "small", "c", "d"], ranks=[3, 1, 2, 4],
             scores=[-20, 40, 10, -30]),
    ]
    rows = standings(["big", "small"], receipts)
    assert [r["candidate_id"] for r in rows] == ["big", "small"]
    big, small = rows
    assert big["mean_score"] == pytest.approx(140 / 3)
    assert small["mean_score"] == pytest.approx(30.0)
    # 名次效用反过来，但不再决定排序。
    assert small["mean_utility"] > big["mean_utility"]
    # 名次分：big 一次第 1（+3）两次第 3（−1）；small 一次第 2（+1）两次第 1（+3）。
    assert big["mean_place_points"] == pytest.approx(1 / 3)
    assert small["mean_place_points"] == pytest.approx(7 / 3)

    tied = [
        dict(players=["a", "b", "c", "d"], ranks=[1, 2, 3, 4], scores=[30, 10, -10, -30]),
        dict(players=["a", "b", "c", "d"], ranks=[2, 1, 3, 4], scores=[10, 30, -10, -30]),
        dict(players=["a", "b", "c", "d"], ranks=[1, 2, 3, 4], scores=[30, 10, -10, -30]),
        dict(players=["a", "b", "c", "d"], ranks=[2, 1, 3, 4], scores=[10, 30, -10, -30]),
    ]
    a, b = standings(["a", "b"], tied)
    assert a["mean_score"] == b["mean_score"] == 20.0
    assert a["mean_place_points"] == b["mean_place_points"]

    # 旧回执没有分数：主口径退回名次效用，排序仍然可用。
    legacy = [dict(players=["x", "y", "c", "d"], ranks=[4, 1, 2, 3])]
    rows = standings(["x", "y"], legacy)
    assert [r["candidate_id"] for r in rows] == ["y", "x"]
    assert all(r["mean_score"] is None for r in rows)


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


def test_play_preserves_block_readouts_for_chunked_competition_jobs(tmp_path):
    assignments = []
    for block, rank in enumerate((1, 4)):
        for seat in range(4):
            index = len(assignments)
            assignments.append(dict(index=index, block=block, players=["same"] * 4))
            atomic_json(
                tmp_path / "matches" / f"{index:06d}.json",
                {"players": ["same"] * 4, "ranks": [rank] * 4},
            )
    result = play({"assignments": assignments, "policies": {"same": {}}}, tmp_path)
    row = result["summary"][0]
    assert row["appearances"] == 32
    assert row["group_means"] == [3.0, 0.0]


