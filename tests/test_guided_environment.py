"""Real-engine rollout boundaries and independent, paired evaluation evidence."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from guided import artifacts
from guided.config import configuration
from guided.environment import Game
from guided.evaluation import schedule, summarize
from guided.statistics import advantages, guidance, interval, seed_for
from mahjong_ai.domain.engine import HandState, MahjongEngine, derive_seed
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.strategies.heuristic import HeuristicPolicy
from selfplay_core.actions import action_index


@pytest.fixture
def policies(monkeypatch):
    created = []

    def build(bundle, candidate_id):
        policy = HeuristicPolicy(style="balanced")
        created.append(policy)
        return policy

    monkeypatch.setattr(artifacts, "build_policy", build)
    bundle = {"teacher": "teacher", "entries": [{"id": "teacher"}]}
    return bundle, {"teacher": bundle}, created


def play_game(game, student):
    while True:
        request = game.advance()
        if request is None:
            return
        action = student.act(game.state.observation())
        game.apply((action_index(action), -0.75, 0.0))


def test_complete_16_hand_score_matches_authoritative_engine_and_rollout(policies):
    bundle, entries, created = policies
    cfg = configuration(rounds=16, lookahead=False, gae_lambda=1.0)
    seed = seed_for(137, "complete-game")
    game = Game(
        cfg,
        bundle,
        entries,
        ["model:current", "teacher", "teacher", "teacher"],
        seed,
        focal=0,
        teacher="teacher",
        train=True,
    )
    student = HeuristicPolicy(style="balanced")
    play_game(game, student)
    reference = MahjongEngine(record_events=False).run_round(
        [HeuristicPolicy(style="balanced") for _ in range(4)],
        RoundConfig(rounds=16, allow_tiebreak=False),
        seed,
        policy_seeds=[seed_for(seed, "policy", seat) for seat in range(4)],
    )
    assert game.complete and game.hand_no == reference.rounds_played == 16
    assert tuple(game.scores) == reference.total_score_by_seat
    assert sum(game.scores) == 0
    assert len(created) == len({id(p) for p in created}) == 4
    assert game.rows
    prefix = getattr(game, "predecision_reward", 0.0)
    assert sum(row["reward"] for row in game.rows) + prefix == game.scores[0]
    for row in game.rows:
        assert row["mask"][row["action"]]
        assert row["mask"][row["teacher"]]
        assert row["label_version"] == "teacher"
        assert row["logprob"] == -0.75
    assert game.rows[-1]["ret"] == game.rows[-1]["reward"]
    assert game.rows[0]["ret"] == sum(row["reward"] for row in game.rows)


@pytest.mark.parametrize("public_only", [False, True])
def test_worker_transport_retains_complete_training_rows(
    policies, monkeypatch, public_only
):
    import multiprocessing as mp
    import threading
    import time

    from guided.environment import worker

    bundle, _, _ = policies
    monkeypatch.setattr(artifacts, "read_bundle", lambda _: bundle)
    monkeypatch.setattr(artifacts, "bind_runtime", lambda _: None)
    parent, child = mp.Pipe()
    cfg = configuration(rounds=1, lookahead=False, tables=1, workers=1)
    command = dict(
        kind="collect",
        iteration=1,
        deadline=time.time() + 20,
        samples=1,
        labels=False,
        policy_inputs_only=public_only,
    )
    thread = threading.Thread(
        target=worker, args=(child, cfg, "bundle", "bundle", 0, command)
    )
    thread.start()
    records = []
    try:
        while True:
            assert parent.poll(20), "worker failed to send a request"
            message = parent.recv()
            assert message[0] != "error", message
            if message[0] == "done":
                assert message[1]["unfinished_games"] == 0
                break
            _, requests, completed = message
            records.extend(completed)
            replies = []
            for ident, request in requests:
                row = request["row"]
                assert ("oracle_planes" in row) != public_only
                replies.append(
                    (ident, (int(np.flatnonzero(row["mask"])[0]), -0.5, 0.0))
                )
            parent.send(replies)
        thread.join(5)
        assert not thread.is_alive()
        assert records and records[0]["rows"]
        for row in records[0]["rows"]:
            assert "oracle_planes" in row and "oracle_scalars" in row
            assert row["mask"][row["action"]]
            assert row["value"] == 0 and row["logprob"] == -0.5
        assert sum(row["reward"] for row in records[0]["rows"]) == records[0]["score"]
    finally:
        parent.close()
        thread.join(5)


def test_teacher_label_does_not_replace_student_action_or_behavior_probability(
    policies,
):
    bundle, entries, _ = policies
    game = Game(
        configuration(rounds=1, lookahead=False),
        bundle,
        entries,
        ["model:current", "teacher", "teacher", "teacher"],
        seed_for(14, "on-policy"),
        focal=0,
        teacher="teacher",
        train=True,
    )
    request = game.advance()
    assert request is not None
    row, legal, _ = game.pending
    teacher_label = row["teacher"]
    alternative = next(
        action for action in legal if action_index(action) != teacher_label
    )
    chosen = action_index(alternative)
    assert not game.rows
    game.apply((chosen, -2.125, 3.25))
    assert game.rows[0]["teacher"] == teacher_label
    assert game.rows[0]["action"] == chosen != teacher_label
    assert game.rows[0]["logprob"] == -2.125
    assert game.rows[0]["value"] == 3.25


def test_illegal_teacher_and_student_actions_fail_explicitly(policies, monkeypatch):
    bundle, entries, _ = policies
    from mahjong_ai.domain.types import Action

    game = Game(
        configuration(rounds=1, lookahead=False),
        bundle,
        entries,
        ["model:current", "teacher", "teacher", "teacher"],
        seed_for(14, "on-policy"),
        focal=0,
        teacher="teacher",
        train=True,
    )
    monkeypatch.setattr(game.teacher, "act", lambda obs: Action("invalid"))
    with pytest.raises(ValueError, match="teacher returned an illegal"):
        game.advance()
    monkeypatch.setattr(game.teacher, "act", HeuristicPolicy(style="balanced").act)
    game.advance()
    with pytest.raises(ValueError, match="neural policy returned an illegal"):
        game.apply((9999, 0.0, 0.0))
    assert not game.rows


def test_teacher_outside_the_bundle_is_borrowed_from_the_trial_versions(policies):
    from guided.environment import External

    bundle, entries, _ = policies
    external = External({"frozen-rule": {}})
    lent = HeuristicPolicy(style="balanced")
    external.free["frozen-rule"].append(lent)
    game = Game(
        configuration(rounds=1, lookahead=False),
        bundle,
        entries,
        ["model:current", "teacher", "teacher", "teacher"],
        seed_for(14, "outside-teacher"),
        focal=0,
        teacher="frozen-rule",
        train=True,
        external=external,
    )
    assert game.teacher is lent and ("frozen-rule", lent) in game.borrowed
    assert game.advance() is not None
    row, legal, _ = game.pending
    assert row["label_version"] == "frozen-rule"
    assert row["teacher"] in {action_index(a) for a in legal}
    external.release(game.borrowed)
    assert external.free["frozen-rule"] == [lent]


def test_teacher_can_be_focal_against_three_independent_copies_of_itself(policies):
    bundle, entries, created = policies
    game = Game(
        configuration(rounds=1, lookahead=False),
        bundle,
        entries,
        ["teacher"] * 4,
        seed_for(3, "teacher-self-play"),
        focal=2,
    )
    assert game.advance() is None
    assert game.complete and game.rows == []
    assert len(created) == len({id(policy) for policy in created}) == 4
    assert sum(game.scores) == 0


def test_forced_predecision_settlement_is_separate_from_future_action_rewards(policies):
    bundle, entries, _ = policies
    game = Game(
        configuration(rounds=2, lookahead=False),
        bundle,
        entries,
        ["model:current", "teacher", "teacher", "teacher"],
        seed_for(28, "prefix"),
        focal=0,
        teacher="teacher",
        train=True,
    )
    # A real settled engine snapshot precedes the first recorded student action.
    # Such score belongs in the game ledger, not a fabricated prior transition.
    first = game.state
    players = [HeuristicPolicy(style="balanced") for _ in range(4)]
    while not first.settled:
        first.step(players[first.turn].act(first.observation()))
    prefix = first.delta[0]
    assert prefix != 0, "fixture must exercise a nonzero prefix"
    play_game(game, players[0])
    assert game.complete and game.rows
    assert game.predecision_reward == prefix
    assert sum(row["reward"] for row in game.rows) + prefix == game.scores[0]


def test_schedule_pairs_opponents_wall_and_all_relative_dealer_positions():
    panel = ["rule-a", "rule-b", "rule-c", "rule-d"]
    targets = ["student", "teacher"]
    jobs = schedule(panel, targets, 8, 314, "development:0")
    assert len(jobs) == 16
    for group in range(2):
        block = [job for job in jobs if job["group"] == group]
        assert len({job["seed"] for job in block}) == 1
        seed = block[0]["seed"]
        for target in targets:
            rows = [job for job in block if job["seats"][job["focal"]] == target]
            assert {row["focal"] for row in rows} == set(range(4))
            assert {(row["focal"] - seed % 4) % 4 for row in rows} == set(range(4))
        for focal in range(4):
            pair = [job for job in block if job["focal"] == focal]
            assert [p for s, p in enumerate(pair[0]["seats"]) if s != focal] == [
                p for s, p in enumerate(pair[1]["seats"]) if s != focal
            ]
        # The scheduler seed reaches the real shuffled wall, not just opponent RNG.
        left = HandState(RoundConfig(), derive_seed(seed, "wall", 1), seed % 4)
        right = HandState(RoundConfig(), derive_seed(seed, "wall", 1), seed % 4)
        assert left.wall == right.wall and left.hands == right.hands


@pytest.mark.parametrize("change", ["seed", "namespace"])
def test_independent_namespace_and_seed_change_actual_wall(change):
    first = schedule(["a", "b", "c", "d"], ["student"], 8, 99, "dev:0")[0]
    second = schedule(
        ["a", "b", "c", "d"],
        ["student"],
        8,
        100 if change == "seed" else 99,
        "final:0" if change == "namespace" else "dev:0",
    )[0]
    assert first["seed"] != second["seed"]
    walls = [
        HandState(RoundConfig(), derive_seed(job["seed"], "wall", 1)).wall
        for job in (first, second)
    ]
    assert walls[0] != walls[1]


def receipts():
    return [
        dict(
            policy=target,
            group=group,
            focal=focal,
            seed=seed_for(3, "paired", group),
            score=(10 if target == "teacher" else 5) + group + focal,
        )
        for target in ("student", "teacher")
        for group in range(2)
        for focal in range(4)
    ]


def test_summarize_rejects_missing_or_duplicate_rotation():
    rows = receipts()
    summary = summarize(rows, ["student", "teacher"], 8)
    assert summary["student"]["groups"] == 2
    assert summary["teacher"]["group_scores"] == [11.5, 12.5]
    with pytest.raises(ValueError, match="incomplete"):
        summarize(rows[:-1], ["student", "teacher"], 8)
    rows[-1]["focal"] = rows[-2]["focal"]
    with pytest.raises(ValueError, match="incomplete"):
        summarize(rows, ["student", "teacher"], 8)


def test_summarize_rejects_unpaired_wall_streams_between_targets():
    rows = receipts()
    for row in rows:
        if row["policy"] == "teacher":
            row["seed"] += 1
    with pytest.raises(ValueError, match="paired|seed|wall"):
        summarize(rows, ["student", "teacher"], 8)


def test_summarize_rejects_different_opponent_arrangements():
    rows = receipts()
    for row in rows:
        row["seats"] = ["rule-a", "rule-b", "rule-c"]
        row["seats"].insert(row["focal"], row["policy"])
    rows[-1]["seats"][0] = "different-rule"
    with pytest.raises(ValueError, match="paired|opponent"):
        summarize(rows, ["student", "teacher"], 8)


def test_guidance_is_continuous_clipped_and_uses_upper_confidence_bound():
    assert guidance([0, 0], [1, 1], 10)[0] == 0
    assert guidance([10, 10], [0, 0], 10)[0] == 1
    assert guidance([20, 20], [0, 0], 10)[0] == 1
    values = [guidance([x, x], [0, 0], 10)[0] for x in (0, 0.001, 2, 5, 9.999, 10)]
    assert values == pytest.approx([0, 0.0001, 0.2, 0.5, 0.9999, 1])
    w, difference = guidance([1, -3], [0, 0], 100)
    assert difference["mean"] < 0 < difference["upper"]
    assert w == pytest.approx(difference["upper"] / 100)
    with pytest.raises(ValueError):
        guidance([0, 1], [0], 10)
    with pytest.raises(ValueError):
        interval([1])
    with pytest.raises(ValueError):
        interval([0, np.nan])
    for scale in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            guidance([1, 1], [0, 0], scale)


def test_gae_bootstraps_only_inside_complete_game():
    rows = [
        dict(reward=0.0, value=4.0),
        dict(reward=3.0, value=2.0),
        dict(reward=-1.0, value=5.0),
    ]
    advantages(rows, lam=1)
    assert [row["ret"] for row in rows] == [2, 2, -1]
    assert [row["adv"] for row in rows] == [-2, 0, -6]


def test_worker_imports_do_not_preload_sdk_or_torch():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import guided.environment, guided.collection, guided.evaluation
assert not any(k == 'mahjong_ai' or k.startswith('mahjong_ai.') for k in sys.modules)
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(PROJECT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
