from dataclasses import replace
import numpy as np
import pytest

from mahjong_ai.domain.engine import MahjongEngine
from mahjong_ai.strategies.common import RulePolicy, RLPolicy, TrainingRecipe
from mahjong_ai.domain.types import RoundConfig, StageKind, RANK_OBJECTIVE
from mahjong_ai.domain.settlement import final_ranks, score_ranks
from mahjong_ai.strategies.training import train
from mahjong_ai.strategies.seeds import default_opponents


class WatchingPolicy(RulePolicy):
    def reset(self, seed):
        self.resets = getattr(self, "resets", 0) + 1
        self.observed = []
        self.ends = []

    def act(self, obs):
        self.observed.append(obs)
        return super().act(obs)

    def finish_round(self, result, seat):
        self.ends.append((result, seat))


def test_multigame_rotating_seats_keeps_identity_ledger_and_one_terminal_update():
    policies = [WatchingPolicy() for _ in range(4)]
    cfg = RoundConfig(rounds=3, games_per_round=3)
    result = MahjongEngine(check_invariants=True).run_round(policies, cfg, 431)
    assert (
        result.completed
        and result.rounds_played == 9
        and len(result.metadata["games"]) == 3
    )
    scores = [0] * 4
    for g in result.metadata["games"]:
        for seat, player in enumerate(g["seat_players"]):
            scores[player] += g["scores_by_seat"][seat]
        assert tuple(scores) == g["stage_scores_by_player"]
    assert tuple(scores) == result.total_score_by_seat and sum(scores) == 0
    assert sorted(result.rank_by_seat) == [1, 2, 3, 4]
    assert sum(result.rank_utility(i) for i in range(4)) == 6
    for i, p in enumerate(policies):
        assert p.resets == 1 and len(p.ends) == 1 and p.ends[0][1] == i
        assert {o.stage.seat for o in p.observed} == {(i - g) % 4 for g in range(3)}
        assert all(o.stage.stage_rounds_total == 9 for o in p.observed)
        assert all(
            e.hand_no <= o.stage.round_no for o in p.observed for e in o.public_events
        )
        assert all(
            o.stage.rank_by_seat == score_ranks(o.stage.effective_scores)
            for o in p.observed
        )
        second = [o for o in p.observed if o.stage.stage_round_no == 4]
        assert second
        for o in second:
            assert o.stage.round_no == 1
            assert o.stage.score_by_seat == (0, 0, 0, 0)
            mapping = result.metadata["games"][1]["seat_players"]
            assert o.stage.stage_scores == tuple(
                result.metadata["games"][0]["stage_scores_by_player"][j]
                for j in mapping
            )


def test_official_place_points_and_stable_identity_tiebreak():
    ranks = score_ranks((10, 10, -10, -10))
    assert ranks == (1.5, 1.5, 3.5, 3.5)
    assert tuple(5 - 2 * r for r in ranks) == (2, 2, -2, -2)
    # Place points before whites, then stable identity (not current physical seat).
    assert final_ranks(
        (0, 0, 0, 0),
        (1, 3, 1, 1),
        (9, 0, 9, 0),
        ("z", "b", "a", "c"),
        StageKind.QUALIFIER,
    ) == (3, 1, 2, 4)
    assert final_ranks(
        (10, 10, 0, -20),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        ("a", "b", "c", "d"),
        StageKind.FINAL,
    ) == (1.5, 1.5, 3, 4)


def test_reward_same_rank_independent_of_score_and_incomplete_has_no_reward():
    r = MahjongEngine().run_round(
        [RulePolicy() for _ in range(4)], RoundConfig(rounds=2), 41
    )
    doubled = replace(
        r, total_score_by_seat=tuple(100 * x for x in r.total_score_by_seat)
    )
    assert [r.rank_utility(i) for i in range(4)] == [
        doubled.rank_utility(i) for i in range(4)
    ]
    with pytest.raises(ValueError, match="completed"):
        replace(r, completed=False).rank_utility(0)


def test_final_tiebreak_only_after_all_configured_games_and_no_partial_reward():
    from mahjong_ai.domain.types import Action

    class NeverHu(WatchingPolicy):
        def act(self, obs):
            self.observed.append(obs)
            return next(
                (a for a in obs.legal_actions if a.action == "discard"), Action("pass")
            )

    policies = [NeverHu() for _ in range(4)]
    r = MahjongEngine().run_round(
        policies,
        RoundConfig(rounds=2, games_per_round=3, stage="final", max_tiebreak_rounds=0),
        1,
    )
    assert r.rounds_played == 6 and not r.completed and not r.rank_by_seat
    assert all(len(p.ends) == 1 for p in policies)


def test_multigame_training_resume_matches_and_legacy_critic_is_reset(
    tmp_path, monkeypatch
):
    import mahjong_ai.strategies.training as t

    cfg = RoundConfig(rounds=2, games_per_round=2)
    original = t.MahjongEngine.run_round
    calls = 0

    def interrupted(self, *a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(self, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(t.MahjongEngine, "run_round", interrupted)
        with pytest.raises(KeyboardInterrupt):
            train(
                TrainingRecipe(), default_opponents(), cfg, 3, tmp_path / "resume", 10
            )
    resumed = train(
        TrainingRecipe(), default_opponents(), cfg, 3, tmp_path / "resume", 10
    )
    full = train(TrainingRecipe(), default_opponents(), cfg, 3, tmp_path / "full", 10)
    assert resumed["episodes"] == full["episodes"]
    assert all(
        e["games_completed"] == 2 and e["reward"] == e["rank_utility"]
        for e in full["episodes"]
    )
    with np.load(full["model"]) as a, np.load(resumed["model"]) as b:
        assert str(a["objective"]) == RANK_OBJECTIVE
        assert all(np.array_equal(a[k], b[k]) for k in a)
        legacy = {k: a[k].copy() for k in a if k != "objective"}
    legacy["value"].fill(123)
    np.savez(tmp_path / "legacy.npz", **legacy)
    loaded = RLPolicy(TrainingRecipe(), tmp_path / "legacy.npz", training=True)
    assert not np.any(loaded.params["value"])
    assert np.array_equal(loaded.params["w"], legacy["w"])


def test_legacy_feature_schema_warm_start_preserves_actor_columns(tmp_path):
    cfg = RoundConfig(rounds=1)
    state = train(
        TrainingRecipe(), default_opponents(), cfg, 1, tmp_path / "trained", 7
    )
    with np.load(state["model"]) as src:
        legacy = {k: src[k].copy() for k in ("w", "b", "out", "value", "schema")}
    legacy["schema"] = np.array("public-state-action-v1")
    legacy["w"] = legacy["w"][:-4]
    legacy["value"] = np.concatenate((legacy["value"][:-5], legacy["value"][-1:]))
    np.savez(tmp_path / "v1.npz", **legacy)
    learner = RLPolicy(TrainingRecipe(), tmp_path / "v1.npz", training=True)
    assert np.array_equal(learner.params["w"][:-4], legacy["w"])
    assert not np.any(learner.params["w"][-4:]) and not np.any(learner.params["value"])
    result = MahjongEngine().run_round(
        [learner, *[RulePolicy() for _ in range(3)]], cfg, 8
    )
    assert result.completed and learner.last_update["updated"]


def test_rank_expression_and_network_use_visible_rank_context():
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.strategies.common import feature_matrix

    obs = HandState(RoundConfig(), 8).observation()
    obs = replace(obs, stage=replace(obs.stage, rank_by_seat=(4, 3, 2, 1)))
    assert np.allclose(feature_matrix(obs)[:, -4:], [1, 2 / 3, 1 / 3, 0])
    base = RulePolicy().utilities(obs)
    rule = RulePolicy(program="base + rank + ranks[0]")
    assert np.allclose(rule.utilities(obs), base + 8)
