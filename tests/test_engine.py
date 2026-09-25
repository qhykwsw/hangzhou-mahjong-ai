from dataclasses import replace

import pytest

from mahjong_ai.domain.engine import HandState, IllegalAction, MahjongEngine
from mahjong_ai.strategies.common import RulePolicy, risk_preference
from mahjong_ai.domain.rules import WHITE, tile_id
from mahjong_ai.domain.types import Action, Meld, RoundConfig, StageKind


def make_hand(text):
    return [tile_id(t) for t in text.split()]


def pass_windows(state):
    while state.phase in ("response_peng", "response_chi"):
        state.step(Action("pass"))


def test_variable_horizon_reproducibility_and_tile_conservation():
    for length in (1, 3, 11):
        engine = MahjongEngine(check_invariants=True)
        config = RoundConfig(rounds=length)
        a = engine.run_round([RulePolicy() for _ in range(4)], config, 431)
        b = engine.run_round([RulePolicy() for _ in range(4)], config, 431)
        assert a == b
        assert a.completed and a.rounds_played == length
        assert sum(a.total_score_by_seat) == 0
        assert a.total_score_by_seat == tuple(
            sum(h["delta"][i] for h in a.metadata["hands"]) for i in range(4)
        )


def test_live_observation_hides_other_draws_and_concealed_kongs():
    state = HandState(RoundConfig(rounds=4), 17)
    private_tile = state.hands[1][0]
    state._event("draw", 1, private_tile, private=1)
    state.melds[1].append(Meld("gang", (private_tile,) * 4, concealed=True))
    obs = state.observation()
    assert obs.stage.seat == 0
    assert obs.public_events[-1].tile is None
    assert not obs.melds[1][-1].tiles
    assert all(
        not e.tiles for e in obs.public_events if e.kind == "deal" and e.seat != 0
    )
    assert len(obs.hand) == 14
    assert not hasattr(obs, "wall") and not hasattr(obs, "seed")


def test_white_discard_forces_catch_play_and_only_drawn_discard():
    state = HandState(RoundConfig(), 12)
    state.hands[0][-1] = WHITE
    state.drawn = WHITE
    state.step(Action("discard", WHITE))
    assert state.turn == 1 and state.phase == "draw"
    assert state.observation().catch_play
    discards = [a for a in state.legal_actions() if a.action == "discard"]
    assert discards == [Action("discard", state.drawn)]
    assert not any(a.action in ("chi", "peng") for a in state.legal_actions())
    for expected in (2, 3, 0):
        state.step(Action("discard", state.drawn))
        pass_windows(state)
        assert state.turn == expected
    assert state.catch_owner == 0
    assert not state.observation().catch_restricted
    state.step(
        next(
            a
            for a in state.legal_actions()
            if a.action == "discard" and a.tile != WHITE
        )
    )
    assert state.catch_owner is None


def test_chi_follows_all_peng_passes_and_cannot_hu_without_draw():
    state = HandState(RoundConfig(), 27)
    state.hands[0] = make_hand("3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 东 东 南 南")
    state.hands[1] = make_hand("1w 2w 4w 5w 6w 7w 8w 9w 1b 2b 3b 东 东")
    state.drawn = tile_id("3w")
    state.step(Action("discard", tile_id("3w")))
    assert state.phase == "response_peng"
    for _ in range(3):
        state.step(Action("pass"))
    assert state.phase == "response_chi" and state.turn == 1
    state.step(Action("chi", tile_id("3w"), (tile_id("1w"), tile_id("2w"))))
    assert state.phase == "draw" and state.drawn is None
    assert Action("hu") not in state.legal_actions()
    assert len(state.melds[1]) == 1


def test_kong_wall_reserve_and_white_restrictions():
    state = HandState(RoundConfig(), 8)
    state.hands[0] = make_hand("1w 1w 1w 1w 2w 3w 4w 5w 6w 7w 东 东 白 白")
    state.drawn = tile_id("1w")
    action = Action("gang", tile_id("1w"), kind="concealed")
    assert action in state.legal_actions()
    state.wall = state.wall[:20]
    assert action not in state.legal_actions()
    with pytest.raises(IllegalAction):
        state.step(action)


def test_failed_execution_budget_does_not_invent_final_winner():
    result = MahjongEngine().run_round(
        [RulePolicy() for _ in range(4)],
        RoundConfig(rounds=3, max_actions_per_hand=1),
        6,
    )
    assert not result.completed
    assert result.winner_seat is None and result.top_two_seats == ()


def test_final_ties_are_not_seat_order_wins():
    # Force draws using a real engine with a legitimate always-discard strategy.
    class NeverHu(RulePolicy):
        def act(self, obs):
            choices = [a for a in obs.legal_actions if a.action == "discard"]
            return choices[0] if choices else Action("pass")

    config = RoundConfig(
        rounds=1, stage=StageKind.FINAL, allow_tiebreak=True, max_tiebreak_rounds=0
    )
    r = MahjongEngine().run_round([NeverHu() for _ in range(4)], config, 4)
    assert not r.completed and r.winner_seat is None
    assert len(set(r.total_score_by_seat)) == 1


def test_risk_changes_with_gap_and_progress_for_any_length():
    obs = HandState(RoundConfig(rounds=16, stage=StageKind.FINAL), 10).observation()
    early = replace(
        obs, stage=replace(obs.stage, score_by_seat=(30, 0, 0, 0), round_no=1)
    )
    leading = replace(early, stage=replace(early.stage, round_no=16))
    trailing = replace(
        leading, stage=replace(leading.stage, score_by_seat=(-30, 0, 0, 0))
    )
    assert risk_preference(early) == 0
    assert risk_preference(leading) < 0 < risk_preference(trailing)
    same_fraction = replace(
        leading, stage=replace(leading.stage, round_no=32, rounds_total=32)
    )
    assert risk_preference(same_fraction) == risk_preference(leading)


def test_v25_third_chi_forbidden_but_peng_does_not_consume_chi_slots():
    state = HandState(RoundConfig(), 1)
    state.phase, state.turn, state.discarded = "response_chi", 1, tile_id("3w")
    state.hands[1] = make_hand("1w 2w 东 东")
    state.melds[1] = [
        Meld("chi", (3, 4, 5)),
        Meld("chi", (12, 13, 14)),
        Meld("peng", (27, 27, 27)),
    ]
    assert not any(a.action == "chi" for a in state.legal_actions())
    state.melds[1][1] = Meld("peng", (28, 28, 28))
    assert any(a.action == "chi" for a in state.legal_actions())


def test_legal_stage_scores_take_precedence_without_resetting_at_new_game():
    obs = HandState(RoundConfig(rounds=4, stage=StageKind.FINAL), 1).observation()
    ctx = replace(
        obs.stage,
        score_by_seat=(0, 0, 0, 0),
        stage_scores=(40, 0, -20, -20),
        stage_round_no=20,
        stage_rounds_total=20,
    )
    assert risk_preference(replace(obs, stage=ctx)) < 0
    assert ctx.own_score == 40 and ctx.progress == 1


def test_piao_preserves_chain_then_ordinary_discard_breaks_it():
    state = HandState(RoundConfig(), 1)
    # Four triplets and a white wait can hu with every drawn tile.
    state.hands[0] = make_hand("1w 1w 1w 2w 2w 2w 3w 3w 3w 4w 4w 4w 白 白")
    state.drawn = WHITE
    state.step(Action("discard", WHITE))
    assert state.chain[0] == state.piao[0] == 1
    for _ in range(3):
        state.step(Action("discard", state.drawn))
        pass_windows(state)
    assert state.turn == 0
    state.step(Action("discard", tile_id("1w")))
    assert state.chain[0] == state.piao[0] == 0


def test_randomized_actual_games_conserve_tiles():
    for seed in range(20):
        r = MahjongEngine(record_events=False, check_invariants=True).run_round(
            [RulePolicy(risk_scale=x) for x in (0.2, 0.7, 1.0, 2.0)],
            RoundConfig(rounds=5),
            seed,
        )
        assert r.completed and sum(r.total_score_by_seat) == 0


def test_winner_becomes_next_dealer_matching_official_eight_hand_replay_contract():
    result = MahjongEngine().run_round(
        [RulePolicy() for _ in range(4)], RoundConfig(rounds=16), 991
    )
    hands = result.metadata["hands"]
    assert any(h["winner"] is not None and h["winner"] != h["dealer"] for h in hands)
    for before, after in zip(hands, hands[1:]):
        assert after["dealer"] == (
            before["dealer"] if before["winner"] is None else before["winner"]
        )
