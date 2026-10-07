"""The race luck (guided.walls, cfg["luck_kind"] = "race"): the same on both engines, mean zero, off by default."""

import copy
import random
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame  # noqa: E402  (fastgame first: members register through it)
from guided.config import configuration  # noqa: E402
from guided.environment import Game  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from guided.walls import race_value, train_hand  # noqa: E402
from mahjong_ai import _fast  # noqa: E402
from mahjong_ai.domain.engine import HandState  # noqa: E402
from mahjong_ai.domain.types import RoundConfig  # noqa: E402
from selfplay_core.actions import action_index  # noqa: E402
from test_guided_fastgame import members, members68, play, rows_of, snapshot_of  # noqa: E402,F401

CONFIG = RoundConfig(rounds=16, allow_tiebreak=False)


def test_the_race_luck_is_the_same_on_both_engines(members68):
    """Members play hands side by side on both engines; at every step both carry the same luck and race luck."""
    plans = [fastgame.race_member(m) for m in members68]
    rng = random.Random(61)
    moved = 0
    for case in range(16):
        seed, dealer, hand_no = rng.getrandbits(64), rng.randrange(4), rng.randrange(1, 17)
        scores = tuple(rng.randrange(-50, 50) for _ in range(4))
        state = HandState(CONFIG, seed, dealer, hand_no, scores)
        state.rounds_total = 16
        hand = fastgame.native.Hand(seed, dealer, hand_no, scores, 16)
        learner, seats = rng.randrange(4), [rng.randrange(len(members68)) for _ in range(4)]
        convert_at = 0 if case % 2 == 0 else rng.randrange(1, 30)  # at the deal, or partway (as an opening)
        steps = 0
        while True:
            if steps == convert_at:
                train_hand(state, stable=case % 4 < 2, luck_seat=learner, race=True)
                if case % 4 < 2:
                    hand.stabilize()
                hand.track_luck(learner, True)
            got = hand.snapshot()
            got.pop("log")
            assert got == snapshot_of(state)
            if steps >= convert_at:
                assert hand.luck == state.luck and hand.race_luck == state.race_luck
            if state.settled:
                break
            legal = state.legal_actions()
            if len(legal) > 1 and rng.random() < 0.9:
                index = hand.act(*plans[seats[state.turn]])
                action = next(a for a in legal if action_index(a) == index)
            else:
                action = legal[rng.randrange(len(legal))]
            state.step(action, legal)
            hand.step(action_index(action))
            steps += 1
        moved += state.race_luck != (0.0, 0.0)
    assert moved >= 8


def racing_position(members68, ready=2):
    """A Python hand with race luck at a response window, members having played it: when everybody
    passes, the next seat draws, and it and ``ready`` - 1 or more other seats are ready (it can win on
    that draw, the others lose copies). Deep copies of it are the tests' small positions."""
    rng = random.Random(67)
    for _ in range(200):
        state = HandState(CONFIG, rng.getrandbits(64), rng.randrange(4), 1, (0, 0, 0, 0))
        state.rounds_total = 16
        train_hand(state, luck_seat=rng.randrange(4), race=True)
        while not state.settled:
            legal = state.legal_actions()
            if state.phase.startswith("response_") and len(state.wall) > 30 and state.catch_owner is None:
                counts = [list(_fast.counts(tuple(h))) for h in state.hands]
                waiting = [_fast.shanten_raw(c, len(m)) <= 0 for c, m in zip(counts, state.melds)]
                drawer = (state.discarder + 1) % 4
                if waiting[drawer] and sum(waiting) >= ready:
                    # The learner: another ready seat, so that both its wait and the drawer's count.
                    state.luck_seat = next(p for p in range(4) if waiting[p] and p != drawer)
                    return state
            action = members68[rng.randrange(len(members68))].act(state.observation(legal)) if len(legal) > 1 else legal[0]
            state.step(action, legal)
    raise AssertionError("no racing position found")


def draw_increment(state):
    """Pass the open window(s) until the next draw; the race luck that draw added."""
    before = state.race_luck
    drawn = len(state.wall)
    while len(state.wall) == drawn and not state.settled:
        state.step(next(a for a in state.legal_actions() if a.action == "pass"))
    return state.race_luck[0] - before[0], state.race_luck[1] - before[1]


def test_the_race_luck_of_a_draw_has_mean_zero(members68):
    """One position, its remaining wall reshuffled again and again: the next draw's race luck averages zero,
    exactly so over the tiles the wall holds."""
    base = racing_position(members68)
    # Every tile of the wall as the next draw (the ordinary wall draws from its back): the mean is zero.
    exact = []
    for i in range(len(base.wall)):
        state = copy.deepcopy(base)
        state.wall[i], state.wall[-1] = state.wall[-1], state.wall[i]
        exact.append(draw_increment(state))
    exact = np.array(exact)
    assert exact[:, 0].std() > 0.5 and len(set(map(tuple, exact))) >= 3
    assert np.abs(exact.mean(axis=0)).max() < 1e-9
    # Reshuffles of the remaining wall.
    rng = random.Random(71)
    shuffled = []
    for _ in range(3000):
        state = copy.deepcopy(base)
        rng.shuffle(state.wall)
        shuffled.append(draw_increment(state))
    shuffled = np.array(shuffled)
    for k in range(2):
        se = shuffled[:, k].std() / np.sqrt(len(shuffled))
        assert abs(shuffled[:, k].mean()) < 4 * se


def test_the_race_value_reads_like_its_formula():
    # One seat, hypergeometric over its own draws: n = 28 tiles leave 8 draws, the seat's at positions 1 and 5.
    pay, deal = race_value(28, 0, [(1, 2, 6.0)], learner=1, learner_deals=False)
    first = 2 / 28 + (26 / 28) * 2 / 27
    assert pay == pytest.approx(first * 3.0) and deal == pytest.approx(first)
    # Nobody can win: the dealer keeps the deal.
    assert race_value(40, 2, [], learner=0, learner_deals=True) == (0.0, 1.0)
    # Two seats: the first to draw wins more often with the same copies.
    a, _ = race_value(60, 0, [(0, 3, 3.0), (1, 3, -3.0)], learner=0, learner_deals=False)
    assert a > 0


def test_rows_with_the_race_luck_are_otherwise_the_same(members, monkeypatch):
    """luck_kind "race" only adds its fields; "immediate" (the default) is the game without the key."""
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    base = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "stable_wall": True, "luck": True}
    seats, seed = ["m1", "m3", "model:current", "m4"], seed_for(73, "race-rows")
    played = {}
    for name, extra in [("default", {}), ("immediate", {"luck_kind": "immediate"}), ("race", {"luck_kind": "race"})]:
        games = [cls(base | extra, {"entries": []}, {k: None for k in by_id}, seats, seed, focal=2, train=True)
                 for cls in (Game, fastgame.FastGame)]
        counts = [play(game) for game in games]
        assert counts[0] == counts[1] > 50
        assert rows_of(games[0]) == rows_of(games[1])
        played[name] = games[1]
    assert rows_of(played["default"]) == rows_of(played["immediate"])
    assert not any("luck_race" in key for row in played["default"].rows for key in row)
    race = played["race"].rows
    strip = SimpleNamespace(rows=[{k: v for k, v in row.items() if "luck_race" not in k} for row in race])
    assert rows_of(strip) == rows_of(played["default"]) and played["race"].scores == played["default"].scores
    assert all("luck_race_pay" in row and "luck_race_deal" in row for row in race)
    assert any("end_luck_race_pay" in row for row in race)
    deviated = [row for row in race if row.get("deviated")]
    assert deviated and all("playout_luck_race_pay" in row and "playout_luck_race_deal" in row for row in deviated)
    assert any(row["luck_race_pay"] != 0.0 for row in race)
    with pytest.raises(ValueError):
        Game(base | {"luck_kind": "both"}, {"entries": []}, {k: None for k in by_id}, seats, seed, focal=2, train=True)
