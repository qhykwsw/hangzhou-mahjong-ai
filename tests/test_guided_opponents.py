"""Training opponents' jitter and timeouts (guided.opponents): both engines alike, off by default, legal moves."""

import random
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame, opponents  # noqa: E402  (fastgame first: members register through it)
from guided.config import configuration  # noqa: E402
from guided.environment import Game  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from mahjong_ai.domain.engine import HandState  # noqa: E402
from mahjong_ai.domain.types import RoundConfig  # noqa: E402
from selfplay_core.actions import action_index  # noqa: E402
from test_guided_fastgame import members, members68, play, rule_table, rows_of  # noqa: E402,F401

CONFIG = RoundConfig(rounds=16, allow_tiebreak=False)
JITTER = {"opponent_jitter": {"sd": 0.1}}
TIMEOUT = {"opponent_timeout": {"mean": 0.15, "shape": 2.0}}


def base_cfg(**extra):
    return configuration(rounds=8, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True} | extra


def both(cfg, entries, seats, seed, **kw):
    """The game on the Python and the C engine, each played out; asserts they are the same game."""
    games = [cls(cfg, {"entries": []}, entries, seats, seed, train=True, **kw) for cls in (Game, fastgame.FastGame)]
    counts = [play(game) for game in games]
    assert counts[0] == counts[1] > 30
    assert games[0].scores == games[1].scores and games[0].predecision_reward == games[1].predecision_reward
    assert rows_of(games[0]) == rows_of(games[1])
    return games


def test_native_race_act_reads_an_observation_as_the_runtime_does(members, members68):
    """native.race_act on a registered profile is _fast.race_act; on an unchanged copy, the same choices too."""
    everyone = members + members68
    plans = [fastgame.race_member(m) for m in everyone]
    copies = [fastgame.native.jitter_profile(p, fastgame.native.race_floats(p)) for p, _ in plans]
    rng = random.Random(83)
    decisions = 0
    for _ in range(10):
        state = HandState(CONFIG, rng.getrandbits(64), rng.randrange(4), rng.randrange(1, 17), (5, -3, 0, -2))
        state.rounds_total = 16
        seats = [rng.randrange(len(everyone)) for _ in range(4)]
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) > 1:
                obs = state.observation(legal)
                k = seats[state.turn]
                action = everyone[k].act(obs)
                assert opponents.jittered_act(obs, plans[k][0], plans[k][1]) == action
                assert opponents.jittered_act(obs, copies[k], plans[k][1]) == action
                decisions += 1
            else:
                action = legal[0]
            state.step(action, legal)
    assert decisions > 300


def test_jittered_profiles_keep_their_fields_and_ranges(members68):
    profile = fastgame.race_member(members68[0])[0]
    floats = fastgame.native.race_floats(profile)
    assert len(floats) == len(opponents.FLOATS)
    rng = random.Random(3)
    for _ in range(2000):
        q, saved, one, far, early, late, hunt, risk, tie = opponents.jittered(floats, 0.5, rng)
        assert 0 <= q <= 0.95 and 0 <= late <= 0.95 and 0 <= early <= late and 0 <= risk <= 0.95
        assert 0 <= far <= 1 and min(saved, one, hunt, tie) >= 0
    assert opponents.jittered(floats, 0.0, random.Random(1)) == floats
    copy = fastgame.native.jitter_profile(profile, (0.2, 0.5, 0.25, 0.4, 0.08, 0.33, 1.1, 0.06, 2e-7))
    assert copy >= 64 and fastgame.native.race_floats(copy) == (0.2, 0.5, 0.25, 0.4, 0.08, 0.33, 1.1, 0.06, 2e-7)
    with pytest.raises(ValueError):
        fastgame.native.race_floats(copy + 7000)


def test_jittered_members_play_the_python_game_in_c(members, members68, monkeypatch):
    everyone = members + members68
    by_id = {f"m{i}": m for i, m in enumerate(everyone)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    entries = {k: None for k in by_id}
    for n, seats in enumerate([["model:current", "m1", "m8", "m12"], ["m3", "m7", "model:current", "m10"]]):
        seed = seed_for(89, "jitter", n)
        focal = seats.index("model:current")
        games = both(base_cfg(**JITTER), entries, seats, seed, focal=focal)
        # Every member seat has its own copy, the learner none; the C game plays the copies.
        assert sorted(games[1].jitter) == [s for s in range(4) if s != focal]
        assert {games[1].seats[0][s] for s in games[1].jitter} == {games[1].jitter[s][0] for s in games[1].jitter}
        assert all(games[1].seats[0][s] >= 64 for s in games[1].jitter)
        plain = Game(base_cfg(), {"entries": []}, entries, seats, seed, focal=focal, train=True)
        play(plain)
        assert rows_of(plain) != rows_of(games[0])  # the copies decide differently somewhere


@pytest.mark.parametrize("extra", [TIMEOUT, JITTER | TIMEOUT,
                                   JITTER | TIMEOUT | {"stable_wall": True, "luck": True, "luck_kind": "race",
                                                       "vine": True}])
def test_timeouts_and_python_rule_seats_play_the_python_game_on_the_c_engine(members, monkeypatch, extra):
    """Rule seats deciding in C and in Python (a Mirror) time out alike; a rule out of time does not act."""
    entries, built = rule_table(members, monkeypatch, {"py": 2})
    for n, seats in enumerate([["model:current", "py", "m1", "m3"], ["m0", "py", "model:current", "m5"]]):
        built.clear()
        focal = seats.index("model:current")
        games = both(base_cfg(**extra), entries, seats, seed_for(97, "timeouts", n, str(sorted(extra))), focal=focal)
        python, fast = built[0][1], built[1][1]
        assert python.seen == fast.seen and python.finished == fast.finished
        rates = games[1].timeouts
        assert rates[focal] == 0.0 and all(rates[s] > 0 for s in range(4) if s != focal)
        members_at = [s for s in range(4) if seats[s] in ("m0", "m1", "m3", "m5")]
        assert sorted(games[1].jitter) == (members_at if "opponent_jitter" in extra else [])
        built.clear()
        plain = Game(base_cfg(), {"entries": []}, entries, seats, seed_for(97, "timeouts", n, str(sorted(extra))),
                     focal=focal, train=True)
        play(plain)
        assert rows_of(plain) != rows_of(games[0]) and len(built[0][1].seen) != len(python.seen)


def test_a_rule_out_of_time_every_time_never_acts(members, monkeypatch):
    entries, built = rule_table(members, monkeypatch, {"py": 2})
    monkeypatch.setattr(opponents.random.Random, "betavariate", lambda self, a, b: 1.0)
    seats = ["model:current", "py", "m1", "m3"]
    games = both(base_cfg(**TIMEOUT), entries, seats, seed_for(101, "always"), focal=0)
    assert games[1].timeouts == (0.0, 1.0, 1.0, 1.0)
    assert built[0][1].seen == built[1][1].seen == []


def test_the_platforms_move_is_legal_and_the_engines(members68):
    """At every decision of random hands: the timeout move is legal, HandState.timeout_action's, and C's."""
    plans = [fastgame.race_member(m) for m in members68]
    rng = random.Random(103)
    kinds = set()
    for _ in range(30):
        seed, dealer = rng.getrandbits(64), rng.randrange(4)
        state = HandState(CONFIG, seed, dealer, 1, (0, 0, 0, 0))
        state.rounds_total = 16
        hand = fastgame.native.Hand(seed, dealer, 1, (0, 0, 0, 0), 16)
        hand.timeouts((1.0, 1.0, 1.0, 1.0), (1, 2, 3, 4))  # every decision out of time
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) > 1:
                move = opponents.default_action(state, legal)
                assert move in legal and move == state.timeout_action(legal)
                assert hand.timeout() == action_index(move)
                kinds.add((state.phase, move.action, state.drawn is not None))
                if state.phase == "draw" and move.action == "discard" and state.drawn is not None:
                    assert move.tile == state.drawn  # after a draw: the drawn tile
                index = hand.act(*plans[rng.randrange(len(plans))]) if rng.random() < 0.8 else action_index(move)
                action = next(a for a in legal if action_index(a) == index)
            else:
                action = legal[0]
            state.step(action, legal)
            hand.step(action_index(action))
    assert {("draw", "discard", True), ("draw", "discard", False), ("response_peng", "pass", False)} <= kinds
    assert any(kind[1] == "hu" for kind in kinds)


def test_streams_and_rates():
    # The Python stream is native.c's (the games above compare them); its uniforms and the rates look right.
    draws = np.array([opponents.uniform(seed_for(1, "u", k // 50), k % 50) for k in range(20000)])
    assert 0 <= draws.min() and draws.max() < 1 and abs(draws.mean() - 0.5) < 0.01
    rates = np.array([opponents.plan(TIMEOUT, seed_for(2, "r", k), ["m", "m", "m", "model:current"], {})[1]
                      for k in range(3000)])
    assert (rates[:, 3] == 0).all()
    assert abs(rates[:, :3].mean() - 0.15) < 0.01 and (rates[:, :3] > 0.5).mean() > 0.04
    assert opponents.plan({}, 1, ["m"] * 4, {}) == ({}, None)
    assert opponents.plan({"opponent_jitter": {"sd": 0}, "opponent_timeout": {"mean": 0}}, 1, ["m"] * 4, {}) == ({}, None)
    for bad in [{"opponent_jitter": {"sd": -1}}, {"opponent_jitter": {"scale": 1}},
                {"opponent_timeout": {"mean": 1.0}}, {"opponent_timeout": {"mean": 0.1, "shape": 0}}]:
        with pytest.raises(ValueError):
            opponents.settings(bad)


def test_zero_settings_and_evaluation_change_nothing(members, members68, monkeypatch):
    everyone = members + members68
    by_id = {f"m{i}": m for i, m in enumerate(everyone)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    entries = {k: None for k in by_id}
    seats, seed = ["m2", "model:current", "m9", "m11"], seed_for(107, "zero")
    zero = {"opponent_jitter": {"sd": 0.0}, "opponent_timeout": {"mean": 0.0}}
    for cls in (Game, fastgame.FastGame):
        plain = cls(base_cfg(), {"entries": []}, entries, seats, seed, focal=1, train=True)
        off = cls(base_cfg(**zero), {"entries": []}, entries, seats, seed, focal=1, train=True)
        assert play(plain) == play(off) and rows_of(plain) == rows_of(off) and plain.scores == off.scores
        assert off.jitter == {} and off.timeouts is None
    # An evaluation game (no train) ignores the keys: same requests and scores.
    games = []
    for cfg in (base_cfg(), base_cfg(**JITTER, **TIMEOUT)):
        game = Game(cfg, {"entries": []}, entries, seats, seed, focal=1, group=0)
        requests = []
        while (request := game.advance()) is not None:
            legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
            requests.append(request["row"]["planes"].tobytes())
            game.apply((legal[len(requests) % len(legal)], 0.0, 0.0))
        assert game.jitter == {} and game.timeouts is None
        games.append((requests, game.scores))
    assert games[0] == games[1]


def test_a_reused_id_reads_nothing_memoised_for_the_copy_before(members68):
    """The id ring wraps after 8000 copies and empties the memos: a reused id decides as its own floats say."""
    plans = [fastgame.race_member(m) for m in members68]
    rng = random.Random(109)
    seen = []
    for _ in range(4):
        state = HandState(CONFIG, rng.getrandbits(64), rng.randrange(4), 1, (0, 0, 0, 0))
        state.rounds_total = 16
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) > 1:
                obs, k = state.observation(legal), rng.randrange(len(members68))
                action = members68[k].act(obs)
                seen.append((obs, k, action))
            else:
                action = legal[0]
            state.step(action, legal)
    native = fastgame.native

    def distorted(k):
        return native.jitter_profile(plans[k][0], (0.9, 0.0, 0.05, 0.95, 0.5, 0.9, 3.0, 0.6, 0.0))

    while distorted(0) != 64:  # the ring has just wrapped: the next copies are 65, 66, ...
        pass
    ids = [64] + [distorted(k) for k in range(1, len(plans))]
    moved = sum(opponents.jittered_act(obs, ids[k], plans[k][1]) != action for obs, k, action in seen)
    assert moved > 0  # the distorted copies decide differently, with their values memoised under these ids
    for _ in range(8000 - len(plans)):
        distorted(0)
    again = [native.jitter_profile(p, native.race_floats(p)) for p, _ in plans]
    assert again == ids  # the same ids, now unchanged copies
    assert all(opponents.jittered_act(obs, again[k], plans[k][1]) == action for obs, k, action in seen)
