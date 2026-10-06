"""fp-rule-131 in the C training game (guided/rule131.h) plays as its frozen Python does, bit for bit.

The package is the rule line's frozen copy (not in Git): the tests run where it
is (RULE131_PACKAGE, or its place under the firstprinciples project).
"""

import hashlib
import importlib.util
import os
from pathlib import Path
import random
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame  # noqa: E402  (fastgame first: profiles register through it)
from guided.config import configuration  # noqa: E402
from guided.environment import Game, rule_seat_kinds  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from mahjong_ai.domain.engine import HandState, derive_seed  # noqa: E402
from mahjong_ai.domain.types import RoundConfig  # noqa: E402
from mahjong_ai.sdk.shape import analyze  # noqa: E402
from selfplay_core.actions import action_index  # noqa: E402
from test_guided_fastgame import members, play, rows_of  # noqa: E402,F401

PACKAGE = Path(os.environ.get("RULE131_PACKAGE")
               or ROOT / "projects/personal/firstprinciples/.local/attempts/fp-rule-131/package")
pytestmark = pytest.mark.skipif(not fastgame._is_rule131(str(PACKAGE)), reason="fp-rule-131's frozen package is not here")


def rule131():
    """A fresh fp-rule-131 policy from its package (a module of its own, as a worker loads it)."""
    name = "_rule131_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, PACKAGE / "strategy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build(PACKAGE / "model")


def test_the_tables_are_the_packages_lines():
    native = (PACKAGE / "native.c").read_text().split("\n")
    copied = [line for line in (PROJECT / "guided/rule131_tables.h").read_text().split("\n")
              if line.startswith("static const ") and "MIMIC_WEIGHTS" not in line]
    assert len(copied) == 20 and all(line in native for line in copied)
    module = rule131().__class__.__module__
    weights = sys.modules[module].MIMIC_WEIGHTS
    text = (PROJECT / "guided/rule131_tables.h").read_text()
    for row in weights:
        assert "{" + ", ".join(repr(float(v)) for v in row) + "}" in text
    assert hashlib.sha256((PACKAGE / "strategy.py").read_bytes()).hexdigest() == fastgame.RULE131["strategy.py"]


def test_every_decision_is_the_pythons():
    """Hands of a game side by side on both engines: at every choice of a 131 seat, both name the same action."""
    policy = rule131()
    member, fallback = fastgame.rule131_member(policy)
    stand_in = fastgame.native.race_profile(0.15, 0.467, 0.22, 0.45, 0.10, 0.30, 1.0, True,
                                            1 - (1 - 0.15) ** (1 / 3), 1e-7, 20, 0)
    rounds = 4
    config = RoundConfig(rounds=rounds, allow_tiebreak=False)
    decisions, seen = 0, set()
    for g in range(4):
        rng = random.Random(f"rule131-{g}")
        seed, seats = rng.getrandbits(63), rng.sample(range(4), 2 + g % 3)
        history, views, scores, dealer = [], None, [0, 0, 0, 0], seed % 4
        for hand_no in range(1, rounds + 1):
            wall = derive_seed(seed, "wall", hand_no)
            state = HandState(config, wall, dealer, hand_no, scores, history, False, previous_views=views)
            state.rounds_total = rounds
            hand = fastgame.native.Hand(wall, dealer, hand_no, scores, rounds)
            while not state.settled:
                legal = state.legal_actions()
                if len(legal) > 1 and state.turn in seats:
                    action = policy.act(state.observation(legal))
                    assert hand.act(member, fallback) == action_index(action)
                    decisions += 1
                    seen.add(frozenset(a.action for a in legal))
                elif len(legal) > 1 and rng.random() > 0.2:
                    index = hand.act(stand_in, analyze)
                    action = next(a for a in legal if action_index(a) == index)
                else:
                    action = legal[rng.randrange(len(legal))]
                state.step(action, legal)
                hand.step(action_index(action))
            scores = [s + d for s, d in zip(scores, state.delta)]
            history, views = state.events, state.public_views()
            dealer = state.winner if state.winner is not None else dealer
    assert decisions > 500 and {frozenset({"discard", "hu"}), frozenset({"pass", "chi"})} <= seen


@pytest.mark.parametrize("extra", [
    {},
    {"stable_wall": True, "luck": True, "critic_features": "guided.encoding:oracle"},
    {"vine": True, "stable_wall": True, "luck": True},
    {"stage": {"kind": "advancement", "games": 10, "per_game_sd": 70}},
    {"encoding": "guided-current-hand-v2", "stage": {"kind": "final", "games": 10, "per_game_sd": 70}},
    {"encoding": "guided-current-hand-v3", "vine": True},
])
def test_131_seats_play_the_python_game_in_c(members, monkeypatch, extra):
    def build(bundle, cid):
        return rule131() if cid == "r131" else members[int(cid[1:])]

    monkeypatch.setattr(artifacts, "build_policy", build)
    entries = {k: None for k in ("r131", "m0", "m1")}
    cfg = configuration(rounds=8, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True} | extra
    for n, seats in enumerate([["model:current", "r131", "m1", "r131"], ["r131", "r131", "model:current", "m0"]]):
        seed = seed_for(53, "fast-rule131", n, str(sorted(extra)))
        focal = seats.index("model:current")
        games = [cls(cfg, {"entries": []}, entries, seats, seed, focal=focal, train=True)
                 for cls in (Game, fastgame.FastGame)]
        assert not games[1].python_seats and not games[1].mirrored
        counts = [play(game) for game in games]
        assert counts[0] == counts[1] > 30
        assert games[0].scores == games[1].scores and games[0].hand_no == games[1].hand_no == 8
        assert games[0].predecision_reward == games[1].predecision_reward
        assert rows_of(games[0]) == rows_of(games[1])
        # What the collection reports: every 131 seat decided in C, none by callback.
        assert rule_seat_kinds(games[1]) == {"c": 3, "callback": 0}
        assert rule_seat_kinds(games[0]) == {"python_engine": 3}


def test_131_teaches_on_the_c_hand(members, monkeypatch):
    monkeypatch.setattr(artifacts, "build_policy",
                        lambda bundle, cid: rule131() if cid == "teach" else members[int(cid[1:])])
    cfg = configuration(rounds=8, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True, "vine": True}
    seats = ["m0", "model:current", "m2", "m3"]
    entries = {k: None for k in ("teach", "m0", "m2", "m3")}
    games = [cls(cfg, {"entries": []}, entries, seats, seed_for(59, "fast-rule131-teacher"), focal=1,
                 teacher="teach", train=True) for cls in (Game, fastgame.FastGame)]
    assert games[1].teacher_member is not None and not games[1].mirrored
    counts = [play(game) for game in games]
    assert counts[0] == counts[1] > 30
    assert rows_of(games[0]) == rows_of(games[1])
    assert all(row["teacher"] >= 0 for row in games[1].rows)
    assert games[0].teacher_errors == games[1].teacher_errors == 0
