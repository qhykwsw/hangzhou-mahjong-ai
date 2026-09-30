"""The C race strategy (_fast.race_*) must decide and value exactly as its Python specification."""

import importlib.util
import itertools
import random
from pathlib import Path

import pytest

from mahjong_ai.domain.engine import HandState, derive_seed
from mahjong_ai.domain.rules import WHITE, counts
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.sdk.shape import analyze

fast = pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

LOADED = itertools.count()
# The fixed pool's members differ from the reference's defaults (peng-speed) in these.
MEMBERS = {
    "peng-speed": ({}, 3),
    "race-classic": ({}, 1),
    "kong-eager": ({}, 2),
    "chi-speed": ({}, 4),
    "closed-early": ({}, 5),
    "patient-big": ({"Q": 0.10, "Q_EARLY": 0.06, "Q_LATE": 0.22}, 0),
    "flat-race": ({"UPGRADE": False, "HUNT": 1.35}, 0),
    "hunting": ({"HUNT": 1.35}, 0),  # no member, but the port's upgrade weight
}


def load(name):
    """A fresh copy of the specification with a member's constants, and its C profile."""
    constants, style = MEMBERS[name]
    spec = importlib.util.spec_from_file_location(
        f"race_reference_{next(LOADED)}", Path(__file__).with_name("race_reference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in constants.items():
        setattr(module, key, value)
    module.DRAW_RISK = 1 - (1 - module.Q) ** (1 / 3)
    profile = fast.race_profile(module.Q, module.SAVED, module.ONE_STEP, module.FAR_STEP, module.Q_EARLY,
                                module.Q_LATE, module.HUNT, module.UPGRADE, module.DRAW_RISK, module.TIE,
                                module.LAST_WALL, style)
    return module, profile, style


def load_complete():
    """A fresh copy of the completed-wait specification (fp-rule-046), and its C profile."""
    spec = importlib.util.spec_from_file_location(
        f"race_complete_reference_{next(LOADED)}", Path(__file__).with_name("race_complete_reference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = fast.race_profile(module.Q, module.SAVED, module.ONE_STEP, module.FAR_STEP, module.Q_EARLY,
                                module.Q_LATE, 1.0, True, module.DRAW_RISK, module.TIE, module.LAST_WALL, 0, True, True)
    return module, profile


def near_hand(rng, opened):
    """A hand of waiting length made of melds, pairs and whites: often waiting or one step away."""
    size = 13 - 3 * opened
    hand = [WHITE] * rng.choice((0, 0, 1, 1, 2, 3))
    while len(hand) < size:
        r = rng.random()
        if r < 0.4:
            s, n = rng.randrange(3), rng.randrange(7)
            hand += [s * 9 + n, s * 9 + n + 1, s * 9 + n + 2]
        elif r < 0.65:
            hand += [rng.randrange(34)] * 3
        elif r < 0.85:
            hand += [rng.randrange(34)] * 2
        else:
            hand.append(rng.randrange(34))
    hand = tuple(sorted(hand[:size]))
    return hand if max(counts(hand)) <= 4 else None


def honor_hand(rng, opened):
    """A white-holding hand with every other honor: baotou must then win on every draw."""
    size = 13 - 3 * opened
    hand = [WHITE] * rng.choice((1, 2, 3)) + list(range(27, WHITE))
    while len(hand) < size:
        hand.append(rng.randrange(WHITE))
    hand = tuple(sorted(hand[:size]))
    return hand if max(counts(hand)) <= 4 and WHITE in hand else None


def availability(rng, hand):
    held = counts(hand)
    return tuple(max(0, 4 - held[t] - rng.choice((0, 0, 0, 1, 2, 4))) for t in range(34))


@pytest.mark.parametrize("name", ["peng-speed", "patient-big", "flat-race", "hunting"])
def test_hand_values_are_the_python_ones(name):
    ref, profile, _ = load(name)
    rng = random.Random(7)
    checked = 0
    while checked < 250:
        opened = rng.choice((0, 0, 0, 1, 1, 2))
        hand = near_hand(rng, opened)
        if hand is None:
            continue
        available = availability(rng, hand)
        draws = rng.randrange(1, 17)
        chain, piao = rng.choice((0, 0, 1, 2)), rng.choice((0, 0, 1))
        assert fast.race_waiting(hand, opened, available, chain, piao, draws, profile, ref.examined) \
            == ref.waiting_value(hand, opened, available, chain, piao, draws), (hand, opened, draws)
        assert fast.race_reshaped(hand, opened, available, draws, profile, ref.examined) \
            == ref.reshaped_value(hand, opened, available, draws), (hand, opened, draws)
        assert fast.race_potential(hand, available) == ref.potential(hand, available)
        checked += 1


def test_baotou_fans_and_upgrade_drops_are_the_python_ones():
    ref, _, _ = load("peng-speed")
    rng = random.Random(11)
    baotou = 0
    for n in range(3000):
        opened = rng.choice((0, 0, 1, 2, 3))
        hand = honor_hand(rng, min(opened, 2)) if n % 5 == 0 else near_hand(rng, opened)
        if hand is None:
            continue
        held = list(counts(hand))
        if WHITE in hand and ref.is_baotou(hand, opened):
            baotou += 1
            available = availability(rng, hand)
            chain, piao = rng.choice((0, 1, 2)), rng.choice((0, 1, 2))
            packed = ref.baotou_draw_fans(hand, opened, available, chain, piao)
            assert fast.race_baotou_fans(held, opened, available, chain, piao) == (packed >> 32, packed & 0xFFFFFFFF)
        # Without a white, the completed wait asks only about drawing one.
        for tile in range(34) if WHITE in hand else (WHITE,):
            if held[tile] >= 4:
                continue
            brute = 0
            for drop in range(34):
                if drop != tile and held[drop]:
                    after = list(held)
                    after[tile] += 1
                    after[drop] -= 1
                    brute |= bool(fast.baotou(after, opened)) << drop
            assert fast.race_upgrade_drops(held, opened, tile) == brute, (hand, opened, tile)
    assert baotou > 50


def test_completed_wait_values_are_the_python_ones():
    ref, profile = load_complete()
    rng = random.Random(13)
    checked = 0
    while checked < 400:
        opened = rng.choice((0, 0, 0, 1, 1, 2))
        hand = near_hand(rng, opened)
        if hand is None:
            continue
        available = availability(rng, hand)
        draws = rng.randrange(1, 17)
        chain, piao = rng.choice((0, 0, 1, 2)), rng.choice((0, 0, 1))
        options, speed = rng.random() < 0.5, rng.choice((1.0, rng.randint(2, 40) / 20))
        assert fast.race_waiting(hand, opened, available, chain, piao, draws, profile, analyze, options, speed) \
            == ref.waiting_value(hand, opened, available, chain, piao, draws, options, speed), (hand, opened, draws)
        assert fast.race_reshaped(hand, opened, available, draws, profile, analyze, options, speed) \
            == ref.reshaped_value(hand, opened, available, draws, options, speed), (hand, opened, draws)
        checked += 1


def test_sums_over_a_set_add_in_pythons_order():
    rng = random.Random(5)
    for _ in range(20000):
        items = sorted(rng.sample(range(34), rng.randint(1, 34)))
        assert fast.race_set_order(items) == list(set(items))
    # A hand's tiles repeat: a repeated tile neither takes a slot nor counts toward a resize.
    for _ in range(20000):
        items = sorted(rng.choice(range(34)) for _ in range(rng.randint(1, 18)))
        assert fast.race_set_order(items) == list(set(items)), items
    assert fast.race_set_order([22, 23, 23, 24, 25]) == [24, 25, 22, 23]


@pytest.mark.parametrize("name", ["race-classic", "kong-eager", "peng-speed", "chi-speed", "closed-early",
                                  "patient-big", "flat-race"])
def test_every_decision_is_the_python_one(name):
    ref, profile, style = load(name)
    players = [ref.STYLES[style]() for _ in range(4)]
    config = RoundConfig(rounds=16, allow_tiebreak=False)
    decisions = 0
    for game in range(2):
        scores, dealer, history, views = [0] * 4, game, [], None
        for hand_no in range(1, 5):
            state = HandState(config, derive_seed(300 + game, "wall", hand_no), dealer, hand_no, scores, history,
                              False, previous_views=views)
            state.rounds_total = 16
            while not state.settled:
                legal = state.legal_actions()
                if len(legal) == 1:
                    state.step(legal[0], legal)
                    continue
                obs = state.observation(legal)
                action = players[state.turn].act(obs)
                assert obs.legal_actions[fast.race_act(obs, profile, ref.examined)] == action, (obs.hand, action)
                decisions += 1
                state.step(action, legal)
            scores = [s + d for s, d in zip(scores, state.delta)]
            history, views = state.events, state.public_views()
            if state.winner is not None:
                dealer = state.winner
    assert decisions > 100
