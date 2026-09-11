"""The C accelerator must agree with the pure-Python reference it replaces."""

import itertools
import random

import pytest

from mahjong_ai.domain import rules
from mahjong_ai.sdk import shape

fast = pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")


def random_vector(rng, total, tiles=range(34)):
    tiles = list(tiles)
    vector = [0] * 34
    left = total
    while left:
        tile = rng.choice(tiles)
        if vector[tile] < 4:
            vector[tile] += 1
            left -= 1
    return tuple(vector)


def random_hand(rng, size, white_bias=0.15):
    hand = []
    while len(hand) < size:
        tile = 33 if rng.random() < white_bias else rng.randrange(34)
        if hand.count(tile) < 4:
            hand.append(tile)
    return tuple(sorted(hand))


def reference_analyze(hand, open_melds, available):
    """analyze() with the accelerator disabled."""
    saved = shape._fast
    shape._fast = None
    try:
        shape.shanten.cache_clear()
        return shape.analyze(hand, open_melds, available)
    finally:
        shape._fast = saved
        shape.shanten.cache_clear()


def test_shape_value_matches_reference_on_random_vectors():
    rng = random.Random(20260911)
    for _ in range(20000):
        vector = random_vector(rng, rng.randint(0, 17))
        open_melds = rng.randint(0, 4)
        assert fast.shape_value(vector, open_melds) == shape._normal(
            vector, open_melds, 0, 0
        )


def test_shape_value_matches_reference_exhaustively_within_two_suits():
    for values in itertools.product(range(5), repeat=8):
        if sum(values) not in (13, 14):
            continue
        vector = (tuple(values) + (0,) * 26)[:34]
        for open_melds in (0, 1, 2):
            assert fast.shape_value(vector, open_melds) == shape._normal(
                vector, open_melds, 0, 0
            )


def test_shanten_and_analyze_match_the_reference_path():
    rng = random.Random(4242)
    assert shape._fast is not None
    for _ in range(1200):
        open_melds = rng.choice([0, 0, 0, 1, 2])
        size = rng.choice([13 - 3 * open_melds, 14 - 3 * open_melds])
        hand = random_hand(rng, size, white_bias=0.3)
        available = tuple(rng.randint(0, 4) for _ in range(34))
        shape.shanten.cache_clear()
        expected = reference_analyze(hand, open_melds, available)
        shape.shanten.cache_clear()
        assert shape.shanten(hand, open_melds) == expected["shanten"]
        shape.shanten.cache_clear()
        assert shape.analyze(hand, open_melds, available) == expected
    shape._normal.cache_clear()


def test_can_hu_matches_the_reference_shape_test():
    rng = random.Random(31337)
    for _ in range(20000):
        open_melds = rng.choice([0, 0, 0, 1, 2, 3, 4])
        size = 14 - 3 * open_melds
        hand = random_hand(rng, size, white_bias=0.18)
        vector = tuple(hand.count(tile) for tile in range(34))
        assert fast.can_hu(vector, open_melds) == rules._shape_reference(
            vector, open_melds
        )


def test_can_hu_preserves_reference_behavior_for_overfull_invalid_input():
    hand = (0,) * 5 + (1,) * 3 + (2,) * 3 + (3,) * 3
    assert rules.can_hu(hand) == rules._shape_reference(
        tuple(hand.count(tile) for tile in range(34)), 0
    )


def test_numeric_fan_matches_reference_details():
    rng = random.Random(20260912)
    for _ in range(20000):
        open_melds = rng.choice([0, 0, 0, 1, 2, 3])
        size = 14 - 3 * open_melds
        hand = random_hand(rng, size, white_bias=0.25)
        drawn = rng.choice(hand)
        chain = rng.randint(0, 3)
        piao = rng.randint(0, 3)
        you_cai = bool(rng.randint(0, 1))
        kong_draw = bool(rng.randint(0, 1))
        expected = rules.calculate_fan(
            hand, drawn, open_melds, chain, piao, you_cai, kong_draw
        )
        actual = fast.fan_value(
            tuple(hand.count(tile) for tile in range(34)),
            drawn,
            open_melds,
            chain,
            piao,
            you_cai,
            kong_draw,
        )
        assert actual == (expected.fan if expected else 0)
