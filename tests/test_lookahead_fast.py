"""Fused lookahead must preserve the full Python draw/discard enumeration."""

import random

import pytest

from mahjong_ai.sdk import lookahead

fast = pytest.importorskip("mahjong_ai._fast")


@pytest.mark.parametrize("opened", range(5))
def test_fused_draw_discard_matches_enumeration(monkeypatch, opened):
    rng = random.Random(92103 + opened)
    hands = []
    for _ in range(240):
        hand = []
        while len(hand) < 13 - 3 * opened:
            tile = 33 if rng.random() < 0.25 else rng.randrange(34)
            if hand.count(tile) < 4:
                hand.append(tile)
        hands.append(tuple(sorted(hand)))
    for hand in hands:
        available = tuple(rng.randint(0, 4 - hand.count(t)) for t in range(34))
        context = rng.randrange(6), rng.randrange(5), rng.choice([False, True])
        monkeypatch.setattr(lookahead, "_fast", fast)
        near = lookahead.baotou_outs(hand, opened, available)
        route = lookahead.tenpai_route_value(hand, opened, available, *context)
        monkeypatch.setattr(lookahead, "_fast", None)
        assert near == lookahead.baotou_outs(hand, opened, available)
        assert route == lookahead.tenpai_route_value(hand, opened, available, *context)


def test_unavailable_draws_and_nonwaiting_route():
    hand = (0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 28, 33)
    assert lookahead.baotou_outs(hand, 0, (0,) * 34) == 0
    assert lookahead.tenpai_route_value(hand, 0, (0,) * 34) == 0.0
    assert lookahead.tenpai_route_value((*hand, 33), 0, (4,) * 34) == 0.0


def test_baotou_absent_honor_representative_matches_every_draw():
    """Exercise joker-heavy wins, all-honors fallback, and all legal draw tiles."""
    from mahjong_ai.domain.rules import counts

    rng = random.Random(250925)
    for opened in range(5):
        for _ in range(6000):
            hand = []
            while len(hand) < 13 - 3 * opened:
                tile = 33 if rng.random() < 0.3 else rng.randrange(34)
                if hand.count(tile) < 4:
                    hand.append(tile)
            c = list(counts(hand))
            # The test deliberately bypasses the optimized baotou predicate.
            expected = bool(c[33])
            if expected:
                for tile in range(34):
                    if c[tile] >= 4:
                        continue
                    c[tile] += 1
                    won = fast.can_hu(c, opened)
                    c[tile] -= 1
                    if not won:
                        expected = False
                        break
            assert fast.baotou(c, opened) == expected
