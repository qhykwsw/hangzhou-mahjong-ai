"""Exact structural distance, public tile availability and legal winning draws."""

from functools import lru_cache
from collections import Counter
from mahjong_ai.domain.rules import WHITE, counts, can_hu, calculate_fan


@lru_cache(maxsize=100000)
def _normal(c, melds, pairs, partial):
    try:
        i = next(i for i, n in enumerate(c) if n)
    except StopIteration:
        return 8 - 2 * melds - min(partial, 4 - melds) - pairs
    x = list(c)
    x[i] -= 1
    best = _normal(tuple(x), melds, pairs, partial)
    if melds < 4:
        if c[i] >= 3:
            x = list(c)
            x[i] -= 3
            best = min(best, _normal(tuple(x), melds + 1, pairs, partial))
        if i < 27 and i % 9 < 7 and c[i + 1] and c[i + 2]:
            x = list(c)
            for t in (i, i + 1, i + 2):
                x[t] -= 1
            best = min(best, _normal(tuple(x), melds + 1, pairs, partial))
    if c[i] >= 2:
        x = list(c)
        x[i] -= 2
        if not pairs:
            best = min(best, _normal(tuple(x), melds, 1, partial))
        if partial < 4 - melds:
            best = min(best, _normal(tuple(x), melds, pairs, partial + 1))
    if i < 27 and partial < 4 - melds:
        for d in (1, 2):
            if i % 9 + d < 9 and c[i + d]:
                x = list(c)
                x[i] -= 1
                x[i + d] -= 1
                best = min(best, _normal(tuple(x), melds, pairs, partial + 1))
    return best


@lru_cache(maxsize=30000)
def shanten(hand, open_melds=0):
    """-1 is a complete structural hand; 0 is one draw from completion.

    White tiles complete missing positions. Four identical tiles may form two
    pairs under the platform's seven-pairs rule. Availability is separate.
    """
    c = counts(hand)
    if not 0 <= open_melds <= 4 or len(hand) not in (
        13 - 3 * open_melds,
        14 - 3 * open_melds,
    ):
        raise ValueError("shanten expects a draw or discard hand")
    if any(n > 4 for n in c):
        raise ValueError("more than four copies")
    j = c[WHITE]
    natural = c[:WHITE] + (0,)
    normal = _normal(natural, open_melds, 0, 0) - j
    if not open_melds:
        pairs = sum(n // 2 for n in natural)
        singles = sum(n % 2 for n in natural)
        seven = 14 - (2 * pairs + min(singles, 7 - pairs)) - j - 1
        normal = min(normal, seven)
    # A fifth copy cannot complete a hand. Explicitly verify the winning boundary.
    if normal < 0 and not can_hu(hand, open_melds):
        normal = 0
    return max(-1, normal)


def remaining_tiles(obs):
    visible = Counter(obs.hand)
    for row in obs.discards:
        visible.update(row)
    for row in obs.melds:
        for meld in row:
            visible.update(meld.tiles)
    return tuple(max(0, 4 - visible[t]) for t in range(34))


def analyze(hand, open_melds=0, remaining=None, chain=0, piao=0, you_cai_bi_kao=False):
    hand = tuple(sorted(hand))
    distance = shanten(hand, open_melds)
    available = (
        remaining if remaining is not None else tuple(4 - n for n in counts(hand))
    )
    improving = []
    wins = []
    fans = {}
    if len(hand) == 13 - 3 * open_melds:
        for tile in range(34):
            if available[tile] <= 0 or hand.count(tile) >= 4:
                continue
            drawn = tuple(sorted((*hand, tile)))
            if shanten(drawn, open_melds) < distance:
                improving.append(tile)
            fan = calculate_fan(drawn, tile, open_melds, chain, piao, you_cai_bi_kao)
            if fan:
                wins.append(tile)
                fans[tile] = fan.fan
    return {
        "shanten": distance,
        "improving": improving,
        "ukeire": sum(available[t] for t in improving),
        "winning_tiles": wins,
        "winning_copies": sum(available[t] for t in wins),
        "fans": fans,
        "fan_value": sum(available[t] * v for t, v in fans.items()),
    }
