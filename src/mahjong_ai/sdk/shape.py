"""Exact structural distance, public tile availability and legal winning draws."""

from functools import lru_cache
from collections import Counter
from mahjong_ai.domain.rules import WHITE, counts, can_hu, calculate_fan

try:  # Optional C accelerator; the pure-Python reference below stays authoritative.
    from mahjong_ai import _fast
except ImportError:  # pragma: no cover - exercised by running without the extension
    _fast = None


@lru_cache(maxsize=300000)
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


@lru_cache(maxsize=200000)
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
    if max(c) > 4:
        raise ValueError("more than four copies")
    if _fast is not None:
        value = _fast.shanten_raw(c, open_melds)
        # Every winning boundary defers to the reference implementation, so the
        # accelerator only has to agree on non-negative distances.
        if value >= 0:
            return value
    return _shanten_reference(hand, open_melds)


def _shanten_reference(hand, open_melds):
    c = counts(hand)
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


def _improving_reference(hand, open_melds, available, distance):
    """Reference scan of every legal draw; used without the accelerator and at
    winning boundaries where the guarded distance can move."""
    hand_counts = counts(hand)
    improving, candidates = [], []
    for tile in range(34):
        if available[tile] <= 0 or hand_counts[tile] >= 4:
            continue
        drawn = tuple(sorted((*hand, tile)))
        drawn_distance = shanten(drawn, open_melds)
        if drawn_distance < distance:
            improving.append(tile)
        if drawn_distance == -1:
            candidates.append(tile)
    return improving, candidates


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
    hand_counts = counts(hand)
    distance = shanten(hand, open_melds)
    available = (
        remaining if remaining is not None else tuple(4 - n for n in hand_counts)
    )
    improving = []
    wins = []
    fans = {}
    if len(hand) == 13 - 3 * open_melds:
        candidates = None
        if _fast is not None:
            raw, fast_improving, negatives = _fast.improving(
                hand_counts, open_melds, available
            )
            if raw >= 0:
                distance = raw
                improving = list(fast_improving)
                # For a physically valid draw (every count <= 4), a
                # negative raw shanten is exactly a winning shape. The C
                # routine already excludes unavailable and fifth-copy
                # draws, so crossing back into Python can_hu here only
                # repeats the same shape proof for every candidate.
                candidates = list(negatives)
                if distance == 0:
                    # 未通过胡牌边界的摸牌，其受保护的距离是 0 而不是负数，
                    # 听牌手上这些摸牌不是进张。
                    boundary = set(negatives) - set(candidates)
                    improving = [tile for tile in improving if tile not in boundary]
        if candidates is None:
            improving, candidates = _improving_reference(
                hand, open_melds, available, distance
            )
        for tile in candidates:
            drawn = tuple(sorted((*hand, tile)))
            numeric_fan = None
            if _fast is not None and hasattr(_fast, "fan_value"):
                numeric_fan = _fast.fan_value(
                    counts(drawn),
                    tile,
                    open_melds,
                    chain,
                    piao,
                    you_cai_bi_kao,
                    False,
                )
            fan = (
                calculate_fan(drawn, tile, open_melds, chain, piao, you_cai_bi_kao)
                if numeric_fan is None
                else None
            )
            if numeric_fan or fan:
                wins.append(tile)
                fans[tile] = numeric_fan if numeric_fan is not None else fan.fan
    return {
        "shanten": distance,
        "improving": improving,
        "ukeire": sum(available[t] for t in improving),
        "winning_tiles": wins,
        "winning_copies": sum(available[t] for t in wins),
        "fans": fans,
        "fan_value": sum(available[t] * v for t, v in fans.items()),
    }
