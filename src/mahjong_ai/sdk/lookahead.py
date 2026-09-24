"""Exact one-draw/discard projections using only public tile availability."""

from mahjong_ai.domain.rules import WHITE, counts, is_baotou
from mahjong_ai.sdk.shape import analyze

try:
    from mahjong_ai import _fast
except ImportError:  # pragma: no cover
    _fast = None


def baotou_outs(hand, open_melds, available):
    """Copies of draws permitting a discard into baotou, from a hand with white."""
    if _fast is not None:
        return _fast.baotou_outs(counts(hand), open_melds, available)
    if WHITE not in hand:
        return 0
    total = 0
    for tile in range(34):
        if available[tile] <= 0 or hand.count(tile) >= 4:
            continue
        drawn = tuple(sorted((*hand, tile)))
        for drop in set(drawn):
            rest = list(drawn)
            rest.remove(drop)
            if is_baotou(tuple(sorted(rest)), open_melds):
                total += available[tile]
                break
    return total


def tenpai_route_value(
    hand, open_melds, available, chain=0, piao=0, you_cai_bi_kao=False
):
    """For a one-shanten waiting hand, availability-weighted best tenpai fan.

    Each improving draw is followed by the best discard into tenpai. The
    destination's fan value uses the original availability, including the
    hypothetical draw, matching the rule policies' route-pricing definition.
    """
    if len(hand) != 13 - 3 * open_melds:
        return 0.0
    if _fast is not None:
        return _fast.tenpai_route_value(
            counts(hand), open_melds, available, chain, piao, int(you_cai_bi_kao)
        )
    row = analyze(hand, open_melds, available, chain, piao, you_cai_bi_kao)
    if row["shanten"] != 1:
        return 0.0
    total, weighted = 0, 0.0
    for tile in row["improving"]:
        copies = available[tile]
        if copies <= 0:
            continue
        drawn = tuple(sorted((*hand, tile)))
        best = 0.0
        for drop in set(drawn):
            rest = list(drawn)
            rest.remove(drop)
            dest = analyze(
                tuple(sorted(rest)), open_melds, available, chain, piao, you_cai_bi_kao
            )
            if dest["shanten"] == 0 and dest["fan_value"] > best:
                best = dest["fan_value"]
        total += copies
        weighted += copies * best
    return weighted / total if total else 0.0
