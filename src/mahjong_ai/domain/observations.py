"""Shared policy-visible chronology and decisions derived from public history."""

from dataclasses import replace

from mahjong_ai.domain.rules import WHITE, is_baotou


def policy_events(events, seat):
    """Canonical game events; transport pass/timeout and initial deals are excluded."""
    result = []
    for event in events:
        if event.kind not in ("draw", "discard", "chi", "peng", "gang", "hand_ended"):
            continue
        tile, tiles = event.tile, tuple(sorted(event.tiles))
        if (
            event.kind == "draw"
            and event.seat != seat
            or event.concealed
            and event.seat != seat
        ):
            tile, tiles = None, ()
        elif event.kind in ("peng", "gang") and tile is not None:
            tiles = (tile,) * (3 if event.kind == "peng" else 4)
        result.append(
            replace(event, tile=tile, tiles=tiles, scores=tuple(event.scores))
        )
    return tuple(result)


def meld_origins(events, hand_no):
    """Recover public claim sources, retaining the original source of added kongs."""
    origins = {}
    discard = None
    for e in events:
        if e.hand_no != hand_no:
            continue
        if e.kind == "discard":
            discard = e
        elif e.kind in ("chi", "peng", "gang"):
            key = (e.seat, e.kind, tuple(e.tiles))
            if e.kind == "gang" and (e.seat, "peng", (e.tile,) * 3) in origins:
                origins[key] = origins[(e.seat, "peng", (e.tile,) * 3)]
            elif not e.concealed and discard is not None and e.tile == discard.tile:
                origins[key] = discard.seat
                discard = None
    return origins


def can_discard_piao(
    hand, drawn_tile, open_melds, events, seat, hand_no, catch_exempt=False
):
    """Price a white discard using the pre-draw or pre-call waiting hand."""
    if WHITE not in hand:
        return False
    after = list(hand)
    after.remove(WHITE)
    if not is_baotou(after, open_melds):
        return False
    if drawn_tile is not None:
        waiting = list(hand)
        waiting.remove(drawn_tile)
        return is_baotou(waiting, open_melds)
    if catch_exempt:
        return True
    # A call contributes one river tile and consumes two previously held tiles.
    # Reconstruct the caller's prior waiting hand without a hidden-state cache.
    last = next(
        (
            e
            for e in reversed(events)
            if e.seat == seat
            and e.hand_no == hand_no
            and e.kind in ("draw", "discard", "chi", "peng", "gang")
        ),
        None,
    )
    if last is None or last.kind not in ("chi", "peng") or open_melds < 1:
        return False
    used = list(last.tiles)
    if last.tile not in used or len(used) != 3:
        return False
    used.remove(last.tile)
    return is_baotou([*hand, *used], open_melds - 1)
