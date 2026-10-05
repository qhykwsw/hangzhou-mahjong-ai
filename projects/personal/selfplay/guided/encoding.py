"""Legal inputs with untruncated current-hand chronology.

Tile/shape features reuse the pure-numpy encoder. Privileged features are a
separate function used exclusively by training workers.

Each encoding is named; an exported Actor records the one it was trained on
(meta.json "encoding"). guided-current-hand-v2 appends CALL_PLANES planes to
the v1 planes, so a v2 row's first N_PLANES planes are its v1 row and an Actor
of either encoding reads the prefix it was built for (networks.HistoryEncoder).
The v2 planes are computed from the v1 row alone (call_planes), so every place
that holds a v1 row, a worker's Python or C encoder or the model service's
binary requests, extends it the same way (augment).
"""

import math

import numpy as np

from mahjong_ai.domain.types import StageKind
from selfplay_core.actions import CHI, GANG_ADDED, GANG_CONCEALED, GANG_EXPOSED, HU, PENG
from selfplay_core.encoding import (
    EVENT_KINDS,
    N_PLANES,
    N_SCALARS,
    P_HAND,
    P_REMAINING,
    encode as base_encode,
)

# The engine silently falls back to pure Python without the native extension,
# which makes shape features about an order of magnitude slower. Fail instead.
try:
    from mahjong_ai import _fast
except ImportError as error:
    import sysconfig
    from pathlib import Path

    import mahjong_ai

    target = Path(mahjong_ai.__file__).parent / (
        "_fast" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    raise ImportError(
        f"guided encoding requires the compiled extension {target}; build "
        "_fast.c there for this interpreter (setuptools build_ext) or copy a "
        "matching build before training, evaluation or serving"
    ) from error

ENCODING = "guided-current-hand-v1"
ENCODING_V2 = "guided-current-hand-v2"
CALL_PLANES = 16
# (planes, scalars) an Actor of each encoding takes.
INPUTS = {ENCODING: (N_PLANES, N_SCALARS), ENCODING_V2: (N_PLANES + CALL_PLANES, N_SCALARS)}

# The own fan state among selfplay_core's scalars, after the score context
# (11), the horizon (4 + stage kinds), the seats (11) and the wall and phase (6).
_CHAIN = 11 + 4 + len(StageKind) + 11 + 6
_PIAO, _YOU_CAI = _CHAIN + 1, _CHAIN + 4


def encode(obs, lookahead=True):
    base = base_encode(obs, lookahead=lookahead, include_events=False)
    # Public events are chronological. Only the current hand's suffix enters
    # this encoder; scanning all previous hands repeats increasing work.
    events = obs.public_events
    first = len(events)
    while first and events[first - 1].hand_no == obs.stage.round_no:
        first -= 1
    events = events[first:]
    result = np.zeros((max(1, len(events)), 5), dtype=np.int16)
    for i, event in enumerate(events):
        result[i] = (
            EVENT_KINDS.get(event.kind, 0),
            4 if event.seat is None else (event.seat - obs.stage.seat) % 4,
            34 if event.tile is None else event.tile,
            min(event.tiles) if event.kind == "chi" and event.tiles else 34,
            0,
        )
    return {
        "planes": base.planes.astype(np.float16),
        "scalars": base.scalars.astype(np.float16),
        "events": result,
        "mask": base.mask,
    }


def call_planes(row):
    """The v2 planes of a v1 row: the hand after each legal call or kong, looked ahead.

    Four groups of four planes hold what v1's per-discard planes hold (shanten
    (8-s)/9, accepted copies, winning copies, availability-weighted fan; from
    _fast.encode_lookahead) for the hand after the move:
      0-3   peng, at the claimed tile: the best discard after it
      4-7   chi, at the sequence's lowest tile: the best discard after it
      8-11  exposed kong, at its tile: the hand waiting for the replacement draw
      12-15 concealed or added kong, at its tile: likewise (an added kong
            turns an open peng into a kong, so no meld is added)
    Moves that are not legal leave their planes at 0. Everything comes from the
    row itself: the hand and the unseen copies from their count planes, the
    open melds from the hand's size, the fan state from the scalars (which hold
    the chain count up to 8 and the piao count up to 4), the moves from the mask.
    """
    out = np.zeros((CALL_PLANES, 34), dtype=np.float32)
    moves = np.flatnonzero(np.asarray(row["mask"])[CHI:HU]) + CHI
    if not len(moves):
        return out
    planes, scalars = row["planes"], row["scalars"]
    hand = (np.asarray(planes[P_HAND : P_HAND + 4]) > 0.5).sum(0)
    remaining = (np.asarray(planes[P_REMAINING : P_REMAINING + 4]) > 0.5).sum(0).tolist()
    size = int(hand.sum())
    opened = (14 - size) // 3 if size % 3 == 2 else (13 - size) // 3
    chain = round(float(scalars[_CHAIN]) * 8)
    piao = round(float(scalars[_PIAO]) * 4)
    you_cai = int(float(scalars[_YOU_CAI]) > 0.5)
    scratch = np.zeros((6, 34), dtype=np.float32)
    for index in moves:
        after, melds = hand.copy(), opened + 1
        if index < PENG:  # the two hand tiles of the sequence leave the hand
            suit, rest = divmod(int(index) - CHI, 21)
            start, position = divmod(rest, 3)
            tile = suit * 9 + start
            after[tile : tile + 3] -= 1
            after[tile + position] += 1
            group = 1
        elif index < GANG_EXPOSED:
            tile, group = int(index) - PENG, 0
            after[tile] -= 2
        elif index < GANG_CONCEALED:
            tile, group = int(index) - GANG_EXPOSED, 2
            after[tile] -= 3
        elif index < GANG_ADDED:
            tile, group = int(index) - GANG_CONCEALED, 3
            after[tile] -= 4
        else:
            tile, group, melds = int(index) - GANG_ADDED, 3, opened
            after[tile] -= 1
        if after.min() < 0 or melds > 4:
            continue
        tiles = np.repeat(np.arange(34), after).tolist()
        best = _fast.encode_lookahead(tiles, melds, remaining, chain, piao, you_cai, scratch)
        if best is None:
            # Past the winning boundary the native analysis declines; v1 then
            # takes the Python analysis (selfplay_core.encoding), and so does this.
            best = _analysis(tiles, melds, remaining, chain, piao, you_cai)
        shanten, ukeire, copies, value = best
        out[4 * group : 4 * group + 4, tile] = (
            (8 - shanten) / 9.0,
            min(ukeire, 40) / 40.0,
            min(copies, 12) / 12.0,
            math.log1p(value) / 8.0,
        )
    return out


def _analysis(tiles, opened, remaining, chain, piao, you_cai):
    """selfplay_core.encoding's Python lookahead: a 13-type hand, or its best discard."""
    from mahjong_ai.sdk.shape import analyze

    if len(tiles) == 13 - 3 * opened:
        rows = [analyze(tuple(tiles), opened, remaining, chain, piao, you_cai)]
    else:
        rows = []
        for tile in sorted(set(tiles)):
            hand = list(tiles)
            hand.remove(tile)
            rows.append(analyze(tuple(hand), opened, remaining, chain, piao, you_cai))
    best = min(rows, key=lambda r: (r["shanten"], -r["ukeire"], -r["fan_value"]))
    return best["shanten"], best["ukeire"], best["winning_copies"], best["fan_value"]


def augment(row, encoding):
    """A v1 row (or one already of ``encoding``) as the input of an Actor of ``encoding``."""
    want = inputs(encoding)[0]
    have = len(row["planes"])
    if have == want:
        return row
    if have == N_PLANES and encoding == ENCODING_V2:
        extra = call_planes(row).astype(np.float16)
        return dict(row, planes=np.concatenate([np.asarray(row["planes"], dtype=np.float16), extra]))
    raise ValueError(f"a row of {have} planes cannot become {encoding}")


def encode_v2(obs, lookahead=True):
    return augment(encode(obs, lookahead), ENCODING_V2)


def inputs(encoding):
    """(planes, scalars) of an Actor of ``encoding``."""
    if encoding not in INPUTS:
        raise ValueError(f"unknown observation encoding: {encoding}")
    return INPUTS[encoding]


def encoder(encoding):
    """The observation encoder of ``encoding``: encoder(name)(obs, lookahead) -> row."""
    inputs(encoding)
    return encode if encoding == ENCODING else encode_v2


def oracle(state, seat):
    from mahjong_ai.sdk.shape import shanten

    planes = np.zeros((16, 34), dtype=np.float32)
    scalars = np.zeros(4, dtype=np.float32)
    for i, tiles in enumerate(
        [state.hands[(seat + j) % 4] for j in (1, 2, 3)] + [state.wall]
    ):
        count = np.bincount(tiles, minlength=34)
        planes[4 * i : 4 * i + 4] = count[None, :] >= np.arange(1, 5)[:, None]
        if i < 3:
            opened = len(state.melds[(seat + i + 1) % 4])
            if len(tiles) in (13 - 3 * opened, 14 - 3 * opened):
                scalars[i] = (8 - shanten(tuple(sorted(tiles)), opened)) / 9
    scalars[3] = len(state.wall) / 136
    return {"oracle_planes": planes, "oracle_scalars": scalars}
