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

guided-current-hand-v3 appends V3_PLANES planes and V3_SCALARS scalars to v2
and gives the events' last field (always 0 before) a value, all again from the
v1 row alone (v3_inputs): per discard the plain and seven-pairs paths apart,
baotou waits and a solitary lookahead of the learner's next draws (native.c
v3_shape), the risk that a discard is called, kong follow-ups and decayed
rivers; whole-hand lookahead, win-or-continue, white-discard and timing
scalars; and per event the rule-forced discards (a catch round's draw-discard,
the discard after one's own chi or peng) with the event's age in turns. An
Actor of an earlier encoding reads the prefixes it was built for and the
events' last field as 0 (networks.HistoryEncoder).

guided-current-hand-v4 appends one scalar to v3: the stage's risk coefficient
beta / BETA_MAX in [-1, 1] (guided.stage_value: -V''/V' of the chance of
advancing, in the learner's own stage score). It is the one input that does
not come from the v1 row but from the observation's stage, so encode leaves
it on the row of a semifinal or final observation as ``risk_beta`` (no key
elsewhere: an input of 0) and augment appends it; the C engine's rows get it
with the stage's other scalars (guided.stage.rewrite), and match workers'
binary requests carry it (selfplay_core.wire).
"""

import math

import numpy as np

from mahjong_ai.domain.types import StageKind
from selfplay_core.actions import CHI, GANG_ADDED, GANG_CONCEALED, GANG_EXPOSED, HU, PENG
from selfplay_core.encoding import (
    EVENT_KINDS,
    N_PLANES,
    N_SCALARS,
    P_DISC_SHANTEN,
    P_DRAWN,
    P_HAND,
    P_MELDS,
    P_REMAINING,
    encode as base_encode,
)

from .stage_value import risk_of

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
ENCODING_V3 = "guided-current-hand-v3"
ENCODING_V4 = "guided-current-hand-v4"
CALL_PLANES = 16
V3_PLANES = 38
V3_SCALARS = 35
# v4's scalar, after v3's: beta / BETA_MAX (guided.stage_value).
RISK = N_SCALARS + V3_SCALARS
# (planes, scalars) an Actor of each encoding takes.
INPUTS = {
    ENCODING: (N_PLANES, N_SCALARS),
    ENCODING_V2: (N_PLANES + CALL_PLANES, N_SCALARS),
    ENCODING_V3: (N_PLANES + CALL_PLANES + V3_PLANES, N_SCALARS + V3_SCALARS),
    ENCODING_V4: (N_PLANES + CALL_PLANES + V3_PLANES, RISK + 1),
}
# Rows of the events' last field an Actor of each encoding embeds: 0 before
# v3; v3 and v4 hold forced * 8 + age bucket (v3_inputs).
EVENT_TAILS = {ENCODING: 4, ENCODING_V2: 4, ENCODING_V3: 24, ENCODING_V4: 24}

# selfplay_core's scalars: the score context (11), the horizon (4 + stage
# kinds), the seats (4 dealer, 4 god discarder, none, catch round, restricted),
# the wall and phase (6), the own fan state (8), then melds and hand sizes.
_SEATS = 11 + 4 + len(StageKind)
_CATCH, _RESTRICTED = _SEATS + 9, _SEATS + 10
_WALL = _SEATS + 11
_PAST_DEAD = _WALL + 1
_CHAIN = _WALL + 6
_PIAO, _HU_FAN, _YOU_CAI = _CHAIN + 1, _CHAIN + 2, _CHAIN + 4
_CHIS = _CHAIN + 8 + 4


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
    row = {
        "planes": base.planes.astype(np.float16),
        "scalars": base.scalars.astype(np.float16),
        "events": result,
        "mask": base.mask,
    }
    risk = risk_of(obs.stage, obs.base_score)
    if risk is not None:  # a semifinal or final: v4's scalar (augment)
        row["risk_beta"] = risk
    return row


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


_native = None


def native():
    """guided/native.c (built once per machine by _fast.load_native), for v3_shape."""
    global _native
    if _native is None:
        _native = _fast.load_native(__file__)
    return _native


def v3_inputs(row):
    """The v3 additions of a v1 (or wider) row: (planes [38, 34], scalars [35], event tails).

    Planes, column = tile:
      0-27  per discard of a hand that discards (native.c v3_shape): 0-2 the
            plain path's (8 - shanten) / 9, accepted copies / 40, winning
            copies / 12; 3-5 the same for seven pairs (0 with melds open);
            6 the waiting hand is a baotou wait; 7 draws after which some
            discard makes a baotou wait / 40; 8-27 for k in (1, 2, 3, 5, own
            draws left), P(ready), P(baotou wait), P(self-drawn win) and
            log2(1 + E[fan | win]) / 9 within k of the learner's own draws,
            drawn uniformly from the tiles it has not seen, other seats ignored
            (ready: exact hypergeometric; one away: each readying draw then its
            best discard by winning copies, walked for the three best discards
            in v1's order; others and two or more away: improving draws at the
            first one's rate, then the walked discards' mean winning copies and
            fan, or the ready ones', or 4 copies at fan 1)
      28    whites short of four once this tile goes, piao discards counted: / 4
      29    chi risk: the next seat may still chi (fewer than two chis) times
            the unseen pairs completing a sequence with this tile / 16, at most 1
      30    peng risk: (unseen copies - 1) / 3, at least 0
      31-33 per legal kong (exposed, concealed or added), at its tile: P(the
            replacement draw wins), the chain after it / 8, P(the replacement
            draw allows another kong)
      34-37 per relative seat, its river decayed: sum of exp(-0.2 * turns since)
            over this hand's discards of the tile it still shows
    No plane holds a white's risk or a non-discarding hand's per-discard values.
    Scalars:
      0-19  the 8-27 lookahead of the reference hand (the best discard in v1's
            order, or a waiting hand itself)
      20    own draws left before the 20-tile end / 20
      21-24 a win is legal: 1; log2(fan) / 9; continuing's log1p(P(win) * E[fan])
            / 8 and P(win) within the draws left, from the reference
      25    a white held while discarding: (shanten letting it go - the best
            otherwise) / 2, clipped to [-1, 1]
      26-28 letting a white go now would be a piao (the hand without it waits on
            baotou, and so did the hand before the draw): 1; the chain then / 8;
            piao then / 4
      29    turns this hand (discards / 4) / 20
      30-33 per relative seat, turns since its last chi, peng or kong this hand
            / 20 (1 without one)
      34    the learner discards after its own chi or peng
    Event tails: forced * 8 + the age bucket of the turns since the event (0, 1,
    2, 3, 4, 5-6, 7-9, 10+), forced 1 for a discard in a catch round by a seat
    other than its owner (it can only let the drawn tile go), 2 for the discard
    right after the same seat's chi or peng (it cannot be the drawn tile), else 0.
    """
    planes = np.asarray(row["planes"], dtype=np.float32)
    scalars = np.asarray(row["scalars"], dtype=np.float32)
    events = np.ascontiguousarray(np.asarray(row["events"], dtype=np.int16).reshape(-1, 5))
    mask = np.asarray(row["mask"])
    out = np.zeros((V3_PLANES, 34), dtype=np.float32)
    extra = np.zeros(V3_SCALARS, dtype=np.float32)
    hand = (planes[P_HAND : P_HAND + 4] > 0.5).sum(0).astype(np.int64)
    unseen = (planes[P_REMAINING : P_REMAINING + 4] > 0.5).sum(0).astype(np.int64)
    size = int(hand.sum())
    discarding = size % 3 == 2
    opened = (14 - size) // 3 if discarding else (13 - size) // 3
    chain = round(float(scalars[_CHAIN]) * 8)
    piao = round(float(scalars[_PIAO]) * 4)
    you_cai = int(float(scalars[_YOU_CAI]) > 0.5)

    # This hand's events (native.c v3_events): tails, decayed rivers, calls.
    tails = np.zeros(len(events), dtype=np.int16)
    rivers = np.zeros((4, 34), dtype=np.float32)
    discards, last_discarder, since_call, after_call = native().v3_events(events, tails, rivers)
    out[34:38] = rivers

    # Own draws left: the hand ends with 20 tiles in the wall; draws go round
    # from the seat after the last discarder (after the learner when it discards).
    left = round(float(scalars[_PAST_DEAD]) * 116)
    if discarding:
        draws = left // 4
    else:
        first = (-(max(last_discarder, 0) + 1)) % 4 + 1
        draws = 0 if left < first else (left - first) // 4 + 1
    shape = np.zeros((29, 34), dtype=np.float32)
    reference = np.zeros(20, dtype=np.float32)
    if size in (13 - 3 * opened, 14 - 3 * opened) and 0 <= opened <= 4:
        native().v3_shape(hand.tolist(), opened, unseen.tolist(), chain, piao, you_cai, draws, shape, reference)
    out[:28] = shape[:28]
    extra[:20] = reference
    extra[20] = min(draws, 20) / 20.0

    # Whites short of four and the risk that a discard is called.
    whites = int(hand[33])
    if discarding:
        for t in np.flatnonzero(hand):
            out[28, t] = (4 - min(4, whites - (t == 33) + piao)) / 4.0
    if round(float(scalars[_CHIS + 1]) * 2) < 2:
        u = np.zeros((3, 13), dtype=np.int64)
        u[:, 2:11] = unseen[:27].reshape(3, 9)
        pairs = u[:, 0:9] * u[:, 1:10] + u[:, 1:10] * u[:, 3:12] + u[:, 3:12] * u[:, 4:13]
        out[29, :27] = np.minimum(pairs.reshape(27) / 16.0, 1.0)
    out[30, :33] = np.maximum(unseen[:33] - 1, 0) / 3.0

    # Kong follow-ups.
    pool = int(unseen.sum())
    own_pengs = (planes[P_MELDS + 2] > 0.5) & (planes[P_MELDS + 3] < 0.5)
    for index in np.flatnonzero(mask[GANG_EXPOSED:HU]) + GANG_EXPOSED:
        tile = (int(index) - GANG_EXPOSED) % 33
        kind = (int(index) - GANG_EXPOSED) // 33
        rest, melds = hand.copy(), opened + (kind != 2)
        rest[tile] -= (3, 4, 1)[kind]
        if rest.min() < 0 or melds > 4 or int(rest.sum()) != 13 - 3 * melds:
            continue
        wins = _fast.analyze_counts(rest.tolist(), melds, unseen.tolist(), chain, piao, you_cai)[5]
        again = sum(int(unseen[t]) for t in range(33) if t != tile and (rest[t] == 3 or own_pengs[t]))
        out[31, tile] = wins / pool if pool else 0.0
        out[32, tile] = min(chain + 1, 8) / 8.0
        out[33, tile] = min(again / pool, 1.0) if pool else 0.0

    # Win or continue, the white's place, piao, turns and calls.
    if mask[HU]:
        # Fans are powers of two: v1's log2(fan) / 9 rounds back exactly.
        extra[21] = 1.0
        extra[22] = round(float(scalars[_HU_FAN]) * 9) / 9.0
        win_p = float(reference[18])
        mean_fan = 2.0 ** (float(reference[19]) * 9) - 1.0
        extra[23] = math.log1p(win_p * mean_fan) / 8.0
        extra[24] = win_p
    if discarding and whites:
        shanten = np.maximum(8 - np.round(planes[P_DISC_SHANTEN] * 9), 0)
        others = [shanten[t] for t in np.flatnonzero(hand) if t != 33]
        if others:
            extra[25] = float(np.clip((shanten[33] - min(others)) / 2.0, -1.0, 1.0))
    if whites:
        held = np.repeat(np.arange(34), hand).tolist()
        rest = list(held)
        rest.remove(33)
        drawn = np.flatnonzero(planes[P_DRAWN] > 0.5)
        piao_now = bool(_fast.is_baotou(rest, opened))
        if piao_now and len(drawn):
            before = list(held)
            before.remove(int(drawn[0]))
            piao_now = bool(_fast.is_baotou(before, opened))
        if piao_now:
            extra[26] = 1.0
            extra[27] = min(chain + 1, 8) / 8.0
            extra[28] = min(piao + 1, 4) / 4.0
    extra[29] = min(discards / 4.0 / 20.0, 1.0)
    for seat, since in enumerate(since_call):
        extra[30 + seat] = 1.0 if since < 0 else min(since / 4.0 / 20.0, 1.0)
    extra[34] = float(after_call)
    return out, extra, tails


def augment(row, encoding):
    """A v1 row (or one already of ``encoding``) as the input of an Actor of ``encoding``.

    A row of a wider encoding than ``encoding`` is returned as it is: an Actor
    reads the prefixes it was built for (networks.HistoryEncoder). v4's scalar
    is the row's ``risk_beta`` (encode), 0 without one.
    """
    want = inputs(encoding)
    have = (len(row["planes"]), len(row["scalars"]))
    if have[0] >= want[0] and have[1] >= want[1]:
        return row
    if have == INPUTS[ENCODING_V3]:  # only v4 is wider
        return dict(row, scalars=np.concatenate([np.asarray(row["scalars"], dtype=np.float16), _risk(row)]))
    v1 = row if have[0] == N_PLANES else dict(row, planes=np.asarray(row["planes"])[:N_PLANES])
    if have[0] not in (N_PLANES, N_PLANES + CALL_PLANES) or have[1] != N_SCALARS:
        raise ValueError(f"a row of {have[0]} planes cannot become {encoding}")
    planes = [np.asarray(row["planes"], dtype=np.float16)]
    if have[0] == N_PLANES:
        planes.append(call_planes(v1).astype(np.float16))
    if encoding == ENCODING_V2:
        return dict(row, planes=np.concatenate(planes))
    added, extra, tails = v3_inputs(v1)
    planes.append(added.astype(np.float16))
    events = np.array(row["events"], dtype=np.int16).reshape(-1, 5)
    events[:, 4] = tails
    scalars = [np.asarray(row["scalars"], dtype=np.float16), extra.astype(np.float16)]
    if encoding == ENCODING_V4:
        scalars.append(_risk(row))
    return dict(row, planes=np.concatenate(planes), scalars=np.concatenate(scalars), events=events)


def _risk(row):
    """v4's scalar of a row: its risk_beta, 0 without one."""
    return np.asarray([row.get("risk_beta", 0.0)], dtype=np.float16)


def encode_v2(obs, lookahead=True):
    return augment(encode(obs, lookahead), ENCODING_V2)


def encode_v3(obs, lookahead=True):
    return augment(encode(obs, lookahead), ENCODING_V3)


def encode_v4(obs, lookahead=True):
    return augment(encode(obs, lookahead), ENCODING_V4)


def inputs(encoding):
    """(planes, scalars) of an Actor of ``encoding``."""
    if encoding not in INPUTS:
        raise ValueError(f"unknown observation encoding: {encoding}")
    return INPUTS[encoding]


def encoder(encoding):
    """The observation encoder of ``encoding``: encoder(name)(obs, lookahead) -> row."""
    inputs(encoding)
    return {ENCODING: encode, ENCODING_V2: encode_v2, ENCODING_V3: encode_v3, ENCODING_V4: encode_v4}[encoding]


def widest(encodings):
    """The encoding whose rows every one of ``encodings`` can read (later ones extend earlier ones)."""
    order = list(INPUTS)
    return max(encodings, key=order.index)


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
