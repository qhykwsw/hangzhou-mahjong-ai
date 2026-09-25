"""Observation -> network inputs.  Pure numpy; identical in training and release.

Three tensors are produced for every decision:
  planes  float32 (N_PLANES, 34)   tile-indexed features
  scalars float32 (N_SCALARS,)     game-state features (no base score: everything
                                   is expressed in units of the room base score)
  events  int16   (K_EVENTS, 5)    the most recent public events, right-aligned:
                                   kind, relative seat, tile, aux tile, hand offset
plus the legal-action mask over the fixed 231-way action index.
"""

from dataclasses import dataclass
import math

import numpy as np

from mahjong_ai.domain.rules import WHITE, counts
from mahjong_ai.domain.types import Observation, StageKind
from mahjong_ai.sdk.shape import analyze, remaining_tiles, shanten

from .actions import N_ACTIONS, legal_indices

try:
    from mahjong_ai import _fast
except ImportError:
    _fast = None

ENCODING_VERSION = "selfplay-obs-v1"
K_EVENTS = 64
EVENT_FIELDS = 5
EVENT_KINDS = {"draw": 1, "discard": 2, "chi": 3, "peng": 4, "gang": 5, "hand_ended": 6}
NO_SEAT = 4
NO_TILE = 34

# ---- plane layout -----------------------------------------------------------
P_HAND = 0  # 4: own hand count >= 1..4
P_DRAWN = 4  # 1
P_MELDS = 5  # 16: per relative seat, meld tile count >= 1..4
P_DISCARDS = 21  # 16: per relative seat, river count >= 1..4
P_LAST_DISCARD = 37  # 4: per relative seat, most recent river tile
P_REMAINING = 41  # 4: unseen copies >= 1..4
P_DISC_SHANTEN = 45  # 1: shanten after discarding this tile (draw phase)
P_DISC_UKEIRE = 46  # 1: accepted copies after discarding this tile
P_DISC_WINCOPIES = 47  # 1: winning copies after discarding this tile
P_DISC_FANVALUE = 48  # 1: availability-weighted fan after discarding this tile
P_IMPROVING = 49  # 1: improving tiles of the reference 13-tile hand
P_WINNING = 50  # 1: winning tiles of the reference 13-tile hand
P_LEGAL_DISCARD = 51  # 1
P_SUIT = 52  # 4: constant suit one-hot (wan, tong, tiao, honor)
P_NUMBER = 56  # 1: constant number position within suit
N_PLANES = 57

_THRESH = np.arange(1, 5, dtype=np.float32)[:, None]
_CONST = np.zeros((5, 34), dtype=np.float32)
for _t in range(34):
    _CONST[min(_t // 9, 3), _t] = 1.0
    _CONST[4, _t] = ((_t % 9) / 8.0) if _t < 27 else 0.5


def symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


@dataclass
class Encoded:
    planes: np.ndarray
    scalars: np.ndarray
    events: np.ndarray
    mask: np.ndarray
    legal: np.ndarray  # action index of each legal action, in legal_actions order
    shanten: int  # reference-hand shanten, reused for reward shaping


def _count_planes(target, tiles):
    if tiles:
        c = np.bincount(np.fromiter(tiles, dtype=np.int64, count=len(tiles)), minlength=34)
        target[:] = c[None, :] >= _THRESH


def encode(obs: Observation, lookahead: bool = True, *, include_events: bool = True) -> Encoded:
    seat = obs.stage.seat
    rel = [(seat + i) % 4 for i in range(4)]  # rel[i] = physical seat at relative offset i
    planes = np.zeros((N_PLANES, 34), dtype=np.float32)
    hand = obs.hand
    opened = len(obs.melds[seat])

    if _fast is not None:
        remaining = _fast.encode_public_planes(
            hand, obs.discards, obs.melds, obs.drawn_tile, seat,
            planes[:P_REMAINING + 4],
        )
    else:
        _count_planes(planes[P_HAND : P_HAND + 4], hand)
        if obs.drawn_tile is not None:
            planes[P_DRAWN, obs.drawn_tile] = 1.0
        for i, s in enumerate(rel):
            meld_tiles = [t for m in obs.melds[s] for t in m.tiles]
            _count_planes(planes[P_MELDS + 4 * i : P_MELDS + 4 * i + 4], meld_tiles)
            river = obs.discards[s]
            _count_planes(planes[P_DISCARDS + 4 * i : P_DISCARDS + 4 * i + 4], river)
            if river:
                planes[P_LAST_DISCARD + i, river[-1]] = 1.0
        remaining = remaining_tiles(obs)
        planes[P_REMAINING : P_REMAINING + 4] = (
            np.asarray(remaining, dtype=np.float32)[None, :] >= _THRESH
        )
    planes[P_SUIT : P_SUIT + 5] = _CONST

    legal = legal_indices(obs.legal_actions)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[legal] = True
    planes[P_LEGAL_DISCARD] = mask[:34]

    # ---- shape lookahead ---------------------------------------------------
    ref_shanten, ref_ukeire, ref_wincopies, ref_fanvalue = 8, 0, 0, 0.0
    native_ref = None
    if lookahead and _fast is not None:
        native_ref = _fast.encode_lookahead(
            hand, opened, remaining, obs.chain_count, obs.piao_count,
            int(obs.you_cai_bi_kao), planes[P_DISC_SHANTEN : P_WINNING + 1],
        )
    if native_ref is not None:
        ref_shanten, ref_ukeire, ref_wincopies, ref_fanvalue = native_ref
    elif lookahead:
        discard_phase = len(hand) == 14 - 3 * opened
        if discard_phase:
            best, best_key = None, None
            hand_list = list(hand)
            for tile in sorted(set(hand)):
                h = hand_list.copy()
                h.remove(tile)
                row = analyze(
                    tuple(h),
                    opened,
                    remaining,
                    obs.chain_count,
                    obs.piao_count,
                    obs.you_cai_bi_kao,
                )
                planes[P_DISC_SHANTEN, tile] = (8 - row["shanten"]) / 9.0
                planes[P_DISC_UKEIRE, tile] = min(row["ukeire"], 40) / 40.0
                planes[P_DISC_WINCOPIES, tile] = min(row["winning_copies"], 12) / 12.0
                planes[P_DISC_FANVALUE, tile] = math.log1p(row["fan_value"]) / 8.0
                key = (row["shanten"], -row["ukeire"], -row["fan_value"])
                if best_key is None or key < best_key:
                    best, best_key = row, key
            ref = best
        elif len(hand) == 13 - 3 * opened:
            ref = analyze(
                tuple(hand),
                opened,
                remaining,
                obs.chain_count,
                obs.piao_count,
                obs.you_cai_bi_kao,
            )
        else:
            ref = None
        if ref is not None:
            ref_shanten = ref["shanten"]
            ref_ukeire = ref["ukeire"]
            ref_wincopies = ref["winning_copies"]
            ref_fanvalue = ref["fan_value"]
            for tile in ref["improving"]:
                planes[P_IMPROVING, tile] = remaining[tile] / 4.0
            for tile, fan in ref["fans"].items():
                planes[P_WINNING, tile] = math.log2(fan + 1) / 9.0
    else:
        if len(hand) in (13 - 3 * opened, 14 - 3 * opened):
            ref_shanten = shanten(tuple(sorted(hand)), opened)

    # ---- scalars -------------------------------------------------------------
    ctx = obs.stage
    scores = ctx.effective_scores
    own = scores[seat]
    ranks = _midranks(scores)
    hc = counts(hand)
    legal_kinds = {a.action for a in obs.legal_actions}
    current, total = ctx.effective_horizon
    whites_visible = sum(1 for row in obs.discards for t in row if t == WHITE) + sum(
        1 for row in obs.melds for m in row for t in m.tiles if t == WHITE
    )
    f = [
        # score context, in units of the room base score
        *(symlog(scores[s] / 8.0) / 4.0 for s in rel),
        *(symlog((own - scores[s]) / 8.0) / 4.0 for s in rel[1:]),
        (ranks[seat] - 1) / 3.0,
        *((ranks[s] - 1) / 3.0 for s in rel[1:]),
        # horizon
        min(1.0, max(0.0, (current - 1) / max(1, total - 1))),
        min(1.0, max(0, total - current) / 16.0),
        min(1.0, total / 20.0),
        float(ctx.in_tiebreak),
        *(float(ctx.kind == k) for k in StageKind),
        # seats
        *(float(ctx.dealer == s) for s in rel),
        *(float(obs.god_discarder_seat == s) for s in rel),
        float(obs.god_discarder_seat < 0),
        float(obs.catch_play),
        float(obs.catch_restricted),
        # wall & phase
        obs.wall_remaining / 136.0,
        max(0, obs.wall_remaining - 20) / 116.0,
        *(float(obs.phase == p) for p in ("draw", "response_peng", "response_chi")),
        float(obs.drawn_tile is not None),
        # own fan state
        min(obs.chain_count, 8) / 8.0,
        min(obs.piao_count, 4) / 4.0,
        math.log2(max(1, obs.hu_fan)) / 9.0,
        float(obs.hu_fan > 0),
        float(obs.you_cai_bi_kao),
        hc[WHITE] / 4.0,
        whites_visible / 4.0,
        remaining[WHITE] / 4.0,
        # melds and hand sizes
        *(len(obs.melds[s]) / 4.0 for s in rel),
        *(sum(m.kind == "chi" for m in obs.melds[s]) / 2.0 for s in rel),
        *(sum(m.kind == "gang" for m in obs.melds[s]) / 4.0 for s in rel),
        *(sum(m.concealed for m in obs.melds[s]) / 4.0 for s in rel),
        *(obs.hand_sizes[s] / 14.0 for s in rel),
        *(len(obs.discards[s]) / 24.0 for s in rel),
        # shape summary
        (8 - ref_shanten) / 9.0,
        float(ref_shanten <= 0),
        min(ref_ukeire, 40) / 40.0,
        min(ref_wincopies, 12) / 12.0,
        math.log1p(ref_fanvalue) / 8.0,
        # legal action summary
        len(obs.legal_actions) / 16.0,
        *(float(k in legal_kinds) for k in ("hu", "discard", "chi", "peng", "gang", "pass")),
    ]
    scalars = np.asarray(f, dtype=np.float32)

    # ---- events --------------------------------------------------------------
    events = np.zeros((K_EVENTS if include_events else 0, EVENT_FIELDS), dtype=np.int16)
    recent = obs.public_events[-K_EVENTS:] if include_events else ()
    current_hand = ctx.round_no
    offset = K_EVENTS - len(recent)
    for i, e in enumerate(recent):
        row = events[offset + i]
        row[0] = EVENT_KINDS.get(e.kind, 0)
        row[1] = NO_SEAT if e.seat is None else (e.seat - seat) % 4
        row[2] = NO_TILE if e.tile is None else e.tile
        if e.kind == "chi" and e.tiles:
            row[3] = min(e.tiles)
        else:
            row[3] = NO_TILE
        row[4] = min(3, max(0, current_hand - e.hand_no))
    return Encoded(planes, scalars, events, mask, legal, ref_shanten)


def _midranks(scores):
    return tuple(
        1 + sum(v > x for v in scores) + (sum(v == x for v in scores) - 1) / 2
        for x in scores
    )


N_SCALARS = None


def _probe_scalar_count():
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig

    return int(encode(HandState(RoundConfig(), 1).observation(), lookahead=False).scalars.shape[0])


N_SCALARS = _probe_scalar_count()
