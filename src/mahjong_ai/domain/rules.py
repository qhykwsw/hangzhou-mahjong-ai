"""Local rules calibrated against the official server and published guide.

This module is pure: no wall, RNG, opponent state, network, or policy code.
"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

try:  # Optional C accelerator; the pure-Python reference below stays authoritative.
    from mahjong_ai import _fast
except ImportError:  # pragma: no cover - exercised by running without the extension
    _fast = None

WHITE = 33
CODES = tuple(f"{n}{s}" for s in "wbt" for n in range(1, 10)) + tuple("东南西北中发白")
CODE_TO_TILE = {s: i for i, s in enumerate(CODES)}


def tile_code(t: int) -> str:
    return CODES[t]


def tile_id(s: str) -> int:
    if s not in CODE_TO_TILE:
        raise ValueError(f"unknown tile: {s}")
    return CODE_TO_TILE[s]


@lru_cache(maxsize=200000)
def _counts(hand: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * 34
    for t in hand:
        if not isinstance(t, int) or not 0 <= t < 34:
            raise ValueError(f"invalid tile: {t}")
        result[t] += 1
    return tuple(result)


def counts(hand):
    # 手牌在搜索里反复出现：元组直接走缓存，列表仍逐张校验。
    if type(hand) is tuple:
        return _counts(hand)
    result = [0] * 34
    for t in hand:
        if not isinstance(t, int) or not 0 <= t < 34:
            raise ValueError(f"invalid tile: {t}")
        result[t] += 1
    return tuple(result)


@lru_cache(maxsize=300000)
def _meldable(c: tuple[int, ...], jokers: int) -> bool:
    try:
        first = next(i for i, n in enumerate(c) if n)
    except StopIteration:
        return jokers % 3 == 0
    # Using fewer natural tiles can leave a tile for another sequence.
    for take in range(min(c[first], 3), 0, -1):
        need = 3 - take
        if need <= jokers:
            x = list(c)
            x[first] -= take
            if _meldable(tuple(x), jokers - need):
                return True
    if first < 27:
        suit = first // 9 * 9
        for start in range(max(suit, first - 2), min(first, suit + 6) + 1):
            x = list(c)
            missing = 0
            for t in range(start, start + 3):
                if x[t]:
                    x[t] -= 1
                else:
                    missing += 1
            if missing <= jokers and _meldable(tuple(x), jokers - missing):
                return True
    return False


@lru_cache(maxsize=200000)
def _shape_accel(c: tuple[int, ...], open_melds: int) -> bool:
    return _fast.can_hu(c, open_melds)


def _shape(c: tuple[int, ...], open_melds: int) -> bool:
    """Reference winning-shape test; the accelerator mirrors it exactly."""
    # The reference path treats overfull physical hands as ordinary input and
    # returns a boolean.  Keep that public behavior instead of leaking the C
    # parser's stricter 0..4 count error for malformed callers.
    if _fast is not None and max(c, default=0) <= 4:
        return _shape_accel(c, open_melds)
    return _shape_reference(c, open_melds)


@lru_cache(maxsize=200000)
def _shape_reference(c: tuple[int, ...], open_melds: int) -> bool:
    if not 0 <= open_melds <= 4 or sum(c) != 14 - 3 * open_melds:
        return False
    j = c[WHITE]
    x = c[:WHITE] + (0,)
    if open_melds == 0 and sum(n % 2 for n in x) <= j:
        return True
    if j >= 2 and _meldable(x, j - 2):
        return True
    for i, n in enumerate(x):
        for take in range(min(2, n), 0, -1):
            need = 2 - take
            if need <= j:
                y = list(x)
                y[i] -= take
                if _meldable(tuple(y), j - need):
                    return True
    return False


def can_hu(hand, open_melds=0) -> bool:
    return _shape(counts(hand), open_melds)


@lru_cache(maxsize=100000)
def _baotou(c: tuple[int, ...], open_melds: int) -> bool:
    if sum(c) != 13 - 3 * open_melds or c[WHITE] == 0:
        return False
    # Test isolated honors early; most hands fail immediately.
    for t in (*range(27, 34), *range(27)):
        if c[t] >= 4:  # no physical fifth copy can be drawn
            continue
        x = list(c)
        x[t] += 1
        if not _shape(tuple(x), open_melds):
            return False
    return True


def is_baotou(waiting_hand, open_melds=0) -> bool:
    return _baotou(counts(waiting_hand), open_melds)


@dataclass(frozen=True)
class Fan:
    fan: int
    baotou: bool
    detail: tuple[str, ...]


def calculate_fan(
    hand,
    drawn_tile,
    open_melds=0,
    chain_count=0,
    piao_count=0,
    you_cai_bi_kao=False,
    kong_draw=False,
):
    # Fan calculation is requested repeatedly while a policy prices waits.
    # Normalize list callers at this boundary, then cache the immutable core.
    return _calculate_fan_cached(
        tuple(hand),
        drawn_tile,
        open_melds,
        chain_count,
        piao_count,
        you_cai_bi_kao,
        kong_draw,
    )


@lru_cache(maxsize=200000)
def _calculate_fan_cached(
    hand,
    drawn_tile,
    open_melds=0,
    chain_count=0,
    piao_count=0,
    you_cai_bi_kao=False,
    kong_draw=False,
):
    if drawn_tile is None or drawn_tile not in hand or not can_hu(hand, open_melds):
        return None
    waiting = list(hand)
    waiting.remove(drawn_tile)
    bt = is_baotou(waiting, open_melds)
    c = counts(hand)
    # Live v34 rejects non-baotou white hands even on a kong replacement.
    # The guide's kong exemption is contradicted by captured rejection/timeout
    # evidence (tests/fixtures/official-v34-youcai-rejected-hu.json).
    # kong_draw records context; the multiplier comes from chain_count.
    if you_cai_bi_kao and c[WHITE] and not bt:
        return None
    detail = []
    fan = 1
    if open_melds == 0 and sum(n % 2 for n in c[:WHITE]) <= c[WHITE]:
        luxury = sum(n == 4 for n in c[:WHITE])
        if c[WHITE] == 4 and not any(n % 2 for n in c[:WHITE]):
            luxury += 1
        fan = 2 ** (1 + luxury)
        detail.append(("七对", "豪华七对", "双豪华七对", "三豪华七对")[min(3, luxury)])
    else:
        detail.append("平胡")
    if chain_count:
        fan *= 2**chain_count
        detail.append(f"动作链{chain_count}")
    if c[WHITE] + piao_count == 4:
        fan *= 2
        detail.append("4个白板")
    if bt:
        fan *= 2
        detail.append("爆头")
    return Fan(fan, bt, tuple(detail))


def payments(winner: int, dealer: int, fan: int, base=1) -> tuple[int, ...]:
    if not 0 <= winner < 4 or not 0 <= dealer < 4 or fan < 1 or base < 1:
        raise ValueError("invalid settlement")
    scores = [0] * 4
    for seat in range(4):
        if seat != winner:
            loss = base * fan * (8 if dealer in (seat, winner) else 1)
            scores[seat] = -loss
            scores[winner] += loss
    return tuple(scores)


@lru_cache(maxsize=200000)
def hand_potential(hand: tuple[int, ...], open_melds=0) -> float:
    """Cheap public-information heuristic, NOT a hu/shanten oracle."""
    c = counts(hand)
    value = c[WHITE] * 2.5 + open_melds * 3.0
    for t, n in enumerate(c[:WHITE]):
        if n >= 2:
            value += 1.7
        if n >= 3:
            value += 1.3
        if t < 27:
            if t % 9 < 8:
                value += min(n, c[t + 1]) * 0.7
            if t % 9 < 7:
                value += min(n, c[t + 2]) * 0.35
    return value
