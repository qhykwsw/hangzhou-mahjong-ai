"""Fixed 231-way action index shared by training and release inference.

Layout (all tile ids 0..33, white = 33 can only be discarded):
  0..33    discard tile
  34..96   chi: suit*21 + sequence_start*3 + position_of_claimed_tile
  97..129  peng tile (0..32)
  130..162 exposed kong tile
  163..195 concealed kong tile
  196..228 added kong tile
  229      hu
  230      pass
"""

import numpy as np

from mahjong_ai.domain.types import Action

N_TILES = 34
DISCARD = 0
CHI = 34
PENG = 97
GANG_EXPOSED = 130
GANG_CONCEALED = 163
GANG_ADDED = 196
HU = 229
PASS = 230
N_ACTIONS = 231

_GANG_BASE = {"exposed": GANG_EXPOSED, "concealed": GANG_CONCEALED, "added": GANG_ADDED}


def action_index(action: Action) -> int:
    kind = action.action
    if kind == "discard":
        return DISCARD + action.tile
    if kind == "chi":
        low = min(action.tiles + (action.tile,))
        suit, start = divmod(low, 9)
        if suit > 2 or start > 6:
            raise ValueError(f"invalid chi {action}")
        return CHI + suit * 21 + start * 3 + (action.tile - low)
    if kind == "peng":
        if action.tile >= 33:
            raise ValueError("white cannot be claimed")
        return PENG + action.tile
    if kind == "gang":
        if action.tile >= 33:
            raise ValueError("white cannot be konged")
        return _GANG_BASE[action.kind] + action.tile
    if kind == "hu":
        return HU
    if kind == "pass":
        return PASS
    raise ValueError(f"unknown action {action}")


def describe_index(index: int) -> Action:
    """Inverse of action_index; used by tests and debugging output."""
    if index < 0 or index >= N_ACTIONS:
        raise ValueError("action index out of range")
    if index < CHI:
        return Action("discard", index)
    if index < PENG:
        rest = index - CHI
        suit, rest = divmod(rest, 21)
        start, pos = divmod(rest, 3)
        low = suit * 9 + start
        tile = low + pos
        return Action("chi", tile, tuple(t for t in (low, low + 1, low + 2) if t != tile))
    if index < GANG_EXPOSED:
        return Action("peng", index - PENG)
    if index < GANG_CONCEALED:
        return Action("gang", index - GANG_EXPOSED, kind="exposed")
    if index < GANG_ADDED:
        return Action("gang", index - GANG_CONCEALED, kind="concealed")
    if index < HU:
        return Action("gang", index - GANG_ADDED, kind="added")
    return Action("hu") if index == HU else Action("pass")


def legal_indices(legal_actions) -> np.ndarray:
    return np.fromiter((action_index(a) for a in legal_actions), dtype=np.int64, count=len(legal_actions))


def legal_mask(legal_actions) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[legal_indices(legal_actions)] = True
    return mask
