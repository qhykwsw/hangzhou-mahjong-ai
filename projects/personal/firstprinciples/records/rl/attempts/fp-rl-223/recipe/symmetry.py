"""Suit permutations of an encoded decision (wan, tong and tiao are interchangeable).

Relabelling the three suits maps a legal position to another legal position
with the same value, so a teacher-labelled row yields five more rows for free.
Tile-indexed planes permute their columns, except the constant suit/number
planes that describe the columns themselves; event tiles, the legal mask and
the action label follow the same relabelling. Honours and white are fixed.
"""

from itertools import permutations

import numpy as np
import torch

from selfplay_core.actions import CHI, GANG_ADDED, GANG_CONCEALED, GANG_EXPOSED, HU, N_ACTIONS, PENG
from selfplay_core.encoding import P_SUIT

SUIT_ORDERS = list(permutations(range(3)))


def maps(order):
    """(tile_map over 35 ids, action_map over the 231 actions) for one suit order."""
    tile = np.arange(35)
    for t in range(27):
        tile[t] = order[t // 9] * 9 + t % 9
    action = np.arange(N_ACTIONS)
    action[:34] = tile[:34]
    for a in range(CHI, PENG):
        suit, rest = divmod(a - CHI, 21)
        action[a] = CHI + order[suit] * 21 + rest
    for base in (PENG, GANG_EXPOSED, GANG_CONCEALED, GANG_ADDED):
        action[base : base + 33] = base + tile[:33]
    assert action[HU] == HU and sorted(action) == list(range(N_ACTIONS))
    return tile, action


class SuitPermuter:
    def __init__(self, device):
        self.tables = []
        for order in SUIT_ORDERS:
            tile, action = maps(order)
            self.tables.append((
                torch.as_tensor(np.argsort(tile[:34]), device=device),  # new column <- old column
                torch.as_tensor(tile, device=device),
                torch.as_tensor(np.argsort(action), device=device),  # new action <- old action
                torch.as_tensor(action, device=device),
            ))

    def __call__(self, batch, label, index):
        if index == 0:
            return batch, label
        column, tile, action_source, action = self.tables[index]
        planes = batch["planes"].index_select(2, column)
        planes[:, P_SUIT:] = batch["planes"][:, P_SUIT:]
        events = batch["events"].clone()
        for field in (2, 3):
            events[..., field] = tile[events[..., field].long()].to(events.dtype)
        out = dict(batch, planes=planes, events=events, mask=batch["mask"].index_select(1, action_source))
        return out, action[label]
