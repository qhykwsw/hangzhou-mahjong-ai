"""Symmetries of an encoded decision: suit permutations and the number reflection.

Relabelling the three suits (wan, tong, tiao), and reading every suit
backwards (1<->9, 2<->8, ...), map a legal position to another legal position
with the same value: no rule looks at a suit's name or at a tile's number.
A teacher-labelled row therefore yields eleven more rows for free.
Tile-indexed planes permute their columns, except the constant suit/number
planes that describe the columns themselves; event tiles, the legal mask and
the action label follow the same relabelling. Honours and white are fixed.
"""

from itertools import permutations

import numpy as np
import torch

from selfplay_core.actions import CHI, GANG_ADDED, GANG_CONCEALED, GANG_EXPOSED, HU, N_ACTIONS, PENG
from selfplay_core.encoding import EVENT_KINDS, P_SUIT

TRANSFORMS = [(order, reflect) for reflect in (False, True) for order in permutations(range(3))]
CHI_EVENT = EVENT_KINDS["chi"]


def maps(order, reflect):
    """(tile_map over 35 ids, action_map over the 231 actions) for one transform."""
    tile = np.arange(35)
    for t in range(27):
        number = t % 9
        tile[t] = order[t // 9] * 9 + (8 - number if reflect else number)
    action = np.arange(N_ACTIONS)
    action[:34] = tile[:34]
    for a in range(CHI, PENG):
        suit, rest = divmod(a - CHI, 21)
        start, position = divmod(rest, 3)
        if reflect:
            start, position = 6 - start, 2 - position
        action[a] = CHI + order[suit] * 21 + start * 3 + position
    for base in (PENG, GANG_EXPOSED, GANG_CONCEALED, GANG_ADDED):
        action[base : base + 33] = base + tile[:33]
    assert action[HU] == HU and sorted(action) == list(range(N_ACTIONS))
    return tile, action


class SuitPermuter:
    """Applies transform `index` (0 is the identity) to a batch and its labels."""

    def __init__(self, device):
        self.tables = []
        for order, reflect in TRANSFORMS:
            tile, action = maps(order, reflect)
            self.tables.append((
                torch.as_tensor(np.argsort(tile[:34]), device=device),  # new column <- old column
                torch.as_tensor(tile, device=device),
                torch.as_tensor(np.argsort(action), device=device),  # new action <- old action
                torch.as_tensor(action, device=device),
                reflect,
            ))

    def __len__(self):
        return len(self.tables)

    def __call__(self, batch, label, index):
        if index == 0:
            return batch, label
        column, tile, action_source, action, reflect = self.tables[index]
        planes = batch["planes"].index_select(2, column)
        planes[:, P_SUIT:] = batch["planes"][:, P_SUIT:]
        events = batch["events"].clone()
        claimed = events[..., 2].long()
        other = events[..., 3].long()
        events[..., 2] = tile[claimed].to(events.dtype)
        if reflect:
            # A chi event keeps the lowest tile of its sequence; read
            # backwards, the sequence starts at the image of that tile + 2.
            chi = events[..., 0] == CHI_EVENT
            events[..., 3] = torch.where(chi, tile[(other + 2).clamp(max=34)], tile[other]).to(events.dtype)
        else:
            events[..., 3] = tile[other].to(events.dtype)
        out = dict(batch, planes=planes, events=events, mask=batch["mask"].index_select(1, action_source))
        return out, action[label]
