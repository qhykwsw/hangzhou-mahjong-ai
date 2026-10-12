"""Critic inputs computed in the game worker from the full hidden state.

critic_features(state, seat) runs for every learner decision; the returned
arrays join that row. The Actor never sees them (ADR-0009 training exception).
is_dealer is used for the dealer premium in targets.py.
"""

import numpy as np


def critic_features(state, seat):
    from mahjong_ai.sdk.shape import shanten

    planes = np.zeros((16, 34), dtype=np.float32)
    scalars = np.zeros(4, dtype=np.float32)
    for i, tiles in enumerate([state.hands[(seat + j) % 4] for j in (1, 2, 3)] + [state.wall]):
        count = np.bincount(tiles, minlength=34)
        planes[4 * i : 4 * i + 4] = count[None, :] >= np.arange(1, 5)[:, None]
        if i < 3:
            opened = len(state.melds[(seat + i + 1) % 4])
            if len(tiles) in (13 - 3 * opened, 14 - 3 * opened):
                scalars[i] = (8 - shanten(tuple(sorted(tiles)), opened)) / 9
    scalars[3] = len(state.wall) / 136
    return {
        "oracle_planes": planes,
        "oracle_scalars": scalars,
        "is_dealer": float(state.dealer == seat),
    }
