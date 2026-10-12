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
    # Where the hand starts (dealer, scores in base units, game length): the
    # next hand's first row gives the game-value target its state (targets.py).
    # The C engine's view keeps them in its snapshot; reading `scores` off the
    # view itself would replay the whole game in Python.
    snap = getattr(state, "_snap", None)
    game = getattr(state, "_game", None)
    config = game.config if game is not None else state.config
    scores = snap["scores"] if snap is not None else state.scores
    return {
        "oracle_planes": planes,
        "oracle_scalars": scalars,
        "is_dealer": float(state.dealer == seat),
        "dealer_rel": float((state.dealer - seat) % 4),
        "seat": float(seat),
        "hand_scores": np.asarray(scores, dtype=np.float32) / config.base_score,
        "rounds": float(config.rounds),
    }


def critic_features_progress(state, seat):
    """critic_features plus the real opponents' progress code (worlds.code_of), for cfg["branch_resample"]."""
    from .worlds import code_of

    out = critic_features(state, seat)
    out["opp_code"] = float(code_of(state.wall, state.hands, [len(state.melds[s]) for s in range(4)], seat))
    return out
