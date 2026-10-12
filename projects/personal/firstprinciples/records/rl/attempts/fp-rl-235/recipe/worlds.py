"""Progress-matched re-dealt worlds (cfg["branch_resample"]; search digest #9).

Under branch_resample a deviation and its control play from one world
re-dealt from the learner's point of view: the others' concealed tiles and
the wall are pooled and shuffled. A uniform re-deal leaves the others slower
than real opponents (their discards and calls say how close they are); the
search measurement found the continuation overvalued by it (census6, R18).

The world's weight here is the rule line's progress matching as an
importance weight: per opponent, its shanten class given the wall stage and
its meld count, P (real hands, from the rollout's own rows) over Q (the
re-dealt worlds of the same rollout), the product over the three opponents,
normalized to mean 1 and clipped. The worker cannot know P and Q, so its
weight function returns the world's cells as a code (progress_code) and the
parent turns codes into weights (importance).
"""

import numpy as np

STAGES, MELDS, CLASSES = 10, 5, 8


def code_of(wall, hands, opened, seat):
    """(wall stage, (melds, shanten class) per opponent) packed into one exact integer."""
    from mahjong_ai.sdk.shape import shanten

    stage = min(STAGES - 1, max(0, (len(wall) - 20) // 8))
    code, base = stage, STAGES
    for i in (1, 2, 3):
        s = (seat + i) % 4
        m = min(int(opened[s]), MELDS - 1)
        try:
            cls = min(shanten(tuple(sorted(hands[s])), m), 5) + 1  # -1 (complete) .. 5
        except ValueError:
            cls = CLASSES - 1  # an unusual hand size: its own class
        code += base * (m * CLASSES + cls)
        base *= MELDS * CLASSES
    return code


def progress_code(row, world):
    """cfg["branch_resample"]["weights"]: the re-dealt world's code (as a float; exact below 2**24)."""
    return float(code_of(world["wall"], world["hands"], world["opened"], world["seat"]))


def cells(codes):
    """(n, 3) cell indices (stage, melds, class) flattened, one per opponent, from codes."""
    codes = np.asarray(codes, dtype=np.int64)
    stage = codes % STAGES
    rest = codes // STAGES
    out = np.zeros((len(codes), 3), dtype=np.int64)
    for i in range(3):
        mc = rest % (MELDS * CLASSES)
        rest //= MELDS * CLASSES
        out[:, i] = stage * (MELDS * CLASSES) + mc
    return out


def importance(real_codes, world_codes, clip=20.0):
    """Each world's weight prod_i P(cell_i) / Q(cell_i) within its (stage, melds) group, mean 1, clipped."""
    size = STAGES * MELDS * CLASSES
    real, world = cells(real_codes), cells(world_codes)
    p = np.bincount(real.reshape(-1), minlength=size).astype(np.float64) + 0.5
    q = np.bincount(world.reshape(-1), minlength=size).astype(np.float64) + 0.5
    # Conditional on the group (stage, melds): normalize over the classes.
    p = p.reshape(-1, CLASSES)
    q = q.reshape(-1, CLASSES)
    p /= p.sum(1, keepdims=True)
    q /= q.sum(1, keepdims=True)
    ratio = (p / q).reshape(-1)
    w = np.prod(ratio[world], axis=1)
    w /= max(w.mean(), 1e-12)
    w = np.minimum(w, clip)
    return w / max(w.mean(), 1e-12)
