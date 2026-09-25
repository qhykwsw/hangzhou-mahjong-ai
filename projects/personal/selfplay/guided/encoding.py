"""Legal inputs with untruncated current-hand chronology.

Tile/shape features reuse the pure-numpy encoder. Privileged features are a
separate function used exclusively by training workers.
"""

import numpy as np

from selfplay_core.encoding import EVENT_KINDS, encode as base_encode

# The engine silently falls back to pure Python without the native extension,
# which makes shape features about an order of magnitude slower. Fail instead.
try:
    from mahjong_ai import _fast  # noqa: F401
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
