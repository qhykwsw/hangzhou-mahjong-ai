"""Critic inputs computed in the game worker from the full hidden state.

critic_features(state, seat) runs for every learner decision; the returned
arrays join that row. The Actor never sees them (ADR-0009 training exception).
is_dealer is used for the dealer premium in targets.py.

Only self-draws win, so the hand is mostly decided by who first draws a
winning tile. Besides every hidden hand and the wall's contents, the Critic
gets the wall's order as each seat will draw it if nobody calls: the tiles
in each seat's next draws, and for each waiting hand which of its own draws
first wins, for how much, and how many of its draws win at all. Calls shift
the order, so these are strong hints, not the outcome.
"""

import numpy as np

ORACLE_PLANES = 20
ORACLE_SCALARS = 20
LAST_WALL = 20  # the hand is drawn once the wall is down to this
NEXT_DRAWS = 4


def critic_features(state, seat):
    from mahjong_ai.sdk.shape import analyze, shanten

    planes = np.zeros((ORACLE_PLANES, 34), dtype=np.float32)
    scalars = np.zeros(ORACLE_SCALARS, dtype=np.float32)
    for i, tiles in enumerate([state.hands[(seat + j) % 4] for j in (1, 2, 3)] + [state.wall]):
        count = np.bincount(tiles, minlength=34)
        planes[4 * i : 4 * i + 4] = count[None, :] >= np.arange(1, 5)[:, None]
        if i < 3:
            opened = len(state.melds[(seat + i + 1) % 4])
            if len(tiles) in (13 - 3 * opened, 14 - 3 * opened):
                scalars[i] = (8 - shanten(tuple(sorted(tiles)), opened)) / 9
    scalars[3] = len(state.wall) / 136
    # Draws pop from the wall's end; without calls they go round from the
    # seat after the current discarder (or after me, when I am to discard).
    after = state.discarder if state.phase.startswith("response") else state.turn
    upcoming = state.wall[LAST_WALL:][::-1]
    for j in range(4):
        who = (seat + j) % 4
        draws = upcoming[(who - after - 1) % 4 :: 4]
        planes[16 + j] = np.bincount(draws[:NEXT_DRAWS], minlength=34)[:34] / 2
        hand, opened = state.hands[who], len(state.melds[who])
        if len(hand) != 13 - 3 * opened:
            continue
        row = analyze(tuple(sorted(hand)), opened)
        if row["shanten"] != 0 or not row["winning_tiles"]:
            continue
        wins = set(row["winning_tiles"])
        first = next((k for k, tile in enumerate(draws) if tile in wins), None)
        base = 4 + 4 * j
        scalars[base] = 1.0
        if first is not None:
            scalars[base + 1] = float(np.exp(-first / 2))
            scalars[base + 2] = min(float(row["fans"][draws[first]]), 64.0) / 16
        scalars[base + 3] = sum(tile in wins for tile in draws) / 4
    return {
        "oracle_planes": planes,
        "oracle_scalars": scalars,
        "is_dealer": float(state.dealer == seat),
    }
