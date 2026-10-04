"""finish_game hook: targets are built after collection (targets.py), so the
per-game hook only fills the fields the harness requires, plus a mark shared
by the game's rows: an opening is a one-hand game, and two in a row can carry
the same hand number, so the hand number alone cannot tell games apart."""

import random


def finish_game(rows, cfg):
    mark = random.getrandbits(62)
    for row in rows:
        row["adv"] = 0.0
        row["ret"] = 0.0
        row["game_mark"] = mark
    return rows
