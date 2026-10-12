"""finish_game hook: targets are built after collection (targets.py), so the
per-game hook only fills the fields the harness requires."""


def finish_game(rows, cfg):
    for row in rows:
        row["adv"] = 0.0
        row["ret"] = 0.0
    return rows
