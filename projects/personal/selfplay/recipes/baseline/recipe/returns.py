"""Returns and advantages, computed in the game worker when a game ends.

finish_game(rows, cfg) receives the learning seat's decisions of one complete
game in order. Each row has ``reward`` (the hand's table score, on the last
decision of that hand, else 0), ``value`` (the Critic's estimate at collection)
and ``hand`` (1-16). It must set ``adv`` and ``ret`` on every row.
"""


def finish_game(rows, cfg):
    # Whole-game GAE with gamma = 1; the terminal value is exactly zero.
    lam = cfg["gae_lambda"]
    adv = 0.0
    next_value = 0.0
    for row in reversed(rows):
        adv = row["reward"] + next_value - row["value"] + lam * adv
        row["adv"] = adv
        row["ret"] = adv + row["value"]
        next_value = row["value"]
    return rows
