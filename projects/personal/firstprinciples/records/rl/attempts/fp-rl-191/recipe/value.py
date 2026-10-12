"""Game value V: my expected table points from the start of a hand to the game's end.

The rule line's fit (.local/models/game-value/game_value.json): fp-rule-128 against
the fixed pool c9e98aa5, 24000 games each of 16 and 8 hands, a Markov
decomposition by who deals (per-hand points and next-dealer odds by place in
the game). Scores do not enter: they did not explain the held-out residual.
V[total][hands_left][dealer_rel], dealer_rel = (dealer - me) % 4 (0 me, 1 after
me, 2 opposite, 3 before me); hands_left counts the hand about to start.
"""

V = {
    16: [
        [0.0, 0.0, 0.0, 0.0],
        [1.213, -0.355, -0.115, 0.135],
        [2.09, -0.434, 0.418, 0.326],
        [1.916, 0.024, 0.374, 0.673],
        [2.055, 0.18, 0.511, 0.821],
        [2.2, 0.325, 0.656, 0.966],
        [2.345, 0.47, 0.8, 1.111],
        [2.49, 0.615, 0.945, 1.256],
        [2.635, 0.76, 1.09, 1.401],
        [2.78, 0.905, 1.235, 1.546],
        [2.925, 1.05, 1.38, 1.691],
        [3.07, 1.195, 1.525, 1.836],
        [3.215, 1.34, 1.67, 1.981],
        [3.36, 1.485, 1.815, 2.126],
        [3.505, 1.63, 1.96, 2.271],
        [3.65, 1.775, 2.105, 2.416],
        [3.795, 1.92, 2.25, 2.561],
    ],
    8: [
        [0.0, 0.0, 0.0, 0.0],
        [1.18, -0.715, -0.325, 0.104],
        [1.477, -0.617, -0.255, 0.17],
        [1.36, -0.366, -0.047, 0.201],
        [1.451, -0.265, 0.045, 0.287],
        [1.544, -0.172, 0.137, 0.38],
        [1.636, -0.08, 0.23, 0.473],
        [1.729, 0.013, 0.322, 0.565],
        [1.822, 0.106, 0.415, 0.658],
    ],
}


def start_value(dealer_rel, hands_left, total_hands, scores=None, seat=None):
    """V at the start of a hand; 0 once the game is over."""
    if hands_left <= 0:
        return 0.0
    table = V.get(int(total_hands), V[16])
    return table[min(int(hands_left), len(table) - 1)][int(dealer_rel) % 4]


def after_hand(info):
    """Search continuation (cfg["search"]["continuation"]): V of the next hand's start, 0 after the last."""
    if info["last"]:
        return 0.0
    return start_value((info["dealer"] - info["seat"]) % 4, info["rounds"] - info["hand_no"] + 1, info["rounds"])
