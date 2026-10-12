"""Per-decision advantages for one rollout: exact counterfactual differences from greedy playouts.

Credit stops at the end of the hand: the rest of the game adds noise several
times larger than one hand's. The one cross-hand effect kept is the dealer
seat: the winner deals next, and the dealer pays and receives 8x. The premium
D (mean hand score as dealer minus as non-dealer, measured on this rollout)
is added to a decision's value when the learner deals the next hand.
"""

import numpy as np


def playout_advantages(rows, cfg):
    """Exact counterfactual advantages from the framework's greedy playouts.

    With cfg["playout_baseline"] every sampled action that is not the argmax
    (a deviation) gets `playout`: the learner's hand score had it played the
    argmax there and at every later decision, on the same wall against the
    same deterministic opponents. So the argmax path's value is known at each
    deviation, and a deviation's own value (deviate, then argmax) is the next
    deviation's playout, or the realized hand after the hand's last one. The
    advantage of a deviation is the difference of the two, on one wall; every
    decision that played the argmax has advantage exactly 0. No Critic, and
    none of a hand's luck lands on the decisions that did not deviate.
    The dealer premium is added to both sides.
    """
    count = len(rows)
    hand = np.array([r["hand"] for r in rows])
    reward = np.array([float(r["reward"]) for r in rows])
    game = np.cumsum(np.r_[1, (np.diff(hand) < 0).astype(int)]) - 1
    new_hand = np.r_[True, (np.diff(hand) != 0) | (np.diff(game) != 0)]
    hand_id = np.cumsum(new_hand) - 1
    first = np.flatnonzero(new_hand)
    outcome = np.bincount(hand_id, weights=reward)
    dealer = np.array([float(r["is_dealer"]) for r in rows])
    dealt = dealer[first] > 0.5
    premium = (float(outcome[dealt].mean() - outcome[~dealt].mean())
               if dealt.any() and (~dealt).any() else 0.0)
    # Whether the learner deals the hand after each hand (the game's last has none).
    same_game_next = np.r_[game[first[1:]] == game[first[:-1]], False]
    deals_next = np.where(same_game_next, np.r_[dealer[first[1:]], 0.0], 0.0)
    last_hand = ~same_game_next
    deviated = np.array([bool(r.get("deviated", False)) for r in rows])
    playout = np.array([float(r.get("playout", 0.0)) for r in rows])
    playout_next = np.array([float(r.get("playout_deals_next", 0.0)) for r in rows])
    greedy_value = playout + premium * np.where(last_hand[hand_id], 0.0, playout_next)
    adv = np.zeros(count)
    for h in np.unique(hand_id[deviated]):
        points = np.flatnonzero(deviated & (hand_id == h))
        taken = np.r_[greedy_value[points[1:]], outcome[h] + premium * deals_next[h]]
        adv[points] = taken - greedy_value[points]
    moved = adv[deviated]
    stats = {
        "premium": premium, "deviation_rate": float(deviated.mean()),
        "deviation_adv_mean": float(moved.mean()) if len(moved) else 0.0,
        "deviation_adv_sd": float(moved.std()) if len(moved) else 0.0,
        "deviation_zero": float((moved == 0).mean()) if len(moved) else 0.0,
        "games": int(game.max() + 1),
    }
    return (adv / max(adv.std(), 1e-9)).astype(np.float32), stats
