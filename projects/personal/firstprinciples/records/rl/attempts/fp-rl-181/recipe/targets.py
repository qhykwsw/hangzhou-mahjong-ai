"""Per-decision advantages for one rollout.

Credit stops at the end of the hand: the rest of the game adds noise several
times larger than one hand's. The one cross-hand effect kept is the dealer
seat: the winner deals next, and the dealer pays and receives 8x. The premium
D (mean hand score as dealer minus as non-dealer, measured on this rollout) is
added to a decision's return when the learner deals the next hand. The
baseline is a Critic cross-fitted on the other half of the games, so no row's
baseline has seen its own return.
"""

import numpy as np

from .critic import fit_predict


def advantages(rows, cfg, rng, device, seed, *, batch_source=None):
    hand = np.array([r["hand"] for r in rows])
    reward = np.array([float(r["reward"]) for r in rows])
    game = np.cumsum(np.r_[1, (np.diff(hand) < 0).astype(int)]) - 1
    count = len(rows)
    new_hand = np.r_[True, (np.diff(hand) != 0) | (np.diff(game) != 0)]
    hand_rest = np.zeros(count)
    next_first = np.full(count, -1)
    running, upcoming = 0.0, -1
    for i in range(count - 1, -1, -1):
        if i + 1 == count or game[i + 1] != game[i]:
            upcoming = -1
        if i + 1 == count or new_hand[i + 1]:
            running = 0.0
        running += reward[i]
        hand_rest[i] = running
        next_first[i] = upcoming
        if new_hand[i]:
            upcoming = i
    dealer = np.array([float(r["is_dealer"]) for r in rows])
    first = np.flatnonzero(new_hand)
    dealt = dealer[first] > 0.5
    premium = float(hand_rest[first][dealt].mean() - hand_rest[first][~dealt].mean()) if dealt.any() and (~dealt).any() else 0.0
    target = hand_rest + np.where(next_first >= 0, premium * dealer[np.maximum(next_first, 0)], 0.0)
    games = int(game.max() + 1)
    half = np.zeros(games, dtype=int)
    half[rng.permutation(games)[: games // 2]] = 1
    side = half[game]
    value = np.zeros(count)
    for h in (0, 1):
        fit_idx, pred_idx = np.flatnonzero(side != h), np.flatnonzero(side == h)
        value[pred_idx] = fit_predict(
            rows, target, fit_idx, pred_idx, cfg, rng, device, seed + h,
            batch_source=batch_source,
        )
    adv = target - value
    r2 = 1.0 - float(((target - value) ** 2).mean() / max(target.var(), 1e-9))
    stats = {"premium": premium, "critic_r2": r2, "games": games, "advantage_sd": float(adv.std())}
    return ((adv - adv.mean()) / max(adv.std(), 1e-9)).astype(np.float32), stats


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
    The dealer premium (as in advantages) is added to both sides.
    """
    count = len(rows)
    hand = np.array([r["hand"] for r in rows])
    reward = np.array([float(r["reward"]) for r in rows])
    if "game_mark" in rows[0]:
        mark = np.array([r["game_mark"] for r in rows])
        game = np.cumsum(np.r_[1, (np.diff(mark) != 0).astype(int)]) - 1
    else:
        game = np.cumsum(np.r_[1, (np.diff(hand) < 0).astype(int)]) - 1
    new_hand = np.r_[True, (np.diff(hand) != 0) | (np.diff(game) != 0)]
    hand_id = np.cumsum(new_hand) - 1
    first = np.flatnonzero(new_hand)
    outcome = np.bincount(hand_id, weights=reward)
    dealer = np.array([float(r["is_dealer"]) for r in rows])
    # The premium comes from whole games only: an opening's one hand starts
    # from a rare window, not from an ordinary deal.
    whole = (np.bincount(game[first]) > 1)[game[first]]
    dealt = (dealer[first] > 0.5) & whole
    undealt = (dealer[first] <= 0.5) & whole
    premium = (float(outcome[dealt].mean() - outcome[undealt].mean())
               if dealt.any() and undealt.any() else 0.0)
    # Whether the learner deals the hand after each hand (the game's last has none).
    same_game_next = np.r_[game[first[1:]] == game[first[:-1]], False]
    deals_next = np.where(same_game_next, np.r_[dealer[first[1:]], 0.0], 0.0)
    last_hand = ~same_game_next
    deviated = np.array([bool(r.get("deviated", False)) for r in rows])
    playout = np.array([float(r.get("playout", 0.0)) for r in rows])
    playout_next = np.array([float(r.get("playout_deals_next", 0.0)) for r in rows])
    if cfg.get("game_value"):
        # The value of the next hand's starting state (who deals, hands left,
        # scores; value.start_value, the rule line's fit) replaces the constant
        # dealer premium: a hand is credited with what the rest of the game is
        # worth from where it leaves the table.
        # The actual path reads it off the next hand's first row, a greedy
        # continuation off its own end; the game's last hand, and an opening's
        # one hand, add nothing on either side.
        from .value import start_value as value
        after = np.zeros(len(first))
        for h in np.flatnonzero(same_game_next):
            r = rows[first[h + 1]]
            after[h] = value(int(r["dealer_rel"]), int(r["rounds"]) - int(r["hand"]) + 1, int(r["rounds"]),
                             list(r["hand_scores"]), int(r["seat"]))
        greedy_value = playout.copy()
        for i in np.flatnonzero(deviated & ~last_hand[hand_id]):
            r = rows[i]
            if "playout_dealer" in r and int(r["hand"]) < int(r["rounds"]):
                greedy_value[i] += value((int(r["playout_dealer"]) - int(r["seat"])) % 4, int(r["rounds"]) - int(r["hand"]),
                                         int(r["rounds"]), list(r["playout_scores"]), int(r["seat"]))
        ends = outcome + after
    else:
        greedy_value = playout + premium * np.where(last_hand[hand_id], 0.0, playout_next)
        ends = outcome + premium * deals_next
    adv = np.zeros(count)
    for h in np.unique(hand_id[deviated]):
        points = np.flatnonzero(deviated & (hand_id == h))
        taken = np.r_[greedy_value[points[1:]], ends[h]]
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
