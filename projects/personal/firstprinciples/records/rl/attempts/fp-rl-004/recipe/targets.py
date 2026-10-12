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
    # GAE within the hand: an early decision's return carries the luck of
    # every later draw, so bootstrap from the cross-fitted value of the next
    # decision. delta = reward + V(next) - V (rewards come from the target
    # differences); the hand's last decision closes against its full target.
    lam = float(cfg["gae_lambda"])
    adv = np.zeros(count)
    carry = 0.0
    for i in range(count - 1, -1, -1):
        if i + 1 == count or new_hand[i + 1]:
            carry = target[i] - value[i]
        else:
            carry = target[i] - target[i + 1] + value[i + 1] - value[i] + lam * carry
        adv[i] = carry
    r2 = 1.0 - float(((target - value) ** 2).mean() / max(target.var(), 1e-9))
    stats = {"premium": premium, "critic_r2": r2, "games": games, "advantage_sd": float(adv.std()),
             "monte_carlo_sd": float((target - value).std())}
    return ((adv - adv.mean()) / max(adv.std(), 1e-9)).astype(np.float32), stats
