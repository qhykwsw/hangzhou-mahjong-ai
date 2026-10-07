"""A second continuation for every deviation (cfg["portfolio"]): fp-rule-131 plays the learner's seat on.

Under vine a deviation's advantage is its copy's value less the hand's, both
continuing with the Actor's argmax (targets.playout_advantages). The
branches hook names the same deviation again; the framework plays it and the
argmax (branch_control) from the same fork on the same wall, the learner's
seat continued by fp-rule-131 in C (cfg["branch_continuation"]). The two
advantages measure the same first move under two continuation policies of
about equal strength; their noise is only partly shared (the continuations
part ways at the learner's next decision), so their mean has the same signal
and less noise. The portfolio advantage is that mean, the luck since the fork
subtracted on each side and the boost weight applied to both.
"""

import numpy as np
import torch


def repeat_deviations(logits, mask, actions):
    """The branches hook: each deviation (a sampled action that is not the argmax) once more; else -1."""
    best = logits.float().masked_fill(~mask, -1e9).argmax(-1)
    return torch.where(actions != best, actions, torch.full_like(actions, -1))[:, None]


def portfolio_advantages(rows, cfg, weights, stats):
    """playout_advantages' normalized weights and stats, with each deviation's 131 advantage averaged in."""
    raw = weights.astype(np.float64) * stats["adv_scale"]
    count = len(rows)
    hand = np.array([r["hand"] for r in rows])
    if "game_mark" in rows[0]:
        mark = np.array([r["game_mark"] for r in rows])
        game = np.cumsum(np.r_[1, (np.diff(mark) != 0).astype(int)]) - 1
    else:
        game = np.cumsum(np.r_[1, (np.diff(hand) < 0).astype(int)]) - 1
    new_hand = np.r_[True, (np.diff(hand) != 0) | (np.diff(game) != 0)]
    hand_id = np.cumsum(new_hand) - 1
    first = np.flatnonzero(new_hand)
    same_game_next = np.r_[game[first[1:]] == game[first[:-1]], False]
    last_hand = ~same_game_next
    premium = stats["premium"]
    luck = bool(cfg.get("luck"))

    def value(r, i):
        scale = 0.0 if last_hand[hand_id[i]] else premium
        v = float(r["playout"]) + scale * float(r["playout_deals_next"])
        if luck:
            v -= float(r.get("playout_luck_pay", 0.0)) + scale * float(r.get("playout_luck_deal", 0.0))
        return v

    deviated = np.array([bool(r.get("deviated", False)) for r in rows])
    both = [i for i in np.flatnonzero(deviated)
            if rows[i].get("branches") and rows[i].get("branch_control") is not None
            and not rows[i]["branches"][0].get("truncated") and not rows[i]["branch_control"].get("truncated")]
    adv = raw.copy()
    if both:
        idx = np.array(both)
        weight = np.exp(np.array([float(rows[i].get("logprob", 0.0)) for i in both])) if cfg.get("deviation_boost") else 1.0
        rule = np.array([value(rows[i]["branches"][0], i) - value(rows[i]["branch_control"], i) for i in both]) * weight
        greedy = raw[idx]
        mix = float(cfg.get("portfolio_mix", 0.5))
        adv[idx] = (1.0 - mix) * greedy + mix * rule
        corr = float(np.corrcoef(greedy, rule)[0, 1]) if len(both) > 2 and greedy.std() > 0 and rule.std() > 0 else 0.0
        stats = stats | {
            "portfolio_rows": len(both) / max(1, int(deviated.sum())),
            "rule_adv_mean": float(rule.mean()), "rule_adv_sd": float(rule.std()),
            "greedy_adv_sd": float(greedy.std()), "portfolio_corr": corr,
            "portfolio_adv_sd": float(adv[idx].std()),
            "rule_sign_agree": float((np.sign(rule) == np.sign(greedy))[(rule != 0) & (greedy != 0)].mean())
            if ((rule != 0) & (greedy != 0)).any() else 0.0,
        }
    else:
        stats = stats | {"portfolio_rows": 0.0}
    stats["adv_scale"] = float(max(adv.std(), 1e-9))
    return (adv / stats["adv_scale"]).astype(np.float32), stats
