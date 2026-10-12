"""Completed-Q over the policy's top candidates (cfg["cq"]; search digest #5).

A sampled learning decision (with probability p = min(1, rate * (1 - pi(argmax))))
plays its k most likely non-argmax actions and the argmax (branch_control),
each in a copy from the same fork on the same wall, continued by the Actor's
argmax: q(a) is a candidate's value less the control's, the luck since the
fork subtracted, and q(argmax) = 0. Actions not played take the completed
value v = sum_visited pi(a) q(a) / sum_visited pi(a) (Gumbel MuZero's v_mix),
so they add nothing. The improved policy softmax(logits + s * (q - v)) is
followed by one KL-sized step; to first order in s (the noise-aware regime:
a single wall's q is ~11 base units of noise around gaps of tenths, so any
sensible s is small) the step is the gradient
sum_visited pi(a) (q(a) - v) grad log pi(a) / p, each visited action's own term,
which is what the rows here carry: the decision's row for the argmax and one
extra row per candidate.

Unlike the sampled deviation (one action drawn as the policy would), every
visited action of the decision gets its own paired value, and an action the
policy gives little mass reaches the gradient whenever it is in the top k.
"""

import numpy as np
import torch

from .batch import actor_args, make_batch, masked_log_probs


def argmax_rate(rate):
    """The sample hook: the argmax; its logprob slot carries log p, the decision's chance of being searched."""
    def sample(logits, mask):
        probs = torch.softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        top, best = probs.max(-1)
        p = (rate * (1.0 - top).clamp_min(0.0)).clamp(min=1e-12, max=1.0)
        return best, p.log()

    sample.capturable = True
    return sample


def top_candidates(k, rate):
    """The branches hook: with probability p the k most likely non-argmax legal actions, else none."""
    def branches(logits, mask, actions):
        probs = torch.softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        top, best = probs.max(-1)
        p = (rate * (1.0 - top).clamp_min(0.0)).clamp(max=1.0)
        others = probs.scatter(1, best[:, None], -1.0)
        values, index = others.topk(k, dim=-1)
        index = torch.where(values > 0, index, torch.full_like(index, -1))
        go = torch.rand(len(probs), device=probs.device) < p
        return torch.where(go[:, None], index, torch.full_like(index, -1))

    return branches


def structure(rows):
    """(hand_id per row, last_hand per hand, dealer premium), as targets.playout_advantages computes them."""
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
    whole = (np.bincount(game[first]) > 1)[game[first]]
    dealt = (dealer[first] > 0.5) & whole
    undealt = (dealer[first] <= 0.5) & whole
    premium = (float(outcome[dealt].mean() - outcome[undealt].mean())
               if dealt.any() and undealt.any() else 0.0)
    same_game_next = np.r_[game[first[1:]] == game[first[:-1]], False]
    return hand_id, ~same_game_next, premium, int(game.max() + 1)


@torch.no_grad()
def policy_probs(actor, rows, device):
    out = []
    actor.eval()
    for s in range(0, len(rows), 1024):
        mb = make_batch(rows[s : s + 1024], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            logits = actor(*actor_args(mb))
        out.append(masked_log_probs(logits, mb["mask"]).exp().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def cq_advantages(rows, cfg, actor, device):
    """(normalized weights for rows + extra, stats, extra rows): each searched decision's visited actions."""
    hand_id, last_hand, premium, games = structure(rows)
    luck = bool(cfg.get("luck"))

    def value(r, i):
        scale = 0.0 if last_hand[hand_id[i]] else premium
        v = float(r["playout"]) + scale * float(r["playout_deals_next"])
        if luck:
            v -= float(r.get("playout_luck_pay", 0.0)) + scale * float(r.get("playout_luck_deal", 0.0))
        return v

    picked = [i for i, r in enumerate(rows) if r.get("branches") and r.get("branch_control") is not None]
    adv = np.zeros(len(rows))
    extra, extra_adv = [], []
    stats = {"premium": premium, "games": games, "cq_rate": len(picked) / max(1, len(rows))}
    if picked:
        probs = policy_probs(actor, [rows[i] for i in picked], device)
        best_gain, positive, visited_mass, counts = [], 0, [], []
        for j, i in enumerate(picked):
            r = rows[i]
            control = value(r["branch_control"], i)
            acts = [int(r["branch_control"]["action"])] + [int(b["action"]) for b in r["branches"]]
            q = np.array([0.0] + [value(b, i) - control for b in r["branches"]])
            pi = probs[j, acts]
            v = float((pi * q).sum() / max(pi.sum(), 1e-12))
            p = float(np.exp(r.get("logprob", 0.0)))
            terms = pi * (q - v) / max(p, 1e-12)
            # The decision's own row played the argmax; each candidate gets a row of its own.
            adv[i] = terms[0]
            for a, t in zip(acts[1:], terms[1:]):
                extra.append(dict(r, action=a))
                extra_adv.append(t)
            best_gain.append(float(q[1:].max()))
            positive += bool((q[1:] > 0).any())
            visited_mass.append(float(pi.sum()))
            counts.append(len(acts) - 1)
        stats |= {"cq_candidates": float(np.mean(counts)), "cq_best_gain": float(np.mean(best_gain)),
                  "cq_any_positive": positive / len(picked), "cq_visited_mass": float(np.mean(visited_mass))}
    full = np.concatenate([adv, np.asarray(extra_adv, dtype=np.float64)])
    stats["adv_scale"] = float(max(full.std(), 1e-9))
    return (full / stats["adv_scale"]).astype(np.float32), stats, extra
