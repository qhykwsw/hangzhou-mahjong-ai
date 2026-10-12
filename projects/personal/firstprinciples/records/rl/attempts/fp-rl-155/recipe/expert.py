"""Expert iteration on searched windows (cfg["expert"], with cfg["search"] and openings).

At a kong or fish window the framework plays every candidate (the Actor's
argmax first) through the same sampled worlds to the end of the hand, the
learner taking its argmax afterwards (cfg["search"]). A window teaches only
when its best alternative beats the argmax on the paired per-world
differences by more than `z` standard errors: then the Actor is pulled toward
that alternative by cross entropy. Every other row, the rest of those games
and a slice of ordinary games, is held at the step's starting policy by KL,
so only the searched decisions are free to move (attempt 124 lost 8 points by
training a few cells without an anchor; 125 held with one).
"""

import copy
import time

import numpy as np
import torch

from .batch import actor_args, make_batch, masked_log_probs


def search_targets(rows, z):
    """[(row index, target action, mean gain)] for windows whose best alternative is significantly better."""
    out, searched = [], 0
    for i, row in enumerate(rows):
        table = row.get("search_values")
        actions = row.get("search_actions")
        if not table or not actions or len(actions) < 2:
            continue
        values = np.asarray(table, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            continue
        searched += 1
        gain = values[:, 1:] - values[:, :1]
        mean = gain.mean(0)
        se = gain.std(0, ddof=1) / np.sqrt(len(gain))
        best = int(np.argmax(mean))
        if mean[best] > 0 and mean[best] > z * max(float(se[best]), 1e-9):
            out.append((i, int(actions[best + 1]), float(mean[best])))
    return out, searched


def edit(actor, anchor, rows, targets, opt, size, device, rng, epochs):
    """Cross entropy toward the targets, KL to the anchor on every other row.

    The targets are a fraction of a percent of the rows, so each minibatch of
    anchored rows carries `size // 8` target rows drawn in turn, and the two
    losses are averaged separately: the anchor holds the policy everywhere
    while each pass still visits every target.
    """
    on = np.zeros(len(rows), dtype=bool)
    label = np.zeros(len(rows), dtype=np.int64)
    for i, action, _ in targets:
        on[i], label[i] = True, action
    held, taught = np.flatnonzero(~on), np.flatnonzero(on)
    per = max(1, size // 8)
    actor.eval()
    total_kl, kl_rows, hits = 0.0, 0, 0
    for epoch in range(epochs):
        order = rng.permutation(held)
        cycle = rng.permutation(taught) if len(taught) else taught
        for k, s in enumerate(range(0, len(order), size)):
            idx = order[s : s + size]
            pick = cycle[(np.arange(per) + k * per) % len(cycle)] if len(cycle) else cycle
            both = np.concatenate([idx, pick])
            mb = make_batch([rows[i] for i in both], device)
            n = len(idx)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref = masked_log_probs(anchor(*actor_args(mb)), mb["mask"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logp = masked_log_probs(actor(*actor_args(mb)), mb["mask"])
            kl = (ref[:n].exp() * (ref[:n] - logp[:n])).masked_fill(~mb["mask"][:n], 0).sum(-1)
            loss = kl.mean()
            if len(pick):
                target = torch.as_tensor(label[pick], device=logp.device)
                loss = loss - logp[n:].gather(1, target[:, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            if epoch == 0:
                total_kl += float(kl.detach().sum())
                kl_rows += n
    opt.zero_grad(set_to_none=True)
    # Agreement with the targets after the edit.
    if len(taught):
        with torch.no_grad():
            mb = make_batch([rows[i] for i in taught], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logp = masked_log_probs(actor(*actor_args(mb)), mb["mask"])
            hits = int((logp.argmax(-1).cpu().numpy() == label[taught]).sum())
    return {"anchor_kl": total_kl / max(1, kl_rows), "target_agreement": hits / max(1, len(taught))}


def train(ctx, actor, cfg, collect_cfg, openings, rng, hands, reserve):
    """Steps of: searched openings + ordinary anchor rows, then one anchored edit."""
    expert = cfg["expert"]
    search_cfg = dict(collect_cfg, search=dict(cfg["search"]))
    windows, anchor_rows = int(expert["windows"]), int(expert["anchor_rows"])
    if ctx.smoke:
        windows, anchor_rows = 40, 2000
    per_call = int(cfg.get("starts_per_call", 1000))
    opt = torch.optim.Adam(actor.parameters(), lr=float(expert.get("learning_rate", 1e-4)), eps=1e-5)
    step, iteration, turn, last = 0, 0, 0, 0.0
    while hands() and ctx.time_left() > 1.15 * last + reserve:
        began = time.time()
        rows, games = [], 0
        searched_rows = []
        # Searched windows: each opening game starts at one (max_windows 1).
        while games < windows and hands() and ctx.time_left() > reserve:
            iteration += 1
            pool = openings[turn % len(openings)]
            turn += 1
            # About three learning decisions per opening game.
            got, records, _ = ctx.collect(
                actor, search_cfg, samples=min(20000, 3 * (windows - games)), iteration=iteration,
                starts=[pool[i] for i in rng.choice(len(pool), min(len(pool), per_call), replace=False)],
            )
            games += len(records)
            searched_rows += got
        searched = time.time() - began
        mark = time.time()
        # Ordinary games: the anchor holds the policy where nothing was searched.
        while len(rows) < anchor_rows and hands() and ctx.time_left() > reserve:
            iteration += 1
            got, _, _ = ctx.collect(actor, collect_cfg, samples=min(40000, anchor_rows), iteration=iteration)
            rows += got
        ordinary = time.time() - mark
        if not searched_rows:
            break
        rows = searched_rows + rows
        targets, n_searched = search_targets(rows, float(expert.get("z", 2.0)))
        step += 1
        anchor = copy.deepcopy(actor).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        stats = edit(actor, anchor, rows, targets, opt, int(cfg["microbatch"]), ctx.device, rng,
                     int(expert.get("epochs", 2)))
        del anchor
        kinds = {}
        for row in searched_rows:
            if row.get("search_kind"):
                kinds[row["search_kind"]] = kinds.get(row["search_kind"], 0) + 1
        taught_kinds = {}
        for i, _, _ in targets:
            k = rows[i].get("search_kind", "?")
            taught_kinds[k] = taught_kinds.get(k, 0) + 1
        ctx.save_actor(actor)
        last = time.time() - began
        ctx.log(step=step, games=games, rows=len(rows), searched=n_searched, targets=len(targets),
                target_gain=float(np.mean([g for _, _, g in targets])) if targets else 0.0,
                **{f"searched_{k}": v for k, v in kinds.items()},
                **{f"targets_{k}": v for k, v in taught_kinds.items()},
                **stats, search_seconds=searched, ordinary_seconds=ordinary, step_seconds=last)
        torch.cuda.empty_cache()
    return actor
