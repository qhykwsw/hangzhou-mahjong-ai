"""Training loop: large on-policy rollout, full-batch gradient, one small KL step.

Why this shape (docs/adr/0012 has the measurements): one decision's advantage
is dominated by luck, so a policy-gradient direction needs hundreds of
thousands of decisions before it beats a random direction, and any step moves
the policy along that direction's noise as well as its signal. Small steps
along directions from large, exactly on-policy rollouts improve the policy
slowly but steadily; small-batch PPO and large steps did not.

Every step: play `rows` decisions with the current Actor (no teacher labels),
build advantages (targets.py), accumulate the policy gradient over all rows
without stepping, then scale the step so that KL(current || new) on a fixed
subset of the rollout equals `step_kl`. A step starts only if the time left
covers the previous step's duration, so the loop never overruns the deadline.
With a hand budget, collection stops when it is spent and the rows gathered so
far make the last (possibly smaller) step.
"""

import json
from pathlib import Path
import time

import numpy as np
import torch

from selfplay_core.actions import GANG_EXPOSED, HU
from guided.collection_session import InferenceSession

from .batch import actor_args, make_batch, masked_log_probs, resident_rows
from .symmetry import SuitPermuter
from .targets import advantages, playout_advantages

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())


def gradient(actor, rows, weights, size, device, *, batch_source=None, capture=False):
    """Sum of advantage-weighted log-probability gradients over every row.

    The sum does not depend on order, so rows are grouped by history length to
    avoid padding short histories to long ones; bf16 autocast halves the cost.
    The direction's precision far exceeds its sampling noise.
    """
    if capture and batch_source is not None and torch.device(device).type == "cuda":
        from .actor_gradient import gradient as resident_gradient

        return resident_gradient(actor, rows, weights, size, device, batch_source=batch_source)
    params = [p for p in actor.parameters() if p.requires_grad]
    total = [torch.zeros_like(p) for p in params]
    order = np.argsort([len(r["events"]) for r in rows], kind="stable")
    actor.eval()
    for s in range(0, len(order), size):
        idx = order[s : s + size]
        mb = (batch_source.batch(idx) if batch_source is not None
              else make_batch([rows[i] for i in idx], device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = actor(*actor_args(mb))
        logp = masked_log_probs(logits, mb["mask"])
        chosen = logp.gather(1, mb["action"][:, None]).squeeze(1)
        loss = (torch.as_tensor(weights[idx], device=device) * chosen).sum() / len(rows)
        for acc, g in zip(total, torch.autograd.grad(loss, params, allow_unused=True)):
            if g is not None:
                acc += g.float()
    return params, total


def tempered(temperature):
    """A sample hook drawing from softmax(logits / temperature) over the legal actions.

    A sharp policy almost never tries its unlikely actions, so the playouts
    never learn whether they are better. Each deviation's advantage is exact
    against the argmax continuation whatever drew it, so the rows keep their
    weight (no importance ratio): a better action is pushed up as soon as it
    is tried, not only as often as the current policy would try it.
    """
    def sample(logits, mask):
        logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e9) / temperature, dim=-1)
        actions = torch.multinomial(logp.exp(), 1).squeeze(1)
        return actions, logp.gather(1, actions[:, None]).squeeze(1)

    return sample


def kl_step(actor, params, direction, batches, goal):
    """Move params to start + eta * direction with KL(start || new) ~= goal."""
    start = [p.detach().clone() for p in params]

    def logps():
        with torch.no_grad():
            return [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in batches]

    reference = logps()
    probabilities = [old.exp().masked_fill(~mb["mask"], 0) for mb, old in zip(batches, reference)]
    latest = reference

    def kl_at(eta):
        nonlocal latest
        with torch.no_grad():
            for p, p0, d in zip(params, start, direction):
                p.copy_(p0 + eta * d)
        latest = logps()
        sums = [(probs * (old - new).masked_fill(~mb["mask"], 0)).sum()
                for mb, old, probs, new in zip(batches, reference, probabilities, latest)]
        # Each batch still sums in fp32, then totals in fp64 like Python float.
        # Only the final scalar crosses the device boundary.
        return float(torch.stack(sums).double().sum()) / sum(len(old) for old in reference)

    norm = float(torch.sqrt(sum((d * d).sum() for d in direction)))
    eta = 1e-3 / max(norm, 1e-12)
    kl = kl_at(eta)
    for _ in range(12):
        if abs(np.log(max(kl, 1e-12) / goal)) < 0.1:
            break
        eta *= float(np.clip((goal / max(kl, 1e-12)) ** 0.5, 0.2, 10.0))
        kl = kl_at(eta)
    hu = [torch.nonzero(mb["mask"][:, HU]).squeeze(1) for mb in batches]
    legal = max(1, sum(len(h) for h in hu))
    before = int(torch.stack([(lp[h].argmax(-1) == HU).sum() for h, lp in zip(hu, reference)]).sum()) / legal
    after = int(torch.stack([(lp[h].argmax(-1) == HU).sum() for h, lp in zip(hu, latest)]).sum()) / legal
    return {"kl": kl, "step_norm": eta * norm, "hu_before": before, "hu_after": after}


def distill(actor, rows, opt, size, device, rng, epochs, *, batch_source=None, permuter=None,
            hard=0.0, window=1.0):
    """Epochs of cross entropy toward the teacher's label on the learner's own states.

    DAgger: the learner played these rows with its own policy, so its mistakes
    lead to states behaviour cloning never visited; the teacher labels them.
    The first epoch's loss is taken before each batch is trained on (fresh rows).
    Later epochs relabel the suits of each batch at random (symmetry.py): the
    data, not the optimisation, limits imitation, and the suits are symmetric.

    A near-converged student already agrees on ~97% of rows, so a uniform loss
    spends almost everything on rows it has right. A row whose current argmax
    differs from the label weighs 1 + `hard`, and a decision where a kong or a
    win is legal weighs `window` times more: those windows are rare but carry
    the multipliers (the teacher's kong-burst options live there).
    """
    teacher = np.array([r["teacher"] for r in rows])
    labeled = np.flatnonzero(teacher >= 0)
    actor.eval()
    total_ce, agree = 0.0, 0.0
    windows = {"kong": [0, 0], "hu": [0, 0], "white": [0, 0]}
    # Each later epoch reshuffles; only the first one reports fresh metrics.
    for epoch in range(epochs):
        order = rng.permutation(labeled)
        for s in range(0, len(order), size):
            idx = order[s : s + size]
            mb = (batch_source.batch(idx) if batch_source is not None
                  else make_batch([rows[i] for i in idx], device))
            label = torch.as_tensor(teacher[idx], device=device)
            if epoch > 0 and permuter is not None:
                mb, label = permuter(mb, label, int(rng.integers(6)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = actor(*actor_args(mb))
            logp = masked_log_probs(logits, mb["mask"])
            nll = -logp.gather(1, label[:, None]).squeeze(1)
            right = logp.argmax(-1) == label
            kong = mb["mask"][:, GANG_EXPOSED:HU].any(1)
            hu = mb["mask"][:, HU]
            weight = 1.0 + hard * (~right).float()
            weight = torch.where(kong | hu, weight * window, weight)
            loss = (weight * nll).sum() / weight.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            if epoch == 0:
                total_ce += float(nll.detach().sum())
                agree += float(right.sum())
                # White in hand: the white discard is legal (white is never claimed).
                for key, sel in (("kong", kong), ("hu", hu), ("white", mb["mask"][:, 33])):
                    windows[key][0] += int(sel.sum())
                    windows[key][1] += int((sel & right).sum())
    opt.zero_grad(set_to_none=True)
    n = max(1, len(labeled))
    stats = {"labeled": len(labeled), "fresh_ce": total_ce / n, "fresh_agreement": agree / n}
    for key, (count, hits) in windows.items():
        stats[f"{key}_rows"] = count
        stats[f"{key}_agreement"] = hits / max(1, count)
    return stats


def train(ctx):
    cfg = dict(CONFIG)
    # Workers, tables and batching are fixed per host by the line's harness.
    collect_cfg = {
        "history_probability": 0.0,
        "critic_features": "recipe.features:critic_features",
        "finish_game": "recipe.returns:finish_game",
        # The strongest rule on the fixed pool labels the distillation.
        "teacher": cfg["teacher"],
    }
    # Policy-gradient rollouts ask for the greedy playout of every deviation.
    pg_cfg = dict(collect_cfg, playout_baseline=bool(cfg.get("playout_baseline")))
    if cfg["teacher"] not in ctx.teachers:
        raise ValueError(f"teacher {cfg['teacher']} is not in ctx.teachers: {ctx.teachers}")
    # The smoke run (about a minute) must still exercise one whole step.
    target_rows = cfg["smoke_rows"] if ctx.smoke else cfg["rows"]
    microbatch = int(cfg["microbatch"])
    reserve = 5 if ctx.smoke else 60
    rng = np.random.default_rng(ctx.seed)
    actor = ctx.actor
    # The first distill_steps rollouts only distil the teacher on the learner's
    # own states (DAgger, where imitation gains fastest); the rest only take
    # policy-gradient steps. Interleaving the two made each undo the other.
    opt = torch.optim.Adam(actor.parameters(), lr=float(cfg["distill_learning_rate"]), eps=1e-5)
    permuter = SuitPermuter(ctx.device)
    chunks = max(1, int(cfg["chunks_per_step"]))
    temperature = float(cfg.get("sample_temperature", 1.0))
    sampler = tempered(temperature) if temperature != 1.0 else None
    step, iteration, rollouts, last = 0, 0, 0, 0.0
    hands = lambda: ctx.hands_left() is None or ctx.hands_left() > 0  # noqa: E731
    # Near the optimum each KL step moves the policy along its noise as much as
    # its signal, and neighbouring steps' noise is uncorrelated: the mean of the
    # iterates over the second half of the policy-gradient steps keeps the
    # signal and averages the noise out (Polyak). Rollouts still use the latest
    # iterate; only the saved result is the mean.
    tail = float(cfg.get("tail_average", 0.0))
    average, averaged, budget, last_batches = None, 0, None, None

    # Openings (ctx.starts): one-hand games from recorded rare windows (a kong
    # the teacher keeps, a self-drawn win it may pass up), where ordinary play
    # reaches one decision in hundreds. `starts_share` of each rollout's rows
    # come from them, the sets taken in turn so each gets an equal share.
    openings = [ctx.starts(name) for name in cfg.get("starts", [])]
    share = float(cfg.get("starts_share", 0.0)) if openings else 0.0
    turn = 0

    def rollout(distilling):
        """One rollout of target_rows decisions, or None when time or hands ran short."""
        nonlocal iteration, turn
        rows, opened = [], []
        ordinary = int(round(target_rows * (1.0 - share)))
        hooks = None if distilling or sampler is None else {"sample": sampler}
        with InferenceSession(actor) as session:
            while len(rows) < ordinary and hands() and ctx.time_left() > reserve:
                iteration += 1
                got, _, _ = ctx.collect(
                    actor, collect_cfg if distilling else pg_cfg, samples=min(40000, ordinary),
                    iteration=iteration,
                    labels=distilling, inference_session=session, hooks=hooks,
                )
                rows += got
            while share and len(rows) + len(opened) < target_rows and hands() and ctx.time_left() > reserve:
                iteration += 1
                got, _, _ = ctx.collect(
                    actor, collect_cfg if distilling else pg_cfg,
                    samples=min(20000, target_rows - len(rows) - len(opened)),
                    iteration=iteration, labels=distilling, inference_session=session, hooks=hooks,
                    starts=openings[turn % len(openings)],
                )
                turn += 1
                opened += got
        rows += opened
        if not rows or (len(rows) < target_rows and hands()):
            return None
        return rows

    while hands() and ctx.time_left() > 1.15 * last + reserve:
        began = time.time()
        distilling = step < int(cfg["distill_steps"])
        if distilling:
            rows = rollout(True)
            if rows is None:
                break
            step += 1
            collected = time.time() - began
            packing_started = time.time()
            batch_source = (resident_rows(rows, ctx.device, ctx.resources["gpu_limit_bytes"], oracle=False)
                            if cfg.get("resident_rows", False) else None)
            packing_seconds = time.time() - packing_started
            stats = distill(actor, rows, opt, microbatch, ctx.device, rng, int(cfg["distill_epochs"]),
                            hard=float(cfg.get("distill_hard", 0.0)), window=float(cfg.get("distill_window", 1.0)),
                            batch_source=batch_source, permuter=permuter)
            count = len(rows)
            del rows, batch_source
            ctx.save_actor(actor)
            last = time.time() - began
            ctx.log(step=step, rows=count, **stats, collect_seconds=collected, step_seconds=last,
                    packing_seconds=packing_seconds)
            torch.cuda.empty_cache()
            continue
        # One policy-gradient step sums the gradients of `chunks` rollouts of
        # the same policy: near the teacher's optimum each step's noise costs
        # quadratically, so a larger batch per KL step settles closer to it.
        # Each rollout is fitted, differentiated and freed before the next.
        if budget is None:
            # The policy-gradient budget as it stands before its first step.
            budget = ctx.hands_left() if ctx.hands_left() is not None else ctx.time_left()
            # A KL step spends its budget on the noise in every direction the
            # gradient can move; the more parameters, the more of it goes to
            # noise (the large Actor gained a third as much per hand). Only the
            # top transformer layers and the heads then take policy-gradient
            # steps; the representation distillation built stays fixed.
            top_layers = int(cfg.get("pg_top_layers", 0))
            if top_layers:
                blocks = actor.encoder.transformer.layers
                trainable = tuple(f"encoder.transformer.layers.{i}." for i in range(len(blocks) - top_layers, len(blocks)))
                trainable += ("encoder.norm.", "discard_head.", "claim_head.", "state_head.")
                for name, parameter in actor.named_parameters():
                    parameter.requires_grad_(name.startswith(trainable))
        params, direction, batches, parts, count = None, None, None, [], 0
        collected = fitted = 0.0
        for part in range(chunks):
            if part and not hands():
                break
            started = time.time()
            rows = rollout(False)
            if rows is None:
                break
            collected += time.time() - started
            started = time.time()
            batch_source = (resident_rows(rows, ctx.device, ctx.resources["gpu_limit_bytes"],
                                          oracle=not cfg.get("playout_baseline"))
                            if cfg.get("resident_rows", False) else None)
            rollouts += 1
            if cfg.get("playout_baseline"):
                weights, stats = playout_advantages(rows, cfg)
            else:
                weights, stats = advantages(
                    rows, cfg, rng, ctx.device, ctx.seed + 7919 * rollouts, batch_source=batch_source,
                )
            fitted += time.time() - started
            params, grad = gradient(
                actor, rows, weights, microbatch, ctx.device, batch_source=batch_source,
                capture=cfg.get("gradient_graphs", False),
            )
            direction = grad if direction is None else [d + g for d, g in zip(direction, grad)]
            # The KL check uses a subset of the latest rollout (same policy).
            subset = rng.choice(len(rows), size=min(len(rows), int(cfg["kl_rows"])), replace=False)
            del batches
            batches = [batch_source.batch(subset[s : s + 512]) if batch_source is not None
                       else make_batch([rows[i] for i in subset[s : s + 512]], ctx.device)
                       for s in range(0, len(subset), 512)]
            parts.append(stats)
            count += len(rows)
            del rows, batch_source, weights, grad
            torch.cuda.empty_cache()
        if direction is None:
            break
        step += 1
        direction = [d / len(parts) for d in direction]
        stats = {key: float(np.mean([p[key] for p in parts])) for key in parts[0]}
        left = ctx.hands_left() if ctx.hands_left() is not None else ctx.time_left()
        # Heat, then cool: the step KL falls linearly from step_kl_start to
        # step_kl over the policy-gradient budget, so the early large steps can
        # leave a local optimum the small ones then settle.
        goal = float(cfg["step_kl"])
        if cfg.get("step_kl_start"):
            done = min(1.0, max(0.0, 1.0 - left / budget))
            goal = float(cfg["step_kl_start"]) + (goal - float(cfg["step_kl_start"])) * done
        stats |= kl_step(actor, params, direction, batches, goal)
        stats["kl_goal"] = goal
        if tail > 0 and left <= tail * budget:
            averaged += 1
            with torch.no_grad():
                if average is None:
                    average = [p.detach().float().clone() for p in params]
                else:
                    for a, p in zip(average, params):
                        a += (p.detach().float() - a) / averaged
            stats["averaged"] = averaged
        # Kept on the host: device tensors held into the next step's rollout
        # would pin freed segments and fragment the capped allocator.
        last_batches = [{k: v.cpu() for k, v in mb.items()} for mb in batches]
        del params, direction, batches
        ctx.save_actor(actor)
        last = time.time() - began
        ctx.log(
            step=step, rows=count, rollouts=len(parts), **stats, collect_seconds=collected,
            critic_seconds=fitted, step_seconds=last,
        )
        torch.cuda.empty_cache()
    if average is not None:
        params = [p for p in actor.parameters() if p.requires_grad]
        last_batches = [{k: v.to(ctx.device) for k, v in mb.items()} for mb in last_batches]
        with torch.no_grad():
            final = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in last_batches]
            for p, a in zip(params, average):
                p.copy_(a.to(p.dtype))
            mean = [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in last_batches]
            kl = sum(float((f.exp().masked_fill(~mb["mask"], 0) * (f - m).masked_fill(~mb["mask"], 0)).sum())
                     for mb, f, m in zip(last_batches, final, mean)) / sum(len(f) for f in final)
            agree = sum(float((f.argmax(-1) == m.argmax(-1)).sum()) for f, m in zip(final, mean)) / sum(
                len(f) for f in final)
        ctx.save_actor(actor)
        ctx.log(step=step, averaged=averaged, average_kl=kl, average_agreement=agree)
    return actor
