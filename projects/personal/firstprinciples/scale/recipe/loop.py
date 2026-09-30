"""Scale run: the large Actor's whole training under one fixed plan (ADR-0016, 2026-09-30 amendment).

The run is a chain of segments on one host, each an attempt the harness
evaluates and the next continues from. segment.json, written by the harness
into each segment's snapshot, says which segment this is (without it, the
policy gradient from the start of its schedules); config.json holds
the plan every segment reads its schedules from, so nothing changes on the
way. A schedule is a function of the hands its phase has trained, counted
from the planned sizes of the phase's earlier segments.

Imitation (the first segments): from a random initialisation, the learning
seat first plays the teacher's moves (behaviour cloning, the first
`bc_fraction` of the imitation's hands), then its own sampled moves while the
teacher labels every decision (DAgger: the states its own mistakes lead to).
The teacher is the RL line's best Actor, frozen. Each rollout's rows are
trained on for `distill_epochs` passes, the suits relabelled at random from
the second. Adam's rate falls linearly from `distill_learning_rate` at the
imitation's first hand to zero at its last; each segment's Adam starts afresh
and warms up over its first `warmup_updates` updates.

Policy gradient (the remaining segments): 098's recipe. Playout counterfactual
advantages; each step sums the gradients of several on-policy rollouts and
moves along them until KL(current || new) on a subset of the last one equals
the step KL; a segment's result is the mean of its iterates over the second
half of its hands (rollouts use the latest). Along the policy gradient's
hands the rollouts per step grow from `chunks_start` to `chunks_end` (the
critical batch grows as the loss falls), and the step KL holds at `step_kl`
until `kl_hold_hands`, then falls linearly to `step_kl_end` at the end. Every
step logs the gradient noise scale (the batch at which the gradient's noise
equals its signal), for diagnosis only.
"""

import json
from pathlib import Path
import time

import numpy as np
import torch

from selfplay_core.actions import HU
from guided.collection_session import InferenceSession

from .batch import actor_args, make_batch, masked_log_probs, resident_rows
from .symmetry import SuitPermuter
from .targets import playout_advantages

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "config.json").read_text())
SEGMENT = json.loads((HERE / "segment.json").read_text()) if (HERE / "segment.json").exists() else None


def place(cfg, index):
    """(phase of segment ``index``, hands its phase's earlier segments train, the phase's hands)."""
    segments = cfg["segments"]
    phase = segments[index]["phase"]
    before = sum(s["hands"] for s in segments[:index] if s["phase"] == phase)
    return phase, before, sum(s["hands"] for s in segments if s["phase"] == phase)


def rate_at(cfg, done, total):
    return float(cfg["distill_learning_rate"]) * max(0.0, 1.0 - done / total)


def chunks_at(cfg, done, total):
    low, high = int(cfg["chunks_start"]), int(cfg["chunks_end"])
    return low + min(high - low, int((high - low + 1) * done / total))


def kl_at(cfg, done, total):
    start, end, hold = float(cfg["step_kl"]), float(cfg["step_kl_end"]), float(cfg["kl_hold_hands"])
    return start if done <= hold else start + (end - start) * min(1.0, (done - hold) / (total - hold))


def gradient(actor, rows, weights, size, device, *, batch_source=None, capture=False):
    """Mean over every row of the advantage-weighted log-probability gradients.

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


def kl_step(actor, params, direction, batches, goal):
    """Move params to start + eta * direction with KL(start || new) ~= goal."""
    start = [p.detach().clone() for p in params]

    def logps():
        with torch.no_grad():
            return [masked_log_probs(actor(*actor_args(mb)), mb["mask"]) for mb in batches]

    reference = logps()
    probabilities = [old.exp().masked_fill(~mb["mask"], 0) for mb, old in zip(batches, reference)]
    latest = reference

    def kl_at_eta(eta):
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
    kl = kl_at_eta(eta)
    for _ in range(12):
        if abs(np.log(max(kl, 1e-12) / goal)) < 0.1:
            break
        eta *= float(np.clip((goal / max(kl, 1e-12)) ** 0.5, 0.2, 10.0))
        kl = kl_at_eta(eta)
    hu = [torch.nonzero(mb["mask"][:, HU]).squeeze(1) for mb in batches]
    legal = max(1, sum(len(h) for h in hu))
    before = int(torch.stack([(lp[h].argmax(-1) == HU).sum() for h, lp in zip(hu, reference)]).sum()) / legal
    after = int(torch.stack([(lp[h].argmax(-1) == HU).sum() for h, lp in zip(hu, latest)]).sum()) / legal
    return {"kl": kl, "step_norm": eta * norm, "hu_before": before, "hu_after": after}


def noise_scale(squares, mean_square, rows):
    """Gradient noise scale B = tr(Sigma) / |G|^2, in rows, from k equal rollouts' mean gradients.

    squares: |g_i|^2 of each rollout's mean gradient over its `rows` rows;
    mean_square: |mean_i g_i|^2. E|g_B|^2 = |G|^2 + tr(Sigma) / B at batch B,
    solved at B = rows and B = k * rows. None when the estimate of |G|^2 is
    not positive (the noise swamps it).
    """
    k = len(squares)
    small = float(np.mean(squares))
    signal = (k * mean_square - small) / (k - 1)
    noise = rows * k * (small - mean_square) / (k - 1)
    return noise / signal if signal > 0 else None


def rollout(ctx, actor, collect_cfg, target_rows, reserve, counter, *, labels):
    """One rollout of target_rows decisions, or None when time or hands ran short."""
    rows = []
    with InferenceSession(actor) as session:
        while len(rows) < target_rows and hands(ctx) and ctx.time_left() > reserve:
            counter[0] += 1
            got, _, _ = ctx.collect(actor, collect_cfg, samples=min(40000, target_rows), iteration=counter[0],
                                    labels=labels, inference_session=session)
            rows += got
    if not rows or (len(rows) < target_rows and hands(ctx)):
        return None
    return rows


def hands(ctx):
    return ctx.hands_left() is None or ctx.hands_left() > 0


def distill(actor, rows, opt, size, device, rng, epochs, rates, updates, warmup, *, batch_source=None,
            permuter=None):
    """Epochs of cross entropy toward the teacher's label; returns (stats, updates so far this segment).

    The rate moves linearly from rates[0] to rates[1] over the rollout's
    updates, scaled down while the segment's first `warmup` updates last.
    The first epoch's loss is taken before each batch is trained on (fresh
    rows). Later epochs relabel the suits of each batch at random (symmetry.py).
    """
    teacher = np.array([r["teacher"] for r in rows])
    labeled = np.flatnonzero(teacher >= 0)
    steps = max(1, epochs * -(-len(labeled) // size))
    actor.eval()
    total_ce, agree, k = 0.0, 0.0, 0
    for epoch in range(epochs):
        order = rng.permutation(labeled)
        for s in range(0, len(order), size):
            rate = (rates[0] + (rates[1] - rates[0]) * k / steps) * min(1.0, (updates + 1) / max(1, warmup))
            for group in opt.param_groups:
                group["lr"] = rate
            idx = order[s : s + size]
            mb = (batch_source.batch(idx) if batch_source is not None
                  else make_batch([rows[i] for i in idx], device))
            label = torch.as_tensor(teacher[idx], device=device)
            if epoch > 0 and permuter is not None:
                mb, label = permuter(mb, label, int(rng.integers(6)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = actor(*actor_args(mb))
            logp = masked_log_probs(logits, mb["mask"])
            loss = -logp.gather(1, label[:, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            k += 1
            updates += 1
            if epoch == 0:
                total_ce += float(loss) * len(idx)
                agree += float((logp.argmax(-1) == label).sum())
    opt.zero_grad(set_to_none=True)
    n = max(1, len(labeled))
    return {"labeled": len(labeled), "fresh_ce": total_ce / n, "fresh_agreement": agree / n}, updates


def imitate(ctx, cfg, before, total):
    """This segment's share of the imitation: behaviour cloning, then DAgger, at a falling rate."""
    if cfg["teacher"] not in ctx.teachers:
        raise ValueError(f"teacher {cfg['teacher']} is not in ctx.teachers: {ctx.teachers}")
    if before == 0:
        # The run starts from a random initialisation, not from the host's start Actor.
        torch.manual_seed(ctx.seed)
        actor = ctx.new_actor()
    else:
        actor = ctx.actor
    collect_cfg = {"history_probability": 0.0, "finish_game": "recipe.returns:finish_game", "teacher": cfg["teacher"]}
    target_rows = cfg["smoke_rows"] if ctx.smoke else cfg["rows"]
    reserve = 5 if ctx.smoke else 60
    rng = np.random.default_rng(ctx.seed)
    opt = torch.optim.Adam(actor.parameters(), lr=float(cfg["distill_learning_rate"]), eps=1e-5)
    permuter = SuitPermuter(ctx.device)
    counter, step, updates, last = [0], 0, 0, 0.0
    while hands(ctx) and ctx.time_left() > 1.15 * last + reserve:
        began = time.time()
        first = before + ctx.hands
        plays = first < float(cfg["bc_fraction"]) * total
        rows = rollout(ctx, actor, dict(collect_cfg, teacher_plays=plays), target_rows, reserve, counter, labels=True)
        if rows is None:
            break
        step += 1
        collected = time.time() - began
        rates = (rate_at(cfg, first, total), rate_at(cfg, before + ctx.hands, total))
        batch_source = (resident_rows(rows, ctx.device, ctx.resources["gpu_limit_bytes"], oracle=False)
                        if cfg.get("resident_rows", False) else None)
        stats, updates = distill(actor, rows, opt, int(cfg["microbatch"]), ctx.device, rng, int(cfg["distill_epochs"]),
                                 rates, updates, int(cfg["warmup_updates"]), batch_source=batch_source,
                                 permuter=permuter)
        count = len(rows)
        del rows, batch_source
        ctx.save_actor(actor)
        last = time.time() - began
        ctx.log(step=step, phase="bc" if plays else "dagger", imitation_hands=before + ctx.hands, rows=count,
                lr_start=rates[0], lr_end=rates[1], updates=updates, **stats, collect_seconds=collected,
                step_seconds=last)
        torch.cuda.empty_cache()
    return actor


def improve(ctx, cfg, before, total):
    """This segment's share of the policy gradient, on the schedules of rollouts per step and step KL."""
    actor = ctx.actor
    pg_cfg = {"history_probability": 0.0, "critic_features": "recipe.features:dealer",
              "finish_game": "recipe.returns:finish_game", "playout_baseline": True}
    target_rows = cfg["smoke_rows"] if ctx.smoke else cfg["rows"]
    microbatch = int(cfg["microbatch"])
    reserve = 5 if ctx.smoke else 60
    rng = np.random.default_rng(ctx.seed)
    # Segments before `pg_top_layers_until` step only the top transformer
    # layers, the last norm and the heads (098 on the small Actor); later ones
    # step every parameter.
    top_layers = int(cfg.get("pg_top_layers", 0)) if before < float(cfg.get("pg_top_layers_until", 0)) else 0
    if top_layers:
        blocks = actor.encoder.transformer.layers
        trainable = tuple(f"encoder.transformer.layers.{i}." for i in range(len(blocks) - top_layers, len(blocks)))
        trainable += ("encoder.norm.", "discard_head.", "claim_head.", "state_head.")
        for name, parameter in actor.named_parameters():
            parameter.requires_grad_(name.startswith(trainable))
    # Near the optimum each KL step moves the policy along its noise as much as
    # its signal, and neighbouring steps' noise is uncorrelated: the mean of the
    # iterates over the second half of the segment keeps the signal and
    # averages the noise out (Polyak). Rollouts still use the latest iterate.
    tail = float(cfg.get("tail_average", 0.0))
    budget = ctx.hands_left() if ctx.hands_left() is not None else ctx.time_left()
    counter, step, last = [0], 0, 0.0
    average, averaged, last_batches = None, 0, None
    while hands(ctx) and ctx.time_left() > 1.15 * last + reserve:
        began = time.time()
        done = before + ctx.hands
        chunks, goal = chunks_at(cfg, done, total), kl_at(cfg, done, total)
        params, direction, batches, parts, squares, count = None, None, None, [], [], 0
        collected = fitted = 0.0
        for part in range(chunks):
            if part and not hands(ctx):
                break
            started = time.time()
            rows = rollout(ctx, actor, pg_cfg, target_rows, reserve, counter, labels=False)
            if rows is None:
                break
            collected += time.time() - started
            started = time.time()
            batch_source = (resident_rows(rows, ctx.device, ctx.resources["gpu_limit_bytes"], oracle=False)
                            if cfg.get("resident_rows", False) else None)
            weights, stats = playout_advantages(rows, cfg)
            fitted += time.time() - started
            params, grad = gradient(actor, rows, weights, microbatch, ctx.device, batch_source=batch_source,
                                    capture=cfg.get("gradient_graphs", False))
            squares.append(float(sum((g.double() ** 2).sum() for g in grad)))
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
        if len(parts) > 1:
            mean_square = float(sum((d.double() ** 2).sum() for d in direction))
            scale = noise_scale(squares, mean_square, count / len(parts))
            stepped = before + ctx.hands - done
            stats |= {"noise_scale_rows": scale,
                      "noise_scale_hands": scale * stepped / count if scale is not None else None}
        stats |= kl_step(actor, params, direction, batches, goal)
        left = ctx.hands_left() if ctx.hands_left() is not None else ctx.time_left()
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
        ctx.log(step=step, pg_hands=before + ctx.hands, rows=count, rollouts=len(parts), chunks=chunks,
                step_kl=goal, top_layers=top_layers, **stats, collect_seconds=collected, advantage_seconds=fitted,
                step_seconds=last)
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


def train(ctx):
    if SEGMENT is None:
        # Outside the run (the RL line's copy of a segment that became its best
        # version): the policy gradient from the start of its schedules.
        phase, before, total = "pg", 0, place(CONFIG, len(CONFIG["segments"]) - 1)[2]
    else:
        phase, before, total = place(CONFIG, int(SEGMENT["index"]))
    ctx.log(segment=SEGMENT and int(SEGMENT["index"]), phase=phase, hands_before=before, phase_hands=total)
    return (imitate if phase == "imitation" else improve)(ctx, CONFIG, before, total)
