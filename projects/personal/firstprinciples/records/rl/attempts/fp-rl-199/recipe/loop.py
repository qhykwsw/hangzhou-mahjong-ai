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


def dense_deviations(rate):
    """A sample hook that deviates at a fixed share of the decisions.

    Sampling the policy deviates at ~3.7% of decisions, ~0.4 per hand, and a
    step's three rollouts then agree on a direction at cosine ~0.01 (183): the
    step is almost all noise. Each deviation's advantage is the exact paired
    difference against the argmax continuation on its wall whatever drew it,
    so deviating at `rate` of the decisions with a choice (the alternative
    drawn among the non-argmax actions by the policy's own preference) gets
    several times the paired samples from the same hands; every other decision
    plays the argmax. A state's rows weigh rate / (1 - pi(argmax)) times their
    on-policy weight: a positive rescaling per state, like the tempered hook's.
    """
    def sample(logits, mask):
        logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        probs = logp.exp()
        best = probs.argmax(-1)
        others = probs.scatter(1, best[:, None], 0.0)
        others = others / others.sum(-1, keepdim=True).clamp_min(1e-12)
        onehot = torch.zeros_like(probs).scatter(1, best[:, None], 1.0)
        choice = (mask.sum(-1) > 1)[:, None]
        mixed = torch.where(choice, (1 - rate) * onehot + rate * others, onehot)
        actions = torch.multinomial(mixed, 1).squeeze(1)
        return actions, mixed.gather(1, actions[:, None]).squeeze(1).clamp_min(1e-12).log()

    return sample


def boosted_deviations(boost):
    """A sample hook that deviates `boost` times as often as the policy would, at the same states.

    With a stable wall and the draw luck subtracted, a deviation's paired
    advantage is still ~11 base units of noise around a gap of a few tenths,
    and the policy deviates at ~3.9% of its decisions (184): a step's three
    rollouts agree on a direction at cosine ~0.03. Under cfg["vine"] the hand
    plays the argmax and each deviation is played in a copy, so deviating more
    often adds paired samples without moving the hand off the argmax path. A
    decision deviates with probability min(1, boost * (1 - pi(argmax))), the
    alternative drawn as the policy would among the non-argmax actions: on the
    policy's own distribution of deviations, `boost` times over. Where the
    rate is capped at 1 the row carries the weight boost * (1 - pi(argmax)) >= 1
    that keeps each state's share what sampling the policy would give it;
    the hook returns the log of that weight in place of a behaviour logprob
    (targets.playout_advantages reads it, nothing else does).
    """
    def sample(logits, mask):
        logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        probs = logp.exp()
        best = probs.argmax(-1)
        leave = (1.0 - probs.gather(1, best[:, None]).squeeze(1)).clamp_min(0.0)
        rate = (boost * leave).clamp(max=1.0)
        others = probs.scatter(1, best[:, None], 0.0)
        others = others / others.sum(-1, keepdim=True).clamp_min(1e-12)
        onehot = torch.zeros_like(probs).scatter(1, best[:, None], 1.0)
        mixed = (1.0 - rate)[:, None] * onehot + rate[:, None] * others
        actions = torch.multinomial(mixed, 1).squeeze(1)
        weight = (boost * leave / rate.clamp_min(1e-12)).clamp_min(1.0)
        return actions, weight.log()

    return sample


def search_rows(rows):
    """Rows for every searched window's other candidates, their Q gain over the argmax, and summary stats."""
    extra, gains = [], []
    windows, best_alt, gain_sum, fish, fish_hu = 0, 0, 0.0, 0, 0
    # The policy's own choice at the windows (the first candidate is its
    # argmax), to watch it drift against the rule's (131: kong 244/1500 at
    # kong windows, win 996/1500 at fish windows).
    kong, kong_kong, fish_argmax_hu = 0, 0, 0
    for row in rows:
        actions, q = row.get("search_actions"), row.get("search_q")
        if not actions or not q or len(actions) < 2 or not row.get("search_worlds"):
            continue
        windows += 1
        best = int(np.argmax(q))
        best_alt += best != 0
        gain_sum += q[best] - q[0]
        if row.get("search_kind") == "fish":
            fish += 1
            fish_hu += actions[best] == HU
            fish_argmax_hu += actions[0] == HU
        elif row.get("search_kind") == "kong":
            kong += 1
            kong_kong += GANG_EXPOSED <= actions[0] < HU
        for action, value in zip(actions[1:], q[1:]):
            extra.append(dict(row, action=int(action)))
            gains.append(float(value) - float(q[0]))
    stats = {"search_windows": windows, "search_best_alt": best_alt / max(1, windows),
             "search_gain": gain_sum / max(1, windows), "search_fish_hu": fish_hu / max(1, fish),
             "argmax_fish_hu": fish_argmax_hu / max(1, fish), "argmax_kong_kong": kong_kong / max(1, kong)}
    return extra, np.asarray(gains, dtype=np.float64), stats


def fisher_product(actor, params, batches, vector):
    """Mean Fisher-vector product of the policy over the KL rows, J^T (diag(pi) - pi pi^T) J v.

    J v is a central difference of the logits (the fused attention has no double
    backward); J^T u is one backward pass. Illegal actions have pi = 0 and drop out.
    """
    start = [p.detach().clone() for p in params]
    norm = float(torch.sqrt(sum((v.float() * v.float()).sum() for v in vector)))
    eps = 1e-3 / max(norm, 1e-12)
    total = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    rows = sum(len(mb["mask"]) for mb in batches)
    for mb in batches:
        args = actor_args(mb)
        with torch.no_grad():
            for p, p0, v in zip(params, start, vector):
                p.copy_(p0 + eps * v)
            plus = actor(*args).float()
            for p, p0, v in zip(params, start, vector):
                p.copy_(p0 - eps * v)
            minus = actor(*args).float()
            for p, p0 in zip(params, start):
                p.copy_(p0)
            jv = ((plus - minus) / (2 * eps)).masked_fill(~mb["mask"], 0)
        logits = actor(*args).float()
        with torch.no_grad():
            pi = masked_log_probs(logits.detach(), mb["mask"]).exp().masked_fill(~mb["mask"], 0)
            u = pi * (jv - (pi * jv).sum(-1, keepdim=True))
        for acc, g in zip(total, torch.autograd.grad((logits * u).sum() / rows, params, allow_unused=True)):
            if g is not None:
                acc += g.float()
    return total


def natural_direction(actor, params, direction, batches, damping, iterations):
    """F^-1 g by conjugate gradient, F the policy's Fisher on the KL rows.

    The KL step measures the move in the policy's own metric, but the plain
    gradient moves a logit in proportion to its action's probability: where the
    policy is nearly sure, a rarely played action that the playouts find better
    barely moves, and the argmax the Actor is deployed with never changes. The
    natural direction (TRPO) moves each decision by its advantage instead and
    spends the same KL where the argmax can still change. Damping is relative to
    the curvature along the plain gradient, so no direction is stretched more
    than 1 / damping times past it.
    """
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]

    def flat(xs):
        return torch.cat([x.float().reshape(-1) for x in xs])

    def unflat(v):
        return [part.reshape(shape) for part, shape in zip(torch.split(v, sizes), shapes)]

    g = flat(direction)
    gg = float(g @ g)
    fg = flat(fisher_product(actor, params, batches, direction))
    curvature = float(g @ fg) / max(gg, 1e-30)
    lam = damping * curvature
    x = torch.zeros_like(g)
    r = g.clone()
    p = r.clone()
    ap = fg + lam * g
    rr = gg
    done = 0
    for done in range(1, iterations + 1):
        alpha = rr / max(float(p @ ap), 1e-30)
        x += alpha * p
        r -= alpha * ap
        new = float(r @ r)
        if new < 1e-8 * gg or done == iterations:
            rr = new
            break
        p = r + (new / rr) * p
        rr = new
        ap = flat(fisher_product(actor, params, batches, unflat(p))) + lam * p
    cosine = float(x @ g) / max(float(torch.sqrt((x @ x) * gg)), 1e-30)
    stats = {"natural_curvature": curvature, "natural_residual": (rr / max(gg, 1e-30)) ** 0.5,
             "natural_cosine": cosine, "natural_iterations": done}
    return [d.to(q.dtype) for d, q in zip(unflat(x), direction)], stats


def kl_step(actor, params, direction, batches, goal, restore=False):
    """Move params to start + eta * direction with KL(start || new) ~= goal.

    `restore` puts the start back afterwards (to measure a direction only).
    """
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
    # Decisions whose argmax (the deployed choice) the step changed, overall and
    # where a kong or a win is legal.
    rows = sum(len(lp) for lp in reference)
    flips = [lp.argmax(-1) != new.argmax(-1) for lp, new in zip(reference, latest)]
    windows = [mb["mask"][:, GANG_EXPOSED:HU + 1].any(1) for mb in batches]
    flip = int(torch.stack([f.sum() for f in flips]).sum()) / max(1, rows)
    flip_window = (int(torch.stack([(f & w).sum() for f, w in zip(flips, windows)]).sum())
                   / max(1, int(torch.stack([w.sum() for w in windows]).sum())))
    if restore:
        with torch.no_grad():
            for p, p0 in zip(params, start):
                p.copy_(p0)
    return {"kl": kl, "step_norm": eta * norm, "hu_before": before, "hu_after": after,
            "flip": flip, "flip_window": flip_window}


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
    # The framework's variance reducers (guided.walls): a stable wall keeps each
    # seat's draws when a branch calls, the draw luck is subtracted from the
    # paired differences (targets.playout_advantages), and openings reshuffle
    # the wall past their recorded position. Under vine the hand plays the
    # argmax and each deviation is played in its copy.
    for key in ("stable_wall", "luck", "starts_reshuffle", "vine"):
        if cfg.get(key):
            collect_cfg[key] = True
    # cfg["stage"] (guided.stage): every training game sits at a random game of
    # a tournament stage; the targets value hands by the chance of advancing.
    if cfg.get("stage"):
        collect_cfg["stage"] = {k: cfg["stage"][k] for k in ("kind", "games", "per_game_sd")}
    # Policy-gradient rollouts ask for the greedy playout of every deviation.
    pg_cfg = dict(collect_cfg, playout_baseline=bool(cfg.get("playout_baseline")))
    if cfg["teacher"] not in ctx.teachers:
        raise ValueError(f"teacher {cfg['teacher']} is not in ctx.teachers: {ctx.teachers}")
    # The smoke run (about a minute) must still exercise one whole step.
    target_rows = cfg["smoke_rows"] if ctx.smoke else cfg["rows"]
    microbatch = int(cfg["microbatch"])
    reserve = 5 if ctx.smoke else 60
    rng = np.random.default_rng(ctx.seed)
    # `fresh_actor` starts from a randomly initialised Actor of the same shape,
    # so nothing of an earlier imitation or teacher carries over.
    actor = ctx.new_actor().to(ctx.device) if cfg.get("fresh_actor") else ctx.actor
    # The first distill_steps rollouts only distil the teacher on the learner's
    # own states (DAgger, where imitation gains fastest); the rest only take
    # policy-gradient steps. Interleaving the two made each undo the other.
    opt = torch.optim.Adam(actor.parameters(), lr=float(cfg["distill_learning_rate"]), eps=1e-5)
    permuter = SuitPermuter(ctx.device)
    chunks = max(1, int(cfg["chunks_per_step"]))
    temperature = float(cfg.get("sample_temperature", 1.0))
    sampler = tempered(temperature) if temperature != 1.0 else None
    if cfg.get("deviation_share"):
        sampler = dense_deviations(float(cfg["deviation_share"]))
    if cfg.get("deviation_boost"):
        if not cfg.get("vine"):
            raise ValueError("deviation_boost needs vine: the hand must stay on the argmax path")
        sampler = boosted_deviations(float(cfg["deviation_boost"]))
    step, iteration, rollouts, last = 0, 0, 0, 0.0
    hands = lambda: ctx.hands_left() is None or ctx.hands_left() > 0  # noqa: E731
    # Near the optimum each KL step moves the policy along its noise as much as
    # its signal, and neighbouring steps' noise is uncorrelated: the mean of the
    # iterates over the second half of the policy-gradient steps keeps the
    # signal and averages the noise out (Polyak). Rollouts still use the latest
    # iterate; only the saved result is the mean.
    tail = float(cfg.get("tail_average", 0.0))
    average, averaged, budget, last_batches = None, 0, None, None
    improver = None

    # Openings (ctx.starts): one-hand games from recorded rare windows (a kong
    # the teacher keeps, a self-drawn win it may pass up), where ordinary play
    # reaches one decision in hundreds. `starts_share` of each rollout's rows
    # come from them, the sets taken in turn so each gets an equal share. Each
    # collection gets a random `starts_per_call` of the set: every worker holds
    # its own copy, and a whole decoded set is about 1 GB.
    openings = [ctx.starts(name) for name in cfg.get("starts", [])]
    per_call = int(cfg.get("starts_per_call", 1000))
    share = float(cfg.get("starts_share", 0.0)) if openings else 0.0
    turn = 0
    if cfg.get("expert"):
        from .expert import train as expert_train

        return expert_train(ctx, actor, cfg, collect_cfg, openings, rng, hands, reserve)

    # The first `bc_steps` distillation rollouts let the teacher play the
    # learning seat (behaviour cloning): a random Actor's own games reach few
    # of the states the teacher needs labelled. DAgger takes over after them.
    bc_cfg = dict(collect_cfg, teacher_plays=True)

    def rollout(distilling, bc=False):
        """One rollout of target_rows decisions, or None when time or hands ran short."""
        nonlocal iteration, turn
        rows, opened = [], []
        ordinary = int(round(target_rows * (1.0 - share)))
        hooks = None if distilling or sampler is None else {"sample": sampler}
        play_cfg = (bc_cfg if bc else collect_cfg) if distilling else pg_cfg
        open_cfg = dict(pg_cfg, search=dict(cfg["search"])) if cfg.get("search") else pg_cfg
        with InferenceSession(actor) as session:
            while len(rows) < ordinary and hands() and ctx.time_left() > reserve:
                iteration += 1
                got, _, _ = ctx.collect(
                    actor, play_cfg, samples=min(40000, ordinary),
                    iteration=iteration,
                    labels=distilling, inference_session=session, hooks=hooks,
                )
                rows += got
            while share and len(rows) + len(opened) < target_rows and hands() and ctx.time_left() > reserve:
                iteration += 1
                pool = openings[turn % len(openings)]
                got, _, _ = ctx.collect(
                    actor, open_cfg if not distilling else play_cfg,
                    samples=min(20000, target_rows - len(rows) - len(opened)),
                    iteration=iteration, labels=distilling, inference_session=session, hooks=hooks,
                    starts=[pool[i] for i in rng.choice(len(pool), min(len(pool), per_call), replace=False)],
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
            rows = rollout(True, bc=step < int(cfg.get("bc_steps", 0)))
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
            # cfg["mpo"] (mpo.py): a Q network fitted to the kept deviations
            # replaces the gradient step after `mpo_warmup` steps.
            if cfg.get("mpo"):
                from .mpo import Improver

                improver = Improver(ctx, actor, cfg, rng)
        params, direction, batches, parts, count = None, None, None, [], 0
        chunk_grads = []
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
            # Searched windows (cfg["search"]): each other candidate becomes a
            # row with its advantage over the argmax, the mean over worlds that
            # keep the opponents' real hands. These worlds know the hidden
            # tiles, so no single row decides: across rows with the same
            # public information the advantages average to its expectation,
            # the way a playout deviation's single wall does.
            extra, gains, searched = search_rows(rows) if cfg.get("search") else ([], np.zeros(0), {})
            if extra:
                batch_source = (resident_rows(rows + extra, ctx.device, ctx.resources["gpu_limit_bytes"],
                                              oracle=not cfg.get("playout_baseline"))
                                if cfg.get("resident_rows", False) else None)
            if cfg.get("playout_baseline"):
                weights, stats = playout_advantages(rows, cfg)
                if improver is not None:
                    improver.add_chunk(actor, rows, weights * stats["adv_scale"])
                if cfg.get("search"):
                    # Same units as the playout advantages, so the same scale;
                    # `search_weight` keeps tens of thousands of searched
                    # windows a step from swamping the ordinary rows.
                    scale = stats["adv_scale"] if stats["adv_scale"] > 1e-6 else 1.0
                    scale /= float(cfg.get("search_weight", 1.0))
                    weights = np.concatenate([weights, (gains / scale).astype(np.float32)])
                    stats |= searched
            else:
                weights, stats = advantages(
                    rows, cfg, rng, ctx.device, ctx.seed + 7919 * rollouts, batch_source=batch_source,
                )
            fitted += time.time() - started
            params, grad = gradient(
                actor, rows + extra if cfg.get("search") else rows, weights, microbatch, ctx.device,
                batch_source=batch_source, capture=cfg.get("gradient_graphs", False),
            )
            direction = grad if direction is None else [d + g for d, g in zip(direction, grad)]
            # Each rollout's own gradient, flattened: how far the rollouts agree
            # on a direction says how much of it is signal rather than noise.
            chunk_grads.append(torch.cat([g.float().reshape(-1) for g in grad]))
            # The KL check uses a subset of the latest rollout (same policy).
            subset = rng.choice(len(rows), size=min(len(rows), int(cfg["kl_rows"])), replace=False)
            # Kong and win decisions share their heads across rare windows and
            # ordinary play but are rare in a random subset, so the KL budget
            # would not see them move: `kl_window_rows` of them join it.
            if cfg.get("kl_window_rows"):
                special = np.flatnonzero([bool(np.asarray(r["mask"])[GANG_EXPOSED:HU + 1].any()) for r in rows])
                if len(special):
                    picked = rng.choice(special, size=min(len(special), int(cfg["kl_window_rows"])), replace=False)
                    subset = np.unique(np.concatenate([subset, picked]))
            del batches
            batches = [batch_source.batch(subset[s : s + 512]) if batch_source is not None
                       else make_batch([rows[i] for i in subset[s : s + 512]], ctx.device)
                       for s in range(0, len(subset), 512)]
            parts.append(stats)
            count += len(rows)
            del rows, extra, batch_source, weights, grad
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
        if len(chunk_grads) > 1:
            cosines = [float(a @ b) / max(float(torch.sqrt((a @ a) * (b @ b))), 1e-30)
                       for i, a in enumerate(chunk_grads) for b in chunk_grads[i + 1:]]
            stats["chunk_cosine"] = float(np.mean(cosines))
        del chunk_grads
        if improver is not None:
            stats |= improver.fit()
            if step > int(cfg.get("mpo_warmup", 2)):
                improved, more = improver.direction(actor, params, batches, goal)
                a = torch.cat([d.float().reshape(-1) for d in improved])
                b = torch.cat([d.float().reshape(-1) for d in direction])
                more["mpo_pg_cosine"] = float(a @ b) / max(float(torch.sqrt((a @ a) * (b @ b))), 1e-30)
                stats |= more
                direction = improved
            else:
                improver.states = []
        if cfg.get("natural_damping"):
            # The plain direction at the same KL, measured and undone, for comparison.
            plain = kl_step(actor, params, direction, batches, goal, restore=True)
            stats |= {"plain_kl": plain["kl"], "plain_flip": plain["flip"], "plain_flip_window": plain["flip_window"]}
            direction, natural = natural_direction(actor, params, direction, batches, float(cfg["natural_damping"]),
                                                   int(cfg.get("natural_iterations", 10)))
            stats |= natural
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
