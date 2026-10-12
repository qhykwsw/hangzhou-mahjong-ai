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

from selfplay_core.actions import HU
from guided.collection_session import InferenceSession

from .batch import actor_args, make_batch, masked_log_probs, resident_rows
from .targets import advantages

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


class OracleHead(torch.nn.Module):
    """Training-only head on the Actor's tile tokens: the hidden tiles per tile id.

    It predicts the Critic's oracle planes (each opponent's hand and the wall,
    count >= 1..4) from what the Actor's encoder made of the public
    observation. It lives outside the Actor, so the returned weights keep the
    fixed structure; only the shared encoder learns to infer hidden tiles.
    """

    def __init__(self, dim):
        super().__init__()
        self.out = torch.nn.Linear(dim, 16)

    def forward(self, tiles):
        return self.out(tiles).transpose(1, 2)


def distill(actor, rows, opt, size, device, rng, epochs, *, batch_source=None, oracle=None):
    """Epochs of cross entropy toward the teacher's label on the learner's own states.

    DAgger: the learner played these rows with its own policy, so its mistakes
    lead to states behaviour cloning never visited; the teacher labels them.
    The first epoch's loss is taken before each batch is trained on (fresh rows).
    With oracle = (head, weight), the encoder also predicts the hidden tiles.
    """
    teacher = np.array([r["teacher"] for r in rows])
    labeled = np.flatnonzero(teacher >= 0)
    actor.eval()
    total_ce, agree, total_aux = 0.0, 0.0, 0.0
    captured = {}
    hook = (actor.encoder.register_forward_hook(lambda module, args, out: captured.__setitem__("tiles", out[1]))
            if oracle else None)
    try:
        # Each later epoch reshuffles; only the first one reports fresh metrics.
        for epoch in range(epochs):
            order = rng.permutation(labeled)
            for s in range(0, len(order), size):
                idx = order[s : s + size]
                mb = (batch_source.batch(idx, oracle=bool(oracle)) if batch_source is not None
                      else make_batch([rows[i] for i in idx], device, oracle=bool(oracle)))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = actor(*actor_args(mb))
                    hidden = oracle[0](captured.pop("tiles")) if oracle else None
                logp = masked_log_probs(logits, mb["mask"])
                label = torch.as_tensor(teacher[idx], device=device)
                ce = -logp.gather(1, label[:, None]).mean()
                loss = ce
                if oracle:
                    aux = torch.nn.functional.binary_cross_entropy_with_logits(hidden.float(), mb["oracle_planes"])
                    loss = ce + oracle[1] * aux
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
                opt.step()
                if epoch == 0:
                    total_ce += float(ce.detach()) * len(idx)
                    agree += float((logp.argmax(-1) == label).sum())
                    if oracle:
                        total_aux += float(aux.detach()) * len(idx)
    finally:
        if hook is not None:
            hook.remove()
    opt.zero_grad(set_to_none=True)
    n = max(1, len(labeled))
    return {"labeled": len(labeled), "fresh_ce": total_ce / n, "fresh_agreement": agree / n,
            "oracle_bce": total_aux / n}


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
    oracle_head = OracleHead(actor.dim).to(ctx.device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(oracle_head.parameters()),
                           lr=float(cfg["distill_learning_rate"]), eps=1e-5)
    step, iteration, last = 0, 0, 0.0
    hands = lambda: ctx.hands_left() is None or ctx.hands_left() > 0  # noqa: E731
    while hands() and ctx.time_left() > 1.15 * last + reserve:
        began = time.time()
        rows = []
        distilling = step < int(cfg["distill_steps"])
        with InferenceSession(actor) as session:
            while len(rows) < target_rows and hands() and ctx.time_left() > reserve:
                iteration += 1
                got, _, _ = ctx.collect(
                    actor, collect_cfg, samples=min(40000, target_rows), iteration=iteration,
                    labels=distilling, inference_session=session,
                )
                rows += got
        if not rows or (len(rows) < target_rows and hands()):
            break
        step += 1
        collected = time.time() - began
        packing_started = time.time()
        batch_source = (resident_rows(rows, ctx.device, ctx.resources["gpu_limit_bytes"])
                        if cfg.get("resident_rows", False) else None)
        packing_seconds = time.time() - packing_started
        if distilling:
            stats = distill(actor, rows, opt, microbatch, ctx.device, rng, int(cfg["distill_epochs"]),
                            batch_source=batch_source,
                            oracle=(oracle_head, float(cfg["oracle_aux_weight"])))
            count = len(rows)
            del rows, batch_source
            ctx.save_actor(actor)
            last = time.time() - began
            ctx.log(step=step, rows=count, **stats, collect_seconds=collected, step_seconds=last,
                    packing_seconds=packing_seconds)
            torch.cuda.empty_cache()
            continue
        weights, stats = advantages(
            rows, cfg, rng, ctx.device, ctx.seed + 7919 * step, batch_source=batch_source,
        )
        fitted = time.time() - began - collected
        params, direction = gradient(
            actor, rows, weights, microbatch, ctx.device, batch_source=batch_source,
            capture=cfg.get("gradient_graphs", False),
        )
        subset = rng.choice(len(rows), size=min(len(rows), int(cfg["kl_rows"])), replace=False)
        batches = [batch_source.batch(subset[s : s + 512]) if batch_source is not None
                   else make_batch([rows[i] for i in subset[s : s + 512]], ctx.device)
                   for s in range(0, len(subset), 512)]
        count = len(rows)
        used_resident = batch_source is not None
        del rows, batch_source
        stats |= kl_step(actor, params, direction, batches, float(cfg["step_kl"]))
        ctx.save_actor(actor)
        last = time.time() - began
        ctx.log(
            step=step, rows=count, **stats, collect_seconds=collected,
            critic_seconds=fitted, step_seconds=last,
            packing_seconds=packing_seconds, resident_rows=used_resident,
        )
        torch.cuda.empty_cache()
    return actor
