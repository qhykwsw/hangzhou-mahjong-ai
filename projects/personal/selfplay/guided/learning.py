"""Teacher-assisted on-policy PPO, with separate Actor and PopArt Critic updates.

Only each microbatch is padded/transferred to the device. In particular a full
rollout with variable, complete histories is never padded on the GPU at once.
The caller owns trajectory/GAE construction and the versioned demonstration
cache. Replay participates only in teacher cross entropy, never the PPO ratio.
"""

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


def make_batch(rows, device):
    """Pad row dictionaries; compact features are promoted inside the networks.

    Missing RL fields are allowed for replay-only demonstrations, but update()
    explicitly requires them for on-policy rows. label_version stays a list of
    original strings, rather than being overwritten with the current teacher.
    When any row has the learner's oracle inputs (guided.oracle), the batch
    has them too, "oracle_input_planes" [n, 16, 34] and "oracle_input_scalars"
    [n, 4] float16, zero for a row without (oracle.argument gives the Actor's
    argument); otherwise it has no such keys.
    """
    if not rows:
        raise ValueError("cannot make an empty batch")
    # Encodings differ in their numbers of planes and scalars (guided.encoding.INPUTS); a batch has one.
    planes = len(rows[0]["planes"]) if "planes" in rows[0] else N_PLANES
    scalars = len(rows[0]["scalars"]) if "scalars" in rows[0] else N_SCALARS
    if planes < N_PLANES or scalars < N_SCALARS:
        raise ValueError("incorrect planes shape")
    batch = {}
    for key, shape, dtype in (
        ("planes", (planes, 34), np.float16),
        ("scalars", (scalars,), np.float16),
        ("mask", (N_ACTIONS,), bool),
        ("oracle_planes", (16, 34), np.float32),
        ("oracle_scalars", (4,), np.float32),
    ):
        values = [
            np.asarray(row.get(key, np.zeros(shape)), dtype=dtype) for row in rows
        ]
        if any(value.shape != shape for value in values):
            raise ValueError(f"incorrect {key} shape; expected {shape}")
        array = np.stack(values)
        if dtype is not bool and not np.isfinite(array).all():
            raise FloatingPointError(f"nonfinite {key}")
        batch[key] = torch.as_tensor(array, device=device)
    events = [np.asarray(row["events"], dtype=np.int64).reshape(-1, 5) for row in rows]
    length = max(1, max(len(value) for value in events))
    padded = np.zeros((len(rows), length, 5), dtype=np.int64)
    for i, value in enumerate(events):
        padded[i, : len(value)] = value
    batch["events"] = torch.as_tensor(padded, device=device)
    from .oracle import FIELDS as ORACLE_FIELDS, stack

    stacked = stack(rows)
    if stacked is not None:
        for key, value in zip(ORACLE_FIELDS, stacked):
            batch[key] = torch.as_tensor(value, device=device)
    for key in ("action", "teacher"):
        default = -1 if key == "teacher" else 0
        batch[key] = torch.as_tensor(
            [row.get(key, default) for row in rows],
            dtype=torch.long,
            device=device,
        )
    for key in ("logprob", "value", "reward", "adv", "ret"):
        values = np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"nonfinite {key}")
        batch[key] = torch.as_tensor(values, device=device)
    batch["label_version"] = [str(row.get("label_version", "")) for row in rows]
    return batch


def masked_log_probs(logits, mask):
    if logits.shape != mask.shape or not bool(mask.any(-1).all()):
        raise ValueError("each policy row must have at least one legal action")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("nonfinite policy logits")
    # A finite sentinel keeps entropy finite without 0 * -inf arithmetic.
    return F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)


def _check_actions(actions, mask, name):
    if bool(((actions < 0) | (actions >= mask.shape[-1])).any()):
        raise ValueError(f"{name} action is out of range")
    if not bool(mask.gather(1, actions[:, None]).all()):
        raise ValueError(f"{name} action is illegal")


def teacher_cross_entropy(logits, mask, teacher):
    _check_actions(teacher, mask, "teacher")
    return -masked_log_probs(logits, mask).gather(1, teacher[:, None]).mean()


def _actor_args(batch):
    return batch["planes"], batch["scalars"], batch["events"]


def _actor_kwargs(batch):
    """The learner's oracle inputs as its Actor saw them in collection (guided.oracle), when the rows have them."""
    from .oracle import argument

    oracle = argument(batch)
    return {} if oracle is None else {"oracle": oracle}


def _critic_args(batch):
    return (*_actor_args(batch), batch["oracle_planes"], batch["oracle_scalars"])


def _permutation(rng, count):
    if hasattr(rng, "permutation"):
        return list(rng.permutation(count))
    indices = list(range(count))
    rng.shuffle(indices)
    return indices


def _sample_replay(rng, replay, count):
    if hasattr(rng, "integers"):
        indices = rng.integers(0, len(replay), size=count)
    elif hasattr(rng, "randrange"):
        indices = [rng.randrange(len(replay)) for _ in range(count)]
    else:
        indices = rng.randint(0, len(replay), size=count)
    return [replay[int(i)] for i in indices]


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}")


def update(actor, critic, actor_opt, critic_opt, rows, cfg, w, replay, rng):
    """Run up to cfg['epochs'] PPO passes, accumulating microbatch gradients.

    Config keys: clip=.1, epochs=2, minibatch=128, effective_minibatch=512,
    target_kl=.01, entropy=.001, lambda_max=1, bc_only=False. Learning rates
    belong to the supplied optimizers (the runner may schedule them).
    For bc_only, teacher CE alone is optimized and Critic/statistics are untouched.
    For critic_only, the Actor is frozen and only the value loss is optimized.
    In joint mode w=0 avoids all label/replay access; teacher=-1 is then valid.
    Otherwise rows labelled -1 (the teacher failed there) stay out of the teacher loss.
    KL is measured before stepping over a complete effective minibatch. Exceeding
    target_kl discards its gradients and stops both updates for this rollout.
    """
    if not rows:
        raise ValueError("cannot update from empty rows")
    if not math.isfinite(w) or not 0 <= w <= 1:
        raise ValueError("teacher guidance w must be finite and in [0, 1]")
    bc_only = bool(cfg.get("bc_only", False))
    micro = int(cfg.get("minibatch", 128))
    effective = int(cfg.get("effective_minibatch", 512))
    epochs = int(cfg.get("epochs", 2))
    clip = float(cfg.get("clip", 0.1))
    target_kl = float(cfg.get("target_kl", 0.01))
    entropy_coef = float(cfg.get("entropy", 0.001))
    lambda_max = float(cfg.get("lambda_max", 1.0))
    if min(micro, effective, epochs) < 1 or not 0 < clip < 1:
        raise ValueError("invalid PPO batch size, epochs or clip")
    if not all(
        math.isfinite(x) and x >= 0 for x in (target_kl, entropy_coef, lambda_max)
    ):
        raise ValueError("invalid PPO coefficients")
    critic_only = bool(cfg.get("critic_only", False))
    if critic_only and bc_only:
        raise ValueError("critic_only and bc_only are exclusive")
    coefficient = 1.0 if bc_only else 0.0 if critic_only else lambda_max * w
    # A history buffer is only dereferenced when its loss is actually enabled.
    use_replay = coefficient > 0 and bool(replay)
    device = next(actor.parameters()).device
    count = len(rows)
    required = {"planes", "scalars", "events", "mask"}
    if not bc_only:
        required |= {
            "action",
            "logprob",
            "value",
            "reward",
            "adv",
            "ret",
            "oracle_planes",
            "oracle_scalars",
        }
    if coefficient:
        required |= {"teacher", "label_version"}
    for row in rows:
        if missing := required - row.keys():
            raise ValueError(f"missing training fields: {sorted(missing)}")
    if bc_only:
        normalized_adv = np.zeros(count, dtype=np.float32)
    else:
        adv = np.asarray([row["adv"] for row in rows], dtype=np.float64)
        returns = torch.as_tensor(
            [row["ret"] for row in rows],
            dtype=torch.float32,
            device=device,
        )
        if not np.isfinite(adv).all():
            raise FloatingPointError("nonfinite rollout advantages")
        _finite(returns, "rollout returns")
        normalized_adv = ((adv - adv.mean()) / max(adv.std(), 1e-8)).astype(np.float32)
        critic.head.update_stats(returns, critic_opt)
    actor.train()
    if not bc_only:
        critic.train()
    metrics = {
        key: 0.0
        for key in (
            "loss",
            "policy_loss",
            "teacher_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
        )
    }
    accepted_samples = 0
    steps = 0
    early_stop = False
    budget_exhausted = False
    stop_kl = 0.0
    try:
        for _ in range(epochs):
            order = _permutation(rng, count)
            for start in range(0, count, effective):
                if time.time() >= cfg.get("deadline", float("inf")):
                    budget_exhausted = True
                    break
                group = order[start : start + effective]
                actor_opt.zero_grad(set_to_none=True)
                if not bc_only:
                    critic_opt.zero_grad(set_to_none=True)
                group_metrics = dict.fromkeys(metrics, 0.0)
                for offset in range(0, len(group), micro):
                    if time.time() >= cfg.get("deadline", float("inf")):
                        budget_exhausted = True
                        break
                    indices = group[offset : offset + micro]
                    mb = make_batch([rows[int(i)] for i in indices], device)
                    logits = actor(*_actor_args(mb), **_actor_kwargs(mb))
                    logp = masked_log_probs(logits, mb["mask"])
                    entropy = (
                        -(logp.exp() * logp.masked_fill(~mb["mask"], 0)).sum(-1).mean()
                    )
                    zero = logits.new_zeros(())
                    policy_loss = approx_kl = clipfrac = value_loss = zero
                    if not bc_only:
                        _check_actions(mb["action"], mb["mask"], "sampled")
                        log_ratio = (
                            logp.gather(1, mb["action"][:, None]).squeeze(1)
                            - mb["logprob"]
                        )
                        ratio = log_ratio.exp()
                        advantages = torch.as_tensor(
                            normalized_adv[indices], device=device
                        )
                        _finite(ratio, "PPO ratio")
                        policy_loss = -torch.minimum(
                            ratio * advantages,
                            ratio.clamp(1 - clip, 1 + clip) * advantages,
                        ).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean().detach()
                        clipfrac = ((ratio - 1).abs() > clip).float().mean().detach()
                        predictions = critic.normalized_value(*_critic_args(mb))
                        value_loss = F.mse_loss(
                            predictions, critic.head.normalize(mb["ret"])
                        )
                    teacher_loss = zero
                    if coefficient > 0:
                        # Rows whose teacher failed (label -1) carry no label.
                        labeled = mb["teacher"] >= 0
                        if bool(labeled.any()):
                            _check_actions(mb["teacher"][labeled], mb["mask"][labeled], "teacher")
                            teacher_loss = -logp[labeled].gather(1, mb["teacher"][labeled][:, None]).mean()
                        else:
                            teacher_loss = logp.sum() * 0.0  # keeps the backward graph
                        if use_replay:
                            # Weight is exactly 90/10, regardless of integer sample rounding.
                            old = make_batch(
                                _sample_replay(
                                    rng, replay, max(1, math.ceil(len(indices) / 9))
                                ),
                                device,
                            )
                            old_loss = teacher_cross_entropy(
                                actor(*_actor_args(old), **_actor_kwargs(old)), old["mask"], old["teacher"]
                            )
                            teacher_loss = 0.9 * teacher_loss + 0.1 * old_loss
                    actor_loss = coefficient * teacher_loss
                    if not bc_only:
                        actor_loss = actor_loss + policy_loss - entropy_coef * entropy
                    _finite(actor_loss, "actor loss")
                    _finite(value_loss, "value loss")
                    weight = len(indices) / len(group)
                    if not critic_only:
                        (actor_loss * weight).backward()
                    if not bc_only:
                        (value_loss * weight).backward()
                    values = dict(
                        loss=actor_loss.detach() + value_loss.detach(),
                        policy_loss=policy_loss.detach(),
                        teacher_loss=teacher_loss.detach(),
                        value_loss=value_loss.detach(),
                        entropy=entropy.detach(),
                        approx_kl=approx_kl.detach(),
                        clipfrac=clipfrac.detach(),
                    )
                    for key, value in values.items():
                        group_metrics[key] += float(value) * weight
                if time.time() >= cfg.get("deadline", float("inf")):
                    budget_exhausted = True
                if budget_exhausted:
                    actor_opt.zero_grad(set_to_none=True)
                    critic_opt.zero_grad(set_to_none=True)
                    break
                if (
                    not bc_only
                    and not critic_only
                    and group_metrics["approx_kl"] > target_kl
                ):
                    early_stop, stop_kl = True, group_metrics["approx_kl"]
                    actor_opt.zero_grad(set_to_none=True)
                    critic_opt.zero_grad(set_to_none=True)
                    break
                if not critic_only:
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), 1.0, error_if_nonfinite=True
                    )
                if not bc_only:
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), 1.0, error_if_nonfinite=True
                    )
                if not critic_only:
                    actor_opt.step()
                if not bc_only:
                    critic_opt.step()
                accepted_samples += len(group)
                steps += 1
                for key, value in group_metrics.items():
                    metrics[key] += value * len(group)
            if early_stop or budget_exhausted:
                break
        for name, model in (("actor", actor), ("critic", critic)):
            if name == "critic" and bc_only:
                continue
            for parameter in model.parameters():
                _finite(parameter, f"{name} parameter")
    finally:
        actor.eval()
        if not bc_only:
            critic.eval()
    stats = {key: value / max(1, accepted_samples) for key, value in metrics.items()}
    stats.update(
        samples=count,
        optimized_samples=accepted_samples,
        sgd_steps=steps,
        early_stop=early_stop,
        stop_kl=stop_kl,
        critic_only=critic_only,
        teacher_coefficient=coefficient,
        budget_exhausted=budget_exhausted,
        replay_fraction=0.1 if use_replay else 0.0,
        popart_mean=float(critic.head.mean),
        popart_std=float(critic.head.std),
    )
    return stats
