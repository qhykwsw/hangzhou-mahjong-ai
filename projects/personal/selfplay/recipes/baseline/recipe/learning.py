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
    """
    if not rows:
        raise ValueError("cannot make an empty batch")
    batch = {}
    for key, shape, dtype in (
        ("planes", (N_PLANES, 34), np.float16),
        ("scalars", (N_SCALARS,), np.float16),
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
    """One rollout: PPO for the Actor under a KL budget, full epochs for the Critic.

    Config keys: clip, epochs, minibatch (microbatch), effective_minibatch,
    target_kl, kl_stop_factor, entropy, lambda_max, actor_lr, warmup_steps,
    actor_step (Actor steps already taken in this run), critic_only, deadline.

    KL to the behaviour policy is measured on each effective minibatch before
    its step. Above kl_stop_factor * target_kl the Actor stops for the rest of
    this rollout and that minibatch's Actor gradients are discarded; the Critic
    keeps fitting every minibatch of every epoch, since its target does not
    depend on how far the Actor moved. Fresh Adam moments make the first steps
    sign-like at full size, so the Actor learning rate warms up linearly over
    warmup_steps. The caller adapts actor_lr from the returned kl_max.
    """
    if not rows:
        raise ValueError("cannot update from empty rows")
    if not math.isfinite(w) or not 0 <= w <= 1:
        raise ValueError("teacher guidance w must be finite and in [0, 1]")
    micro = int(cfg.get("minibatch", 128))
    effective = int(cfg.get("effective_minibatch", 512))
    epochs = int(cfg.get("epochs", 2))
    clip = float(cfg.get("clip", 0.1))
    target_kl = float(cfg.get("target_kl", 0.01))
    stop_threshold = target_kl * float(cfg.get("kl_stop_factor", 2.0))
    entropy_coef = float(cfg.get("entropy", 0.001))
    lambda_max = float(cfg.get("lambda_max", 1.0))
    actor_lr = float(cfg["actor_lr"])
    warmup = int(cfg.get("warmup_steps", 0))
    actor_step = int(cfg.get("actor_step", 0))
    deadline = cfg.get("deadline", float("inf"))
    if min(micro, effective, epochs) < 1 or not 0 < clip < 1:
        raise ValueError("invalid PPO batch size, epochs or clip")
    if not all(
        math.isfinite(x) and x >= 0
        for x in (target_kl, stop_threshold, entropy_coef, lambda_max, actor_lr)
    ):
        raise ValueError("invalid PPO coefficients")
    critic_only = bool(cfg.get("critic_only", False))
    coefficient = 0.0 if critic_only else lambda_max * w
    # A history buffer is only dereferenced when its loss is actually enabled.
    use_replay = coefficient > 0 and bool(replay)
    device = next(actor.parameters()).device
    count = len(rows)
    required = {"planes", "scalars", "events", "mask", "action", "logprob"}
    required |= {"value", "reward", "adv", "ret", "oracle_planes", "oracle_scalars"}
    if coefficient:
        required |= {"teacher", "label_version"}
    for row in rows:
        if missing := required - row.keys():
            raise ValueError(f"missing training fields: {sorted(missing)}")
    adv = np.asarray([row["adv"] for row in rows], dtype=np.float64)
    returns = torch.as_tensor(
        [row["ret"] for row in rows], dtype=torch.float32, device=device
    )
    if not np.isfinite(adv).all():
        raise FloatingPointError("nonfinite rollout advantages")
    _finite(returns, "rollout returns")
    normalized_adv = ((adv - adv.mean()) / max(adv.std(), 1e-8)).astype(np.float32)
    critic.head.update_stats(returns, critic_opt)
    actor.train()
    critic.train()
    actor_keys = ("policy_loss", "teacher_loss", "entropy", "approx_kl", "clipfrac")
    actor_sums = dict.fromkeys(actor_keys, 0.0)
    value_sum = 0.0
    actor_active = not critic_only
    actor_steps = critic_steps = actor_samples = critic_samples = 0
    kl_max = 0.0
    stopped_kl = None
    budget_exhausted = False
    try:
        for _ in range(epochs):
            order = _permutation(rng, count)
            for start in range(0, count, effective):
                if time.time() >= deadline:
                    budget_exhausted = True
                    break
                group = order[start : start + effective]
                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
                group_actor = dict.fromkeys(actor_keys, 0.0)
                group_value = 0.0
                for offset in range(0, len(group), micro):
                    if time.time() >= deadline:
                        budget_exhausted = True
                        break
                    indices = group[offset : offset + micro]
                    mb = make_batch([rows[int(i)] for i in indices], device)
                    weight = len(indices) / len(group)
                    if actor_active:
                        logits = actor(*_actor_args(mb))
                        logp = masked_log_probs(logits, mb["mask"])
                        entropy = (
                            -(logp.exp() * logp.masked_fill(~mb["mask"], 0))
                            .sum(-1)
                            .mean()
                        )
                        _check_actions(mb["action"], mb["mask"], "sampled")
                        log_ratio = (
                            logp.gather(1, mb["action"][:, None]).squeeze(1)
                            - mb["logprob"]
                        )
                        ratio = log_ratio.exp()
                        _finite(ratio, "PPO ratio")
                        advantages = torch.as_tensor(
                            normalized_adv[indices], device=device
                        )
                        policy_loss = -torch.minimum(
                            ratio * advantages,
                            ratio.clamp(1 - clip, 1 + clip) * advantages,
                        ).mean()
                        teacher_loss = logits.new_zeros(())
                        if coefficient > 0:
                            _check_actions(mb["teacher"], mb["mask"], "teacher")
                            teacher_loss = -logp.gather(
                                1, mb["teacher"][:, None]
                            ).mean()
                            if use_replay:
                                # Weight is exactly 90/10, regardless of integer rounding.
                                old = make_batch(
                                    _sample_replay(
                                        rng, replay, max(1, math.ceil(len(indices) / 9))
                                    ),
                                    device,
                                )
                                old_loss = teacher_cross_entropy(
                                    actor(*_actor_args(old)),
                                    old["mask"],
                                    old["teacher"],
                                )
                                teacher_loss = 0.9 * teacher_loss + 0.1 * old_loss
                        actor_loss = (
                            coefficient * teacher_loss
                            + policy_loss
                            - entropy_coef * entropy
                        )
                        _finite(actor_loss, "actor loss")
                        (actor_loss * weight).backward()
                        values = dict(
                            policy_loss=policy_loss,
                            teacher_loss=teacher_loss,
                            entropy=entropy,
                            approx_kl=((ratio - 1) - log_ratio).mean(),
                            clipfrac=((ratio - 1).abs() > clip).float().mean(),
                        )
                        for key, value in values.items():
                            group_actor[key] += float(value.detach()) * weight
                    predictions = critic.normalized_value(*_critic_args(mb))
                    value_loss = F.mse_loss(
                        predictions, critic.head.normalize(mb["ret"])
                    )
                    _finite(value_loss, "value loss")
                    (value_loss * weight).backward()
                    group_value += float(value_loss.detach()) * weight
                if budget_exhausted or time.time() >= deadline:
                    budget_exhausted = True
                    actor_opt.zero_grad(set_to_none=True)
                    critic_opt.zero_grad(set_to_none=True)
                    break
                if actor_active:
                    kl_max = max(kl_max, group_actor["approx_kl"])
                    if group_actor["approx_kl"] > stop_threshold:
                        actor_active = False
                        stopped_kl = group_actor["approx_kl"]
                        actor_opt.zero_grad(set_to_none=True)
                    else:
                        taken = actor_step + actor_steps + 1
                        scale = min(1.0, taken / warmup) if warmup > 0 else 1.0
                        for param_group in actor_opt.param_groups:
                            param_group["lr"] = actor_lr * scale
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(), 1.0, error_if_nonfinite=True
                        )
                        actor_opt.step()
                        actor_steps += 1
                        actor_samples += len(group)
                        for key, value in group_actor.items():
                            actor_sums[key] += value * len(group)
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), 1.0, error_if_nonfinite=True
                )
                critic_opt.step()
                critic_steps += 1
                critic_samples += len(group)
                value_sum += group_value * len(group)
            if budget_exhausted:
                break
        for name, model in (("actor", actor), ("critic", critic)):
            for parameter in model.parameters():
                _finite(parameter, f"{name} parameter")
    finally:
        actor.eval()
        critic.eval()
    stats = {key: value / max(1, actor_samples) for key, value in actor_sums.items()}
    stats.update(
        value_loss=value_sum / max(1, critic_samples),
        samples=count,
        optimized_samples=actor_samples,
        sgd_steps=actor_steps,
        critic_steps=critic_steps,
        kl_max=kl_max,
        early_stop=stopped_kl is not None,
        stop_kl=stopped_kl or 0.0,
        actor_lr=actor_lr,
        critic_only=critic_only,
        teacher_coefficient=coefficient,
        budget_exhausted=budget_exhausted,
        replay_fraction=0.1 if use_replay else 0.0,
        popart_mean=float(critic.head.mean),
        popart_std=float(critic.head.std),
    )
    return stats
