"""Actor-only batches for frozen evaluation; training uses the full learning batch."""

import numpy as np
import torch

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


def policy_batch(rows, device):
    if not rows:
        raise ValueError("cannot infer an empty batch")
    batch = {}
    for key, shape, dtype in (
        ("planes", (N_PLANES, 34), np.float16),
        ("scalars", (N_SCALARS,), np.float16),
        ("mask", (N_ACTIONS,), bool),
    ):
        values = [np.asarray(row[key], dtype=dtype) for row in rows]
        if any(value.shape != shape for value in values):
            raise ValueError(f"incorrect {key} shape")
        array = np.stack(values)
        if dtype is not bool and not np.isfinite(array).all():
            raise FloatingPointError(f"nonfinite {key}")
        if key == "mask" and not array.any(axis=1).all():
            raise ValueError("each observation needs a legal action")
        batch[key] = torch.as_tensor(array, device=device)
    events = [np.asarray(row["events"], dtype=np.int64).reshape(-1, 5) for row in rows]
    padded = np.zeros((len(rows), max(1, max(map(len, events))), 5), dtype=np.int64)
    for i, value in enumerate(events):
        padded[i, : len(value)] = value
    batch["events"] = torch.as_tensor(padded, device=device)
    return batch


def evaluate_requests(pending, actor, history, device, max_batch=128):
    """Keep the same logits/log-softmax/argmax, but synchronize actions only once.

    Evaluation does not consume behavior probabilities or privileged values.
    Group ordering and within-group ordering match the original collector.
    """
    if max_batch < 1:
        raise ValueError("positive inference batch limit required")
    grouped = {}
    for item in pending:
        grouped.setdefault(item[2]["slot"], []).append(item)
    ordered, outputs, sizes = [], [], []
    with torch.inference_mode():
        for slot in sorted(grouped):
            for first in range(0, len(grouped[slot]), max_batch):
                group = grouped[slot][first : first + max_batch]
                batch = policy_batch([item[2]["row"] for item in group], device)
                model = actor if slot == "model:current" else history[slot]
                logits = model(
                    batch["planes"], batch["scalars"], batch["events"]
                ).float()
                actions = (
                    logits.masked_fill(~batch["mask"], -1e9).log_softmax(-1).argmax(-1)
                )
                ordered.extend(group)
                outputs.append(actions)
                sizes.append(len(group))
        if not outputs:
            return [], sizes
        actions = torch.cat(outputs).cpu().tolist()
    replies = [
        (conn, ident, (index, 0.0, 0.0))
        for (conn, ident, _), index in zip(ordered, actions, strict=True)
    ]
    return replies, sizes
