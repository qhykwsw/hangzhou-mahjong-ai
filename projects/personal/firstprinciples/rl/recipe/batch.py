"""Microbatch padding and masked log-probabilities for learner rows."""

import numpy as np
import torch
from torch.nn import functional as F

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


def make_batch(rows, device):
    batch = {}
    for key, shape, dtype in (
        ("planes", (N_PLANES, 34), np.float16),
        ("scalars", (N_SCALARS,), np.float16),
        ("mask", (N_ACTIONS,), bool),
        ("oracle_planes", (16, 34), np.float32),
        ("oracle_scalars", (4,), np.float32),
    ):
        batch[key] = torch.as_tensor(
            np.stack([np.asarray(row[key], dtype=dtype) for row in rows]), device=device
        )
    events = [np.asarray(row["events"], dtype=np.int64).reshape(-1, 5) for row in rows]
    padded = np.zeros((len(rows), max(1, max(len(e) for e in events)), 5), dtype=np.int64)
    for i, value in enumerate(events):
        padded[i, : len(value)] = value
    batch["events"] = torch.as_tensor(padded, device=device)
    batch["action"] = torch.as_tensor([row["action"] for row in rows], dtype=torch.long, device=device)
    return batch


def actor_args(batch):
    return batch["planes"], batch["scalars"], batch["events"]


def masked_log_probs(logits, mask):
    # A finite sentinel keeps the arithmetic finite for illegal actions.
    return F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
