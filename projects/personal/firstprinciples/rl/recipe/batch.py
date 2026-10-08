"""Microbatch padding and masked log-probabilities for learner rows."""

import numpy as np
import torch
from torch.nn import functional as F

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


def make_batch(rows, device, *, oracle=False):
    batch = {}
    fields = [
        ("planes", (N_PLANES, 34), np.float16),
        ("scalars", (N_SCALARS,), np.float16),
        ("mask", (N_ACTIONS,), bool),
    ]
    if oracle:
        fields += [("oracle_planes", (16, 34), np.float32),
                   ("oracle_scalars", (4,), np.float32)]
        if rows and "opp_waits" in rows[0]:
            fields.append(("opp_waits", (3, 34), np.float32))
    for key, shape, dtype in fields:
        batch[key] = torch.as_tensor(
            np.stack([np.asarray(row[key], dtype=dtype) for row in rows]), device=device
        )
    events = [np.asarray(row["events"], dtype=np.int16).reshape(-1, 5) for row in rows]
    padded = np.zeros((len(rows), max(1, max(len(e) for e in events)), 5), dtype=np.int16)
    for i, value in enumerate(events):
        padded[i, : len(value)] = value
    batch["events"] = torch.as_tensor(padded, device=device)
    batch["action"] = torch.as_tensor([row["action"] for row in rows], dtype=torch.long, device=device)
    return batch


class ResidentRows:
    """Immutable rollout columns, shared by both Critic folds and the Actor.

    Histories stay ragged in storage. Each gather reproduces make_batch's
    padding, dtypes and order, including repeated indices and empty histories.
    Metadata/rewards stay in the original rows; only learner inputs live here.
    """

    FIELDS = (
        ("planes", np.float16), ("scalars", np.float16), ("mask", bool),
        ("action", np.int64), ("oracle_planes", np.float32),
        ("oracle_scalars", np.float32),
    )

    @staticmethod
    def storage_bytes(rows):
        fixed = 2 * (N_PLANES * 34 + N_SCALARS) + N_ACTIONS + 8 + 4 * (16 * 34 + 4)
        return len(rows) * (fixed + 16) + 10 * (1 + sum(len(r["events"]) for r in rows))

    def __init__(self, rows, device):
        self.device = device
        events = [np.asarray(r["events"], dtype=np.int16).reshape(-1, 5) for r in rows]
        self.lengths = np.array([len(e) for e in events], dtype=np.int64)
        # Index zero is always a padding sentinel, including an all-empty rollout.
        self.events = torch.as_tensor(
            np.concatenate([np.zeros((1, 5), dtype=np.int16), *events]), device=device
        )
        self.offsets = torch.as_tensor(np.r_[1, 1 + np.cumsum(self.lengths)[:-1]], device=device)
        self.device_lengths = torch.as_tensor(self.lengths, device=device)
        self.fields = {
            key: torch.as_tensor(np.stack([np.asarray(r[key], dtype=dtype) for r in rows]), device=device)
            for key, dtype in self.FIELDS
        }

    def batch(self, indices, *, oracle=False):
        indices = np.asarray(indices, dtype=np.int64)
        width = max(1, int(self.lengths[indices].max()))
        index = torch.as_tensor(indices, device=self.device)
        positions = torch.arange(width, device=self.device)[None, :]
        valid = positions < self.device_lengths[index, None]
        locations = (self.offsets[index, None] + positions).masked_fill(~valid, 0)
        batch = {key: value.index_select(0, index) for key, value in self.fields.items()
                 if oracle or not key.startswith("oracle_")}
        batch["events"] = self.events[locations].masked_fill(~valid[..., None], 0)
        return batch


class ActorRows(ResidentRows):
    """Only the Actor's inputs and actions: steps that fit no Critic skip the oracle columns."""

    FIELDS = (("planes", np.float16), ("scalars", np.float16), ("mask", bool), ("action", np.int64))

    @staticmethod
    def storage_bytes(rows):
        fixed = 2 * (N_PLANES * 34 + N_SCALARS) + N_ACTIONS + 8
        return len(rows) * (fixed + 16) + 10 * (1 + sum(len(r["events"]) for r in rows))


def resident_rows(rows, device, gpu_limit_bytes, *, oracle=True):
    """Use residency only within the existing cap, leaving room for backward.

    Without `oracle` (distillation, playout advantages) the Critic's columns,
    about a third of each row, stay on the host.
    """
    if torch.device(device).type != "cuda":
        return None
    store = ResidentRows if oracle else ActorRows
    allocated = torch.cuda.memory_allocated(device)
    free, _ = torch.cuda.mem_get_info(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    available = min(gpu_limit_bytes - allocated, free + reusable)
    if store.storage_bytes(rows) + 2 * 1024**3 > available:
        return None
    return store(rows, device)


def actor_args(batch):
    return batch["planes"], batch["scalars"], batch["events"]


def masked_log_probs(logits, mask):
    # A finite sentinel keeps the arithmetic finite for illegal actions.
    return F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
