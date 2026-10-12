"""Exact-shape Actor gradient replay over the immutable resident rollout."""

from collections import Counter

import numpy as np
import torch

from .batch import policy_log_probs
from .cuda_execution import warmup_stream


class ResidentGradient:
    def __init__(self, actor, source, weights, count, eligible):
        self.actor = actor
        self.source = source
        self.weights = torch.as_tensor(weights, device=source.device)
        self.count = count
        self.eligible = eligible
        self.params = [p for p in actor.parameters() if p.requires_grad]
        for parameter in self.params:
            parameter.grad = torch.zeros_like(parameter)
        self.entries = {}
        self.pool = None

    def backward(self, index, active, positions):
        source = self.source
        valid = positions < source.device_lengths[index, None]
        locations = (source.offsets[index, None] + positions).masked_fill(~valid, 0)
        events = source.events[locations].masked_fill(~valid[..., None], 0)
        # The Actor never receives the privileged Critic columns.
        fields = {
            key: value.index_select(0, index)
            for key, value in source.fields.items()
            if not key.startswith("oracle_")
        }
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = self.actor(fields["planes"], fields["scalars"], events)
        chosen = (
            policy_log_probs(logits, fields["mask"])
            .gather(1, fields["action"][:, None])
            .squeeze(1)
        )
        ((self.weights[index] * active * chosen).sum() / self.count).backward()

    def entry(self, shape):
        if shape in self.entries:
            return self.entries[shape]
        if self.entries:
            # Sorted lengths never return to an earlier shape.
            torch.cuda.synchronize()
            self.entries.clear()
            self.pool = None
            torch.cuda.empty_cache()
        size, width = shape
        device = self.source.device
        index = torch.zeros(size, dtype=torch.int64, device=device)
        active = torch.zeros((), dtype=torch.float32, device=device)
        positions = torch.arange(width, device=device)[None, :]
        graph = None
        if shape in self.eligible:
            side = warmup_stream(device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.backward(index, active, positions)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # Warmup/capture contribute zero, retaining any accumulated gradient.
            with torch.cuda.graph(graph, pool=self.pool):
                self.backward(index, active, positions)
            self.pool = graph.pool()
        active.fill_(1)
        result = index, active, positions, graph
        self.entries[shape] = result
        return result

    def add(self, indices):
        width = max(1, int(self.source.lengths[indices].max()))
        index, active, positions, graph = self.entry((len(indices), width))
        # The pageable array must finish copying before its next reuse.
        index.copy_(torch.from_numpy(indices))
        if graph is None:
            self.backward(index, active, positions)
        else:
            graph.replay()


def gradient(actor, rows, weights, size, device, *, batch_source=None, capture=True):
    assert batch_source is not None
    actor.eval()
    order = np.argsort(batch_source.lengths, kind="stable")
    batches = [order[start : start + size] for start in range(0, len(order), size)]
    counts = Counter(
        (len(indices), max(1, int(batch_source.lengths[indices].max())))
        for indices in batches
    )
    # Rare long-tail shapes do not repay capture or its retained device storage.
    eligible = {shape for shape, uses in counts.items() if capture and uses >= 8}
    runner = ResidentGradient(actor, batch_source, weights, len(rows), eligible)
    try:
        for indices in batches:
            runner.add(indices)
        torch.cuda.synchronize()
        return runner.params, [p.grad for p in runner.params]
    finally:
        # No stored closure points back to runner; release captures at step end.
        runner.entries.clear()
        for parameter in runner.params:
            parameter.grad = None
