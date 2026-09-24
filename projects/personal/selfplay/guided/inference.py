"""Actor-only batches for frozen evaluation; training uses the full learning batch."""

import numpy as np
import torch

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


def prepare_frozen(actor, device):
    """Precast immutable inference weights to their existing autocast dtype.

    Embeddings, residuals and layer norms stay fp32. Each linear sees exactly
    the fp16 coefficients autocast would produce, but no longer converts them
    on every decision (including GPU copy kernels inside a CUDA graph).
    Never apply this to the learning Actor: its fp32 parameters are optimized.
    """
    if torch.device(device).type in ("cuda", "mps"):
        for module in actor.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.MultiheadAttention)):
                module.half()
        actor._frozen_fp16 = True
    return actor


def policy_logits(model, planes, scalars, events):
    """Eager inference also handles frozen precast weights, without CUDA graphs."""
    if not getattr(model, "_frozen_fp16", False):
        return model(planes, scalars, events)
    with torch.autocast(planes.device.type, torch.float16, cache_enabled=False):
        return model(planes, scalars, events)


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
                logits = policy_logits(
                    model, batch["planes"], batch["scalars"], batch["events"]
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


class ActorGraphs:
    """CUDA graphs of a model's decision, one per padded input shape.

    The collection parent answers many small batches; launching a few dozen
    kernels from Python per batch costs more than the kernels. Inputs are
    written into pinned host buffers of a few padded shapes and copied into
    the graph's static tensors; the replayed computation is the eager one
    (forward, mask, log-softmax, then multinomial sampling or argmax and the
    chosen log-probability), so its outputs are identical in distribution and,
    for argmax, in value. Shapes beyond the buckets run eagerly.
    """

    BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
    EVENT_BUCKET, MAX_WIDTH = 32, 256

    IN_FLIGHT = 2  # launches of one model and shape that may be unread at a time

    def __init__(self):
        self.graphs, self.hosts, self.turn, self.pool = {}, {}, {}, None

    @staticmethod
    def shape(rows):
        size = next((b for b in ActorGraphs.BATCH_BUCKETS if b >= len(rows)), None)
        width = (
            -(
                -max(1, max(len(np.asarray(r["events"]).reshape(-1, 5)) for r in rows))
                // ActorGraphs.EVENT_BUCKET
            )
            * ActorGraphs.EVENT_BUCKET
        )
        return (
            (size, width)
            if size is not None and width <= ActorGraphs.MAX_WIDTH
            else None
        )

    def _host(self, key, size, width):
        # A ring of IN_FLIGHT buffer sets per model and shape: a buffer is not
        # refilled while its earlier copy may still be on the way, and other
        # models' batches or the next round may be launched in between.
        ring = self.hosts.get((key, size, width))
        if ring is None:
            ring = self.hosts[(key, size, width)] = []
            for _ in range(self.IN_FLIGHT):
                tensors = (
                    torch.zeros((size, N_PLANES, 34), dtype=torch.float16).pin_memory(),
                    torch.zeros((size, N_SCALARS), dtype=torch.float16).pin_memory(),
                    torch.zeros((size, width, 5), dtype=torch.int64).pin_memory(),
                    torch.zeros((size, N_ACTIONS), dtype=torch.bool).pin_memory(),
                )
                views = [t.numpy() for t in tensors]
                views[3][:, 0] = (
                    True  # padding rows stay valid; they never reach a caller
                )
                answers = torch.zeros((size, 2), dtype=torch.float32).pin_memory()
                ring.append((tensors, views, answers, torch.cuda.Event()))
        turn = self.turn.get((key, size, width), 0)
        self.turn[(key, size, width)] = (turn + 1) % self.IN_FLIGHT
        return ring[turn]

    @staticmethod
    def _decide(model, planes, scalars, events, mask, sample):
        # fp16 tensor cores: the large pool Actor answers about twice as fast
        # (measured 1.8-2.6x on a 4070), the small one 1.3-2x. Collection only
        # samples from these policies, so the last-bit logit differences from
        # fp32 change nothing that the training measures; evaluation is the
        # research model service, not this class.
        with torch.autocast("cuda", torch.float16, cache_enabled=False):
            logits = model(planes, scalars, events).float()
        logp = logits.masked_fill(~mask, -1e9).log_softmax(-1)
        actions = (
            torch.multinomial(logp.exp(), 1).squeeze(1) if sample else logp.argmax(-1)
        )
        chosen = logp.gather(1, actions[:, None]).squeeze(1)
        return torch.stack((actions.float(), chosen), -1)

    def decide(self, key, model, rows, sample):
        """Launch (action, log-probability) per row: sampled from the policy, or its argmax.

        Returns a Decided whose result() waits for the device. The answers are
        copied straight out of the graph into a pinned buffer on the same
        stream: graphs share one memory pool, so a later replay of any other
        graph may overwrite this one's device output. Up to IN_FLIGHT calls
        for one model and shape may be unread (their host buffers rotate), so
        the device runs one batch while the host fills the next.
        """
        shape = self.shape(rows)
        device = next(model.parameters()).device
        if shape is None:
            batch = policy_batch(rows, device)
            return Decided(
                self._decide(
                    model,
                    batch["planes"],
                    batch["scalars"],
                    batch["events"],
                    batch["mask"],
                    sample,
                )
            )
        size, width = shape
        tensors, (planes, scalars, events, mask), answers, done = self._host(
            key, size, width
        )
        for i, row in enumerate(rows):
            planes[i] = row["planes"]
            scalars[i] = row["scalars"]
            history = np.asarray(row["events"]).reshape(-1, 5)
            events[i, : len(history)] = history
            events[i, len(history) :] = 0
            mask[i] = row["mask"]
        entry = self.graphs.get((key, size, width, sample))
        if entry is None:
            static = [t.to(device) for t in tensors]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._decide(model, *static, sample)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self._decide(model, *static, sample)
            self.pool = graph.pool()
            entry = self.graphs[(key, size, width, sample)] = (static, out, graph)
        static, out, graph = entry
        for target, source in zip(static, tensors):
            target.copy_(source, non_blocking=True)
        graph.replay()
        answers.copy_(out, non_blocking=True)
        done.record()
        return Decided(answers.numpy()[: len(rows)], done)


class Decided:
    """A launched decision; result() gives its rows as a numpy array once the device is done."""

    def __init__(self, rows, event=None):
        self.rows, self.event = rows, event

    def result(self):
        if self.event is not None:
            self.event.synchronize()
            return self.rows
        return self.rows.cpu().numpy() if hasattr(self.rows, "cpu") else self.rows
