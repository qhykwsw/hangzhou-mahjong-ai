"""Actor-only batches for frozen evaluation; training uses the full learning batch."""

import time

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


def policy_logits(model, planes, scalars, events, oracle=None):
    """Eager inference also handles frozen precast weights, without CUDA graphs.

    ``oracle`` (planes, scalars): the learner's oracle inputs (guided.oracle), passed only when given.
    """
    if not getattr(model, "_frozen_fp16", False):
        return forward(model, planes, scalars, events, oracle)
    with torch.autocast(planes.device.type, torch.float16, cache_enabled=False):
        return forward(model, planes, scalars, events, oracle)


def forward(model, planes, scalars, events, oracle=None):
    """The model's logits; its ``oracle`` argument (guided.oracle) only when there is one."""
    if oracle is None:
        return model(planes, scalars, events)
    return model(planes, scalars, events, oracle=oracle)


def widen(rows, model):
    """Rows extended to the inputs ``model`` reads, when they were encoded for a narrower encoding.

    Collection workers encode for the widest encoding the trial names
    (cfg["encoding"]); this covers callers that leave it at v1.
    """
    wanted = (model.encoder.tile_in.in_features, model.encoder.state_in.in_features)
    if not rows or (len(rows[0]["planes"]) >= wanted[0] and len(rows[0]["scalars"]) >= wanted[1]):
        return rows
    from .encoding import INPUTS, augment

    encoding = next(e for e, inputs in INPUTS.items() if inputs == wanted)
    return [augment(r, encoding) for r in rows]


def policy_batch(rows, device, oracle=False):
    """The rows' Actor inputs on ``device``; with ``oracle``, batch["oracle"] holds the rows' oracle
    inputs (guided.oracle.stack: zero for a row without) when any row has them, else None."""
    if not rows:
        raise ValueError("cannot infer an empty batch")
    # Encodings differ in their numbers of planes and scalars (guided.encoding.INPUTS); a batch has one.
    planes, scalars = len(rows[0]["planes"]), len(rows[0]["scalars"])
    if planes < N_PLANES or scalars < N_SCALARS:
        raise ValueError("incorrect planes shape")
    batch = {}
    for key, shape, dtype in (
        ("planes", (planes, 34), np.float16),
        ("scalars", (scalars,), np.float16),
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
    if oracle:
        from .oracle import stack

        stacked = stack(rows)
        batch["oracle"] = None if stacked is None else tuple(torch.as_tensor(x, device=device) for x in stacked)
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
                model = actor if slot == "model:current" else history[slot]
                batch = policy_batch(widen([item[2]["row"] for item in group], model), device)
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


def hooked_decision(logits, mask, hook):
    """(action, logprob, argmax) per row from a recipe's sample hook, as the collection parent's eager path takes them.

    hook(logits, mask) -> (actions, logprobs or None); a None logprob is the
    chosen action's log-probability under the policy.
    """
    logp = logits.masked_fill(~mask, -1e9).log_softmax(-1)
    actions, selected = hook(logits, mask)
    if selected is None:
        selected = logp.gather(1, actions[:, None]).squeeze(1)
    return torch.stack((actions.float(), selected.float(), logp.argmax(-1).float()), -1)


def multinomial(logits, mask):
    """The default sampling (an unhooked decision's, ActorGraphs._decide), as a sample hook."""
    logp = logits.masked_fill(~mask, -1e9).log_softmax(-1)
    return torch.multinomial(logp.exp(), 1).squeeze(1), None


def branched_decision(logits, mask, hook, branches):
    """hooked_decision (with the default sampling when ``hook`` is None), then the row's branch candidates.

    branches(logits, mask, actions) -> (n, K) candidates (collection's
    hooks["branches"], made K wide); they follow the three answer columns.
    """
    logp = logits.masked_fill(~mask, -1e9).log_softmax(-1)
    actions, selected = (hook or multinomial)(logits, mask)
    if selected is None:
        selected = logp.gather(1, actions[:, None]).squeeze(1)
    decision = torch.stack((actions.float(), selected.float(), logp.argmax(-1).float()), -1)
    return torch.cat((decision, branches(logits, mask, actions).float()), -1)


class ActorGraphs:
    """CUDA graphs of a model's decision, one per padded input shape.

    The collection parent answers many small batches; launching a few dozen
    kernels from Python per batch costs more than the kernels. Inputs are
    written into pinned host buffers of a few padded shapes and copied into
    the graph's static tensors; the replayed computation is the eager one
    (forward, mask, log-softmax, then multinomial sampling or argmax, the
    chosen log-probability and the argmax), so its outputs are identical in
    distribution and, for argmax, in value. Shapes beyond the buckets run eagerly.

    A recipe's sample hook (collection's hooks["sample"]) is arbitrary Python,
    so by default it is not captured: its decision replays a graph of the
    logits alone, and the hook then runs on the batch's rows of the graph's
    output, on the same stream, before any later replay may overwrite it
    (hooked_decision). A hook marked ``capturable = True`` declares itself
    graph-safe: tensor operations only, no host synchronization, row by row
    (it also sees the padding rows of a shape, which never reach a caller),
    and its constants fixed for the life of the hook. It is then captured
    with the forward, one graph per shape, and only the three answer columns
    are read back. Its random draws come from the default CUDA generator,
    whose graph-safe Philox offset advances on every replay, as the captured
    multinomial of an unhooked sample does.

    A branches hook (collection's hooks["branches"], cfg["branches"]) also
    runs outside the graph: a sampled decision with one replays the logits
    alone, then the sample hook (captured or not, or the default
    multinomial) and the branches hook run on the batch's rows of the
    graph's output, as on the eager path (branched_decision), and the
    candidates come back after the three answer columns.

    The learner's oracle inputs (decide's ``oracle``, guided.oracle) are two
    more static inputs, float16 [size, 16, 34] and [size, 4], of graphs of
    their own (the key's ``inputs`` count gains the oracle's 16 planes): the
    forward then passes them to the Actor. Batches without them replay the
    graphs they always did.
    """

    BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
    FINE_BATCH_BUCKETS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    EVENT_BUCKET, MAX_WIDTH = 32, 256

    IN_FLIGHT = 2  # launches of one model and shape that may be unread at a time

    def __init__(self, *, fine_batches=False, history_band=0):
        # Distinct history groups must map to distinct graph widths. Otherwise
        # one round could reuse a two-slot ring before its answers are consumed.
        if (not isinstance(history_band, int) or history_band < 0
                or history_band % self.EVENT_BUCKET):
            raise ValueError("history band must be a nonnegative multiple of 32")
        self.history_band = history_band
        self.batch_buckets = (
            self.FINE_BATCH_BUCKETS if fine_batches else self.BATCH_BUCKETS
        )
        self.graphs, self.hosts, self.turn, self.pool = {}, {}, {}, None
        # Host seconds of decide() by part, cumulative (the collection parent
        # reports the change over a rollout): filling the pinned buffers, the
        # four uploads, the replay, a sample hook's launch, the readback and
        # event, first captures, and batches outside the graph shapes (run eagerly).
        self.seconds = dict.fromkeys(("fill", "upload", "replay", "hook", "readback", "capture", "eager"), 0.0)
        # "hooked" counts graphed decisions of a sample hook, "in_graph" those
        # whose hook was captured with the forward (no separate hook launch).
        self.counts = {"graphed": 0, "hooked": 0, "in_graph": 0, "eager": 0, "captures": 0}

    def shape(self, rows):
        size = next((b for b in self.batch_buckets if b >= len(rows)), None)
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

    def _host(self, key, size, width, inputs):
        # A ring of IN_FLIGHT buffer sets per model and shape: a buffer is not
        # refilled while its earlier copy may still be on the way, and other
        # models' batches or the next round may be launched in between.
        planes, scalars, *oracle = inputs
        ring = self.hosts.get((key, size, width, inputs))
        if ring is None:
            ring = self.hosts[(key, size, width, inputs)] = []
            for _ in range(self.IN_FLIGHT):
                tensors = (
                    torch.zeros((size, planes, 34), dtype=torch.float16).pin_memory(),
                    torch.zeros((size, scalars), dtype=torch.float16).pin_memory(),
                    torch.zeros((size, width, 5), dtype=torch.int64).pin_memory(),
                    torch.zeros((size, N_ACTIONS), dtype=torch.bool).pin_memory(),
                )
                if oracle:  # the learner's oracle inputs (guided.oracle): planes, scalars
                    tensors += (
                        torch.zeros((size, oracle[0], 34), dtype=torch.float16).pin_memory(),
                        torch.zeros((size, 4), dtype=torch.float16).pin_memory(),
                    )
                views = [t.numpy() for t in tensors]
                views[3][:, 0] = (
                    True  # padding rows stay valid; they never reach a caller
                )
                answers = torch.zeros((size, 3), dtype=torch.float32).pin_memory()
                ring.append((tensors, views, answers, torch.cuda.Event()))
        turn = self.turn.get((key, size, width, inputs), 0)
        self.turn[(key, size, width, inputs)] = (turn + 1) % self.IN_FLIGHT
        return ring[turn]

    def _answers(self, key, size, width, inputs, columns):
        """A pinned answer buffer of ``columns`` per row, for a decision with branch candidates.

        A ring of IN_FLIGHT per model and shape like _host's, turned with it
        (every decision of a branching model and shape takes one), so a buffer
        is reused only after its decision's event, recorded on _host's ring
        entry, has been read.
        """
        name = ("answers", key, size, width, inputs, columns)
        ring = self.hosts.get(name)
        if ring is None:
            ring = self.hosts[name] = [torch.zeros((size, columns), dtype=torch.float32).pin_memory()
                                       for _ in range(self.IN_FLIGHT)]
        turn = self.turn.get(name, 0)
        self.turn[name] = (turn + 1) % self.IN_FLIGHT
        return ring[turn]

    @staticmethod
    def _decide(model, planes, scalars, events, mask, sample, oracle=None):
        # fp16 tensor cores: the large pool Actor answers about twice as fast
        # (measured 1.8-2.6x on a 4070), the small one 1.3-2x. Collection only
        # samples from these policies, so the last-bit logit differences from
        # fp32 change nothing that the training measures; evaluation is the
        # research model service, not this class.
        with torch.autocast("cuda", torch.float16, cache_enabled=False):
            logits = forward(model, planes, scalars, events, oracle).float()
        logp = logits.masked_fill(~mask, -1e9).log_softmax(-1)
        greedy = logp.argmax(-1)
        actions = torch.multinomial(logp.exp(), 1).squeeze(1) if sample else greedy
        chosen = logp.gather(1, actions[:, None]).squeeze(1)
        return torch.stack((actions.float(), chosen, greedy.float()), -1)

    @staticmethod
    def _logits(model, planes, scalars, events, mask, oracle=None):
        # The forward of _decide, for a sample hook to choose from (the mask is the hook's).
        with torch.autocast("cuda", torch.float16, cache_enabled=False):
            return forward(model, planes, scalars, events, oracle).float()

    @staticmethod
    def _hooked(model, planes, scalars, events, mask, hook, oracle=None):
        # The forward and a capturable hook's decision, as one graph.
        return hooked_decision(ActorGraphs._logits(model, planes, scalars, events, mask, oracle), mask, hook)

    def _capture(self, compute, model, static):
        """Warm up and capture ``compute(model, *static)`` into a graph sharing the pool; (output, graph)."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                compute(model, *static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = compute(model, *static)
        self.pool = graph.pool()
        return out, graph

    def decide(self, key, model, rows, sample, hook=None, branches=None, oracle=False):
        """Launch (action, log-probability, argmax) per row: the action sampled from the policy, or its argmax.

        With a sample hook (and ``sample``) the hook chooses the action and its
        logprob column from the logits and mask of the batch's rows
        (hooked_decision), as the collection parent's eager path does; a hook
        marked ``capturable`` runs inside the graph (see the class). With a
        branches hook (and ``sample``) each row's candidates follow the three
        columns (branched_decision), the hooks outside the graph.

        Returns a Decided whose result() waits for the device. The answers are
        copied straight out of the graph into a pinned buffer on the same
        stream: graphs share one memory pool, so a later replay of any other
        graph may overwrite this one's device output. Up to IN_FLIGHT calls
        for one model and shape may be unread (their host buffers rotate), so
        the device runs one batch while the host fills the next.

        With ``oracle`` (the collection parent sets it for the learner's slots
        only, guided.oracle) the rows' oracle inputs reach the model, zero for
        a row without them; when no row has them the decision is the plain one.
        """
        from .oracle import PLANES, SCALARS

        start = time.perf_counter()
        hook = hook if sample else None
        branches = branches if sample else None
        rows = widen(rows, model)
        oracle = oracle and any(PLANES in row for row in rows)
        shape = self.shape(rows)
        device = next(model.parameters()).device
        if shape is None:
            batch = policy_batch(rows, device, oracle=oracle)
            inputs = (batch["planes"], batch["scalars"], batch["events"], batch["mask"])
            fed = {"oracle": batch["oracle"]} if oracle else {}
            if branches is not None:
                decided = Decided(branched_decision(self._logits(model, *inputs, **fed), batch["mask"], hook, branches))
            elif hook is None:
                decided = Decided(self._decide(model, *inputs, sample, **fed))
            else:
                decided = Decided(hooked_decision(self._logits(model, *inputs, **fed), batch["mask"], hook))
            self.seconds["eager"] += time.perf_counter() - start
            self.counts["eager"] += 1
            return decided
        size, width = shape
        # The oracle's graphs and buffers are their own: the inputs count gains its planes.
        count = (len(rows[0]["planes"]), len(rows[0]["scalars"])) + ((16,) if oracle else ())
        tensors, views, answers, done = self._host(key, size, width, count)
        planes, scalars, events, mask = views[:4]
        for i, row in enumerate(rows):
            planes[i] = row["planes"]
            scalars[i] = row["scalars"]
            history = np.asarray(row["events"]).reshape(-1, 5)
            events[i, : len(history)] = history
            events[i, len(history) :] = 0
            mask[i] = row["mask"]
        if oracle:
            oracle_planes, oracle_scalars = views[4:]
            for i, row in enumerate(rows):
                if PLANES in row:
                    oracle_planes[i] = row[PLANES]
                    oracle_scalars[i] = row[SCALARS]
                else:
                    oracle_planes[i] = 0
                    oracle_scalars[i] = 0
        mark = time.perf_counter()
        self.seconds["fill"] += mark - start
        # A hooked decision replays the logits alone, unless the hook is
        # capturable: then hook and forward are one graph per hook (the entry
        # keeps the hook alive, so its id cannot name another one later).
        in_graph = hook is not None and branches is None and getattr(hook, "capturable", False) is True
        if in_graph:
            mode = ("hooked", id(hook))
            compute = lambda *inputs, **fed: self._hooked(*inputs, hook, **fed)  # noqa: E731
        elif hook is not None or branches is not None:
            mode, compute = "logits", self._logits
        else:
            mode = sample
            compute = lambda *inputs, **fed: self._decide(*inputs, sample, **fed)  # noqa: E731
        if oracle:
            # The static oracle tensors follow the four inputs; the forward takes them as its oracle.
            plain = compute
            compute = lambda model, p, s, e, m, op, os: plain(model, p, s, e, m, oracle=(op, os))  # noqa: E731
        entry = self.graphs.get((key, size, width, count, mode))
        if entry is None:
            static = [t.to(device) for t in tensors]
            out, graph = self._capture(compute, model, static)
            entry = self.graphs[(key, size, width, count, mode)] = (static, out, graph, hook)
            now = time.perf_counter()
            self.seconds["capture"] += now - mark
            self.counts["captures"] += 1
            mark = now
        static, out, graph, _ = entry
        for target, source in zip(static, tensors):
            target.copy_(source, non_blocking=True)
        now = time.perf_counter()
        self.seconds["upload"] += now - mark
        graph.replay()
        mark = time.perf_counter()
        self.seconds["replay"] += mark - now
        if in_graph:
            self.counts["hooked"] += 1
            self.counts["in_graph"] += 1
        elif branches is not None:
            # As a hook out of the graph, with the candidates' columns after the answer's three.
            n = len(rows)
            out = branched_decision(out[:n], static[3][:n], hook, branches)
            answers = self._answers(key, size, width, count, out.shape[1])[:n]
            now = time.perf_counter()
            self.seconds["hook"] += now - mark
            self.counts["hooked"] += 1
            mark = now
        elif hook is not None:
            # On the replay's stream, so the hook reads these logits before any
            # later replay can overwrite the graphs' shared output memory.
            n = len(rows)
            out = hooked_decision(out[:n], static[3][:n], hook)
            answers = answers[:n]
            now = time.perf_counter()
            self.seconds["hook"] += now - mark
            self.counts["hooked"] += 1
            mark = now
        answers.copy_(out, non_blocking=True)
        done.record()
        self.seconds["readback"] += time.perf_counter() - mark
        self.counts["graphed"] += 1
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
