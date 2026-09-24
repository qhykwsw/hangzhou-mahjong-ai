"""Shared batched inference for model seats in research matches.

One process per host serves every exported Actor under ``--root``; match
workers post public observations (or their Actor encoding) and receive the
chosen legal action. One thread reads requests, groups them by model and keeps
up to two padded batches in flight on the device: a batch is launched without
waiting for it, the next one is assembled from requests that arrive while it
runs, and answers go out once its completion event has fired. Parsing never
waits for the device and the device never waits for parsing, which is what
bounds throughput when many workers each send one small request at a time.
(Several such processes behind one port were measured too: on CUDA the
contexts contend for the GPU and throughput drops, on Apple GPUs it gains
nothing.)

Rows are written straight into pinned host buffers of a few padded shapes and
copied to the device at once; on CUDA the whole decision of each shape is a
captured graph replayed in place, on Apple GPUs it runs eagerly. Both compute
the linear layers in fp16 (autocast; about twice fp32 throughput on either),
so a near-tie argmax can differ between devices and from fp32, which does not
affect paired comparisons because both sides face the same seat.

    python -m guided.model_server --root DIR --port 8830 [--device auto]
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import threading
import time

from . import bootstrap  # noqa: F401

MAX_BODY = 4 * 1024 * 1024
# Inputs are padded to a few shapes: a new shape compiles a graph on Apple GPUs
# (hundreds of ms) or captures one on CUDA, which must happen at load time, not
# inside an action window. Padded rows and history positions are ignored.
EVENT_BUCKET = 32
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
WARM_EVENTS = (32, 64, 128, 192, 256, 320, 384, 448, 512)
IN_FLIGHT = 2  # batches launched ahead of their answers
# CUDA event waits spin a whole core (also with blocking-sync flags, at least
# under WSL) and a waiting thread's wake-up rides on the GIL switch interval,
# so the serving loop polls CUDA itself; Apple GPU waits block properly.
CUDA_POLL = 0.0001


def pick_device(name="auto"):
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Models:
    """Actor artifacts below one root, loaded on first use and kept resident."""

    def __init__(self, root, device):
        self.root, self.device = Path(root).resolve(), device
        self.loaded, self.lock = {}, threading.Lock()

    def get(self, name):
        from .deployment import load_actor

        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or not (path / "meta.json").is_file():
            raise ValueError(f"unknown model artifact: {name}")
        with self.lock:
            if name not in self.loaded:
                actor, meta = load_actor(path, self.device)
                self.loaded[name] = (prepare(actor, self.device), meta)
            return self.loaded[name]


def prepare(actor, device):
    """Hold the linear weights in the dtype the device computes them in.

    Under fp16 autocast every linear casts its fp32 weight and bias again each
    forward (the cast cache needs grad-requiring leaves and is cleared when the
    context exits): ~150 small copies per forward of the large Actor, most of
    the host time of a small batch on Apple GPUs. A weight that is already fp16
    passes through the cast untouched, so the computation and its outputs are
    the same; everything else (embeddings, layer norms, the token parameters)
    stays fp32 exactly as autocast runs it. CUDA replays captured graphs, where
    the casts cost nothing.
    """
    import torch.nn as nn

    if device == "mps":
        for module in actor.modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                module.half()
    return actor


def decide(actor, planes, scalars, events, mask):
    """The whole decision on the device: chosen action per row and one validity flag.

    masked_log_probs without its host-side checks, which would sync (and, on
    CUDA, break a graph capture); the flag is read together with the choices.
    """
    import torch
    import torch.nn.functional as F

    device = planes.device.type
    with torch.autocast(device, torch.float16, enabled=device != "cpu", cache_enabled=False):
        logits = actor(planes, scalars, events).float()
    choice = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1).argmax(-1)
    valid = mask.any(-1).all() & torch.isfinite(logits).all()
    return choice, valid


def bucket(count, length):
    """Padded batch size and event width for ``count`` rows of at most ``length`` events."""
    size = next((b for b in BATCH_BUCKETS if b >= count), count)
    return size, -(-max(1, length) // EVENT_BUCKET) * EVENT_BUCKET


class Slot:
    """Host staging for one padded shape: rows are written in place, answers read in place.

    Pinned where the device copies asynchronously, so filling a slot is plain
    numpy work and neither direction of the transfer blocks the thread. Each
    row's computation is independent of the others in its batch, so padding
    rows keep whatever they last held; they only need one legal action so the
    batch stays valid.
    """

    def __init__(self, size, width, device):
        import torch
        from selfplay_core.actions import N_ACTIONS
        from selfplay_core.encoding import N_PLANES, N_SCALARS

        pin = device.type in ("cuda", "mps")

        def host(shape, dtype):
            tensor = torch.zeros(shape, dtype=dtype)
            return tensor.pin_memory() if pin else tensor

        self.size, self.width = size, width
        self.inputs = (
            host((size, N_PLANES, 34), torch.float16),
            host((size, N_SCALARS), torch.float16),
            host((size, width, 5), torch.int64),
            host((size, N_ACTIONS), torch.bool),
        )
        self.choice = host((size,), torch.int64)
        # 0-d like the flag itself: a broadcasting non-blocking copy silently reads
        # as False on Apple GPUs.
        self.valid = host((), torch.bool)
        self.views = [t.numpy() for t in self.inputs]
        self.views[3][:, 0] = True
        self.rows = 0
        self.marker = None  # the device event recorded after each launch
        self.event = None  # set while a launched batch has not been taken
        self.done = False  # set once the event has fired

    def fill(self, rows):
        planes, scalars, events, mask = self.views
        for i, row in enumerate(rows):
            planes[i] = row["planes"]
            scalars[i] = row["scalars"]
            history = row["events"]
            events[i, : len(history)] = history
            events[i, len(history) :] = 0
            mask[i] = row["mask"]
        self.rows = len(rows)


class Graphs:
    """CUDA graphs of the whole decision, one per model and input shape.

    Small batches spend most of their time launching kernels from Python, not
    computing: replaying a captured graph launches them all at once. The
    captured computation is the eager one, so the chosen actions are identical.
    """

    def __init__(self):
        self.cache, self.pool = {}, None

    def replay(self, name, actor, inputs):
        """Copy the host inputs into the graph's static tensors and replay; nothing waits."""
        import torch

        key = (name,) + tuple(tuple(t.shape) for t in inputs)
        entry = self.cache.get(key)
        if entry is None:
            device = next(actor.parameters()).device
            static = [t.to(device) for t in inputs]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):  # warm-up outside the capture, as CUDA graphs require
                    decide(actor, *static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                outputs = decide(actor, *static)
            self.pool = graph.pool()
            entry = self.cache[key] = (static, outputs, graph)
        static, outputs, graph = entry
        for target, source in zip(static, inputs):
            target.copy_(source, non_blocking=True)
        graph.replay()
        return outputs


class Runner:
    """Launches padded batches on one device and tells when they have completed.

    On CUDA the caller polls ``ready``. Elsewhere a waiter thread blocks on
    each batch's event in launch order and writes a byte to ``wake`` when it
    has fired, so the serving loop can sleep in select on its sockets and this
    pipe alike; the loop itself never waits on the device.
    """

    def __init__(self, device, graphs=True):
        import queue
        import torch

        self.device = torch.device(device)
        self.graphs = Graphs() if graphs and self.device.type == "cuda" else None
        self.polls = self.device.type == "cuda"
        self.slots = {}  # (size, width) -> IN_FLIGHT slots
        self.wake, self._signal = os.pipe()
        os.set_blocking(self.wake, False)
        self.launched = queue.Queue()
        if not self.polls:
            threading.Thread(target=self._wait, name="device-waiter", daemon=True).start()

    def slot(self, count, length):
        """A free slot of the shape for these rows; None while all are in flight."""
        shape = bucket(count, length)
        ring = self.slots.get(shape)
        if ring is None:
            ring = self.slots[shape] = [Slot(*shape, self.device) for _ in range(IN_FLIGHT)]
        return next((s for s in ring if s.event is None), None)

    def launch(self, name, actor, slot):
        import torch

        with torch.inference_mode():
            if self.graphs is not None:
                choice, valid = self.graphs.replay(name, actor, slot.inputs)
            else:
                tensors = [t.to(self.device, non_blocking=True) for t in slot.inputs]
                choice, valid = decide(actor, *tensors)
            slot.choice.copy_(choice, non_blocking=True)
            slot.valid.copy_(valid, non_blocking=True)
            slot.done = False
            slot.event = self._record(slot)
        if not self.polls:
            self.launched.put(slot)

    def _record(self, slot):
        import torch

        if self.device.type == "cpu":
            return True  # CPU work is complete on return
        if slot.marker is None:
            slot.marker = torch.cuda.Event() if self.device.type == "cuda" else torch.mps.Event()
        slot.marker.record()
        return slot.marker

    def _wait(self):
        while True:
            slot = self.launched.get()
            if slot.event is not True:
                slot.event.synchronize()
            slot.done = True
            os.write(self._signal, b"\0")

    def drain(self):
        """Consume wake-up bytes; the waiter may have signalled several batches."""
        try:
            os.read(self.wake, 4096)
        except BlockingIOError:
            pass

    def ready(self, slot):
        if slot.done or slot.event is True:
            return True
        if self.polls and slot.event.query():
            slot.done = True
        return slot.done

    @staticmethod
    def take(slot):
        """Choices and the validity flag of a completed batch; frees the slot."""
        choice = slot.choice.tolist()[: slot.rows]
        valid = bool(slot.valid)
        slot.event, slot.done = None, False
        return choice, valid


class Service:
    """Groups requests per model and keeps the device fed; nothing here blocks on it."""

    def __init__(self, models, *, max_batch=64, graphs=True):
        self.models, self.max_batch = models, max_batch
        self.runner = Runner(models.device, graphs)
        self.pending = {}  # (model, pre-encoded) -> deque of (sequence, context, request)
        self.sequence = 0
        self.inflight = deque()  # (items, parsed, slot, launched at)
        self.failed = []  # (items, answers) rejected before reaching the device
        self.warmed = set()
        self.stats = dict(requests=0, batches=0, seconds=0.0, prepare_seconds=0.0,
                          forward_seconds=0.0)

    def enqueue(self, context, request):
        """Queue one /act request; ``context`` comes back with its answer."""
        group = (request["model"], "encoded" in request)
        self.pending.setdefault(group, deque()).append((self.sequence, context, request))
        self.sequence += 1

    def busy(self):
        return bool(self.inflight or self.pending or self.failed)

    def launch(self):
        """Start batches, oldest request first, while the device has room."""
        while self.pending and len(self.inflight) < IN_FLIGHT:
            group = min(self.pending, key=lambda g: self.pending[g][0][0])
            queue = self.pending[group]
            items = [queue.popleft() for _ in range(min(len(queue), self.max_batch))]
            if not queue:
                del self.pending[group]
            self._launch(group, [(context, request) for _, context, request in items])

    def _launch(self, group, items):
        model, pre_encoded = group
        started = time.perf_counter()
        try:
            actor, meta = self.models.get(model)
            rows, parsed = self._rows([request for _, request in items], meta, pre_encoded)
            slot = self.runner.slot(len(rows), max(len(r["events"]) for r in rows))
            if slot is None:
                raise RuntimeError("no free staging slot")
            slot.fill(rows)
            self.stats["prepare_seconds"] += time.perf_counter() - started
            self.runner.launch(model, actor, slot)
        except Exception as exc:  # a bad request fails its batch, not the server
            self.failed.append((items, [{"error": str(exc)[:2000]}] * len(items)))
            return
        finally:
            self.stats["seconds"] += time.perf_counter() - started
        self.inflight.append((items, parsed, slot, started))

    def _rows(self, requests, meta, pre_encoded):
        if pre_encoded:
            if any(p.get("lookahead") != meta["config"]["lookahead"] for p in requests):
                raise ValueError("encoding does not match the model's lookahead")
            return [decode(p["encoded"]) for p in requests], None
        from .deployment import parse_observation
        from .encoding import encode

        parsed = [parse_observation(p["observation"]) for p in requests]
        return [encode(obs, lookahead=meta["config"]["lookahead"]) for obs in parsed], parsed

    def finished(self):
        """(items, answers) of every batch whose answers are ready, oldest first."""
        out, self.failed = self.failed, []
        while self.inflight and self.runner.ready(self.inflight[0][2]):
            items, parsed, slot, launched = self.inflight.popleft()
            started = time.perf_counter()
            choice, valid = self.runner.take(slot)
            self.stats["forward_seconds"] += started - launched
            if not valid:
                answers = [{"error": "each policy row must have at least one legal action "
                            "and finite logits"}] * len(items)
            elif parsed is None:
                answers = [{"index": int(index)} for index in choice]
            else:
                from selfplay_core.actions import action_index

                answers = []
                for obs, index in zip(parsed, choice):
                    legal = [action_index(a) for a in obs.legal_actions]
                    answers.append({"legal_index": legal.index(index)} if index in legal
                                   else {"error": "Actor chose an illegal action"})
            self.stats["requests"] += len(items)
            self.stats["batches"] += 1
            self.stats["seconds"] += time.perf_counter() - started
            out.append((items, answers))
        return out

    def wait(self):
        """Block until an in-flight batch has completed (nothing to do meanwhile)."""
        if self.runner.polls:
            time.sleep(CUDA_POLL)
            return
        import select

        select.select([self.runner.wake], [], [])
        self.runner.drain()

    def decide(self, model, payloads):
        """Chosen actions for one model's requests, in order (waits for the device)."""
        answers = {}
        for i, payload in enumerate(payloads):
            self.enqueue(i, dict(payload, model=model))
        while self.busy():
            self.launch()
            for items, replies in self.finished():
                answers.update({context: reply for (context, _), reply in zip(items, replies)})
            if self.inflight:
                self.wait()
        return [answers[i] for i in range(len(payloads))]

    def load(self, model):
        """Load a model and run every padded shape once, so play never pays for it."""
        actor, _ = self.models.get(model)
        if model in self.warmed:
            return {"ok": True}
        for size in BATCH_BUCKETS[:5]:
            for length in WARM_EVENTS:
                row = blank()
                row["events"] = row["events"].repeat(length, axis=0)
                # A slot of its own: the ring's may be in flight for another worker.
                slot = Slot(*bucket(size, length), self.runner.device)
                slot.fill([row] * size)
                self.runner.launch(model, actor, slot)
                while not self.runner.ready(slot):
                    self.wait()
                self.runner.take(slot)
        self.warmed.add(model)
        return {"ok": True}


def blank():
    """A minimal valid Actor input, for loading a model and its kernels ahead of play."""
    import numpy as np
    from selfplay_core.actions import N_ACTIONS
    from selfplay_core.encoding import N_PLANES, N_SCALARS

    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True
    return {"planes": np.zeros((N_PLANES, 34), np.float16), "scalars": np.zeros(N_SCALARS, np.float16),
            "events": np.zeros((1, 5), np.int16), "mask": mask}


def decode(data):
    """Actor input packed by a match worker (model_seat.pack)."""
    import base64
    import numpy as np
    from selfplay_core.actions import N_ACTIONS
    from selfplay_core.encoding import N_PLANES, N_SCALARS

    raw = {k: base64.b64decode(v) for k, v in data.items()}
    return {
        "planes": np.frombuffer(raw["planes"], dtype=np.float16).reshape(N_PLANES, 34),
        "scalars": np.frombuffer(raw["scalars"], dtype=np.float16).reshape(N_SCALARS),
        "events": np.frombuffer(raw["events"], dtype=np.int16).reshape(-1, 5),
        "mask": np.frombuffer(raw["mask"], dtype=bool).reshape(N_ACTIONS),
    }


class Connection:
    def __init__(self, sock):
        self.sock, self.inbox, self.outbox = sock, bytearray(), bytearray()


def reply(connection, code, body):
    data = json.dumps(body).encode()
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 413: "Payload Too Large"}[code]
    head = (f"HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n\r\n").encode()
    connection.outbox += head + data  # one write: no Nagle stall between header and body


def requests(connection):
    """Complete HTTP/1.1 requests in the connection's buffer: (method, path, body)."""
    while True:
        end = connection.inbox.find(b"\r\n\r\n")
        if end < 0:
            return
        head = bytes(connection.inbox[:end]).decode("latin-1").split("\r\n")
        method, path, _ = head[0].split(" ", 2)
        length = 0
        for line in head[1:]:
            name, _, value = line.partition(":")
            if name.strip().lower() == "content-length":
                length = int(value)
        if length > MAX_BODY:
            raise ValueError("invalid body size")
        if len(connection.inbox) < end + 4 + length:
            return
        body = bytes(connection.inbox[end + 4 : end + 4 + length])
        del connection.inbox[: end + 4 + length]
        yield method, path, body


def serve(service, port, host="127.0.0.1", code=""):
    """One thread: read ready requests, launch and collect batches, answer, repeat.

    While a batch is on the device the loop keeps reading, so the next batch
    forms from everything that arrives meanwhile; the runner's pipe (or, on
    CUDA, a poll between iterations) tells it when a batch has finished. The
    standard library's threaded server spent more CPU per request (a thread
    and a GIL handover each, slow header parsing) than the inference itself.
    """
    import selectors
    import socket

    selector = selectors.DefaultSelector()
    listener = socket.create_server((host, port), backlog=512)
    listener.setblocking(False)
    selector.register(listener, selectors.EVENT_READ)
    selector.register(service.runner.wake, selectors.EVENT_READ)
    print(json.dumps({"port": port, "device": service.models.device}), flush=True)

    def close(connection):
        selector.unregister(connection.sock)
        connection.sock.close()

    def adopt(sock):
        sock.setblocking(False)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        selector.register(sock, selectors.EVENT_READ, Connection(sock))

    def flush(connection):
        try:
            sent = connection.sock.send(connection.outbox)
        except BlockingIOError:
            sent = 0
        except OSError:
            close(connection)
            return
        del connection.outbox[:sent]
        selector.modify(connection.sock, selectors.EVENT_READ | (selectors.EVENT_WRITE if connection.outbox else 0),
                        connection)

    polling = service.runner.polls
    reported = time.monotonic()
    while True:
        ready = selector.select(0 if polling and service.inflight else None)
        if time.monotonic() - reported > 60:
            # Device load of the run: forward seconds against wall seconds.
            reported = time.monotonic()
            print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in service.stats.items()}),
                  flush=True)
        for key, events in ready:
            if key.fileobj == service.runner.wake:
                service.runner.drain()
                continue
            if key.fileobj is listener:
                try:
                    sock, _ = listener.accept()
                except BlockingIOError:
                    continue
                adopt(sock)
                continue
            connection = key.data
            if events & selectors.EVENT_WRITE:
                flush(connection)
            if not events & selectors.EVENT_READ:
                continue
            try:
                data = connection.sock.recv(1 << 20)
            except BlockingIOError:
                continue
            except ConnectionError:
                data = b""
            if not data:
                close(connection)
                continue
            connection.inbox += data
            try:
                for method, path, body in requests(connection):
                    if method == "GET" and path == "/health":
                        reply(connection, 200, {"ok": True, "device": service.models.device, "code": code, **service.stats})
                    elif method == "POST" and path == "/load":
                        try:
                            reply(connection, 200, service.load(json.loads(body)["model"]))
                        except Exception as exc:
                            reply(connection, 400, {"error": str(exc)[:2000]})
                    elif method == "POST" and path == "/act":
                        try:
                            service.enqueue(connection, json.loads(body))
                        except Exception as exc:
                            reply(connection, 400, {"error": str(exc)[:2000]})
                    else:
                        reply(connection, 404, {"error": "not found"})
            except ValueError as exc:
                reply(connection, 413, {"error": str(exc)})
                flush(connection)
                close(connection)
                continue
            if connection.outbox:
                flush(connection)
        done = service.finished()
        service.launch()
        for items, answers in done:
            for (connection, _), answer in zip(items, answers):
                if connection.sock.fileno() >= 0:  # the worker may have gone meanwhile
                    reply(connection, 400 if "error" in answer else 200, answer)
                    flush(connection)
        if polling and service.inflight and not ready and not done:
            time.sleep(CUDA_POLL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--port", type=int, default=8830)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--no-graphs", action="store_true", help="eager CUDA inference (for comparison)")
    args = p.parse_args()
    service = Service(Models(args.root, pick_device(args.device)), max_batch=args.max_batch,
                      graphs=not args.no_graphs)
    serve(service, args.port, code=str(Path(__file__).resolve().parents[4]))


if __name__ == "__main__":
    main()
