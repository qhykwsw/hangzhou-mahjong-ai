"""Shared batched inference for model seats in research matches.

One process per host serves every exported Actor under ``--root``; match
workers post public observations and receive the chosen legal action. One
inference thread groups pending requests by model, so concurrent workers are
batched on one device instead of competing for it. On Apple GPUs inference
runs under fp16 autocast (about twice fp32 throughput); near-tie argmax can
differ from CUDA fp32, which does not affect paired comparisons because both
sides face the same seat.

    python -m guided.model_server --root DIR --port 8830 [--device auto]
"""

import argparse
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time

from . import bootstrap  # noqa: F401

MAX_BODY = 4 * 1024 * 1024


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
                self.loaded[name] = load_actor(path, self.device)
            return self.loaded[name]


class Service:
    def __init__(self, models, *, max_batch=64, batch_wait=0.002, timeout=30.0):
        self.models, self.max_batch = models, max_batch
        self.batch_wait, self.timeout = batch_wait, timeout
        self.queue, self.closed = queue.Queue(), threading.Event()
        self.stats = dict(requests=0, batches=0, seconds=0.0)
        self.thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self.thread.start()

    def submit(self, model, payload):
        future = Future()
        self.queue.put((model, payload, future, time.perf_counter() + self.timeout))
        return future

    def _loop(self):
        import torch
        from selfplay_core.actions import action_index
        from .deployment import parse_observation
        from .encoding import encode
        from .learning import masked_log_probs

        autocast = self.models.device == "mps"
        while not self.closed.is_set():
            try:
                pending = [self.queue.get(timeout=0.05)]
            except queue.Empty:
                continue
            until = time.perf_counter() + self.batch_wait
            while len(pending) < self.max_batch:
                try:
                    pending.append(self.queue.get(timeout=max(0.0, until - time.perf_counter())))
                except queue.Empty:
                    break
            groups = {}
            for model, payload, future, deadline in pending:
                groups.setdefault(model, []).append((payload, future, deadline))
            for model, rows in groups.items():
                started = time.perf_counter()
                try:
                    actor, meta = self.models.get(model)
                    parsed = [parse_observation(payload) for payload, _, _ in rows]
                    encoded = [encode(obs, lookahead=meta["config"]["lookahead"]) for obs in parsed]
                    with torch.inference_mode(), torch.autocast("mps", torch.float16, enabled=autocast):
                        planes, scalars, events, mask = inputs(encoded, actor_device(actor))
                        logits = actor(planes, scalars, events)
                        choice = masked_log_probs(logits.float(), mask).argmax(-1).tolist()
                    for (payload, future, _), obs, index in zip(rows, parsed, choice):
                        legal = [action_index(a) for a in obs.legal_actions]
                        if index not in legal:
                            future.set_exception(RuntimeError("Actor chose an illegal action"))
                        else:
                            future.set_result({"legal_index": legal.index(index)})
                except Exception as exc:  # A bad request fails its batch, not the server.
                    for _, future, _ in rows:
                        if not future.done():
                            future.set_exception(exc)
                self.stats["requests"] += len(rows)
                self.stats["batches"] += 1
                self.stats["seconds"] += time.perf_counter() - started

    def close(self):
        self.closed.set()
        self.thread.join(timeout=5)


def actor_device(actor):
    return next(actor.parameters()).device


def inputs(rows, device):
    """Only the Actor inputs, assembled on the host and copied once each.

    Per-tensor host-to-device copies dominate small batches on Apple GPUs,
    so this skips the training fields that learning.make_batch also builds.
    """
    import numpy as np
    import torch

    planes = np.stack([np.asarray(r["planes"], dtype=np.float16) for r in rows])
    scalars = np.stack([np.asarray(r["scalars"], dtype=np.float16) for r in rows])
    mask = np.stack([np.asarray(r["mask"], dtype=bool) for r in rows])
    history = [np.asarray(r["events"], dtype=np.int64).reshape(-1, 5) for r in rows]
    events = np.zeros((len(rows), max(1, max(len(h) for h in history)), 5), dtype=np.int64)
    for i, h in enumerate(history):
        events[i, : len(h)] = h
    return tuple(torch.from_numpy(a).to(device) for a in (planes, scalars, events, mask))


def serve(service, port, host="127.0.0.1"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"ok": True, "device": service.models.device, **service.stats})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/act":
                return self._reply(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self._reply(413, {"error": "invalid body size"})
            try:
                request = json.loads(self.rfile.read(length))
                future = service.submit(request["model"], request["observation"])
                self._reply(200, future.result(timeout=service.timeout))
            except Exception as exc:
                self._reply(400, {"error": str(exc)[:2000]})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--port", type=int, default=8830)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-batch", type=int, default=64)
    args = p.parse_args()
    service = Service(Models(args.root, pick_device(args.device)), max_batch=args.max_batch)
    server = serve(service, args.port)
    print(json.dumps({"port": args.port, "device": service.models.device}), flush=True)
    try:
        server.serve_forever()
    finally:
        service.close()


if __name__ == "__main__":
    main()
