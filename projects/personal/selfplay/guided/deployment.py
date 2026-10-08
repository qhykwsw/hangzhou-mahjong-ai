"""Actor-only artifacts and an independent batched HTTP inference service.

Run with ``python -m guided.deployment export|benchmark|serve`` from a path that
contains the selfplay project. This is a GPU-service artifact, not a replacement
for the existing platform CPU/ONNX release contract. No ONNX export is implied.
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import random
import tempfile
import threading
import time
import urllib.error
import urllib.request

from . import bootstrap  # noqa: F401

FORMAT = "guided-actor-v1"
MAX_BODY = 2 * 1024 * 1024


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _actor_config(config):
    from .networks import actor_options

    result = {key: int(config[key]) for key in ("dim", "layers", "heads")}
    if min(result.values()) < 1 or result["dim"] % result["heads"]:
        raise ValueError("invalid actor architecture")
    result["lookahead"] = bool(config.get("lookahead", True))
    # Optional Actor structures (networks.actor_options); absent by default.
    return result | actor_options(config)


def load_actor(source, device="cuda", *, oracle=False):
    """Local checkpoints are trusted pickle inputs; exported weights use safe load.

    Service deployment accepts only exported directories. Loading a training
    checkpoint here is for local export/benchmark, never an HTTP-supplied path.

    An artifact's meta.json may turn on the stage re-rank (``"outcome_rerank"``,
    guided.outcome): it is checked against the Actor (it needs the outcome
    head) and returned normalized in the metadata (None when off), for the
    services that act on it.

    An Actor with the training-only oracle inputs (ADR-0018, guided.oracle:
    ``oracle_inputs`` in its config, or any oracle parameter) is refused unless
    ``oracle``: only a trial's training start (guided.trial.Context) and
    expand ask for one. Everything that evaluates, serves, deploys or plays
    against an Actor loads it here without, so none of them can ever feed or
    read the hidden state, even if an export had failed to strip it.
    """
    import torch
    from .encoding import ENCODING, INPUTS
    from .networks import ORACLE_PREFIX, actor_of, actor_options, has_oracle

    source = Path(source)
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU inference")
    if source.is_dir():
        metadata = json.loads((source / "meta.json").read_text())
        if metadata.get("format") != FORMAT or metadata.get("encoding") not in INPUTS:
            raise ValueError("incompatible Actor artifact or observation encoding")
        if _sha256(source / "actor.pt") != metadata.get("weights_sha256"):
            raise ValueError("Actor weight checksum mismatch")
        if has_oracle(metadata["config"]) and not oracle:
            raise ValueError(f"refusing an Actor with oracle inputs (ADR-0018: training only): {source}")
        config = _actor_config(metadata["config"])
        weights = torch.load(source / "actor.pt", map_location="cpu", weights_only=True)
    else:
        state = torch.load(source, map_location="cpu", weights_only=False)
        if state.get("format") != "guided-v1":
            raise ValueError("expected a guided-v1 training checkpoint")
        if has_oracle(state["config"]) and not oracle:
            raise ValueError(f"refusing an Actor with oracle inputs (ADR-0018: training only): {source}")
        config = _actor_config(state["config"])
        weights = state["actor"]
        encoding = state["config"].get("encoding", ENCODING)
        if encoding not in INPUTS:
            raise ValueError("incompatible observation encoding")
        metadata = {"format": FORMAT, "encoding": encoding, "config": config}
    if not oracle and any(name.startswith(ORACLE_PREFIX) for name in weights):
        raise ValueError(f"refusing Actor weights with oracle parameters (ADR-0018: training only): {source}")
    if "outcome_rerank" in metadata:
        # The stage re-rank switch (guided.outcome), normalized; it needs the outcome head.
        from .outcome import rerank_options

        rerank = rerank_options(metadata["outcome_rerank"])
        if rerank is not None and "outcome_head" not in config:
            raise ValueError(f"outcome_rerank needs an Actor with the outcome head: {source}")
        metadata = dict(metadata, outcome_rerank=rerank)
    actor = actor_of(metadata["encoding"], config["dim"], config["layers"], config["heads"], **actor_options(config))
    actor.load_state_dict(weights, strict=True)
    for value in actor.state_dict().values():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite Actor weights")
    return actor.to(selected).eval(), metadata


def export_actor(checkpoint, output):
    """Atomically export ONLY Actor tensors, architecture and public encoding ID."""
    import torch

    actor, metadata = load_actor(checkpoint, "cpu")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".actor-export-", dir=output.parent) as tmp:
        directory = Path(tmp) / "artifact"
        directory.mkdir()
        weights = {
            name: value.detach().cpu().clone()
            for name, value in actor.state_dict().items()
        }
        torch.save(weights, directory / "actor.pt")
        meta = {
            "format": FORMAT,
            "encoding": metadata["encoding"],
            "config": metadata["config"],
            "weights_sha256": _sha256(directory / "actor.pt"),
            "source_checkpoint_sha256": _sha256(checkpoint)
            if Path(checkpoint).is_file()
            else None,
            "backend": "pytorch",
            "actions": 231,
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n"
        )
        # Round-trip strict loading verifies exactly the Actor architecture before publication.
        restored, _ = load_actor(directory, "cpu")
        for name, value in actor.state_dict().items():
            if not torch.equal(value, restored.state_dict()[name]):
                raise RuntimeError("Actor export round-trip mismatch")
        directory.rename(output)
    return output


def parse_observation(payload):
    """Decode a complete public observation; reject malformed/private extensions.

    This validates the public protocol and action representation. Hidden state is
    intentionally unavailable, so the server cannot re-adjudicate a client's
    claimed legal actions against the real wall or opponents' hands.
    """
    from mahjong_ai.execution.candidate_worker import observation
    from selfplay_core.actions import action_index, describe_index

    if not isinstance(payload, dict) or "public_events" not in payload:
        raise ValueError(
            "a complete observation_payload with public_events is required"
        )
    try:
        obs = observation(payload)
        if obs.phase not in {"draw", "response_peng", "response_chi"}:
            raise ValueError("invalid decision phase")
        if type(obs.stage.seat) is not int or not 0 <= obs.stage.seat < 4:
            raise ValueError("invalid seat")
        if type(obs.stage.dealer) is not int or not 0 <= obs.stage.dealer < 4:
            raise ValueError("invalid dealer")
        if any(
            len(value) != 4
            for value in (
                obs.stage.score_by_seat,
                obs.melds,
                obs.discards,
                obs.hand_sizes,
            )
        ):
            raise ValueError("four-seat fields must have exactly four elements")
        if obs.stage.stage_scores is not None and len(obs.stage.stage_scores) != 4:
            raise ValueError("stage_scores must have four elements")
        if not 0 <= len(obs.hand) <= 14 or not 0 <= obs.wall_remaining <= 136:
            raise ValueError("invalid hand or wall size")
        if any(type(size) is not int or not 0 <= size <= 14 for size in obs.hand_sizes):
            raise ValueError("invalid hand sizes")
        if obs.stage.round_no < 1 or obs.stage.rounds_total < 1 or obs.base_score < 1:
            raise ValueError("invalid round or base score")
        if len(obs.public_events) > 16384:
            raise ValueError("public history exceeds request limit")

        def tile(value):
            if type(value) is not int or not 0 <= value < 34:
                raise ValueError("tile must be an integer in [0, 33]")

        for value in obs.hand:
            tile(value)
        if obs.drawn_tile is not None:
            tile(obs.drawn_tile)
        for river in obs.discards:
            if len(river) > 136:
                raise ValueError("invalid river size")
            for value in river:
                tile(value)
        for melds in obs.melds:
            if len(melds) > 4:
                raise ValueError("invalid meld count")
            for meld in melds:
                for value in meld.tiles:
                    tile(value)
        for event in obs.public_events:
            if event.seat is not None and (
                type(event.seat) is not int or not 0 <= event.seat < 4
            ):
                raise ValueError("invalid event seat")
            if event.tile is not None:
                tile(event.tile)
            for value in event.tiles:
                tile(value)
        if not obs.legal_actions or len(obs.legal_actions) > 231:
            raise ValueError("nonempty legal actions are required")
        indices = []
        for action in obs.legal_actions:
            if action.tile is not None:
                tile(action.tile)
            for value in action.tiles:
                tile(value)
            if (
                action.action in {"discard", "peng", "gang", "chi"}
                and action.tile is None
            ):
                raise ValueError("tile action requires a tile")
            if action.action == "discard" and action.tile not in obs.hand:
                raise ValueError("discard tile is absent from the hand")
            if action.action == "chi":
                sequence = sorted((*action.tiles, action.tile))
                if (
                    len(sequence) != 3
                    or sequence[-1] >= 27
                    or sequence[0] // 9 != sequence[-1] // 9
                    or sequence != list(range(sequence[0], sequence[0] + 3))
                ):
                    raise ValueError("invalid chi sequence")
            index = action_index(action)
            if not 0 <= index < 231:
                raise ValueError("action index out of range")
            if action != describe_index(index):
                raise ValueError(
                    "action does not match the canonical action representation"
                )
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate legal action indices")
        # Scores/numeric metadata must also survive strict JSON serialization.
        json.dumps(payload, allow_nan=False)
        return obs
    except (TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"invalid observation_payload: {exc}") from exc


class OverloadedError(RuntimeError):
    pass


class BatchService:
    """One inference thread, bounded backlog, at most ten requests per batch."""

    def __init__(
        self,
        actor,
        metadata,
        *,
        max_pending=100,
        max_batch=10,
        batch_wait=0.002,
        timeout=5.0,
    ):
        if (
            max_pending < 1
            or not 1 <= max_batch <= 10
            or batch_wait < 0
            or timeout <= 0
        ):
            raise ValueError("invalid service queue configuration")
        self.actor, self.metadata = actor, metadata
        self.queue = queue.Queue(max_pending)
        self.max_batch, self.batch_wait, self.timeout = max_batch, batch_wait, timeout
        self.closed = threading.Event()
        self.thread = threading.Thread(
            target=self._loop, name="actor-inference", daemon=True
        )
        self.thread.start()

    def submit(self, payload):
        if self.closed.is_set():
            raise RuntimeError("service is closed")
        future = Future()
        try:
            self.queue.put_nowait((payload, future, time.perf_counter() + self.timeout))
        except queue.Full as exc:
            raise OverloadedError("inference queue is full") from exc
        return future

    @staticmethod
    def _complete(future, *, result=None, error=None):
        if not future.done():
            try:
                future.set_exception(error) if error is not None else future.set_result(
                    result
                )
            except Exception:
                # HTTP timeouts may cancel the future concurrently.
                if not future.cancelled():
                    raise

    def _loop(self):
        import torch
        from .encoding import encoder
        from .learning import make_batch, masked_log_probs
        from .outcome import choice_options, choose, outcome_bins, stage_of_row, utilities
        from selfplay_core.actions import action_index

        device = next(self.actor.parameters()).device
        encode = encoder(self.metadata["encoding"])
        rerank = self.metadata.get("outcome_rerank")
        bins = outcome_bins(self.metadata["config"]["outcome_head"]) if rerank else None
        while not self.closed.is_set():
            try:
                first = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            requests = [first]
            until = time.perf_counter() + self.batch_wait
            while len(requests) < self.max_batch:
                try:
                    requests.append(
                        self.queue.get(timeout=max(0.0, until - time.perf_counter()))
                    )
                except queue.Empty:
                    break
            valid, rows = [], []
            for payload, future, deadline in requests:
                if future.cancelled():
                    continue
                if time.perf_counter() >= deadline:
                    self._complete(
                        future, error=TimeoutError("request expired in queue")
                    )
                    continue
                try:
                    obs = parse_observation(payload)
                    rows.append(
                        encode(obs, lookahead=self.metadata["config"]["lookahead"])
                    )
                    valid.append((obs, future, deadline))
                except Exception as exc:
                    self._complete(future, error=ValueError(str(exc)))
            if not valid:
                continue
            try:
                with torch.inference_mode():
                    batch = make_batch(rows, device)
                    inputs = (batch["planes"], batch["scalars"], batch["events"])
                    # The stage re-rank (meta.json outcome_rerank) acts only on semifinal and final rows.
                    stages = [stage_of_row(row) for row in rows] if rerank else []
                    if any(s is not None for s in stages):
                        logits, outcome = self.actor(*inputs, return_outcome=True)
                        log_probs = masked_log_probs(logits, batch["mask"])
                        utility = torch.as_tensor(utilities(stages, bins, rerank["per_hand_sd"]), device=device)
                        chosen = choose(log_probs, outcome, batch["mask"], utility, log_probs.argmax(-1),
                                        **choice_options(rerank))
                    else:
                        chosen = masked_log_probs(self.actor(*inputs), batch["mask"]).argmax(-1)
                    indices = chosen.cpu().tolist()
                for (obs, future, deadline), index in zip(valid, indices):
                    if time.perf_counter() >= deadline:
                        self._complete(
                            future,
                            error=TimeoutError("request expired during inference"),
                        )
                        continue
                    action = next(
                        (a for a in obs.legal_actions if action_index(a) == index), None
                    )
                    if action is None:
                        raise RuntimeError("Actor returned an illegal action")
                    self._complete(
                        future, result={"index": int(index), "action": asdict(action)}
                    )
            except Exception as exc:
                for _, future, _ in valid:
                    self._complete(
                        future, error=RuntimeError(f"inference failed: {exc}")
                    )
        while True:
            try:
                _, future, _ = self.queue.get_nowait()
            except queue.Empty:
                break
            self._complete(future, error=RuntimeError("service stopped"))

    def close(self):
        self.closed.set()
        self.thread.join(timeout=self.timeout + 1)
        if self.thread.is_alive():
            raise RuntimeError("inference thread did not stop within shutdown timeout")


def create_server(service, host="127.0.0.1", port=8810, max_connections=128):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(service.timeout + 1)

        def log_message(self, *_):
            pass

        def respond(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "not found"})
            healthy = service.thread.is_alive() and not service.closed.is_set()
            self.respond(
                200 if healthy else 503,
                {
                    "ok": healthy,
                    "format": FORMAT,
                    "queued": service.queue.qsize(),
                    "max_batch": service.max_batch,
                },
            )

        def do_POST(self):
            if self.path != "/act":
                return self.respond(404, {"error": "not found"})
            future = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.respond(413, {"error": "invalid body size"})
                message = json.loads(self.rfile.read(length))
                if not isinstance(message, dict) or set(message) != {
                    "observation_payload"
                }:
                    raise ValueError("expected only observation_payload")
                future = service.submit(message["observation_payload"])
                self.respond(200, future.result(timeout=service.timeout))
            except OverloadedError as exc:
                self.respond(429, {"error": str(exc)})
            except TimeoutError:
                if future is not None:
                    future.cancel()
                self.respond(504, {"error": "inference request timed out"})
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(503, {"error": str(exc)})

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128

        def __init__(self):
            self.slots = threading.BoundedSemaphore(max_connections)
            super().__init__((host, port), Handler)

        def process_request(self, request, client_address):
            if not self.slots.acquire(blocking=False):
                try:
                    request.sendall(
                        b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
                    )
                finally:
                    self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.slots.release()

    return Server()


def sample_observations(count=100, seed=20260920):
    """Real legal decision states, including claims and nonempty public histories."""
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig
    from mahjong_ai.execution.candidate_runtime import observation_payload

    if count < 1:
        raise ValueError("observation count must be positive")
    rng, result, game = random.Random(seed), [], 0
    while len(result) < count:
        state = HandState(RoundConfig(rounds=16, allow_tiebreak=False), seed + game)
        while not state.settled and len(result) < count:
            legal = state.legal_actions()
            if len(legal) > 1:
                payload, _ = observation_payload(state.observation())
                result.append(payload)
            state.step(rng.choice(legal))
        game += 1
    return result


def _http_act(url, payload, timeout=10):
    body = json.dumps({"observation_payload": payload}, allow_nan=False).encode()
    request = urllib.request.Request(
        url + "/act", data=body, headers={"Content-Type": "application/json"}
    )
    # Explicitly bypass machine proxy settings for the local benchmark server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def benchmark(source, observations=None, count=100, *, device="cuda", concurrency=10):
    """Measure real loopback HTTP receive/queue/encode/infer/decode/return latency.

    Both sustained concurrency and synchronized bursts are measured. This replays
    real engine observations; it is not a measurement of the platform transport
    or its production arrival distribution. Generation and model loading are
    outside timing. Every timed response is checked against its legal actions.
    """
    import numpy as np
    import torch
    from selfplay_core.actions import action_index

    if count < 1 or not 1 <= concurrency <= 10:
        raise ValueError("positive count and concurrency in [1, 10] required")
    payloads = (
        list(observations) if observations is not None else sample_observations(count)
    )
    if not payloads:
        raise ValueError("no benchmark observations")
    parsed = [parse_observation(payload) for payload in payloads]
    actor, metadata = load_actor(source, device)
    service = BatchService(actor, metadata)
    server = create_server(service, port=0)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    url = f"http://127.0.0.1:{server.server_port}"
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    report = {
        "device": str(device),
        "format": FORMAT,
        "config": metadata["config"],
        "workload": "real-engine-observation-replay-over-loopback-http",
        "concurrency": concurrency,
        "observations": len(payloads),
        "latency_boundary": "client JSON serialization through complete HTTP response decoding",
        "loads": {},
    }

    def request(index, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=10)
        started = time.perf_counter()
        i = index % len(payloads)
        try:
            result = _http_act(url, payloads[i])
            elapsed = (time.perf_counter() - started) * 1000
            legal = {
                action_index(action): asdict(action)
                for action in parsed[i].legal_actions
            }
            expected = legal.get(result.get("index"))
            # JSON changes action.tiles from tuple to list.
            illegal = expected is None or json.loads(
                json.dumps(expected)
            ) != result.get("action")
            return elapsed, int(illegal), None
        except Exception as exc:
            return (time.perf_counter() - started) * 1000, 0, str(exc)

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            warmup = list(executor.map(request, range(concurrency)))
            if any(illegal or error for _, illegal, error in warmup):
                raise RuntimeError(f"benchmark warmup failed: {warmup}")
            for mode in ("sustained", "synchronized_burst"):
                started = time.perf_counter()
                if mode == "sustained":
                    results = list(executor.map(request, range(count)))
                else:
                    results = []
                    for start in range(0, count, concurrency):
                        indices = range(start, min(start + concurrency, count))
                        barrier = threading.Barrier(len(indices))
                        futures = [
                            executor.submit(request, i, barrier) for i in indices
                        ]
                        results.extend(future.result() for future in futures)
                duration = time.perf_counter() - started
                latencies = [row[0] for row in results]
                errors = [row[2] for row in results if row[2] is not None]
                report["loads"][mode] = {
                    "requests": count,
                    "seconds": duration,
                    "requests_per_second": count / duration,
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                    "illegal_actions": sum(row[1] for row in results),
                    "errors": len(errors),
                    "error_examples": errors[:5],
                }
        if torch.device(device).type == "cuda":
            gpu = torch.device(device)
            report["gpu"] = {
                "name": torch.cuda.get_device_name(gpu),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(gpu),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(gpu),
                "total_bytes": torch.cuda.get_device_properties(gpu).total_memory,
            }
        else:
            report["gpu"] = None
        report["passed_legality"] = all(
            not load["illegal_actions"] and not load["errors"]
            for load in report["loads"].values()
        )
        report["p95_under_100ms"] = all(
            load["p95_ms"] <= 100 for load in report["loads"].values()
        )
        return report
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("checkpoint")
    export.add_argument("output")
    bench = sub.add_parser("benchmark")
    bench.add_argument("source")
    bench.add_argument("--count", type=int, default=100)
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--output")
    serve = sub.add_parser("serve")
    serve.add_argument("artifact")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8810)
    serve.add_argument("--device", default="cuda")
    serve.add_argument("--max-pending", type=int, default=100)
    serve.add_argument("--timeout", type=float, default=5.0)
    for command in (bench, serve):
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "export":
        print(export_actor(args.checkpoint, args.output))
        return
    import torch

    torch.set_num_threads(args.threads)
    if args.command == "benchmark":
        report = benchmark(args.source, count=args.count, device=args.device)
        rendered = (
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if args.output:
            Path(args.output).write_text(rendered)
        print(rendered, end="")
        if not report["passed_legality"]:
            raise SystemExit(1)
        return
    if not Path(args.artifact).is_dir():
        raise ValueError(
            "serve requires an exported Actor directory, not a training checkpoint"
        )
    actor, metadata = load_actor(args.artifact, args.device)
    service = BatchService(
        actor, metadata, max_pending=args.max_pending, timeout=args.timeout
    )
    server = create_server(service, args.host, args.port)
    print(
        json.dumps(
            {"listen": f"http://{args.host}:{server.server_port}", "format": FORMAT}
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
