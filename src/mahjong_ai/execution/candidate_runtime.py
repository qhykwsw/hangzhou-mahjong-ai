"""Frozen code packages and an observation-only, time-bounded process interface."""

from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import threading
from dataclasses import asdict, fields
from pathlib import Path

from mahjong_ai.persistence.artifacts import atomic_json, digest, file_hash, read_json
from mahjong_ai.domain.types import RoundResult
from mahjong_ai.execution.process_usage import memory_kb


class JobCancelled(InterruptedError):
    """An operator/scheduler interruption, never a candidate quality failure."""


class ActionUnavailable(RuntimeError):
    """The environment must apply its own timeout/default action."""


# Set while a deadline-cut job freezes its checkpoint. The budget signal already
# fired, and aborting here would discard the frozen version it was meant to keep.
_FINALIZING = False


def begin_finalize():
    """Enter the checkpoint-freeze window that a deadline cut has to be able to finish.

    The deadline signal writes cancel.flag before freezing starts, so the flag must
    stop interrupting the job here: discarding the committed checkpoint would lose
    the version the iteration limit was supposed to preserve.
    """
    global _FINALIZING
    _FINALIZING = True


def is_finalizing():
    return _FINALIZING


def check_cancelled():
    if _FINALIZING:
        return
    cancel = os.environ.get("MAHJONG_JOB_CANCEL")
    if cancel and Path(cancel).exists():
        raise JobCancelled("任务已取消")


_MEMORY_LOCK = threading.Lock()
_MEMORY_WATCHED = {}
_MEMORY_THREAD = None
_MEMORY_INTERVAL = 0.5


def _watch_memory(owner, limit_kb):
    """Register a candidate process with the single shared memory watcher.

    One poller for the whole job replaces one `ps` fork per process per 200ms:
    at twelve concurrent matches that fork storm dominated host CPU. The
    watcher still kills a runaway candidate at the same RSS limit.
    """
    global _MEMORY_THREAD
    with _MEMORY_LOCK:
        _MEMORY_WATCHED[owner.process.pid] = (owner, limit_kb)
        if _MEMORY_THREAD is None or not _MEMORY_THREAD.is_alive():
            _MEMORY_THREAD = threading.Thread(target=_memory_watch_loop, daemon=True)
            _MEMORY_THREAD.start()


def _unwatch_memory(pid):
    with _MEMORY_LOCK:
        _MEMORY_WATCHED.pop(pid, None)


def _memory_watch_loop():
    while True:
        with _MEMORY_LOCK:
            targets = {
                pid: (entry[0], entry[1], entry[0].process)
                for pid, entry in list(_MEMORY_WATCHED.items())
                if entry[0].process is not None and entry[0].process.poll() is None
            }
            _MEMORY_WATCHED.clear()
            _MEMORY_WATCHED.update(
                (pid, (owner, limit))
                for pid, (owner, limit, _process) in targets.items()
            )
            if not targets:
                # A new registration can race the final empty poll. Clear the
                # sentinel while holding the same lock used by _watch_memory so
                # that it will start a fresh watcher instead of going unpolled.
                global _MEMORY_THREAD
                _MEMORY_THREAD = None
                return
        usage = memory_kb(sorted(targets))
        for pid, (owner, limit, process) in targets.items():
            rss = usage.get(pid)
            if rss is None:
                continue
            if rss > owner.peak_rss_kb:
                owner.peak_rss_kb = rss
            if rss > limit:
                owner.memory_exceeded = True
                try:
                    # close() may have detached the owner after the snapshot;
                    # kill the captured process object, if it is still alive.
                    if process.poll() is None:
                        process.kill()
                except (OSError, ProcessLookupError):
                    pass
        time.sleep(_MEMORY_INTERVAL)


def tree_hash(root):
    root = Path(root)
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("候选包不能包含符号链接")
        if p.is_file() and "__pycache__" not in p.parts:
            files[str(p.relative_to(root))] = file_hash(p)
    return digest(files)


def sandbox_command(argv, readable, writable, *, no_fork=True, network=False):
    """Enforce equivalent candidate boundaries with the host OS sandbox."""
    if sys.platform == "linux":
        from mahjong_ai.execution.linux_sandbox import command

        return command(argv, readable, writable, no_fork=no_fork, network=network)
    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        raise RuntimeError(
            "隔离候选进程需要 macOS sandbox-exec 或 Linux/WSL bubblewrap"
        )

    def sub(path):
        return "(subpath " + json.dumps(str(Path(path).resolve())) + ")"

    base = [
        "/System",
        "/usr",
        "/bin",
        "/sbin",
        "/Library",
        "/private/var/db",
        "/dev",
        sys.base_prefix,
        sys.prefix,
        Path(__file__).parents[2],
        "/opt/homebrew/opt/libomp",
        "/opt/homebrew/Cellar/libomp",
    ]
    profile = "(version 1)(allow default)(deny file-read*)(deny file-write*)(deny process-info*)"
    if not network:
        profile += "(deny network*)"
    profile += "(allow process-info* (target self))"
    profile += (
        '(allow file-read-metadata)(allow file-read* (literal "/") '
        + " ".join(sub(p) for p in base + list(readable))
        + ")"
    )
    profile += (
        '(allow file-read* file-write* (literal "/dev/null") (literal "/dev/urandom") '
        + " ".join(sub(p) for p in writable)
        + ")"
    )
    if no_fork:
        profile += "(deny process-fork)"
    # No network, no host home, no experiment files, no inherited secrets.
    return ["/usr/bin/sandbox-exec", "-p", profile, *argv]


def sandbox_popen(argv, **kwargs):
    if sys.platform == "linux":
        from mahjong_ai.execution.linux_sandbox import spawn

        return spawn(argv, **kwargs)
    return subprocess.Popen(argv, **kwargs)


def public_result(result):
    return RoundResult(
        result.engine,
        0,
        result.stage,
        result.rounds_played,
        result.total_score_by_seat,
        result.winner_seat,
        result.top_two_seats,
        result.completed,
        {
            "objective": result.metadata.get("objective"),
            "reason": result.metadata.get("reason", ""),
        },
        result.rank_by_seat,
    )


class CandidateProcess:
    def __init__(self, package, python=None, output=None, runtime=None):
        self.package = Path(package).resolve()
        if python is not None:
            interpreter = Path(python).expanduser()
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                raise FileNotFoundError(f"候选解释器不可用：{interpreter}")
        self.temp = tempfile.TemporaryDirectory(prefix="mahjong-candidate-")
        self.output = (
            Path(output).resolve() if output else Path(self.temp.name) / "output"
        )
        self.output.mkdir(parents=True, exist_ok=True)
        runtime = runtime or str(Path(__file__).parents[2])
        argv = [
            python or sys.executable,
            "-B",
            "-m",
            "mahjong_ai.execution.candidate_worker",
            str(self.package),
            str(self.output),
        ]
        readable = [self.package, runtime]
        if python:
            readable.append(Path(python).parent.parent)
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": runtime,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "1"),
            "TMPDIR": self.temp.name,
            "HOME": self.temp.name,
        }
        self.process = sandbox_popen(
            sandbox_command(argv, readable, [self.temp.name, self.output]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.package,
            env=env,
            start_new_session=True,
            bufsize=0,
        )
        self.pending = bytearray()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.stderr = bytearray()
        self.failure = None

        def drain():
            while True:
                data = self.process.stderr.read(4096)
                if not data:
                    break
                self.stderr.extend(data)
                del self.stderr[:-65536]

        self.stderr_thread = threading.Thread(target=drain, daemon=True)
        self.stderr_thread.start()
        limit = int(os.environ.get("MAHJONG_MEMORY_MB", "4096")) * 1024
        self.peak_rss_kb = 0
        self.memory_exceeded = False
        _watch_memory(self, limit)

    def send(self, data):
        self.process.stdin.write((json.dumps(data, allow_nan=False) + "\n").encode())
        self.process.stdin.flush()

    def receive(self, seconds):
        end = time.monotonic() + max(0.001, seconds)
        while b"\n" not in self.pending:
            cancel = os.environ.get("MAHJONG_JOB_CANCEL")
            if cancel and Path(cancel).exists() and not _FINALIZING:
                raise JobCancelled("任务已取消")
            if not self.selector.select(min(0.25, max(0, end - time.monotonic()))):
                if time.monotonic() >= end:
                    raise TimeoutError("候选进程响应超时")
                continue
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(
                    "候选进程超出内存上限" if self.memory_exceeded else "候选进程已退出"
                )
            self.pending.extend(data)
            if len(self.pending) > 8_000_000:
                raise ValueError("候选进程回执过大")
        line, _, rest = self.pending.partition(b"\n")
        self.pending = bytearray(rest)
        value = json.loads(line)
        if "error" in value:
            self.failure = value
            raise RuntimeError(value["error"])
        return value

    def call(self, data, seconds=10):
        self.send(data)
        return self.receive(seconds)

    def close(self):
        if getattr(self, "process", None):
            _unwatch_memory(self.process.pid)
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait()
            self.stderr_thread.join(timeout=1)
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                pipe.close()
            self.selector.close()
            self.process = None
        self.temp.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def observation_payload(observation, previous_events=None):
    """Serialize an observation and append only newly published public events."""
    events = observation.public_events
    data = {}
    for field in fields(observation):
        if field.name == "public_events":
            continue
        value = getattr(observation, field.name)
        if field.name == "legal_actions":
            data[field.name] = [asdict(action) for action in value]
        elif field.name == "melds":
            data[field.name] = [[asdict(meld) for meld in row] for row in value]
        elif field.name == "stage":
            data[field.name] = asdict(value)
        else:
            data[field.name] = value
    if (
        previous_events is not None
        and len(events) > len(previous_events)
        and events[: len(previous_events)] == previous_events
    ):
        data["events_from"] = len(previous_events)
        data["events"] = [asdict(event) for event in events[len(previous_events) :]]
    else:
        data["public_events"] = [asdict(event) for event in events]
    return data, events


class RemotePolicy:
    """Never imports candidate code in the engine process; validates action indices."""

    name = "code-policy"

    def __init__(
        self,
        package,
        package_hash,
        action_seconds=3.0,
        python=None,
        version="code",
        response_seconds=1.0,
        runtime=None,
    ):
        if tree_hash(package) != package_hash:
            raise ValueError("冻结的候选发生变化")
        self.package, self.package_hash, self.action_seconds = (
            package,
            package_hash,
            action_seconds,
        )
        self.python, self.version, self.worker = python, version, None
        self.response_seconds = response_seconds
        self.runtime = runtime
        self.reuse_process = True
        self.reset_count = 0
        self.pending_act = False
        self.sent_events = None
        self.stats = {"actions": 0, "timeouts": 0, "errors": 0, "elapsed_seconds": 0.0}
        self.stats.update(compute_seconds=0.0, ipc_seconds=0.0, fallback_actions=0)
        self.last_error = None
        self.latencies = []

    def _start(self):
        self.worker = CandidateProcess(self.package, self.python, runtime=self.runtime)
        try:
            response = self.worker.call({"op": "build"})
            self.reuse_process = bool(response.get("reuse_process", True))
        except Exception:
            self.close()
            raise
        self.sent_events = None

    def start(self):
        """Eagerly start the sandbox so several policies can start in parallel."""
        if self.worker is None:
            self._start()

    def reset(self, seed):
        try:
            if self.pending_act:
                self.close()
            elif (
                self.worker is not None and self.reset_count and not self.reuse_process
            ):
                self.close()
            if self.worker is None:
                self._start()
            self.worker.call({"op": "reset", "seed": seed})
            self.sent_events = None
            self.reset_count += 1
        except JobCancelled:
            self.close()
            raise
        except Exception as exc:
            self._error(exc)
            self.stats["errors"] += 1
            self.close()
            # A reset/build failure means this seat is not an evaluable policy.
            # Let the enclosing task fail instead of silently converting an
            # entire candidate into timeout fallback actions.
            raise RuntimeError(f"候选启动或 reset 失败：{exc}") from exc

    def act(self, observation):
        started = time.monotonic()
        end = started + (
            self.response_seconds
            if observation.phase.startswith("response_")
            else self.action_seconds
        )
        self.stats["actions"] += 1
        try:
            if self.worker is None:
                raise ActionUnavailable("候选进程不可用")
            # Keep a slow policy alive. Drain and discard its previous late reply
            # before sending another observation; replies cannot cross decisions.
            if self.pending_act:
                self.worker.receive(max(0.001, end - time.monotonic()))
                self.pending_act = False
            if time.monotonic() >= end:
                raise TimeoutError("上一个动作仍占用决策窗口")
            request = {"op": "act", "observation": self._payload(observation)}
            self.worker.send(request)
            self.pending_act = True
            reply = self.worker.receive(max(0.001, end - time.monotonic()))
            self.pending_act = False
            index = reply["index"]
            compute = reply.get("compute_seconds", 0.0)
            if (
                not isinstance(compute, (float, int))
                or not 0 <= compute <= time.monotonic() - started
            ):
                compute = 0.0
            # Worker-reported compute timing is diagnostic; host elapsed is authoritative.
            self.stats["compute_seconds"] += compute
            self.stats["ipc_seconds"] += max(0, time.monotonic() - started - compute)
            if type(index) is not int or not 0 <= index < len(
                observation.legal_actions
            ):
                raise ValueError("动作下标不合法")
            return observation.legal_actions[index]
        except JobCancelled:
            self.close()
            raise
        except ActionUnavailable:
            pass
        except TimeoutError as exc:
            self._error(exc)
            self.stats["timeouts"] += 1
        except Exception as exc:
            self._error(exc)
            self.stats["errors"] += 1
            self.close()
        finally:
            elapsed = time.monotonic() - started
            self.stats["elapsed_seconds"] += elapsed
            self.latencies.append(elapsed)
            if len(self.latencies) > 4096:
                del self.latencies[:2048]
        self.stats["fallback_actions"] += 1
        raise ActionUnavailable("候选未在时限内给出合法动作")

    def _payload(self, observation):
        """Full observation, with public events sent as an append-only delta.

        The event log only grows inside one game and the worker rebuilds it once,
        so a round trip carries the newly published events instead of the whole
        history (26KB per action on average in the fast pools).
        """
        data, events = observation_payload(observation, self.sent_events)
        self.sent_events = events
        return data

    def finish_round(self, result, seat):
        if self.worker:
            try:
                if self.pending_act:
                    self.worker.receive(self.action_seconds)
                    self.pending_act = False
                self.worker.call(
                    {
                        "op": "finish",
                        "result": asdict(public_result(result)),
                        "seat": seat,
                    },
                    self.action_seconds,
                )
            except JobCancelled:
                self.close()
                raise
            except Exception as exc:
                self._error(exc)
                self.stats["errors"] += 1
                self.close()

    def close(self):
        self.pending_act = False
        if self.worker:
            self.worker.close()
            self.worker = None

    def _error(self, exc):
        self.last_error = {"message": str(exc)[:4000], "type": type(exc).__name__}
        if self.worker:
            self.last_error["traceback"] = (self.worker.failure or {}).get(
                "traceback", []
            )
            self.last_error["stderr"] = self.worker.stderr.decode(errors="replace")[
                -8000:
            ]

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def train_package(
    package,
    output,
    config,
    opponents,
    episodes,
    seed,
    deadline,
    python=None,
    policy_seed=None,
    runtime=None,
):
    """Candidate owns train(ctx); ctx.play(policy) delegates real games to this host."""
    from mahjong_ai.domain.engine import MahjongEngine, derive_seed

    worker = CandidateProcess(package, python, output, runtime=runtime)
    records = []
    progress = Path(output).parent / "training-checkpoint.json"
    checkpoint = None
    if progress.exists():
        checkpoint = read_json(progress)
        if checkpoint["hash"] != tree_hash(output):
            # Candidate may have been interrupted midway through a newer save.
            # Restore the last atomically committed model directory.
            shutil.rmtree(output)
            committed = Path(output).parent / checkpoint["directory"]
            if tree_hash(committed) != checkpoint["hash"]:
                raise ValueError("训练检查点损坏")
            shutil.copytree(committed, output)
        records = checkpoint["episodes"]

    class Bridge:
        name, version = "training", "code"

        def __init__(self):
            self.sent_events = None

        def reset(self, seed):
            self.sent_events = None
            worker.call(
                {"op": "reset", "seed": seed}, max(0.001, deadline - time.monotonic())
            )

        def act(self, obs):
            payload, self.sent_events = observation_payload(obs, self.sent_events)
            reply = worker.call(
                {"op": "act", "observation": payload},
                max(0.001, deadline - time.monotonic()),
            )
            index = reply["index"]
            if type(index) is not int or not 0 <= index < len(obs.legal_actions):
                raise ValueError("训练中出现非法动作")
            return obs.legal_actions[index]

        def finish_round(self, result, seat):
            worker.call(
                {"op": "finish", "result": asdict(public_result(result)), "seat": seat},
                max(0.001, deadline - time.monotonic()),
            )

    try:
        worker.send(
            {
                "op": "train",
                "episodes": episodes,
                "config": config.as_dict(),
                "completed_episodes": len(records),
                "checkpoint": checkpoint.get("metadata") if checkpoint else None,
            }
        )
        while True:
            reply = worker.receive(max(0.001, deadline - time.monotonic()))
            if reply.get("request") == "checkpoint":
                fingerprint = tree_hash(output)
                committed = Path(output).parent / ("training-committed-" + fingerprint)
                staging = Path(output).parent / "training-committing"
                if staging.exists():
                    shutil.rmtree(staging)
                if not committed.exists():
                    shutil.copytree(output, staging)
                    staging.rename(committed)
                atomic_json(
                    progress,
                    {
                        "episodes": records,
                        "hash": fingerprint,
                        "metadata": reply.get("metadata"),
                        "directory": committed.name,
                    },
                )
                for previous in Path(output).parent.glob("training-committed-*"):
                    if previous != committed:
                        shutil.rmtree(previous)
                worker.send({"ok": True})
                continue
            if reply.get("done"):
                return {
                    "status": "completed",
                    "episodes": records,
                    "reported_metrics": reply.get("metrics", {}),
                }
            if reply.get("request") != "play" or len(records) >= episodes:
                raise ValueError("训练对局额度用尽或请求不合法")
            i = len(records)
            seat = i % 4
            chosen_opponents = opponents(i) if callable(opponents) else opponents
            players = [p.build() for p in chosen_opponents]
            players.insert(seat, Bridge())
            try:
                seeds = (
                    [
                        derive_seed(policy_seed, "training-policy", i, j)
                        for j in range(4)
                    ]
                    if policy_seed is not None
                    else None
                )
                result = MahjongEngine().run_round(
                    players,
                    config,
                    derive_seed(seed, "training", i),
                    policy_seeds=seeds,
                )
            finally:
                for p in players:
                    if hasattr(p, "close"):
                        p.close()
            if not result.completed:
                raise ValueError("训练结算不完整")
            records.append(
                {
                    "episode": i,
                    "rank": result.rank_by_seat[seat],
                    "rank_utility": result.rank_utility(seat),
                    "hands": result.rounds_played,
                    "score": result.score(seat),
                }
            )
            worker.send({"result": asdict(public_result(result)), "seat": seat})
    finally:
        worker.close()
