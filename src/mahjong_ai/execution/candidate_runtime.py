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
from dataclasses import asdict
from pathlib import Path

from mahjong_ai.persistence.artifacts import atomic_json, digest, file_hash, read_json
from mahjong_ai.domain.types import RoundResult


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
    """macOS seatbelt is an enforced allowlist, not a prompt-based restriction."""
    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        raise RuntimeError("隔离候选进程需要 macOS sandbox-exec，没有不安全的降级路径")

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
        self.process = subprocess.Popen(
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

        def monitor_memory():
            process = self.process
            while process is not None and process.poll() is None:
                row = subprocess.run(
                    ["/bin/ps", "-p", str(process.pid), "-o", "rss="],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if row.isdigit():
                    self.peak_rss_kb = max(self.peak_rss_kb, int(row))
                    if int(row) > limit:
                        self.memory_exceeded = True
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        break
                time.sleep(0.2)

        self.memory_thread = threading.Thread(target=monitor_memory, daemon=True)
        self.memory_thread.start()

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
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait()
            self.stderr_thread.join(timeout=1)
            self.memory_thread.join(timeout=1)
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
        self.pending_act = False
        self.stats = {"actions": 0, "timeouts": 0, "errors": 0, "elapsed_seconds": 0.0}
        self.stats.update(compute_seconds=0.0, ipc_seconds=0.0, fallback_actions=0)
        self.last_error = None
        self.latencies = []

    def _start(self):
        self.worker = CandidateProcess(self.package, self.python, runtime=self.runtime)
        self.worker.call({"op": "build"})

    def reset(self, seed):
        try:
            if self.pending_act:
                self.close()
            if self.worker is None:
                self._start()
            self.worker.call({"op": "reset", "seed": seed})
        except JobCancelled:
            self.close()
            raise
        except Exception as exc:
            self._error(exc)
            self.stats["errors"] += 1
            self.close()

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
            request = {"op": "act", "observation": asdict(observation)}
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

        def reset(self, seed):
            worker.call(
                {"op": "reset", "seed": seed}, max(0.001, deadline - time.monotonic())
            )

        def act(self, obs):
            reply = worker.call(
                {"op": "act", "observation": asdict(obs)},
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
