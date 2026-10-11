import io
from concurrent.futures import ThreadPoolExecutor
import selectors
import time
from types import SimpleNamespace

import pytest

from mahjong_ai.execution import candidate_runtime
from mahjong_ai.execution.candidate_runtime import (
    CandidateProcess,
    JobCancelled,
    RemotePolicy,
    tree_hash,
)
from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.strategies.common import PolicySpec


def test_candidate_outlives_parallel_startup_thread(tmp_path):
    (tmp_path / "strategy.py").write_text(
        "from mahjong_ai.strategies.common import RulePolicy\n"
        "def build(model): return RulePolicy()\n"
    )
    policy = RemotePolicy(tmp_path, tree_hash(tmp_path))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(policy.start).result()
        time.sleep(0.05)
        policy.reset(17)
        assert policy.worker.process.poll() is None
    finally:
        policy.close()


def test_candidate_cannot_read_host_files_or_network(tmp_path):
    secret = tmp_path / "private.txt"
    secret.write_text("secret")
    package = tmp_path / "package"
    package.mkdir()
    source = f"""from pathlib import Path
import socket
from mahjong_ai.strategies.common import RulePolicy
def build(model):
    try: Path({str(secret)!r}).read_text()
    except (PermissionError, FileNotFoundError): pass
    else: raise AssertionError("private read allowed")
    try: socket.socket().connect(("127.0.0.1", 9))
    except OSError: pass
    else: raise AssertionError("network allowed")
    return RulePolicy()
"""
    (package / "strategy.py").write_text(source)
    worker = CandidateProcess(package)
    try:
        assert worker.call({"op": "build"}) == {
            "ok": True,
            "reuse_process": True,
        }
    finally:
        worker.close()


def test_candidate_process_rejects_missing_interpreter_before_spawn(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "strategy.py").write_text("def build(model): return None\n")
    with pytest.raises(FileNotFoundError, match="候选解释器不可用"):
        CandidateProcess(package, str(tmp_path / "missing-python"))


def test_reset_start_failure_is_not_converted_to_fallback(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "strategy.py").write_text("def build(model): return None\n")
    policy = RemotePolicy(
        package,
        tree_hash(package),
        python=str(tmp_path / "missing-python"),
    )
    with pytest.raises(RuntimeError, match="候选启动或 reset 失败"):
        policy.reset(1)


def test_policy_spec_rejects_stale_candidate_interpreter(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "strategy.py").write_text("def build(model): return None\n")
    with pytest.raises(ValueError, match="interpreter is unavailable"):
        PolicySpec(
            family="code",
            package_path=str(package),
            package_hash=tree_hash(package),
            python_path=str(tmp_path / "missing-python"),
        )


def test_candidate_can_opt_out_of_process_reuse(tmp_path):
    (tmp_path / "strategy.py").write_text(
        "PROCESS_REUSE = False\n"
        "class Policy:\n"
        " def reset(self, seed): pass\n"
        " def act(self, obs): return obs.legal_actions[0]\n"
        " def finish_round(self, result, seat): pass\n"
        "def build(model): return Policy()\n"
    )
    p = RemotePolicy(tmp_path, tree_hash(tmp_path), action_seconds=0.2)
    try:
        p.start()
        p.reset(1)
        first = p.worker.process.pid
        p.reset(2)
        assert p.worker.process.pid != first
    finally:
        p.close()


def test_memory_watcher_uses_snapshot_process_after_owner_closes(monkeypatch):
    """A close racing the RSS poll must not dereference owner.process=None."""

    class Process:
        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    owner = SimpleNamespace(process=Process(), peak_rss_kb=0, memory_exceeded=False)
    process = owner.process
    candidate_runtime._MEMORY_WATCHED.clear()
    candidate_runtime._MEMORY_WATCHED[123] = (owner, 1)

    monkeypatch.setattr(candidate_runtime, "memory_kb", lambda pids: {123: 2})

    # Simulate close() detaching the owner immediately after the locked snapshot.
    def sleep(_seconds):
        owner.process = None
        raise StopIteration

    monkeypatch.setattr(candidate_runtime.time, "sleep", sleep)
    with pytest.raises(StopIteration):
        candidate_runtime._memory_watch_loop()
    assert process.killed
    assert owner.memory_exceeded
    candidate_runtime._MEMORY_WATCHED.clear()


def test_slow_policy_times_out_and_falls_back(tmp_path):
    (tmp_path / "strategy.py").write_text(
        "from mahjong_ai.strategies.common import RulePolicy\nclass Slow(RulePolicy):\n def act(self,obs):\n  while True: pass\ndef build(model):return Slow()\n"
    )
    p = RemotePolicy(str(tmp_path), tree_hash(tmp_path), action_seconds=0.02)
    hand = HandState(RoundConfig(rounds=1), 123, 0, 1, [0] * 4, [], False)
    obs = hand.observation()
    p.reset(1)
    started = time.monotonic()
    from mahjong_ai.execution.candidate_runtime import ActionUnavailable

    with pytest.raises(ActionUnavailable):
        p.act(obs)
    assert time.monotonic() - started < 1
    assert p.stats["timeouts"] == 1
    assert p.stats["errors"] == 0
    p.close()



def test_finalize_window_survives_the_deadline_cancel_flag(tmp_path, monkeypatch):
    """时限信号落盘的 cancel.flag 只能打断训练，不能打断它自己触发的检查点冻结。"""
    flag = tmp_path / "cancel.flag"
    flag.touch()
    monkeypatch.setenv("MAHJONG_JOB_CANCEL", str(flag))
    monkeypatch.setattr(candidate_runtime, "_FINALIZING", False)

    with pytest.raises(JobCancelled):
        candidate_runtime.check_cancelled()

    candidate_runtime.begin_finalize()
    candidate_runtime.check_cancelled()

    probe = object.__new__(CandidateProcess)
    probe.pending = bytearray()
    probe.selector = selectors.DefaultSelector()
    probe.process = SimpleNamespace(stdout=io.BytesIO(b""))
    with pytest.raises(TimeoutError):
        probe.receive(0.01)


FIRST = """
#include <Python.h>
static PyObject *first(PyObject *self, PyObject *args) { return PyLong_FromLong(0); }
static PyMethodDef methods[] = {{"first", first, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, methods};
PyMODINIT_FUNC PyInit_native(void) { return PyModule_Create(&module); }
"""


def test_a_sandboxed_candidate_runs_its_own_native_code(tmp_path, monkeypatch):
    """A package with native.c (ADR-0017) plays sandboxed, as on the platform: the
    sandbox can neither compile nor write the machine's cache, so the module is
    built outside it and only read inside."""
    pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")
    monkeypatch.setenv("MAHJONG_NATIVE_CACHE", str(tmp_path / "native-cache"))
    package = tmp_path / "package"
    package.mkdir()
    (package / "native.c").write_text(FIRST)
    (package / "strategy.py").write_text(
        "from mahjong_ai import _fast\n"
        "from mahjong_ai.strategies.common import RulePolicy\n"
        "native = _fast.load_native(__file__)\n"
        "class Policy(RulePolicy):\n"
        "    def act(self, obs):\n"
        "        return obs.legal_actions[native.first()]\n"
        "def build(model):\n"
        "    return Policy()\n"
    )
    policy = RemotePolicy(package, tree_hash(package))
    try:
        policy.reset(1)
        obs = HandState(RoundConfig(rounds=1), 3).observation()
        assert policy.act(obs) == obs.legal_actions[0]
        assert policy.stats["errors"] == policy.stats["fallback_actions"] == 0
    finally:
        policy.close()
