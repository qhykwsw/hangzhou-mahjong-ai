import io
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
from mahjong_ai.execution.agent import codex_command
from mahjong_ai.persistence.artifacts import read_json
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.strategies.common import PolicySpec


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
    except PermissionError: pass
    else: raise AssertionError("private read allowed")
    try: socket.socket().connect(("127.0.0.1", 9))
    except PermissionError: pass
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

    class Ps:
        stdout = "123 2\n"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(candidate_runtime.subprocess, "run", lambda *a, **k: Ps())

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


def test_codex_only_receives_research_tools(tmp_path):
    command = codex_command(tmp_path, "run-id", "rule")
    assert "--ignore-user-config" in command
    assert "features.shell_tool=false" in command
    assert "features.unified_exec=false" in command
    assert not any("mcp_servers.playwright" in x for x in command)


def test_training_checkpoint_recovers_same_model(tmp_path):
    from mahjong_ai.execution.candidate_runtime import train_package
    from mahjong_ai.strategies.seeds import default_opponents
    import numpy as np

    package = tmp_path / "package"
    package.mkdir()
    (
        package / "strategy.py"
    ).write_text("""from mahjong_ai.strategies.common import RLPolicy,TrainingRecipe
import time
def train(ctx):
    path=ctx.output/'model.npz'
    learner=RLPolicy(TrainingRecipe(),path if ctx.completed_episodes else None,training=True)
    start=ctx.completed_episodes
    for i in range(start,ctx.episodes):
        ctx.play(learner)
        learner.save(path)
        ctx.checkpoint({'index':i})
        if start==0 and i==0:time.sleep(2)
""")
    out = tmp_path / "partial" / "model"
    with pytest.raises(TimeoutError):
        train_package(
            package,
            out,
            RoundConfig(rounds=1),
            default_opponents(),
            3,
            19,
            time.monotonic() + 0.5,
        )
    progress = read_json(out.parent / "training-checkpoint.json")
    assert len(progress["episodes"]) == 1
    resumed = train_package(
        package,
        out,
        RoundConfig(rounds=1),
        default_opponents(),
        3,
        19,
        time.monotonic() + 10,
    )
    whole = tmp_path / "whole" / "model"
    direct = train_package(
        package,
        whole,
        RoundConfig(rounds=1),
        default_opponents(),
        3,
        19,
        time.monotonic() + 10,
    )
    assert resumed["episodes"] == direct["episodes"]
    with np.load(out / "model.npz") as a, np.load(whole / "model.npz") as b:
        for key in ("w", "b", "out", "value"):
            np.testing.assert_array_equal(a[key], b[key])


def test_frozen_checkpoint_requires_committed_episodes(tmp_path):
    from mahjong_ai.execution.jobs import frozen_checkpoint

    out = tmp_path / "candidate"
    assert frozen_checkpoint(out) is None
    committed = out / "training-committed-abc"
    committed.mkdir(parents=True)
    (committed / "model.npz").write_bytes(b"weights")
    checkpoint = out / "training-checkpoint.json"
    checkpoint.write_text(
        '{"episodes": [{"episode": 0}], "hash": "abc", '
        '"directory": "training-committed-abc", "metadata": {"algorithm": "mlp"}}'
    )
    path, record = frozen_checkpoint(out)
    assert path == committed
    assert record["status"] == "partial"
    assert record["metadata"] == {"algorithm": "mlp"}
    assert record["interrupted"] == "iteration execution limit reached"
    checkpoint.write_text('{"episodes": [], "directory": "training-committed-abc"}')
    assert frozen_checkpoint(out) is None


def test_iteration_deadline_freezes_checkpoint_for_evaluation(tmp_path):
    """Training cut by the iteration limit still yields an evaluable version."""
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    from mahjong_ai.execution.candidate_runtime import tree_hash
    from mahjong_ai.persistence.artifacts import read_json, runtime_fingerprint

    package = tmp_path / "candidate"
    package.mkdir()
    (package / "strategy.py").write_text(
        "COMBINATION = 'heuristic-unrestricted'\n"
        "from mahjong_ai.strategies.common import RLPolicy, RulePolicy, TrainingRecipe\n"
        "import time\n"
        "def train(ctx):\n"
        "    path = ctx.output / 'model.npz'\n"
        "    learner = RLPolicy(TrainingRecipe(), None, training=True)\n"
        "    ctx.play(learner)\n"
        "    learner.save(path)\n"
        "    ctx.checkpoint({'index': 0})\n"
        # Commit one checkpoint first, then stall: the deadline always lands on a
        # job that already has resumable state, so the freeze path is not racing
        # the first commit on a loaded machine.
        "    deadline = time.monotonic() + 300\n"
        "    while time.monotonic() < deadline:\n"
        "        time.sleep(1)\n"
        "def build(model):\n"
        "    return RulePolicy()\n"
    )
    root = tmp_path / "job"
    root.mkdir()
    output = tmp_path / "versions" / "candidate-1"
    task = {
        "kind": "prepare",
        "project": str(tmp_path),
        "experiment": "test",
        "runtime_path": str(Path(__file__).resolve().parents[1] / "src"),
        "runtime": runtime_fingerprint(),
        "config": RoundConfig(rounds=1).as_dict(),
        "timeouts": {"discard": 3.0, "response": 1.0},
        "resources": {"memory_mb": 1024, "threads": 1, "device": "cpu"},
        "seed": 7,
        "packages": {
            "candidate-1": {
                "input": str(package),
                "hash": tree_hash(package),
                "requirements": [],
                "output": str(output),
            }
        },
        "combination": "heuristic-unrestricted",
        "definition": None,
        "training_games": 100000,
        "opponents": [],
        # Enough wall time to import and commit one checkpoint even while a live
        # experiment saturates the machine; the stall keeps training unfinished.
        "deadline": time.time() + 30,
        "owner_pid": os.getpid(),
    }
    (root / "task.json").write_text(json.dumps(task))
    source = Path(__file__).resolve().parents[1] / "src"
    subprocess.run(
        [sys.executable, "-B", "-m", "mahjong_ai.execution.jobs", str(root)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(source)},
        check=True,
        timeout=180,
    )
    result = read_json(root / "result.json")
    row = result["candidates"]["candidate-1"]
    assert result["status"] == "completed"
    assert row["status"] == "ready"
    assert row["training"]["status"] == "partial"
    assert 0 < len(row["training"]["episodes"]) < 100000
    assert (output / "strategy.py").exists()
    assert (output / "model" / "model.npz").exists()
    assert row["spec"]["package_path"] == str(output)


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
