import json
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import time

import pytest

from mahjong_ai.execution.candidate_runtime import sandbox_command


def clean_env(temp):
    """A sandboxed child's whole environment: the repository's sources and a private home."""
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(temp),
        "TMPDIR": str(temp),
        "OMP_NUM_THREADS": "1",
    }


def run_sandbox(source, readable, writable, **options):
    return subprocess.run(
        sandbox_command(
            [sys.executable, "-B", "-c", source], readable, writable, **options
        ),
        env=clean_env(writable[0]),
        cwd=writable[0],
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_sandbox_files_network_processes_and_threads(tmp_path):
    private = tmp_path / "private.txt"
    private.write_text("hidden state")
    public = tmp_path / "public"
    public.mkdir()
    allowed = public / "allowed.txt"
    allowed.write_text("public observation")
    output = tmp_path / "output"
    output.mkdir()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        source = f"""
import json, os, socket, subprocess, threading
from pathlib import Path
result = {{}}
result['read'] = Path({str(allowed)!r}).read_text()
Path({str(output / "artifact")!r}).write_text('model')
for name, action in [
    ('private', lambda: Path({str(private)!r}).read_text()),
    ('readonly', lambda: Path({str(allowed)!r}).write_text('changed')),
    ('network', lambda: socket.create_connection({server.getsockname()!r}, timeout=1)),
    ('subprocess', lambda: subprocess.run(['/bin/true'], check=True)),
]:
    try:
        action()
        result[name] = 'allowed'
    except OSError:
        result[name] = 'denied'
try:
    child = os.fork()
except OSError:
    result['fork'] = 'denied'
else:
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    result['fork'] = 'allowed'
thread = threading.Thread(target=lambda: result.update(thread='allowed'))
thread.start()
thread.join()
print(json.dumps(result))
"""
        result = run_sandbox(source, [public], [output])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "read": "public observation",
        "private": "denied",
        "readonly": "denied",
        "network": "denied",
        "subprocess": "denied",
        "fork": "denied",
        "thread": "allowed",
    }
    assert (output / "artifact").read_text() == "model"


def test_missing_sandbox_fails_closed(monkeypatch):
    from mahjong_ai.execution import candidate_runtime

    monkeypatch.setattr(candidate_runtime.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="sandbox-exec|bubblewrap"):
        sandbox_command([sys.executable, "-c", "pass"], [], [])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death behavior")
def test_sandbox_dies_with_host_process(tmp_path):
    source = f"""
import subprocess, sys, time
from mahjong_ai.execution.candidate_runtime import sandbox_command, sandbox_popen
child = sandbox_popen(
    sandbox_command([sys.executable, '-c', 'import time; print("ready", flush=True); time.sleep(60)'], [], [{str(tmp_path)!r}]),
    stdout=subprocess.PIPE, text=True, env={clean_env(tmp_path)!r},
)
assert child.stdout.readline() == 'ready\\n'
print(child.pid, flush=True)
time.sleep(60)
"""
    host = subprocess.Popen(
        [sys.executable, "-B", "-c", source], stdout=subprocess.PIPE, text=True
    )
    descendants = set()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(host.stdout, selectors.EVENT_READ)
            assert selector.select(15), "sandbox startup timed out"
        descendants.add(int(host.stdout.readline()))
        pending = list(descendants)
        while pending:
            pid = pending.pop()
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            for child in children:
                descendants.add(int(child))
                pending.append(int(child))
        assert len(descendants) > 1
        host.kill()
        host.wait(timeout=5)
        deadline = time.monotonic() + 5
        while descendants and time.monotonic() < deadline:
            for pid in list(descendants):
                status = Path(f"/proc/{pid}/status")
                try:
                    state = next(
                        line
                        for line in status.read_text().splitlines()
                        if line.startswith("State:")
                    )
                except FileNotFoundError:
                    descendants.remove(pid)
                    continue
                if "Z (zombie)" in state:
                    descendants.remove(pid)
            time.sleep(0.05)
        assert not descendants, f"orphaned sandbox processes: {descendants}"
    finally:
        if host.poll() is None:
            host.kill()
        host.wait(timeout=5)
        host.stdout.close()
