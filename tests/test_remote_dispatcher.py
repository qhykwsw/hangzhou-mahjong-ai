import json
import time

import mahjong_ai.execution.remote as remote
from mahjong_ai.execution.remote import Dispatcher


class Host:
    """A remote host whose reachability, reboots and training the test controls."""

    name, service_code = "gpu", "/code"

    def __init__(self):
        self.up, self.boot, self.alive, self.signals = True, 0.0, True, []

    def python(self, script, timeout=None):
        if not self.up:
            raise RuntimeError("unreachable")
        if "SIGSTOP" in script or "SIGCONT" in script:
            self.signals.append("SIGSTOP" if "SIGSTOP" in script else "SIGCONT")
            return [1]
        if "/proc/uptime" in script:
            return self.boot
        return True

    def sync(self, paths):
        if not self.up:
            raise RuntimeError("unreachable")

    def build_native(self, runtimes):
        pass

    def launch_many(self, jobs):
        if not self.up:
            raise RuntimeError("unreachable")
        return list(range(len(jobs)))

    def poll(self, jobs):
        if not self.up:
            raise RuntimeError("unreachable")
        return {j: {"alive": self.alive} for j in jobs}

    def cancel(self, root, pid):
        pass


def wait(condition, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end and not condition():
        time.sleep(0.05)
    return condition()


def dispatcher(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "job_inputs", lambda task: set())
    monkeypatch.setattr(remote, "model_weights", lambda task: set())
    host = Host()
    return host, Dispatcher(host, tmp_path, poll_seconds=0.1, probe_seconds=0.1)


def receipt(root):
    path = root / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_unreachable_launch_and_reboot_are_host_faults(tmp_path, monkeypatch):
    host, d = dispatcher(tmp_path, monkeypatch)
    host.up = False
    (tmp_path / "a").mkdir()
    d.submit("a", tmp_path / "a", {"scope": "paired"}, None, {})
    assert wait(lambda: receipt(tmp_path / "a"))
    assert receipt(tmp_path / "a")["status"] == "host_fault"
    # Unreachable while running: the job keeps waiting, then a reboot loses it.
    host.up = True
    (tmp_path / "b").mkdir()
    d.submit("b", tmp_path / "b", {"scope": "paired"}, None, {})
    assert wait(lambda: "b" in d.running)
    host.up = False
    time.sleep(0.4)
    assert receipt(tmp_path / "b") is None
    host.up, host.alive, host.boot = True, False, time.time() + 5
    assert wait(lambda: receipt(tmp_path / "b"))
    assert receipt(tmp_path / "b")["status"] == "host_fault"


def test_a_crash_without_reboot_is_a_failure_and_offline_needs_time(tmp_path, monkeypatch):
    host, d = dispatcher(tmp_path, monkeypatch)
    (tmp_path / "c").mkdir()
    d.submit("c", tmp_path / "c", {"scope": "paired"}, None, {})
    assert wait(lambda: "c" in d.running)
    host.alive = False
    assert wait(lambda: receipt(tmp_path / "c"))
    assert receipt(tmp_path / "c")["status"] == "failed"
    host.up = False
    time.sleep(0.3)
    assert not d.offline()
    d.last_ok -= d.OFFLINE_SECONDS
    assert d.offline()


def test_adopted_job_is_polled_and_a_moved_job_is_ignored(tmp_path, monkeypatch):
    host, d = dispatcher(tmp_path, monkeypatch)
    root = tmp_path / "d"
    root.mkdir()
    (root / "remote.json").write_text(json.dumps({"host": "gpu", "pid": 3, "at": time.time()}))
    assert d.adopt("d", root) and "d" in d.running
    d.forget("d")
    host.alive = False
    time.sleep(0.4)
    assert receipt(root) is None


def test_a_long_sync_does_not_make_a_reachable_host_offline(tmp_path, monkeypatch):
    host, d = dispatcher(tmp_path, monkeypatch)
    d.OFFLINE_SECONDS = 0.5
    host.sync = lambda paths: time.sleep(1.5)  # the dispatch loop is busy for 3x the limit
    (tmp_path / "e").mkdir()
    d.submit("e", tmp_path / "e", {"scope": "competition"}, None, {})
    for _ in range(12):
        time.sleep(0.1)
        assert not d.offline()


def test_a_host_last_known_offline_starts_offline_until_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "job_inputs", lambda task: set())
    host = Host()
    host.up = False
    d = Dispatcher(host, tmp_path, poll_seconds=0.1, probe_seconds=0.1, online=False)
    assert d.offline()
    host.up = True
    assert wait(lambda: not d.offline())
