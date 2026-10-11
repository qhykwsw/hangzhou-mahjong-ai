import os
import plistlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mahjong_ai.interfaces import app, service_host


def test_launchd_keeps_existing_lifecycle_and_qos(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(service_host.sys, "platform", "darwin")
    monkeypatch.setattr(
        service_host.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    service_host.start_service(
        tmp_path, "mahjong-test", tmp_path, ["python", "serve"], {"PATH": "/bin"}
    )
    payload = plistlib.loads((tmp_path / "service.plist").read_bytes())
    assert payload["ProcessType"] == "Interactive"
    assert payload["ProgramArguments"] == ["python", "serve"]
    assert payload["KeepAlive"] is False
    assert calls[-1][:2] == ["launchctl", "bootstrap"]
    assert (tmp_path / "service.plist").stat().st_mode & 0o777 == 0o600


def test_systemd_uses_private_environment_and_transient_service(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(service_host.sys, "platform", "linux")
    monkeypatch.setattr(service_host.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        service_host.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    service_host.start_service(
        tmp_path,
        "mahjong-test",
        tmp_path,
        ["python", "$literal"],
        {"TOKEN": 'quote"backslash\\value'},
    )
    assert "--collect" in calls[-1]
    assert "--property=KillMode=mixed" in calls[-1]
    assert calls[-1][-1] == "$$literal"
    assert not any("TOKEN" in argument for argument in calls[-1])
    assert (
        tmp_path / "service.env"
    ).read_text() == 'TOKEN="quote\\"backslash\\\\value"\n'
    assert (tmp_path / "service.env").stat().st_mode & 0o777 == 0o600


def test_browser_failure_does_not_fail_start(monkeypatch, capsys):
    monkeypatch.setattr(service_host.shutil, "which", lambda name: None)
    service_host.open_url("http://127.0.0.1:8765")
    assert "http://127.0.0.1:8765" in capsys.readouterr().err


def test_wsl_browser_prefers_windows_browser(monkeypatch):
    calls = []
    monkeypatch.setattr(service_host.sys, "platform", "linux")
    monkeypatch.setattr(service_host.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        service_host.subprocess, "run", lambda argv, **kwargs: calls.append(argv)
    )
    service_host.open_url("http://127.0.0.1:8765")
    assert calls == [["/usr/bin/wslview", "http://127.0.0.1:8765"]]


def test_real_service_start_restart_stop(tmp_path, monkeypatch):
    if sys.platform == "linux":
        ready = subprocess.run(
            ["systemctl", "--user", "show-environment"], capture_output=True
        )
        if ready.returncode:
            pytest.skip("systemd user manager is not available")
    monkeypatch.setattr(app, "tailscale_ip", lambda: None)
    tmp_path = tmp_path / "workspace with spaces $HOME %n 中文"
    tmp_path.mkdir()
    root, label = app.service_paths(tmp_path)
    try:
        url = app.start(tmp_path, open_browser=False)
        first = app.health_at(url, tmp_path)
        assert first and first["pid"] != os.getpid()
        assert app.start(tmp_path, open_browser=False) == url
        assert app.health_at(url, tmp_path)["pid"] == first["pid"]
        assert service_host.stop_service(root, label, first)
        assert app.health_at(url, tmp_path) is None
        url = app.start(tmp_path, open_browser=False)
        assert app.health_at(url, tmp_path)["pid"] != first["pid"]
    finally:
        service_host.stop_service(root, label)
