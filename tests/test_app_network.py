import subprocess
import pytest
from types import SimpleNamespace

from mahjong_ai.interfaces import app


def test_two_interfaces_share_one_port():
    remote_ip = app.tailscale_ip()
    if remote_ip is None:
        pytest.skip("Tailscale is not connected")
    sockets = app.listener_sockets(0, remote_ip)
    try:
        addresses = [sock.getsockname() for sock in sockets]
        assert [address[0] for address in addresses] == ["127.0.0.1", remote_ip]
        assert addresses[0][1] == addresses[1][1] != 0
    finally:
        for sock in sockets:
            sock.close()


def test_tailscale_discovery_handles_offline_or_invalid_output(monkeypatch):
    monkeypatch.setattr(app.shutil, "which", lambda name: "/usr/local/bin/tailscale")
    for output, expected in [
        ("100.64.0.1\n", "100.64.0.1"),
        ("192.168.1.1", None),
        ("offline", None),
    ]:
        monkeypatch.setattr(
            app.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout=output),
        )
        assert app.tailscale_ip() == expected

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("tailscale", 3)

    monkeypatch.setattr(app.subprocess, "run", timeout)
    assert app.tailscale_ip() is None


def test_ssh_start_prints_both_urls_without_opening_browser(
    tmp_path, monkeypatch, capsys
):
    calls = []
    monkeypatch.setenv("SSH_CONNECTION", "100.64.0.2 12345 100.64.0.1 22")
    monkeypatch.setattr(
        app,
        "start",
        lambda workspace, open_browser: (
            calls.append(open_browser) or "http://127.0.0.1:8765"
        ),
    )
    monkeypatch.setattr(
        app, "health_at", lambda *args: {"tailscale_url": "http://100.64.0.1:8765"}
    )
    app.main(["--workspace", str(tmp_path), "start"])
    assert calls == [False]
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8765" in output and "http://100.64.0.1:8765" in output
