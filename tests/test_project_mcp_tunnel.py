"""Verify configuration and supervision without touching the user's live services."""
import importlib.util
import os
from pathlib import Path
import plistlib
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('project_mcp_tunnel', Path(__file__).resolve().parents[1] / 'scripts/project_mcp_tunnel.py')
tunnel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tunnel)


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(tunnel, 'ROOT', tmp_path)
    monkeypatch.setattr(tunnel, 'LOCAL', tmp_path / '.local/project_mcp')
    monkeypatch.setattr(tunnel, 'PROFILE', tunnel.LOCAL / 'profiles/leihuo-majiang.yaml')
    monkeypatch.setattr(tunnel, 'PLIST', tmp_path / 'LaunchAgents/test.plist')
    key = tmp_path / 'runtime.key'
    key.write_text('FAKE_KEY_NOT_A_CREDENTIAL')
    key.chmod(0o600)
    client = tmp_path / 'tunnel-client'
    client.write_text('#!/bin/sh\nexit 0\n')
    client.chmod(0o700)
    return SimpleNamespace(tunnel_id='tunnel_fixture', client=client, key_file=key)


def test_configure_keeps_key_reference_and_existing_profile(local, monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        tunnel.PROFILE.parent.mkdir(parents=True)
        tunnel.PROFILE.write_text('fixture profile')
    monkeypatch.setattr(tunnel.subprocess, 'run', run)
    monkeypatch.setenv('OPENAI_API_KEY', 'FAKE_AMBIENT')
    tunnel.configure(local)
    cmd, kwargs = calls[0]
    assert 'FAKE_KEY_NOT_A_CREDENTIAL' not in str(cmd)
    assert 'OPENAI_API_KEY' not in kwargs['env']
    assert cmd[cmd.index('--health-listen-addr') + 1] == '127.0.0.1:8789'
    assert f'file:{local.key_file}' in cmd
    assert tunnel.command('run')[-2:] == ['--log.level', 'warn']
    assert tunnel.PROFILE.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match='already exists'):
        tunnel.configure(local)


def test_insecure_key_rejected_before_configuration(local):
    local.key_file.chmod(0o644)
    with pytest.raises(ValueError, match='private file'):
        tunnel.configure(local)
    assert not tunnel.LOCAL.exists()


def test_launchd_lifecycle(local, monkeypatch):
    tunnel.LOCAL.mkdir(parents=True)
    tunnel.PROFILE.parent.mkdir()
    tunnel.PROFILE.write_text('fixture')
    (tunnel.LOCAL / 'key-path.txt').write_text(str(local.key_file))
    calls = []
    monkeypatch.setattr(tunnel.subprocess, 'run', lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(tunnel.sys, 'platform', 'darwin')
    monkeypatch.setattr(tunnel.sys, 'argv', ['project_mcp_tunnel.py', 'start'])
    tunnel.main()
    payload = plistlib.loads(tunnel.PLIST.read_bytes())
    assert payload['KeepAlive'] is True and payload['RunAtLoad'] is True
    assert payload['Label'] == 'com.netease.leihuo-majiang.project-mcp'
    assert payload['ProgramArguments'][-1] == 'run'
    assert 'FAKE_KEY_NOT_A_CREDENTIAL' not in str(payload)
    assert calls[0][:2] == ['launchctl', 'bootstrap']
    monkeypatch.setattr(tunnel.sys, 'argv', ['project_mcp_tunnel.py', 'stop'])
    tunnel.main()
    assert not tunnel.PLIST.exists()
    assert calls[1] == ['launchctl', 'bootout', f'gui/{os.getuid()}/{tunnel.LABEL}']
