#!/usr/bin/env python3
"""Configure the private tunnel and manage its independent macOS LaunchAgent."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local/project_mcp'
PROFILE = LOCAL / 'profiles/leihuo-majiang.yaml'
LABEL = 'com.netease.leihuo-majiang.project-mcp'
PLIST = Path.home() / 'Library/LaunchAgents' / f'{LABEL}.plist'
HEALTH = 'http://127.0.0.1:8789/readyz'


def environment():
    return {k: v for k, v in os.environ.items() if k not in {'OPENAI_API_KEY', 'OPENAI_ADMIN_KEY', 'CONTROL_PLANE_API_KEY'}}


def configure(args):
    if not re.fullmatch(r'tunnel_[A-Za-z0-9]+', args.tunnel_id):
        raise ValueError('Invalid tunnel ID')
    client = args.client.expanduser().resolve(strict=True)
    key = args.key_file.expanduser().resolve(strict=True)
    if not key.is_file() or key.stat().st_mode & 0o077 or not key.stat().st_size:
        raise ValueError('The runtime key must be a nonempty private file (chmod 600)')
    if PROFILE.exists():
        raise ValueError('Profile already exists; inspect it before replacing the local configuration')
    LOCAL.mkdir(parents=True, exist_ok=True, mode=0o700)
    LOCAL.chmod(0o700)
    (LOCAL / 'bin').mkdir(exist_ok=True)
    target = LOCAL / 'bin/tunnel-client'
    if client != target:
        shutil.copy2(client, target)
    target.chmod(0o700)
    # Store only a secret reference. The key remains at the operator-selected path.
    subprocess.run([
        str(target), 'init', '--sample', 'sample_mcp_stdio_local',
        '--profile', 'leihuo-majiang', '--profile-dir', str(LOCAL / 'profiles'),
        '--tunnel-id', args.tunnel_id, '--control-plane-api-key-ref', f'file:{key}',
        '--mcp-command', str(ROOT / 'scripts/run_project_mcp.sh'),
        '--health-listen-addr', '127.0.0.1:8789',
    ], env=environment(), check=True, stdout=subprocess.DEVNULL)
    PROFILE.chmod(0o600)
    (LOCAL / 'key-path.txt').write_text(str(key))
    (LOCAL / 'key-path.txt').chmod(0o600)
    print('Tunnel configured; run doctor, then start.')


def command(action):
    if not PROFILE.is_file() or not (LOCAL / 'key-path.txt').is_file():
        raise ValueError('Run configure first; see docs/reference/project-mcp.md')
    key = Path((LOCAL / 'key-path.txt').read_text())
    if not key.is_file() or key.stat().st_mode & 0o077:
        raise ValueError('Runtime key missing or not private')
    return [str(LOCAL / 'bin/tunnel-client'), action, '--profile-file', str(PROFILE),
            '--control-plane.api-key', f'file:{key}', '--log.level', 'warn']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    config = sub.add_parser('configure')
    config.add_argument('--tunnel-id', required=True)
    config.add_argument('--client', type=Path, required=True)
    config.add_argument('--key-file', type=Path, required=True)
    for action in ('run', 'doctor', 'start', 'stop', 'status'):
        sub.add_parser(action)
    args = parser.parse_args()
    try:
        if args.action == 'configure':
            configure(args)
        elif args.action in {'run', 'doctor'}:
            cmd = command(args.action)
            os.execve(cmd[0], cmd, environment())
        elif args.action == 'status':
            try:
                with urlopen(HEALTH, timeout=3) as response:
                    health = response.read(1024).decode()
            except (OSError, URLError):
                health = 'unavailable'
            print(json.dumps({'configured': PROFILE.is_file(), 'launch_agent_installed': PLIST.is_file(), 'readyz': health}))
        else:
            if sys.platform != 'darwin':
                raise ValueError('start/stop use macOS launchd; on Linux supervise the run command with systemd')
            domain = f'gui/{os.getuid()}'
            if args.action == 'stop':
                result = subprocess.run(['launchctl', 'bootout', f'{domain}/{LABEL}'], capture_output=True)
                if result.returncode:
                    raise ValueError('LaunchAgent is not loaded or could not be stopped')
                PLIST.unlink(missing_ok=True)
                print('Tunnel stopped and login startup removed')
                return
            command('run')  # validate before registering the service
            if PLIST.exists():
                raise ValueError('LaunchAgent already installed; stop before replacing it')
            logs = LOCAL / 'logs'
            logs.mkdir(exist_ok=True, mode=0o700)
            PLIST.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'Label': LABEL,
                'ProgramArguments': [str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()), 'run'],
                'WorkingDirectory': str(ROOT), 'RunAtLoad': True, 'KeepAlive': True,
                'ThrottleInterval': 10, 'Umask': 0o077,
                'StandardOutPath': str(logs / 'tunnel.out.log'),
                'StandardErrorPath': str(logs / 'tunnel.err.log'),
            }
            PLIST.write_bytes(plistlib.dumps(payload))
            PLIST.chmod(0o600)
            subprocess.run(['launchctl', 'bootstrap', domain, str(PLIST)], check=True, capture_output=True)
            print('Tunnel registered with launchd; use status to verify readiness')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        # Never expose subprocess diagnostics or secret material.
        parser.exit(1, f'{exc if isinstance(exc, ValueError) else "Local tunnel operation failed; inspect configuration and service status"}\n')


if __name__ == '__main__':
    main()
