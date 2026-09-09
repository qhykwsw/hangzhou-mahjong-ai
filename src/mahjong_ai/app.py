"""Local dashboard and one launchd-owned service per project."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import plistlib
import socket
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from .arena import Arena, catalog
from .arena_controller import Controller
from .storage import atomic_json, exclusive_lock, read_json


def application(project, tailscale_url=None):
    arena = Arena(project)
    @asynccontextmanager
    async def lifespan(app):
        controller = Controller(arena)
        stopped = asyncio.Event()
        async def loop():
            while not stopped.is_set():
                try:
                    await asyncio.to_thread(controller.tick)
                except Exception:
                    traceback.print_exc()
                    for s in arena.runs():
                        if s['status'] == 'running':
                            arena.control(s['id'], 'pause')
                await asyncio.sleep(.25)
        worker = asyncio.create_task(loop())
        try:
            yield
        finally:
            stopped.set()
            await worker
            await asyncio.to_thread(controller.close)
    async def home(request):
        return FileResponse(Path(__file__).parent / 'web/index.html')
    async def health(request):
        return JSONResponse({'project': str(arena.project), 'pid': os.getpid(), 'tailscale_url': tailscale_url})
    async def state(request):
        return JSONResponse(await asyncio.to_thread(arena.snapshot, request.query_params.get('run')))
    async def activity(request):
        try:
            return JSONResponse(await asyncio.to_thread(arena.activity, request.query_params['run']))
        except (KeyError, ValueError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
    async def directions(request):
        return JSONResponse(list(catalog(arena.project).values()))
    async def mutate(request):
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'error': 'local same-origin request required'}, status_code=403)
        try:
            value = await request.json()
            if request.url.path == '/api/runs':
                identity = await asyncio.to_thread(arena.create, value.get('directions'), value.get('seconds'))
                result = await asyncio.to_thread(arena.control, identity, 'resume')
            else:
                result = await asyncio.to_thread(arena.control, value['run'], value['action'], value.get('seconds', 0))
            return JSONResponse(result)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
    app = Starlette(lifespan=lifespan, routes=[Route('/', home), Route('/api/health', health),
        Route('/api/state', state), Route('/api/activity', activity), Route('/api/directions', directions),
        Route('/api/runs', mutate, methods=['POST']), Route('/api/control', mutate, methods=['POST'])])
    app.mount('/assets', StaticFiles(directory=Path(__file__).parent / 'web'), name='assets')
    return app


def service_paths(project):
    root = Path(project).resolve() / '.local/service'
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    label = 'com.mahjong.' + hashlib.sha256(str(Path(project).resolve()).encode()).hexdigest()[:16]
    return root, label


def health_at(url, project):
    try:
        with urllib.request.urlopen(url + '/api/health', timeout=1) as response:
            state = json.load(response)
        return state if state['project'] == str(Path(project).resolve()) else None
    except (OSError, ValueError, KeyError):
        return None


def tailscale_ip():
    executable = shutil.which('tailscale')
    if not executable:
        bundled = Path('/Applications/Tailscale.app/Contents/MacOS/Tailscale')
        executable = str(bundled) if bundled.exists() else None
    if not executable:
        return None
    try:
        result = subprocess.run([executable, 'ip', '-4'], capture_output=True, text=True, timeout=3)
        if result.returncode != 0:
            return None
        address = ipaddress.IPv4Address(result.stdout.strip())
        return str(address) if address in ipaddress.IPv4Network('100.64.0.0/10') else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def listener_sockets(port, remote_ip=None):
    """Listen only on loopback and the connected tailnet interface."""
    sockets = []
    try:
        for host in ['127.0.0.1', *([remote_ip] if remote_ip else [])]:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            # Port zero must select the same port on both interfaces.
            port = sock.getsockname()[1]
            sock.listen(128)
        return sockets
    except BaseException:
        for sock in sockets:
            sock.close()
        raise


def start(project, open_browser=True):
    root, label = service_paths(project)
    with exclusive_lock(root / 'start.lock', blocking=True):
        record = root / 'service.json'
        if record.exists():
            url = read_json(record)['url']
            if health_at(url, project):
                if open_browser:
                    subprocess.run(['open', url], check=True)
                return url
        arena = Arena(project)
        if not arena.runs():
            # Controller recovery happens first; activate through HTTP after readiness.
            initial = arena.create()
        else:
            initial = None
        remote_ip = tailscale_ip()
        try:
            probes = listener_sockets(8765, remote_ip)
        except OSError:
            probes = listener_sockets(0, remote_ip)
        port = probes[0].getsockname()[1]
        for sock in probes:
            sock.close()
        url = f'http://127.0.0.1:{port}'
        env = {key: os.environ[key] for key in ('HOME', 'PATH', 'CODEX_HOME') if key in os.environ}
        env['PYTHONPATH'] = str(Path(__file__).parent.parent)
        # Custom providers sometimes use an environment-backed credential.
        import tomllib
        config_path = Path(env.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'config.toml'
        config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
        provider = config.get('model_providers', {}).get(config.get('model_provider'), {})
        key = provider.get('env_key')
        if key and key in os.environ:
            env[key] = os.environ[key]
        plist = root / 'service.plist'
        payload = dict(Label=label, ProgramArguments=[sys.executable, '-B', '-m', 'mahjong_ai.app',
                       '--project', str(Path(project).resolve()), 'serve', '--port', str(port),
                       *(['--tailscale-ip', remote_ip] if remote_ip else [])],
                       WorkingDirectory=str(Path(project).resolve()), EnvironmentVariables=env,
                       RunAtLoad=True, KeepAlive=False, ExitTimeOut=15,
                       StandardOutPath=str(root / 'service.log'), StandardErrorPath=str(root / 'service.log'))
        plist.write_bytes(plistlib.dumps(payload))
        plist.chmod(0o600)
        domain = f'gui/{os.getuid()}'
        subprocess.run(['launchctl', 'bootout', domain + '/' + label], capture_output=True)
        # bootout releases its registration asynchronously on macOS.
        for attempt in range(50):
            launched = subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], capture_output=True, text=True)
            if launched.returncode == 0:
                break
            if launched.returncode != 5 or attempt == 49:
                raise RuntimeError('launchd could not start service: ' + launched.stderr.strip())
            time.sleep(.1)
        atomic_json(record, {'url': url, 'label': label})
        for _ in range(100):
            if health_at(url, project):
                break
            time.sleep(.1)
        else:
            raise RuntimeError(f'service did not become ready; see {root / "service.log"}')
        if initial:
            req = urllib.request.Request(url + '/api/control',
                json.dumps({'run': initial, 'action': 'resume'}).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10):
                pass
        if open_browser:
            subprocess.run(['open', url], check=True)
        return url


def main(argv=None):
    parser = argparse.ArgumentParser(description='本地麻将研究控制台')
    parser.add_argument('--project', default=os.getcwd())
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('start')
    p.add_argument('--no-open', action='store_true')
    sub.add_parser('status')
    sub.add_parser('stop')
    p = sub.add_parser('serve')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--tailscale-ip', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root, label = service_paths(args.project)
    if args.command == 'serve':
        import uvicorn
        with exclusive_lock(root / 'server.lock'):
            remote_ip = args.tailscale_ip or tailscale_ip()
            sockets = listener_sockets(args.port, remote_ip)
            remote_url = f'http://{remote_ip}:{args.port}' if remote_ip else None
            try:
                server = uvicorn.Server(uvicorn.Config(application(args.project, remote_url), log_level='warning'))
                server.run(sockets=sockets)
            finally:
                for sock in sockets:
                    sock.close()
    elif args.command == 'start':
        through_ssh = bool(os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'))
        url = start(args.project, not args.no_open and not through_ssh)
        print(url)
        running = health_at(url, args.project) or {}
        if running.get('tailscale_url'):
            print('Tailscale: ' + running['tailscale_url'])
    elif args.command == 'stop':
        record = root / 'service.json'
        value = read_json(record) if record.exists() else {}
        previous = health_at(value['url'], args.project) if value else None
        result = subprocess.run(['launchctl', 'bootout', f'gui/{os.getuid()}/{label}'], capture_output=True)
        if result.returncode == 0:
            for _ in range(100):
                present = subprocess.run(['launchctl', 'print', f'gui/{os.getuid()}/{label}'], capture_output=True)
                alive = subprocess.run(['/bin/ps', '-p', str(previous['pid']), '-o', 'command='],
                                       capture_output=True, text=True).stdout if previous else ''
                if present.returncode != 0 and 'mahjong_ai.app' not in alive:
                    break
                time.sleep(.1)
        print('服务已停止' if result.returncode == 0 else '服务未运行')
    else:
        record = root / 'service.json'
        value = read_json(record) if record.exists() else {}
        running = health_at(value['url'], args.project) if value else None
        print(json.dumps({'running': bool(running), **value, 'tailscale_url': (running or {}).get('tailscale_url')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
