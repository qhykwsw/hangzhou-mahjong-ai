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
from dataclasses import asdict
from pathlib import Path

from .storage import atomic_json, digest, file_hash, read_json
from .types import RoundResult


def tree_hash(root):
    root = Path(root)
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('candidate packages may not contain symlinks')
        if p.is_file() and '__pycache__' not in p.parts:
            files[str(p.relative_to(root))] = file_hash(p)
    return digest(files)


def sandbox_command(argv, readable, writable):
    """macOS seatbelt is an enforced allowlist, not a prompt-based restriction."""
    if sys.platform != 'darwin' or not shutil.which('sandbox-exec'):
        raise RuntimeError('isolated candidate runner requires macOS sandbox-exec; no unsafe fallback')
    def sub(path):
        return '(subpath ' + json.dumps(str(Path(path).resolve())) + ')'
    base = ['/System', '/usr', '/bin', '/sbin', '/Library', '/private/var/db', '/dev',
            sys.base_prefix, sys.prefix, Path(__file__).parent.parent]
    profile = '(version 1)(allow default)(deny network*)(deny file-read*)(deny file-write*)(deny process-info*)'
    profile += '(allow process-info* (target self))'
    profile += '(allow file-read-metadata)(allow file-read* (literal "/") ' + ' '.join(sub(p) for p in base + list(readable)) + ')'
    profile += '(allow file-read* file-write* (literal "/dev/null") (literal "/dev/urandom") ' + ' '.join(sub(p) for p in writable) + ')'
    # No network, no host home, no experiment files, no inherited secrets.
    return ['/usr/bin/sandbox-exec', '-p', profile, *argv]


def public_result(result):
    return RoundResult(result.engine, 0, result.stage, result.rounds_played,
                       result.total_score_by_seat, result.winner_seat,
                       result.top_two_seats, result.completed,
                       {'objective': result.metadata.get('objective'), 'reason': result.metadata.get('reason', '')},
                       result.rank_by_seat)


class CandidateProcess:
    def __init__(self, package, python=None, output=None):
        self.package = Path(package).resolve()
        self.temp = tempfile.TemporaryDirectory(prefix='mahjong-candidate-')
        self.output = Path(output).resolve() if output else Path(self.temp.name) / 'output'
        self.output.mkdir(parents=True, exist_ok=True)
        runtime = str(Path(__file__).parent.parent)
        argv = [python or sys.executable, '-B', '-m', 'mahjong_ai.candidate_worker', str(self.package), str(self.output)]
        readable = [self.package]
        if python: readable.append(Path(python).parent.parent)
        env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': runtime, 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
               'TMPDIR': self.temp.name, 'HOME': self.temp.name}
        self.process = subprocess.Popen(sandbox_command(argv, readable, [self.temp.name, self.output]),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        cwd=self.package, env=env, start_new_session=True, bufsize=0)
        self.pending = bytearray()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def send(self, data):
        self.process.stdin.write((json.dumps(data, allow_nan=False) + '\n').encode())
        self.process.stdin.flush()

    def receive(self, seconds):
        end = time.monotonic() + max(.001, seconds)
        while b'\n' not in self.pending:
            if not self.selector.select(max(0, end - time.monotonic())):
                raise TimeoutError('candidate response deadline exceeded')
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data: raise RuntimeError('candidate worker exited')
            self.pending.extend(data)
            if len(self.pending) > 8_000_000: raise ValueError('candidate response too large')
        line, _, rest = self.pending.partition(b'\n')
        self.pending = bytearray(rest)
        value = json.loads(line)
        if 'error' in value: raise RuntimeError(value['error'])
        return value

    def call(self, data, seconds=10):
        self.send(data)
        return self.receive(seconds)

    def close(self):
        if getattr(self, 'process', None):
            if self.process.poll() is None:
                try: os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            self.process.wait()
            for pipe in (self.process.stdin, self.process.stdout): pipe.close()
            self.selector.close()
            self.process = None
        self.temp.cleanup()

    def __del__(self):
        try: self.close()
        except Exception: pass


class RemotePolicy:
    """Never imports candidate code in the engine process; validates action indices."""
    name = 'code-policy'
    def __init__(self, package, package_hash, action_seconds=.05, python=None, version='code'):
        if tree_hash(package) != package_hash: raise ValueError('frozen candidate changed')
        self.package, self.package_hash, self.action_seconds = package, package_hash, action_seconds
        self.python, self.version, self.worker = python, version, None
        self.stats = {'actions': 0, 'timeouts': 0, 'errors': 0, 'elapsed_seconds': 0.}

    def _start(self):
        self.worker = CandidateProcess(self.package, self.python)
        self.worker.call({'op': 'build'})

    def reset(self, seed):
        try:
            if self.worker is None: self._start()
            self.worker.call({'op': 'reset', 'seed': seed})
        except Exception:
            self.stats['errors'] += 1
            self.close()

    def act(self, observation):
        started = time.monotonic()
        self.stats['actions'] += 1
        try:
            if self.worker is None:return observation.legal_actions[0]
            request={'op':'act','observation':asdict(observation)}
            reply = self.worker.call(request, max(.001,self.action_seconds-(time.monotonic()-started)))
            index = reply['index']
            if type(index) is not int or not 0 <= index < len(observation.legal_actions):
                raise ValueError('invalid action index')
            return observation.legal_actions[index]
        except TimeoutError:
            self.stats['timeouts'] += 1
            self.close()
        except Exception:
            self.stats['errors'] += 1
            self.close()
        finally:
            self.stats['elapsed_seconds'] += time.monotonic() - started
        # Stable, cheap and identical across development and private evaluation.
        return observation.legal_actions[0]

    def finish_round(self, result, seat):
        if self.worker:
            try: self.worker.call({'op': 'finish', 'result': asdict(public_result(result)), 'seat': seat})
            except Exception:
                self.stats['errors'] += 1
                self.close()

    def close(self):
        if self.worker:
            self.worker.close()
            self.worker = None

    def __del__(self):
        try: self.close()
        except Exception: pass


def train_package(package, output, config, opponents, episodes, seed, deadline, python=None):
    """Candidate owns train(ctx); ctx.play(policy) delegates real games to this host."""
    from .engine import MahjongEngine, derive_seed
    worker = CandidateProcess(package, python, output)
    records = []
    progress=Path(output).parent/'training-checkpoint.json'
    checkpoint=None
    if progress.exists():
        checkpoint=read_json(progress)
        if checkpoint['hash']!=tree_hash(output):
            # Candidate may have been interrupted midway through a newer save.
            # Restore the last atomically committed model directory.
            shutil.rmtree(output)
            committed=Path(output).parent/checkpoint['directory']
            if tree_hash(committed)!=checkpoint['hash']:raise ValueError('training checkpoint corrupted')
            shutil.copytree(committed,output)
        records=checkpoint['episodes']
    class Bridge:
        name, version = 'training', 'code'
        def reset(self, seed): worker.call({'op': 'reset', 'seed': seed}, max(.001, deadline-time.monotonic()))
        def act(self, obs):
            reply = worker.call({'op': 'act', 'observation': asdict(obs)}, max(.001, deadline-time.monotonic()))
            index = reply['index']
            if type(index) is not int or not 0 <= index < len(obs.legal_actions): raise ValueError('illegal training action')
            return obs.legal_actions[index]
        def finish_round(self, result, seat):
            worker.call({'op': 'finish', 'result': asdict(public_result(result)), 'seat': seat}, max(.001, deadline-time.monotonic()))
    try:
        worker.send({'op': 'train', 'episodes': episodes, 'config': config.as_dict(),
                     'completed_episodes':len(records),'checkpoint':checkpoint.get('metadata') if checkpoint else None})
        while True:
            reply = worker.receive(max(.001, deadline-time.monotonic()))
            if reply.get('request')=='checkpoint':
                fingerprint=tree_hash(output)
                committed=Path(output).parent/('training-committed-'+fingerprint)
                staging=Path(output).parent/'training-committing'
                if staging.exists():shutil.rmtree(staging)
                if not committed.exists():
                    shutil.copytree(output,staging);staging.rename(committed)
                atomic_json(progress,{'episodes':records,'hash':fingerprint,'metadata':reply.get('metadata'),'directory':committed.name})
                for previous in Path(output).parent.glob('training-committed-*'):
                    if previous!=committed:shutil.rmtree(previous)
                worker.send({'ok':True})
                continue
            if reply.get('done'):
                if not records or not any(Path(output).iterdir()): raise ValueError('RL training produced no games or model artifacts')
                return {'status': 'completed', 'episodes': records, 'reported_metrics': reply.get('metrics', {})}
            if reply.get('request') != 'play' or len(records) >= episodes:
                raise ValueError('training episode allowance exceeded or invalid request')
            i = len(records)
            seat = i % 4
            players = [p.build() for p in opponents]
            players.insert(seat, Bridge())
            try: result = MahjongEngine().run_round(players, config, derive_seed(seed, 'training', i))
            finally:
                for p in players:
                    if hasattr(p, 'close'): p.close()
            if not result.completed: raise ValueError('incomplete training settlement')
            records.append({'episode': i, 'rank': result.rank_by_seat[seat], 'rank_utility': result.rank_utility(seat),
                            'hands': result.rounds_played, 'score': result.score(seat)})
            worker.send({'result': asdict(public_result(result)), 'seat': seat})
    finally: worker.close()
