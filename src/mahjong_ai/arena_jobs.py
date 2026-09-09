"""One admitted compute job. All strategy code lives in isolated child processes."""
import os
from pathlib import Path
import random
import resource
import shutil
import signal
import sys
import threading
import time

from .candidate_runtime import CandidateProcess, JobCancelled, check_cancelled, train_package, tree_hash
from .dependencies import prepare_environment
from .engine import MahjongEngine, derive_seed
from .policy import PolicySpec
from .storage import atomic_json, digest, read_json, runtime_fingerprint
from .types import RoundConfig


def prepare(task, root, deadline):
    rows = {}
    for cid, item in task['packages'].items():
        check_cancelled()
        out = root / cid
        out.mkdir(exist_ok=True)
        artifact = out / 'artifact.json'
        if artifact.exists():
            row = read_json(artifact)
            PolicySpec(**row['spec']).identity
            rows[cid] = row
            continue
        try:
            source = Path(item['input'])
            if tree_hash(source) != item['hash']:
                raise ValueError('submitted artifact changed')
            python, wheels = prepare_environment(item['requirements'], out / 'environment', deadline)
            probe = CandidateProcess(source, python)
            try:
                capabilities = probe.call({'op': 'describe'}, min(10, max(.01, deadline-time.monotonic())))
            finally:
                probe.close()
            model = out / 'training-model'
            training = None
            if capabilities['train']:
                def opponents(index):
                    rng = random.Random(derive_seed(task['seed'], 'training-opponents', cid, index))
                    return [PolicySpec(**p) for p in rng.sample(task['opponents'], 3)]
                training = train_package(source, model, RoundConfig(**task['config']), opponents,
                                         task['training_games'], derive_seed(task['seed'], 'training', cid),
                                         deadline, python, policy_seed=derive_seed(task['seed'], 'policy', cid))
            package = out / 'frozen'
            if package.exists():
                shutil.rmtree(package)
            shutil.copytree(source, package)
            if training is not None:
                if (package / 'model').exists():
                    shutil.rmtree(package / 'model')
                shutil.copytree(model, package / 'model')
            if wheels:
                atomic_json(package / 'dependencies.json', wheels)
            probe = CandidateProcess(package, python)
            try:
                probe.call({'op': 'build'}, min(10, max(.01, deadline-time.monotonic())))
            finally:
                probe.close()
            spec = PolicySpec(family='code', version=cid, package_path=str(package), package_hash=tree_hash(package),
                              python_path=python, action_seconds=task['timeouts']['discard'],
                              response_seconds=task['timeouts']['response']).as_dict()
            rows[cid] = dict(status='ready', spec=spec, training=training)
            atomic_json(artifact, rows[cid])
        except JobCancelled:
            raise
        except Exception as exc:
            rows[cid] = dict(status='failed', error=f'{type(exc).__name__}: {exc}'[:4000])
        atomic_json(root / 'progress.json', {'prepared': len(rows), 'total': len(task['packages'])})
    return {'status': 'completed', 'candidates': rows}


def play(task, root):
    players = [PolicySpec(**p).build() for p in task['policies']]
    engine = MahjongEngine(record_events=False)
    engine.progress = lambda progress: atomic_json(root / 'progress.json', progress)
    try:
        result = engine.run_round(players, RoundConfig(**task['config']), task['seed'],
                                  policy_seeds=[derive_seed(task['policy_seed'], cid) for cid in task['players']])
        if not result.completed:
            raise ValueError('incomplete settlement: ' + result.metadata['reason'])
        return {'status': 'completed', 'players': task['players'], 'ranks': result.rank_by_seat,
                'scores': result.total_score_by_seat, 'hands': result.rounds_played,
                'execution': [dict(p.stats) for p in players],
                'diagnostics': [{'candidate_id': cid, 'error': p.last_error} for cid, p in zip(task['players'], players) if p.last_error],
                'task_hash': digest(task)}
    finally:
        for player in players:
            player.close()


def main():
    root = Path(sys.argv[1])
    task = read_json(root / 'task.json')
    if task['runtime'] != runtime_fingerprint():
        raise ValueError('job requires its frozen runtime')
    cancel = root / 'cancel.flag'
    os.environ['MAHJONG_JOB_CANCEL'] = str(cancel)
    def stop(signum, frame):
        cancel.touch()
        raise JobCancelled('operator or budget interrupted job')
    signal.signal(signal.SIGTERM, stop)
    def watch_owner():
        while True:
            time.sleep(.2)
            if os.getppid() != task['owner_pid'] or time.time() >= task['deadline']:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    threading.Thread(target=watch_owner, daemon=True).start()
    started = time.monotonic()
    deadline = started + max(0, task['deadline']-time.time())
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        check_cancelled()
        value = prepare(task, root, deadline) if task['kind'] == 'prepare' else play(task, root)
    except JobCancelled:
        value = {'status': 'paused', 'reason': 'operator or budget interruption'}
    except Exception as exc:
        value = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:4000]}
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    value['usage'] = dict(wall_seconds=time.monotonic()-started, host_cpu_seconds=time.process_time(),
                          child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    atomic_json(root / 'result.json', value)


if __name__ == '__main__':
    main()
