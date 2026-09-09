"""Owns all processes, resource admission, cancellation and durable result ingestion."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .arena import Arena
from .storage import atomic_json, digest, read_json
from .engine import derive_seed


class Controller:
    def __init__(self, arena):
        self.arena = arena
        self.processes = {}
        self.caffeine = None
        self.recover()

    def recover(self):
        for stored in self.arena.runs():
            with self.arena.edit(stored['id']) as (s, db):
                self.consume(s, db)
                for job in s['jobs'].values():
                    if job['status'] in ('running', 'admitted'):
                        self.stop(s, job, force=True)
                        job['status'] = 'completed' if job['kind'] == 'agent' else 'paused'
                for direction in s['directions'].values():
                    if direction['status'] == 'researching':
                        direction['status'] = 'ready'
                if s['status'] in ('running', 'pausing'):
                    s.update(status='paused', pause_reason='service restarted; resume to continue')

    def stop(self, s, job, force=False):
        root = Path(s['root']) / 'jobs' / job['id']
        root.mkdir(parents=True, exist_ok=True)
        (root / 'cancel.flag').touch()
        process_record = root / 'process.json'
        pid = job.get('pid') or (read_json(process_record)['pid'] if process_record.exists() else None)
        if not pid:
            return
        if force and job['kind'] == 'agent':
            child_record = root / 'codex-process.json'
            if child_record.exists():
                child = read_json(child_record)['pid']
                command = subprocess.run(['/bin/ps', '-p', str(child), '-o', 'command='], capture_output=True, text=True).stdout
                if 'codex' in command and s['id'] in command:
                    try:
                        os.killpg(child, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        process = self.processes.get((s['id'], job['id']))
        if process is not None and process.poll() is not None:
            return
        command = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
        if str(root) not in command or 'mahjong_ai.' not in command:
            return
        try:
            os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass
        job.setdefault('stop_requested_at', time.time())

    def consume(self, s, db):
        for job in list(s['jobs'].values()):
            if job['status'] in ('completed', 'failed'):
                continue
            root = Path(s['root']) / 'jobs' / job['id']
            path = root / 'result.json'
            if not path.exists():
                continue
            result = read_json(path)
            if (result['status'] == 'paused' and job['kind'] == 'prepare' and s['status'] == 'running'
                    and time.time() >= read_json(root / 'task.json')['deadline']):
                result = {**result, 'status': 'failed', 'error': '训练准备达到单次时间上限；检查点已保留。'}
            fingerprint = digest(result)
            if job.get('result_hash') == fingerprint:
                continue
            job.update(result_hash=fingerprint, status=result['status'], usage=result.get('usage', {}))
            job.pop('stop_requested_at', None)
            process = self.processes.pop((s['id'], job['id']), None)
            if process is not None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.processes[(s['id'], job['id'])] = process
            if result.get('error'):
                job['error'] = result['error']
            if job['kind'] == 'agent':
                direction = s['directions'][job['direction']]
                if result.get('thread_id'):
                    direction['thread_id'] = result['thread_id']
                waiting = any(b['direction'] == job['direction'] and b['status'] not in ('completed', 'failed') for b in s['batches'].values())
                progressed = sum(b['direction'] == job['direction'] for b in s['batches'].values()) > job['batches_before']
                direction['failures'] = 0 if progressed else direction['failures'] + (0 if result.get('interrupted') else 1)
                direction['status'] = 'waiting' if waiting else 'error' if direction['failures'] >= 3 else 'ready'
                if result.get('error'):
                    direction['last_error'] = result['error']
                elif not progressed:
                    direction['last_error'] = '本次会话未提交候选；已保留草稿和会话。'
                job['status'] = 'completed'
                self.arena.event(db, s, 'agent_turn', {'direction': job['direction'], 'usage': result.get('usage', {}),
                                                      'usage_complete': result.get('usage_complete', False),
                                                      'error': result.get('error')})
            elif job['kind'] == 'prepare' and result['status'] in ('completed', 'failed'):
                batch = s['batches'][job['batch_id']]
                identities = {c['spec']['package_hash']: c['id'] for c in s['candidates'].values() if c.get('spec')}
                ready = 0
                for cid in batch['candidates']:
                    c = s['candidates'][cid]
                    row = result.get('candidates', {}).get(cid, {'status': 'failed', 'error': result.get('error', 'preparation incomplete')})
                    c['status'] = row['status']
                    if row.get('spec'):
                        identity = row['spec']['package_hash']
                        c['spec'], c['training'] = row['spec'], row.get('training')
                        if identity in identities:
                            c['status'] = 'duplicate'
                            c['duplicate_of'] = identities[identity]
                        else:
                            identities[identity] = cid
                            ready += 1
                    if row.get('error'):
                        c['error'] = row['error']
                batch['status'] = 'ready' if ready else 'completed'
                if not ready:
                    s['directions'][batch['direction']]['status'] = 'ready'
                self.arena.event(db, s, 'batch_prepared', {'batch_id': batch['id'], 'direction': batch['direction'], 'ready': ready})
            elif job['kind'] == 'match':
                if result['status'] == 'completed':
                    task = read_json(root / 'task.json')
                    if result.get('task_hash') != digest(task) or result['players'] != job['assignment']['players']:
                        raise ValueError('match result does not match its frozen task')
                    payload = {k: result[k] for k in ('players', 'ranks', 'scores', 'hands', 'execution', 'diagnostics', 'task_hash')}
                    old = db.execute('SELECT payload FROM results WHERE run_id=? AND period=? AND match_index=?',
                                     (s['id'], job['period'], job['assignment']['index'])).fetchone()
                    encoded = json.dumps(payload)
                    if old is not None and old[0] != encoded:
                        raise ValueError('completed match receipt changed')
                    db.execute('INSERT OR IGNORE INTO results VALUES(?,?,?,?)',
                               (s['id'], job['period'], job['assignment']['index'], encoded))
                elif result['status'] == 'failed':
                    s.update(status='pausing', pause_reason='match failed: ' + result.get('error', 'unknown error'))

    def launch(self, s, job):
        root = Path(s['root']) / 'jobs' / job['id']
        root.mkdir(parents=True, exist_ok=True)
        receipt = root / 'result.json'
        if receipt.exists():
            attempts = root / 'attempts'
            attempts.mkdir(exist_ok=True)
            receipt.rename(attempts / f'{len(list(attempts.glob("*.json"))):05d}.json')
        (root / 'cancel.flag').unlink(missing_ok=True)
        remaining = s['seconds_granted'] - s['seconds_used']
        task_path = root / 'task.json'
        if task_path.exists() and job['kind'] != 'agent':
            task = read_json(task_path)
        else:
            key = read_json(Path(s['root']) / 'private/random.json')['key']
            task = dict(kind=job['kind'], run=s['id'], project=str(self.arena.project),
                        config=s['config']['game'], timeouts=s['config']['timeouts'], runtime=s['runtime'],
                        seed=derive_seed(key, 'job', job['id']))
            if job['kind'] == 'prepare':
                task.update(packages=job['packages'], training_games=job['training_games'],
                            opponents=[c['spec'] for c in s['candidates'].values() if c['status'] == 'active'])
            elif job['kind'] == 'match':
                a = job['assignment']
                task.update(players=a['players'], policies=[s['candidates'][cid]['spec'] for cid in a['players']],
                            seed=derive_seed(key, 'wall', job['period'], a['block']),
                            policy_seed=derive_seed(key, 'policy', job['period'], a['block']))
            else:
                d = s['directions'][job['direction']]
                task.update(direction=job['direction'], brief=d['brief'], thread_id=d['thread_id'])
        limit = s['config']['agent_turn_seconds'] if job['kind'] == 'agent' else s['config']['training_seconds'] if job['kind'] == 'prepare' else remaining
        task.update(deadline=time.time() + min(remaining, limit), owner_pid=os.getpid())
        atomic_json(task_path, task)
        module = 'research_agent' if job['kind'] == 'agent' else 'arena_jobs'
        runtime = str(Path(s['root']) / 'runtime')
        with (root / 'worker.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-B', '-m', 'mahjong_ai.' + module, str(root)],
                                       cwd=self.arena.project, env=dict(os.environ, PYTHONPATH=runtime),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        job.update(status='running', pid=process.pid, started_at=time.time())
        job.pop('error', None)
        atomic_json(root / 'process.json', {'pid': process.pid})
        job.pop('stop_requested_at', None)
        self.processes[(s['id'], job['id'])] = process

    def tick(self):
        running = False
        admissions = []
        for stored in self.arena.runs(active_only=True):
            with self.arena.edit(stored['id']) as (s, db):
                self.consume(s, db)
                for job in s['jobs'].values():
                    process = self.processes.get((s['id'], job['id']))
                    if job['status'] == 'running' and process is not None and process.poll() is not None:
                        path = Path(s['root']) / 'jobs' / job['id'] / 'result.json'
                        if not path.exists():
                            atomic_json(path, {'status': 'paused' if s['status'] != 'running' else 'failed',
                                               'error': 'worker exited without a receipt'})
                    if job['status'] == 'running' and s['status'] != 'running':
                        self.stop(s, job, force=time.time()-job.get('stop_requested_at', time.time()) > 5)
                    if job['status'] == 'running' and job['kind'] != 'agent':
                        task = read_json(Path(s['root']) / 'jobs' / job['id'] / 'task.json')
                        if time.time() >= task['deadline']:
                            self.stop(s, job, force=time.time()-task['deadline'] > 5)
                if s['status'] == 'pausing' and not any(j['status'] == 'running' for j in s['jobs'].values()):
                    s['status'] = 'paused'
                    self.arena.event(db, s, 'paused', {'reason': s['pause_reason']})
                if s['status'] != 'running':
                    continue
                running = True
                self.arena.advance(s, db)
                compute = sum(j['status'] == 'running' and j['kind'] != 'agent' for j in s['jobs'].values())
                queued = sorted((j for j in s['jobs'].values() if j['status'] == 'queued' and j['kind'] != 'agent'),
                                key=lambda j: (j['kind'] != 'prepare', j['created_at'], j['id']))
                for job in queued[:max(0, s['config']['workers']-compute)]:
                    job['status'] = 'admitted'
                    admissions.append((s['id'], job['id']))
                agents = sum(j['status'] == 'running' and j['kind'] == 'agent' for j in s['jobs'].values())
                ready = sorted((d for d in s['directions'].values() if d['status'] == 'ready'), key=lambda d: (d['last_run'], d['id']))
                for d in ready[:max(0, s['config']['agent_concurrency']-agents)]:
                    jid = f'agent-{d["id"]}-{d["turns"]:05d}'
                    job = dict(id=jid, kind='agent', status='queued', direction=d['id'], created_at=time.time(),
                               batches_before=sum(b['direction'] == d['id'] for b in s['batches'].values()))
                    s['jobs'][jid] = job
                    d.update(status='researching', last_run=time.time(), turns=d['turns']+1)
                    job['status'] = 'admitted'
                    admissions.append((s['id'], job['id']))
        for key, process in list(self.processes.items()):
            if process.poll() is not None:
                self.processes.pop(key, None)
        # Admission is committed before a process can exist. A failed spawn or
        # service crash always leaves a task that recovery can account for.
        for run_id, job_id in admissions:
            with self.arena.edit(run_id) as (s, db):
                job = s['jobs'][job_id]
                if s['status'] != 'running':
                    job['status'] = 'paused'
                    if job['kind'] == 'agent':
                        s['directions'][job['direction']]['status'] = 'ready'
                    continue
                try:
                    self.launch(s, job)
                except Exception as exc:
                    job['status'] = 'running'
                    atomic_json(Path(s['root']) / 'jobs' / job['id'] / 'result.json',
                                {'status': 'failed', 'error': f'worker launch failed: {exc}'[:2000]})
        if running and self.caffeine is None and Path('/usr/bin/caffeinate').exists():
            self.caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i'], stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif not running and self.caffeine is not None:
            self.caffeine.terminate()
            self.caffeine.wait()
            self.caffeine = None

    def close(self):
        for s in self.arena.runs():
            if s['status'] == 'running':
                self.arena.control(s['id'], 'pause')
        until = time.monotonic() + 7
        while self.processes and time.monotonic() < until:
            self.tick()
            time.sleep(.1)
        for s in self.arena.runs():
            for job in s['jobs'].values():
                if job['status'] == 'running':
                    self.stop(s, job, force=True)
        if self.caffeine is not None:
            self.caffeine.terminate()
            self.caffeine.wait()
