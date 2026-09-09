"""Research directions and period-based competition behind one transactional interface."""
from contextlib import contextmanager
from datetime import date
from itertools import combinations
import copy
import json
import math
import random
import re
import secrets
import shutil
import sqlite3
import sys
import time
import tomllib
from pathlib import Path

from .candidate_runtime import tree_hash
from .policy import PolicySpec
from .seeds import seed_sources
from .storage import atomic_json, digest, read_json, runtime_fingerprint
from .types import RoundConfig


DEFAULTS = dict(directions=['rule', 'rl'], seconds=3600, agent_concurrency=1,
                agent_turn_seconds=300, workers=2, pool_size=16, survivors=8,
                matches_per_period=128, new_candidates_per_period=4, pending_limit=16,
                training_seconds=300, game=dict(rounds=8, games_per_round=1),
                timeouts=dict(discard=3., response=1.))


def configuration(project):
    path = Path(project) / 'research.toml'
    given = tomllib.loads(path.read_text()) if path.exists() else {}
    if set(given) - set(DEFAULTS):
        raise ValueError('unknown research configuration')
    c = copy.deepcopy(DEFAULTS) | given
    c['game'] = RoundConfig(**c['game']).as_dict()
    for key, minimum, maximum in [('workers', 1, 2), ('agent_concurrency', 1, 8),
                                 ('pool_size', 4, 16), ('survivors', 4, 16),
                                 ('new_candidates_per_period', 1, 4), ('pending_limit', 4, 64),
                                 ('matches_per_period', 4, 4096)]:
        if type(c[key]) is not int or not minimum <= c[key] <= maximum:
            raise ValueError(f'invalid {key}')
    if c['survivors'] > c['pool_size'] or c['matches_per_period'] % 4:
        raise ValueError('survivors exceed pool or match count is not divisible by four')
    if min(c['pool_size']//4, c['pool_size']-c['survivors']) < c['new_candidates_per_period']:
        raise ValueError('pool must free enough places for one full batch after elimination')
    for key in ('seconds', 'agent_turn_seconds', 'training_seconds'):
        if not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or not 0 < c[key] <= 86400:
            raise ValueError(f'invalid {key}')
    if c['agent_turn_seconds'] > 300:
        raise ValueError('agent turns are limited to five minutes')
    if set(c['timeouts']) != {'discard', 'response'} or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 10 for v in c['timeouts'].values()):
        raise ValueError('invalid action timeouts')
    return c


def catalog(project):
    result = {}
    for path in sorted((Path(project) / 'directions').glob('*/direction.toml')):
        identity = path.parent.name
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', identity):
            raise ValueError('invalid direction name')
        d = tomllib.loads(path.read_text())
        if set(d) - {'name', 'brief', 'template', 'training_games'} or not isinstance(d.get('brief'), str):
            raise ValueError(f'invalid direction configuration: {identity}')
        template = path.parent / d.get('template', 'template')
        if not template.resolve().is_relative_to(path.parent.resolve()) or template.is_symlink():
            raise ValueError('template escapes direction')
        if template.exists():
            tree_hash(template)
            if not (template / 'strategy.py').is_file():
                raise ValueError('strategy.py missing from direction template')
        games = d.get('training_games', 64)
        if type(games) is not int or not 1 <= games <= 100000:
            raise ValueError('invalid training game allowance')
        result[identity] = dict(id=identity, name=d.get('name', identity), brief=d['brief'],
                                template=str(template.resolve()) if template.exists() else None,
                                training_games=games)
    return result


def schedule(candidate_ids, matches, seed):
    """Balance appearances first, repeated opponents second, then rotate all seats."""
    ids = sorted(candidate_ids)
    if len(ids) < 4 or len(ids) != len(set(ids)) or matches < 4 or matches % 4:
        raise ValueError('unique roster and four-seat match blocks required')
    rng = random.Random(seed)
    appearances, together = dict.fromkeys(ids, 0), {}
    output = []
    for block in range(matches // 4):
        chosen = []
        for _ in range(4):
            tie = {cid: rng.random() for cid in ids}
            cid = min((cid for cid in ids if cid not in chosen), key=lambda cid: (
                appearances[cid], sum(together.get(tuple(sorted((cid, p))), 0) for p in chosen), tie[cid]))
            chosen.append(cid)
        for cid in chosen:
            appearances[cid] += 1
        for pair in combinations(sorted(chosen), 2):
            together[pair] = together.get(pair, 0) + 1
        for seat_rotation in range(4):
            output.append(dict(index=len(output), block=block,
                               players=chosen[seat_rotation:] + chosen[:seat_rotation]))
    return output


def standings(roster, receipts):
    rows = {cid: dict(candidate_id=cid, appearances=0, utility_sum=0., errors=0, timeouts=0,
                      fallback_actions=0, opponents=set()) for cid in roster}
    for receipt in receipts:
        for seat, cid in enumerate(receipt['players']):
            row = rows[cid]
            row['appearances'] += 1
            row['utility_sum'] += 4 - receipt['ranks'][seat]
            row['opponents'].update(set(receipt['players']) - {cid})
            for key in ('errors', 'timeouts', 'fallback_actions'):
                row[key] += receipt['execution'][seat].get(key, 0)
    for row in rows.values():
        row['mean_utility'] = row['utility_sum'] / row['appearances'] if row['appearances'] else None
        row['opponent_count'] = len(row.pop('opponents'))
    return sorted(rows.values(), key=lambda r: (-(r['mean_utility'] if r['mean_utility'] is not None else -1),
                                                -r['appearances'], r['candidate_id']))


class Arena:
    def __init__(self, project):
        self.project = Path(project).resolve()
        private = self.project / '.local/service'
        private.mkdir(parents=True, exist_ok=True)
        private.chmod(0o700)
        self.database = private / 'arena.sqlite'
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, state TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL, at REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS results(run_id TEXT, period INTEGER, match_index INTEGER,
                payload TEXT NOT NULL, PRIMARY KEY(run_id,period,match_index));
            ''')
        self.database.chmod(0o600)

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=10, isolation_level=None)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA foreign_keys=ON')
        return db

    @contextmanager
    def connect(self):
        db = self._connect()
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def charge(s):
        now = time.time()
        if s['status'] == 'running':
            s['seconds_used'] += max(0, now - s['accounted_at'])
            if s['seconds_used'] >= s['seconds_granted']:
                s['status'] = 'pausing'
                s['pause_reason'] = 'budget exhausted'
        s['accounted_at'] = now

    @contextmanager
    def edit(self, run_id):
        db = self._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM runs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                raise ValueError('unknown run')
            s = json.loads(row[0])
            self.charge(s)
            yield s, db
            db.execute('UPDATE runs SET state=? WHERE id=?', (json.dumps(s, allow_nan=False), run_id))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, s, kind, data):
        db.execute('INSERT INTO events(run_id,at,kind,data) VALUES(?,?,?,?)',
                   (s['id'], time.time(), kind, json.dumps(data, ensure_ascii=False)))

    def runs(self, active_only=False):
        with self.connect() as db:
            query = 'SELECT state FROM runs'
            if active_only:
                query += " WHERE json_extract(state,'$.status') IN ('running','pausing')"
            return [json.loads(row[0]) for row in db.execute(query + ' ORDER BY rowid DESC')]

    def headers(self):
        with self.connect() as db:
            return [dict(id=r[0], status=r[1]) for r in db.execute(
                "SELECT id,json_extract(state,'$.status') FROM runs ORDER BY rowid DESC")]

    def activity(self, run_id):
        s = self.read(run_id)
        entries = []
        for job in list(s['jobs'].values())[::-1]:
            if job['kind'] != 'agent':
                continue
            path = Path(s['root']) / 'jobs' / job['id'] / 'codex.jsonl'
            if not path.exists():
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-131072))
                lines = stream.read().decode(errors='replace').splitlines()
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                item = event.get('item', {})
                if event.get('type') == 'item.completed' and item.get('type') == 'agent_message':
                    entries.append(dict(direction=job['direction'], turn=job['id'], text=item.get('text', '')[:8000]))
            if len(entries) >= 10:
                break
        return entries[-10:]

    def read(self, run_id):
        with self.connect() as db:
            row = db.execute('SELECT state FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('unknown run')
        return json.loads(row[0])

    def create(self, directions=None, seconds=None):
        config = configuration(self.project)
        chosen = directions if directions is not None else config['directions']
        available = catalog(self.project)
        if not chosen or len(set(chosen)) != len(chosen) or any(d not in available for d in chosen):
            raise ValueError('select distinct configured directions')
        duration = config['seconds'] if seconds is None else seconds
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 86400:
            raise ValueError('budget must be 0..86400 seconds')
        if any(s['status'] in ('running', 'pausing') for s in self.runs()):
            raise ValueError('pause the active run before creating another')
        base = self.project / '.local/runs'
        base.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            root = base / f'{date.today():%Y%m%d}-arena-{index:02d}'
            try:
                root.mkdir(mode=0o700)
                break
            except FileExistsError:
                index += 1
        run_id = root.name
        for name in ('private', 'reports', 'artifacts', 'jobs', 'directions'):
            (root / name).mkdir(mode=0o700)
        shutil.copytree(Path(__file__).parent, root / 'runtime/mahjong_ai',
                        ignore=shutil.ignore_patterns('__pycache__'))
        atomic_json(root / 'private/random.json', {'key': secrets.token_hex(32)})
        s = dict(id=run_id, root=str(root), version=3, created_at=time.time(), config=config,
                 runtime=runtime_fingerprint(), status='paused', pause_reason='ready',
                 seconds_granted=float(duration), seconds_used=0., accounted_at=time.time(),
                 directions={}, candidates={}, batches={}, jobs={}, periods=[], initial_period=True,
                 direction_cursor=0)
        for i, (name, code) in enumerate(seed_sources().items()):
            cid = f'seed-{i}'
            package = root / 'artifacts' / cid
            package.mkdir()
            (package / 'strategy.py').write_text(code)
            s['candidates'][cid] = dict(id=cid, label=name, direction='_seed', parent=None,
                                        hypothesis='ordinary starting entrant', status='active',
                                        requirements=[], spec=self.package_spec(s, package, cid))
        for identity in chosen:
            direction = available[identity]
            dest = root / 'directions' / identity
            template = dest / 'template'
            if direction['template']:
                shutil.copytree(direction['template'], template)
            else:
                template.mkdir(parents=True)
                (template / 'strategy.py').write_text(next(iter(seed_sources().values())))
            workspace = dest / 'candidates/initial'
            shutil.copytree(template, workspace)
            s['directions'][identity] = {**direction, 'template': str(template), 'status': 'ready',
                                         'thread_id': None, 'turns': 0, 'failures': 0, 'last_run': 0.}
            cid = identity + '--initial'
            s['candidates'][cid] = dict(id=cid, label='initial', direction=identity, parent=None,
                                        hypothesis=direction['brief'], status='draft', requirements=[],
                                        workspace=str(workspace), created_at=time.time())
        atomic_json(root / 'manifest.json', dict(version=3, config=config, runtime=s['runtime'],
                                               directions=s['directions'], scope='period-relative mixed competition'))
        with self.connect() as db:
            db.execute('INSERT INTO runs VALUES(?,?)', (run_id, json.dumps(s)))
        return run_id

    @staticmethod
    def package_spec(s, package, cid, python=None):
        return PolicySpec(family='code', version=cid, package_path=str(package), package_hash=tree_hash(package),
                          python_path=python or sys.executable, action_seconds=s['config']['timeouts']['discard'],
                          response_seconds=s['config']['timeouts']['response']).as_dict()

    def control(self, run_id, action, seconds=0):
        if action not in ('resume', 'pause', 'budget'):
            raise ValueError('unknown control action')
        with self.edit(run_id) as (s, db):
            if action == 'budget':
                if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 86400:
                    raise ValueError('positive budget addition, at most 24 hours')
                s['seconds_granted'] += seconds
            elif action == 'pause':
                s['status'] = 'pausing' if s['status'] == 'running' else s['status']
                s['pause_reason'] = 'operator paused'
            else:
                if s['status'] == 'pausing':
                    raise ValueError('wait until pause completes')
                if s['seconds_granted'] <= s['seconds_used']:
                    raise ValueError('add budget before resuming')
                others = db.execute('SELECT id,state FROM runs WHERE id<>?', (run_id,)).fetchall()
                if any(json.loads(row[1])['status'] in ('running', 'pausing') for row in others):
                    raise ValueError('another run is active')
                s['status'], s['pause_reason'] = 'running', ''
                for d in s['directions'].values():
                    if d['status'] == 'error':
                        d.update(status='ready', failures=0)
                for job in s['jobs'].values():
                    if job['status'] == 'paused' or job['status'] == 'failed' and job['kind'] == 'match':
                        job['status'] = 'queued'
            self.event(db, s, action, {'seconds': seconds})
        return self.snapshot(run_id)

    @staticmethod
    def require_running(s):
        if s['status'] != 'running':
            raise ValueError('research is paused')

    @staticmethod
    def own(s, direction, cid):
        if cid not in s['candidates']:
            cid = direction + '--' + cid
        if cid not in s['candidates'] or s['candidates'][cid]['direction'] != direction:
            raise ValueError('candidate is not owned by this research direction')
        return s['candidates'][cid]

    def candidate(self, run_id, direction, cid, *, path=None, content=None):
        with self.edit(run_id) as (s, db):
            c = self.own(s, direction, cid)
            if path is None:
                return copy.deepcopy(c)
            if not isinstance(path, str) or len(path) > 240 or Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('use a relative candidate file path')
            source = Path(c['spec']['package_path']) if c.get('spec') else Path(c['workspace'])
            target = source / path
            if not target.resolve().is_relative_to(source.resolve()) or any(p.is_symlink() for p in [target, *target.parents]):
                raise ValueError('candidate path escapes package')
            if content is not None:
                self.require_running(s)
                if c['status'] != 'draft' or not isinstance(content, str) or len(content.encode()) > 1_000_000:
                    raise ValueError('only draft text files up to 1 MB are editable')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                return {'candidate_id': c['id'], 'path': path, 'written': True}
            if target.stat().st_size > 1_000_000:
                raise ValueError('file too large for research source view')
            return {'candidate_id': c['id'], 'path': path, 'content': target.read_text()}

    def fork(self, run_id, direction, name, hypothesis, parent=None):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name):
            raise ValueError('invalid candidate name')
        with self.edit(run_id) as (s, db):
            self.require_running(s)
            if any(b['direction'] == direction and b['status'] not in ('completed', 'failed') for b in s['batches'].values()):
                raise ValueError('wait for the current batch result')
            cid = direction + '--' + name
            if cid in s['candidates']:
                raise ValueError('candidate already exists')
            if parent:
                previous = self.own(s, direction, parent)
                source = previous.get('spec', {}).get('package_path') or previous['workspace']
            else:
                previous = None
                source = s['directions'][direction]['template']
            dest = Path(s['root']) / 'directions' / direction / 'candidates' / name
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns('__pycache__'))
            s['candidates'][cid] = dict(id=cid, label=name, direction=direction, parent=previous['id'] if previous else None,
                                        hypothesis=hypothesis[:8000], status='draft', workspace=str(dest),
                                        requirements=previous.get('requirements', []) if previous else [], created_at=time.time())
            self.event(db, s, 'candidate_created', {'candidate_id': cid, 'direction': direction})
        return {'candidate_id': cid}

    def dependencies(self, run_id, direction, cid, requirements):
        if len(requirements) > 32 or any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+-]*', x) for x in requirements):
            raise ValueError('dependencies must be exact name==version wheels')
        with self.edit(run_id) as (s, db):
            self.require_running(s)
            c = self.own(s, direction, cid)
            if c['status'] != 'draft':
                raise ValueError('dependencies are frozen after submission')
            c['requirements'] = list(requirements)
        return {'requirements': requirements}

    def submit(self, run_id, direction, candidates, training_games=None):
        if not 1 <= len(candidates) <= 4 or len(set(candidates)) != len(candidates):
            raise ValueError('submit one to four distinct candidates')
        with self.edit(run_id) as (s, db):
            self.require_running(s)
            if any(b['direction'] == direction and b['status'] not in ('completed', 'failed') for b in s['batches'].values()):
                raise ValueError('a batch is already pending')
            pending = sum(c['status'] in ('submitted', 'ready') for c in s['candidates'].values())
            if pending + len(candidates) > s['config']['pending_limit']:
                raise ValueError('candidate queue is full; wait for a period to finish')
            if len(candidates) > s['config']['new_candidates_per_period']:
                raise ValueError('batch exceeds configured per-period admission capacity')
            rows = [self.own(s, direction, cid) for cid in candidates]
            if any(c['status'] != 'draft' for c in rows):
                raise ValueError('derive a new draft before submitting frozen candidates')
            games = training_games if training_games is not None else s['directions'][direction]['training_games']
            if type(games) is not int or not 1 <= games <= 100000:
                raise ValueError('invalid training allowance')
            bid = f'batch-{len(s["batches"]):05d}'
            packages = {}
            submission = Path(s['root']) / 'submissions' / bid
            # Only uncommitted residue can exist under this next unused batch ID.
            if submission.exists():
                shutil.rmtree(submission)
            try:
                for c in rows:
                    original = Path(c['workspace'])
                    before = tree_hash(original)
                    if not (original / 'strategy.py').is_file():
                        raise ValueError('strategy.py is required')
                    target = submission / c['id']
                    shutil.copytree(original, target)
                    if tree_hash(target) != before or tree_hash(original) != before:
                        raise ValueError('candidate changed while freezing')
                    packages[c['id']] = dict(input=str(target), hash=before, requirements=c['requirements'])
                    c.update(status='submitted', batch_id=bid)
            except BaseException:
                shutil.rmtree(submission, ignore_errors=True)
                raise
            s['batches'][bid] = dict(id=bid, direction=direction, candidates=[c['id'] for c in rows],
                                      status='preparing', created_at=time.time())
            job = dict(id='prepare-' + bid, kind='prepare', status='queued', direction=direction,
                       batch_id=bid, packages=packages, training_games=games, created_at=time.time())
            s['jobs'][job['id']] = job
            s['directions'][direction]['status'] = 'waiting'
            self.event(db, s, 'batch_submitted', {'batch_id': bid, 'direction': direction, 'candidates': list(packages)})
        return {'batch_id': bid, 'status': 'preparing'}

    def receipts(self, run_id, period, db=None):
        if db is not None:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM results WHERE run_id=? AND period=? ORDER BY match_index', (run_id, period))]
        with self.connect() as connection:
            return self.receipts(run_id, period, connection)

    def advance(self, s, db):
        if s['periods'] and s['periods'][-1]['status'] == 'running':
            period = s['periods'][-1]
            results = self.receipts(s['id'], period['id'], db)
            if len(results) != len(period['schedule']):
                return
            table = standings(period['roster'], results)
            count = min(len(table) // 4, max(0, len(table) - s['config']['survivors']))
            removed = [r['candidate_id'] for r in table[-count:]] if count else []
            for row in table:
                c = s['candidates'][row['candidate_id']]
                c['last_result'] = {**row, 'period': period['id']}
                c['status'] = 'archived' if c['id'] in removed else 'active'
            period.update(status='completed', table=table, eliminated=removed, completed_at=time.time())
            for bid in period['batches']:
                s['batches'][bid].update(status='completed', period=period['id'])
                s['directions'][s['batches'][bid]['direction']]['status'] = 'ready'
            self.event(db, s, 'period_completed', {'period': period['id'], 'eliminated': removed})
        active = [c['id'] for c in s['candidates'].values() if c['status'] == 'active']
        ready = [b for b in s['batches'].values() if b['status'] == 'ready']
        if not ready and not s['initial_period']:
            return
        capacity = min(s['config']['new_candidates_per_period'], s['config']['pool_size'] - len(active))
        admitted, batches = [], []
        direction_ids = list(s['directions'])
        order = direction_ids[s['direction_cursor']:] + direction_ids[:s['direction_cursor']]
        for direction in order:
            for batch in ready:
                ids = [cid for cid in batch['candidates'] if s['candidates'][cid]['status'] == 'ready']
                if batch['direction'] == direction and len(ids) <= capacity - len(admitted):
                    admitted.extend(ids)
                    batches.append(batch['id'])
        if not admitted and not s['initial_period']:
            return
        roster = active + admitted
        for cid in admitted:
            s['candidates'][cid]['status'] = 'active'
        number = len(s['periods']) + 1
        key = read_json(Path(s['root']) / 'private/random.json')['key']
        assignments = schedule(roster, s['config']['matches_per_period'], digest([key, 'schedule', number]))
        period = dict(id=number, status='running', roster=roster, batches=batches,
                      schedule=assignments, created_at=time.time())
        s['periods'].append(period)
        s['initial_period'] = False
        s['direction_cursor'] = (s['direction_cursor'] + 1) % len(direction_ids)
        for bid in batches:
            s['batches'][bid]['status'] = 'competing'
        for assignment in assignments:
            jid = f'match-{number:04d}-{assignment["index"]:05d}'
            s['jobs'][jid] = dict(id=jid, kind='match', status='queued', period=number,
                                  assignment=assignment, created_at=time.time())
        atomic_json(Path(s['root']) / 'periods' / f'{number:04d}.json', period)
        self.event(db, s, 'period_started', {'period': number, 'roster': roster})

    def snapshot(self, run_id=None, direction=None, after=0):
        if run_id is None:
            runs = self.runs()
            if not runs:
                return {'run': None, 'directions': list(catalog(self.project).values()), 'runs': []}
            run_id = runs[0]['id']
        s = self.read(run_id)
        self.charge(s)
        if direction is not None and direction not in s['directions']:
            raise ValueError('unknown research direction')
        period = copy.deepcopy(s['periods'][-1]) if s['periods'] else None
        if period:
            receipts = self.receipts(run_id, period['id'])
            period['completed_matches'] = len(receipts)
            period['total_matches'] = len(period.pop('schedule'))
            if not direction or period['status'] == 'completed':
                period['table'] = standings(period['roster'], receipts)
        candidates = []
        for c in s['candidates'].values():
            visible = {k: c[k] for k in ('id', 'label', 'direction', 'status', 'last_result') if k in c}
            if not direction or c['direction'] == direction:
                visible.update({k: c[k] for k in ('hypothesis', 'parent', 'error', 'batch_id', 'requirements') if k in c})
            if (not direction or c['direction'] == direction) and c.get('training'):
                training = c['training']
                visible['training'] = dict(status=training['status'], episodes=len(training['episodes']),
                                           reported_metrics=training.get('reported_metrics', {}))
            candidates.append(visible)
        jobs = [{k: job[k] for k in ('id', 'kind', 'status', 'direction', 'period', 'assignment', 'error', 'usage') if k in job}
                for job in s['jobs'].values() if job['status'] == 'running' and (not direction or job.get('direction') == direction)]
        if not direction:
            for job in jobs:
                path = Path(s['root']) / 'jobs' / job['id'] / 'progress.json'
                if path.exists():
                    job['progress'] = read_json(path)
        with self.connect() as db:
            events = [dict(seq=r[0], at=r[1], kind=r[2], data=json.loads(r[3])) for r in db.execute(
                'SELECT seq,at,kind,data FROM events WHERE run_id=? AND seq>? ORDER BY seq DESC LIMIT 100', (run_id, after))]
        if direction:
            events = [e for e in events if e['data'].get('direction') == direction or e['kind'] in ('period_completed', 'period_started')]
        public_directions = {d: {k: v[k] for k in ('id', 'name', 'status', 'turns', 'failures', 'last_error') if k in v}
                             for d, v in s['directions'].items() if not direction or d == direction}
        return dict(run=dict(id=s['id'], status=s['status'], pause_reason=s['pause_reason'],
                             seconds_remaining=max(0, s['seconds_granted']-s['seconds_used']), config=s['config']),
                    period=period, directions=public_directions, candidates=candidates, jobs=jobs,
                    batches=[b for b in s['batches'].values() if not direction or b['direction'] == direction],
                    periods=[{k: p[k] for k in ('id', 'status', 'table', 'eliminated') if k in p} for p in s['periods']],
                    events=events[::-1], runs=self.headers())
