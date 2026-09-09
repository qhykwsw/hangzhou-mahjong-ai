"""Official HTTP transport and strict, explicitly mapped player snapshots.

The guide omits several snapshot fields. A mapping verified against an actual
player snapshot is required for play; the probe needs no mapping and never acts.
"""
from __future__ import annotations

import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from .engine import HandState, derive_seed
from .rules import tile_id
from .storage import atomic_json, digest, exclusive_lock, read_json
from .types import Meld, Observation, PublicEvent, RoundConfig, StageContext, StageKind


class ApiError(RuntimeError):
    def __init__(self, status, code=''):
        self.status, self.code = status, code
        super().__init__(f'platform HTTP {status}')  # never echo credentials or HTTP bodies


class SnapshotError(ValueError):
    pass


class RateLimiter:
    """Shared across all game workers for one credential/user process."""
    def __init__(self, rate=14, clock=time.monotonic, sleep=time.sleep):
        self.interval, self.clock, self.sleep = 1 / rate, clock, sleep
        self.next = 0.
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = self.clock()
            scheduled = max(now, self.next)
            self.next = scheduled + self.interval
        self.sleep(max(0., scheduled - now))


class PlatformClient:
    def __init__(self, base, token, insecure=False, retries=3):
        if not token: raise ValueError('token environment variable is empty')
        self.base, self._token, self.retries = base.rstrip('/'), token, retries
        self.context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        self.limiter = RateLimiter()

    def request(self, method, path, body=None):
        for attempt in range(self.retries + 1):
            if '/state?' in path: self.limiter.acquire()
            req = urllib.request.Request(self.base + path, method=method,
                     data=json.dumps(body).encode() if body is not None else None,
                     headers={'Authorization': 'Bearer ' + self._token, 'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=35, context=self.context) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self.retries:
                    try: delay = float(exc.headers.get('Retry-After', 2 ** attempt))
                    except ValueError: delay = 2 ** attempt
                    time.sleep(min(10, max(.1, delay)))
                    continue
                try:
                    error = json.load(exc)
                    code = error.get('error', error.get('code', ''))
                    code = code if isinstance(code, str) and re.fullmatch(r'[A-Z_0-9]+', code) else ''
                except (ValueError, TypeError, AttributeError):
                    code = ''
                raise ApiError(exc.code, code) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                # A POST may have been accepted: caller must resync, never replay it blindly.
                if method != 'GET' or attempt == self.retries:
                    raise ConnectionError('platform connection failed; action outcome may be unknown') from None
                time.sleep(min(4, 2 ** attempt))

    def discover(self, tournament_id=None):
        me = self.request('GET', '/api/me')
        scoped = me.get('tournament_id')
        if scoped and tournament_id and scoped != tournament_id:
            raise ValueError('requested tournament differs from scoped token')
        tid = scoped or tournament_id
        if not tid: raise ValueError('global token requires --tournament-id')
        return str(tid), bool(scoped), me


def _id(value):
    return quote(str(value), safe='')


def _redact(value):
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()
                if not any(x in k.lower() for x in ('token', 'password', 'secret', 'authorization'))}
    if isinstance(value, list): return [_redact(x) for x in value]
    return value


def probe(client, output, tournament_id=None):
    tid, scoped, me = client.discover(tournament_id)
    tournament = client.request('GET', f'/api/tournaments/{_id(tid)}')
    games = active_games(me, tournament, scoped)
    data = {'me': me, 'tournament': tournament, 'games': {}}
    for gid in games:
        data['games'][gid] = client.request('GET', f'/api/games/{_id(gid)}/state?seq=0')
    root = Path(output).parent
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    atomic_json(output, _redact(data))
    return {'games': len(games), 'output': str(output), 'actions_submitted': 0}


def active_games(me, tournament, scoped):
    ids = [str(g['game_id']) for g in me['active_games']]
    return ids if scoped else [g for g in ids if g in tournament.get('my_games', [])]


# Null entries deliberately prevent play until verified against a player snapshot.
MAPPING_TEMPLATE = {
    'seat': 'snapshot.seat', 'phase': 'snapshot.phase', 'turn': 'snapshot.turn',
    'responding_seats': 'snapshot.responding_seats', 'hand': 'snapshot.my_hand',
    'drawn_tile': 'snapshot.drawn_tile', 'melds': 'snapshot.melds',
    'scores': 'snapshot.scores', 'round_no': 'snapshot.round_no',
    'catch_play': 'snapshot.god.catch_play', 'chain_count': 'snapshot.god.chain_count',
    'god_discarder_seat': 'snapshot.god.god_discarder_seat',
    'dealer': None, 'rounds_total': None, 'base_score': None, 'you_cai_bi_kao': None,
    'discards': None, 'hand_sizes': None, 'wall_remaining': None, 'discarded_tile': None,
    'kong_draw': None, 'piao_count': None, 'public_events': None,
    'stage_kind': None, 'stage_no': None, 'stage_scores': None,
    'stage_round_no': None, 'stage_rounds_total': None, 'in_tiebreak': None,
}


def _mapped(document, locator, key):
    if isinstance(locator, dict) and set(locator) == {'value'}:
        return locator['value']
    if not isinstance(locator, str): raise SnapshotError(f'unverified mapping: {key}')
    value = document
    try:
        for part in locator.split('.'):
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError):
        raise SnapshotError(f'missing mapped field: {key}') from None
    return value


def _tile(value):
    if value is None or value == '': return None
    if not isinstance(value, str): raise SnapshotError('wire tiles must use official tile strings')
    return tile_id(value)


class SnapshotAdapter:
    """Map only the supplied player's wire data, then use the local action rules.

    Complex upstream representations should be normalized by a custom adapter
    implementing this same callable interface, not inferred from hidden state.
    """
    def __init__(self, mapping):
        if set(mapping) != set(MAPPING_TEMPLATE) or any(v is None for v in mapping.values()):
            raise SnapshotError('complete and verify every field in the mapping template before play')
        self.mapping = mapping

    def __call__(self, document):
        s = {k: _mapped(document, locator, k) for k, locator in self.mapping.items()}
        seat = s['seat']
        if type(seat) is not int or not 0 <= seat < 4:
            raise SnapshotError('snapshot has no player seat')
        phase = s['phase']
        if not (phase == 'draw' and s['turn'] == seat or
                phase in ('response_peng', 'response_chi') and seat in s['responding_seats']):
            return None
        for key in ('melds', 'discards', 'hand_sizes', 'scores'):
            if not isinstance(s[key], list) or len(s[key]) != 4:
                raise SnapshotError(f'{key} must contain four seats')
        if s['stage_scores'] is not None and len(s['stage_scores']) != 4:
            raise SnapshotError('stage_scores must align with current four player identities')
        for key in ('catch_play', 'kong_draw', 'you_cai_bi_kao', 'in_tiebreak'):
            if type(s[key]) is not bool: raise SnapshotError(f'{key} must be boolean')
        owner = s['god_discarder_seat']
        if (type(owner) is not int or owner not in range(-1, 4)
                or s['catch_play'] != (owner >= 0)):
            raise SnapshotError('invalid god_discarder_seat for catch_play')
        for key in ('dealer', 'round_no', 'rounds_total', 'wall_remaining', 'chain_count', 'piao_count'):
            if type(s[key]) is not int or s[key] < 0: raise SnapshotError(f'invalid {key}')
        if s['dealer'] > 3 or s['round_no'] < 1: raise SnapshotError('invalid dealer/hand number')
        melds = tuple(tuple(Meld(m['kind'], tuple(_tile(t) for t in m['tiles']),
                                m.get('from_seat'), m.get('concealed', False)) for m in row) for row in s['melds'])
        if any(m.kind not in ('chi', 'peng', 'gang') for row in melds for m in row):
            raise SnapshotError('unknown meld kind; provide a verified normalizer')
        events = tuple(PublicEvent(e['kind'], e.get('seat'), _tile(e.get('tile')),
                       tuple(_tile(t) for t in e.get('tiles', [])), e['hand_no'], tuple(e.get('scores', [])))
                       for e in s['public_events'])
        config = RoundConfig(rounds=s['rounds_total'], base_score=s['base_score'],
                             stage=s['stage_kind'], you_cai_bi_kao=s['you_cai_bi_kao'])
        # This is a legal-action projection, never a simulated unknown tile wall.
        view = HandState.__new__(HandState)
        view.config, view.settled, view.turn, view.phase = config, False, seat, phase
        view.hands = [[] for _ in range(4)]
        view.hands[seat] = [_tile(t) for t in s['hand']]
        view.melds, view.drawn = melds, _tile(s['drawn_tile'])
        view.chain, view.piao = [0] * 4, [0] * 4
        view.chain[seat], view.piao[seat] = s['chain_count'], s['piao_count']
        view.kong_draw = s['kong_draw']
        view.catch_owner = owner if s['catch_play'] else None
        view.wall = range(s['wall_remaining'])  # only len() is read by legal_actions
        view.discarded = _tile(s['discarded_tile'])
        if phase in ('response_peng', 'response_chi') and view.discarded is None:
            raise SnapshotError('response window requires the actual discarded tile')
        if view.drawn is not None and phase == 'draw' and view.drawn not in view.hands[seat]:
            raise SnapshotError('drawn tile missing from own hand')
        legal = view.legal_actions()
        if not legal: raise SnapshotError('no legal action in an actionable state')
        fan = view.current_fan()
        ctx = StageContext(StageKind(s['stage_kind']), seat, tuple(s['scores']), s['round_no'],
                           s['rounds_total'], s['dealer'], s['stage_no'],
                           tuple(s['stage_scores']) if s['stage_scores'] is not None else None,
                           in_tiebreak=s['in_tiebreak'], stage_round_no=s['stage_round_no'],
                           stage_rounds_total=s['stage_rounds_total'])
        return Observation(ctx, tuple(sorted(view.hands[seat])), events, legal, melds,
                           tuple(tuple(_tile(t) for t in row) for row in s['discards']),
                           tuple(s['hand_sizes']), s['wall_remaining'], phase, view.drawn,
                           s['catch_play'], s['chain_count'], s['piao_count'], fan.fan if fan else 0,
                           s['you_cai_bi_kao'], god_discarder_seat=owner)


def wire_action(action):
    result = action.as_dict()
    result.pop('kind', None)  # official API infers the kong kind; this key is local only
    return result


class TournamentRunner:
    def __init__(self, client, policy, adapter, output, workers=16, join=False):
        if not 1 <= workers <= 16: raise ValueError('workers must be 1..16')
        self.client, self.policy, self.adapter = client, policy, adapter
        self.root, self.workers, self.join = Path(output), workers, join
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.tournament = None

    def _play(self, gid):
        policy = self.policy.build()
        policy.reset(derive_seed(gid, self.policy.identity))
        path = self.root / ('game-' + digest(gid)[:24] + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        state = read_json(path) if path.exists() else {'responses': [], 'status': 'running', 'events': []}
        if state.get('policy_id', self.policy.identity) != self.policy.identity:
            raise ValueError('cannot switch policy while resuming a game')
        state.update(policy_id=self.policy.identity, game_id=gid)
        seen = set(state['responses'])
        seq = state.get('event_seq', 0)
        events = state.setdefault('events', [])
        event_ids = {e['seq'] for e in events}
        hand_no = state.get('round_no', 1)
        force_snapshot = False
        def ingest(payload):
            nonlocal seq, hand_no
            for event in payload.get('events') or []:
                if event['seq'] not in event_ids:
                    event = dict(event)
                    event['_round_no'] = (event.get('data') or {}).get('round_no', hand_no)
                    events.append(event)
                    event_ids.add(event['seq'])
                    if event['type'] == 'round_ended': hand_no = event['_round_no'] + 1
                seq = max(seq, event['seq'])
        while not self.stop.is_set():
            # A fully answered peng window can become chi without emitting
            # any event. Snapshot polling must therefore be independent of seq.
            cursor = 0
            force_snapshot = False
            res = self.client.request('GET', f'/api/games/{_id(gid)}/state?seq={cursor}')
            ingest(res)
            if res.get('finished'):
                state.update(status='finished', final=_redact(res), event_seq=seq)
                atomic_json(path, state)
                return
            if res.get('pending'): continue
            if res.get('snapshot') is None:
                res = self.client.request('GET', f'/api/games/{_id(gid)}/state?seq=0')
            snap = res.get('snapshot')
            if snap is None: raise SnapshotError('full state response lacks snapshot')
            hand_no = snap['round_no']
            # seq=0 is the server's full-snapshot sentinel. Preserve subsequent
            # events instead of repeatedly discarding them after each action.
            if seq == 0 and res.get('seq', 0) > 0:
                seq = 1
                # First event is recoverable as the initial dealer discard when
                # the current river still contains it. Otherwise flag the gap.
                river = snap['discards'][snap['dealer']]
                if river and 1 not in event_ids:
                    events.append({'seq': 1, 'type': 'tile_discarded', 'seat': snap['dealer'],
                                   'tile': river[0], 'data': {}, '_round_no': hand_no, 'reconstructed': True})
                    event_ids.add(1)
                else:
                    state['initial_history_gap'] = True
            if res.get('seq', 0) > seq:
                delta = self.client.request('GET', f'/api/games/{_id(gid)}/state?seq={seq}')
                ingest(delta)
                if delta.get('snapshot'):
                    snap = delta['snapshot']
                    seq = max(seq, delta.get('seq', seq))
                elif delta.get('seq', 0) > res.get('seq', 0):
                    res = self.client.request('GET', f'/api/games/{_id(gid)}/state?seq=0')
                    if res.get('snapshot'): snap = res['snapshot']
            with self.lock: tournament = self.tournament
            obs = self.adapter({'snapshot': snap, 'tournament': tournament, 'response': res,
                                'wire_events': sorted(events, key=lambda e: e['seq'])})
            state.update(event_seq=seq, round_no=hand_no)
            if obs is None: continue
            key = digest({'window': snap.get('window_deadline_ms'), 'hand': obs.stage.round_no, 'phase': obs.phase, 'turn': snap['turn'],
                          'melds': snap['melds'], 'discards': obs.discards, 'own_hand': obs.hand,
                          'drawn': obs.drawn_tile, 'wall': obs.wall_remaining})
            if key in seen: continue
            begin = time.perf_counter()
            from .candidate_runtime import ActionUnavailable, RemotePolicy
            if isinstance(policy, RemotePolicy) and snap.get('window_deadline_ms'):
                remaining = max(.001, snap['window_deadline_ms']/1000-time.time())
                policy.action_seconds = min(self.policy.action_seconds, remaining)
                policy.response_seconds = min(self.policy.response_seconds, remaining)
            try:
                action = policy.act(obs)
            except ActionUnavailable:
                # The server owns timeout fallback. Never submit a stale action.
                seen.add(key)
                force_snapshot = True
                continue
            decision_ms = (time.perf_counter() - begin) * 1000
            if action not in obs.legal_actions: raise SnapshotError('policy returned an illegal action')
            seen.add(key)
            state['responses'].append(key)
            atomic_json(path, state)
            record = {'game_id': gid, 'seq': res.get('seq', seq), 'snapshot': snap,
                      'action': wire_action(action), 'decision_ms': decision_ms,
                      'request_ms': None, 'accepted': False}
            begin = time.perf_counter()
            try:
                response = self.client.request('POST', f'/api/games/{_id(gid)}/action', wire_action(action))
                record.update(accepted=True, response=response)
                if seq == 0:
                    kind = {'discard': 'tile_discarded', 'hu': 'round_ended'}.get(action.action, action.action)
                    events.append({'seq': 1, 'type': kind, 'seat': obs.stage.seat,
                                   'tile': wire_action(action).get('tile', ''),
                                   'data': {'kind': {'concealed': 'an', 'exposed': 'ming', 'added': 'bu'}.get(action.kind, '')},
                                   '_round_no': hand_no, 'reconstructed': True})
                    event_ids.add(1)
                    seq = 1
            except ApiError as exc:
                record.update(http_status=exc.status, error_code=exc.code)
                if exc.status != 409: raise
                force_snapshot = True
            except ConnectionError:
                force_snapshot = True
                record['error_code'] = 'ACCEPTANCE_UNKNOWN'
            finally:
                record['request_ms'] = (time.perf_counter() - begin) * 1000
                with path.with_suffix('.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            state['event_seq'] = seq
            atomic_json(path, state)
        state['status'] = 'paused'
        atomic_json(path, state)

    def run(self, tournament_id=None, seconds=300):
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        deadline = time.monotonic() + seconds
        if seconds <= 0: raise ValueError('positive online time budget required')
        tid, scoped, _ = self.client.discover(tournament_id)
        if self.join:
            # Tournament detail is forbidden before registration for global tokens.
            for operation in ('register', 'ready'):
                try:
                    self.client.request('POST', f'/api/tournaments/{_id(tid)}/{operation}')
                except ApiError as exc:
                    if exc.status != 409: raise
        futures, completed = {}, set()
        with exclusive_lock(self.root / '.online.lock'), ThreadPoolExecutor(max_workers=self.workers) as pool:
            try:
                while time.monotonic() < deadline:
                    t = self.client.request('GET', f'/api/tournaments/{_id(tid)}')
                    with self.lock: self.tournament = t
                    # Persist authoritative stage ledger; never sum repeated ranking snapshots.
                    atomic_json(self.root / 'tournament.json', _redact(t))
                    status = t['status']
                    if status in ('finished', 'closed', 'void'): break
                    if status == 'stage_open' and not t.get('qualified'): break
                    if self.join and status in ('registering', 'stage_open'):
                        try:
                            if status == 'registering':
                                self.client.request('POST', f'/api/tournaments/{_id(tid)}/register')
                            self.client.request('POST', f'/api/tournaments/{_id(tid)}/ready')
                        except ApiError as exc:
                            if exc.status != 409: raise
                    for gid, future in list(futures.items()):
                        if future.done():
                            future.result()  # worker failures stop this run, never disappear
                            completed.add(gid)
                            del futures[gid]
                    if status == 'running':
                        me = self.client.request('GET', '/api/me')
                        games = active_games(me, t, scoped)
                        if len(set(games) - completed) > self.workers:
                            raise RuntimeError('insufficient independent workers for active games')
                        for gid in games:
                            if gid not in futures and gid not in completed:
                                futures[gid] = pool.submit(self._play, gid)
                    elif status not in ('registering', 'stage_open', 'stage_done'):
                        raise SnapshotError('unknown tournament status')
                    self.stop.wait(1)
            finally:
                self.stop.set()
            for future in futures.values(): future.result()
        return {'status': 'stopped', 'games_finished': len(completed), 'output': str(self.root)}
