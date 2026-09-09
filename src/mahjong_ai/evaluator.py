from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import ENGINE_VERSION, MahjongEngine, derive_seed
from .policy import PolicySpec
from .storage import atomic_json, digest, read_json
from .types import RANK_OBJECTIVE, RoundConfig


@dataclass
class EvaluationReport:
    protocol: dict
    pairs: list[dict]
    alpha: float = 0.05

    @property
    def failures(self):
        return [p for p in self.pairs if not p['completed']]

    @property
    def completed(self):
        return [p for p in self.pairs if p['completed']]

    @property
    def mean_rank_utility_delta(self):
        return float(np.mean([p['delta'] for p in self.completed])) if self.completed else 0.0

    def summary(self):
        pairs = self.completed
        deltas = np.asarray([p['delta'] for p in pairs], dtype=float)
        interval = None
        method = 'paired percentile bootstrap; approximate, small/tail samples may be underpowered'
        if len(pairs) >= 30 and self.alpha < .0001:
            # Rank utility difference lies in [-3,3]. Avoid ever-growing bootstrap
            # work at tiny alpha; a bounded-mean interval is conservative but finite.
            radius = math.sqrt(18 * math.log(2 / self.alpha) / len(pairs))
            interval = [max(-3.,float(deltas.mean())-radius),min(3.,float(deltas.mean())+radius)]
            method = 'Hoeffding bounded paired mean interval; independent settlement seeds, conservative'
        elif len(pairs) >= 30:
            # Paired percentile bootstrap. Fresh seeds are the sampling units;
            # hands within a round are not treated as independent observations.
            rng = np.random.default_rng(271828)
            reps = max(2000, math.ceil(40 / self.alpha))
            means = []
            for _ in range(0, reps, 256):
                chunk = rng.choice(deltas, size=(min(256, reps - len(means)), len(deltas)), replace=True).mean(axis=1)
                means.extend(chunk)
            interval = [float(np.quantile(means, self.alpha / 2)),
                        float(np.quantile(means, 1 - self.alpha / 2))]
        def mean(key, field):
            return float(np.mean([p[key][field] for p in pairs])) if pairs else 0.0
        return {
            'engine': self.protocol['engine'],
            'protocol_id': digest(self.protocol),
            'requested_pairs': self.protocol['pairs'],
            'completed_pairs': len(pairs),
            'attempted_pairs': len(self.pairs),
            'actual_rounds_started': sum(p['rounds_started'] for p in self.pairs),
            'failure_count': len(self.failures),
            'baseline_mean_score': mean('baseline', 'score'),
            'candidate_mean_score': mean('candidate', 'score'),
            'objective': RANK_OBJECTIVE,
            'mean_rank_utility_delta': self.mean_rank_utility_delta,
            'mean_score_delta': mean('candidate', 'score') - mean('baseline', 'score'),
            'baseline_mean_rank': mean('baseline', 'rank'),
            'candidate_mean_rank': mean('candidate', 'rank'),
            'baseline_mean_rank_utility': mean('baseline', 'rank_utility'),
            'candidate_mean_rank_utility': mean('candidate', 'rank_utility'),
            'baseline_rank_distribution': [sum(p['baseline']['rank'] == rank for p in pairs) / len(pairs) if pairs else 0 for rank in range(1, 5)],
            'candidate_rank_distribution': [sum(p['candidate']['rank'] == rank for p in pairs) / len(pairs) if pairs else 0 for rank in range(1, 5)],
            'bootstrap_interval': interval,
            'alpha': self.alpha,
            'uncertainty_method': method,
            'baseline_first_rate': mean('baseline', 'first'),
            'candidate_first_rate': mean('candidate', 'first'),
            'baseline_top_two_rate': mean('baseline', 'top_two'),
            'candidate_top_two_rate': mean('candidate', 'top_two'),
            'stage': self.protocol['config']['stage'],
            'rounds_per_game': self.protocol['config']['rounds'],
            'games_per_sample': self.protocol['config'].get('games_per_round', 1),
            'verified_hands': sum(p[k]['rounds_played'] for p in pairs for k in ('baseline','candidate')),
            'elo_used_for_promotion': False,
            'execution': {side: {key: sum(p[side].get('execution', {}).get(key,0) for p in pairs)
                                for key in ('actions','timeouts','errors','elapsed_seconds')}
                          for side in ('baseline','candidate')},
            'calibration': 'research engine; see versioned oracle report; full transition parity pending',
        }


_code_players = {}


def close_code_players():
    for players in _code_players.values():
        for player in players:
            if hasattr(player, 'close'): player.close()
    _code_players.clear()


def _pair_job(job):
    index, seed, old, new, opponents, config, record = job
    seat = index % 4
    output = {'index': index, 'seed': seed, 'seat': seat, 'completed': False, 'rounds_started': 0}
    try:
        for label, spec in (('baseline', old), ('candidate', new)):
            lineup = [*opponents[:seat], spec, *opponents[seat:]]
            use_cache = any(p.get('package_path') for p in lineup)
            cache_key = digest(lineup)
            if use_cache and cache_key in _code_players:
                players = _code_players[cache_key]
            else:
                players = [PolicySpec(**x).build() for x in lineup]
                if use_cache: _code_players[cache_key] = players
            output['rounds_started'] += 1
            before = dict(getattr(players[seat], 'stats', {}))
            try:
                result = MahjongEngine(record_events=record).run_round(players, RoundConfig(**config), seed)
            finally:
                if not use_cache:
                    for player in players:
                        if hasattr(player, 'close'): player.close()
            output[label] = {
                'score': result.score(seat), 'first': result.winner_seat == seat,
                'top_two': seat in result.top_two_seats, 'rounds_played': result.rounds_played,
                'rank': result.rank_by_seat[seat] if result.completed else None,
                'rank_utility': result.rank_utility(seat) if result.completed else None,
                'rank_by_player': result.rank_by_seat,
                'execution': {k:v-before.get(k,0) for k,v in getattr(players[seat], 'stats', {}).items()},
                'policy_ids': [PolicySpec(**x).identity for x in ([*opponents[:seat], spec, *opponents[seat:]])],
                'result': result.as_dict() if record else {'reason': result.metadata['reason']},
            }
            if not result.completed:
                output['error'] = result.metadata['reason']
                return output
        output['delta'] = output['candidate']['rank_utility'] - output['baseline']['rank_utility']
        output['completed'] = True
    except Exception as exc:
        output['error'] = f'{type(exc).__name__}: {exc}'
    return output


class Evaluator:
    def evaluate(self, baseline: PolicySpec, candidate: PolicySpec, opponents, config,
                 pairs=1000, seed=0, output=None, workers=1, record_events=False,
                 alpha=0.05, deadline=None):
        if len(opponents) != 3:
            raise ValueError('a tested policy requires exactly THREE frozen opponents')
        if pairs < 1 or workers < 1 or not 0 < alpha < 0.5:
            raise ValueError('invalid evaluation budget')
        protocol = {'version': 2, 'objective': RANK_OBJECTIVE, 'engine': ENGINE_VERSION,
                    'baseline': baseline.identity, 'candidate': candidate.identity,
                    'opponents': [p.identity for p in opponents], 'config': config.as_dict(),
                    'pairs': pairs, 'seed': seed, 'alpha': alpha, 'record_events': record_events}
        root = Path(output) if output else None
        if root:
            root.mkdir(parents=True, exist_ok=True)
            root.chmod(0o700)
            manifest = root / 'protocol.json'
            if manifest.exists() and read_json(manifest) != protocol:
                raise ValueError('resume protocol differs; use a new evaluation directory')
            atomic_json(manifest, protocol)
        results, jobs = [], []
        for i in range(pairs):
            path = root / f'pair-{i:06}.json' if root else None
            if path and path.exists():
                r = read_json(path)
                checksum = r.pop('checksum', None)
                if (checksum != digest(r) or r.get('protocol_id') != digest(protocol)
                        or r['index'] != i or r['seed'] != derive_seed(seed, 'pair', i)):
                    raise ValueError('corrupt cached pair')
                r['checksum'] = checksum
                results.append(r)
            else:
                jobs.append((i, derive_seed(seed, 'pair', i), baseline.as_dict(), candidate.as_dict(),
                             [p.as_dict() for p in opponents], config.as_dict(), record_events))
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            # Submit bounded batches so the time budget does not enqueue an
            # unbounded backlog. In-flight complete rounds are allowed to finish.
            for start in range(0, len(jobs), workers):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                batch = jobs[start:start + workers]
                iterator = pool.map(_pair_job, batch) if pool else map(_pair_job, batch)
                for result in iterator:
                    result['protocol_id'] = digest(protocol)
                    result['checksum'] = digest(result)
                    if root: atomic_json(root / f"pair-{result['index']:06}.json", result)
                    results.append(result)
        finally:
            if pool: pool.shutdown(wait=True)
            else: close_code_players()
        report = EvaluationReport(protocol, sorted(results, key=lambda p: p['index']), alpha)
        if root: atomic_json(root / 'summary.json', report.summary())
        return report


def promotion_decision(report):
    if report.protocol.get('objective') != RANK_OBJECTIVE:
        return {'status': 'inconclusive', 'reason': 'rank objective protocol required; legacy score evidence cannot promote'}
    summary = report.summary()
    if any(v['errors'] for v in summary['execution'].values()):
        return {'status': 'inconclusive', 'reason': 'runtime errors or illegal actions; repair before promotion'}
    if report.failures or len(report.completed) != report.protocol['pairs']:
        return {'status': 'inconclusive', 'reason': 'failed or unfinished full rounds'}
    interval = summary['bootstrap_interval']
    if interval is None:
        return {'status': 'inconclusive', 'reason': 'at least 30 independent paired rounds required'}
    if interval[0] > 0:
        return {'status': 'promoted', 'reason': 'positive paired rank-utility interval; local research baseline only'}
    if interval[1] < 0:
        return {'status': 'rejected', 'reason': 'negative paired rank-utility interval'}
    return {'status': 'inconclusive', 'reason': 'rank-utility interval includes zero'}
