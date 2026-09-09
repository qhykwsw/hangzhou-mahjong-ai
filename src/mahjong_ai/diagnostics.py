"""Behavior diagnostics on developer trajectories; never changes score criteria."""
from __future__ import annotations

from .engine import MahjongEngine, derive_seed


class ComparisonPolicy:
    def __init__(self, baseline, candidate):
        self.baseline, self.candidate = baseline, candidate
        self.decisions, self.differences = 0, 0

    def reset(self, seed):
        self.baseline.reset(seed)
        self.candidate.reset(seed)
        self.decisions = self.differences = 0

    def act(self, observation):
        old = self.baseline.act(observation)
        new = self.candidate.act(observation)
        self.decisions += 1
        self.differences += old != new
        return old

    def finish_round(self, result, seat):
        pass


def compare_behavior(baseline, candidate, opponents, config, rounds=2, seed=0):
    decisions = differences = 0
    for i in range(rounds):
        compared = ComparisonPolicy(baseline.build(), candidate.build())
        players = [p.build() for p in opponents]
        players.insert(i % 4, compared)
        result = MahjongEngine(record_events=False).run_round(players, config, derive_seed(seed, 'behavior', i))
        if not result.completed:
            return {'status': 'incomplete', 'reason': result.metadata['reason']}
        decisions += compared.decisions
        differences += compared.differences
    return {'status': 'completed', 'developer_rounds': rounds, 'decisions': decisions,
            'different_actions': differences, 'difference_rate': differences / max(1, decisions),
            'scope': 'same public observations on baseline developer trajectories; NOT score evidence'}
