"""Version Elo from completed research settlements; never a promotion criterion."""
import math
from pathlib import Path

from .storage import atomic_json, digest, exclusive_lock, read_json
from .types import RANK_OBJECTIVE


def elo_changes(policy_ids, ranks, ratings, k=24.0):
    if len(policy_ids) != 4 or len(ranks) != 4 or not math.isfinite(k) or k <= 0:
        raise ValueError('four participants and a positive finite K required')
    if any(not math.isfinite(r) or not 1 <= r <= 4 for r in ranks):
        raise ValueError('invalid settlement ranks')
    changes = {pid: 0.0 for pid in policy_ids}
    for i in range(4):
        for j in range(i+1, 4):
            a, b = policy_ids[i], policy_ids[j]
            if a == b: continue  # self-play against the identical version carries no evidence
            expected = 1 / (1 + 10 ** (max(-1200, min(1200, ratings.get(b,1500.) - ratings.get(a,1500.))) / 400))
            actual = 1. if ranks[i] < ranks[j] else .5 if ranks[i] == ranks[j] else 0.
            change = k / 3 * (actual - expected)
            changes[a] += change
            changes[b] -= change
    return changes


class EloLedger:
    def __init__(self, path, context, k=24.0):
        self.path = Path(path)
        self.protocol = {'version':1, 'objective':RANK_OBJECTIVE, 'context':context,
                         'initial':1500., 'k':k, 'pair_normalization':3}

    def record(self, event_id, policy_ids, ranks, *, completed, source):
        if source not in ('research-development','research-league'):
            raise ValueError('only research settlements can update research Elo')
        if not completed: raise ValueError('incomplete settlements cannot update Elo')
        payload = {'policy_ids':list(policy_ids),'ranks':list(ranks),'source':source}
        fingerprint = digest(payload)
        with exclusive_lock(self.path.with_suffix('.lock')):
            state = read_json(self.path) if self.path.exists() else {
                'protocol':self.protocol,'ratings':{},'appearances':{},'events':{}}
            if state['protocol'] != self.protocol: raise ValueError('Elo protocol changed; use another ledger')
            if event_id in state['events']:
                if state['events'][event_id] != fingerprint: raise ValueError('Elo event changed')
                return self._table(state)
            changes = elo_changes(policy_ids,ranks,state['ratings'],self.protocol['k'])
            for pid, change in changes.items():
                state['ratings'][pid] = state['ratings'].get(pid,1500.) + change
                state['appearances'][pid] = state['appearances'].get(pid,0) + policy_ids.count(pid)
            state['events'][event_id] = fingerprint
            atomic_json(self.path,state)
            return self._table(state)

    def table(self):
        if not self.path.exists(): return []
        state=read_json(self.path)
        if state['protocol'] != self.protocol: raise ValueError('Elo protocol changed')
        return self._table(state)

    @staticmethod
    def _table(state):
        return [{'policy_id':pid,'elo':rating,'appearances':state['appearances'][pid]}
                for pid,rating in sorted(state['ratings'].items(),key=lambda p:(-p[1],p[0]))]
