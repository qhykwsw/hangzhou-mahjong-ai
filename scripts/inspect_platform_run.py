#!/usr/bin/env python3
"""Read-only progress/latency report from an in-progress or finished platform run."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def records(path):
    output = []
    for line in path.read_text().splitlines():
        try: output.append(json.loads(line))
        except json.JSONDecodeError: pass  # a concurrent writer may have an incomplete last line
    return output


def inspect(root):
    result = {'output': str(root), 'players': []}
    directories = [p for p in root.glob('player-*') if p.is_dir()]
    if (root / 'player').is_dir(): directories.append(root / 'player')
    for directory in sorted(directories):
        rows = [r for path in directory.glob('*.jsonl') for r in records(path)]
        games, events = [], {}
        for path in directory.glob('game-*.json'):
            state = json.loads(path.read_text())
            games.append({'game_id': state.get('game_id'), 'status': state['status'],
                          'hand': state.get('round_no'), 'score': (state.get('final', {}).get('snapshot') or {}).get('scores')})
            for e in state.get('events', []): events[(state.get('game_id'), e['seq'])] = e
        errors = [r for r in rows if not r['accepted']]
        waits = [r['request_ms'] for r in rows if r.get('request_ms') is not None]
        decisions = [r['decision_ms'] for r in rows]
        seats = {r['game_id']: r['snapshot']['seat'] for r in rows}
        def percentile(values, p):
            if not values: return None
            return sorted(values)[min(len(values) - 1, int((len(values)-1) * p))]
        result['players'].append({'player': directory.name, 'games': games,
            'actions': len(rows), 'accepted': len(rows)-len(errors),
            'action_types': dict(Counter(r['action']['action'] for r in rows)),
            'errors': dict(Counter(f"{r['action']['action']}:{r.get('error_code')}" for r in errors)),
            'decision_p95_ms': percentile(decisions, .95), 'request_p95_ms': percentile(waits, .95),
            'observed_discard_timeouts': sum(e['type']=='timeout' and (e.get('data') or {}).get('kind')=='discard' for e in events.values()),
            'observed_own_discard_timeouts': sum(e['type']=='timeout' and (e.get('data') or {}).get('kind')=='discard'
                                               and e.get('seat') == seats.get(gid) for (gid, _), e in events.items())})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--output')
    args = p.parse_args()
    result = inspect(Path(args.run))
    value = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(value)
    print(value)


if __name__ == '__main__': main()
