#!/usr/bin/env python3
"""Export reproducible, developer-visible campaign aggregates, without eval seeds."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('campaign')
    p.add_argument('--output', required=True)
    p.add_argument('--followup')
    args = p.parse_args()
    root = Path(args.campaign)
    run = json.loads((root / 'experiments/runs/campaign/run.json').read_text())
    rows = []
    for index, outcome in enumerate(run['iterations']):
        feedback = outcome['feedback']
        proposal = json.loads((root / f'experiments/runs/campaign/iteration-{index:04}/proposal.json').read_text())
        rows.append({'index': index, 'family': outcome['family'], 'status': outcome['status'],
                     'proposal': proposal, 'validation': feedback.get('validation'),
                     'behavior': feedback.get('behavior'), 'training': feedback.get('training')})
    result = {'status': run['status'], 'protocol': run['protocol'], 'iterations': rows,
              'note': 'Local actual-match experiments; not official platform scores.'}
    if args.followup:
        result['followup'] = json.loads(Path(args.followup).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'status': result['status'], 'iterations': len(rows), 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__': main()
