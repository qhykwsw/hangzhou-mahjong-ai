#!/usr/bin/env python3
"""Freeze code once, then run a resumable two-family Agent campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--iterations-per-family', type=int, default=3)
    p.add_argument('--training-rounds', type=int, default=128)
    p.add_argument('--evaluation-rounds', type=int, default=2000)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--games-per-round', type=int, default=1)
    p.add_argument('--seconds', type=float, default=1200)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=901)
    p.add_argument('--generator', choices=['codex', 'search'], default='codex')
    args = p.parse_args()
    root = Path(args.output).resolve()
    source = root / 'runtime' / 'mahjong_ai'
    manifest = root / 'campaign.json'
    settings = vars(args) | {'output': str(root)}
    settings.pop('seconds')
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['settings'] != settings: p.error('campaign settings changed; use another output directory')
        if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in saved['source'].items()):
            p.error('frozen campaign source changed')
    else:
        source.mkdir(parents=True, exist_ok=True)
        for file in (Path(__file__).resolve().parents[1] / 'src' / 'mahjong_ai').glob('*.py'):
            shutil.copy2(file, source / file.name)
        saved = {'settings': settings, 'source': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')}}
        manifest.write_text(json.dumps(saved, indent=2))
    env = dict(os.environ, PYTHONPATH=str(root / 'runtime'))
    argv = [sys.executable, '-m', 'mahjong_ai.cli', 'iterate', '--root', str(root / 'experiments'),
            '--run-id', 'campaign', '--families', 'rule', 'rl', '--generator', args.generator,
            '--iterations', str(args.iterations_per_family * 2), '--training-rounds', str(args.training_rounds),
            '--evaluation-rounds', str(args.evaluation_rounds), '--rounds', str(args.rounds), '--games-per-round', str(args.games_per_round),
            '--workers', str(args.workers), '--seconds', str(args.seconds), '--seed', str(args.seed)]
    return subprocess.call(argv, env=env)


if __name__ == '__main__': raise SystemExit(main())
