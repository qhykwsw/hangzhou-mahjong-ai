#!/usr/bin/env python3
"""Freeze platform runtime, execute four-token games, then audit actual results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    args, remaining = parser.parse_known_args()
    root = Path(args.output).resolve()
    runtime = root / 'runtime' / 'mahjong_ai'
    marker = root / 'runtime-manifest.json'
    if not runtime.exists():
        runtime.mkdir(parents=True)
        for file in (Path(__file__).resolve().parents[1] / 'src/mahjong_ai').glob('*.py'):
            shutil.copy2(file, runtime / file.name)
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in runtime.glob('*.py')}
    if marker.exists():
        if json.loads(marker.read_text()) != hashes:
            raise SystemExit('frozen platform runtime changed; restore it or use a new output directory')
    else: marker.write_text(json.dumps(hashes, indent=2))
    env = dict(os.environ, PYTHONPATH=str(root / 'runtime'))
    code = subprocess.call([sys.executable, '-m', 'mahjong_ai.live_workflow',
                            '--output', str(root), *remaining], env=env)
    if code or '--capture-only' in remaining: return code
    return subprocess.call([sys.executable, str(Path(__file__).with_name('audit_platform_run.py')),
                            str(root), '--output', str(root/'audit.json')], env=env)


if __name__ == '__main__': raise SystemExit(main())
