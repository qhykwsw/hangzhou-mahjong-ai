#!/usr/bin/env python3
"""Freeze runtime and play a registered tournament with one frozen strategy.

The scoped token comes from a credentials file ``{"token": "..."}``; the runner
registers and readies, keeps polling so the entrant counts as online when the
stage freezes, plays every assigned game and exits when the tournament ends.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--policy", required=True, help="exported policy.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=12 * 3600)
    parser.add_argument("--base-url", default="https://10.240.169.190:18080")
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    runtime = root / "runtime" / "mahjong_ai"
    marker = root / "runtime-manifest.json"
    if not runtime.exists():
        shutil.copytree(
            Path(__file__).resolve().parents[1] / "src/mahjong_ai",
            runtime,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    hashes = {
        str(f.relative_to(runtime)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in runtime.rglob("*.py")
    }
    if marker.exists():
        if json.loads(marker.read_text()) != hashes:
            raise SystemExit(
                "frozen platform runtime changed; restore it or use a new output directory"
            )
    else:
        marker.write_text(json.dumps(hashes, indent=2))
    token = json.loads(Path(args.credentials).read_text())["token"]
    env = dict(os.environ, PYTHONPATH=str(root / "runtime"), MAHJONG_TOKEN=token)
    command = [
        sys.executable,
        "-c",
        "import sys; from mahjong_ai.interfaces.cli import main; sys.exit(main())",
        "online-play",
        "--base-url", args.base_url,
        "--policy", str(Path(args.policy).resolve()),
        "--output", str(root / "play"),
        "--seconds", str(args.seconds),
        "--join",
    ]
    if args.insecure:
        command.append("--insecure")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
