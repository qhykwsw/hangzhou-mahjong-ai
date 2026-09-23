#!/usr/bin/env python3
"""Freeze runtime and execute one public matchmaking session without iteration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def model_service(root, remaining, env):
    """A model policy plays through a local inference service, loaded before the room.

    The exported weights are copied into the run directory and served from
    there; the service address and the encoder location go to the player.
    """
    if "--policy" not in remaining:
        return None
    policy = Path(remaining[remaining.index("--policy") + 1])
    if not policy.is_file() or json.loads(policy.read_text()).get("family") != "model":
        return None
    weights = json.loads(policy.read_text())["parameters"]["weights"]
    models = root / "models"
    if not (models / weights).exists():
        shutil.copytree(policy.parent / "models" / weights, models / weights)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mahjong_ai.execution.model_service import ModelService

    service = ModelService(root=models, log=root / "model-server.log")
    env["MAHJONG_MODEL_SERVER"] = service.url()
    env["MAHJONG_SELFPLAY"] = str(service.project)
    service.warm([weights])
    return service


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    args, remaining = parser.parse_known_args()
    root = Path(args.output).resolve()
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
    env = dict(os.environ, PYTHONPATH=str(root / "runtime"))
    service = model_service(root, remaining, env)
    try:
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "mahjong_ai.platform.free_play",
                "--output",
                str(root),
                *remaining,
            ],
            env=env,
        )
    finally:
        if service:
            service.close()
    if code or "--help" in remaining or "-h" in remaining:
        return code
    return subprocess.call(
        [
            sys.executable,
            str(Path(__file__).with_name("audit_free_play_run.py")),
            str(root),
            *(["--insecure"] if "--insecure" in remaining else []),
            *(
                ["--base-url", remaining[remaining.index("--base-url") + 1]]
                if "--base-url" in remaining
                else []
            ),
        ],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
