"""Repository paths only; importing this module must not import torch or the SDK."""

import sys
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
for path in (PROJECT, PROJECT / "src", REPO / "src"):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def verify_source():
    """Check a deployed snapshot; development checkouts have no manifest."""
    path = REPO / "source-manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    files = manifest["files"]
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity != manifest["id"]:
        raise ValueError("source manifest identity mismatch")
    for name, expected in files.items():
        source = REPO / name
        if source.is_symlink() or not source.resolve().is_relative_to(REPO):
            raise ValueError("source path escapes the frozen snapshot")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen source changed: {name}")
    return identity
