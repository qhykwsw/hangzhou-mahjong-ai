"""Export an immutable arena entrant and its matching runtime for platform use."""

import shutil
from pathlib import Path
from mahjong_ai.research.service import Research
from mahjong_ai.strategies.common import PolicySpec
from mahjong_ai.persistence.artifacts import atomic_json


def export_candidate(root, candidate, output):
    root = Path(root).resolve()
    arena = Research(root.parents[2])
    state = arena.read(root.name)
    row = state["candidates"].get(candidate)
    if not row or not row.get("spec"):
        raise ValueError("select a prepared candidate by its full ID")
    original = PolicySpec(**row["spec"])
    original.identity  # verify immutable source
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("export destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original.package_path, destination / "candidate")
    shutil.copytree(root / "runtime", destination / "runtime")
    value = original.as_dict() | {"package_path": "candidate", "python_path": None}
    atomic_json(destination / "policy.json", value)
    atomic_json(
        destination / "manifest.json",
        {
            "candidate": candidate,
            "run": root.name,
            "runtime": state["runtime"],
            "requirements": row.get("requirements", []),
            "scope": "local competition result; external platform has separate records",
        },
    )
    return {"policy": str(destination / "policy.json"), "candidate": candidate}
