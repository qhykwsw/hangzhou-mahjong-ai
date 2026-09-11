"""Export an immutable arena entrant and its matching runtime for platform use."""

import shutil
from pathlib import Path
from mahjong_ai.research.service import Research
from mahjong_ai.strategies.common import PolicySpec
from mahjong_ai.persistence.artifacts import atomic_json


def export_candidate(project, candidate, output):
    arena = Research(project)
    state = arena.read()
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
    shutil.copytree(original.runtime_path, destination / "runtime")
    value = original.as_dict() | {
        "package_path": "candidate",
        "python_path": None,
        "runtime_path": "runtime",
    }
    atomic_json(destination / "policy.json", value)
    atomic_json(
        destination / "manifest.json",
        {
            "candidate": candidate,
            "branch": row["branch"],
            "experiment": row.get("experiment"),
            "runtime": Path(original.runtime_path).name,
            "requirements": row.get("requirements", []),
            "scope": "local competition result; external platform has separate records",
        },
    )
    return {"policy": str(destination / "policy.json"), "candidate": candidate}
