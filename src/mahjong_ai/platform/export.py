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
    if original.family == "model":
        # An exported Actor: its weights directory, named by SHA-256, is what
        # an inference service loads; the policy itself is only the address.
        weights = original.parameters["weights"]
        shutil.copytree(arena.layout.research / "models" / weights, destination / "models" / weights)
        atomic_json(destination / "policy.json", original.as_dict())
        atomic_json(destination / "manifest.json", {
            "candidate": candidate,
            "branch": row["branch"],
            "family": "model",
            "weights": weights,
            "scope": "local competition result; external platform has separate records",
        })
        return {"policy": str(destination / "policy.json"), "candidate": candidate}
    # An experiment that plays every frozen package on one candidate runtime
    # evaluated this entrant there; the platform entrant plays the same way.
    runtime = state.get("candidate_runtime_path") or original.runtime_path
    shutil.copytree(original.package_path, destination / "candidate")
    shutil.copytree(runtime, destination / "runtime")
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
            "runtime": Path(runtime).name,
            "original_runtime": Path(original.runtime_path).name,
            "requirements": row.get("requirements", []),
            "scope": "local competition result; external platform has separate records",
        },
    )
    return {"policy": str(destination / "policy.json"), "candidate": candidate}
