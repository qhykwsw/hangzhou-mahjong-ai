"""Immutable rule teachers, selected only from completed research competitions.

This module intentionally has no mahjong_ai imports at module scope. Spawned
workers must bind the bundle runtime before importing any engine or policy SDK.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid
import weakref


BRANCHES = (
    "heuristic-efficiency--01",
    "heuristic-fan--01",
    "heuristic-adaptive--01",
    "heuristic-unrestricted--01",
)
# Runtime parts that never affect how a frozen rule candidate plays: the platform
# client, project MCP and service interfaces, plus per-machine native builds whose
# C source is itself compared. Runtimes equal outside these may share one bundle.
NON_GAMEPLAY_PREFIXES = (
    "mahjong_ai/platform/",
    "mahjong_ai/project_mcp/",
    "mahjong_ai/interfaces/",
)
NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")
_BOUND_RUNTIME: Path | None = None
_BOUND_RUNTIME_HASH: str | None = None


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
        or value in (".", "..")
    ):
        raise ValueError("invalid artifact identifier")
    return value


def _path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("bundle paths must be relative POSIX paths")
    part = Path(relative)
    if (
        not relative
        or part.is_absolute()
        or any(p in (".", "..") for p in relative.split("/"))
    ):
        raise ValueError("bundle path escapes its root")
    path = root
    for component in part.parts:
        path = path / component
        if path.is_symlink():
            raise ValueError("symbolic links are not allowed in bundles")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("bundle path escapes its root")
    return path


def _source_tree(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("source must be a directory without symbolic links")
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError("symbolic links are not allowed in bundles")
    return path.resolve()


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symbolic links are not allowed in bundles")
        if path.is_file() and "__pycache__" not in path.parts:
            result[path.relative_to(root).as_posix()] = _hash(path)
    return result


def gameplay_inventory(inventory):
    """The part of a runtime inventory that can change how candidates play."""
    return {
        key: value
        for key, value in inventory.items()
        if not key.startswith(NON_GAMEPLAY_PREFIXES)
        and not key.endswith(NATIVE_SUFFIXES)
    }


def _atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def export_bundle(project_root, output_root) -> Path:
    """Publish the latest completed four-branch roster without writing research state.

    output_root is normally projects/personal/selfplay/.local/teachers. An
    existing current.json is replaced only after a complete verified export.
    """
    project = Path(project_root).resolve()
    database = project / ".local/service/research.sqlite"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT state FROM registry").fetchone()
    if not row:
        raise ValueError("research registry is empty")
    state = json.loads(row[0])
    completed = [c for c in state.get("cycles", []) if c.get("status") == "completed"]
    if not completed:
        raise ValueError("no completed research competition")
    cycle = completed[-1]
    ranked = [r for r in cycle.get("ranking", []) if r.get("branch") in BRANCHES]
    if len(ranked) != 4 or {r["branch"] for r in ranked} != set(BRANCHES):
        raise ValueError(
            "latest completed competition must contain all four rule branches"
        )
    if len({r["candidate_id"] for r in ranked}) != 4:
        raise ValueError("rule branches must refer to four distinct candidate versions")

    # Export runs in the host process; consumers bind a frozen SDK in a fresh worker.
    from mahjong_ai.strategies.common import PolicySpec

    entries, sources, runtime_paths, source_files, originals = [], {}, {}, {}, []
    for ranking in ranked:
        cid = _name(ranking["candidate_id"])
        candidate = state.get("candidates", {}).get(cid)
        if (
            not candidate
            or candidate.get("branch") != ranking["branch"]
            or not candidate.get("spec")
        ):
            raise ValueError("completed ranking references an unavailable candidate")
        original = PolicySpec(**candidate["spec"])
        package = _source_tree(original.package_path)
        runtime = _source_tree(original.runtime_path)
        identity = original.identity  # Verify the original immutable candidate first.
        originals.append((original, identity))
        runtime_name = _name(runtime.name)
        runtime_hash = _digest(_inventory(runtime))
        if (
            runtime_name in runtime_paths
            and runtime_paths[runtime_name][1] != runtime_hash
        ):
            raise ValueError("runtime name refers to different source contents")
        runtime_paths[runtime_name] = (runtime, runtime_hash)
        package_relative = f"candidates/{cid}"
        entries.append(
            dict(
                id=cid,
                branch=ranking["branch"],
                path=package_relative,
                original_runtime=runtime_name,
                original_runtime_hash=runtime_hash,
                source_identity=identity,
                package_hash=original.package_hash,
            )
        )
        sources[package_relative] = package
        source_files[package_relative] = _inventory(package)
    # Candidates carried into a newer research experiment keep their older runtime.
    # They share one bundle runtime only when the runtimes play identically.
    gameplay = {
        name: gameplay_inventory(_inventory(path))
        for name, (path, _) in runtime_paths.items()
    }
    if len({_digest(value) for value in gameplay.values()}) != 1:
        raise ValueError(
            "mixed frozen runtimes differ in gameplay files; current teacher remains unchanged"
        )
    current = Path(state.get("runtime_path") or "").name
    chosen = current if current in runtime_paths else entries[0]["original_runtime"]
    chosen_path, chosen_hash = runtime_paths[chosen]
    runtime_relative = f"runtimes/{chosen}"
    for entry in entries:
        entry.update(runtime=runtime_relative, runtime_hash=chosen_hash)
    sources[runtime_relative] = chosen_path
    source_files[runtime_relative] = _inventory(chosen_path)
    if not (chosen_path / "mahjong_ai/__init__.py").is_file():
        raise ValueError("frozen runtime does not contain mahjong_ai")

    output = Path(output_root)
    if output.is_symlink():
        raise ValueError("teacher output directory must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
    try:
        for relative, source in sources.items():
            shutil.copytree(
                source, staging / relative, ignore=shutil.ignore_patterns("__pycache__")
            )
            if _inventory(staging / relative) != source_files[relative]:
                raise ValueError("source changed while exporting immutable bundle")
        if any(original.identity != identity for original, identity in originals):
            raise ValueError(
                "original identity changed while exporting immutable bundle"
            )
        manifest = dict(
            teacher=entries[0]["id"],
            entries=entries,
            runtime=runtime_relative,
            gameplay_hash=_digest(gameplay[chosen]),
            source_cycle=cycle["id"],
            source_experiment=cycle["experiment"],
            files=_inventory(staging),
        )
        manifest["id"] = _digest(manifest)
        _atomic_json(staging / "manifest.json", manifest)
        read_bundle(staging / "manifest.json")
        destination = output / manifest["id"]
        if destination.exists():
            existing = read_bundle(destination / "manifest.json")
            if existing["id"] != manifest["id"]:
                raise ValueError("existing teacher artifact has a conflicting identity")
        else:
            os.rename(staging, destination)
        _atomic_json(
            output / "current.json",
            {
                "id": manifest["id"],
                "manifest": f"{manifest['id']}/manifest.json",
            },
        )
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_bundle(path, verify=True) -> dict:
    """Validate and relocate a manifest; original source checksums remain relative.

    Additional __pycache__ files and newly compiled native extensions beneath
    the frozen runtime are allowed. Original native files remain checksummed.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symbolic links are not allowed in bundles")
    if path.is_dir():
        path = path / "manifest.json"
    if path.parent.is_symlink():
        raise ValueError("symbolic links are not allowed in bundles")
    manifest = json.loads(path.read_text())
    if "manifest" in manifest and "entries" not in manifest:
        target = _path(path.parent.resolve(), manifest["manifest"])
        result = read_bundle(target, verify=verify)
        if result["id"] != manifest.get("id"):
            raise ValueError("current pointer and manifest disagree")
        return result
    root = path.parent.resolve()
    if manifest.get("id") != _digest({k: v for k, v in manifest.items() if k != "id"}):
        raise ValueError("manifest identity mismatch")
    entries = manifest.get("entries", [])
    if len(entries) != 4 or {e.get("branch") for e in entries} != set(BRANCHES):
        raise ValueError("bundle must contain exactly the four rule branches")
    ids = {e["id"] for e in entries}
    if len(ids) != 4 or manifest.get("teacher") not in ids:
        raise ValueError("invalid teacher or duplicate candidate identity")
    runtime = _path(root, manifest["runtime"])
    if not (runtime / "mahjong_ai/__init__.py").is_file():
        raise ValueError("frozen runtime does not contain mahjong_ai")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("bundle has no source checksums")
    prefix = manifest["runtime"] + "/"
    runtime_hash = _digest(
        {
            key[len(prefix) :]: value
            for key, value in files.items()
            if key.startswith(prefix)
        }
    )
    if any(entry.get("runtime_hash") != runtime_hash for entry in entries):
        raise ValueError("runtime source identity disagrees with source checksums")
    for relative, expected in files.items():
        source = _path(root, relative)
        if not source.is_file():
            raise ValueError(f"missing source file: {relative}")
        if verify and _hash(source) != expected:
            raise ValueError(f"source checksum mismatch: {relative}")
    for source in root.rglob("*"):
        if source.is_symlink():
            raise ValueError("symbolic links are not allowed in bundles")
        if (
            not source.is_file()
            or source == path.resolve()
            or "__pycache__" in source.parts
        ):
            continue
        relative = source.relative_to(root).as_posix()
        if relative not in files and not (
            source.is_relative_to(runtime)
            and source.suffix in (".so", ".pyd", ".dll", ".dylib")
        ):
            raise ValueError(f"unrecorded source file: {relative}")
    relocated = []
    for entry in entries:
        if entry["runtime"] != manifest["runtime"]:
            raise ValueError("mixed frozen runtimes cannot be enabled")
        package = _path(root, entry["path"])
        if not (package / "strategy.py").is_file():
            raise ValueError("candidate package has no strategy.py")
        relocated.append({**entry, "path": str(package), "runtime": str(runtime)})
    return {
        **manifest,
        "entries": relocated,
        "runtime": str(runtime),
        "runtime_hash": runtime_hash,
        "manifest_path": str(path.resolve()),
        "root": str(root),
    }


def bind_runtime(bundle) -> None:
    """Bind once per process; never replace an already imported different SDK."""
    global _BOUND_RUNTIME, _BOUND_RUNTIME_HASH
    runtime = Path(bundle["runtime"]).resolve()
    runtime_hash = bundle["runtime_hash"]
    if _BOUND_RUNTIME_HASH is not None and _BOUND_RUNTIME_HASH != runtime_hash:
        raise RuntimeError("worker is already bound to a different frozen runtime")
    active = _BOUND_RUNTIME or runtime
    for name, module in tuple(sys.modules.items()):
        if name != "mahjong_ai" and not name.startswith("mahjong_ai."):
            continue
        source = getattr(module, "__file__", None)
        if source is None or not Path(source).resolve().is_relative_to(active):
            raise RuntimeError(
                "mahjong_ai was imported from a different runtime; spawn a fresh worker"
            )
    if str(active) in sys.path:
        sys.path.remove(str(active))
    sys.path.insert(0, str(active))
    _BOUND_RUNTIME = active
    _BOUND_RUNTIME_HASH = runtime_hash


def build_policy(bundle, candidate_id):
    """Build a fresh in-process rule policy under its already-bound frozen SDK."""
    if _BOUND_RUNTIME_HASH is None or _BOUND_RUNTIME_HASH != bundle["runtime_hash"]:
        raise RuntimeError("bind_runtime must run before building a teacher policy")
    bind_runtime(bundle)  # Detect intervening SDK imports from a different tree.
    entry = next((e for e in bundle["entries"] if e["id"] == candidate_id), None)
    if entry is None:
        raise ValueError("candidate is not in this teacher bundle")
    return load_package(entry["path"])


def bound_runtime_name():
    """Directory name of the bound frozen runtime, i.e. its fingerprint."""
    return Path(_BOUND_RUNTIME).name if _BOUND_RUNTIME else None


def load_package(package):
    """Build a rule policy from a frozen package under the already-bound SDK."""
    if _BOUND_RUNTIME_HASH is None:
        raise RuntimeError("bind_runtime must run before building a rule policy")
    package = Path(package)
    name = "_guided_rule_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(
        name,
        package / "strategy.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Candidate contracts permit sibling imports. Keep this insertion temporary
    # and prevent a previous teacher's helper module being reused accidentally.
    inserted = str(package) not in sys.path
    if inserted:
        sys.path.insert(0, str(package))
    before = set(sys.modules)
    try:
        spec.loader.exec_module(module)
        policy = module.build(package / "model")
        if not all(
            callable(getattr(policy, method, None))
            for method in ("reset", "act", "finish_round")
        ):
            raise TypeError("teacher must implement reset, act and finish_round")
        try:
            weakref.finalize(policy, _release_module, name)
        except TypeError:
            # Slot-only policies may not expose a weakref slot. Their methods
            # keep the loaded globals alive without retaining a global module.
            _release_module(name)
        return policy
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        if inserted:
            sys.path.remove(str(package))
        for imported in set(sys.modules) - before:
            helper = sys.modules.get(imported)
            source = getattr(helper, "__file__", None)
            if (
                source
                and Path(source).resolve().is_relative_to(package)
                and not imported.startswith(name + ".")
            ):
                sys.modules.pop(imported, None)


def _release_module(name):
    for key in tuple(sys.modules):
        if key == name or key.startswith(name + "."):
            sys.modules.pop(key, None)
