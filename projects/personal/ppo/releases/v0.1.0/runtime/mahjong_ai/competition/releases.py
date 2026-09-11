"""Portable immutable releases, independent of automatic research state."""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time

from mahjong_ai.execution.candidate_runtime import tree_hash
from mahjong_ai.persistence.artifacts import (
    atomic_json,
    digest,
    exclusive_lock,
    file_hash,
)
from mahjong_ai.project import ProjectDescriptor
from mahjong_ai.strategies.common import PolicySpec


def _files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("release cannot contain symlinks")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path != root / "manifest.json"
        ):
            result[str(path.relative_to(root))] = file_hash(path)
    return result


def verify_release(path):
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text())
    files = _files(root)
    identity = {
        k: v for k, v in manifest.items() if k not in ("content_hash", "entrant_id")
    }
    if files != manifest["files"] or digest(identity) != manifest["content_hash"]:
        raise ValueError("frozen release has changed")
    if (
        manifest["entrant_id"]
        != f"{manifest['project_id']}@{manifest['version']}#{manifest['content_hash']}"
    ):
        raise ValueError("release identity does not match its manifest")
    return manifest


def _interpreter(root, manifest):
    requirements = manifest.get("dependencies", [])
    if not requirements:
        return sys.executable
    environment = Path.home() / ".cache/mahjong/releases" / manifest["content_hash"]
    with exclusive_lock(environment.with_suffix(".lock"), blocking=True):
        python = environment / "bin/python"
        if not (environment / "complete.json").exists():
            if environment.exists():
                shutil.rmtree(environment)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(environment),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    str(root / "wheels"),
                    *requirements,
                ],
                check=True,
                capture_output=True,
            )
            atomic_json(
                environment / "complete.json", {"release": manifest["content_hash"]}
            )
    return str(python)


def load_policy(path):
    if str(path) == "rule":
        return PolicySpec()
    if str(path) == "rl":
        return PolicySpec("rl", "rl-initialized")
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "policy.json"
    root = source.parent
    manifest = None
    if (root / "manifest.json").exists():
        metadata = json.loads((root / "manifest.json").read_text())
        if metadata.get("schema") == "strategy-release-v1":
            manifest = verify_release(root)
    data = json.loads(source.read_text())
    for key in ("model_path", "package_path", "python_path", "runtime_path"):
        if data.get(key) and not Path(data[key]).is_absolute():
            data[key] = str((root / data[key]).resolve())
    if manifest:
        data["python_path"] = _interpreter(root, manifest)
    return PolicySpec(**data)


def build_release(project, version, *, parent=None):
    descriptor = ProjectDescriptor.load(project)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", version):
        raise ValueError("invalid release version")
    entrypoint = descriptor.root / (descriptor.entrypoint or "src/strategy.py")
    if not entrypoint.is_file() or entrypoint.name != "strategy.py":
        raise ValueError("release entrypoint must be an existing strategy.py")
    if not entrypoint.resolve().is_relative_to(descriptor.root):
        raise ValueError("entrypoint must be inside the project")
    for requirement in descriptor.dependencies:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", requirement):
            raise ValueError("dependencies require exact name==version")
    destination = descriptor.releases / version
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(destination.parent / ".build.lock", blocking=True):
        if destination.exists():
            raise ValueError("release versions cannot be overwritten")
        temporary = Path(tempfile.mkdtemp(prefix=".build-", dir=destination.parent))
        try:
            shutil.copytree(
                entrypoint.parent,
                temporary / "candidate",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            models = descriptor.root / "models"
            if models.exists():
                shutil.copytree(
                    models, temporary / "candidate/model", dirs_exist_ok=True
                )
            shutil.copytree(
                Path(__file__).parents[1],
                temporary / "runtime/mahjong_ai",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            wheels = {}
            python = sys.executable
            if descriptor.dependencies:
                from mahjong_ai.execution.dependencies import prepare_environment

                environment = (
                    descriptor.local
                    / "environments"
                    / digest(list(descriptor.dependencies))
                )
                python, wheels = prepare_environment(
                    list(descriptor.dependencies), environment, time.monotonic() + 900
                )
                shutil.copytree(environment / "wheels", temporary / "wheels")
            spec = PolicySpec(
                family="code",
                version=f"{descriptor.id}@{version}",
                package_path="candidate",
                package_hash=tree_hash(temporary / "candidate"),
                runtime_path="runtime",
            )
            atomic_json(temporary / "policy.json", spec.as_dict())
            from mahjong_ai.execution.candidate_runtime import CandidateProcess

            probe = CandidateProcess(
                temporary / "candidate", python, runtime=str(temporary / "runtime")
            )
            try:
                probe.call({"op": "build"}, 60)
            finally:
                probe.close()
            metadata = {
                "schema": "strategy-release-v1",
                "project_id": descriptor.id,
                "version": version,
                "family": descriptor.family,
                "parent": parent,
                "dependencies": list(descriptor.dependencies),
                "wheels": wheels,
                "runtime": tree_hash(temporary / "runtime"),
                "build": {"python": sys.version, "validated": True},
                "files": _files(temporary),
            }
            content_hash = digest(metadata)
            metadata.update(
                content_hash=content_hash,
                entrant_id=f"{descriptor.id}@{version}#{content_hash}",
            )
            atomic_json(temporary / "manifest.json", metadata)
            verify_release(temporary)
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return {"path": str(destination), **metadata}
