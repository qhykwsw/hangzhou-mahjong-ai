"""Teacher publication preserves completed evidence and frozen source identity."""

import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest


MODULE = (
    Path(__file__).resolve().parents[1]
    / "projects/personal/selfplay/guided/artifacts.py"
)
spec = importlib.util.spec_from_file_location("guided_artifacts_tests", MODULE)
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


@pytest.fixture
def research(tmp_path, monkeypatch):
    from mahjong_ai.strategies import common

    verified = []

    class SourceSpec:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        @property
        def identity(self):
            verified.append(self.version)
            return "original-" + self.version

    monkeypatch.setattr(common, "PolicySpec", SourceSpec)
    root = tmp_path / "research"
    runtime = root / ".local/research/runtimes/frozen-sdk"
    (runtime / "mahjong_ai").mkdir(parents=True)
    (runtime / "mahjong_ai/__init__.py").write_text("FROZEN = True\n")
    candidates, ranking = {}, []
    for index, branch in enumerate(artifacts.BRANCHES):
        cid = f"candidate-{index}"
        package = root / ".local/strategies" / cid
        package.mkdir(parents=True)
        (package / "strategy.py").write_text(
            "from mahjong_ai import FROZEN\n"
            "class Policy:\n"
            "    def __init__(self): self.calls = 0\n"
            "    def reset(self, seed): self.calls = 0\n"
            "    def act(self, observation): self.calls += 1; return self.calls\n"
            "    def finish_round(self, result, seat): pass\n"
            "def build(model_dir): return Policy()\n"
        )
        candidates[cid] = dict(
            branch=branch,
            spec=dict(
                version=cid,
                package_path=str(package),
                runtime_path=str(runtime),
                package_hash="frozen-" + cid,
            ),
        )
        ranking.append(dict(branch=branch, candidate_id=cid))
    state = dict(
        candidates=candidates,
        cycles=[
            dict(id=1, experiment="experiment-1", status="completed", ranking=ranking),
            dict(
                id=2,
                experiment="experiment-1",
                status="competing",
                ranking=list(reversed(ranking)),
            ),
        ],
        last_ranking=list(reversed(ranking)),
    )
    database = root / ".local/service/research.sqlite"
    database.parent.mkdir(parents=True)

    def save():
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS registry (state TEXT)")
            connection.execute("DELETE FROM registry")
            connection.execute("INSERT INTO registry VALUES (?)", (json.dumps(state),))

    save()
    return root, tmp_path / "teachers", state, save, verified


def test_completed_roster_wins_over_temporary_and_live_rankings(research):
    root, output, _, _, verified = research
    database = root / ".local/service/research.sqlite"
    before = database.read_bytes()
    path = artifacts.export_bundle(root, output)
    result = artifacts.read_bundle(path)
    assert result["teacher"] == "candidate-0"
    assert result["source_cycle"] == 1
    assert result["source_experiment"] == "experiment-1"
    assert verified == [f"candidate-{i}" for i in range(4)] * 2
    assert result["entries"][0]["source_identity"] == "original-candidate-0"
    assert all(Path(entry["path"]).is_absolute() for entry in result["entries"])
    assert artifacts.read_bundle(output / "current.json")["id"] == result["id"]
    assert artifacts.export_bundle(root, output) == path
    assert database.read_bytes() == before


def test_latest_completed_must_contain_all_four_branches(research):
    root, output, state, save, _ = research
    path = artifacts.export_bundle(root, output)
    current = (output / "current.json").read_bytes()
    state["cycles"].append(
        dict(id=3, experiment="experiment-1", status="completed", ranking=[])
    )
    save()
    with pytest.raises(ValueError, match="all four"):
        artifacts.export_bundle(root, output)
    assert (output / "current.json").read_bytes() == current
    assert path.exists()


def test_mixed_runtime_refused_without_replacing_current(research):
    root, output, state, save, _ = research
    artifacts.export_bundle(root, output)
    before = (output / "current.json").read_bytes()
    candidate = state["candidates"]["candidate-1"]
    original = Path(candidate["spec"]["runtime_path"])
    other = original.with_name("another-sdk")
    shutil.copytree(original, other)
    (other / "mahjong_ai/rules.py").write_text("CHANGED = True\n")
    candidate["spec"]["runtime_path"] = str(other)
    save()
    with pytest.raises(ValueError, match="mixed frozen runtimes"):
        artifacts.export_bundle(root, output)
    assert (output / "current.json").read_bytes() == before


def test_runtimes_differing_only_outside_gameplay_share_the_current_runtime(research):
    root, output, state, save, _ = research
    candidate = state["candidates"]["candidate-1"]
    original = Path(candidate["spec"]["runtime_path"])
    newer = original.with_name("newer-sdk")
    shutil.copytree(original, newer)
    (newer / "mahjong_ai/platform").mkdir()
    (newer / "mahjong_ai/platform/online.py").write_text("CLIENT = 2\n")
    (newer / "mahjong_ai/_fast.cpython-312-darwin.so").write_bytes(b"local build")
    candidate["spec"]["runtime_path"] = str(newer)
    state["runtime_path"] = str(newer)
    save()
    result = artifacts.read_bundle(artifacts.export_bundle(root, output))
    assert Path(result["runtime"]).name == "newer-sdk"
    assert {e["original_runtime"] for e in result["entries"]} == {
        "frozen-sdk",
        "newer-sdk",
    }
    assert {e["runtime"] for e in result["entries"]} == {result["runtime"]}


def test_copy_failure_does_not_publish_or_leave_partial_bundle(research, monkeypatch):
    root, output, _, _, _ = research
    artifacts.export_bundle(root, output)
    before = (output / "current.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("interrupted copy")

    monkeypatch.setattr(artifacts.shutil, "copytree", fail)
    with pytest.raises(OSError, match="interrupted"):
        artifacts.export_bundle(root, output)
    assert (output / "current.json").read_bytes() == before
    assert not list(output.glob(".export-*"))


@pytest.mark.parametrize("change", ["missing", "modified", "extra"])
def test_source_tampering_rejected(research, change):
    root, output, _, _, _ = research
    manifest = artifacts.export_bundle(root, output)
    package = manifest.parent / "candidates/candidate-0"
    if change == "missing":
        (package / "strategy.py").unlink()
    elif change == "modified":
        (package / "strategy.py").write_text("tampered = True\n")
    else:
        (package / "extra.py").write_text("injected = True\n")
    with pytest.raises(ValueError, match="source"):
        artifacts.read_bundle(manifest)


def test_manifest_tampering_rejected(research):
    root, output, _, _, _ = research
    manifest = artifacts.export_bundle(root, output)
    value = json.loads(manifest.read_text())
    value["teacher"] = "candidate-3"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest identity"):
        artifacts.read_bundle(manifest)


@pytest.mark.parametrize(
    "relative", ["../outside", "/tmp/outside", "candidates/../../outside"]
)
def test_paths_cannot_escape_bundle(research, relative):
    root, output, _, _, _ = research
    manifest = artifacts.export_bundle(root, output)
    value = json.loads(manifest.read_text())
    value["entries"][0]["path"] = relative
    value["id"] = artifacts._digest({k: v for k, v in value.items() if k != "id"})
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="escapes"):
        artifacts.read_bundle(manifest)


def test_symlinks_rejected_even_in_cache(research):
    root, output, _, _, _ = research
    manifest = artifacts.export_bundle(root, output)
    cache = manifest.parent / "__pycache__"
    cache.mkdir()
    (cache / "escape").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        artifacts.read_bundle(manifest)


def test_relocation_and_added_native_extension_preserve_source_identity(
    research, tmp_path
):
    root, output, _, _, _ = research
    manifest = artifacts.export_bundle(root, output)
    moved = tmp_path / "moved"
    shutil.copytree(manifest.parent, moved)
    runtime = moved / "runtimes/frozen-sdk/mahjong_ai"
    (runtime / "_fast.cpython-312-linux.so").write_bytes(b"local build")
    (runtime / "__pycache__").mkdir()
    (runtime / "__pycache__/init.pyc").write_bytes(b"cache")
    value = artifacts.read_bundle(moved)
    assert value["id"] == manifest.parent.name
    assert value["runtime"] == str(runtime.parent)
    assert (
        "runtimes/frozen-sdk/mahjong_ai/_fast.cpython-312-linux.so"
        not in value["files"]
    )


def test_four_identical_teacher_seats_are_independent_in_bound_worker(
    research, tmp_path
):
    root, output, _, _, _ = research
    manifest = artifacts.export_bundle(root, output)
    panel = tmp_path / "fixed-panel"
    shutil.copytree(manifest.parent, panel)
    code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('artifacts', sys.argv[1])
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
assert 'mahjong_ai' not in sys.modules
b = a.read_bundle(sys.argv[2]); a.bind_runtime(b)
panel = a.read_bundle(sys.argv[3]); a.bind_runtime(panel)
policies = [a.build_policy(b if n == 0 else panel, b['teacher']) for n in range(4)]
assert len({id(p) for p in policies}) == 4
policies[0].act(None)
assert [p.calls for p in policies] == [1, 0, 0, 0]
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(MODULE), str(manifest), str(panel)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_binding_refuses_already_imported_host_sdk(research):
    root, output, _, _, _ = research
    manifest = artifacts.export_bundle(root, output)
    with pytest.raises(RuntimeError, match="different runtime"):
        artifacts.bind_runtime(artifacts.read_bundle(manifest))


def test_build_requires_binding(research):
    root, output, _, _, _ = research
    bundle = artifacts.read_bundle(artifacts.export_bundle(root, output))
    with pytest.raises(RuntimeError, match="bind_runtime"):
        artifacts.build_policy(bundle, bundle["teacher"])
