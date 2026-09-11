import importlib.util
import json
import sqlite3
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / "scripts/migrate_strategy_project.py"
    spec = importlib.util.spec_from_file_location("migration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_source(tmp_path):
    root = tmp_path / "repo"
    (root / ".local/service").mkdir(parents=True)
    (root / ".local/research/runtimes/r1").mkdir(parents=True)
    (root / ".local/research/jobs/j1/c1").mkdir(parents=True)
    (root / ".local/strategies/combo/b1/submissions").mkdir(parents=True)
    (root / ".local/research/runtimes/r1/runtime.py").write_bytes(b"frozen-runtime")
    (root / ".local/research/environments/e/bin").mkdir(parents=True)
    (root / ".local/research/environments/e/bin/tool").write_text(
        str(root) + "/.local/research\n", encoding="utf-8"
    )
    (root / ".local/strategies/combo/b1/strategy.py").write_bytes(b"frozen-package")
    (root / ".local/research/jobs/j1/matches.json").write_bytes(b"frozen-receipt")
    (root / "research.toml").write_text("[research]\n", encoding="utf-8")
    state = {
        "id": "experiment-1",
        "status": "paused",
        "phase": "researching",
        "root": str(root / ".local/research"),
        "runtime_path": str(root / ".local/research/runtimes/r1"),
        "jobs": {"j1": {"id": "j1", "status": "paused"}},
        "candidates": {
            "c1": {
                "workspace": str(root / ".local/strategies/combo/b1"),
                "spec": {
                    "package_path": str(root / ".local/strategies/combo/b1"),
                    "runtime_path": str(root / ".local/research/runtimes/r1"),
                },
            }
        },
    }
    db = sqlite3.connect(root / ".local/service/research.sqlite")
    db.execute("CREATE TABLE registry(id INTEGER PRIMARY KEY, state TEXT NOT NULL)")
    db.execute("INSERT INTO registry VALUES(1, ?)", (json.dumps(state),))
    db.commit()
    db.close()
    return root, state


def test_dry_run_is_read_only_and_destination_layout(tmp_path):
    m = module()
    source, state = make_source(tmp_path)
    destination = source / "projects/agent-research"
    destination.mkdir(parents=True)
    (destination / "project.toml").write_text("name='agent'\n")
    before = (source / ".local/strategies/combo/b1/strategy.py").read_bytes()
    result = m.migrate(source, destination)
    assert result["status"] == "dry-run"
    assert not (destination / ".local").exists()
    assert (source / ".local/service/research.sqlite").exists()
    assert (source / ".local/strategies/combo/b1/strategy.py").read_bytes() == before


def test_apply_uses_sqlite_backup_rewrites_paths_and_preserves_frozen_bytes(tmp_path):
    m = module()
    source, _ = make_source(tmp_path)
    destination = source / "projects/agent-research"
    destination.mkdir(parents=True)
    (destination / "project.toml").write_text("name='agent'\n")
    frozen = (source / ".local/strategies/combo/b1/strategy.py").read_bytes()
    receipt = (source / ".local/research/jobs/j1/matches.json").read_bytes()
    result = m.migrate(source, destination, apply=True)
    assert result["status"] == "migrated" and result["source_retained"]
    assert (destination / "project.toml").read_text() == "name='agent'\n"
    assert (
        destination / ".local/strategies/combo/b1/strategy.py"
    ).read_bytes() == frozen
    assert (
        destination / ".local/research/jobs/j1/matches.json"
    ).read_bytes() == receipt
    db = sqlite3.connect(
        f"file:{destination / '.local/service/research.sqlite'}?mode=ro", uri=True
    )
    moved = json.loads(
        db.execute("SELECT state FROM registry WHERE id=1").fetchone()[0]
    )
    db.close()
    assert moved["root"] == str(destination / ".local/research")
    assert moved["candidates"]["c1"]["workspace"].startswith(str(destination))
    assert moved["id"] == "experiment-1"
    assert (str(source) + "/.local/research") not in (
        destination / ".local/research/environments/e/bin/tool"
    ).read_text()


def test_failed_commit_restores_existing_project(tmp_path, monkeypatch):
    m = module()
    source, _ = make_source(tmp_path)
    destination = source / "projects/agent-research"
    destination.mkdir(parents=True)
    marker = destination / "project.toml"
    marker.write_text("keep\n")
    monkeypatch.setattr(
        m, "_rewrite_db", lambda *args: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    try:
        m.migrate(source, destination, apply=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("migration should fail")
    assert marker.read_text() == "keep\n"
    assert not (destination / ".local").exists()
