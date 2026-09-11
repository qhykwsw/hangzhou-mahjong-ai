#!/usr/bin/env python3
"""Controlled, dry-run-by-default migration of the agent research project."""

from __future__ import annotations
import argparse, hashlib, json, shutil, sqlite3, subprocess, tempfile
from pathlib import Path
from contextlib import contextmanager


def _sha(path):
    path = Path(path)
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    elif path.exists():
        for p in sorted(path.rglob("*")):
            if p.is_file() and not p.is_symlink():
                h.update(str(p.relative_to(path)).encode())
                h.update(_sha(p).encode())
    return h.hexdigest()


def _rewrite(value, source, destination):
    if isinstance(value, str):
        return value.replace(str(source), str(destination))
    if isinstance(value, list):
        return [_rewrite(v, source, destination) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite(v, source, destination) for k, v in value.items()}
    return value


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _state(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        row = db.execute("SELECT state FROM registry WHERE id=1").fetchone()
    if not row:
        raise ValueError("research.sqlite has no registry state")
    return json.loads(row[0])


def _assert_stopped(state, source):
    active = [
        j["id"]
        for j in state.get("jobs", {}).values()
        if j.get("status") in ("running", "admitted")
    ]
    if active:
        raise RuntimeError("暂停或停止执行后再迁移：在途任务 " + ", ".join(active))
    try:
        ps = subprocess.run(
            ["/bin/ps", "-axo", "command="], capture_output=True, text=True, check=True
        ).stdout
        if str(source) in ps and "migrate_strategy_project.py" not in ps:
            raise RuntimeError("检测到仍在运行的研究进程，请先停止服务")
    except FileNotFoundError:
        pass


@contextmanager
def _lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _backup_db(source, destination):
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def _copy_tree(source, staging):
    local = staging / ".local"
    (local / "service").mkdir(parents=True)
    shutil.copytree(source / ".local/research", local / "research", symlinks=True)
    shutil.copytree(source / ".local/strategies", local / "strategies", symlinks=True)
    _backup_db(
        source / ".local/service/research.sqlite", local / "service/research.sqlite"
    )
    if (source / "research.toml").exists():
        shutil.copy2(source / "research.toml", staging / "research.toml")


def _rewrite_mutable_jobs(staging, state, source, destination):
    for job in state.get("jobs", {}).values():
        if job.get("status") == "completed":
            continue
        root = staging / ".local/research/jobs" / job["id"]
        for name in ("task.json", "result.json", "progress.json"):
            p = root / name
            if p.exists():
                try:
                    _write_json(p, _rewrite(_read_json(p), source, destination))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
        for p in root.glob("*/artifact.json"):
            _write_json(p, _rewrite(_read_json(p), source, destination))


def _rewrite_db(path, source, destination):
    with sqlite3.connect(path) as db:
        row = db.execute("SELECT state FROM registry WHERE id=1").fetchone()
        if not row:
            raise ValueError("copied research.sqlite has no registry state")
        db.execute(
            "UPDATE registry SET state=? WHERE id=1",
            (
                json.dumps(
                    _rewrite(json.loads(row[0]), source, destination),
                    ensure_ascii=False,
                ),
            ),
        )


def _rewrite_environment_scripts(staging, source, destination):
    root = staging / ".local/research/environments"
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        old = str(source).encode()
        new = str(destination).encode()
        if old not in raw:
            continue
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        mode = p.stat().st_mode
        p.write_bytes(raw.replace(old, new))
        p.chmod(mode)


def migrate(source, destination, apply=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination != source / "projects/agent-research":
        raise ValueError("destination must be source/projects/agent-research")
    db = source / ".local/service/research.sqlite"
    if not db.exists():
        raise ValueError("source has no .local/service/research.sqlite")
    if destination.exists() and {p.name for p in destination.iterdir()} != {
        "project.toml"
    }:
        raise ValueError("destination must be absent or contain only project.toml")
    state = _state(db)
    _assert_stopped(state, source)
    hashes = {
        "runtime": _sha(source / ".local/research/runtimes"),
        "strategies": _sha(source / ".local/strategies"),
        "research": _sha(source / ".local/research"),
    }
    report = {
        "source": str(source),
        "destination": str(destination),
        "status": "dry-run",
        "state_id": state.get("id"),
        "jobs": len(state.get("jobs", {})),
        "immutable_hashes": hashes,
    }
    if not apply:
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _lock(source / ".local/service/registry.lock"):
        state = _state(db)
        _assert_stopped(state, source)
        staging = Path(
            tempfile.mkdtemp(prefix=".agent-research-", dir=destination.parent)
        )
        backup = destination.with_name(destination.name + ".rollback")
        committed = False
        try:
            if destination.exists():
                shutil.copy2(destination / "project.toml", staging / "project.toml")
            _copy_tree(source, staging)
            _rewrite_db(staging / ".local/service/research.sqlite", source, destination)
            _rewrite_mutable_jobs(staging, state, source, destination)
            _rewrite_environment_scripts(staging, source, destination)
            if _sha(staging / ".local/strategies") != hashes["strategies"]:
                raise ValueError("strategy package hash changed during migration")
            if destination.exists():
                destination.rename(backup)
            staging.rename(destination)
            committed = True
            if backup.exists():
                shutil.rmtree(backup)
            report.update(status="migrated", source_retained=True)
            return report
        except BaseException:
            if committed and destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            if backup.exists() and not destination.exists():
                backup.rename(destination)
            shutil.rmtree(staging, ignore_errors=True)
            raise


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", default=".")
    p.add_argument("destination", nargs="?")
    p.add_argument("--apply", action="store_true")
    a = p.parse_args(argv)
    source = Path(a.source).resolve()
    destination = (
        Path(a.destination).resolve()
        if a.destination
        else source / "projects/agent-research"
    )
    print(
        json.dumps(migrate(source, destination, a.apply), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
