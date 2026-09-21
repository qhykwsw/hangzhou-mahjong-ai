"""Remote publication is ordered after validation and stops at its deadline."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from guided import sync


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    store = tmp_path / "teachers"
    bundle = {
        "id": "source-v1",
        "runtime_hash": "same-frozen-runtime",
        "teacher": "teacher-a",
        "files": {
            "candidates/teacher-a/strategy.py": "sha",
            "runtimes/sdk/mahjong_ai/_fast.c": "sha",
        },
    }
    events = []
    failures = set()

    def export(source, output):
        output = Path(output)
        folder = output / bundle["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps(bundle))
        return folder / "manifest.json"

    def upload(host, root, files, destination):
        events.append((host, "upload", list(files), str(destination)))
        if (host, "upload") in failures:
            raise RuntimeError("remote output that should not leak into logs")

    def command(host, argv, **kwargs):
        phase = (
            "validate"
            if argv[2] == sync._PREPARE
            else ("discard" if argv[2] == sync._DISCARD else "publish")
        )
        events.append((host, phase, argv, kwargs))
        if (host, phase) in failures:
            raise RuntimeError("remote output that should not leak into logs")
        return json.dumps(
            {
                "id": bundle["id"],
                "runtime_hash": bundle["runtime_hash"],
                "hands": 1,
                "policies": 4,
            }
        )

    monkeypatch.setattr(sync.artifacts, "export_bundle", export)
    monkeypatch.setattr(sync.artifacts, "read_bundle", lambda path: dict(bundle))
    monkeypatch.setattr(sync.remote, "upload", upload)
    monkeypatch.setattr(sync.remote, "command", command)
    return store, bundle, events, failures


def perform(transfer, **kwargs):
    store, _, _, _ = transfer
    return sync.sync_once(
        "research-project",
        store,
        "/home/user/code/leihuo_majiang/.local/guided-teachers",
        **kwargs,
    )


def test_publish_follows_complete_upload_and_target_validation(transfer):
    store, bundle, events, _ = transfer
    result = perform(transfer)
    assert [r["status"] for r in result] == ["published", "published"]
    for host in ("4090", "4070"):
        steps = [event for event in events if event[0] == host]
        assert [event[1] for event in steps] == ["upload", "validate", "publish"]
        assert set(steps[0][2]) == {*bundle["files"], "manifest.json"}
        assert "current.json" not in steps[0][2]
        assert Path(steps[0][3]).name.startswith(".incoming-" + bundle["id"])
        assert steps[1][2][0] == sync.remote.PYTHON
    assert json.loads((store / "sync-state.json").read_text()) == {
        "4090": bundle["id"],
        "4070": bundle["id"],
    }
    assert len((store / "sync-log.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("phase", ["upload", "validate", "publish"])
def test_failure_preserves_current_and_only_failed_host_retries(transfer, phase):
    store, bundle, events, failures = transfer
    failures.add(("4070", phase))
    result = perform(transfer)
    assert [r["status"] for r in result] == ["published", "failed"]
    if phase != "publish":
        assert not any(e[:2] == ("4070", "publish") for e in events)
    assert json.loads((store / "sync-state.json").read_text()) == {"4090": bundle["id"]}
    assert "remote output" not in (store / "sync-log.jsonl").read_text()
    events.clear()
    failures.clear()
    result = perform(transfer)
    assert [r["status"] for r in result] == ["unchanged", "published"]
    assert {event[0] for event in events} == {"4070"}


def test_successful_version_is_not_retransmitted(transfer):
    _, _, events, _ = transfer
    perform(transfer)
    events.clear()
    assert all(r["status"] == "unchanged" for r in perform(transfer))
    assert events == []


def test_remote_commands_use_explicit_frozen_framework_checkout(transfer):
    _, _, events, failures = transfer
    code = sync.remote.ROOT + "/.local/guided-dev/20260920"
    failures.add(("4070", "validate"))
    result = perform(transfer, code_path=code)
    assert [row["status"] for row in result] == ["published", "failed"]
    commands = [event for event in events if event[1] != "upload"]
    assert {event[1] for event in commands} == {"validate", "publish", "discard"}
    assert all(event[3]["cwd"] == code for event in commands)


def test_new_version_retransmits_but_incompatible_runtime_never_publishes(transfer):
    store, bundle, events, _ = transfer
    perform(transfer)
    prior_state = (store / "sync-state.json").read_bytes()
    bundle["id"] = "source-v2"
    events.clear()
    result = perform(transfer, expected_runtime_hash="different-sdk")
    assert all(r["status"] == "rejected" for r in result)
    assert events == []
    assert (store / "sync-state.json").read_bytes() == prior_state
    assert all(
        r["status"] == "published"
        for r in perform(transfer, expected_runtime_hash=bundle["runtime_hash"])
    )


def test_mismatched_validation_receipt_cannot_publish(transfer, monkeypatch):
    _, _, events, _ = transfer

    def command(host, argv, **kwargs):
        events.append((host, "validate", argv))
        return json.dumps(
            {
                "id": "wrong-version",
                "runtime_hash": "wrong-sdk",
                "hands": 1,
                "policies": 4,
            }
        )

    monkeypatch.setattr(sync.remote, "command", command)
    result = perform(transfer)
    assert all(r["status"] == "failed" and r["phase"] == "validate" for r in result)
    assert not any(event[1] == "publish" for event in events)


def test_expired_deadline_performs_no_transfer(transfer):
    _, _, events, _ = transfer
    assert perform(transfer, deadline=0) == []
    assert events == []


def test_poll_loop_has_one_finite_deadline(monkeypatch, tmp_path):
    now = [10.0]
    calls = []
    monkeypatch.setattr(sync, "_until", lambda deadline: nullcontext())
    monkeypatch.setattr(sync.time, "time", lambda: now[0])
    monkeypatch.setattr(
        sync.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(
        sync,
        "sync_once",
        lambda *args, **kwargs: calls.append((kwargs["deadline"], kwargs["code_path"])),
    )
    sync.run_until(
        "source", tmp_path, "/remote", deadline=16, interval=4, code_path="/frozen/code"
    )
    assert calls == [(16, "/frozen/code"), (16, "/frozen/code")]
    assert now[0] == 16


def test_deadline_interrupt_is_not_swallowed_as_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "_until", lambda deadline: nullcontext())

    def interrupted(*args, **kwargs):
        raise sync.DeadlineReached()

    monkeypatch.setattr(sync, "sync_once", interrupted)
    sync.run_until("source", tmp_path, "/remote", deadline=sync.time.time() + 60)


def test_remote_programs_are_valid_python_and_validation_binds_before_sdk_import():
    for label in ("_VALIDATE", "_PREPARE", "_PUBLISH", "_DISCARD"):
        compile(getattr(sync, label), label, "exec")
    assert sync._VALIDATE.index("bind_runtime(bundle)") < sync._VALIDATE.index(
        "from mahjong_ai"
    )
    assert "TemporaryDirectory" in sync._PREPARE
    assert "read_bundle(manifest)" in sync._PUBLISH
    assert sync._PUBLISH.index("receipt['hands']") < sync._PUBLISH.index(
        "root/'current.json'"
    )


@pytest.mark.parametrize(
    "valid,preexisting", [(True, False), (False, False), (True, True)]
)
def test_publication_program_atomically_replaces_only_validated_version(
    tmp_path, valid, preexisting
):
    # Execute the actual remote publication program locally, with a minimal
    # read_bundle test double. No SSH, installed SDK, CUDA or teacher is needed.
    package = tmp_path / "projects/personal/selfplay/guided"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "artifacts.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def read_bundle(path): return json.loads(Path(path).read_text())\n"
    )
    store = tmp_path / "remote-store"
    staging = store / ".incoming-new-version-123"
    staging.mkdir(parents=True)
    native = staging / "_fast.so"
    native.write_bytes(b"derived-native")
    manifest = staging / "manifest.json"
    manifest.write_text(
        json.dumps({"id": "new-version", "runtime_hash": "runtime-sha"})
    )
    if preexisting:
        shutil.copytree(staging, store / "new-version")
    receipts = store / ".receipts"
    receipts.mkdir()
    (receipts / (staging.name + ".json")).write_text(
        json.dumps(
            {
                "id": "new-version",
                "runtime_hash": "runtime-sha",
                "hands": 1 if valid else 0,
                "policies": 4,
                "native": {"_fast.so": hashlib.sha256(native.read_bytes()).hexdigest()},
            }
        )
    )
    pointer = store / "current.json"
    original = json.dumps(
        {"id": "old-version", "manifest": "old-version/manifest.json"}
    )
    pointer.write_text(original)
    validate = (
        "import json,sys;from pathlib import Path;"
        "assert Path(sys.argv[1]).parent.name == 'new-version';"
        "print(json.dumps({'id':'new-version','runtime_hash':'runtime-sha','hands':1,'policies':4}))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            sync._PUBLISH,
            str(manifest),
            "new-version",
            "runtime-sha",
            str(time.time() + 30),
            validate,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if valid:
        assert result.returncode == 0, result.stderr
        assert json.loads(pointer.read_text())["id"] == "new-version"
        assert (store / "new-version/manifest.json").is_file()
        assert (store / "new-version/_fast.so").read_bytes() == b"derived-native"
        assert not staging.exists()
    else:
        assert result.returncode != 0
        assert pointer.read_text() == original
        assert not (store / "new-version").exists()
