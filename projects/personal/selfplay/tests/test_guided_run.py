"""Recovery must preserve healthy weights after a partially failed update."""

import json
import time

import pytest

torch = pytest.importorskip("torch")

from guided.config import configuration
from guided import artifacts, collection, learning
from guided.run import train, pending_bundle


def test_timed_snapshots_preserve_distinct_weights_and_training_progress(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from guided import run as runner

    now = [0.0]
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=lambda: now[0]))
    manifest = tmp_path / "teacher.json"
    monkeypatch.setattr(
        artifacts,
        "read_bundle",
        lambda *a, **kw: {
            "manifest_path": str(manifest),
            "id": "fixed",
            "teacher": "teacher",
        },
    )
    monkeypatch.setattr(
        collection, "collect", lambda *a, **kw: ([{"teacher": -1}], [], {})
    )

    def update(actor, *args):
        with torch.no_grad():
            next(actor.parameters()).add_(1)
        now[0] += 6
        return {"optimized_samples": 8, "sgd_steps": 2}

    monkeypatch.setattr(learning, "update", update)
    cfg = configuration(dim=16, layers=1, heads=2, device="cpu", samples=1)
    root = tmp_path / "run"
    train(
        root,
        cfg,
        manifest,
        manifest,
        deadline=12,
        bc_seconds=30,
        eval_enabled=False,
        snapshot_seconds=5,
    )
    states = [
        torch.load(root / "snapshots" / (name + ".pt"), weights_only=False)
        for name in ["initial", "seconds-000005", "seconds-000010", "final"]
    ]
    key = next(iter(states[0]["actor"]))
    for state, expected in zip(states, [0, 1, 2, 2]):
        torch.testing.assert_close(
            state["actor"][key], states[0]["actor"][key] + expected
        )
        assert state["updates"] == expected
        assert state["sgd_steps"] == expected * 2
        assert state["optimized_samples"] == expected * 8
    assert states[1]["scheduled_seconds"] == 5
    assert states[1]["elapsed_seconds"] == 6


def test_bc_boundary_is_retained_before_first_joint_update(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from guided import run as runner

    now = [0.0]
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=lambda: now[0]))
    manifest = tmp_path / "teacher.json"
    monkeypatch.setattr(
        artifacts,
        "read_bundle",
        lambda *a, **kw: {
            "manifest_path": str(manifest),
            "id": "fixed",
            "teacher": "teacher",
        },
    )
    monkeypatch.setattr(
        collection, "collect", lambda *a, **kw: ([{"teacher": -1}], [], {})
    )

    def update(actor, critic, actor_opt, critic_opt, rows, cfg, *args):
        with torch.no_grad():
            next(actor.parameters()).add_(1 if cfg["bc_only"] else 10)
        now[0] += 6
        return {}

    monkeypatch.setattr(learning, "update", update)
    cfg = configuration(
        dim=16, layers=1, heads=2, device="cpu", workers=1, tables=1, samples=1
    )
    path = tmp_path / "run"
    train(
        path,
        cfg,
        manifest,
        manifest,
        deadline=30,
        bc_seconds=5,
        eval_enabled=False,
        max_updates=2,
    )
    initial = torch.load(path / "initial.pt", weights_only=False)["actor"]
    bc = torch.load(path / "bc-final.pt", weights_only=False)["actor"]
    final = torch.load(path / "latest.pt", weights_only=False)["actor"]
    key = next(iter(initial))
    torch.testing.assert_close(bc[key], initial[key] + 1)
    torch.testing.assert_close(final[key], initial[key] + 11)


def test_update_limit_and_resume_preserve_progress(tmp_path, monkeypatch):
    manifest = tmp_path / "teacher.json"
    monkeypatch.setattr(
        artifacts,
        "read_bundle",
        lambda *a, **kw: {
            "manifest_path": str(manifest),
            "id": "fixed",
            "teacher": "teacher",
        },
    )
    monkeypatch.setattr(
        collection, "collect", lambda *a, **kw: ([{"teacher": -1}], [], {})
    )
    monkeypatch.setattr(learning, "update", lambda *a, **kw: {})
    cfg = configuration(dim=16, layers=1, heads=2, device="cpu", samples=1)
    run = tmp_path / "run"
    deadline = time.time() + 60
    train(
        run,
        cfg,
        manifest,
        manifest,
        deadline=deadline,
        eval_enabled=False,
        max_updates=1,
    )
    train(
        run,
        cfg,
        manifest,
        manifest,
        deadline=deadline,
        resume=True,
        eval_enabled=False,
        max_updates=1,
    )
    state = torch.load(run / "latest.pt", weights_only=False)
    assert state["iteration"] == 2
    assert state["deadline"] == deadline
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert sum(row["kind"] == "update" for row in rows) == 2


def test_failed_update_does_not_replace_healthy_checkpoint(tmp_path, monkeypatch):
    manifest = tmp_path / "teacher.json"
    manifest.write_text("{}")
    monkeypatch.setattr(
        artifacts,
        "read_bundle",
        lambda path, **kw: {
            "manifest_path": str(manifest),
            "id": "fixed",
            "teacher": "teacher",
            "runtime_hash": "runtime",
        },
    )
    monkeypatch.setattr(
        collection, "collect", lambda *a, **kw: ([{"teacher": 0}], [], {})
    )

    def fail_update(actor, *args, **kwargs):
        with torch.no_grad():
            next(actor.parameters()).fill_(float("nan"))
        raise FloatingPointError("injected numerical failure")

    monkeypatch.setattr(learning, "update", fail_update)
    cfg = configuration(
        dim=16, layers=1, heads=2, device="cpu", workers=1, tables=1, samples=1
    )
    run = tmp_path / "run"
    with pytest.raises(FloatingPointError):
        train(
            run, cfg, manifest, manifest, deadline=time.time() + 60, eval_enabled=False
        )
    latest = torch.load(run / "latest.pt", weights_only=False)
    initial = torch.load(run / "initial.pt", weights_only=False)
    assert all(torch.isfinite(value).all() for value in latest["actor"].values())
    assert all(
        torch.equal(value, initial["actor"][name])
        for name, value in latest["actor"].items()
    )
    assert json.loads((run / "status.json").read_text())["state"] == "failed"


def test_teacher_compatibility_checks_content_not_directory_name(tmp_path, monkeypatch):
    pointer = tmp_path / "current.json"
    pointer.write_text("{}")
    active = tmp_path / "active.json"
    monkeypatch.setattr(
        artifacts,
        "read_bundle",
        lambda path, **kw: {
            "manifest_path": str(path),
            "runtime": "/same/name",
            "runtime_hash": "new" if path == pointer else "old",
        },
    )
    with pytest.raises(ValueError, match="different engine"):
        pending_bundle(pointer, active)


def test_carry_preserves_inherited_best_even_when_latest_is_different(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "teacher.json"
    manifest.write_text("{}")
    monkeypatch.setattr(
        artifacts,
        "read_bundle",
        lambda path, **kw: {
            "manifest_path": str(manifest),
            "id": "fixed",
            "teacher": "teacher",
            "runtime_hash": "runtime",
        },
    )
    cfg = configuration(
        dim=16, layers=1, heads=2, device="cpu", workers=1, tables=1, samples=1
    )
    source = tmp_path / "trial"
    train(source, cfg, manifest, manifest, deadline=0, eval_enabled=False)
    state = torch.load(source / "latest.pt", weights_only=False)
    state["best_score"] = 42.0
    torch.save(state, source / "latest.pt")
    key = next(iter(state["actor"]))
    state["actor"][key] = state["actor"][key] + 1
    torch.save(state, source / "best.pt")
    target = tmp_path / "main"
    train(
        target,
        cfg,
        manifest,
        manifest,
        initial=source / "latest.pt",
        carry=True,
        deadline=0,
        eval_enabled=False,
    )
    best = torch.load(target / "best.pt", weights_only=False)
    latest = torch.load(target / "latest.pt", weights_only=False)
    assert torch.equal(best["actor"][key], state["actor"][key])
    assert not torch.equal(best["actor"][key], latest["actor"][key])
