"""Recipe iteration state machine: promotion, crash, retries and discards."""

import shutil
from pathlib import Path

import pytest

from mahjong_ai.recipes.service import RecipeIteration

REPO = Path(__file__).resolve().parents[1]
SELFPLAY = "projects/personal/selfplay"


@pytest.fixture
def service(tmp_path):
    project = tmp_path / SELFPLAY
    (project / "recipes").mkdir(parents=True)
    shutil.copy(REPO / SELFPLAY / "recipes/iteration.toml", project / "recipes")
    shutil.copytree(
        REPO / SELFPLAY / "recipes/baseline", project / "recipes/baseline"
    )
    return RecipeIteration(tmp_path)


def prepared(lines):
    return {
        "status": "completed",
        "output": {
            "source_id": "s",
            "code": "/remote/code",
            "remote_root": "/remote/recipes",
            "bundle": "/remote/bundle/manifest.json",
            "bundle_id": "b",
            "teacher": "rule-a",
            "rules": ["rule-a", "rule-b", "rule-c", "rule-d"],
            "lines": {
                key: {
                    "start_actor": f"/remote/{key}/start",
                    "start_meta": {"config": {"dim": 8, "layers": 1, "heads": 1}},
                }
                for key in lines
            },
        },
    }


def brawl_result():
    rows = {
        name: {"mean": 0.0, "lower": -1.0, "upper": 1.0, "appearances": 16}
        for name in ["model:incumbent", "rule-a", "rule-b", "rule-c", "rule-d"]
    }
    return {
        "status": "completed",
        "output": {
            "ok": True,
            "result": {
                "results": rows,
                "ranking": list(rows),
                "games": 20,
                "seconds": 1.0,
            },
        },
    }


def live(s, kind, line=None):
    jobs = [
        j
        for j in s["jobs"].values()
        if j["kind"] == kind
        and j.get("line") == line
        and j["status"] in ("queued", "running")
    ]
    assert len(jobs) == 1, (kind, line, [j["status"] for j in s["jobs"].values()])
    return jobs[0]


def step(service, kind, line, result):
    with service.store.edit() as (s, db):
        job = live(s, kind, line)
        job["status"] = "running"
        service.accept(s, db, job, result)
        service.advance(s, db)


def start(service):
    service.create(limit=3)
    service.control("resume")
    with service.store.edit() as (s, db):
        service.advance(s, db)
        lines = list(s["lines"])
    step(service, "prepare", None, prepared(lines))
    for line in lines:
        step(service, "brawl", line, brawl_result())
    return lines


def submit(service, line):
    with service.store.edit() as (s, db):
        trial = s["trials"][s["lines"][line]["trial"]]
        trial.update(status="submitted", submitted=trial["workspace"], summary="x")
    step(service, "agent", line, {"status": "completed", "usage": {}})


def trained(ok=True):
    status = {"state": "complete" if ok else "failed", "training_seconds": 900.0}
    return {
        "status": "completed",
        "output": {
            "ok": ok,
            "status": status,
            "actor": "/remote/trained" if ok else None,
            "metrics": [{"iteration": 1, "loss": 0.5}],
        },
    }


def evaluated(lower):
    difference = {"mean": lower + 1, "lower": lower, "upper": lower + 2}
    return {
        "status": "completed",
        "output": {
            "ok": True,
            "result": {
                "games": 1024,
                "challenger": {"mean": 1.0},
                "incumbent": {"mean": 0.0},
                "difference": difference,
                "promoted": lower > 0,
                "seconds": 1.0,
            },
        },
    }


def test_promotion_brawls_the_new_version_before_the_next_trial(service):
    line = start(service)[0]
    s = service.store.read()
    assert s["lines"][line]["curve"][0]["version"] == 0
    assert live(s, "agent", line)["trial"] == f"{line}-001"
    submit(service, line)
    step(service, "train", line, trained())
    step(service, "evaluate", line, evaluated(0.5))
    s = service.store.read()
    state = s["lines"][line]
    assert state["version"] == 1 and state["used"] == 1
    assert state["versions"][1]["actor"] == "/remote/trained"
    assert live(s, "brawl", line)["version"] == 1
    assert not any(
        j["kind"] == "agent" and j["status"] == "queued" and j["line"] == line
        for j in s["jobs"].values()
    )
    step(service, "brawl", line, brawl_result())
    s = service.store.read()
    assert [p["version"] for p in s["lines"][line]["curve"]] == [0, 1]
    assert s["trials"][f"{line}-002"]["incumbent_version"] == 1


def test_rejected_and_crashed_trials_consume_the_allowance(service):
    line = start(service)[0]
    submit(service, line)
    step(service, "train", line, trained(ok=False))
    s = service.store.read()
    assert s["trials"][f"{line}-001"]["outcome"] == "crashed"
    submit(service, line)
    step(service, "train", line, trained())
    step(service, "evaluate", line, evaluated(-0.5))
    s = service.store.read()
    assert s["trials"][f"{line}-002"]["outcome"] == "rejected"
    assert s["lines"][line]["used"] == 2 and s["lines"][line]["version"] == 0


def test_infrastructure_failures_retry_then_pause(service):
    line = start(service)[0]
    submit(service, line)
    retries = service.store.read()["config"]["trial"]["retries"]
    for _ in range(retries + 1):
        step(service, "train", line, {"status": "failed", "error": "ssh down"})
    s = service.store.read()
    assert s["status"] == "pausing" and "ssh down" in s["pause_reason"]
    assert s["trials"][f"{line}-001"]["status"] == "training"
    assert s["lines"][line]["used"] == 0


def test_stop_now_discards_trials_without_counting_them(service):
    lines = start(service)
    service.control("stop_now")
    s = service.store.read()
    assert all(s["lines"][key]["used"] == 0 for key in lines)
    assert {t["status"] for t in s["trials"].values()} == {"discarded"}
    assert s["status"] == "pausing"


def test_unsubmitted_session_continues_while_time_remains(service):
    line = start(service)[0]
    step(service, "agent", line, {"status": "completed", "usage": {"wall_seconds": 30}})
    s = service.store.read()
    job = live(s, "agent", line)
    assert job["continuations"] == 1
    step(service, "agent", line, {"status": "completed", "usage": {"wall_seconds": 500}})
    s = service.store.read()
    assert s["trials"][f"{line}-001"]["outcome"] == "unsubmitted"
    assert s["lines"][line]["used"] == 1
