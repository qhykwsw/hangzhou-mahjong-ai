"""Recipe iteration state machine: promotion, crash, retries and discards."""

from pathlib import Path
import re
import shutil

import pytest

from mahjong_ai.recipes.service import RecipeIteration

REPO = Path(__file__).resolve().parents[1]
SELFPLAY = "projects/personal/selfplay"


@pytest.fixture
def service(tmp_path):
    return make_service(tmp_path, "start")


def make_service(tmp_path, incumbent):
    project = tmp_path / SELFPLAY
    (project / "recipes").mkdir(parents=True)
    config = (REPO / SELFPLAY / "recipes/iteration.toml").read_text()
    config = re.sub(r'incumbent = "\w+"', f'incumbent = "{incumbent}"', config)
    (project / "recipes/iteration.toml").write_text(config)
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


def start(service, tmp_path=None):
    service.create(limit=3)
    service.control("resume")
    with service.store.edit() as (s, db):
        service.advance(s, db)
        lines = list(s["lines"])
        baseline = s["config"]["trial"]["incumbent"] == "baseline"
    step(service, "prepare", None, prepared(lines))
    for line in lines:
        if baseline:
            step(service, "baseline", line, trained())
    return lines


def request(service, line, root, candidate="rl--01--trial-00001"):
    service.request(line, root, "trial", candidate=candidate,
                    opponents={"rule-x": {"family": "code"}}, iteration=1)
    with service.store.edit() as (s, db):
        service.advance(s, db)


def reply(root):
    import json

    return json.loads((Path(root) / "result.json").read_text())


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
            "local_actor": "/local/trained" if ok else None,
            "metrics": [{"iteration": 1, "loss": 0.5}],
        },
    }


def test_trials_start_on_research_request_and_hand_off_after_training(service, tmp_path):
    line = start(service)[0]
    s = service.store.read()
    assert not s["trials"]
    request(service, line, tmp_path)
    s = service.store.read()
    trial = s["trials"][f"{line}-001"]
    assert trial["request"]["candidate"] == "rl--01--trial-00001"
    submit(service, line)
    s = service.store.read()
    task = (service.job_root(s, live(s, "train", line)["id"]) / "task.json").read_text()
    assert '"rule-x"' in task and '"fetch_actor": true' in task
    step(service, "train", line, trained())
    s = service.store.read()
    assert s["trials"][f"{line}-001"]["status"] == "handed_off"
    receipt = reply(tmp_path)
    assert receipt["candidates"]["rl--01--trial-00001"] == {
        "status": "ready", "local_actor": "/local/trained",
        "training": s["trials"][f"{line}-001"]["training"],
    }
    # Nothing else runs until research concludes the evaluation.
    assert not any(j["status"] == "queued" for j in s["jobs"].values())
    service.conclude(f"{line}-001", True, {"paired": {"delta": 1.0}})
    s = service.store.read()
    state = s["lines"][line]
    assert state["version"] == 1 and state["versions"][1]["local_actor"] == "/local/trained"
    assert s["trials"][f"{line}-001"]["outcome"] == "promoted"
    request(service, line, tmp_path / "next")
    assert service.store.read()["trials"][f"{line}-002"]["incumbent_version"] == 1


def test_crashed_or_unsubmitted_trials_reply_without_a_candidate(service, tmp_path):
    line = start(service)[0]
    request(service, line, tmp_path)
    submit(service, line)
    step(service, "train", line, trained(ok=False))
    s = service.store.read()
    assert s["trials"][f"{line}-001"]["outcome"] == "crashed"
    assert reply(tmp_path)["candidates"]["rl--01--trial-00001"]["status"] == "failed"
    request(service, line, tmp_path / "second")
    step(service, "agent", line, {"status": "completed", "usage": {"wall_seconds": 30}})
    step(service, "agent", line, {"status": "completed", "usage": {"wall_seconds": 900}})
    s = service.store.read()
    assert s["trials"][f"{line}-002"]["outcome"] == "unsubmitted"
    assert reply(tmp_path / "second")["candidates"]


def test_baseline_request_replies_with_version_zero(tmp_path):
    service = make_service(tmp_path, "baseline")
    line = start(service)[0]
    service.request(line, tmp_path / "baseline", "baseline")
    with service.store.edit() as (s, db):
        service.advance(s, db)
    receipt = reply(tmp_path / "baseline")
    assert receipt["baseline"] and receipt["local_actor"] == "/local/trained"




def test_infrastructure_failures_retry_then_pause(service, tmp_path):
    line = start(service)[0]
    request(service, line, tmp_path)
    submit(service, line)
    retries = service.store.read()["config"]["trial"]["retries"]
    for _ in range(retries + 1):
        step(service, "train", line, {"status": "failed", "error": "ssh down"})
    s = service.store.read()
    assert s["status"] == "pausing" and "ssh down" in s["pause_reason"]
    assert s["trials"][f"{line}-001"]["status"] == "training"


def test_research_cancel_discards_the_trial(service, tmp_path):
    line = start(service)[0]
    request(service, line, tmp_path)
    service.cancel(line)
    s = service.store.read()
    assert {t["status"] for t in s["trials"].values()} == {"discarded"}
    assert s["lines"][line]["trial"] is None and s["lines"][line]["request"] is None


def test_a_lost_host_redoes_the_same_training_job_without_counting_it(service, tmp_path):
    line = start(service)[0]
    request(service, line, tmp_path)
    submit(service, line)
    retries = service.store.read()["config"]["trial"]["retries"]
    for _ in range(retries + 2):
        with service.store.edit() as (s, db):
            job = live(s, "train", line)
            first = job["id"]
            job["status"] = "running"
            service.accept(s, db, job, {"status": "host_fault", "error": "rebooted"})
            service.advance(s, db)
            assert live(s, "train", line)["id"] == first
    s = service.store.read()
    assert s["status"] == "running" and s["trials"][f"{line}-001"]["status"] == "training"
    # After a service restart the trial continues once research runs again.
    with service.store.edit() as (s, db):
        live(s, "train", line)["status"] = "running"
    service.recover()
    assert service.store.read()["status"] == "paused"
    service.wake()
    s = service.store.read()
    assert s["status"] == "running" and live(s, "train", line)["status"] == "queued"
