from pathlib import Path
import time
import pytest
from starlette.testclient import TestClient
from mahjong_ai.research.service import Research
from mahjong_ai.research.config import configuration
from mahjong_ai.evaluation.schedule import schedule, eliminated
from mahjong_ai.execution.controller import Controller
from mahjong_ai.execution.jobs import prepare, play
from mahjong_ai.interfaces.app import application


def small_config():
    return dict(
        total_iterations=1,
        first_cycle=1,
        cycle_iterations=1,
        evaluation_games=4,
        competition_games_per_branch=4,
        training_games=1,
        agent_concurrency=2,
        training_workers=1,
        match_workers=2,
        game={"rounds": 1},
        combinations=[
            dict(
                id="heuristic-balanced",
                branches=4,
                models=["astra"] * 4,
                parameters={"style": "balanced"},
                training_games=1,
            )
        ],
    )


@pytest.fixture
def research(tmp_path):
    r = Research(tmp_path)
    run = r.create(small_config())
    r.control(run, "resume")
    c = Controller(r)
    r.control(run, "resume")
    with r.edit(run) as (s, db):
        job = next(iter(s["jobs"].values()))
        task = c.task(s, job)
        root = Path(s["root"]) / "jobs" / job["id"]
        root.mkdir(parents=True)
        result = prepare(task, root, time.monotonic() + 30)
        r.accept(s, db, job, result)
        r.advance(s, db)
    return r, run, c


def test_defaults_and_model_validation(tmp_path):
    cfg = configuration(tmp_path)
    assert sum(c["branches"] for c in cfg["combinations"]) == 24
    assert all(
        c["models"] == ["astra", "deepseek", "qwen"] for c in cfg["combinations"]
    )
    assert cfg["models"]["astra"]["effort"] == "medium"
    with pytest.raises(ValueError, match="effort"):
        configuration(tmp_path, {"models": {"qwen": {"effort": "medium"}}})
    with pytest.raises(ValueError, match="multiple"):
        configuration(tmp_path, {"evaluation_games": 5})


def test_branch_isolation_atomic_submission_and_full_cycle(research):
    r, run, controller = research
    initial = r.read(run)
    assert initial["phase"] == "researching"
    branches = list(initial["branches"])
    for branch in branches:
        with r.edit(run) as (s, db):
            job = r.start_iteration(s, branch)
        r.fork(run, branch, "v1", "try unchanged parent")
        with pytest.raises(ValueError, match="owned"):
            r.fork(
                run,
                branch,
                "bad",
                "steal",
                branches[-1 if branch != branches[-1] else 0] + "--initial",
            )
        with pytest.raises(ValueError, match="relative"):
            r.candidate(run, branch, "v1", path="../private/random.json")
        r.submit(run, branch, ["v1"])
        with pytest.raises(ValueError):
            r.candidate(run, branch, "v1", path="strategy.py", content="changed")
        with r.edit(run) as (s, db):
            r.accept(
                s,
                db,
                s["jobs"][job["id"]],
                dict(status="completed", usage={"wall_seconds": 0.1}),
            )
            r.advance(s, db)
            prep = next(
                j
                for j in s["jobs"].values()
                if j["kind"] == "prepare" and j.get("branch") == branch
            )
            task = controller.task(s, prep)
            root = Path(s["root"]) / "jobs" / prep["id"]
            root.mkdir(parents=True)
            result = prepare(task, root, time.monotonic() + 30)
            r.accept(s, db, prep, result)
            matches = [
                j
                for j in s["jobs"].values()
                if j["kind"] == "match" and j.get("branch") == branch
            ]
            assert len(matches) == 2
            tasks = [controller.task(s, j) for j in matches]
            assert tasks[0]["seed"] == tasks[1]["seed"]
            for j, task in zip(matches, tasks):
                root = Path(s["root"]) / "jobs" / j["id"]
                root.mkdir(parents=True)
                result = play(task, root)
                assert result["focal"]["appearances"] == 4
                assert result["focal"]["errors"] == 0
                r.accept(s, db, j, result)
                r.accept(s, db, j, result)
            assert s["branches"][branch]["completed"] == 1
            assert s["branches"][branch]["representative"].endswith("--initial")
            r.advance(s, db)
    with r.edit(run) as (s, db):
        assert s["phase"] == "competing"
        for j in [j for j in s["jobs"].values() if j.get("scope") == "competition"]:
            root = Path(s["root"]) / "jobs" / j["id"]
            root.mkdir(parents=True)
            r.accept(s, db, j, play(controller.task(s, j), root))
        r.advance(s, db)
        assert s["phase"] == "finished" and s["status"] == "paused"
        assert len(s["last_ranking"]) == 4
    snap = r.snapshot(run, branches[0])
    encoded = str(snap)
    assert (
        "package_path" not in encoded
        and "thread_id" not in encoded
        and "workspace" not in encoded
    )
    assert all(c["branch"] == branches[0] for c in snap["candidates"])


def test_failure_consumes_once_and_config_boundaries(research):
    r, run, c = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
        result = dict(
            status="failed",
            error="iteration execution limit reached",
            usage={"wall_seconds": 3600},
        )
        r.accept(s, db, job, result)
        r.accept(s, db, job, result)
        assert (
            s["branches"][branch]["completed"] == 1
            and s["branches"][branch]["failed"] == 1
        )
    r.configure(
        run, {"match_workers": 3, "evaluation_games": 8, "iteration_seconds": 20}
    )
    s = r.read(run)
    assert s["config"]["match_workers"] == 3 and s["config"]["evaluation_games"] == 4
    assert s["pending"]["evaluation_games"] == 8
    with pytest.raises(ValueError, match="new research"):
        r.configure(run, {"game": {"rounds": 2}})
    r.control(run, "extend", 3)
    assert r.read(run)["config"]["total_iterations"] == 4


def test_protection_skips_without_replacement():
    branches = {
        f"b{i}": {
            "id": f"b{i}",
            "combination": "protected" if i >= 8 else "large",
            "status": "ready",
        }
        for i in range(10)
    }
    table = [{"branch": f"b{i}"} for i in range(10)]
    assert eliminated(table, branches, 0.2, 2) == []
    branches["b8"]["combination"] = "large"
    assert eliminated(table, branches, 0.2, 2) == ["b8"]


def test_balanced_four_seat_schedule():
    games = schedule([str(i) for i in range(64)], 1024, 8)
    for cid in map(str, range(64)):
        assert [sum(g["players"][seat] == cid for g in games) for seat in range(4)] == [
            16
        ] * 4


def test_api_configuration_and_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(Controller, "tick", lambda self: None)
    with TestClient(application(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert (
            client.get("/api/directions").json()["defaults"]["total_iterations"] == 100
        )
        result = client.post("/api/runs", json={"config": small_config()})
        assert result.status_code == 200
        run = result.json()["run"]["id"]
        assert (
            client.post(
                "/api/control",
                json={"run": run, "action": "pause"},
                headers={"Origin": "https://other.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/config", json={"run": run, "patch": {"match_workers": 3}}
            ).json()["config"]["match_workers"]
            == 3
        )


def test_pause_submission_recovery_does_not_repeat_agent(research):
    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
    r.fork(run, branch, "submitted", "recover")
    r.submit(run, branch, ["submitted"])
    with r.edit(run) as (s, db):
        j = s["jobs"][job["id"]]
        j.update(status="running", started_at=time.time() - 1)
        root = Path(s["root"]) / "jobs" / j["id"]
        root.mkdir(parents=True)
        (root / "codex.jsonl").write_text(
            '{"type":"thread.started","thread_id":"saved-thread"}\n'
        )
    Controller(r)
    s = r.read(run)
    assert s["jobs"][job["id"]]["status"] == "completed"
    assert s["branches"][branch]["thread_id"] == "saved-thread"
    assert s["branches"][branch]["completed"] == 0
    r.control(run, "resume")
    with r.edit(run) as (s, db):
        r.advance(s, db)
        assert s["branches"][branch]["status"] == "preparing"
        assert sum(j["kind"] == "agent" for j in s["jobs"].values()) == 1


def test_partial_prepare_failure_evaluates_all_successes(research):
    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
    for name in ("good", "bad"):
        r.fork(run, branch, name, "partial batch")
    r.candidate(
        run,
        branch,
        "bad",
        path="strategy.py",
        content="def build(model_dir): return None",
    )
    r.submit(run, branch, ["good", "bad"])
    with r.edit(run) as (s, db):
        r.accept(s, db, s["jobs"][job["id"]], {"status": "completed"})
        r.advance(s, db)
        prep = next(
            j
            for j in s["jobs"].values()
            if j.get("branch") == branch and j["kind"] == "prepare"
        )
        root = Path(s["root"]) / "jobs" / prep["id"]
        root.mkdir(parents=True)
        r.accept(
            s, db, prep, prepare(controller.task(s, prep), root, time.monotonic() + 30)
        )
        matches = [
            j
            for j in s["jobs"].values()
            if j.get("branch") == branch and j["kind"] == "match"
        ]
        assert len(matches) == 2
        assert s["candidates"][branch + "--bad"]["status"] == "failed"
        assert all(len(j["assignments"]) == 4 for j in matches)


def test_64_branches_use_dynamic_agent_slots(tmp_path, monkeypatch):
    r = Research(tmp_path)
    cfg = configuration(tmp_path)
    for combination in cfg["combinations"]:
        combination.update(branches=8, models=["deepseek"] * 8)
    run = r.create(cfg)
    controller = Controller(r)
    r.control(run, "resume")
    # Exercise admission under a populated, initialized 64-branch state without
    # making 64 external LLM calls or changing production scheduling behavior.
    with r.edit(run) as (s, db):
        for j in s["jobs"].values():
            j["status"] = "completed"
        for b in s["branches"].values():
            b.update(status="ready", representative="placeholder-" + b["id"])
        s["phase"] = "researching"
        s["cycles"] = [
            dict(
                id=1,
                status="researching",
                targets={b: 20 for b in s["branches"]},
                config=s["config"],
            )
        ]
    launched = []

    def launch(s, j):
        launched.append(j["branch"])
        j["status"] = "running"

    monkeypatch.setattr(controller, "launch", launch)
    controller.tick()
    controller.tick()
    assert len(launched) == 10 and len(set(launched)) == 10


def test_custom_combination_and_model_are_configurable(tmp_path):
    cfg = small_config()
    cfg["combinations"][0].update(
        id="custom-shape", family="heuristic", name="Custom shape"
    )
    r = Research(tmp_path)
    run = r.create(cfg)
    assert len(r.read(run)["branches"]) == 4
    assert "custom-shape" in r.read(run)["baselines"]


def test_submission_is_idempotent_and_releases_slot(research):
    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
    r.fork(run, branch, "idempotent", "one batch")
    first = r.submit(run, branch, ["idempotent"])
    second = r.submit(run, branch, ["idempotent"])
    assert first["iteration"] == second["iteration"]
    with r.edit(run) as (s, db):
        j = s["jobs"][job["id"]]
        j["status"] = "running"
        it = s["iterations"][j["iteration"]]
        it["submitted_at"] = time.time() - 10
        root = Path(s["root"]) / "jobs" / j["id"]
        root.mkdir(parents=True)
        # Fill slots so this check never launches unrelated model calls.
        s["config"]["agent_concurrency"] = 1
    controller.tick()
    assert (root / "submitted.flag").exists()


def test_round_config_can_end_midcycle_without_elimination(research):
    r, run, c = research
    with r.edit(run) as (s, db):
        s["cycles"] = []
        s["config"].update(first_cycle=20, total_iterations=1)
        r.begin_cycle(s, db)
        assert s["cycles"][-1]["full"] is False
        assert set(s["cycles"][-1]["targets"].values()) == {1}


def test_pause_finishes_when_worker_exits_between_poll_and_cleanup(
    research, monkeypatch
):
    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
        job.update(status="running", started_at=time.time())
        s["status"] = "pausing"

    class ExitingProcess:
        calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else -9

    controller.processes[(run, job["id"])] = ExitingProcess()
    monkeypatch.setattr(controller, "stop", lambda *args, **kwargs: None)
    for _ in range(3):
        controller.tick()
    state = r.read(run)
    assert state["status"] == "paused"
    assert state["jobs"][job["id"]]["status"] == "paused"
    assert state["branches"][branch]["completed"] == 0
    assert not controller.processes
