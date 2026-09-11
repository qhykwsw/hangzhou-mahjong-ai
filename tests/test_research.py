from pathlib import Path
import time
import pytest
from starlette.testclient import TestClient
from mahjong_ai.research.service import Research
from mahjong_ai.research.config import COMBINATIONS, configuration
from mahjong_ai.evaluation.schedule import (
    schedule,
    reference_groups,
    group_uncertainty,
)
from mahjong_ai.execution.controller import Controller
from mahjong_ai.execution.jobs import prepare, play
from mahjong_ai.interfaces.app import application


def small_config():
    return dict(
        total_rounds=1,
        iterations_per_round=1,
        evaluation_games=4,
        competition_games_per_branch=4,
        training_games=1,
        training_caps={"heuristic": 64},
        candidate_limit=4,
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
    return start_research(tmp_path)


@pytest.fixture(autouse=True)
def project_manifest(tmp_path):
    (tmp_path / "project.toml").write_text(
        '[project]\nid = "test-project"\nfamily = "test"\n'
    )


def start_research(tmp_path, config=None):
    r = Research(tmp_path)
    run = r.create(config or small_config())
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
    assert sum(c["branches"] for c in cfg["combinations"]) == 14
    assert all(c["models"] == ["deepseek", "qwen"] for c in cfg["combinations"])
    assert cfg["candidate_limit"] == 1
    assert cfg["opponent_limit"] == 4
    assert cfg["agent_concurrency"] == 14
    assert cfg["training_caps"] == {
        "heuristic": 0,
        "search": 0,
        "value": 256,
        "ppo": 256,
    }
    assert cfg["training_games"] == 128
    assert cfg["models"]["astra"]["effort"] == "medium"
    with pytest.raises(ValueError, match="effort"):
        configuration(tmp_path, {"models": {"qwen": {"effort": "medium"}}})
    with pytest.raises(ValueError, match="multiple"):
        configuration(tmp_path, {"evaluation_games": 5})
    with pytest.raises(ValueError, match="training_caps"):
        configuration(tmp_path, {"training_caps": {"heuristic": -1}})


def test_reduced_model_roster_preserves_branch_histories(tmp_path):
    r = Research(tmp_path)
    old = configuration(tmp_path)
    for combination in old["combinations"]:
        combination.update(branches=3, models=["astra", "deepseek", "qwen"])
        combination.pop("branch_ids")
    r.register(old)
    with r.edit() as (s, db):
        for branch in s["branches"].values():
            branch.update(thread_id="thread-" + branch["id"], completed=3)
    before = r.read()["branches"]
    for branch in before.values():
        if branch["model"] == "astra":
            r.archive(branch["id"])
    r.register()
    after = r.read()
    active = [b for b in after["branches"].values() if not b["archived"]]
    assert len(active) == 14
    assert not after["model_history"]
    for branch in active:
        assert branch == before[branch["id"]]
        assert branch["model"] in ("deepseek", "qwen")
    with pytest.raises(ValueError, match="branch_ids"):
        invalid = configuration(tmp_path)
        invalid["combinations"][0]["branch_ids"] = ["02", "02"]
        configuration(tmp_path, invalid)


def test_branch_isolation_atomic_submission_and_full_cycle(research):
    r, run, controller = research
    initial = r.read(run)
    assert initial["phase"] == "researching"
    branches = list(initial["branches"])
    for branch in branches:
        with r.edit(run) as (s, db):
            job = r.start_iteration(s, branch)
        r.declare_plan(
            run, branch, "unchanged parent", "copy the incumbent", "isolation"
        )
        r.fork(run, branch, "v1", "try unchanged parent")
        with pytest.raises(ValueError, match="属于本分支"):
            r.fork(
                run,
                branch,
                "bad",
                "steal",
                branches[-1 if branch != branches[-1] else 0] + "--initial",
            )
        with pytest.raises(ValueError, match="相对"):
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
    r.configure(run, {"match_workers": 3})
    assert r.read(run)["config"]["match_workers"] == 3
    with pytest.raises(ValueError, match="新建实验"):
        r.configure(run, {"evaluation_games": 8})
    with pytest.raises(ValueError, match="新建实验"):
        r.configure(run, {"game": {"rounds": 2}})
    r.control(run, "extend", 3)
    assert r.read(run)["config"]["total_rounds"] == 4


def test_opponents_sample_with_replacement_and_allow_self():
    assert reference_groups(["own"], 3, 2) == [["own"] * 3] * 2
    groups = reference_groups(["a", "b"], 3, 20)
    assert all(len(g) == 3 and set(g) <= {"a", "b"} for g in groups)
    assert groups == reference_groups(["a", "b"], 3, 20)


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
        assert client.get("/api/directions").json()["defaults"]["total_rounds"] == 10
        result = client.post("/api/experiments", json={"config": small_config()})
        assert result.status_code == 200
        run = result.json()["experiment"]["id"]
        assert (
            client.post(
                "/api/control",
                json={"experiment": run, "action": "pause"},
                headers={"Origin": "https://other.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/config", json={"experiment": run, "patch": {"match_workers": 3}}
            ).json()["config"]["match_workers"]
            == 3
        )


def test_pause_submission_recovery_does_not_repeat_agent(research):
    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
    r.declare_plan(run, branch, "recover", "resubmit", "pause recovery")
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


def test_training_cost_feedback_survives_pause_and_new_experiment(research):
    from mahjong_ai.execution.agent import agent_prompt
    from mahjong_ai.persistence.artifacts import atomic_json

    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        agent = r.start_iteration(s, branch)
    r.declare_plan(run, branch, "cost feedback", "measured run", "training budget")
    r.fork(run, branch, "measured", "adjust training using actual cost")
    r.submit(run, branch, ["measured"], training_games=48)
    with r.edit(run) as (s, db):
        r.accept(
            s,
            db,
            s["jobs"][agent["id"]],
            {"status": "completed", "usage": {"wall_seconds": 194.0}},
        )
        r.advance(s, db)
        prep = next(
            j
            for j in s["jobs"].values()
            if j["kind"] == "prepare" and j.get("branch") == branch
        )
        root = Path(s["root"]) / "jobs" / prep["id"]
        atomic_json(
            root / "result.json", {"status": "paused", "usage": {"wall_seconds": 45.0}}
        )
        controller.consume(s, db)
        controller.consume(s, db)  # Repeated receipt must not count twice.
        cid = branch + "--measured"
        spec = s["candidates"][s["branches"][branch]["representative"]]["spec"]
        r.accept(
            s,
            db,
            prep,
            {
                "status": "completed",
                "usage": {"wall_seconds": 300.0},
                "candidates": {
                    cid: {
                        "status": "ready",
                        "spec": spec,
                        "training": {
                            "status": "completed",
                            "episodes": [{}] * 48,
                            "reported_metrics": {"updates": 11},
                        },
                    }
                },
            },
        )
        for job in list(s["jobs"].values()):
            if job["kind"] == "match" and job.get("branch") == branch:
                r.accept(
                    s,
                    db,
                    job,
                    {
                        "status": "completed",
                        "usage": {"wall_seconds": 9000.0},
                        "summary": [],
                        "focal": {
                            "candidate_id": job["candidate"],
                            "appearances": 4,
                            "mean_utility": 1.5,
                        },
                    },
                )
    feedback = r.snapshot(run, branch)["iterations"][-1]["execution_feedback"]
    assert feedback["training_games"] == 48
    assert feedback["agent_seconds"] == 194
    assert feedback["preparation_seconds"] == 345
    assert feedback["execution_seconds"] == 539
    assert feedback["execution_limit_seconds"] == 3600
    assert feedback["training"][0]["completed_games"] == 48
    assert feedback["training"][0]["updates"] == 11
    r.control(run, "pause")
    controller.tick()
    r.control(run, "close")
    run2 = r.create(small_config() | {"iteration_seconds": 7200}, targets=[branch])
    r.control(run2, "resume")
    with r.edit(run2) as (s, db):
        r.advance(s, db)
        job = r.start_iteration(s, branch)
        task = controller.task(s, job)
    assert task["previous_iteration_feedback"] == feedback
    assert task["previous_plan"]["direction"] == "measured run"
    assert '"assessment": "cost feedback"' in agent_prompt(task)
    assert '"preparation_seconds": 345.0' in agent_prompt(task)
    assert "自行调整本次 training_games" in agent_prompt(task)
    assert len(r.snapshot(run2, branch)["iterations"]) == 2
    own = r.snapshot(run2, branch)
    assert all(it["branch"] == branch for it in own["iterations"])
    assert "workspace" not in str(feedback) and "runtime_path" not in str(feedback)


def test_snapshot_exposes_live_branch_progress_without_private_paths(research):
    r, run, _ = research
    snapshot = r.snapshot(run)
    assert len(snapshot["branch_progress"]) == len(snapshot["branches"])
    for progress in snapshot["branch_progress"].values():
        assert "preparation" in progress and "evaluation" in progress
        assert "workspace" not in str(progress)
        assert "runtime_path" not in str(progress)


def test_heuristic_family_has_no_training_slot(tmp_path):
    config = small_config()
    config["training_caps"] = {"heuristic": 0}
    r, run, _ = start_research(tmp_path, config)
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        r.start_iteration(s, branch)
    r.declare_plan(run, branch, "no training", "rule tweak", "heuristics do not learn")
    r.fork(run, branch, "plain", "rule tweak")
    r.fork(run, branch, "hooked", "rule tweak")
    state = r.read(run)
    source = (
        Path(state["candidates"][branch + "--plain"]["workspace"]) / "strategy.py"
    ).read_text()
    r.candidate(
        run,
        branch,
        "hooked",
        path="strategy.py",
        content=source + "\n\ndef train(ctx):\n    ctx.play(None)\n",
    )
    check = r.check(run, branch, "hooked")
    assert check["ok"] is False
    assert "没有训练环节" in check["error"]
    with pytest.raises(ValueError, match="没有训练环节"):
        r.submit(run, branch, ["plain"], training_games=8)
    r.submit(run, branch, ["plain"])
    iteration = r.read(run)["iterations"][state["branches"][branch]["iteration"]]
    assert iteration["training_games"] == 0


def test_partial_prepare_failure_evaluates_all_successes(research):
    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
        parent = s["candidates"][s["branches"][branch]["representative"]]
        source = (Path(parent["workspace"]) / "strategy.py").read_text()
    r.declare_plan(run, branch, "partial batch", "two drafts", "failure isolation")
    for name in ("good", "bad"):
        r.fork(run, branch, name, "partial batch")
    r.candidate(
        run,
        branch,
        "bad",
        path="strategy.py",
        content=source
        + "\n\ndef train(ctx):\n"
        + "    if ctx.episodes > 1:\n"
        + "        raise RuntimeError('intentional preparation failure')\n",
    )
    # The draft runs for one episode in the submit check and only fails in the
    # real preparation job, which is what this test covers.
    r.submit(run, branch, ["good", "bad"], training_games=4)
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


def test_iteration_plan_is_required_before_drafts_and_exposed(research):
    r, run, _ = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        r.start_iteration(s, branch)
    with pytest.raises(ValueError, match="research_declare_plan"):
        r.fork(run, branch, "unplanned", "code before judgement")
    with pytest.raises(ValueError, match="非空"):
        r.declare_plan(run, branch, "", "direction only", "reason")
    r.declare_plan(
        run, branch, "现役基线不设防", "收紧终局放铳惩罚", "上一轮大样本证明防守有收益"
    )
    r.fork(run, branch, "planned", "one knob at a time")
    r.submit(run, branch, ["planned"])
    plan = next(b for b in r.snapshot(run)["branches"] if b["id"] == branch)["plan"]
    assert plan["assessment"] == "现役基线不设防"
    assert plan["direction"] == "收紧终局放铳惩罚"
    assert plan["iteration"] == 1 and plan["status"] == "submitted"


def test_each_iteration_starts_a_fresh_thread_with_handoff(research):
    from mahjong_ai.execution.agent import agent_prompt

    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        first = r.start_iteration(s, branch)
        s["iterations"][first["iteration"]].update(
            status="completed", improved=True, spent=600.0
        )
        s["branches"][branch].update(thread_id="thread-1", status="ready", completed=1)
        s["cycles"][-1]["targets"][branch] = 2
    with r.edit(run) as (s, db):
        second = r.start_iteration(s, branch)
        task = controller.task(s, s["jobs"][second["id"]])
    state = r.read(run)
    assert state["iterations"][first["iteration"]]["thread_id"] == "thread-1"
    assert state["branches"][branch]["thread_id"] is None
    assert task["thread_id"] is None
    handoff = task["handoff"]
    assert handoff["branch"] == branch
    assert handoff["recent_iterations"][-1]["iteration"] == 1
    assert (
        handoff["incumbent"]["candidate_id"]
        == state["branches"][branch]["representative"]
    )
    assert "结构化交接" in agent_prompt(task)


def test_repeated_create_returns_existing_candidate(research):
    """A dropped tool reply must not turn a retry into a conflict error."""

    r, run, _ = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        r.start_iteration(s, branch)
    r.declare_plan(run, branch, "retry", "same draft", "channel flake")
    first = r.fork(run, branch, "retried", "same intent")
    retry = r.fork(run, branch, "retried", "same intent")
    assert first == {
        "candidate_id": branch + "--retried",
        "status": "draft",
        "existing": False,
    }
    assert retry["candidate_id"] == first["candidate_id"]
    assert retry["existing"] is True and retry["status"] == "draft"
    r.submit(run, branch, ["retried"])
    late = r.fork(run, branch, "retried", "late retry after submission")
    assert late["existing"] is True and late["status"] == "submitted"
    assert "新名字" in late["note"]
    assert len([c for c in r.snapshot(run)["candidates"] if c["branch"] == branch]) == 2


def test_submission_rejects_draft_that_does_not_run(research):
    r, run, _ = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        r.start_iteration(s, branch)
    r.declare_plan(run, branch, "broken module", "submit anyway", "run check")
    r.fork(run, branch, "broken", "module without a build entry point")
    r.candidate(
        run,
        branch,
        "broken",
        path="strategy.py",
        content="COMBINATION = 'heuristic-balanced'\n",
    )
    check = r.check(run, branch, "broken")
    assert check["ok"] is False and "build" in check["error"]
    with pytest.raises(ValueError, match="运行自检"):
        r.submit(run, branch, ["broken"])
    state = r.read(run)
    assert state["candidates"][branch + "--broken"]["status"] == "draft"
    iteration = state["iterations"][state["branches"][branch]["iteration"]]
    assert iteration["status"] == "researching"
    root = (
        Path(r.project)
        / ".local/strategies"
        / state["branches"][branch]["combination"]
        / branch
        / "submissions"
        / iteration["id"]
    )
    assert not root.exists()


def test_64_branches_use_dynamic_agent_slots(tmp_path, monkeypatch):
    r = Research(tmp_path)
    cfg = configuration(tmp_path)
    cfg["combinations"] = [
        dict(id=key, branches=8, models=["deepseek"] * 8, parameters=params)
        for key, (_, _, params) in list(COMBINATIONS.items())[:8]
    ]
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
    controller.agent_launch_interval = 0
    controller.tick()
    controller.tick()
    assert len(launched) == 14 and len(set(launched)) == 14


def test_agent_launches_are_paced(tmp_path, monkeypatch):
    r = Research(tmp_path)
    cfg = configuration(tmp_path)
    cfg["combinations"] = [
        dict(
            id=key,
            branches=2,
            models=["deepseek"] * 2,
            parameters=params,
        )
        for key, (_, _, params) in list(COMBINATIONS.items())[:4]
    ]
    run = r.create(cfg)
    controller = Controller(r)
    r.control(run, "resume")
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
        controller.agent_launch_at = time.time()

    monkeypatch.setattr(controller, "launch", launch)
    controller.tick()
    assert len(launched) == 1
    controller.tick()
    assert len(launched) == 1
    controller.agent_launch_at = time.time() - controller.agent_launch_interval
    controller.tick()
    assert len(launched) == 2


def test_custom_combination_and_model_are_configurable(tmp_path):
    cfg = small_config()
    cfg["combinations"][0].update(
        id="custom-shape", family="heuristic", name="Custom shape"
    )
    r = Research(tmp_path)
    run = r.create(cfg)
    assert len(r.read(run)["branches"]) == 4
    assert "custom-shape" in r.read(run)["definitions"]


def test_submission_is_idempotent_and_releases_slot(research):
    r, run, controller = research
    branch = next(iter(r.read(run)["branches"]))
    with r.edit(run) as (s, db):
        job = r.start_iteration(s, branch)
    r.declare_plan(run, branch, "one batch", "idempotent submit", "slot release")
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


def test_total_rounds_and_inner_iterations_are_independent(research):
    r, run, c = research
    with r.edit(run) as (s, db):
        s["cycles"] = []
        s["config"].update(total_rounds=2, iterations_per_round=3)
        r.begin_cycle(s, db)
        assert s["cycles"][-1]["number"] == 1
        assert set(s["cycles"][-1]["targets"].values()) == {3}


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


def finish_branch_with_receipts(r, experiment, controller, branch, name, improve=True):
    """Drive real submission/selection with deterministic executor receipts."""

    with r.edit(experiment) as (s, db):
        job = r.start_iteration(s, branch)
    r.declare_plan(experiment, branch, "controlled", "single variant", "receipts")
    candidate = r.fork(experiment, branch, name, "controlled experiment")[
        "candidate_id"
    ]
    r.submit(experiment, branch, [candidate])
    with r.edit(experiment) as (s, db):
        r.accept(s, db, s["jobs"][job["id"]], {"status": "completed"})
        r.advance(s, db)
        it = s["iterations"][s["branches"][branch]["iteration"]]
        preparation = next(
            j
            for j in s["jobs"].values()
            if j.get("iteration") == it["id"] and j["kind"] == "prepare"
        )
        spec = {**s["candidates"][it["incumbent"]]["spec"], "version": candidate}
        r.accept(
            s,
            db,
            preparation,
            {
                "status": "completed",
                "candidates": {candidate: {"status": "ready", "spec": spec}},
            },
        )
        matches = [
            j
            for j in s["jobs"].values()
            if j.get("iteration") == it["id"] and j["kind"] == "match"
        ]
        tasks = [controller.task(s, j) for j in matches]
        assert tasks[0]["seed"] == tasks[1]["seed"]
        for task in tasks:
            assert all(len(a["players"]) == 4 for a in task["assignments"])
        for job in matches:
            mean = 2 if job["candidate"] == candidate and improve else 1
            focal = dict(
                candidate_id=job["candidate"],
                appearances=4,
                utility_sum=mean * 4,
                mean_utility=mean,
                errors=0,
                timeouts=0,
                fallback_actions=0,
            )
            r.accept(s, db, job, {"status": "completed", "summary": [], "focal": focal})
        r.advance(s, db)
    return candidate


def finish_comparison(r, experiment):
    from mahjong_ai.evaluation.schedule import standings

    with r.edit(experiment) as (s, db):
        assert s["phase"] == "competing"
        cycle = s["cycles"][-1]
        for job in [
            j
            for j in s["jobs"].values()
            if j.get("scope") == "competition" and j["cycle"] == cycle["id"]
        ]:
            receipts = [
                {"players": a["players"], "ranks": [1, 2, 3, 4]}
                for a in job["assignments"]
            ]
            summary = standings(
                {p for a in job["assignments"] for p in a["players"]}, receipts
            )
            r.accept(s, db, job, {"status": "completed", "summary": summary})
        r.advance(s, db)


def test_selective_iterations_all_eligible_compete_and_top_k_automatically_refresh(
    research,
):
    r, experiment, controller = research
    ids = list(r.read(experiment)["branches"])
    with r.edit(experiment) as (s, db):
        # One already evaluated branch remains a competitor, but is not iterated.
        s["branches"][ids[1]]["successful"] = 1
        s["branches"][ids[1]]["completed"] = 1
    r.control(experiment, "pause")
    controller.tick()
    r.control(experiment, "close")
    config = small_config() | {"total_rounds": 2, "opponent_limit": 1}
    experiment = r.create(config, targets=[ids[0]], opponents=[ids[1] + "--initial"])
    r.control(experiment, "resume")
    with r.edit(experiment) as (s, db):
        r.advance(s, db)
        assert list(s["cycles"][-1]["targets"]) == [ids[0]]
    first = finish_branch_with_receipts(r, experiment, controller, ids[0], "first")
    with r.edit(experiment) as (s, db):
        cycle = s["cycles"][-1]
        assert set(cycle["roster"]) == {ids[0], ids[1]}
        assert cycle["reference"] == [ids[1] + "--initial"]
        assignments = [
            a
            for j in s["jobs"].values()
            if j.get("cycle") == cycle["id"] and j.get("scope") == "competition"
            for a in j["assignments"]
        ]
        assert any(p in s["bootstrap"] for a in assignments for p in a["players"])
    finish_comparison(r, experiment)
    s = r.read(experiment)
    assert s["phase"] == "researching"
    assert s["cycles"][-1]["number"] == 2
    assert s["cycles"][-1]["reference"] == [s["last_ranking"][0]["candidate_id"]]
    assert not any(cid in s["bootstrap"] for cid in s["cycles"][-1]["reference"])
    finish_branch_with_receipts(
        r, experiment, controller, ids[0], "second", improve=False
    )
    finish_comparison(r, experiment)
    s = r.read(experiment)
    assert s["phase"] == "finished"
    assert s["branches"][ids[0]]["representative"] == first
    assert s["branches"][ids[0]]["completed"] == 2
    assert s["branches"][ids[1]]["completed"] == 1
    assert s["branches"][ids[2]]["completed"] == 0
    assert s["experiments"][-1]["completed_rounds"] == 2
    assert len(s["last_ranking"]) == 2


def test_new_experiment_preserves_branch_history_and_rejects_old_sessions(research):
    r, experiment, controller = research
    branch = next(iter(r.read(experiment)["branches"]))
    original = r.read(experiment)["branches"][branch]["representative"]
    r.control(experiment, "pause")
    controller.tick()
    r.control(experiment, "close")
    assert all(
        b["status"] in ("ready", "uninitialized") for b in r.read()["branches"].values()
    )
    second = r.create(
        small_config(), targets=[branch], opponents=["bootstrap--balanced"]
    )
    assert second != experiment
    state = r.read(second)
    assert state["branches"][branch]["representative"] == original
    assert len(state["branches"]) == 4
    assert len(state["experiments"]) == 2
    assert (
        "/.local/strategies/heuristic-balanced/"
        in state["candidates"][original]["spec"]["package_path"]
    )
    assert not (r.project / ".local/runs").exists()
    with pytest.raises(ValueError, match="已不是当前实验"):
        r.fork(experiment, branch, "stale", "old agent")
    with pytest.raises(ValueError, match="当前实验"):
        r.create(small_config())


def test_registry_does_not_train_unselected_combinations(tmp_path):
    r = Research(tmp_path)
    r.register()
    assert not r.read()["jobs"]
    assert len(r.read()["bootstrap"]) == 3
    experiment = r.create(targets=["heuristic-balanced--02"])
    state = r.read(experiment)
    assert set(state["baselines"]) == {"heuristic-balanced"}
    assert len(state["jobs"]) == 1
    assert state["branches"]["ppo--02"]["status"] == "uninitialized"


def test_evaluation_shards_split_rotation_blocks_and_merge_into_one_result(tmp_path):
    config = small_config() | {"evaluation_games": 8, "evaluation_shards": 2}
    r, experiment, controller = start_research(tmp_path, config)
    branch = next(iter(r.read(experiment)["branches"]))
    with r.edit(experiment) as (s, db):
        agent = r.start_iteration(s, branch)
    r.declare_plan(experiment, branch, "shards", "split evaluation", "parallel")
    candidate = r.fork(experiment, branch, "sharded", "split work")["candidate_id"]
    r.submit(experiment, branch, [candidate])
    with r.edit(experiment) as (s, db):
        r.accept(s, db, s["jobs"][agent["id"]], {"status": "completed"})
        r.advance(s, db)
        it = s["iterations"][s["branches"][branch]["iteration"]]
        preparation = next(
            j
            for j in s["jobs"].values()
            if j.get("iteration") == it["id"] and j["kind"] == "prepare"
        )
        spec = {**s["candidates"][it["incumbent"]]["spec"], "version": candidate}
        r.accept(
            s,
            db,
            preparation,
            {
                "status": "completed",
                "candidates": {candidate: {"status": "ready", "spec": spec}},
            },
        )
        matches = [
            j
            for j in s["jobs"].values()
            if j.get("iteration") == it["id"] and j["kind"] == "match"
        ]
        assert len(matches) == 4  # 两个候选 × 两个分片
        for cid in (it["incumbent"], candidate):
            shards = [j for j in matches if j["candidate"] == cid]
            assert sorted(j["shard"] for j in shards) == [0, 1]
            assert all(j["shards"] == 2 for j in shards)
            assert sorted(
                (a["block"], a["focal"]) for j in shards for a in j["assignments"]
            ) == [(block, seat) for block in (0, 1) for seat in range(4)]
        assert len({controller.task(s, j)["seed"] for j in matches}) == 1

        def focal(job, mean):
            return dict(
                candidate_id=job["candidate"],
                appearances=4,
                utility_sum=mean * 4,
                mean_utility=mean,
                errors=0,
                timeouts=0,
                fallback_actions=0,
                group_means=[mean],
            )

        first = matches[0]
        r.accept(
            s,
            db,
            first,
            {"status": "completed", "summary": [], "focal": focal(first, 1.0)},
        )
        partial = s["iterations"][it["id"]]["results"][first["candidate"]]
        assert partial["appearances"] == 4 and partial["evaluation_groups"] == 1
        for job in matches:
            if job["id"] == first["id"]:
                continue
            r.accept(
                s,
                db,
                job,
                {"status": "completed", "summary": [], "focal": focal(job, 2.0)},
            )
        assert s["iterations"][it["id"]]["results"][first["candidate"]] == {
            "candidate_id": first["candidate"],
            "appearances": 8,
            "utility_sum": 12.0,
            "errors": 0,
            "timeouts": 0,
            "fallback_actions": 0,
            "mean_utility": 1.5,
            **group_uncertainty([1.0, 2.0]),
        }
        assert s["branches"][branch]["completed"] == 1


def test_focal_scoring_excludes_same_version_in_opponent_seats(tmp_path):
    from mahjong_ai.persistence.artifacts import atomic_json

    assignments = []
    for seat in range(4):
        assignments.append(dict(index=seat, block=0, focal=seat, players=["same"] * 4))
        atomic_json(
            tmp_path / "matches" / f"{seat:06d}.json",
            {"players": ["same"] * 4, "ranks": [1, 2, 3, 4]},
        )
    result = play(
        {"assignments": assignments, "policies": {"same": {}}, "candidate": "same"},
        tmp_path,
    )
    assert result["summary"][0]["appearances"] == 16
    assert result["focal"]["appearances"] == 4
    assert result["focal"]["mean_utility"] == 1.5


def test_same_branch_can_be_only_training_and_evaluation_opponent(research):
    r, experiment, controller = research
    branch = next(iter(r.read(experiment)["branches"]))
    with r.edit(experiment) as (s, db):
        own = s["branches"][branch]["representative"]
        s["cycles"][-1]["reference"] = [own]
        agent = r.start_iteration(s, branch)
    r.declare_plan(experiment, branch, "self play", "own opponent", "opponent pool")
    r.fork(experiment, branch, "self", "self opposition")
    r.submit(experiment, branch, ["self"])
    with r.edit(experiment) as (s, db):
        r.accept(s, db, s["jobs"][agent["id"]], {"status": "completed"})
        r.advance(s, db)
        job = next(
            j
            for j in s["jobs"].values()
            if j["kind"] == "prepare" and j.get("branch") == branch
        )
        assert controller.task(s, job)["opponents"] == [s["candidates"][own]["spec"]]


def test_old_opponents_obey_current_experiment_action_windows(research):
    r, experiment, controller = research
    branch = next(iter(r.read(experiment)["branches"]))
    with r.edit(experiment) as (s, db):
        s["config"]["timeouts"] = {"discard": 2.0, "response": 0.5}
        s["cycles"][-1]["config"]["timeouts"] = dict(s["config"]["timeouts"])
        agent = r.start_iteration(s, branch)
    r.declare_plan(experiment, branch, "action windows", "same policy", "budgets")
    r.fork(experiment, branch, "windows", "uniform action budgets")
    r.submit(experiment, branch, ["windows"])
    with r.edit(experiment) as (s, db):
        r.accept(s, db, s["jobs"][agent["id"]], {"status": "completed"})
        r.advance(s, db)
        prep = next(
            j
            for j in s["jobs"].values()
            if j["kind"] == "prepare" and j.get("branch") == branch
        )
        assert all(
            p["action_seconds"] == 2 and p["response_seconds"] == 0.5
            for p in controller.task(s, prep)["opponents"]
        )
        initial = s["branches"][branch]["representative"]
        spec = s["candidates"][initial]["spec"]
        r.accept(
            s,
            db,
            prep,
            {
                "status": "completed",
                "candidates": {branch + "--windows": {"status": "ready", "spec": spec}},
            },
        )
        jobs = [j for j in s["jobs"].values() if j["kind"] == "match"]
        assert jobs
        for job in jobs:
            assert all(
                p["action_seconds"] == 2 and p["response_seconds"] == 0.5
                for p in controller.task(s, job)["policies"].values()
            )
        assert s["candidates"][initial]["spec"]["action_seconds"] == 3


def test_all_failed_branches_use_bootstrap_comparison_and_finish_round(research):
    r, experiment, controller = research
    for branch in r.read(experiment)["branches"]:
        with r.edit(experiment) as (s, db):
            job = r.start_iteration(s, branch)
            r.accept(s, db, job, {"status": "failed", "error": "no candidate"})
            r.advance(s, db)
    state = r.read(experiment)
    assert state["phase"] == "competing"
    assert not state["cycles"][-1]["roster"]
    finish_comparison(r, experiment)
    state = r.read(experiment)
    assert state["phase"] == "finished" and not state["last_ranking"]
    assert r.snapshot()["default_opponents"] == state["bootstrap"]
