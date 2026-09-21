"""Finite-budget coordination, recovery and selection without SSH or GPUs."""

import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided.experiment import Coordinator, HOUR, NodeFailure


class Clock:
    def __init__(self):
        self.now = 100000.0

    def __call__(self):
        return self.now


class FakeRemote:
    ROOT = "/remote/repo"
    PYTHON = "/remote/python"
    HOSTS = {"4090": "first", "4070": "second"}

    def __init__(self, clock, *, teacher_mean=0, failing=None):
        self.clock, self.teacher_mean, self.failing = clock, teacher_mean, failing
        self.files, self.jobs, self.launches, self.uploads = {}, {}, [], []
        self.calls = []

    def upload(self, host, root, files, destination):
        self.uploads.append((host, list(files), destination))
        for name in files:
            self.files[host, str(Path(destination) / name)] = (
                Path(root) / name
            ).read_bytes()
        return "uploaded"

    def command(self, host, argv, **kwargs):
        path = argv[-1]
        task = json.loads(self.files[host, path])
        key = (host, path)
        if key not in self.jobs:
            finish = (
                task.get("deadline", self.clock())
                if task["kind"] == "train"
                else self.clock()
            )
            if task["kind"] == "evaluate":
                finish += 30
            self.jobs[key] = dict(task=task, finish=finish, state="running")
            self.launches.append((host, path, task))
        return json.dumps({"pid": 100 + len(self.launches)})

    def _result(self, host, path, task):
        name = Path(path).stem
        kind = task["kind"]
        if kind == "initialize":
            return {"checkpoint": task["output"]}
        if kind == "train":
            return {"checkpoint": task["run"] + "/latest.pt"}
        if kind == "export":
            actor = ("weights:" + task["checkpoint"]).encode()
            meta = dict(
                format="guided-actor-v1",
                weights_sha256=hashlib.sha256(actor).hexdigest(),
            )
            self.files[host, task["output"] + "/actor.pt"] = actor
            self.files[host, task["output"] + "/meta.json"] = json.dumps(meta).encode()
            return {"artifact": task["output"]}
        if kind == "benchmark":
            p95 = 110 if "scale-large" in name else 30
            load = dict(
                p95_ms=p95, p50_ms=20, p99_ms=p95 + 1, errors=0, illegal_actions=0
            )
            return dict(
                loads={"sustained": load, "synchronized_burst": load},
                passed_legality=True,
                gpu={"peak_allocated_bytes": 20, "total_bytes": 100},
            )
        if kind == "evaluate":
            mean = {
                "scale-small-evaluate": 1,
                "scale-medium-evaluate": 2,
                "startup-A-evaluate": 2,
                "startup-B-evaluate": 3,
                "final-4090": 3,
                "final-4070": 4,
            }[name]
            noise = [(i % 5 - 2) / 10 for i in range(task["games"] // 4)]
            return {
                "results": {
                    "model:current": {
                        "group_scores": [mean + n for n in noise],
                        "games": task["games"],
                    },
                    "teacher-final"
                    if name.startswith("final")
                    else "teacher-initial": {
                        "group_scores": [self.teacher_mean + n for n in noise],
                        "games": task["games"],
                    },
                }
            }
        raise AssertionError(kind)

    def execute(self, host, script, **kwargs):
        line = next(line for line in script.splitlines() if line.startswith("request="))
        request = ast.literal_eval(line[len("request=") :])
        self.calls.append((host, request))
        op = request["operation"]
        if op == "status":
            key = (host, request["path"])
            job = self.jobs.get(key)
            if job is None:
                return "null"
            if self.clock() >= job["finish"] and job["state"] == "running":
                if Path(request["path"]).stem == self.failing:
                    job.update(state="failed", error="deliberate arithmetic failure")
                else:
                    job.update(
                        state="complete",
                        result=self._result(host, request["path"], job["task"]),
                    )
            return json.dumps(
                {key: job[key] for key in ("state", "result", "error") if key in job}
            )
        if op == "download_actor":
            return json.dumps(
                {
                    name: base64.b64encode(
                        self.files[host, request["path"] + "/" + name]
                    ).decode()
                    for name in ("actor.pt", "meta.json")
                }
            )
        if op == "select_checkpoint":
            return json.dumps(
                {"checkpoint": request["run"] + "/best.pt", "selection": "best"}
            )
        if op == "freeze_bundle":
            return json.dumps(
                {
                    "manifest": "/remote/teachers/final/manifest.json",
                    "id": "final",
                    "teacher": "teacher-final",
                }
            )
        if op == "verify_bundle":
            return json.dumps({"id": "final", "teacher": "teacher-final"})
        if op == "terminate":
            return json.dumps({"stopped": True})
        raise AssertionError(op)


def coordinator(tmp_path, clock, remote):
    return Coordinator(
        tmp_path / "experiment",
        "/remote/frozen-code",
        "/remote/teachers/initial/manifest.json",
        "/remote/panel/manifest.json",
        clock=clock,
        remote_module=remote,
    )


def drive(instance, remote, clock, stop_stage=None):
    for _ in range(250):
        if stop_stage and instance.state["stage"] == stop_stage:
            return
        if instance.state["status"] == "complete":
            return
        instance.tick()
        pending = [
            job["finish"] for job in remote.jobs.values() if job["state"] == "running"
        ]
        if pending:
            clock.now = max(clock.now, min(pending))
    raise AssertionError("coordinator did not finish bounded mock workflow")


def test_construction_never_dispatches_and_resume_preserves_original_deadline(tmp_path):
    clock = Clock()
    remote = FakeRemote(clock)
    first = coordinator(tmp_path, clock, remote)
    assert first.state["status"] == "ready" and first.state["started"] is None
    assert not remote.calls and not remote.launches
    first.start(168)
    deadline = first.state["deadline"]
    first.tick()
    assert len(remote.launches) == 1
    clock.now += HOUR
    second = coordinator(tmp_path, clock, remote)
    second.start(999)
    assert second.state["deadline"] == deadline
    second.tick()
    # The initialization is recovered by durable remote status, never relaunched.
    assert (
        sum(
            Path(path).stem == "scale-small-initialize"
            for _, path, _ in remote.launches
        )
        == 1
    )


def test_whole_workflow_fairness_two_hosts_and_actor_only_transfer(tmp_path):
    clock = Clock()
    remote = FakeRemote(clock)
    instance = coordinator(tmp_path, clock, remote)
    instance.start()
    drive(instance, remote, clock)
    state = instance.state
    assert state["status"] == "complete"
    assert state["selection"]["scale"] == "medium"
    assert state["selection"]["route"] == "B"
    assert state["selection"]["kind"] == "actor"
    assert state["selection"]["winner"] == "4070"
    assert state["platform_started"] is False
    assert state["finished"] <= state["deadline"]
    tasks = {Path(path).stem: (host, task) for host, path, task in remote.launches}
    assert "scale-large-train" not in tasks  # preflight rejects before spending 3h
    a, b = tasks["startup-A"][1], tasks["startup-B"][1]
    assert a["initial"] == b["initial"]
    assert a["config"] == b["config"]
    assert a["bc_seconds"] == 0 and b["bc_seconds"] == HOUR
    assert a["updates"] is None and b["updates"] is None
    main_a, main_b = tasks["main-4090"][1], tasks["main-4070"][1]
    assert (
        main_a["carry"] and main_a["initial"] == state["selection"]["carry_checkpoint"]
    )
    assert not main_b["carry"] and main_b["initial"] is None
    assert main_a["deadline"] == main_b["deadline"] == state["deadline"] - 24 * HOUR
    assert main_a["config"]["seed"] != main_b["config"]["seed"]
    assert main_b["bc_seconds"] == HOUR
    assert main_a["updates"] == "/remote/teachers/current.json"
    for host in ("4090", "4070"):
        final = tasks["final-" + host][1]
        assert final["games"] == 8192 and final["namespace"] == "final-test"
        assert final["bundle"] == "/remote/teachers/final/manifest.json"
    assert tasks["final-4090"][1]["eval_seed"] == tasks["final-4070"][1]["eval_seed"]
    transfers = [files for _, files, _ in remote.uploads if "actor.pt" in files]
    assert transfers and all(
        set(files) == {"actor.pt", "meta.json"} for files in transfers
    )
    assert not list(instance.local.glob(".transfer-*"))


def test_teacher_remains_recommendation_when_it_has_the_highest_final_score(tmp_path):
    clock = Clock()
    remote = FakeRemote(clock, teacher_mean=5)
    instance = coordinator(tmp_path, clock, remote)
    instance.start()
    drive(instance, remote, clock)
    assert instance.state["selection"]["kind"] == "teacher"
    assert (
        instance.state["selection"]["recommendation_4070"]
        == "/remote/teachers/final/manifest.json"
    )
    assert instance.state["actor_artifact_role"] == "research"
    assert instance.state["artifact_4070"]


def test_non_oom_node_failure_stops_stage_instead_of_skipping_candidate(tmp_path):
    clock = Clock()
    remote = FakeRemote(clock, failing="scale-small-benchmark")
    instance = coordinator(tmp_path, clock, remote)
    instance.start()
    with pytest.raises(NodeFailure, match="arithmetic failure"):
        drive(instance, remote, clock)
    assert instance.state["status"] == "failed"
    assert not any("medium" in path for _, path, _ in remote.launches)
    restored = coordinator(tmp_path, clock, remote)
    with pytest.raises(NodeFailure):
        restored.tick()


def test_elapsed_budget_is_never_restarted_or_silently_shortens_trial(tmp_path):
    clock = Clock()
    remote = FakeRemote(clock)
    instance = coordinator(tmp_path, clock, remote)
    instance.start()
    deadline = instance.state["deadline"]
    clock.now += 10 * HOUR
    with pytest.raises(NodeFailure, match="full scale-small-train duration"):
        drive(instance, remote, clock)
    assert instance.state["deadline"] == deadline
    assert not any(
        Path(path).stem == "scale-small-train" for _, path, _ in remote.launches
    )


def test_insignificant_pairwise_difference_keeps_simpler_or_joint_route():
    # Same helper is used for scale and B-minus-A decisions, whose gate is lower>0.
    evidence = Coordinator._difference([3, -1, 3, -1], [0, 0, 0, 0])
    assert evidence["mean"] > 0 and evidence["lower"] < 0
    equal = Coordinator._difference([1, 2, 3], [1, 2, 3])
    assert equal["lower"] == 0


def test_main_jobs_are_both_launched_before_polling_for_completion(tmp_path):
    clock = Clock()
    remote = FakeRemote(clock)
    instance = coordinator(tmp_path, clock, remote)
    instance.start()
    drive(instance, remote, clock, stop_stage="main")
    instance.tick()
    main = [
        (host, task)
        for host, path, task in remote.launches
        if Path(path).stem.startswith("main-")
    ]
    assert {host for host, _ in main} == {"4090", "4070"}
    assert len({task["deadline"] for _, task in main}) == 1


def test_stale_coordinator_reloads_budget_after_acquiring_lock(tmp_path, monkeypatch):
    clock = Clock()
    remote = FakeRemote(clock)
    first = coordinator(tmp_path, clock, remote)
    stale = coordinator(tmp_path, clock, remote)
    first.start()
    original = first.state["deadline"]
    clock.now += HOUR

    def finish():
        stale.state["status"] = "complete"
        stale._save()

    monkeypatch.setattr(stale, "tick", finish)
    stale.run()
    assert stale.state["deadline"] == original


def test_failed_coordinator_lock_does_not_start_budget(tmp_path):
    import fcntl

    clock = Clock()
    instance = coordinator(tmp_path, clock, FakeRemote(clock))
    with (instance.local / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            instance.run()
    assert json.loads(instance.path.read_text())["started"] is None
