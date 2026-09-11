"""Durable admission, process ownership and independent Agent/training/match slots."""

from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from mahjong_ai.persistence.artifacts import atomic_json, read_json, digest
from mahjong_ai.domain.engine import derive_seed
import json

# 逐个放出 Agent，避免同一秒把整个并发上限压给模型网关触发限流。
AGENT_LAUNCH_INTERVAL = 5.0


class Controller:
    def __init__(self, research):
        self.research = research
        self.processes = {}
        self.agent_launch_interval = AGENT_LAUNCH_INTERVAL
        self.agent_launch_at = 0.0
        self.recover()

    def recover(self):
        for run in self.research.states():
            with self.research.edit(run["id"]) as (s, db):
                self.consume(s, db)
                for j in s["jobs"].values():
                    if j["status"] in ("running", "admitted"):
                        self.stop(s, j, force=True)
                        root = Path(s["root"]) / "jobs" / j["id"]
                        if j["kind"] == "agent":
                            log = root / "codex.jsonl"
                            if log.exists():
                                for line in log.read_text(
                                    errors="replace"
                                ).splitlines():
                                    try:
                                        event = json.loads(line)
                                    except ValueError:
                                        continue
                                    if event.get("type") == "thread.started":
                                        s["branches"][j["branch"]]["thread_id"] = (
                                            event.get("thread_id")
                                        )
                        if j.get("iteration") and j["kind"] in ("agent", "prepare"):
                            task = root / "task.json"
                            ended = (
                                min(time.time(), read_json(task)["deadline"])
                                if task.exists()
                                else time.time()
                            )
                            elapsed = max(0, ended - j.get("started_at", ended))
                            j["spent"] += elapsed
                            s["iterations"][j["iteration"]]["spent"] += elapsed
                        if (
                            j["kind"] == "agent"
                            and s["iterations"][j["iteration"]]["status"] == "submitted"
                        ):
                            j["status"] = "completed"
                        else:
                            j["status"] = "paused"
                if s["status"] in ("running", "pausing"):
                    s.update(
                        status="paused",
                        pause_reason="服务已重启，继续实验以恢复",
                    )

    def stop(self, s, job, force=False):
        root = Path(s["root"]) / "jobs" / job["id"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "cancel.flag").touch()
        job.setdefault("stop_requested", time.time())
        for file in ("codex-process.json", "process.json"):
            p = root / file
            if not p.exists():
                continue
            pid = read_json(p)["pid"]
            command = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
            ).stdout
            if (
                file == "codex-process.json"
                and "codex" in command
                and job["experiment"] in command
            ) or (str(root) in command and "mahjong_ai." in command):
                try:
                    os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def consume(self, s, db):
        for j in list(s["jobs"].values()):
            if j["status"] == "completed" or j["experiment"] != s["id"]:
                continue
            root = Path(s["root"]) / "jobs" / j["id"]
            receipt = root / "result.json"
            if not receipt.exists():
                continue
            result = read_json(receipt)
            if j.get("result_hash") == digest(result):
                continue
            j["result_hash"] = digest(result)
            if (
                result["status"] == "paused"
                and j["kind"] == "agent"
                and result.get("thread_id")
            ):
                s["branches"][j["branch"]]["thread_id"] = result["thread_id"]
            if (
                result["status"] == "paused"
                and j["kind"] == "agent"
                and s["iterations"][j["iteration"]]["status"] == "submitted"
            ):
                result = {**result, "status": "completed"}
            if result["status"] == "paused":
                elapsed = result.get("usage", {}).get("wall_seconds", 0.0)
                j["spent"] += elapsed
                if j.get("iteration") and j["kind"] in ("agent", "prepare"):
                    s["iterations"][j["iteration"]]["spent"] += elapsed
            self.research.accept(s, db, j, result)

    def task(self, s, j):
        root = Path(s["root"])
        key = read_json(root / "private/random.json")["key"]
        it = s["iterations"].get(j.get("iteration"))
        cfg = it["config"] if it else s["config"]

        def policy_spec(cid, config):
            spec = {
                **s["candidates"][cid]["spec"],
                "action_seconds": config["timeouts"]["discard"],
                "response_seconds": config["timeouts"]["response"],
            }
            # An explicit new experiment may evaluate frozen packages against
            # a freshly frozen framework runtime. Older experiments keep their
            # own runtime path exactly as recorded.
            if s.get("candidate_runtime_path"):
                spec["runtime_path"] = s["candidate_runtime_path"]
            return spec

        task = dict(
            kind=j["kind"],
            project=str(self.research.project),
            experiment=s["id"],
            runtime_path=j["runtime_path"],
            runtime=s["runtime"],
            config=cfg["game"],
            timeouts=cfg["timeouts"],
            resources=cfg["resources"],
            seed=derive_seed(key, j["id"]),
        )
        if j["kind"] == "agent":
            b = s["branches"][j["branch"]]
            comb = s["definitions"][b["combination"]]
            previous = next(
                (
                    i
                    for i in reversed(list(s["iterations"].values()))
                    if i["branch"] == b["id"] and i["status"] == "completed"
                ),
                None,
            )
            task.update(
                branch=b["id"],
                thread_id=b["thread_id"],
                model_config=b["model_config"],
                brief=comb["name"] + ": " + str(comb["parameters"]),
                iteration=it["number"],
                handoff=self.research.handoff(s, b["id"]),
                previous_plan=(previous or {}).get("plan"),
                previous_iteration_feedback=(
                    self.research.iteration_feedback(s, previous) if previous else None
                ),
            )
        elif j["kind"] == "prepare":
            packages = {}
            for cid in j["candidates"]:
                c = s["candidates"][cid]
                source = Path(c["workspace"])
                packages[cid] = dict(
                    input=str(source),
                    hash=__import__(
                        "mahjong_ai.execution.candidate_runtime", fromlist=["tree_hash"]
                    ).tree_hash(source),
                    requirements=c["requirements"],
                    output=str(
                        self.research.layout.strategies
                        / c["combination"]
                        / (c["branch"] or "baseline")
                        / "versions"
                        / cid
                    ),
                )
            pool = (
                s["cycles"][it["cycle"] - 1]["reference"]
                if it
                else s["experiments"][-1]["initial_opponents"]
            )
            opponents = [policy_spec(cid, cfg) for cid in pool]
            task.update(
                packages=packages,
                dependency_root=str(root / "environments"),
                combination=j["combination"],
                definition=s["definitions"][j["combination"]],
                training_games=j["training_games"],
                opponents=opponents,
            )
        else:
            cycle = s["cycles"][j["cycle"] - 1]
            cfg = cycle["config"]
            task.update(
                config=cfg["game"],
                timeouts=cfg["timeouts"],
                assignments=j["assignments"],
                scope=j["scope"],
                candidate=j.get("candidate"),
                policies={
                    cid: policy_spec(cid, cfg)
                    for a in j["assignments"]
                    for cid in a["players"]
                },
            )
            # Identical scheduling keys across incumbent and variants; no candidate ID enters wall/policy seed.
            task["seed"] = derive_seed(
                key, "evaluation", j["cycle"], j.get("iteration", "competition")
            )
        return task

    def launch(self, s, j):
        root = Path(s["root"]) / "jobs" / j["id"]
        root.mkdir(parents=True, exist_ok=True)
        if (root / "result.json").exists():
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            j["attempt"] = int(j.get("attempt", 0)) + 1
            archive = attempts / str(time.time_ns())
            archive.mkdir()
            for name in ("result.json", "progress.json", "matches"):
                path = root / name
                if path.exists():
                    path.rename(archive / name)
        (root / "cancel.flag").unlink(missing_ok=True)
        task = (
            read_json(root / "task.json")
            if (root / "task.json").exists()
            else self.task(s, j)
        )
        it = s["iterations"].get(j.get("iteration"))
        if j["kind"] == "match":
            remaining = 86400 * 7
        elif it:
            remaining = max(0.01, it["config"]["iteration_seconds"] - it["spent"])
        else:
            remaining = max(0.01, s["config"]["initialization_seconds"] - j["spent"])
        task.update(deadline=time.time() + remaining, owner_pid=os.getpid())
        if j["kind"] == "agent":
            task["thread_id"] = s["branches"][j["branch"]]["thread_id"]
        atomic_json(root / "task.json", task)
        module = (
            "mahjong_ai.execution.agent"
            if j["kind"] == "agent"
            else "mahjong_ai.execution.jobs"
        )
        env = dict(
            os.environ,
            PYTHONPATH=j["runtime_path"],
            MAHJONG_MEMORY_MB=str(task["resources"]["memory_mb"]),
        )
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = str(task["resources"]["threads"])
        with (root / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", "-m", module, str(root)],
                cwd=self.research.project,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        j.update(status="running", pid=process.pid, started_at=time.time())
        j.pop("stop_requested", None)
        if j["kind"] == "agent":
            self.agent_launch_at = time.time()
        self.processes[(s["id"], j["id"])] = process
        atomic_json(root / "process.json", {"pid": process.pid})

    def tick(self):
        admissions = []
        for run in self.research.states(active_only=True):
            with self.research.edit(run["id"]) as (s, db):
                self.consume(s, db)
                for j in s["jobs"].values():
                    process = self.processes.get((s["id"], j["id"]))
                    root = Path(s["root"]) / "jobs" / j["id"]
                    if (
                        j["status"] == "running"
                        and process is not None
                        and process.poll() is not None
                        and not (root / "result.json").exists()
                    ):
                        atomic_json(
                            root / "result.json",
                            {
                                "status": "paused"
                                if s["status"] != "running"
                                else "failed",
                                "error": "任务进程退出但没有留下回执",
                                "usage": {
                                    "wall_seconds": max(
                                        0, time.time() - j["started_at"]
                                    )
                                },
                            },
                        )
                    if j["status"] == "running" and j["kind"] == "agent":
                        it = s["iterations"][j["iteration"]]
                        if (
                            it["status"] == "submitted"
                            and time.time() - it.get("submitted_at", time.time()) > 3
                        ):
                            (root / "submitted.flag").touch()
                    if (
                        j["status"] == "running"
                        and s["status"] == "running"
                        and (root / "task.json").exists()
                    ):
                        deadline = read_json(root / "task.json")["deadline"]
                        # Training that hits the iteration limit still needs a
                        # window to freeze its last committed checkpoint.
                        grace = 180 if j["kind"] == "prepare" else 6
                        if time.time() > deadline + grace:
                            self.stop(s, j, force=True)
                            if not (root / "result.json").exists():
                                result = {
                                    "status": "failed",
                                    "error": "执行超过时限",
                                    "usage": {
                                        "wall_seconds": max(
                                            0, deadline - j.get("started_at", deadline)
                                        )
                                    },
                                }
                                if j["kind"] == "prepare":
                                    result["candidates"] = {
                                        cid: read_json(root / cid / "artifact.json")
                                        for cid in j["candidates"]
                                        if (root / cid / "artifact.json").exists()
                                    }
                                atomic_json(root / "result.json", result)
                    # 作业已离开运行态（被取消/暂停）或整台实验不再运行时，
                    # 都要真正收掉还在跑的进程，否则迭代删了进程还在算。
                    if (
                        process is not None
                        and process.poll() is None
                        and (
                            j["status"] not in ("running", "admitted")
                            or s["status"] != "running"
                        )
                    ):
                        self.stop(
                            s,
                            j,
                            force=time.time() - j.get("stop_requested", time.time())
                            > 5,
                        )
                if s["status"] == "pausing" and not any(
                    j["status"] in ("running", "admitted") for j in s["jobs"].values()
                ):
                    s["status"] = "paused"
                if s["status"] != "running":
                    continue
                self.research.advance(s, db)
                if s["status"] != "running":
                    continue
                limits = {
                    "agent": s["config"]["agent_concurrency"],
                    "prepare": s["config"]["training_workers"],
                    "match": s["config"]["match_workers"],
                }
                if s["phase"] == "researching":
                    active = sum(
                        j["kind"] == "agent"
                        and j["status"] in ("running", "queued", "admitted")
                        for j in s["jobs"].values()
                    )
                    ready = sorted(
                        (
                            b
                            for b in s["branches"].values()
                            if b["status"] == "ready"
                            and b["id"] in s["cycles"][-1]["targets"]
                        ),
                        key=lambda b: (b["last_run"], b["id"]),
                    )
                    for b in ready[: max(0, limits["agent"] - active)]:
                        self.research.start_iteration(s, b["id"])
                now = time.time()
                for kind, limit in limits.items():
                    used = sum(
                        j["kind"] == kind and j["status"] in ("running", "admitted")
                        for j in s["jobs"].values()
                    )
                    waiting = sorted(
                        (
                            j
                            for j in s["jobs"].values()
                            if j["kind"] == kind and j["status"] == "queued"
                        ),
                        key=lambda j: j["created_at"],
                    )
                    room = max(0, limit - used)
                    if kind == "agent" and self.agent_launch_interval > 0:
                        room = (
                            min(room, 1)
                            if now - self.agent_launch_at >= self.agent_launch_interval
                            else 0
                        )
                    for j in waiting[:room]:
                        j["status"] = "admitted"
                        admissions.append((s["id"], j["id"]))
        for run, jid in admissions:
            with self.research.edit(run) as (s, db):
                j = s["jobs"][jid]
                if s["status"] != "running":
                    j["status"] = "paused"
                    continue
                try:
                    self.launch(s, j)
                except Exception as exc:
                    j["status"] = "running"
                    atomic_json(
                        Path(s["root"]) / "jobs" / jid / "result.json",
                        {"status": "failed", "error": str(exc)[:2000]},
                    )
        for key, p in list(self.processes.items()):
            if p.poll() is None:
                continue
            # An exit can occur after the job poll above. Keep ownership until a
            # durable receipt exists so the next tick can reconcile that exit.
            root = Path(self.research.read(key[0])["root"]) / "jobs" / key[1]
            if (root / "result.json").exists():
                self.processes.pop(key)

    def close(self):
        for s in self.research.states(active_only=True):
            self.research.control(s["id"], "pause")
        end = time.monotonic() + 7
        while self.processes and time.monotonic() < end:
            self.tick()
            time.sleep(0.1)
        for s in self.research.states():
            for j in s["jobs"].values():
                if j["status"] == "running":
                    self.stop(s, j, force=True)
