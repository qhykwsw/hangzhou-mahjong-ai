"""Durable admission, process ownership and independent Agent/training/match slots."""

from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from mahjong_ai.persistence.artifacts import atomic_json, read_json, digest
from mahjong_ai.domain.engine import derive_seed
from mahjong_ai.execution.model_service import ModelService, SERVER_ENV
from mahjong_ai.execution.model_seat import ENCODER_ENV
from mahjong_ai.execution.remote import Dispatcher, RemoteHost, opponent_specs
import json

# 逐个放出 Agent，避免同一秒把整个并发上限压给模型网关触发限流。
AGENT_LAUNCH_INTERVAL = 5.0


class Controller:
    def __init__(self, research, recipes=None):
        self.research = research
        # The recipe iteration service designs and trains recipe branches'
        # candidates; research only hands trials over and evaluates them.
        self.recipes = recipes
        self.processes = {}
        self.agent_launch_interval = AGENT_LAUNCH_INTERVAL
        self.agent_launch_at = 0.0
        self.models = ModelService(research)
        self.dispatchers = {}
        self.hosts_checked = 0.0
        self.recover()

    def dispatcher(self, s, name):
        if name not in self.dispatchers:
            config = s["config"]["hosts"].get(name)
            if config is None:
                raise ValueError("未配置的远程主机：" + name)
            self.dispatchers[name] = Dispatcher(
                RemoteHost(name, config, self.research.project),
                self.research.layout.research / "models",
            )
        return self.dispatchers[name]

    def recover(self):
        for run in self.research.states():
            with self.research.edit(run["id"]) as (s, db):
                self.consume(s, db)
                # Also shrinks state that accumulated before compaction existed.
                self.research.compact_jobs(s)
                for j in s["jobs"].values():
                    if j["status"] == "running" and self.detached(j):
                        # Remote work and recipe trials outlive this service:
                        # take them over again instead of stopping them.
                        if j["kind"] == "trial" or self.dispatcher(s, j["host"]).adopt(
                            j["id"], Path(s["root"]) / "jobs" / j["id"]
                        ):
                            continue
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

    @staticmethod
    def detached(job):
        """Work that runs outside this machine and survives pauses and restarts."""
        return job["kind"] == "trial" or bool(job.get("host"))

    def local_live(self, s):
        return any(
            j["status"] in ("running", "admitted") and not self.detached(j)
            for j in s["jobs"].values()
        )

    def hosts(self, s, db):
        """Refresh host availability; work that must not wait leaves offline hosts."""
        known = s.setdefault("hosts", {})
        for name in s["config"].get("hosts", {}):
            online = not self.dispatcher(s, name).offline()
            if known.get(name, {}).get("online", True) == online and name in known:
                continue
            known[name] = {"online": online, "since": time.time()}
            self.research.event(db, s, "host_status", {"host": name, "online": online})
            if not online:
                for jid in self.research.rehome(s, db, name):
                    self.dispatcher(s, name).forget(jid)

    def stop(self, s, job, force=False):
        root = Path(s["root"]) / "jobs" / job["id"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "cancel.flag").touch()
        if job.get("host") and "stop_requested" not in job:
            self.dispatcher(s, job["host"]).cancel(
                job["id"], root, brawl=job.get("scope") == "competition"
            )
        if job["kind"] == "trial" and "stop_requested" not in job and self.recipes:
            self.recipes.cancel(job["line"])
            if not (root / "result.json").exists():
                atomic_json(root / "result.json", {"status": "paused", "reason": "operator interruption"})
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
                except (ProcessLookupError, PermissionError):
                    # A worker can outlive the service user/session boundary;
                    # the cancel flag is still durable, so keep the controller
                    # loop alive and let the process watchdog reap it later.
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
            try:
                self.research.accept(s, db, j, result)
            except ValueError as exc:
                if "不可变回执发生变化" not in str(exc):
                    raise
                # Keep the first immutable receipt and quarantine a changed
                # duplicate instead of crashing the whole service loop.
                j.update(status="failed", error="回执冲突：" + str(exc))
                self.research.event(
                    db,
                    s,
                    "receipt_conflict",
                    {"job": j["id"], "attempt": j.get("attempt", 0)},
                )

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
                candidate_limit=cfg["candidate_limit"],
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
                it.get("opponents", s["cycles"][it["cycle"] - 1]["reference"])
                if it
                else s["experiments"][-1]["initial_opponents"] or s["bootstrap"]
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

    def admissible(self, s, j, local_busy):
        if j["kind"] == "trial" and not j.get("baseline"):
            it = s["iterations"][j["iteration"]]
            return self.research.online(s, it["config"].get("evaluation_host"))
        # A brawl played here starts once this machine's iterations are done.
        return not (j.get("scope") == "competition" and not j.get("host") and local_busy)

    def overflow(self, s, local_limit):
        """Move local paired evaluations that cannot start now to idle hosts."""
        live = ("running", "admitted")
        local_used = sum(
            j["kind"] == "match" and not j.get("host") and j["status"] in live
            for j in s["jobs"].values()
        )
        backlog = sorted(
            (
                j for j in s["jobs"].values()
                if j["kind"] == "match" and not j.get("host")
                and j.get("scope") == "paired" and j["status"] == "queued"
                # Candidates with their own locked environment stay local.
                and not any(
                    s["candidates"].get(cid, {}).get("requirements")
                    for a in j["assignments"] for cid in a["players"]
                )
            ),
            key=lambda j: j["created_at"],
        )[max(0, local_limit - local_used):]
        for host in s["config"].get("overflow_hosts", []):
            if (
                not backlog
                or not self.research.online(s, host)
                or not self.dispatcher(s, host).idle_for_overflow()
            ):
                continue
            used = sum(
                j["kind"] == "match" and j.get("host") == host
                and j["status"] in live + ("queued",)
                for j in s["jobs"].values()
            )
            room = max(0, s["config"]["hosts"][host]["workers"] - used)
            for j in backlog[:room]:
                j.update(host=host, overflow=True)
            backlog = backlog[room:]

    def launch_trial(self, s, j, root):
        if self.recipes is None:
            raise RuntimeError("训练方案服务未启用，方案分支无法迭代")
        j.update(status="running", started_at=time.time())
        j.pop("stop_requested", None)
        if j.get("baseline"):
            self.recipes.request(j["line"], root, "baseline")
            return
        it = s["iterations"][j["iteration"]]
        host = it["config"].get("evaluation_host")
        override = s.get("candidate_runtime_path")
        candidates = {
            cid: dict(c, spec=dict(c["spec"], runtime_path=override))
            if override and c["spec"].get("package_path") else c
            for cid, c in ((cid, s["candidates"][cid]) for cid in it["opponents"])
        }
        timeouts, models = it["config"]["timeouts"], self.research.layout.research / "models"
        recipes, line, candidate, number = self.recipes, j["line"], j["candidate"], it["number"]

        def hand_over(remote):
            specs = opponent_specs(remote, candidates, models, timeouts)
            recipes.request(line, root, "trial", candidate=candidate, opponents=specs,
                            iteration=number)

        self.dispatcher(s, host).call(hand_over, root)

    def launch(self, s, j):
        root = Path(s["root"]) / "jobs" / j["id"]
        root.mkdir(parents=True, exist_ok=True)
        if j["kind"] == "trial":
            (root / "result.json").unlink(missing_ok=True)
            return self.launch_trial(s, j, root)
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
        # 迭代总预算随任务下发：提示词用它说明节奏，研究 MCP 用它回注剩余时间。
        limit = it["config"]["iteration_seconds"] if it and j["kind"] == "agent" else None
        task.update(
            deadline=time.time() + remaining,
            execution_limit_seconds=limit,
            owner_pid=os.getpid(),
        )
        if j["kind"] == "agent":
            task["thread_id"] = s["branches"][j["branch"]]["thread_id"]
            task["continuation"] = j.get("continuations", 0)
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
        if not j.get("host") and j["kind"] == "match" and any(
            p.get("family") == "model" for p in task.get("policies", {}).values()
        ):
            env[SERVER_ENV] = self.models.url()
            env[ENCODER_ENV] = str(self.models.project)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = str(task["resources"]["threads"])
        if j.get("host"):
            # Remote workers get the same environment; the dispatcher adds the
            # host's own inference service and runtime path.
            remote_env = {
                k: v for k, v in env.items()
                if k.startswith(("MAHJONG_", "OMP_", "OPENBLAS_", "VECLIB_"))
            }
            self.dispatcher(s, j["host"]).submit(
                j["id"], root, task, j["runtime_path"], remote_env
            )
            j.update(status="running", started_at=time.time())
            j.pop("stop_requested", None)
            return
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

    def idle_hosts(self):
        """Keep host status current while no experiment runs (the page shows it)."""
        if self.research.states(active_only=True) or time.monotonic() < self.hosts_checked + 30:
            return
        self.hosts_checked = time.monotonic()
        s = self.research.store.read()
        if not s or not s["config"].get("hosts"):
            return
        known = s.get("hosts", {})
        current = {name: not self.dispatcher(s, name).offline() for name in s["config"]["hosts"]}
        if any(known.get(n, {}).get("online") != v or n not in known for n, v in current.items()):
            with self.research.edit() as (s, db):
                self.hosts(s, db)

    def tick(self):
        self.idle_hosts()
        admissions = []
        concluded = []
        waking = False
        for run in self.research.states(active_only=True):
            with self.research.edit(run["id"]) as (s, db):
                self.consume(s, db)
                for it in s["iterations"].values():
                    if (
                        it.get("recipe_trial")
                        and it["status"] == "completed"
                        and not it.get("recipe_notified")
                    ):
                        it["recipe_notified"] = True
                        host = s["config"].get("brawl_host")
                        winner = s["candidates"].get(it.get("representative"))
                        if it.get("improved") and host and winner:
                            # A new RL version will enter the next brawl: move
                            # its weights to the brawl host now, not when the
                            # brawl starts (relayed uploads take minutes).
                            model = self.research.layout.research / "models" / winner["spec"]["parameters"]["weights"]
                            self.dispatcher(s, host).call(lambda remote, m=model: remote.sync([m]), None)
                        result = it["results"].get(it["candidates"][0]) or {}
                        concluded.append((it["recipe_trial"], bool(it.get("improved")),
                                          {k: result.get(k) for k in ("paired", "promotion",
                                           "appearances", "mean_score")}))
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
                self.hosts(s, db)
                # Pausing only waits for this machine's work; remote work and
                # trials continue and are collected after resuming.
                if s["status"] == "pausing" and not self.local_live(s):
                    s["status"] = "paused"
                if s["status"] != "running":
                    continue
                waking = waking or any(
                    j["kind"] == "trial" and j["status"] == "running" for j in s["jobs"].values()
                )
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
                        self.research.startable(s),
                        key=lambda b: (b["last_run"], b["id"]),
                    )
                    for b in ready[: max(0, limits["agent"] - active)]:
                        self.research.start_iteration(s, b["id"])
                now = time.time()
                self.overflow(s, limits["match"])
                # Local work shares this machine's limits; each remote host
                # has its own worker count.
                lanes = [(kind, None, limit) for kind, limit in limits.items()]
                lanes.append(("trial", None, 10**6))
                lanes += [
                    ("match", host, cfg["workers"])
                    for host, cfg in s["config"].get("hosts", {}).items()
                ]
                local_busy = self.research.local_busy(s)
                for kind, host, limit in lanes:
                    if host and not self.research.online(s, host):
                        continue
                    used = sum(
                        j["kind"] == kind
                        and j.get("host") == host
                        and j["status"] in ("running", "admitted")
                        for j in s["jobs"].values()
                    )
                    waiting = sorted(
                        (
                            j
                            for j in s["jobs"].values()
                            if j["kind"] == kind
                            and j.get("host") == host
                            and j["status"] == "queued"
                            and self.admissible(s, j, local_busy)
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
        for trial, promoted, evaluation in concluded:
            if self.recipes is not None:
                self.recipes.conclude(trial, promoted, evaluation)
        if self.recipes is not None and waking:
            # Trials handed over before a restart continue once research runs.
            self.recipes.wake()
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
                if j["status"] == "running" and not self.detached(j):
                    self.stop(s, j, force=True)
        self.models.close()
