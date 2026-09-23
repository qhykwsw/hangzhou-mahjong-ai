"""Recipe iteration: one serial trial loop per recipe line.

A trial starts from a copy of the line's incumbent recipe. The Agent edits it and
submits it (designing); the frozen harness trains it for the fixed budget from
the line's fixed start (training); the trained Actor and the incumbent's play
the same games against the frozen rules (evaluating). A paired 95% lower bound
above zero promotes it. The line's start and every promotion get a five-way
brawl against the four rules, which draws the promotion curve.

State transitions are owned here. The Agent reaches them only through
recipes.mcp; remote work runs in guided.recipe_jobs child processes.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from mahjong_ai.persistence.artifacts import atomic_json, read_json
from mahjong_ai.recipes.store import RecipeStore, configuration

ACTIVE = ("queued", "admitted", "running")
LIVE = ACTIVE + ("paused",)
# Evaluation belongs to the research framework (ADR-0013): a trained trial is
# handed off and waits there until research concludes it.
STAGE_JOB = {"designing": "agent", "training": "train"}
OUTCOMES = {
    "promoted": "晋级",
    "rejected": "未晋级",
    "crashed": "训练失败",
    "unsubmitted": "未提交",
}
AGENT_FAILURE_PAUSE = 2  # consecutive Agent session errors before pausing


def seed_for(*parts):
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def recipe_files(root):
    root = Path(root)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )


def tree_hash(root):
    h = hashlib.sha256()
    for name in recipe_files(root):
        h.update(name.encode() + b"\0" + (Path(root) / name).read_bytes() + b"\0")
    return h.hexdigest()


def copy_recipe(source, target):
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )


def budget(s, line):
    """Training seconds of one line; lines may override the experiment default."""
    return line.get("budget_seconds") or s["config"]["trial"]["budget_seconds"]


def smoke_extension(trial, limit):
    """Smoke checks run on the remote host; their time does not count as thinking."""
    return sum(x.get("seconds", 0.0) for x in trial.get("smokes", [])[:limit])


def agent_deadline(root):
    task = read_json(Path(root) / "task.json")
    path = Path(root) / "extension.json"
    extra = 0.0
    if path.exists():
        try:
            extra = float(read_json(path)["seconds"]) - task.get("extension_base", 0.0)
        except (ValueError, KeyError):
            extra = 0.0
    return task["deadline"] + max(0.0, extra)


def memory_summary(status):
    """Peak usage against the harness caps, in GB, for the Agent and the page."""
    gb = 2**30
    limits = status.get("limits") or {}
    values = {
        "reason": status.get("reason"),
        "gpu_peak_gb": status.get("gpu_peak_reserved_bytes"),
        "gpu_limit_gb": limits.get("gpu_limit_bytes"),
        "ram_peak_gb": status.get("ram_peak_bytes"),
        "ram_limit_gb": limits.get("ram_limit_bytes"),
    }
    return {
        k: round(v / gb, 2) if isinstance(v, (int, float)) and k != "reason" else v
        for k, v in values.items()
        if v is not None
    }


def compact_metrics(row):
    return {
        k: round(v, 5) if isinstance(v, float) else v
        for k, v in (row or {}).items()
        if isinstance(v, (int, float, bool)) and k not in ("time",)
    }


class RecipeIteration:
    def __init__(self, workspace):
        self.store = RecipeStore(workspace)
        self.processes = {}

    @property
    def project(self):
        return self.store.project

    def experiment_root(self, s):
        return self.store.root / s["id"]

    def job_root(self, s, jid):
        return self.experiment_root(s) / "jobs" / jid

    def trial_root(self, s, trial):
        return (
            self.experiment_root(s)
            / "lines"
            / trial["line"]
            / "trials"
            / f"{trial['number']:03d}"
        )

    # ---- experiment lifecycle -------------------------------------------------

    def create(self, limit=None):
        current = self.store.read()
        if current and current["phase"] != "closed":
            raise ValueError("先结束当前的训练方案迭代实验")
        from mahjong_ai.research.config import MODELS

        config = configuration(self.project)
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError("尝试次数上限必须是正整数")
            config["trial"]["limit"] = limit
        if config["agent"]["model"] not in MODELS:
            raise ValueError("未知的研究模型：" + config["agent"]["model"])
        identity = time.strftime("%Y%m%d-%H%M%S")
        root = self.store.root / identity
        now = time.time()
        lines = {}
        for spec in config["lines"]:
            recipe = root / "lines" / spec["id"] / "versions" / "000" / "recipe"
            copy_recipe(self.project / config["baseline"], recipe)
            lines[spec["id"]] = {
                **spec,
                "limit": config["trial"]["limit"],
                "used": 0,
                "version": 0,
                "versions": [
                    {
                        "version": 0,
                        "trial": None,
                        "recipe": str(recipe),
                        "actor": None,
                        "at": now,
                    }
                ],
                "curve": [],
                "pending_brawl": None,
                "request": None,
                "conclusions": [],
                "failures": 0,
                "trial": None,
                "model": None,
            }
        s = {
            "id": identity,
            "created_at": now,
            "status": "paused",
            "phase": "active",
            "pause_reason": "尚未开始",
            "stop_after_trial": False,
            "config": config,
            "model_config": MODELS[config["agent"]["model"]],
            "prepared": None,
            "lines": lines,
            "trials": {},
            "jobs": {},
            "sequence": 0,
        }
        self.store.create(s)
        return identity

    def control(self, action, line=None, count=0):
        with self.store.edit() as (s, db):
            if s["phase"] == "closed" and action != "close":
                raise ValueError("实验已结束")
            if action == "pause":
                if s["status"] == "running":
                    s.update(status="pausing", pause_reason="用户暂停")
            elif action == "resume":
                if s["status"] == "pausing":
                    raise ValueError("请等待暂停完成")
                if s["phase"] == "finished":
                    if all(v["used"] >= v["limit"] for v in s["lines"].values()):
                        raise ValueError("继续之前先追加尝试次数")
                    s["phase"] = "active"
                s.update(status="running", pause_reason="", stop_after_trial=False)
                for j in s["jobs"].values():
                    if j["status"] == "paused":
                        j["status"] = "queued"
                    elif j["status"] == "failed":
                        # Resuming grants the infrastructure a fresh set of retries.
                        j["status"] = "abandoned"
                for v in s["lines"].values():
                    v["failures"] = 0
            elif action == "pause_after":
                if s["status"] != "running":
                    raise ValueError("只有运行中的实验可以安排停止")
                s["stop_after_trial"] = True
            elif action == "clear_stop":
                s["stop_after_trial"] = False
            elif action == "stop_now":
                discarded = self.discard(s, db)
                if s["status"] == "running":
                    s["status"] = "pausing"
                s.update(pause_reason="已立即结束当前尝试", stop_after_trial=False)
                self.store.event(db, s, "discarded", {"trials": discarded})
            elif action == "extend":
                if type(count) is not int or count < 1:
                    raise ValueError("追加的尝试次数必须为正整数")
                targets = [line] if line else list(s["lines"])
                for key in targets:
                    s["lines"][key]["limit"] += count
                if s["phase"] == "finished":
                    s["phase"] = "active"
            elif action == "close":
                if s["status"] in ("running", "pausing"):
                    raise ValueError("结束实验前先暂停并等待完成")
                self.discard(s, db)
                for j in s["jobs"].values():
                    if j["status"] in LIVE:
                        j["status"] = "cancelled"
                s.update(phase="closed", status="paused", pause_reason="实验已结束")
            else:
                raise ValueError("未知操作")
            self.store.event(db, s, action, {"line": line, "count": count})
        return self.snapshot()

    def discard(self, s, db):
        """Drop in-flight trials without counting them; remote work is killed."""
        discarded = []
        for line in s["lines"].values():
            trial = s["trials"].get(line["trial"])
            if not trial:
                continue
            trial.update(status="discarded", finished_at=time.time())
            line["trial"] = None
            discarded.append(trial["id"])
            for j in s["jobs"].values():
                if j.get("trial") == trial["id"] and j["status"] in LIVE:
                    j["status"] = "cancelled"
                    self.stop(s, j)
                    self.kill_remote(s, j)
        return discarded

    # ---- state machine ----------------------------------------------------------

    def queue(self, s, kind, line=None, trial=None, version=None):
        s["sequence"] += 1
        jid = f"{s['sequence']:05d}-{kind}" + (f"-{line}" if line else "")
        attempt = sum(
            j["kind"] == kind
            and j.get("line") == line
            and j.get("trial") == trial
            and j.get("version") == version
            for j in s["jobs"].values()
        )
        job = {
            "id": jid,
            "kind": kind,
            "line": line,
            "trial": trial,
            "version": version,
            "attempt": attempt,
            "status": "queued",
            "created_at": time.time(),
            "spent": 0.0,
            "continuations": 0,
        }
        s["jobs"][jid] = job
        if kind != "agent":
            atomic_json(self.job_root(s, jid) / "task.json", self.job_task(s, job))
        return job

    def ensure(self, s, db, kind, line=None, trial=None, version=None):
        """Keep exactly one live job for a stage; bounded retries, then pause."""
        same = [
            j
            for j in s["jobs"].values()
            if j["kind"] == kind
            and j.get("line") == line
            and j.get("trial") == trial
            and j.get("version") == version
        ]
        if any(j["status"] in LIVE for j in same):
            return
        failed = [j for j in same if j["status"] == "failed"]
        if len(failed) > s["config"]["trial"]["retries"]:
            error = failed[-1].get("error") or "未知错误"
            if kind == "brawl":
                # A missing curve point must not block the line.
                target = s["lines"][line]
                target["curve"].append({"version": version, "error": error[:500]})
                target["pending_brawl"] = None
                self.store.event(
                    db, s, "brawl_skipped", {"line": line, "error": error[:500]}
                )
                return
            s.update(
                status="pausing",
                pause_reason=f"{kind} 连续失败 {len(failed)} 次：{error[:300]}",
            )
            return
        self.queue(s, kind, line, trial, version)

    # ---- research framework hand-off -------------------------------------------

    def request(self, line_id, root, kind, candidate=None, opponents=None, iteration=None):
        """Queue work asked for by a research iteration; the reply is its receipt."""
        with self.store.edit() as (s, db):
            line = s["lines"][line_id]
            line["request"] = dict(
                kind=kind, root=str(root), candidate=candidate,
                opponents=opponents or {}, iteration=iteration, at=time.time(),
            )
            if s["status"] == "paused" and s["phase"] == "active":
                s.update(status="running", pause_reason="")
            self.store.event(db, s, "requested", {"line": line_id, "kind": kind})

    @staticmethod
    def reply(request, payload):
        atomic_json(Path(request["root"]) / "result.json", payload)

    @staticmethod
    def baseline_reply(line):
        v0 = line["versions"][0]
        return {"status": "completed", "baseline": True, "local_actor": v0.get("local_actor"),
                "recipe": v0["recipe"], "training": v0.get("training")}

    def conclude(self, trial_id, promoted, evaluation=None):
        """Research decided a handed-off trial; promotion makes it the new base."""
        with self.store.edit() as (s, db):
            trial = s["trials"].get(trial_id)
            if not trial or trial["status"] != "handed_off":
                return
            line = s["lines"][trial["line"]]
            trial["evaluation"] = evaluation
            if promoted:
                line["version"] += 1
                line["versions"].append({
                    "version": line["version"], "trial": trial["id"],
                    "recipe": trial["submitted"], "actor": trial["actor"],
                    "local_actor": trial.get("local_actor"), "at": time.time(),
                })
            self.complete_trial(s, db, trial, "promoted" if promoted else "rejected")

    def cancel(self, line_id):
        with self.store.edit() as (s, db):
            s["lines"][line_id]["request"] = None
            trial = s["trials"].get(s["lines"][line_id]["trial"])
            if trial:
                trial.update(status="discarded", finished_at=time.time())
                s["lines"][line_id]["trial"] = None
                for j in s["jobs"].values():
                    if j.get("trial") == trial["id"] and j["status"] in LIVE:
                        j["status"] = "cancelled"
                        self.stop(s, j)
                        self.kill_remote(s, j)

    def start_trial(self, s, db, line, request=None):
        number = 1 + sum(t["line"] == line["id"] for t in s["trials"].values())
        tid = f"{line['id']}-{number:03d}"
        incumbent = line["versions"][-1]
        trial = {
            "id": tid,
            "line": line["id"],
            "number": number,
            "status": "designing",
            "outcome": None,
            "parent": {"source": "incumbent", "version": incumbent["version"]},
            "incumbent_version": incumbent["version"],
            "plan": None,
            "summary": None,
            "smokes": [],
            "reviewed": False,
            "thread_id": None,
            "created_at": time.time(),
            "request": request,
        }
        workspace = self.trial_root(s, trial) / "workspace" / "recipe"
        copy_recipe(incumbent["recipe"], workspace)
        trial["workspace"] = str(workspace)
        s["trials"][tid] = trial
        line["trial"] = tid
        self.queue(s, "agent", line["id"], tid)
        self.store.event(db, s, "trial_started", {"trial": tid})

    def complete_trial(self, s, db, trial, outcome, error=None):
        line = s["lines"][trial["line"]]
        request = trial.get("request")
        if request and outcome not in ("promoted", "rejected"):
            # Nothing to evaluate: the research iteration ends without candidates.
            self.reply(request, {
                "status": "completed",
                "trial": trial["id"],
                "plan": trial.get("plan"),
                "summary": trial.get("summary"),
                "candidates": {request["candidate"]: {
                    "status": "failed", "error": error or OUTCOMES.get(outcome, outcome)}},
            })
        trial.update(
            status="completed", outcome=outcome, finished_at=time.time(), error=error
        )
        line["used"] += 1
        line["trial"] = None
        self.store.event(
            db, s, "trial_completed", {"trial": trial["id"], "outcome": outcome}
        )

    def advance(self, s, db):
        if s["prepared"] is None:
            self.ensure(s, db, "prepare")
            return
        idle = True
        for line in s["lines"].values():
            if any(
                j.get("line") == line["id"] and j["status"] in LIVE
                for j in s["jobs"].values()
            ):
                idle = False
                continue
            trial = s["trials"].get(line["trial"])
            if line["versions"][0]["actor"] is None:
                idle = False
                self.ensure(s, db, "baseline", line["id"])
            elif trial:
                idle = False
                if trial["status"] == "handed_off":
                    continue
                if trial["status"] == "submitted":
                    trial["status"] = "training"
                if trial["status"] == "designing":
                    # The Agent job ended in a state accept() did not resolve.
                    self.complete_trial(
                        s, db, trial, "unsubmitted", "Agent 会话结束但未提交"
                    )
                else:
                    self.ensure(
                        s, db, STAGE_JOB[trial["status"]], line["id"], trial["id"]
                    )
            elif line.get("request"):
                # Trials start only when the research framework asks for one.
                idle = False
                request = line["request"]
                line["request"] = None
                if request["kind"] == "baseline":
                    self.reply(request, self.baseline_reply(line))
                else:
                    self.start_trial(s, db, line, request)
            if s["status"] != "running":
                return
        if not idle:
            return
        if all(v["used"] >= v["limit"] for v in s["lines"].values()):
            s.update(
                status="paused",
                phase="finished",
                pause_reason="所有方案线已用完尝试次数",
            )
        elif s["stop_after_trial"]:
            s.update(
                status="paused",
                stop_after_trial=False,
                pause_reason="已按安排在当前尝试后停止",
            )

    def accept(self, s, db, j, result):
        if j["status"] == "cancelled":
            return
        if j["kind"] == "agent":
            return self.accept_agent(s, db, j, result)
        if result.get("status") == "paused":
            j["status"] = "paused"
            return
        getattr(self, "accept_" + j["kind"])(s, db, j, result)

    def failed(self, j, result):
        j.update(status="failed", error=(result.get("error") or "")[:2000])

    def accept_prepare(self, s, db, j, result):
        if result["status"] != "completed":
            return self.failed(j, result)
        j["status"] = "completed"
        s["prepared"] = result["output"]
        baseline = s["config"]["trial"].get("incumbent", "start") == "baseline"
        for key, line in s["lines"].items():
            prepared = result["output"]["lines"][key]
            # Version 0 is either the fixed start itself or what the baseline
            # recipe trains from it in one budget; trials always start from it.
            line["versions"][0]["actor"] = None if baseline else prepared["start_actor"]
            line["model"] = prepared["start_meta"]["config"]
        self.store.event(db, s, "prepared", {"bundle": result["output"]["bundle_id"]})

    def accept_agent(self, s, db, j, result):
        trial = s["trials"][j["trial"]]
        line = s["lines"][j["line"]]
        cfg = s["config"]["agent"]
        if result.get("thread_id"):
            trial["thread_id"] = result["thread_id"]
        j["spent"] += result.get("usage", {}).get("wall_seconds", 0.0)
        trial["agent_seconds"] = round(j["spent"], 1)
        if trial["status"] == "submitted":
            j["status"] = "completed"
            trial["status"] = "training"
            line["failures"] = 0
            return
        if trial["status"] != "designing":
            j["status"] = "completed"
            return
        if result["status"] == "paused":
            j["status"] = "paused"
            return
        remaining = (
            cfg["seconds"] + smoke_extension(trial, cfg["smoke_limit"]) - j["spent"]
        )
        if (
            result["status"] == "completed"
            and remaining > 45
            and j["continuations"] < cfg["continuations"]
        ):
            j["continuations"] += 1
            j["status"] = "queued"
            return
        j["status"] = "completed"
        error = result.get("error")
        if result["status"] == "failed":
            line["failures"] += 1
            if line["failures"] >= AGENT_FAILURE_PAUSE:
                s.update(
                    status="pausing",
                    pause_reason="研究模型会话连续失败：" + str(error)[:300],
                )
        self.complete_trial(
            s, db, trial, "unsubmitted", error or "Agent 时限内没有提交训练方案"
        )

    @staticmethod
    def training_summary(j, out):
        status = out["status"]
        metrics = out.get("metrics") or []
        return {
            "job": j["id"],
            "state": status.get("state"),
            "seconds": status.get("training_seconds"),
            "overrun": status.get("overrun_seconds"),
            "updates": len(metrics),
            "last": compact_metrics(metrics[-1] if metrics else None),
            "error": (status.get("error") or "")[-3000:] or None,
            **memory_summary(status),
        }

    def accept_baseline(self, s, db, j, result):
        if result["status"] != "completed":
            return self.failed(j, result)
        j["status"] = "completed"
        line = s["lines"][j["line"]]
        out = result["output"]
        line["versions"][0]["training"] = self.training_summary(j, out)
        if out["ok"]:
            line["versions"][0]["actor"] = out["actor"]
            line["versions"][0]["local_actor"] = out.get("local_actor")
            self.store.event(db, s, "baseline_trained", {"line": line["id"]})
        else:
            s.update(
                status="pausing",
                pause_reason=f"{line['name']} 基线方案训练失败："
                + str(out["status"].get("error") or out["status"].get("state"))[-300:],
            )

    def accept_train(self, s, db, j, result):
        if result["status"] != "completed":
            return self.failed(j, result)
        j["status"] = "completed"
        trial = s["trials"][j["trial"]]
        out = result["output"]
        status = out["status"]
        trial["training"] = self.training_summary(j, out)
        if out["ok"]:
            trial.update(actor=out["actor"], local_actor=out.get("local_actor"),
                         status="handed_off")
            request = trial.get("request")
            if request:
                self.reply(request, {
                    "status": "completed",
                    "trial": trial["id"],
                    "plan": trial.get("plan"),
                    "summary": trial.get("summary"),
                    "candidates": {request["candidate"]: {
                        "status": "ready", "local_actor": out.get("local_actor"),
                        "training": trial["training"]}},
                })
        else:
            reason = {
                "timeout": "训练超时",
                "gpu_memory": "显存超过上限",
                "ram_limit": "内存超过上限",
            }.get(status.get("reason") or status.get("state"), "训练报错")
            self.complete_trial(s, db, trial, "crashed", reason)

    def accept_evaluate(self, s, db, j, result):
        out = result.get("output") or {}
        if result["status"] != "completed" or not out.get("ok"):
            error = result.get("error") or str(out.get("status", {}).get("error"))
            return self.failed(j, {"error": error})
        j["status"] = "completed"
        trial = s["trials"][j["trial"]]
        line = s["lines"][j["line"]]
        r = out["result"]
        trial["evaluation"] = {
            k: r[k]
            for k in (
                "games",
                "challenger",
                "incumbent",
                "difference",
                "promoted",
                "seconds",
            )
        }
        line["failures"] = 0
        if r["promoted"]:
            line["version"] += 1
            line["versions"].append(
                {
                    "version": line["version"],
                    "trial": trial["id"],
                    "recipe": trial["submitted"],
                    "actor": trial["actor"],
                    "difference": r["difference"],
                    "at": time.time(),
                }
            )
            line["pending_brawl"] = line["version"]
            self.complete_trial(s, db, trial, "promoted")
        else:
            self.complete_trial(s, db, trial, "rejected")

    def accept_brawl(self, s, db, j, result):
        out = result.get("output") or {}
        if result["status"] != "completed" or not out.get("ok"):
            error = result.get("error") or str(out.get("status", {}).get("error"))
            return self.failed(j, {"error": error})
        j["status"] = "completed"
        line = s["lines"][j["line"]]
        r = out["result"]
        line["curve"].append(
            {
                "version": j["version"],
                "trial": line["versions"][j["version"]]["trial"],
                "results": {
                    name: {k: v[k] for k in ("mean", "lower", "upper", "appearances")}
                    for name, v in r["results"].items()
                },
                "ranking": r["ranking"],
                "games": r["games"],
                "seconds": r["seconds"],
                "at": time.time(),
            }
        )
        line["pending_brawl"] = None
        self.store.event(db, s, "brawl", {"line": line["id"], "version": j["version"]})

    # ---- tasks ------------------------------------------------------------------

    def job_task(self, s, j):
        cfg = s["config"]["trial"]
        if j["kind"] == "prepare":
            return {
                "kind": "prepare",
                "root": str(self.experiment_root(s)),
                "remote_root": f".local/guided-recipes/{s['id']}",
                "lines": {
                    key: {"host": v["host"], "start_checkpoint": v["start_checkpoint"]}
                    for key, v in s["lines"].items()
                },
                "research_project": str(
                    self.store.workspace / "projects/agent-research"
                ),
                "teacher_store": str(self.project / ".local/teachers"),
            }
        p = s["prepared"]
        line = s["lines"][j["line"]]
        task = {
            "kind": j["kind"],
            "host": line["host"],
            "line": line["id"],
            "remote_dir": f"{p['remote_root']}/{line['id']}/{j['id']}",
            "code": p["code"],
            "source_id": p["source_id"],
            "bundle": p["bundle"],
            "execution": line["execution"],
        }
        trial = s["trials"].get(j.get("trial"))
        if j["kind"] == "baseline":
            task.update(
                kind="train",
                limits=line.get("limits") or {},
                recipe=line["versions"][0]["recipe"],
                start_actor=p["lines"][line["id"]]["start_actor"],
                budget_seconds=budget(s, line),
                smoke_seconds=cfg["smoke_seconds"],
                seed=seed_for(s["id"], line["id"], "baseline"),
                fetch_actor=True,
            )
        elif j["kind"] == "train":
            task.update(
                limits=line.get("limits") or {},
                recipe=trial["submitted"],
                start_actor=p["lines"][line["id"]]["start_actor"],
                budget_seconds=budget(s, line),
                smoke_seconds=cfg["smoke_seconds"],
                seed=seed_for(s["id"], line["id"], trial["number"], "train"),
                opponents=(trial.get("request") or {}).get("opponents") or {},
                fetch_actor=True,
            )
        return task

    def smoke_task(self, s, trial, index):
        p = s["prepared"]
        line = s["lines"][trial["line"]]
        jid = f"smoke-{trial['id']}-{index}"
        return jid, {
            "kind": "smoke",
            "host": line["host"],
            "line": line["id"],
            "remote_dir": f"{p['remote_root']}/{line['id']}/{jid}",
            "code": p["code"],
            "source_id": p["source_id"],
            "bundle": p["bundle"],
            "execution": line["execution"],
            "limits": line.get("limits") or {},
            "recipe": trial["workspace"],
            "opponents": (trial.get("request") or {}).get("opponents") or {},
            "start_actor": p["lines"][line["id"]]["start_actor"],
            "budget_seconds": budget(s, line),
            "smoke_seconds": s["config"]["trial"]["smoke_seconds"],
            "seed": seed_for(s["id"], line["id"], trial["number"], "smoke", index),
        }

    def agent_task(self, s, j):
        from mahjong_ai.recipes.contract import prompt

        trial = s["trials"][j["trial"]]
        cfg = s["config"]["agent"]
        extension = smoke_extension(trial, cfg["smoke_limit"])
        remaining = max(30.0, cfg["seconds"] + extension - j["spent"])
        root = self.job_root(s, j["id"])
        return {
            "project": str(self.project),
            "experiment": s["id"],
            "branch": trial["line"],
            "thread_id": trial.get("thread_id"),
            "model_config": s["model_config"],
            "deadline": time.time() + remaining,
            "execution_limit_seconds": cfg["seconds"],
            "extension_base": extension,
            "owner_pid": os.getpid(),
            "prompt": prompt(s, trial, j, remaining),
            "mcp_server": {
                "name": "mahjong_recipes",
                "module": "mahjong_ai.recipes.mcp",
                "args": [str(self.store.workspace), s["id"], trial["id"], str(root)],
            },
            "tool_timeout_seconds": s["config"]["trial"]["smoke_seconds"] + 420,
        }

    # ---- processes --------------------------------------------------------------

    def environment(self, kind):
        src = str(self.store.workspace / "src")
        path = src if kind == "agent" else str(self.project) + os.pathsep + src
        return dict(os.environ, PYTHONPATH=path, PYTHONUNBUFFERED="1")

    def launch(self, s, j):
        root = self.job_root(s, j["id"])
        root.mkdir(parents=True, exist_ok=True)
        if (root / "result.json").exists():
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            (root / "result.json").rename(attempts / f"{time.time_ns()}.json")
        for flag in ("cancel.flag", "submitted.flag"):
            (root / flag).unlink(missing_ok=True)
        if j["kind"] == "agent":
            atomic_json(root / "task.json", self.agent_task(s, j))
        module = (
            "mahjong_ai.execution.agent"
            if j["kind"] == "agent"
            else "guided.recipe_jobs"
        )
        with (root / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", "-m", module, str(root)],
                cwd=self.project,
                env=self.environment(j["kind"]),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        j.update(status="running", pid=process.pid, started_at=time.time())
        self.processes[j["id"]] = process
        atomic_json(root / "process.json", {"pid": process.pid})

    def stop(self, s, j, force=False):
        root = self.job_root(s, j["id"])
        if not root.exists():
            return
        (root / "cancel.flag").touch()
        for name in ("codex-process.json", "process.json"):
            path = root / name
            if not path.exists():
                continue
            pid = read_json(path)["pid"]
            command = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
            ).stdout
            if str(root) in command or (
                name == "codex-process.json" and "codex" in command
            ):
                try:
                    os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

    def kill_remote(self, s, j):
        """Stop a discarded trial's remote training or evaluation; best effort."""
        root = self.job_root(s, j["id"])
        if j["kind"] in ("train", "evaluate", "brawl") and any(
            root.glob("launch-*.json")
        ):
            subprocess.Popen(
                [sys.executable, "-B", "-m", "guided.recipe_jobs", "--kill", str(root)],
                cwd=self.project,
                env=self.environment(j["kind"]),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    def consume(self, s, db):
        for j in s["jobs"].values():
            if j["status"] not in ("running", "admitted", "cancelled"):
                continue
            receipt = self.job_root(s, j["id"]) / "result.json"
            if not receipt.exists() or j.get("consumed") == receipt.stat().st_mtime_ns:
                continue
            j["consumed"] = receipt.stat().st_mtime_ns
            self.accept(s, db, j, read_json(receipt))

    def recover(self):
        experiment = self.store.current()
        if not experiment:
            return
        with self.store.edit(experiment) as (s, db):
            self.consume(s, db)
            for j in s["jobs"].values():
                if j["status"] not in ("running", "admitted"):
                    continue
                self.stop(s, j, force=True)
                root = self.job_root(s, j["id"])
                if j["kind"] == "agent":
                    log = root / "codex.jsonl"
                    for line in (
                        log.read_text(errors="replace").splitlines()
                        if log.exists()
                        else []
                    ):
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get("type") == "thread.started":
                            s["trials"][j["trial"]]["thread_id"] = event.get(
                                "thread_id"
                            )
                    if (root / "task.json").exists():
                        ended = min(time.time(), agent_deadline(root))
                        j["spent"] += max(0.0, ended - j.get("started_at", ended))
                    if s["trials"][j["trial"]]["status"] == "submitted":
                        self.accept_agent(s, db, j, {"status": "completed"})
                        continue
                j["status"] = "paused"
            if s["status"] in ("running", "pausing"):
                s.update(status="paused", pause_reason="服务已重启，继续实验以恢复")

    def tick(self):
        experiment = self.store.current()
        if not experiment:
            return
        admissions = []
        with self.store.edit(experiment) as (s, db):
            if s["phase"] == "closed":
                return
            self.consume(s, db)
            now = time.time()
            for j in s["jobs"].values():
                process = self.processes.get(j["id"])
                root = self.job_root(s, j["id"])
                alive = process is not None and process.poll() is None
                if j["status"] == "running" and not alive and process is not None:
                    if not (root / "result.json").exists():
                        atomic_json(
                            root / "result.json",
                            {
                                "status": "paused"
                                if s["status"] != "running"
                                else "failed",
                                "error": "任务进程退出但没有留下回执",
                                "usage": {
                                    "wall_seconds": max(0.0, now - j["started_at"])
                                },
                            },
                        )
                if j["status"] == "running" and j["kind"] == "agent" and alive:
                    trial = s["trials"][j["trial"]]
                    if (
                        trial["status"] == "submitted"
                        and now - trial.get("submitted_at", now) > 3
                    ):
                        (root / "submitted.flag").touch()
                    if s["status"] == "running" and now > agent_deadline(root) + 20:
                        self.stop(s, j, force=True)
                if alive and (
                    j["status"] not in ("running", "admitted")
                    or s["status"] != "running"
                ):
                    self.stop(s, j, force=now - j.setdefault("stop_requested", now) > 5)
            self.consume(s, db)
            if s["status"] == "pausing" and not any(
                j["status"] in ("running", "admitted") for j in s["jobs"].values()
            ):
                s["status"] = "paused"
            if s["status"] == "running":
                self.advance(s, db)
            if s["status"] == "running":
                for j in sorted(s["jobs"].values(), key=lambda j: j["created_at"]):
                    if j["status"] == "queued":
                        j["status"] = "admitted"
                        j.pop("stop_requested", None)
                        admissions.append(j["id"])
        for jid in admissions:
            with self.store.edit(experiment) as (s, db):
                j = s["jobs"][jid]
                if s["status"] != "running" or j["status"] != "admitted":
                    if j["status"] == "admitted":
                        j["status"] = "paused"
                    continue
                try:
                    self.launch(s, j)
                except Exception as exc:
                    j["status"] = "running"
                    atomic_json(
                        self.job_root(s, jid) / "result.json",
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"[:2000],
                        },
                    )
        s = self.store.read(experiment)
        for jid, process in list(self.processes.items()):
            # Keep ownership of an exited process until its receipt exists, so
            # the next tick can record the exit.
            if process.poll() is not None and (
                (self.job_root(s, jid) / "result.json").exists()
                or s["jobs"][jid]["status"] not in ("running", "admitted")
            ):
                self.processes.pop(jid)

    def close(self):
        experiment = self.store.current()
        if not experiment:
            return
        with self.store.edit(experiment) as (s, db):
            if s["status"] == "running":
                s.update(status="pausing", pause_reason="服务停止")
        end = time.monotonic() + 7
        while self.processes and time.monotonic() < end:
            self.tick()
            time.sleep(0.1)
        with self.store.edit(experiment) as (s, db):
            for j in s["jobs"].values():
                if j["status"] == "running":
                    self.stop(s, j, force=True)

    # ---- views ------------------------------------------------------------------

    def line_stage(self, s, line):
        live = [
            j
            for j in s["jobs"].values()
            if j.get("line") == line["id"] and j["status"] in LIVE
        ]
        if s["prepared"] is None:
            return {"label": "部署框架与冻结对手", "kind": "prepare"}
        if not live:
            if line["used"] >= line["limit"]:
                return {"label": "尝试次数已用完", "kind": "done"}
            return {"label": "等待", "kind": "idle"}
        j = live[0]
        stage = {
            "kind": j["kind"],
            "label": {
                "agent": "Agent 设计方案",
                "baseline": "训练第 0 版（基线方案）",
                "train": "限时训练",
                "evaluate": "配对评估",
                "brawl": f"大乱斗（第 {j.get('version')} 版）",
            }[j["kind"]],
            "status": j["status"],
            "started_at": j.get("started_at"),
        }
        root = self.job_root(s, j["id"])
        if (
            j["kind"] == "agent"
            and j["status"] == "running"
            and (root / "task.json").exists()
        ):
            stage["deadline"] = agent_deadline(root)
        if j["kind"] in ("train", "baseline") and j.get("started_at"):
            stage["deadline"] = j["started_at"] + budget(s, line)
        progress = root / "progress.json"
        if progress.exists():
            stage["progress"] = read_json(progress).get("message")
        return stage

    def trial_view(self, trial):
        evaluation = trial.get("evaluation") or {}
        training = trial.get("training") or {}
        plan = trial.get("plan") or {}
        return {
            "id": trial["id"],
            "line": trial["line"],
            "number": trial["number"],
            "status": trial["status"],
            "outcome": trial.get("outcome"),
            "outcome_label": OUTCOMES.get(trial.get("outcome"), ""),
            "direction": plan.get("direction"),
            "summary": trial.get("summary"),
            "difference": evaluation.get("difference"),
            "training": {
                k: training.get(k)
                for k in ("state", "seconds", "updates", "gpu_peak_gb", "gpu_limit_gb")
            },
            "incumbent_version": trial.get("incumbent_version"),
            "error": trial.get("error"),
            "created_at": trial.get("created_at"),
            "finished_at": trial.get("finished_at"),
        }

    def snapshot(self):
        s = self.store.read()
        if s is None:
            return {"experiment": None, "experiments": []}
        lines = []
        for line in s["lines"].values():
            trial = s["trials"].get(line["trial"])
            incumbent = line["versions"][-1]
            lines.append(
                {
                    "id": line["id"],
                    "name": line["name"],
                    "host": line["host"],
                    "model": line["model"],
                    "budget_seconds": budget(s, line),
                    "limit": line["limit"],
                    "used": line["used"],
                    "version": line["version"],
                    "incumbent": {
                        "version": incumbent["version"],
                        "trial": incumbent["trial"],
                        "difference": incumbent.get("difference"),
                    },
                    "stage": self.line_stage(s, line),
                    "trial": self.trial_view(trial) | {"plan": trial.get("plan")}
                    if trial
                    else None,
                    "curve": line["curve"],
                    "conclusions": line["conclusions"],
                    "promotions": line["version"],
                }
            )
        trials = sorted(
            (
                self.trial_view(t)
                for t in s["trials"].values()
                if t["status"] != "discarded"
            ),
            key=lambda t: t["created_at"],
            reverse=True,
        )
        prepared = s["prepared"] or {}
        return {
            "experiment": {
                "id": s["id"],
                "status": s["status"],
                "phase": s["phase"],
                "pause_reason": s["pause_reason"],
                "stop_after_trial": s["stop_after_trial"],
                "created_at": s["created_at"],
                "incumbent": s["config"]["trial"].get("incumbent", "start"),
                "agent_seconds": s["config"]["agent"]["seconds"],
                "model": s["config"]["agent"]["model"],
                "teacher": prepared.get("teacher"),
                "rules": prepared.get("rules"),
            },
            "lines": lines,
            "trials": trials,
            "events": self.store.events(s["id"], 60),
            "experiments": self.store.experiments(),
        }

    def trial_detail(self, trial_id):
        s = self.store.read()
        trial = s["trials"].get(trial_id) if s else None
        if trial is None:
            raise ValueError("未知的方案尝试")
        root = self.trial_root(s, trial)
        detail = dict(trial)
        diff = root / "diff.txt"
        detail["diff"] = diff.read_text()[:60000] if diff.exists() else None
        training = trial.get("training") or {}
        metrics = (
            self.job_root(s, training["job"]) / "metrics.jsonl"
            if training.get("job")
            else None
        )
        detail["metrics"] = (
            [
                compact_metrics(json.loads(x))
                for x in metrics.read_text().splitlines()
                if x.strip()
            ]
            if metrics and metrics.exists()
            else []
        )
        return detail
