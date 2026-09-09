"""Research lifecycle: branches own exploration, cycles own shared competition."""

from copy import deepcopy
from datetime import date
from pathlib import Path
import json
import re
import secrets
import shutil
import time
from mahjong_ai.persistence.store import Store
from mahjong_ai.persistence.artifacts import (
    atomic_json,
    digest,
    read_json,
    runtime_fingerprint,
)
from mahjong_ai.execution.candidate_runtime import tree_hash
from mahjong_ai.strategies.catalog import baseline_source
from mahjong_ai.research.config import (
    configuration,
    merge,
    validate,
    IMMEDIATE,
    ITERATION,
    NEW_RUN,
)
from mahjong_ai.evaluation.schedule import schedule, reference_groups, eliminated


class Research:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.store = Store(self.project)

    def read(self, run):
        return self.store.read(run)

    def edit(self, run):
        return self.store.edit(run)

    def runs(self, active_only=False):
        return [
            s
            for s in self.store.runs()
            if not active_only or s["status"] in ("running", "pausing")
        ]

    def event(self, db, s, kind, data):
        self.store.event(db, s, kind, data)

    def create(self, overrides=None):
        config = configuration(self.project, overrides)
        base = self.project / ".local/runs"
        base.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            root = base / f"{date.today():%Y%m%d}-research-{index:02d}"
            try:
                root.mkdir(mode=0o700)
                break
            except FileExistsError:
                index += 1
        try:
            shutil.copytree(
                Path(__file__).parents[1],
                root / "runtime/mahjong_ai",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            (root / "reports").mkdir()
            (root / "private").mkdir(mode=0o700)
            atomic_json(root / "private/random.json", {"key": secrets.token_hex(32)})
            s = dict(
                id=root.name,
                root=str(root),
                runtime=runtime_fingerprint(),
                status="paused",
                phase="initializing",
                pause_reason="ready",
                created_at=time.time(),
                config=config,
                pending={},
                revisions=[],
                branches={},
                candidates={},
                iterations={},
                cycles=[],
                jobs={},
                baselines={},
                last_ranking=[],
                model_history=[],
            )
            for combination in config["combinations"]:
                self.add_combination(s, combination)
            self.store.create(s)
        except BaseException:
            shutil.rmtree(root)
            raise
        return s["id"]

    def add_combination(self, s, combination):
        key = combination["id"]
        cid = "baseline--" + key
        if key not in s["baselines"]:
            package = Path(s["root"]) / "baselines" / key / "source"
            package.mkdir(parents=True)
            (package / "strategy.py").write_text(
                baseline_source(
                    key, combination.get("parameters"), combination["family"]
                )
            )
            candidate = dict(
                id=cid,
                branch=None,
                combination=key,
                status="submitted",
                workspace=str(package),
                parent=None,
                hypothesis="shared initialized baseline",
                requirements=["lightgbm==4.6.0"]
                if combination["parameters"].get("algorithm") == "lightgbm"
                else [],
            )
            s["candidates"][cid] = candidate
            s["baselines"][key] = cid
            self.queue(
                s,
                "prepare",
                branch=None,
                candidates=[cid],
                combination=key,
                initialization=True,
                training_games=combination.get(
                    "training_games", s["config"]["training_games"]
                ),
            )
        for i, model in enumerate(combination["models"]):
            bid = f"{key}--{i + 1:02d}"
            if bid in s["branches"]:
                b = s["branches"][bid]
                if (
                    b["model"] != model
                    or b["model_config"] != s["config"]["models"][model]
                ):
                    s["model_history"].append(
                        dict(
                            branch=bid,
                            old=b["model"],
                            new=model,
                            at=time.time(),
                            iteration=b["completed"],
                        )
                    )
                    b["model"] = model
                    b["model_config"] = deepcopy(s["config"]["models"][model])
                    b["thread_id"] = None
                continue
            s["branches"][bid] = dict(
                id=bid,
                combination=key,
                model=model,
                model_config=deepcopy(s["config"]["models"][model]),
                status="initializing",
                completed=0,
                successful=0,
                failed=0,
                representative=None,
                thread_id=None,
                turns=0,
                last_run=0.0,
            )

    @staticmethod
    def queue(s, kind, **fields):
        jid = f"{kind}-{len(s['jobs']):07d}"
        job = dict(
            id=jid,
            kind=kind,
            status="queued",
            created_at=time.time(),
            spent=0.0,
            **fields,
        )
        s["jobs"][jid] = job
        return job

    def control(self, run, action, iterations=0):
        with self.edit(run) as (s, db):
            if action == "pause":
                s.update(
                    status="pausing" if s["status"] == "running" else "paused",
                    pause_reason="user paused",
                )
            elif action == "resume":
                if s["status"] == "pausing":
                    raise ValueError("wait for pause to finish")
                if any(
                    r["id"] != run and r["status"] in ("running", "pausing")
                    for r in self.runs()
                ):
                    raise ValueError("pause other research first")
                if s["phase"] == "finished" and all(
                    b["completed"] >= s["config"]["total_iterations"]
                    for b in s["branches"].values()
                    if b["status"] != "eliminated"
                ):
                    raise ValueError("add iterations before resuming")
                s.update(status="running", pause_reason="")
                for j in s["jobs"].values():
                    if j["status"] == "paused":
                        j["status"] = "queued"
                if s["phase"] == "finished":
                    s["phase"] = "boundary"
                for combination, cid in s["baselines"].items():
                    if s["candidates"][cid]["status"] == "failed":
                        s["candidates"][cid]["status"] = "submitted"
                        cfg = next(
                            c
                            for c in s["config"]["combinations"]
                            if c["id"] == combination
                        )
                        self.queue(
                            s,
                            "prepare",
                            branch=None,
                            candidates=[cid],
                            combination=combination,
                            initialization=True,
                            training_games=cfg.get(
                                "training_games", s["config"]["training_games"]
                            ),
                        )
            elif action == "extend":
                if type(iterations) is not int or iterations < 1:
                    raise ValueError("positive additional iterations required")
                s["config"]["total_iterations"] += iterations
            else:
                raise ValueError("unknown action")
            self.event(db, s, action, {"iterations": iterations})
        return self.snapshot(run)

    def configure(self, run, patch):
        with self.edit(run) as (s, db):
            if set(patch) & NEW_RUN:
                raise ValueError("game and action windows require new research")
            prospective = validate(merge(merge(s["config"], s["pending"]), patch))
            committed = max(
                [b["completed"] for b in s["branches"].values()]
                + list(
                    s["cycles"][-1]["targets"].values()
                    if s["cycles"] and s["cycles"][-1]["status"] != "completed"
                    else []
                )
            )
            if prospective["total_iterations"] < committed:
                raise ValueError("cannot remove completed or current-cycle iterations")
            old = {c["id"]: c for c in s["config"]["combinations"]}
            for key, c in old.items():
                new = next(
                    (x for x in prospective["combinations"] if x["id"] == key), None
                )
                if new and (
                    new["family"] != c["family"]
                    or any(
                        new["parameters"].get(k) != c["parameters"].get(k)
                        for k in ("style", "algorithm")
                    )
                ):
                    raise ValueError(
                        "changing algorithm identity requires a new combination"
                    )
                if not new or new["branches"] < c["branches"]:
                    raise ValueError(
                        "remove branches through elimination, not configuration"
                    )
            for k in patch:
                if k in IMMEDIATE:
                    s["config"][k] = prospective[k]
                else:
                    s["pending"][k] = prospective[k]
            s["revisions"].append(dict(at=time.time(), patch=patch))
            self.event(db, s, "configuration", {"keys": list(patch)})
        return self.snapshot(run)

    @staticmethod
    def own(s, branch, cid):
        if cid not in s["candidates"]:
            cid = branch + "--" + cid
        c = s["candidates"].get(cid)
        if not c or c["branch"] != branch:
            raise ValueError("candidate must be owned by this branch")
        return c

    @staticmethod
    def require_turn(s, branch):
        if s["status"] != "running" or s["branches"][branch]["status"] != "researching":
            raise ValueError("branch has no active research turn")
        iteration = s["iterations"][s["branches"][branch]["iteration"]]
        if iteration["status"] != "researching":
            raise ValueError("batch already submitted")
        return iteration

    def fork(self, run, branch, name, hypothesis, parent=None):
        if not re.fullmatch("[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("invalid candidate name")
        with self.edit(run) as (s, db):
            self.require_turn(s, branch)
            cid = branch + "--" + name
            if cid in s["candidates"]:
                raise ValueError("candidate already exists")
            b = s["branches"][branch]
            previous = self.own(s, branch, parent or b["representative"])
            source = (
                previous.get("spec", {}).get("package_path") or previous["workspace"]
            )
            dest = Path(s["root"]) / "branches" / branch / "drafts" / name
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__"))
            if previous["status"] == "failed":
                for job in reversed(list(s["jobs"].values())):
                    if job["kind"] == "prepare" and previous["id"] in job.get(
                        "candidates", []
                    ):
                        checkpoint = (
                            Path(s["root"])
                            / "jobs"
                            / job["id"]
                            / previous["id"]
                            / "training-checkpoint.json"
                        )
                        if checkpoint.exists():
                            row = read_json(checkpoint)
                            committed = checkpoint.parent / row["directory"]
                            if tree_hash(committed) != row["hash"]:
                                raise ValueError("training checkpoint changed")
                            shutil.rmtree(dest / "model", ignore_errors=True)
                            shutil.copytree(committed, dest / "model")
                        break
            s["candidates"][cid] = dict(
                id=cid,
                branch=branch,
                combination=b["combination"],
                status="draft",
                workspace=str(dest),
                parent=previous["id"],
                hypothesis=hypothesis[:8000],
                requirements=previous["requirements"],
            )
        return {"candidate_id": cid}

    def candidate(self, run, branch, cid, path=None, content=None):
        with self.edit(run) as (s, db):
            c = self.own(s, branch, cid)
            if content is not None:
                self.require_turn(s, branch)
                if c["status"] != "draft":
                    raise ValueError("only drafts can be edited")
            base = Path(c.get("spec", {}).get("package_path") or c["workspace"])
            if path is None:
                return {k: v for k, v in c.items() if k not in ("workspace", "spec")}
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("relative candidate path required")
            file = base / relative
            if not file.resolve().is_relative_to(base.resolve()):
                raise ValueError("candidate path escaped")
            if content is not None:
                if not isinstance(content, str) or len(content.encode()) > 2_000_000:
                    raise ValueError("text file too large")
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(content)
                return {"written": path}
            return {"path": path, "content": file.read_text()}

    def dependencies(self, run, branch, cid, requirements):
        if len(requirements) > 32 or any(
            not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+-]*", r
            )
            for r in requirements
        ):
            raise ValueError("exact name==version wheels required")
        with self.edit(run) as (s, db):
            self.require_turn(s, branch)
            c = self.own(s, branch, cid)
            if c["status"] != "draft":
                raise ValueError("dependencies frozen")
            c["requirements"] = requirements
        return {"requirements": requirements}

    def submit(self, run, branch, candidates, training_games=None):
        with self.edit(run) as (s, db):
            previous = s["iterations"].get(s["branches"][branch].get("iteration"))
            if previous and previous["status"] != "researching":
                ids = [self.own(s, branch, cid)["id"] for cid in candidates]
                if ids == previous["candidates"]:
                    return {
                        "iteration": previous["number"],
                        "status": "submitted",
                        "instruction": "Batch accepted. End this reply now; evaluation is managed by the controller.",
                    }
            it = self.require_turn(s, branch)
            if not 1 <= len(candidates) <= it["config"]["candidate_limit"] or len(
                set(candidates)
            ) != len(candidates):
                raise ValueError("submit one to four distinct drafts")
            rows = [self.own(s, branch, cid) for cid in candidates]
            if any(c["status"] != "draft" for c in rows):
                raise ValueError("only draft candidates can be submitted")
            games = (
                training_games
                if training_games is not None
                else it["config"]["training_games"]
            )
            if type(games) is not int or not 1 <= games <= 1000000:
                raise ValueError("invalid training games")
            root = Path(s["root"]) / "submissions" / it["id"]
            if root.exists():
                shutil.rmtree(root)
            try:
                for c in rows:
                    source = Path(c["workspace"])
                    if not (source / "strategy.py").is_file():
                        raise ValueError("strategy.py required")
                    target = root / c["id"]
                    shutil.copytree(source, target)
                    if tree_hash(source) != tree_hash(target):
                        raise ValueError("source changed during submission")
                for c in rows:
                    c.update(status="submitted", workspace=str(root / c["id"]))
                it.update(
                    status="submitted",
                    candidates=[c["id"] for c in rows],
                    training_games=games,
                )
            except BaseException:
                shutil.rmtree(root, ignore_errors=True)
                raise
            it["submitted_at"] = time.time()
            self.event(
                db,
                s,
                "batch_submitted",
                {
                    "branch": branch,
                    "iteration": it["number"],
                    "candidates": it["candidates"],
                },
            )
        return {"iteration": it["number"], "status": "submitted"}

    def initialize_ready(self, s):
        for b in s["branches"].values():
            if b["status"] != "initializing":
                continue
            base = s["candidates"][s["baselines"][b["combination"]]]
            if base["status"] != "ready":
                continue
            cid = b["id"] + "--initial"
            dest = Path(s["root"]) / "branches" / b["id"] / "initial"
            if not dest.exists():
                shutil.copytree(base["spec"]["package_path"], dest)
            spec = {**base["spec"], "package_path": str(dest), "version": cid}
            s["candidates"][cid] = {
                **base,
                "id": cid,
                "branch": b["id"],
                "spec": spec,
                "status": "representative",
                "hypothesis": "shared initial baseline",
            }
            b.update(status="ready", representative=cid)

    def begin_cycle(self, s, db):
        for key in list(s["pending"]):
            s["config"][key] = s["pending"].pop(key)
        for c in s["config"]["combinations"]:
            self.add_combination(s, c)
        self.initialize_ready(s)
        if any(b["status"] == "initializing" for b in s["branches"].values()):
            s["phase"] = "initializing"
            return
        active = [b for b in s["branches"].values() if b["status"] != "eliminated"]
        length = (
            s["config"]["first_cycle"]
            if not s["cycles"]
            else s["config"]["cycle_iterations"]
        )
        remaining = max(
            0, s["config"]["total_iterations"] - min(b["completed"] for b in active)
        )
        if not remaining:
            s.update(
                status="paused",
                phase="finished",
                pause_reason="iteration target reached",
            )
            return
        length = min(length, remaining)
        number = len(s["cycles"]) + 1
        cycle = dict(
            id=number,
            status="researching",
            config=deepcopy(s["config"]),
            targets={
                b["id"]: min(s["config"]["total_iterations"], b["completed"] + length)
                for b in active
            },
            reference={b["id"]: b["representative"] for b in active},
            opponents={},
            ranking=[],
            eliminated=[],
            full=length
            == (
                s["config"]["first_cycle"]
                if not s["cycles"]
                else s["config"]["cycle_iterations"]
            ),
        )
        key = read_json(Path(s["root"]) / "private/random.json")["key"]
        for b in active:
            cycle["opponents"][b["id"]] = reference_groups(
                active,
                b["id"],
                digest([key, number, b["id"]]),
                cycle["config"]["evaluation_games"] // 4,
            )
            b["status"] = "ready"
        s["cycles"].append(cycle)
        s["phase"] = "researching"
        self.event(db, s, "cycle_started", {"cycle": number})

    def start_iteration(self, s, branch):
        b = s["branches"][branch]
        cfg = merge(
            s["config"], {k: v for k, v in s["pending"].items() if k in ITERATION}
        )
        combination = next(
            c for c in cfg["combinations"] if c["id"] == b["combination"]
        )
        if "training_games" not in s["pending"]:
            cfg["training_games"] = combination.get(
                "training_games", cfg["training_games"]
            )
        number = b["completed"] + 1
        iid = f"{branch}--iteration-{number:05d}"
        s["iterations"][iid] = dict(
            id=iid,
            branch=branch,
            number=number,
            cycle=s["cycles"][-1]["id"],
            status="researching",
            config=deepcopy(cfg),
            incumbent=b["representative"],
            candidates=[],
            spent=0.0,
            results={},
        )
        b.update(
            status="researching",
            iteration=iid,
            last_run=time.time(),
            turns=b["turns"] + 1,
        )
        return self.queue(s, "agent", branch=branch, iteration=iid)

    def advance(self, s, db):
        if s["status"] != "running":
            return
        self.initialize_ready(s)
        if s["phase"] in ("initializing", "boundary"):
            if any(b["status"] == "initializing" for b in s["branches"].values()):
                return
            self.begin_cycle(s, db)
        if s["phase"] == "researching":
            cycle = s["cycles"][-1]
            for b in s["branches"].values():
                if b["status"] == "eliminated":
                    continue
                it = s["iterations"].get(b.get("iteration"))
                if (
                    it
                    and it["status"] == "submitted"
                    and not any(
                        j["kind"] == "agent"
                        and j.get("iteration") == it["id"]
                        and j["status"] in ("running", "queued", "admitted")
                        for j in s["jobs"].values()
                    )
                ):
                    it["status"] = "preparing"
                    b["status"] = "preparing"
                    self.queue(
                        s,
                        "prepare",
                        branch=b["id"],
                        iteration=it["id"],
                        candidates=it["candidates"],
                        combination=b["combination"],
                        training_games=it["training_games"],
                    )
                if (
                    b["status"] == "ready"
                    and b["completed"] >= cycle["targets"][b["id"]]
                ):
                    b["status"] = "waiting"
            if all(
                b["status"] in ("waiting", "eliminated") for b in s["branches"].values()
            ):
                cycle["status"] = "competing"
                s["phase"] = "competing"
                roster = [
                    b["representative"]
                    for b in s["branches"].values()
                    if b["status"] != "eliminated"
                ]
                matches = (
                    len(roster) * cycle["config"]["competition_games_per_branch"] // 4
                )
                matches = (matches + 3) // 4 * 4
                key = read_json(Path(s["root"]) / "private/random.json")["key"]
                assignments = schedule(
                    roster, matches, digest([key, "competition", cycle["id"]])
                )
                for block in range(0, len(assignments), 4):
                    self.queue(
                        s,
                        "match",
                        scope="competition",
                        cycle=cycle["id"],
                        assignments=assignments[block : block + 4],
                    )
                self.event(
                    db,
                    s,
                    "competition_started",
                    {"cycle": cycle["id"], "matches": matches},
                )
        if s["phase"] == "competing":
            cycle = s["cycles"][-1]
            jobs = [
                j
                for j in s["jobs"].values()
                if j.get("scope") == "competition" and j["cycle"] == cycle["id"]
            ]
            if not jobs or any(j["status"] != "completed" for j in jobs):
                return
            table = self.competition_table(s, cycle)
            removed = (
                eliminated(
                    table,
                    s["branches"],
                    cycle["config"]["elimination_fraction"],
                    cycle["config"]["minimum_branches"],
                )
                if cycle["full"]
                else []
            )
            previous = {r["branch"]: i + 1 for i, r in enumerate(s["last_ranking"])}
            for rank, row in enumerate(table, 1):
                row.update(
                    rank=rank, rank_change=previous.get(row["branch"], rank) - rank
                )
            for bid in removed:
                s["branches"][bid]["status"] = "eliminated"
            cycle.update(
                status="completed",
                ranking=table,
                eliminated=removed,
                completed_at=time.time(),
            )
            s["last_ranking"] = table
            s["phase"] = "boundary"
            self.event(
                db, s, "cycle_completed", {"cycle": cycle["id"], "eliminated": removed}
            )
            self.begin_cycle(s, db)

    def complete_iteration(self, s, db, it, error=None):
        b = s["branches"][it["branch"]]
        if it["status"] == "completed":
            return
        success = bool(it["results"])
        if success:
            choices = [it["incumbent"], *sorted(it["candidates"])]
            valid = [cid for cid in choices if cid in it["results"]]
            winner = max(
                valid,
                key=lambda cid: (
                    it["results"][cid]["mean_utility"],
                    cid == it["incumbent"],
                    -choices.index(cid),
                ),
            )
            for cid in it["candidates"]:
                s["candidates"][cid]["status"] = (
                    "evaluated"
                    if cid in it["results"]
                    else s["candidates"][cid]["status"]
                )
            s["candidates"][b["representative"]]["status"] = "evaluated"
            b["representative"] = winner
            s["candidates"][winner]["status"] = "representative"
        b["completed"] += 1
        b["successful"] += int(success)
        b["failed"] += int(not success)
        b["status"] = "ready"
        it.pop("config", None)
        for finished in s["jobs"].values():
            if (
                finished.get("iteration") == it["id"]
                and finished["status"] == "completed"
            ):
                finished.pop("assignments", None)
        it.update(
            status="completed",
            representative=b["representative"],
            error=error,
            completed_at=time.time(),
        )
        self.event(
            db,
            s,
            "iteration_completed",
            {
                "branch": b["id"],
                "iteration": it["number"],
                "representative": b["representative"],
                "success": success,
            },
        )

    def accept(self, s, db, job, result):
        if job["status"] == "completed":
            return
        if result["status"] == "paused":
            job["status"] = "paused"
            return
        if not (job["kind"] == "match" and result["status"] != "completed"):
            self.store.receipt(db, s["id"], job["id"], result)
        job.update(
            status="completed", error=result.get("error"), usage=result.get("usage", {})
        )
        it = s["iterations"].get(job.get("iteration"))
        if it and job["kind"] in ("agent", "prepare"):
            it["spent"] += result.get("usage", {}).get("wall_seconds", 0.0)
        if job["kind"] == "agent":
            b = s["branches"][job["branch"]]
            if result.get("thread_id"):
                b["thread_id"] = result["thread_id"]
            if it["status"] == "researching":
                self.complete_iteration(
                    s, db, it, result.get("error") or "no candidate submitted"
                )
        elif job["kind"] == "prepare":
            ready = []
            for cid in job["candidates"]:
                c = s["candidates"][cid]
                row = result.get("candidates", {}).get(
                    cid,
                    {
                        "status": "failed",
                        "error": result.get("error", "preparation incomplete"),
                    },
                )
                c.update(row)
                if row["status"] == "ready":
                    ready.append(cid)
            if job.get("initialization"):
                if not ready:
                    s.update(
                        status="pausing",
                        pause_reason="baseline initialization failed: "
                        + job["combination"],
                    )
            elif not ready:
                self.complete_iteration(s, db, it, "no prepared candidates")
            else:
                it["status"] = "evaluating"
                s["branches"][it["branch"]]["status"] = "evaluating"
                cycle = s["cycles"][it["cycle"] - 1]
                for cid in [it["incumbent"], *ready]:
                    assignments = []
                    for block, opponents in enumerate(cycle["opponents"][it["branch"]]):
                        for seat in range(4):
                            players = list(opponents)
                            players.insert(seat, cid)
                            assignments.append(
                                dict(
                                    index=len(assignments),
                                    block=block,
                                    players=players,
                                    focal=seat,
                                )
                            )
                    self.queue(
                        s,
                        "match",
                        scope="paired",
                        branch=it["branch"],
                        iteration=it["id"],
                        cycle=cycle["id"],
                        candidate=cid,
                        assignments=assignments,
                    )
        elif job["kind"] == "match":
            if result["status"] != "completed":
                job["status"] = "paused"
                s.update(
                    status="pausing",
                    pause_reason="evaluation job failed: " + str(result.get("error")),
                )
                return
            job["summary"] = result["summary"]
            if job["scope"] == "paired":
                it["results"][job["candidate"]] = result["focal"]
                s["candidates"][job["candidate"]]["last_result"] = result["focal"]
                jobs = [
                    j
                    for j in s["jobs"].values()
                    if j.get("iteration") == it["id"] and j["kind"] == "match"
                ]
                if all(j["status"] == "completed" for j in jobs):
                    self.complete_iteration(s, db, it)

    @staticmethod
    def competition_table(s, cycle):
        rows = {}
        for j in s["jobs"].values():
            if (
                j.get("scope") != "competition"
                or j["cycle"] != cycle["id"]
                or j["status"] != "completed"
            ):
                continue
            for r in j.get("summary", []):
                dst = rows.setdefault(
                    r["candidate_id"],
                    {
                        k: 0
                        for k in (
                            "appearances",
                            "utility_sum",
                            "errors",
                            "timeouts",
                            "fallback_actions",
                        )
                    },
                )
                for k in dst:
                    dst[k] += r[k]
        result = []
        for b in s["branches"].values():
            cid = b["representative"]
            if cid in rows:
                r = rows[cid]
                result.append(
                    dict(
                        **r,
                        candidate_id=cid,
                        branch=b["id"],
                        combination=b["combination"],
                        model=b["model"],
                        effort=b["model_config"]["effort"],
                        mean_utility=r["utility_sum"] / r["appearances"],
                    )
                )
        return sorted(
            result, key=lambda r: (-r["mean_utility"], -r["appearances"], r["branch"])
        )

    def snapshot(self, run=None, branch=None, after=0):
        runs = self.runs()
        if not runs:
            return {"run": None, "runs": [], "defaults": configuration(self.project)}
        s = self.read(run or runs[0]["id"])
        if branch and branch not in s["branches"]:
            raise ValueError("unknown branch")
        cycle = deepcopy(s["cycles"][-1]) if s["cycles"] else None
        if cycle:
            cycle.pop("opponents")
            cycle.pop("reference")
            cycle.pop("config")
            if s["phase"] == "competing":
                cycle["ranking"] = self.competition_table(s, s["cycles"][-1])
        branches = []
        for b in s["branches"].values():
            row = {k: v for k, v in b.items() if k not in ("thread_id", "model_config")}
            row["effort"] = b["model_config"]["effort"]
            branches.append(row)
        candidates = []
        for c in s["candidates"].values():
            if branch and c["branch"] != branch:
                continue
            candidates.append(
                {
                    k: v
                    for k, v in c.items()
                    if k not in ("workspace", "spec", "training")
                }
            )
        iterations = [
            {k: v for k, v in it.items() if k != "config"}
            for it in s["iterations"].values()
            if not branch or it["branch"] == branch
        ]
        ranking = [
            r
            for r in (
                cycle["ranking"]
                if cycle and s["phase"] == "competing"
                else s["last_ranking"]
            )
            if s["branches"][r["branch"]]["status"] != "eliminated"
        ]
        jobs = []
        for j in s["jobs"].values():
            if j["status"] not in ("running", "admitted", "queued"):
                continue
            if branch and j.get("branch") != branch:
                continue
            row = {
                k: j[k]
                for k in ("id", "kind", "status", "branch", "scope", "started_at")
                if k in j
            }
            progress = Path(s["root"]) / "jobs" / j["id"] / "progress.json"
            if not branch and progress.exists():
                row["progress"] = read_json(progress)
            jobs.append(row)
        cfg = deepcopy(s["config"])
        # Model profile paths are configuration, but provider authentication never enters state.
        return dict(
            run={k: s[k] for k in ("id", "status", "phase", "pause_reason")},
            config=cfg,
            pending=s["pending"],
            branches=branches,
            candidates=candidates,
            iterations=iterations,
            cycle=cycle,
            ranking=ranking,
            jobs=jobs,
            history=[
                dict(cycle=c["id"], ranking=c["ranking"], eliminated=c["eliminated"])
                for c in s["cycles"]
                if c["status"] == "completed"
            ],
            model_history=s["model_history"],
            events=[
                e
                for e in self.store.events(s["id"], after)
                if not branch or e["data"].get("branch") in (None, branch)
            ],
            runs=[{"id": r["id"], "status": r["status"]} for r in runs],
        )

    def activity(self, run):
        s = self.read(run)
        out = []
        for job in reversed(list(s["jobs"].values())):
            if job["kind"] != "agent":
                continue
            path = Path(s["root"]) / "jobs" / job["id"] / "codex.jsonl"
            if not path.exists():
                continue
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 131072))
                lines = stream.read().decode(errors="replace").splitlines()
            for line in lines:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                item = e.get("item", {})
                if (
                    e.get("type") == "item.completed"
                    and item.get("type") == "agent_message"
                ):
                    out.append(
                        dict(branch=job["branch"], text=item.get("text", "")[:8000])
                    )
            if len(out) >= 20:
                break
        return out[:20]
