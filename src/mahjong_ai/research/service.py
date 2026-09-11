"""Persistent strategy registry and sequential, two-phase experiments."""

from copy import deepcopy
from pathlib import Path
import json
import re
import secrets
import shutil
import time
from math import sqrt
from statistics import variance
from mahjong_ai.persistence.store import Store
from mahjong_ai.project import ProjectDescriptor
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
)
from mahjong_ai.evaluation.schedule import (
    schedule,
    reference_groups,
    metric_uncertainty,
)


def _competition_registry(research):
    """Return the shared registry, when this checkout provides one."""
    try:
        from mahjong_ai.competition.registry import CompetitionRegistry

        root = getattr(research.layout, "workspace", None) or research.project.parent
        return CompetitionRegistry(root)
    except (ImportError, AttributeError):
        return None


def paired_readout(reference, candidate, metric="field"):
    """同一副牌、同一对手下的配对数：逐块差值的均值与聚类标准误。

    主口径是场（平台按场结算，场内局与局不对等）；逐局配对读数只作辅助。
    """
    if not reference or not candidate:
        return None
    key = "hand_group_means" if metric == "hand" else "field_group_means"
    lookup = dict(zip(reference.get("group_blocks") or [], reference.get(key) or []))
    diffs = [
        value - lookup[block]
        for block, value in zip(
            candidate.get("group_blocks") or [], candidate.get(key) or []
        )
        if block in lookup
    ]
    if not diffs:
        return None
    count = len(diffs)
    return dict(
        blocks=count,
        delta=sum(diffs) / count,
        standard_error=sqrt(variance(diffs) / count) if count >= 2 else None,
    )


class Research:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.layout = ProjectDescriptor.load(self.project)
        self.store = Store(self.project)

    def read(self, experiment=None):
        state = self.store.read(experiment)
        if state is None:
            raise ValueError("策略注册表尚未初始化")
        return state

    def edit(self, experiment=None):
        return self.store.edit(experiment)

    def states(self, active_only=False):
        state = self.store.read()
        return (
            [state]
            if state and (not active_only or state["status"] in ("running", "pausing"))
            else []
        )

    def event(self, db, s, kind, data):
        self.store.event(db, s, kind, data)

    def freeze_runtime(self):
        fingerprint = runtime_fingerprint()
        runtime = self.layout.research / "runtimes" / fingerprint
        if not runtime.exists():
            temporary = runtime.with_name(fingerprint + "." + secrets.token_hex(5))
            shutil.copytree(
                Path(__file__).parents[1],
                temporary / "mahjong_ai",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            runtime.parent.mkdir(parents=True, exist_ok=True)
            try:
                temporary.rename(runtime)
            except OSError:
                if not runtime.exists():
                    raise
                shutil.rmtree(temporary)
        return fingerprint, str(runtime)

    def register(self, overrides=None):
        """Register configured combinations without starting agents or model training."""
        from mahjong_ai.persistence.artifacts import exclusive_lock

        with exclusive_lock(self.layout.service / "registry.lock", blocking=True):
            config = configuration(self.project, overrides)
            if self.store.read() is None:
                root = self.layout.research
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
                (root / "reports").mkdir(exist_ok=True)
                (root / "private").mkdir(exist_ok=True, mode=0o700)
                atomic_json(
                    root / "private/random.json", {"key": secrets.token_hex(32)}
                )
                (root / "private/random.json").chmod(0o600)
                fingerprint, runtime = self.freeze_runtime()
                self.store.create(
                    dict(
                        id=None,
                        root=str(root),
                        runtime=fingerprint,
                        runtime_path=runtime,
                        status="idle",
                        phase="idle",
                        pause_reason="",
                        stop_plan=None,
                        created_at=time.time(),
                        config=config,
                        revisions=[],
                        branches={},
                        candidates={},
                        iterations={},
                        cycles=[],
                        jobs={},
                        baselines={},
                        definitions={},
                        last_ranking=[],
                        model_history=[],
                        experiments=[],
                        bootstrap=[],
                    )
                )
            with self.edit() as (state, db):
                if state["status"] in ("running", "pausing"):
                    raise ValueError("注册策略前先暂停实验")
                for combination in config["combinations"]:
                    self.add_combination(state, combination, config["models"])
                if not state["bootstrap"]:
                    for style in ("conservative", "balanced", "adventurous"):
                        key = "heuristic-" + style
                        cid = "bootstrap--" + style
                        folder = self.layout.strategies / key / "bootstrap"
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "strategy.py").write_text(baseline_source(key))
                        from mahjong_ai.strategies.common import PolicySpec

                        spec = PolicySpec(
                            family="code",
                            version=cid,
                            package_path=str(folder),
                            package_hash=tree_hash(folder),
                            runtime_path=state["runtime_path"],
                        ).as_dict()
                        state["candidates"][cid] = dict(
                            id=cid,
                            branch=None,
                            combination=key,
                            status="ready",
                            workspace=str(folder),
                            spec=spec,
                            parent=None,
                            hypothesis="启动启发式",
                            requirements=[],
                            runtime_path=state["runtime_path"],
                        )
                        state["bootstrap"].append(cid)
        return self.read()

    def sync_resident_releases(self, s):
        """Mirror active resident releases as read-only external candidates.

        The candidate snapshot is updated only at a research boundary; cycles
        retain their own reference list, so a release removed mid-round still
        completes the round it entered.
        """
        registry = _competition_registry(self)
        if registry is None:
            return []
        try:
            releases = registry.list(active_only=True, resident_only=True)
        except Exception:
            return []
        active = []
        for release in releases:
            cid = "external--" + release["entrant_id"]
            try:
                from mahjong_ai.competition.releases import load_policy, verify_release

                verify_release(release["bundle_path"])
                spec = load_policy(release["bundle_path"])
                spec = spec.as_dict() if hasattr(spec, "as_dict") else dict(spec)
            except (ImportError, OSError, ValueError, KeyError):
                continue
            spec["version"] = cid
            metadata = release.get("metadata") or {}
            row = s["candidates"].get(cid, {})
            row.update(
                id=cid,
                branch=None,
                external=True,
                entrant_id=release["entrant_id"],
                project_id=release["project_id"],
                combination=metadata.get("combination", "external"),
                status="resident",
                iterable=False,
                workspace=release["bundle_path"],
                spec=spec,
                requirements=[],
                runtime_path=s["runtime_path"],
                metadata=metadata,
            )
            s["candidates"][cid] = row
            active.append(cid)
        return active

    def add_combination(self, s, combination, models):
        key = combination["id"]
        old = s["definitions"].get(key)
        if old and (
            old["family"] != combination["family"]
            or old["parameters"] != combination["parameters"]
        ):
            raise ValueError("修改已注册的组合需要使用新的组合 ID")
        s["definitions"][key] = deepcopy(combination)
        for i, model in enumerate(combination["models"]):
            suffix = combination.get(
                "branch_ids", [f"{n + 1:02d}" for n in range(combination["branches"])]
            )[i]
            bid = f"{key}--{suffix}"
            if bid in s["branches"]:
                b = s["branches"][bid]
                if b["model"] != model or b["model_config"] != models[model]:
                    if s["id"] and s["phase"] not in ("finished", "closed", "idle"):
                        raise ValueError("切换研究模型前先结束当前实验")
                    s["model_history"].append(
                        dict(
                            branch=bid,
                            old=b["model"],
                            new=model,
                            at=time.time(),
                            iteration=b["completed"],
                        )
                    )
                    b.update(
                        model=model,
                        model_config=deepcopy(models[model]),
                        thread_id=None,
                    )
                continue
            s["branches"][bid] = dict(
                id=bid,
                combination=key,
                model=model,
                model_config=deepcopy(models[model]),
                status="uninitialized",
                archived=False,
                completed=0,
                successful=0,
                improved=0,
                failed=0,
                representative=None,
                thread_id=None,
                turns=0,
                last_run=0.0,
            )

    def create(
        self,
        overrides=None,
        targets=None,
        opponents=None,
        candidate_runtime_path=None,
    ):
        config = configuration(self.project, overrides)
        current = self.store.read()
        if current and current["id"] and current["phase"] not in ("finished", "closed"):
            raise ValueError("开始新实验前先结束或关闭当前实验")
        self.register(overrides)
        fingerprint, runtime = self.freeze_runtime()
        if candidate_runtime_path is not None:
            candidate_runtime = Path(candidate_runtime_path).expanduser().resolve()
            if not (candidate_runtime / "mahjong_ai").is_dir():
                raise ValueError("候选运行时必须包含 mahjong_ai 包目录")
            candidate_runtime_path = str(candidate_runtime)
        with self.edit() as (s, db):
            if s["id"] and s["phase"] not in ("finished", "closed"):
                raise ValueError("已存在未完成的实验")
            selected = (
                list(targets)
                if targets is not None
                else [b for b, row in s["branches"].items() if not row["archived"]]
            )
            if (
                not selected
                or len(selected) != len(set(selected))
                or any(
                    bid not in s["branches"] or s["branches"][bid]["archived"]
                    for bid in selected
                )
            ):
                raise ValueError("请选择互不重复、已注册且未归档的迭代分支")
            if set(config["branch_iterations"]) - set(selected):
                raise ValueError("单支迭代额度只能指向本次选中的迭代对象")
            pool = (
                self.default_opponents(s, config["opponent_limit"])
                if opponents is None
                else list(opponents)
            )
            if not pool or len(pool) != len(set(pool)):
                raise ValueError("至少选择一个互不重复的对手版本")
            for cid in pool:
                c = s["candidates"].get(cid)
                if (
                    not c
                    or not c.get("spec")
                    or (
                        c.get("branch") and not s["branches"][c["branch"]]["successful"]
                    )
                ):
                    raise ValueError("对手必须是已完成有效评估分支的准备就绪版本")
                if self.eligible(s) and cid in s["bootstrap"]:
                    raise ValueError("启动策略只在还没有实际策略时用作对手")
            identity = secrets.token_hex(8)
            experiment = dict(
                id=identity,
                created_at=time.time(),
                status="paused",
                targets=selected,
                initial_opponents=pool,
                total_rounds=config["total_rounds"],
                completed_rounds=0,
                config=deepcopy(config),
                runtime=fingerprint,
                runtime_path=runtime,
                candidate_runtime_path=candidate_runtime_path,
            )
            s["experiments"].append(experiment)
            s.update(
                id=identity,
                status="paused",
                phase="initializing",
                pause_reason="ready",
                config=config,
                runtime=fingerprint,
                runtime_path=runtime,
                candidate_runtime_path=candidate_runtime_path,
            )
            for b in s["branches"].values():
                b["status"] = "ready" if b["representative"] else "uninitialized"
            for bid in selected:
                b = s["branches"][bid]
                if b["representative"]:
                    b["status"] = "ready"
                    continue
                b["status"] = "initializing"
                key = b["combination"]
                if key in s["baselines"]:
                    continue
                combination = s["definitions"][key]
                cid = "baseline--" + key
                package = self.layout.strategies / key / "baseline/source"
                package.mkdir(parents=True, exist_ok=True)
                (package / "strategy.py").write_text(
                    baseline_source(
                        key, combination["parameters"], combination["family"]
                    )
                )
                s["candidates"][cid] = dict(
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
                    runtime_path=runtime,
                )
                s["baselines"][key] = cid
                self.queue(
                    s,
                    "prepare",
                    branch=None,
                    candidates=[cid],
                    combination=key,
                    initialization=True,
                    training_games=combination.get(
                        "training_games", config["training_games"]
                    ),
                )
            root = Path(s["root"]) / "experiments" / identity
            atomic_json(root / "experiment.json", experiment)
            self.event(
                db, s, "experiment_created", {"targets": selected, "opponents": pool}
            )
        return identity

    @staticmethod
    def eligible(s):
        return [
            b
            for b in s["branches"].values()
            if not b["archived"] and b["successful"] > 0 and b["representative"]
        ]

    def default_opponents(self, s, limit):
        residents = self.sync_resident_releases(s)
        eligible = {b["id"]: b for b in self.eligible(s)}
        if not eligible:
            return residents + list(s["bootstrap"])
        ranked = [
            r["candidate_id"] for r in s["last_ranking"] if r["branch"] in eligible
        ]
        ranked += [
            b["representative"]
            for bid, b in sorted(eligible.items())
            if bid not in {r["branch"] for r in s["last_ranking"]}
        ]
        # Residents are permanent comparison objects and do not get displaced
        # by the automatic top-K pool.
        return ranked[:limit] + [cid for cid in residents if cid not in ranked[:limit]]

    @staticmethod
    def queue(s, kind, **fields):
        jid = f"{kind}-{len(s['jobs']):07d}"
        job = dict(
            id=jid,
            kind=kind,
            status="queued",
            created_at=time.time(),
            spent=0.0,
            attempt=0,
            experiment=s["id"],
            runtime_path=s["runtime_path"],
            **fields,
        )
        s["jobs"][jid] = job
        return job

    def control(self, experiment, action, rounds=0, options=None, reason=""):
        with self.edit(experiment) as (s, db):
            if not s["id"]:
                raise ValueError("请先开始实验")
            current = s["experiments"][-1]
            if action == "pause":
                s.update(
                    status="pausing" if s["status"] == "running" else "paused",
                    pause_reason=reason or "user paused",
                )
            elif action == "pause_after":
                if s["phase"] == "closed":
                    raise ValueError("已关闭的实验不能安排停止方式")
                if s["status"] != "running":
                    raise ValueError("只有运行中的实验可以安排停止方式")
                self.arrange_stop(s, options or {})
            elif action == "stop_now":
                if s["phase"] == "closed":
                    raise ValueError("已关闭的实验不能结束当前迭代")
                if s["status"] != "running":
                    raise ValueError("只有运行中的实验可以直接结束当前迭代")
                competition = bool((options or {}).get("competition", True))
                if s["phase"] == "researching":
                    discarded = self.discard_iterations(s, db)
                    self.arrange_stop(s, {"scope": "now", "competition": competition})
                else:
                    # 大乱斗已经在跑，直接结束只表示停在轮末。
                    discarded = []
                    self.arrange_stop(s, {"scope": "round", "competition": True})
                self.event(
                    db,
                    s,
                    "stop_now",
                    {"discarded": discarded, "competition": competition},
                )
            elif action == "clear_stop":
                self.restore_targets(s)
                s["stop_plan"] = None
            elif action == "resume":
                if s["status"] == "pausing":
                    raise ValueError("请等待暂停完成")
                if s["phase"] == "closed":
                    raise ValueError("已关闭的实验不能继续")
                if current["completed_rounds"] >= s["config"]["total_rounds"]:
                    raise ValueError("继续之前先追加研究轮数")
                s.update(status="running", pause_reason="")
                for j in s["jobs"].values():
                    if j["experiment"] == s["id"] and j["status"] == "paused":
                        j["status"] = "queued"
                if s["phase"] == "finished":
                    s["phase"] = "boundary"
                selected_combinations = {
                    s["branches"][bid]["combination"] for bid in current["targets"]
                }
                for combination, cid in s["baselines"].items():
                    if (
                        combination in selected_combinations
                        and s["candidates"][cid]["status"] == "failed"
                    ):
                        s["candidates"][cid]["status"] = "submitted"
                        cfg = s["definitions"][combination]
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
                if type(rounds) is not int or rounds < 1:
                    raise ValueError("追加的研究轮数必须为正数")
                s["config"]["total_rounds"] += rounds
                current["total_rounds"] = s["config"]["total_rounds"]
            elif action == "close":
                if s["status"] in ("running", "pausing"):
                    raise ValueError("结束本次实验前先暂停并等待完成")
                for it in s["iterations"].values():
                    if it["experiment"] == s["id"] and it["status"] != "completed":
                        self.complete_iteration(
                            s, db, it, "experiment closed", cancelled=True
                        )
                for j in s["jobs"].values():
                    if j["status"] != "completed":
                        j["status"] = "completed"
                        j["cancelled"] = True
                for cid in s["baselines"].values():
                    if s["candidates"][cid]["status"] == "submitted":
                        s["candidates"][cid]["status"] = "failed"
                for b in s["branches"].values():
                    b["status"] = "ready" if b["representative"] else "uninitialized"
                s.update(
                    status="paused", phase="closed", pause_reason="experiment closed"
                )
            else:
                raise ValueError("未知操作")
            current["status"] = (
                s["phase"] if s["phase"] in ("closed", "finished") else s["status"]
            )
            self.event(db, s, action, {"rounds": rounds})
        return self.snapshot(experiment)

    def arrange_stop(self, s, options):
        """记录停止安排；默认先跑完大乱斗，除非显式不要。"""
        scope = options.get("scope", "round")
        if scope not in ("round", "iteration", "now"):
            raise ValueError("停止范围必须是本轮、本次迭代或立即结束")
        competition = bool(options.get("competition", True))
        self.restore_targets(s)
        cycle = self.current_cycle(s)
        plan = {"scope": scope, "competition": competition}
        if scope == "iteration" and s["phase"] == "researching" and cycle:
            # 本轮迭代只收当前这一拍：在跑的收尾，没在跑的立即收工。
            plan["targets"] = deepcopy(cycle["targets"])
            for bid in list(cycle["targets"]):
                branch = s["branches"][bid]
                iteration = s["iterations"].get(branch.get("iteration"))
                inflight = bool(iteration) and iteration["status"] != "completed"
                cycle["targets"][bid] = branch["completed"] + (1 if inflight else 0)
        s["stop_plan"] = plan
        s.pop("pause_after_round", None)

    @staticmethod
    def current_cycle(s):
        cycles = s["cycles"]
        if cycles and cycles[-1]["experiment"] == s["id"]:
            return cycles[-1]
        return None

    def restore_targets(self, s):
        """撤销「本次迭代后停」留下的目标收缩，安排可以随时改主意。"""
        plan = s.get("stop_plan") or {}
        saved = plan.get("targets")
        cycle = self.current_cycle(s)
        if saved and cycle is not None:
            cycle["targets"] = deepcopy(saved)

    def discard_iterations(self, s, db):
        """结束当前迭代并删除未完成的迭代记录，让分支回到就绪状态。"""
        cycle = (
            s["cycles"][-1]
            if s["cycles"] and s["cycles"][-1]["experiment"] == s["id"]
            else None
        )
        if cycle is None:
            return []
        discarded = []
        for bid in list(cycle["targets"]):
            branch = s["branches"][bid]
            cycle["targets"][bid] = branch["completed"]
            iteration = s["iterations"].get(branch.get("iteration"))
            if iteration is None or iteration["status"] == "completed":
                continue
            for job in s["jobs"].values():
                if job.get("iteration") == iteration["id"] and job["status"] != (
                    "completed"
                ):
                    job.update(status="completed", cancelled=True)
            for cid in iteration["candidates"]:
                s["candidates"].pop(cid, None)
            for cid, candidate in list(s["candidates"].items()):
                if candidate.get("iteration") == iteration["id"]:
                    s["candidates"].pop(cid, None)
            s["iterations"].pop(iteration["id"], None)
            branch.update(status="ready", iteration=None)
            discarded.append(iteration["id"])
        if discarded:
            self.event(db, s, "iterations_discarded", {"iterations": discarded})
        return discarded

    def configure(self, experiment, patch):
        with self.edit(experiment) as (s, db):
            if set(patch) - IMMEDIATE:
                raise ValueError("已冻结的实验设置需要新建实验才能修改")
            config = validate(merge(s["config"], patch))
            if config["total_rounds"] < s["config"]["total_rounds"]:
                raise ValueError("总轮数只能追加")
            s["config"] = config
            s["experiments"][-1]["total_rounds"] = config["total_rounds"]
            s["revisions"].append(dict(experiment=s["id"], at=time.time(), patch=patch))
            self.event(db, s, "configuration", {"keys": list(patch)})
        return self.snapshot(experiment)

    def archive(self, branch, archived=True):
        if type(archived) is not bool:
            raise ValueError("archived 必须是布尔值")
        with self.edit() as (s, db):
            if s["phase"] not in ("idle", "finished", "closed"):
                raise ValueError("归档前先结束或关闭实验")
            s["branches"][branch]["archived"] = archived
            self.event(
                db, s, "branch_archived", {"branch": branch, "archived": archived}
            )
        return self.snapshot()

    @staticmethod
    def own(s, branch, cid):
        if cid not in s["candidates"]:
            cid = branch + "--" + cid
        c = s["candidates"].get(cid)
        if not c or c["branch"] != branch:
            raise ValueError("候选必须属于本分支")
        return c

    @staticmethod
    def require_turn(s, branch):
        if s["status"] != "running" or s["branches"][branch]["status"] != "researching":
            raise ValueError("本分支当前没有进行中的研究迭代")
        iteration = s["iterations"][s["branches"][branch]["iteration"]]
        if iteration["status"] != "researching":
            raise ValueError("本批次已经提交")
        return iteration

    PLAN_FIELDS = ("assessment", "direction", "reason")

    @staticmethod
    def training_allowance(family, cap, requested, default):
        """Resolve the training budget an iteration may actually run.

        Rule families have no training slot at all; learners are capped so that
        a branch cannot convert wall clock into an unbounded training run.
        """
        if cap == 0:
            if requested:
                raise ValueError(
                    f"{family} 组合没有训练环节：不要传 training_games，并移除 "
                    "strategy.train，由配对评估来衡量候选"
                )
            return 0
        games = default if requested is None else requested
        if type(games) is not int or not 1 <= games <= 1000000:
            raise ValueError("训练场数不合法")
        if cap is not None and games > cap:
            raise ValueError(f"{family} 组合的训练上限是 {cap} 场，本次申请 {games} 场")
        return games

    @staticmethod
    def require_plan(iteration):
        if not iteration.get("plan"):
            raise ValueError(
                "创建或提交候选之前，先用 research_declare_plan 声明本次迭代的"
                "研究判断（assessment、direction、reason）"
            )
        return iteration["plan"]

    def declare_plan(self, experiment, branch, assessment, direction, reason):
        """Record the pre-implementation judgement that gates this iteration's drafts."""
        text = {}
        for key, value in (
            ("assessment", assessment),
            ("direction", direction),
            ("reason", reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"研究判断的 {key} 必须是非空描述")
            text[key] = value.strip()[:4000]
        with self.edit(experiment) as (s, db):
            iteration = self.require_turn(s, branch)
            now = time.time()
            plan = dict(text, declared_at=now, updated_at=now)
            if iteration.get("plan"):
                plan["declared_at"] = iteration["plan"].get("declared_at", now)
            iteration["plan"] = plan
            self.event(
                db,
                s,
                "plan_declared",
                {
                    "branch": branch,
                    "iteration": iteration["number"],
                    "direction": text["direction"][:200],
                },
            )
            number = iteration["number"]
        return {"iteration": number, "plan": plan}

    def fork(self, experiment, branch, name, hypothesis, parent=None):
        if not re.fullmatch("[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("候选名不合法")
        with self.edit(experiment) as (s, db):
            cid = branch + "--" + name
            existing = s["candidates"].get(cid)
            if existing and existing["branch"] == branch:
                # A dropped tool reply makes the agent retry a call that already
                # landed; retrying a create must hand back the same draft instead
                # of an error the agent cannot tell apart from a real conflict.
                return {
                    "candidate_id": cid,
                    "status": existing["status"],
                    "existing": True,
                    "parent": existing.get("parent"),
                    "note": (
                        "该草稿已存在于本分支，直接修改并提交它"
                        if existing["status"] == "draft"
                        else "这个候选 ID 已被 "
                        + str(existing["status"])
                        + " 版本使用；继续使用它，或换一个新名字"
                    ),
                }
            if existing:
                raise ValueError("同名候选已存在于其他分支")
            self.require_plan(self.require_turn(s, branch))
            b = s["branches"][branch]
            previous = self.own(s, branch, parent or b["representative"])
            source = (
                previous.get("spec", {}).get("package_path") or previous["workspace"]
            )
            dest = self.layout.strategies / b["combination"] / branch / "drafts" / name
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
                                raise ValueError("训练检查点已变化")
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
                requirements=list(previous["requirements"]),
                runtime_path=s["runtime_path"],
                experiment=s["id"],
                iteration=b.get("iteration"),
            )
        return {"candidate_id": cid, "status": "draft", "existing": False}

    def candidate(self, experiment, branch, cid, path=None, content=None):
        with self.edit(experiment) as (s, db):
            c = self.own(s, branch, cid)
            if content is not None:
                self.require_turn(s, branch)
                if c["status"] != "draft":
                    raise ValueError("只有草稿可以修改")
            base = Path(c.get("spec", {}).get("package_path") or c["workspace"])
            if path is None:
                return {k: v for k, v in c.items() if k not in ("workspace", "spec")}
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("必须使用相对候选路径")
            file = base / relative
            if not file.resolve().is_relative_to(base.resolve()):
                raise ValueError("候选路径越出候选目录")
            if content is not None:
                if not isinstance(content, str) or len(content.encode()) > 2_000_000:
                    raise ValueError("文本文件过大")
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(content)
                return {"written": path}
            return {"path": path, "content": file.read_text()}

    def dependencies(self, experiment, branch, cid, requirements):
        if len(requirements) > 32 or any(
            not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+-]*", r
            )
            for r in requirements
        ):
            raise ValueError("必须使用精确的 name==version wheel 依赖")
        with self.edit(experiment) as (s, db):
            self.require_turn(s, branch)
            c = self.own(s, branch, cid)
            if c["status"] != "draft":
                raise ValueError("依赖已冻结")
            c["requirements"] = requirements
        return {"requirements": requirements}

    def check(self, experiment, branch, cid):
        """Same run check as submission, without freezing the draft."""
        s = self.read(experiment)
        draft = self.own(s, branch, cid)
        if draft["status"] != "draft":
            return {"candidate": draft["id"], "ok": True, "frozen": True}
        combination = s["branches"][branch]["combination"]
        return {
            "candidate": draft["id"],
            **self.preflight(
                Path(draft["workspace"]),
                draft["requirements"],
                combination,
                s["definitions"][combination],
                s["config"]["game"],
            ),
        }

    def preflight(self, source, requirements, combination, definition=None, game=None):
        """Run a draft the way its job will: build probe plus one training episode.

        Drafts that cannot import, build or survive a single episode are rejected
        at submission time instead of burning the iteration in preparation.
        """
        from mahjong_ai.execution.candidate_runtime import (
            CandidateProcess,
            train_package,
        )
        from mahjong_ai.execution.dependencies import prepare_environment
        from mahjong_ai.domain.engine import derive_seed
        from mahjong_ai.domain.types import RoundConfig
        from mahjong_ai.strategies.common import PolicySpec
        import numpy as np
        import sys
        import tempfile

        started = time.monotonic()
        deadline = started + 900
        source = Path(source)
        try:
            python = sys.executable
            if requirements:
                environment = (
                    self.layout.research
                    / "environments"
                    / digest([sorted(requirements), sys.version, np.__version__])
                )
                python, _ = prepare_environment(requirements, environment, deadline)
            root_state = self.read()
            runtime = root_state["runtime_path"]
            family = (definition or {}).get("family")
            cap = (root_state["config"].get("training_caps") or {}).get(family)
            probe = CandidateProcess(source, python, runtime=runtime)
            try:
                capabilities = probe.call(
                    {"op": "describe"}, min(60, max(1, deadline - time.monotonic()))
                )
                if capabilities.get("train") and cap == 0:
                    raise ValueError(
                        f"{family} 组合没有训练环节：请移除 strategy.train，"
                        "让配对评估来衡量这个草稿"
                    )
                probe.call(
                    {
                        "op": "build",
                        "combination": combination,
                        "definition": definition,
                    },
                    min(120, max(1, deadline - time.monotonic())),
                )
            finally:
                probe.close()
            if capabilities.get("train"):
                with tempfile.TemporaryDirectory(prefix="mahjong-preflight-") as tmp:
                    train_package(
                        source,
                        Path(tmp) / "model",
                        RoundConfig(**(game or {})),
                        lambda index: [PolicySpec() for _ in range(3)],
                        1,
                        derive_seed(source.name, "preflight"),
                        deadline,
                        python,
                        runtime=runtime,
                    )
        except Exception as exc:  # noqa: BLE001 - the message is the payload
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "seconds": round(time.monotonic() - started, 1),
            }
        return {
            "ok": True,
            "error": None,
            "seconds": round(time.monotonic() - started, 1),
        }

    def submit(self, experiment, branch, candidates, training_games=None):
        # Drafts must run before they freeze: a candidate that cannot import,
        # build or finish one training episode is rejected with the real error
        # so the agent fixes it inside the same turn.
        snapshot = self.read(experiment)
        previous = snapshot["iterations"].get(
            snapshot["branches"][branch].get("iteration")
        )
        if previous and previous["status"] != "researching":
            ids = [self.own(snapshot, branch, cid)["id"] for cid in candidates]
            if ids == previous["candidates"]:
                return {
                    "iteration": previous["number"],
                    "status": "submitted",
                    "instruction": "Batch accepted. End this reply now; evaluation is managed by the controller.",
                }
        combination = snapshot["branches"][branch]["combination"]
        turn = (
            snapshot["iterations"].get(snapshot["branches"][branch].get("iteration"))
            or {}
        )
        turn_config = turn.get("config") or snapshot["config"]
        family = (snapshot["definitions"].get(combination) or {}).get("family")
        # Reject a doomed budget before running the run checks, so a rejected
        # request costs the agent a message instead of a subprocess sweep.
        self.training_allowance(
            family,
            (turn_config.get("training_caps") or {}).get(family),
            training_games,
            turn_config["training_games"],
        )
        checks = {}
        for cid in [self.own(snapshot, branch, c)["id"] for c in candidates]:
            draft = snapshot["candidates"][cid]
            if draft["status"] != "draft":
                continue
            check = self.preflight(
                Path(draft["workspace"]),
                draft["requirements"],
                combination,
                snapshot["definitions"][combination],
                snapshot["config"]["game"],
            )
            if not check["ok"]:
                raise ValueError(
                    f"候选 {cid} 未通过运行自检，本次提交被拒绝："
                    f"{check['error']}。请在同一次回复里修好并重新提交。"
                )
            checks[cid] = check
        with self.edit(experiment) as (s, db):
            previous = s["iterations"].get(s["branches"][branch].get("iteration"))
            if previous and previous["status"] != "researching":
                ids = [self.own(s, branch, cid)["id"] for cid in candidates]
                if ids == previous["candidates"]:
                    return {
                        "iteration": previous["number"],
                        "status": "submitted",
                        "instruction": "批次已接收。立即结束本次回复，评估由控制器处理。",
                    }
            it = self.require_turn(s, branch)
            self.require_plan(it)
            if not 1 <= len(candidates) <= it["config"]["candidate_limit"] or len(
                set(candidates)
            ) != len(candidates):
                raise ValueError("请提交 1 到 4 个互不重复的草稿")
            rows = [self.own(s, branch, cid) for cid in candidates]
            if any(c["status"] != "draft" for c in rows):
                raise ValueError("只有草稿状态的候选可以提交")
            family = (s["definitions"].get(combination) or {}).get("family")
            cap = (it["config"].get("training_caps") or {}).get(family)
            games = self.training_allowance(
                family, cap, training_games, it["config"]["training_games"]
            )
            if type(games) is not int or not 0 <= games <= 1000000:
                raise ValueError("训练场数不合法")
            root = (
                self.layout.strategies
                / s["branches"][branch]["combination"]
                / branch
                / "submissions"
                / it["id"]
            )
            if root.exists():
                shutil.rmtree(root)
            try:
                for c in rows:
                    source = Path(c["workspace"])
                    if not (source / "strategy.py").is_file():
                        raise ValueError("缺少 strategy.py")
                    target = root / c["id"]
                    shutil.copytree(source, target)
                    if tree_hash(source) != tree_hash(target):
                        raise ValueError("提交过程中源码发生变化")
                for c in rows:
                    c.update(
                        status="submitted",
                        workspace=str(root / c["id"]),
                        preflight=checks.get(
                            c["id"], {"ok": None, "error": None, "seconds": None}
                        ),
                    )
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
            dest = self.layout.strategies / b["combination"] / b["id"] / "initial"
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
        experiment = s["experiments"][-1]
        if experiment["completed_rounds"] >= s["config"]["total_rounds"]:
            s.update(
                status="paused",
                phase="finished",
                pause_reason="research round target reached",
            )
            experiment.update(status="finished", completed_at=time.time())
            return
        selected = [s["branches"][bid] for bid in experiment["targets"]]
        if any(not b["representative"] for b in selected):
            s["phase"] = "initializing"
            return
        self.sync_resident_releases(s)
        number = len(s["cycles"]) + 1
        pool = (
            experiment["initial_opponents"]
            if not experiment["completed_rounds"]
            else self.default_opponents(s, s["config"]["opponent_limit"])
        )
        cycle = dict(
            id=number,
            experiment=s["id"],
            number=experiment["completed_rounds"] + 1,
            status="researching",
            config=deepcopy(s["config"]),
            targets={
                b["id"]: b["completed"]
                + s["config"]["branch_iterations"].get(
                    b["id"], s["config"]["iterations_per_round"]
                )
                for b in selected
            },
            starts={b["id"]: b["completed"] for b in selected},
            reference=list(pool),
            ranking=[],
            roster={},
        )
        for b in selected:
            b["status"] = "ready"
        s["cycles"].append(cycle)
        s["phase"] = "researching"
        self.event(
            db, s, "round_started", {"round": cycle["number"], "opponents": list(pool)}
        )

    def start_iteration(self, s, branch):
        b = s["branches"][branch]
        cycle = s["cycles"][-1]
        if (
            branch not in cycle["targets"]
            or b["completed"] >= cycle["targets"][branch]
            or b["status"] != "ready"
        ):
            raise ValueError("分支尚未就绪，无法开始本次迭代")
        cfg = deepcopy(s["config"])
        combination = s["definitions"][b["combination"]]
        cfg["training_games"] = combination.get("training_games", cfg["training_games"])
        number = b["completed"] + 1
        iid = f"{branch}--iteration-{number:05d}"
        s["iterations"][iid] = dict(
            id=iid,
            branch=branch,
            number=number,
            cycle=s["cycles"][-1]["id"],
            experiment=s["id"],
            status="researching",
            config=deepcopy(cfg),
            incumbent=b["representative"],
            candidates=[],
            spent=0.0,
            results={},
        )
        finished = s["iterations"].get(b.get("iteration"))
        if finished and b.get("thread_id"):
            finished.setdefault("thread_id", b["thread_id"])
        b.update(
            status="researching",
            iteration=iid,
            last_run=time.time(),
            turns=b["turns"] + 1,
            # Every iteration starts its own conversation: the handoff carries the
            # state forward, so a branch's context stops growing with its history.
            thread_id=None,
        )
        return self.queue(s, "agent", branch=branch, iteration=iid)

    def advance(self, s, db):
        if s["status"] != "running":
            return
        self.initialize_ready(s)
        if s["phase"] in ("initializing", "boundary"):
            if any(
                not s["branches"][bid]["representative"]
                for bid in s["experiments"][-1]["targets"]
            ):
                return
            self.begin_cycle(s, db)
        if s["phase"] == "researching":
            cycle = s["cycles"][-1]
            for b in s["branches"].values():
                if b["id"] not in cycle["targets"]:
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
                s["branches"][bid]["status"] == "waiting" for bid in cycle["targets"]
            ):
                plan = s.get("stop_plan") or {}
                if plan and not plan.get("competition", True):
                    s.update(
                        status="paused",
                        pause_reason="迭代阶段结束，按安排暂停（未跑大乱斗）",
                        stop_plan=None,
                    )
                    s["experiments"][-1]["status"] = "paused"
                    return
                cycle["status"] = "competing"
                s["phase"] = "competing"
                cycle["roster"] = {
                    b["id"]: dict(
                        candidate_id=b["representative"],
                        combination=b["combination"],
                        model=b["model"],
                        effort=b["model_config"]["effort"],
                    )
                    for b in self.eligible(s)
                }
                for cid in self.sync_resident_releases(s):
                    c = s["candidates"][cid]
                    cycle["roster"][cid] = dict(
                        candidate_id=cid,
                        entrant_id=c.get("entrant_id"),
                        combination=c.get("combination", "external"),
                        model="external",
                        effort="external",
                        external=True,
                        iterable=False,
                    )
                roster = [b["candidate_id"] for b in cycle["roster"].values()]
                for cid in s["bootstrap"]:
                    if len(roster) >= 4:
                        break
                    roster.append(cid)
                # An entirely unsuccessful first round still uses the three seed styles.
                if len(roster) < 4:
                    alias = "bootstrap--balanced--extra"
                    base = s["candidates"][s["bootstrap"][1]]
                    s["candidates"][alias] = {
                        **base,
                        "id": alias,
                        "spec": {**base["spec"], "version": alias},
                    }
                    roster.append(alias)
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
            previous = {r["branch"]: i + 1 for i, r in enumerate(s["last_ranking"])}
            for rank, row in enumerate(table, 1):
                row.update(
                    rank=rank, rank_change=previous.get(row["branch"], rank) - rank
                )
            cycle.update(status="completed", ranking=table, completed_at=time.time())
            s["last_ranking"] = table
            s["experiments"][-1]["completed_rounds"] += 1
            s["phase"] = "boundary"
            atomic_json(
                Path(s["root"])
                / "experiments"
                / s["id"]
                / f"round-{cycle['number']:04d}.json",
                cycle,
            )
            self.event(
                db, s, "round_completed", {"round": cycle["number"], "ranking": table}
            )
            # 停止安排：本轮结果落盘后再停，不在下一轮调度任何任务。
            plan = s.get("stop_plan") or {}
            if plan:
                reason = {
                    "iteration": "本次迭代结束后暂停",
                    "now": "已结束当前迭代并暂停",
                }.get(plan.get("scope"), "本轮结束后暂停")
                s.update(status="paused", pause_reason=reason, stop_plan=None)
                s["experiments"][-1]["status"] = "paused"
                return
            self.begin_cycle(s, db)

    def complete_iteration(self, s, db, it, error=None, cancelled=False):
        b = s["branches"][it["branch"]]
        if it["status"] == "completed":
            return
        success = bool(it["results"]) and not cancelled
        improved = False
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
            # 候选与基线共用同一份赛程：场口径配对差供页面对照选优是否落在
            # 噪声以内；逐局配对只作辅助读数。
            for cid in it["candidates"]:
                reference = it["results"].get(it["incumbent"])
                result = it["results"].get(cid)
                for metric, field in (("field", "paired"), ("hand", "hand_paired")):
                    paired = paired_readout(reference, result, metric)
                    if paired and result is not None:
                        result[field] = paired
            for cid in it["candidates"]:
                s["candidates"][cid]["status"] = (
                    "evaluated"
                    if cid in it["results"]
                    else s["candidates"][cid]["status"]
                )
            s["candidates"][b["representative"]]["status"] = "evaluated"
            improved = winner != it["incumbent"]
            b["representative"] = winner
            s["candidates"][winner]["status"] = "representative"
        b["completed"] += 1
        b["successful"] += int(success)
        b["improved"] += int(improved)
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
            improved=improved,
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
                "improved": improved,
            },
        )

    def accept(self, s, db, job, result):
        if job["status"] == "completed":
            return
        if job["experiment"] != s["id"]:
            raise ValueError("实验回执已过期")
        if result["status"] == "paused":
            job["status"] = "paused"
            return
        if not (job["kind"] == "match" and result["status"] != "completed"):
            self.store.receipt(
                db, s["id"], job["id"], result, int(job.get("attempt", 0))
            )
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
                        pause_reason="初始版本训练失败：" + job["combination"],
                    )
            elif not ready:
                self.complete_iteration(s, db, it, "no prepared candidates")
            else:
                it["status"] = "evaluating"
                s["branches"][it["branch"]]["status"] = "evaluating"
                cycle = s["cycles"][it["cycle"] - 1]
                key = read_json(Path(s["root"]) / "private/random.json")["key"]
                groups = reference_groups(
                    cycle["reference"],
                    digest([key, "opponents", it["id"]]),
                    cycle["config"]["evaluation_games"] // 4,
                )
                # 一次评估按四座轮换组切成若干分片任务并行跑；对手、种子与
                # 计分口径都由 block 决定，分片不改变任何一局。
                shards = max(
                    1, min(int(cycle["config"]["evaluation_shards"]), len(groups))
                )
                for cid in [it["incumbent"], *ready]:
                    blocks = []
                    for block, opponents in enumerate(groups):
                        seats = []
                        for seat in range(4):
                            players = list(opponents)
                            players.insert(seat, cid)
                            seats.append(dict(block=block, players=players, focal=seat))
                        blocks.append(seats)
                    per = -(-len(blocks) // shards)
                    for shard, start in enumerate(range(0, len(blocks), per)):
                        part = []
                        for seats in blocks[start : start + per]:
                            for seat in seats:
                                part.append(dict(index=len(part), **seat))
                        self.queue(
                            s,
                            "match",
                            scope="paired",
                            branch=it["branch"],
                            iteration=it["id"],
                            cycle=cycle["id"],
                            candidate=cid,
                            shard=shard,
                            shards=shards,
                            assignments=part,
                        )
        elif job["kind"] == "match":
            if result["status"] != "completed":
                job["status"] = "paused"
                s.update(
                    status="pausing",
                    pause_reason="评估任务失败：" + str(result.get("error")),
                )
                return
            job["summary"] = result["summary"]
            if job["scope"] == "paired":
                job["focal"] = result["focal"]
                jobs = [
                    j
                    for j in s["jobs"].values()
                    if j.get("iteration") == it["id"] and j["kind"] == "match"
                ]
                for cid in sorted({j["candidate"] for j in jobs}):
                    shards = [j for j in jobs if j["candidate"] == cid]
                    # 分支未到齐前不发布读数：半程均值既不是选优口径，也会让
                    # 卡片把进行中的评估显示成完成。
                    if any(j["status"] != "completed" for j in shards):
                        continue
                    merged = self.merge_focal([j["focal"] for j in shards])
                    it["results"][cid] = merged
                    s["candidates"][cid]["last_result"] = merged
                if all(j["status"] == "completed" for j in jobs):
                    self.complete_iteration(s, db, it)

    @staticmethod
    def merge_focal(parts):
        """把同一候选的评估分片合成一份读数；口径等于一次跑完整批。"""
        merged = {
            key: sum(part.get(key, 0) for part in parts)
            for key in (
                "appearances",
                "utility_sum",
                "hand_appearances",
                "hand_utility_sum",
                "hand_score_sum",
                "errors",
                "timeouts",
                "fallback_actions",
            )
        }
        merged["candidate_id"] = parts[-1]["candidate_id"]
        merged["mean_utility"] = (
            merged["utility_sum"] / merged["appearances"]
            if merged["appearances"]
            else None
        )
        merged["hand_mean_utility"] = (
            merged["hand_utility_sum"] / merged["hand_appearances"]
            if merged["hand_appearances"]
            else None
        )
        merged["hand_mean_score"] = (
            merged["hand_score_sum"] / merged["hand_appearances"]
            if merged["hand_appearances"]
            else None
        )
        # 分片按 block 升序拼接，块编号与均值一一对齐：候选与基线共用同一
        # 份赛程，配对比较靠这份编号对齐，不靠数组位置猜测。
        metrics = (
            ("", "group_means", "field_group_means"),
            ("hand_", "hand_group_means", "hand_group_means"),
            ("score_", "score_group_means", "score_group_means"),
        )
        if all(not (part.get("group_blocks") or []) for part in parts):
            # 旧回执没有块编号：按分片顺序拼接，只报每条分片的合计口径。
            merged["group_blocks"] = []
            for prefix, key, dest in metrics:
                means = [mean for part in parts for mean in part.get(key) or []]
                merged.update(metric_uncertainty(means, prefix=prefix))
                merged[dest] = means
        else:
            blocks = []
            aligned = {key: [] for _, key, _ in metrics}
            for part in parts:
                part_blocks = part.get("group_blocks") or []
                part_values = {key: part.get(key) or [] for key in aligned}
                for index, block in enumerate(part_blocks):
                    if block in blocks:
                        continue
                    blocks.append(block)
                    for key in aligned:
                        values = part_values[key]
                        aligned[key].append(
                            values[index] if index < len(values) else None
                        )
            order = sorted(range(len(blocks)), key=lambda i: blocks[i])
            merged["group_blocks"] = [blocks[i] for i in order]
            for prefix, key, dest in metrics:
                means = [aligned[key][i] for i in order if aligned[key][i] is not None]
                merged.update(metric_uncertainty(means, prefix=prefix))
                merged[dest] = means
        merged["group_means"] = merged["field_group_means"]
        return merged

    @staticmethod
    def competition_table(s, cycle):
        rows = {}
        groups, hand_groups, score_groups = {}, {}, {}
        for j in s["jobs"].values():
            if (
                j.get("scope") != "competition"
                or j["cycle"] != cycle["id"]
                or j["status"] != "completed"
            ):
                continue
            for r in j.get("summary", []):
                # A competition job is exactly one complete four-seat block.
                if r["appearances"] != 4:
                    raise ValueError("共同竞技回执必须是完整的四座轮换组")
                groups.setdefault(r["candidate_id"], []).append(r["utility_sum"] / 4)
                hand_mean = r.get("hand_mean_utility")
                score_mean = r.get("hand_mean_score")
                if hand_mean is not None:
                    hand_groups.setdefault(r["candidate_id"], []).append(hand_mean)
                    score_groups.setdefault(r["candidate_id"], []).append(score_mean)
                dst = rows.setdefault(
                    r["candidate_id"],
                    {
                        k: 0
                        for k in (
                            "appearances",
                            "utility_sum",
                            "hand_appearances",
                            "hand_utility_sum",
                            "hand_score_sum",
                            "errors",
                            "timeouts",
                            "fallback_actions",
                        )
                    },
                )
                for k in dst:
                    dst[k] += r.get(k, 0)
        result = []
        for bid, b in cycle["roster"].items():
            cid = b["candidate_id"]
            if cid in rows:
                r = rows[cid]
                hand_appearances = r["hand_appearances"]
                result.append(
                    dict(
                        **r,
                        **metric_uncertainty(groups[cid]),
                        **metric_uncertainty(
                            hand_groups.get(cid) or [], prefix="hand_"
                        ),
                        **metric_uncertainty(
                            score_groups.get(cid) or [], prefix="score_"
                        ),
                        candidate_id=cid,
                        branch=None if b.get("external") else bid,
                        entrant_id=b.get("entrant_id"),
                        external=bool(b.get("external")),
                        iterable=bool(b.get("iterable", not b.get("external"))),
                        combination=b["combination"],
                        model=b["model"],
                        effort=b["effort"],
                        mean_utility=r["utility_sum"] / r["appearances"],
                        hand_mean_utility=(
                            r["hand_utility_sum"] / hand_appearances
                            if hand_appearances
                            else None
                        ),
                        hand_mean_score=(
                            r["hand_score_sum"] / hand_appearances
                            if hand_appearances
                            else None
                        ),
                    )
                )
        return sorted(
            result,
            key=lambda r: (
                -r["mean_utility"],
                -r["appearances"],
                r.get("branch") or r.get("entrant_id") or r["candidate_id"],
            ),
        )

    @staticmethod
    def handoff(s, branch):
        """Compact state transfer that replaces the previous iteration's conversation."""
        b = s["branches"][branch]
        own = sorted(
            (
                i
                for i in s["iterations"].values()
                if i["branch"] == branch and i["id"] != b.get("iteration")
            ),
            key=lambda i: i["number"],
        )
        recent = []
        for it in own[-3:]:
            row = {
                "iteration": it["number"],
                "round": it["cycle"],
                "status": it["status"],
                "improved": it.get("improved"),
                "candidates": list(it.get("candidates") or []),
                "spent_minutes": round((it.get("spent") or 0) / 60),
            }
            if it.get("error"):
                row["error"] = str(it["error"])[:200]
            if it.get("plan"):
                row["direction"] = it["plan"]["direction"][:200]
            if it.get("results"):
                row["paired"] = {
                    cid: {
                        "mean_utility": round(v["mean_utility"], 3),
                        "hand_mean_utility": (
                            None
                            if v.get("hand_mean_utility") is None
                            else round(v["hand_mean_utility"], 3)
                        ),
                        "hand_mean_score": (
                            None
                            if v.get("hand_mean_score") is None
                            else round(v["hand_mean_score"], 3)
                        ),
                        "standard_error": (
                            None
                            if v.get("standard_error") is None
                            else round(v["standard_error"], 3)
                        ),
                        "appearances": v["appearances"],
                        "hand_appearances": v.get("hand_appearances"),
                    }
                    for cid, v in it["results"].items()
                }
            recent.append(row)
        incumbent = s["candidates"].get(b["representative"]) or {}
        ranking = [
            dict(
                rank=index + 1,
                branch=row["branch"],
                mean_utility=round(row["mean_utility"], 3),
                hand_mean_utility=(
                    None
                    if row.get("hand_mean_utility") is None
                    else round(row["hand_mean_utility"], 3)
                ),
            )
            for index, row in enumerate(s["last_ranking"][:8])
        ]
        return {
            "branch": branch,
            "combination": b["combination"],
            "iterations_completed": b["completed"],
            "improved_iterations": b["improved"],
            "failed_iterations": b["failed"],
            "note": "本轮是新的研究会话，上一轮的对话已归档；"
            "下面是结构化交接，细节用 research_status / research_history 查。",
            "incumbent": {
                "candidate_id": b["representative"],
                "hypothesis": (incumbent.get("hypothesis") or "")[:1200],
            },
            "recent_iterations": recent,
            "last_ranking": ranking,
        }

    @staticmethod
    def iteration_feedback(s, iteration, jobs=None):
        """Own-branch observed execution cost, excluding queueing and evaluation."""
        if jobs is None:
            jobs = [
                j for j in s["jobs"].values() if j.get("iteration") == iteration["id"]
            ]

        def elapsed(kind):
            measured = [
                j
                for j in jobs
                if j["kind"] == kind
                and ("wall_seconds" in j.get("usage", {}) or j.get("spent", 0) > 0)
            ]
            return (
                sum(
                    j.get("spent", 0) + j.get("usage", {}).get("wall_seconds", 0)
                    for j in measured
                )
                if measured
                else None
            )

        config = s["cycles"][iteration["cycle"] - 1]["config"]
        training = []
        for cid in iteration["candidates"]:
            candidate = s["candidates"][cid]
            record = candidate.get("training")
            training.append(
                dict(
                    candidate_id=cid,
                    status=record["status"] if record else candidate["status"],
                    completed_games=len(record["episodes"]) if record else None,
                    updates=record.get("reported_metrics", {}).get("updates")
                    if record
                    else None,
                    interrupted=record.get("interrupted") if record else None,
                    error=candidate.get("error"),
                )
            )
        return dict(
            iteration=iteration["id"],
            experiment=iteration["experiment"],
            training_games=iteration.get("training_games"),
            execution_limit_seconds=config["iteration_seconds"],
            execution_seconds=iteration["spent"],
            agent_seconds=elapsed("agent"),
            preparation_seconds=elapsed("prepare"),
            training=training,
        )

    def snapshot(self, experiment=None, branch=None, after=0):
        if self.store.read() is None:
            self.register()
        s = self.read(experiment)
        if branch and branch not in s["branches"]:
            raise ValueError("未知分支")
        cycle = (
            deepcopy(s["cycles"][-1])
            if s["cycles"] and s["cycles"][-1]["experiment"] == s["id"]
            else None
        )
        if cycle:
            cycle.pop("config")
            if s["phase"] == "competing":
                cycle["ranking"] = self.competition_table(s, s["cycles"][-1])
        latest_plans = {}
        for it in s["iterations"].values():
            if not it.get("plan"):
                continue
            row = latest_plans.get(it["branch"])
            if row is None or it["number"] > row["iteration"]:
                latest_plans[it["branch"]] = {
                    "iteration": it["number"],
                    "cycle": it["cycle"],
                    "status": it["status"],
                    "improved": it.get("improved"),
                    **it["plan"],
                }
        branches = [
            {
                **{
                    k: v for k, v in b.items() if k not in ("thread_id", "model_config")
                },
                "effort": b["model_config"]["effort"],
                "plan": latest_plans.get(b["id"]),
            }
            for b in s["branches"].values()
        ]
        candidates = [
            {
                **{
                    k: v
                    for k, v in c.items()
                    if k not in ("workspace", "spec", "training", "runtime_path")
                },
                "available": bool(c.get("spec")),
            }
            for c in s["candidates"].values()
            if not branch or c["branch"] == branch
        ]
        iteration_jobs = {}
        for job in s["jobs"].values():
            if job.get("iteration"):
                iteration_jobs.setdefault(job["iteration"], []).append(job)
        iterations = [
            {
                **{k: v for k, v in it.items() if k != "config"},
                "execution_feedback": self.iteration_feedback(
                    s, it, iteration_jobs.get(it["id"], [])
                ),
            }
            for it in s["iterations"].values()
            if not branch or it["branch"] == branch
        ]
        ranking = (
            cycle["ranking"]
            if cycle and s["phase"] == "competing"
            else s["last_ranking"]
        )
        jobs = []
        for j in s["jobs"].values():
            if (
                j["experiment"] != s["id"]
                or j["status"] not in ("running", "admitted", "queued")
                or (branch and j.get("branch") != branch)
            ):
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
        current = s["experiments"][-1] if s["experiments"] else None
        public_experiment = (
            (
                {
                    k: v
                    for k, v in current.items()
                    if k not in ("runtime_path", "config", "runtime")
                }
                | {
                    "status": s["status"],
                    "phase": s["phase"],
                    "pause_reason": s["pause_reason"],
                    "stop_plan": s.get("stop_plan"),
                }
            )
            if current
            else None
        )
        opponents = self.default_opponents(s, s["config"]["opponent_limit"])
        from mahjong_ai.research.progress import branch_progress

        return dict(
            experiment=public_experiment,
            config=deepcopy(s["config"]),
            combinations=list(s["definitions"].values()),
            branches=branches,
            candidates=candidates,
            iterations=iterations,
            cycle=cycle,
            ranking=ranking,
            jobs=jobs,
            **({"branch_progress": branch_progress(s)} if branch is None else {}),
            default_opponents=opponents,
            bootstrap=list(s["bootstrap"]),
            history=[
                dict(
                    cycle=c["id"],
                    experiment=c["experiment"],
                    round=c["number"],
                    ranking=c["ranking"],
                )
                for c in s["cycles"]
                if c["status"] == "completed"
            ],
            experiments=[
                {
                    k: v
                    for k, v in e.items()
                    if k not in ("config", "runtime_path", "runtime")
                }
                for e in s["experiments"]
            ],
            model_history=s["model_history"],
            events=[
                e
                for e in self.store.events(s["id"], after)
                if not branch or e["data"].get("branch") in (None, branch)
            ],
        )

    def activity(self, experiment):
        s = self.read(experiment)
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
