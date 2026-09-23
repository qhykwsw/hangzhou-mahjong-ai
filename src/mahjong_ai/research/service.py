"""Persistent strategy registry and continuous experiments.

Branches iterate independently; every iteration is evaluated against the
evaluation opponents frozen when it starts (the top of the latest common
brawl). A common brawl starts after enough promotions and does not wait for
branches; its ranking only affects iterations that start afterwards.
"""

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
from mahjong_ai.research.conclusions import ledger, validate_assessment
from mahjong_ai.research.selection import (
    SELECTION_RULE,
    paired_interval,
    promotion_decision,
)
from mahjong_ai.research.config import (
    configuration,
    merge,
    validate,
    IMMEDIATE,
    LIVE,
)
from mahjong_ai.evaluation.schedule import (
    schedule,
    reference_groups,
    metric_uncertainty,
)


COMPETITION_BLOCKS_PER_JOB = 16


def evaluation_shard_count(config, group_count, candidate_count):
    """Choose deterministic shard count without leaving match workers idle."""
    if group_count < 1 or candidate_count < 1:
        raise ValueError("evaluation groups and candidates must be positive")
    configured = max(1, min(int(config["evaluation_shards"]), group_count))
    target = min(
        group_count,
        -(-int(config["match_workers"]) // candidate_count),
    )
    return max(configured, target)


def _competition_registry(research):
    """Return the shared registry, when this checkout provides one."""
    try:
        from mahjong_ai.competition.registry import CompetitionRegistry

        root = getattr(research.layout, "workspace", None) or research.project.parent
        return CompetitionRegistry(root)
    except (ImportError, AttributeError):
        return None


# 主口径是场桌面分（平台第一排序键）；名次效用与逐局读数只作对照。
# 旧实验的回执没有场分数块均值，回退到它自己冻结时用的名次效用口径。
PAIRED_KEYS = {
    "score": ("field_score_group_means", "field_group_means"),
    "field": ("field_group_means", None),
    "hand": ("hand_group_means", None),
    "hand_score": ("score_group_means", None),
}


def _primary_mean(row):
    """主口径均值：有场均分就用它，旧实验回退名次效用。"""
    if row is None:
        return None
    return row["mean_score"] if row.get("mean_score") is not None else row.get("mean_utility")


def paired_readout(reference, candidate, metric="score", confidence=0.95):
    """同一副牌、同一对手下的配对数：逐块差值的均值与聚类标准误。

    主口径是场桌面分（平台按场结算并按总分排序）；名次与逐局读数只作辅助。
    """
    if not reference or not candidate:
        return None
    primary, fallback = PAIRED_KEYS[metric]
    key = primary
    if fallback and not (reference.get(primary) and candidate.get(primary)):
        key = fallback
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
    delta = sum(diffs) / count
    standard_error = sqrt(variance(diffs) / count) if count >= 2 else None
    return dict(
        blocks=count,
        delta=delta,
        standard_error=standard_error,
        **paired_interval(delta, standard_error, count, confidence),
    )


def distinct_conditions(records):
    """Label each distinct evaluation condition within an epoch separately.

    Iterations of one epoch can stop at different evaluation stages, so their
    conditions differ in games per side; agent runtimes frozen before this
    change group conditions by the epoch number alone and reject a mismatch.
    """
    from mahjong_ai.research.agent_context import CONDITION_FIELDS

    seen = {}
    for record in records:
        evidence = record.get("evidence") or {}
        if "cycle" not in evidence:
            continue
        context = {k: evidence.get(k) for k in CONDITION_FIELDS if k in evidence}
        label = str(evidence["cycle"])
        if seen.setdefault(label, context) != context:
            label = f"{evidence['cycle']}/{evidence.get('evaluation_games')}"
            seen.setdefault(label, context)
            evidence["cycle"] = label
    return records


def strip_groups(result):
    if result:
        for key in [k for k in result if k.endswith("group_means") or k == "group_blocks"]:
            result.pop(key)


MODES = ("iterating", "competing", "archived")


def branch_mode(b):
    """Iterating, competing only, or archived (see CONTEXT.md)."""
    return b.get("mode") or ("archived" if b["archived"] else "iterating")


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
                self.apply_modes(state, db, config)
                if not state["bootstrap"]:
                    self.ensure_bootstrap(state, state["runtime_path"])
        return self.read()

    def ensure_bootstrap(self, state, runtime):
        """New experiments get fresh startup opponents when their runtime changes."""
        from mahjong_ai.strategies.common import PolicySpec

        if state["bootstrap"] and all(
            state["candidates"][cid]["runtime_path"] == runtime
            for cid in state["bootstrap"]
        ):
            return
        generation = Path(runtime).name if state["bootstrap"] else None
        bootstrap = []
        for style in ("conservative", "balanced", "adventurous"):
            key = "heuristic-" + style
            cid = "bootstrap--" + style + ("--" + generation if generation else "")
            bootstrap.append(cid)
            if cid in state["candidates"]:
                continue
            # Keep new packages outside the old package tree so its hash and
            # recorded source path stay unchanged, including across restarts.
            folder = self.layout.strategies / key
            folder = (
                folder / "bootstrap-versions" / generation
                if generation
                else folder / "bootstrap"
            )
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "strategy.py").write_text(baseline_source(key))
            spec = PolicySpec(
                family="code",
                version=cid,
                package_path=str(folder),
                package_hash=tree_hash(folder),
                runtime_path=runtime,
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
                runtime_path=runtime,
            )
        state["bootstrap"] = bootstrap

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

    def register_model(self, cid, source, *, branch=None, combination="model", note=""):
        """Register an exported Actor as a playable version keyed by its weights.

        The artifact is copied under research/models/<weights SHA-256>, the
        name every host's inference service uses to find it.
        """
        with self.edit() as (s, db):
            self.model_candidate(s, db, cid, source, branch=branch,
                                 combination=combination, note=note)
        return cid

    def model_candidate(self, s, db, cid, source, *, branch=None, combination="model",
                        note="", status=None):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", cid):
            raise ValueError("模型版本 ID 不合法")
        source = Path(source)
        meta = json.loads((source / "meta.json").read_text())
        import hashlib

        weights = hashlib.sha256((source / "actor.pt").read_bytes()).hexdigest()
        if weights != meta.get("weights_sha256"):
            raise ValueError("模型权重与 meta.json 校验不一致")
        target = self.layout.research / "models" / weights
        if not target.exists():
            temporary = target.with_name(weights + "." + secrets.token_hex(4))
            temporary.mkdir(parents=True)
            for name in ("actor.pt", "meta.json"):
                shutil.copy2(source / name, temporary / name)
            temporary.rename(target)
        spec = dict(family="model", version=cid, parameters={"weights": weights})
        existing = s["candidates"].get(cid)
        if existing and existing.get("spec") != spec:
            raise ValueError("同名版本已存在且内容不同")
        s["candidates"][cid] = dict(
            (existing or {}),
            id=cid,
            branch=branch,
            combination=combination,
            status=status or ("resident" if branch is None else "ready"),
            workspace=str(target),
            spec=spec,
            parent=(existing or {}).get("parent"),
            hypothesis=note or (existing or {}).get("hypothesis") or "exported Actor",
            requirements=[],
            runtime_path=s["runtime_path"],
            model=dict(config=meta.get("config"), weights=weights, source=str(source)),
        )
        self.event(db, s, "model_registered", {"candidate": cid, "weights": weights})
        return s["candidates"][cid]

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
                    # Takes effect from the branch's next iteration.
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
                mode="iterating",
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
            if targets is not None:
                # Selected branches iterate; the other active ones only compete.
                if (
                    not targets
                    or len(targets) != len(set(targets))
                    or any(
                        bid not in s["branches"]
                        or branch_mode(s["branches"][bid]) == "archived"
                        for bid in targets
                    )
                ):
                    raise ValueError("请选择互不重复、已注册且未归档的迭代分支")
                for bid, b in s["branches"].items():
                    if branch_mode(b) != "archived":
                        b.update(mode="iterating" if bid in targets else "competing")
            selected = [
                bid for bid, b in s["branches"].items() if branch_mode(b) == "iterating"
            ]
            if not selected:
                raise ValueError("没有处于迭代中的分支")
            self.ensure_bootstrap(s, runtime)
            # Without explicit opponents the experiment opens with a common
            # brawl; its ranking supplies the first evaluation opponents.
            pool = list(opponents or [])
            if len(pool) != len(set(pool)):
                raise ValueError("对手版本不能重复")
            for cid in pool:
                c = s["candidates"].get(cid)
                if not c or not c.get("spec"):
                    raise ValueError("对手必须是准备就绪的已注册版本")
            identity = secrets.token_hex(8)
            experiment = dict(
                id=identity,
                created_at=time.time(),
                status="paused",
                targets=selected,
                initial_opponents=pool,
                completed_brawls=0,
                selection_rule=SELECTION_RULE,
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
                promotions_pending=0,
                promotions_since=None,
                runtime=fingerprint,
                runtime_path=runtime,
                candidate_runtime_path=candidate_runtime_path,
            )
            for b in s["branches"].values():
                b["status"] = "ready" if b["representative"] else "uninitialized"
            for bid in selected:
                self.initialize_branch(s, bid)
            root = Path(s["root"]) / "experiments" / identity
            atomic_json(root / "experiment.json", experiment)
            self.event(
                db, s, "experiment_created", {"targets": selected, "opponents": pool}
            )
        return identity

    def initialize_branch(self, s, bid):
        """Give an iterating branch without a version its initial version."""
        b = s["branches"][bid]
        if b["representative"]:
            b["status"] = "ready"
            return
        if b["status"] == "initializing":
            return
        b["status"] = "initializing"
        key = b["combination"]
        combination = s["definitions"][key]
        if combination["family"] == "recipe":
            # Version 0 is the baseline recipe trained once by the recipe
            # service; it does not hold up the other branches.
            self.queue(s, "trial", branch=bid, baseline=True,
                       line=combination["parameters"]["line"])
            return
        if key in s["baselines"] and s["candidates"][s["baselines"][key]]["status"] != "failed":
            return
        cid = "baseline--" + key
        package = self.layout.strategies / key / "baseline/source"
        package.mkdir(parents=True, exist_ok=True)
        (package / "strategy.py").write_text(
            baseline_source(key, combination["parameters"], combination["family"])
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
            runtime_path=s["runtime_path"],
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
                "training_games", s["config"]["training_games"]
            ),
        )

    def apply(self, experiment=None):
        """Apply research.toml to the running experiment (ADR-0014).

        Branches, their modes, research models, fixed entrants and host
        placement change live. Evaluation conditions and scoring stay frozen:
        a file that changes them is not applied and the reason is kept.
        """
        if self.store.read() is None:
            self.register()
        config = configuration(self.project)
        with self.edit(experiment) as (s, db):
            frozen = sorted(
                k for k in set(config) | set(s["config"])
                if k not in LIVE and config.get(k) != s["config"].get(k)
            )
            known = {c["id"]: c for c in s["config"]["combinations"]}
            for c in config["combinations"]:
                old = known.get(c["id"])
                if old and any(
                    c.get(k) != old.get(k)
                    for k in ("family", "parameters", "evaluation_games", "training_games")
                ):
                    frozen.append("combinations." + c["id"])
            if frozen and s["id"] and s["phase"] not in ("finished", "closed"):
                error = "以下设置属于评估条件，运行中不能修改：" + ", ".join(frozen)
                if s.get("config_error") != error:
                    s["config_error"] = error
                    self.event(db, s, "configuration_rejected", {"keys": frozen})
                return False
            for k in LIVE:
                s["config"][k] = deepcopy(config[k])
            for combination in config["combinations"]:
                self.add_combination(s, combination, config["models"])
            self.apply_modes(s, db, config)
            s["config_error"] = None
            self.event(db, s, "configuration", {"applied": True})
        return True

    def apply_modes(self, s, db, config):
        configured = {
            f"{c['id']}--{suffix}"
            for c in config["combinations"]
            for suffix in c.get("branch_ids", [f"{n + 1:02d}" for n in range(c["branches"])])
        }
        active = bool(s["id"]) and s["phase"] not in ("idle", "finished", "closed")
        for bid, b in s["branches"].items():
            new = config["modes"].get(bid, "iterating") if bid in configured else "archived"
            old = branch_mode(b)
            if new != old or "mode" not in b:
                b.update(mode=new, archived=new == "archived")
                self.event(db, s, "branch_mode", {"branch": bid, "mode": new, "previous": old})
            # A branch that joins a running experiment starts from its initial version.
            if active and new == "iterating" and b["status"] == "uninitialized":
                self.initialize_branch(s, bid)

    @staticmethod
    def online(s, host):
        return host is None or s.get("hosts", {}).get(host, {}).get("online", True)

    @staticmethod
    def evaluation_host(s, b):
        return s["definitions"][b["combination"]].get("evaluation_host")

    def local_busy(self, s):
        """Iterations whose evaluation runs on this machine are still in flight."""
        return any(
            it["experiment"] == s["id"]
            and it["status"] != "completed"
            and not it["config"].get("evaluation_host")
            for it in s["iterations"].values()
        )

    @staticmethod
    def local_brawl(s):
        return any(
            j.get("scope") == "competition" and not j.get("host")
            and j["status"] != "completed" and j["experiment"] == s["id"]
            for j in s["jobs"].values()
        )

    def rehome(self, s, db, host):
        """A host went offline: bring back work that must not wait for it.

        Overflowed rule evaluations return to this machine; brawl chunks are
        split to local size and wait for the brawl drain. Work that belongs
        to the host (a recipe branch's evaluation) waits for it.
        """
        moved = []
        for j in list(s["jobs"].values()):
            if j.get("host") != host or j["kind"] != "match" or j["status"] == "completed":
                continue
            if j.get("overflow"):
                j.update(host=None, overflow=False, status="queued")
                j.pop("stop_requested", None)
                moved.append(j["id"])
            elif j.get("scope") == "competition":
                j.update(status="completed", cancelled=True, rehomed=True)
                moved.append(j["id"])
                chunk = 4 * COMPETITION_BLOCKS_PER_JOB
                for block in range(0, len(j["assignments"]), chunk):
                    self.queue(
                        s, "match", scope="competition", cycle=j["cycle"], host=None,
                        assignments=j["assignments"][block : block + chunk],
                    )
        if moved:
            self.event(db, s, "host_offline", {"host": host, "moved": moved})
        return moved

    def count_entrants(self, s, cycle):
        """A version newly entering the brawl counts as one pending promotion."""
        previous = next(
            (c for c in reversed(s["cycles"][:-1]) if c.get("roster")), None
        )
        if previous is None or cycle["status"] != "researching":
            return
        keys = {
            bid for bid, b in s["branches"].items()
            if branch_mode(b) != "archived" and b["representative"]
        } | set(s["config"].get("anchors", [])) | set(s["config"].get("guests", []))
        counted = set(cycle.setdefault("counted_entrants", []))
        for key in sorted(keys - set(previous["roster"]) - counted):
            cycle["counted_entrants"].append(key)
            s["promotions_pending"] = s.get("promotions_pending", 0) + 1
            s["promotions_since"] = s.get("promotions_since") or time.time()

    @staticmethod
    def eligible(s):
        return [
            b
            for b in s["branches"].values()
            if not b["archived"] and b["successful"] > 0 and b["representative"]
        ]

    def default_opponents(self, s, limit):
        """The top of the latest common brawl; anchors only measure drift."""
        anchors = set(s["config"].get("anchors", []))
        ranked = [
            r["candidate_id"]
            for r in s["last_ranking"]
            if r["candidate_id"] not in anchors
        ]
        return ranked[:limit]

    @staticmethod
    def queue(s, kind, **fields):
        # A persistent sequence, never the job count: a removed job's number
        # (and its directory, with a stale task.json) must never be reused.
        number = s.get("job_sequence")
        if number is None:
            number = 1 + max((int(k.rsplit("-", 1)[1]) for k in s["jobs"]), default=-1)
        s["job_sequence"] = number + 1
        jid = f"{kind}-{number:07d}"
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
                discarded = self.discard_iterations(s, db)
                self.arrange_stop(s, {"scope": "now"})
                self.event(db, s, "stop_now", {"discarded": discarded})
            elif action == "clear_stop":
                s["stop_plan"] = None
            elif action == "resume":
                if s["status"] == "pausing":
                    raise ValueError("请等待暂停完成")
                if s["phase"] == "closed":
                    raise ValueError("已关闭的实验不能继续")
                s.update(status="running", pause_reason="")
                for j in s["jobs"].values():
                    if j["experiment"] == s["id"] and j["status"] == "paused":
                        j["status"] = "queued"
                selected_combinations = {
                    b["combination"] for b in s["branches"].values()
                    if branch_mode(b) == "iterating"
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
        """Stop after the running iterations finish, or now (discarding them).

        No new iteration starts once a stop is arranged; a running common
        brawl still completes so its ranking is not lost.
        """
        scope = options.get("scope", "iteration")
        if scope not in ("iteration", "now"):
            raise ValueError("停止范围必须是本次迭代后或立即结束")
        s["stop_plan"] = {"scope": scope}

    @staticmethod
    def current_cycle(s):
        cycles = s["cycles"]
        if cycles and cycles[-1]["experiment"] == s["id"]:
            return cycles[-1]
        return None

    def discard_iterations(self, s, db):
        """结束当前迭代并删除未完成的迭代记录，让分支回到就绪状态。"""
        discarded = []
        for branch in s["branches"].values():
            iteration = s["iterations"].get(branch.get("iteration"))
            if (
                iteration is None
                or iteration["experiment"] != s["id"]
                or iteration["status"] == "completed"
            ):
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
            s["config"] = config
            s["revisions"].append(dict(experiment=s["id"], at=time.time(), patch=patch))
            self.event(db, s, "configuration", {"keys": list(patch)})
        return self.snapshot(experiment)

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
            pending = [r["record_id"] for r in ledger(s, branch) if not r["assessment"]]
            if pending:
                raise ValueError(
                    "先用 research_record_conclusion 整理累计研究结论，再声明下一步计划："
                    + ", ".join(pending)
                )
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

    def record_conclusion(
        self,
        experiment,
        branch,
        record_id,
        status,
        conclusion,
        conditions,
        failure_kind="none",
        failure_reason="",
    ):
        """Review own completed evidence; preserve revisions and dropped-reply idempotency."""
        with self.edit(experiment) as (s, db):
            current = self.require_turn(s, branch)
            record = next(
                (r for r in ledger(s, branch) if r["record_id"] == record_id), None
            )
            if record is None:
                raise ValueError("只能总结本分支已完成尝试的 record_id")
            value = validate_assessment(
                record, status, conclusion, conditions, failure_kind, failure_reason
            )
            iteration = s["iterations"][record["iteration_id"]]
            history = iteration.setdefault("conclusions", {}).setdefault(record_id, [])
            if not history or any(history[-1][k] != v for k, v in value.items()):
                history.append(
                    {**value, "recorded_in": current["id"], "at": time.time()}
                )
                self.event(
                    db,
                    s,
                    "conclusion_recorded",
                    {
                        "branch": branch,
                        "record_id": record_id,
                        "status": status,
                    },
                )
            return {
                "record_id": record_id,
                "assessment": deepcopy(history[-1]),
                "revision_count": len(history),
            }

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
        result = self.preflight(
            Path(draft["workspace"]),
            draft["requirements"],
            combination,
            s["definitions"][combination],
            s["config"]["game"],
        )
        self.record_check(experiment, branch, draft["id"], result)
        return {"candidate": draft["id"], **result}

    def record_check(self, experiment, branch, cid, result):
        with self.edit(experiment) as (s, db):
            candidate = self.own(s, branch, cid)
            checks = candidate.setdefault("checks", [])
            value = {"ok": result["ok"], "error": result.get("error")}
            if not checks or any(checks[-1][k] != v for k, v in value.items()):
                checks.append({**value, "at": time.time()})

    def preflight(self, source, requirements, combination, definition=None, game=None):
        """Run a draft the way its job will: build probe plus one training episode.

        Drafts that cannot import, build or survive a single episode are rejected
        at submission time instead of burning the iteration in preparation.
        """
        from mahjong_ai.execution.candidate_runtime import (
            ActionUnavailable,
            CandidateProcess,
            RemotePolicy,
            train_package,
        )
        from mahjong_ai.execution.dependencies import prepare_environment
        from mahjong_ai.domain.engine import MahjongEngine, derive_seed
        from mahjong_ai.domain.types import RoundConfig
        from mahjong_ai.strategies.common import PolicySpec, RulePolicy
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
            if not capabilities.get("train"):
                timeouts = root_state["config"]["timeouts"]
                policy = RemotePolicy(
                    source,
                    tree_hash(source),
                    timeouts["discard"],
                    python,
                    response_seconds=timeouts["response"],
                    runtime=runtime,
                )
                smoke_deadline = min(deadline, time.monotonic() + 30)

                class StrictProbe:
                    def check(self):
                        if any(
                            policy.stats[key]
                            for key in ("errors", "timeouts", "fallback_actions")
                        ):
                            raise ValueError(
                                (policy.last_error or {}).get("message")
                                or "候选动作超时或非法"
                            )
                        if time.monotonic() >= smoke_deadline:
                            raise TimeoutError("候选对局自检超过 30 秒")

                    def reset(self, seed):
                        policy.reset(seed)
                        self.check()

                    def act(self, observation):
                        self.check()
                        try:
                            action = policy.act(observation)
                        except ActionUnavailable:
                            self.check()
                            raise
                        self.check()
                        return action

                    def finish_round(self, result, seat):
                        policy.finish_round(result, seat)
                        self.check()

                try:
                    seed = derive_seed(source.name, "preflight-actions")
                    smoke_config = RoundConfig(
                        **{
                            **(game or {}),
                            "rounds": 1,
                            "games_per_round": 1,
                            "allow_tiebreak": False,
                        }
                    )
                    result = MahjongEngine(record_events=False).run_round(
                        [StrictProbe(), RulePolicy(), RulePolicy(), RulePolicy()],
                        smoke_config,
                        seed - seed % 4,
                    )
                    if not result.completed:
                        raise ValueError("候选对局自检未完成")
                finally:
                    policy.close()
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

    def submission_candidates(self, state, branch, candidates, limit):
        if not 1 <= len(candidates) <= limit:
            raise ValueError(
                f"本次最多提交 {limit} 个候选（candidate_limit={limit}），"
                f"实际收到 {len(candidates)} 个；请提交 1 到 {limit} 个互不重复的草稿"
            )
        rows = [self.own(state, branch, cid) for cid in candidates]
        if len({c["id"] for c in rows}) != len(rows):
            raise ValueError("请提交互不重复的草稿；完整 ID 与短名可能指向同一候选")
        if any(c["status"] != "draft" for c in rows):
            raise ValueError("只有草稿状态的候选可以提交")
        if any(
            c.get("iteration") != state["branches"][branch]["iteration"] for c in rows
        ):
            raise ValueError(
                "只能提交本次迭代派生的草稿；历史草稿请作为 parent，"
                "用 research_create_candidate 派生新候选后再提交"
            )
        return rows

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
        self.require_plan(self.require_turn(snapshot, branch))
        drafts = self.submission_candidates(
            snapshot, branch, candidates, turn_config["candidate_limit"]
        )
        checks = {}
        for draft in drafts:
            cid = draft["id"]
            check = self.preflight(
                Path(draft["workspace"]),
                draft["requirements"],
                combination,
                snapshot["definitions"][combination],
                snapshot["config"]["game"],
            )
            self.record_check(experiment, branch, cid, check)
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
            rows = self.submission_candidates(
                s, branch, candidates, it["config"]["candidate_limit"]
            )
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
            # Recipe branches get version 0 from the recipe service instead.
            if b["status"] != "initializing" or b["combination"] not in s["baselines"]:
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

    def open_epoch(self, s, db, reference):
        """Start the stretch between two common brawls.

        Iterations started in it are evaluated against ``reference``; a
        stretch without reference only waits for its opening brawl.
        """
        experiment = s["experiments"][-1]
        cycle = dict(
            id=len(s["cycles"]) + 1,
            experiment=s["id"],
            number=experiment["completed_brawls"] + 1,
            status="researching",
            config=deepcopy(s["config"]),
            reference=list(reference),
            ranking=[],
            roster={},
            started_at=time.time(),
        )
        s["cycles"].append(cycle)
        s["phase"] = "researching"
        self.event(
            db, s, "epoch_started", {"epoch": cycle["number"], "opponents": reference}
        )
        return cycle

    def brawl_due(self, s, cycle, now=None):
        if cycle["status"] != "researching" or not cycle["reference"]:
            return None
        pending = s.get("promotions_pending", 0)
        if pending >= s["config"]["brawl_promotions"]:
            return "promotions"
        waited = (now or time.time()) - cycle["started_at"]
        if pending and waited >= s["config"]["brawl_fallback_seconds"]:
            return "fallback"
        return None

    def start_brawl(self, s, db, cycle, reason, host=None):
        """Freeze every active branch's representative plus anchors and play."""
        cycle["status"] = "competing"
        cycle["brawl_host"] = host
        cycle["brawl_reason"] = reason
        cycle["brawl_started_at"] = time.time()
        cycle["roster"] = {
            b["id"]: dict(
                candidate_id=b["representative"],
                combination=b["combination"],
                model=b["model"],
                effort=b["model_config"]["effort"],
            )
            for b in s["branches"].values()
            if branch_mode(b) != "archived" and b["representative"]
        }
        for role in ("anchors", "guests"):
            for cid in s["config"].get(role, []):
                if cid not in s["candidates"]:
                    raise ValueError("大乱斗参赛版本尚未注册：" + cid)
                cycle["roster"][cid] = dict(
                    candidate_id=cid, combination=role[:-1], model=role[:-1],
                    effort=role[:-1], anchor=role == "anchors", iterable=False,
                )
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
        roster = [row["candidate_id"] for row in cycle["roster"].values()]
        for cid in s["bootstrap"]:
            if len(roster) >= 4:
                break
            roster.append(cid)
        if len(roster) < 4:
            raise ValueError("共同大乱斗至少需要四个参赛版本")
        cycle["entrants"] = roster
        s["promotions_pending"] = 0
        matches = len(roster) * cycle["config"]["brawl_appearances"] // 4
        matches = (matches + 3) // 4 * 4
        key = read_json(Path(s["root"]) / "private/random.json")["key"]
        assignments = schedule(roster, matches, digest([key, "competition", cycle["id"]]))
        chunk = 4 * COMPETITION_BLOCKS_PER_JOB
        if host:
            # Remote launches cost SSH round trips: one job per host worker.
            workers = s["config"]["hosts"][host]["workers"]
            chunk = 4 * -(-len(assignments) // (4 * workers))
        for block in range(0, len(assignments), chunk):
            self.queue(
                s,
                "match",
                scope="competition",
                cycle=cycle["id"],
                host=host,
                assignments=assignments[block : block + chunk],
            )
        s["drain"] = None
        self.event(
            db, s, "competition_started",
            {"cycle": cycle["id"], "matches": matches, "reason": reason, "host": host},
        )

    def startable(self, s):
        """Branches that may begin their next iteration now."""
        cycle = self.current_cycle(s)
        if (
            s["status"] != "running"
            or s["phase"] != "researching"
            or s.get("stop_plan")
            or cycle is None
            or not cycle["reference"]
        ):
            return []
        draining = bool(s.get("drain")) or self.local_brawl(s)
        ready = []
        for b in s["branches"].values():
            if branch_mode(b) != "iterating" or b["status"] != "ready":
                continue
            host = self.evaluation_host(s, b)
            # A branch evaluated on another host waits while that host is
            # offline; branches evaluated here wait while a brawl drains them.
            if (host and not self.online(s, host)) or (draining and not host):
                continue
            ready.append(b)
        return ready

    def start_iteration(self, s, branch):
        b = s["branches"][branch]
        cycle = self.current_cycle(s)
        if b not in self.startable(s):
            raise ValueError("分支尚未就绪，无法开始本次迭代")
        cfg = deepcopy(s["config"])
        combination = s["definitions"][b["combination"]]
        cfg["training_games"] = combination.get("training_games", cfg["training_games"])
        for key in ("evaluation_games", "evaluation_host"):
            if key in combination:
                cfg[key] = combination[key]
        number = b["completed"] + 1
        iid = f"{branch}--iteration-{number:05d}"
        s["iterations"][iid] = dict(
            id=iid,
            branch=branch,
            number=number,
            cycle=cycle["id"],
            experiment=s["id"],
            status="researching",
            config=deepcopy(cfg),
            incumbent=b["representative"],
            opponents=list(cycle["reference"]),
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
        if combination["family"] == "recipe":
            # The recipe service designs and trains; research evaluates.
            cid = f"{branch}--trial-{number:05d}"
            s["iterations"][iid].update(status="submitted", candidates=[cid])
            b["status"] = "preparing"
            return self.queue(s, "trial", branch=branch, iteration=iid, candidate=cid,
                              line=combination["parameters"]["line"])
        return self.queue(s, "agent", branch=branch, iteration=iid)

    def advance(self, s, db):
        if s["status"] != "running":
            return
        self.initialize_ready(s)
        experiment = s["experiments"][-1]
        if s["phase"] == "initializing":
            if any(
                branch_mode(b) == "iterating"
                and not b["representative"]
                and s["definitions"][b["combination"]]["family"] != "recipe"
                for b in s["branches"].values()
            ):
                return
            cycle = self.open_epoch(s, db, experiment["initial_opponents"])
            if not cycle["reference"]:
                cycle["opening"] = True
        cycle = self.current_cycle(s)
        for b in s["branches"].values():
            it = s["iterations"].get(b.get("iteration"))
            if (
                it
                and it["status"] == "submitted"
                and not any(
                    (j["kind"] == "agent"
                     and j["status"] in ("running", "queued", "admitted"))
                    or j["kind"] == "trial"
                    for j in s["jobs"].values()
                    if j.get("iteration") == it["id"]
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
        if cycle["status"] == "competing":
            jobs = [
                j
                for j in s["jobs"].values()
                if j.get("scope") == "competition" and j["cycle"] == cycle["id"]
            ]
            if jobs and all(j["status"] == "completed" for j in jobs):
                self.finish_brawl(s, db, cycle)
                cycle = self.open_epoch(
                    s, db, self.default_opponents(s, s["config"]["opponent_limit"])
                )
        elif cycle.get("opening") or not s.get("stop_plan"):
            self.count_entrants(s, cycle)
            reason = "opening" if cycle.get("opening") else self.brawl_due(s, cycle)
            host = s["config"].get("brawl_host")
            if not reason:
                s["drain"] = None
            elif self.online(s, host):
                self.start_brawl(s, db, cycle, reason, host)
            elif not self.local_busy(s):
                # The brawl host is offline: play here once this machine's
                # iterations have finished (brawl drain).
                self.start_brawl(s, db, cycle, reason, None)
            elif s.get("drain") != reason:
                s["drain"] = reason
                self.event(db, s, "brawl_drain", {"reason": reason, "host": host})
        plan = s.get("stop_plan")
        if plan and cycle["status"] != "competing" and not any(
            it["experiment"] == s["id"] and it["status"] != "completed"
            for it in s["iterations"].values()
        ):
            reason = "已结束当前迭代并暂停" if plan["scope"] == "now" else "本次迭代结束后暂停"
            s.update(status="paused", pause_reason=reason, stop_plan=None)
            experiment["status"] = "paused"

    def finish_brawl(self, s, db, cycle):
        table = self.competition_table(s, cycle)
        previous = {r["branch"]: i + 1 for i, r in enumerate(s["last_ranking"])}
        for rank, row in enumerate(table, 1):
            row.update(rank=rank, rank_change=previous.get(row["branch"], rank) - rank)
        cycle.update(status="completed", ranking=table, completed_at=time.time())
        self.compact_jobs(s)
        s["last_ranking"] = table
        s["experiments"][-1]["completed_brawls"] += 1
        atomic_json(
            Path(s["root"]) / "experiments" / s["id"] / f"brawl-{cycle['number']:04d}.json",
            cycle,
        )
        self.event(db, s, "round_completed", {"round": cycle["number"], "ranking": table})

    def accept_trial(self, s, db, job, result):
        """A recipe trial came back: its trained model enters paired evaluation."""
        b = s["branches"][job["branch"]]
        if job.get("baseline"):
            if result.get("baseline") and result.get("local_actor"):
                cid = f"{b['id']}--initial"
                self.model_candidate(
                    s, db, cid, result["local_actor"], branch=b["id"],
                    combination=b["combination"], note="基线方案训练的第 0 版",
                    status="representative",
                )
                b.update(status="ready", representative=cid)
            else:
                s.update(status="pausing",
                         pause_reason="方案分支第 0 版训练失败：" + str(result.get("error")))
            return
        it = s["iterations"][job["iteration"]]
        it["recipe_trial"] = result.get("trial")
        if result.get("plan"):
            it["plan"] = dict(result["plan"], declared_at=time.time(), updated_at=time.time())
        ready = []
        for cid, row in (result.get("candidates") or {}).items():
            if row.get("status") == "ready" and row.get("local_actor"):
                candidate = self.model_candidate(
                    s, db, cid, row["local_actor"], branch=b["id"],
                    combination=b["combination"], note=result.get("summary") or "",
                )
                candidate.update(parent=it["incumbent"], iteration=it["id"],
                                 experiment=s["id"], training=row.get("training"))
                ready.append(cid)
            else:
                s["candidates"][cid] = dict(
                    id=cid, branch=b["id"], combination=b["combination"],
                    status="failed", error=row.get("error"), parent=it["incumbent"],
                    iteration=it["id"], experiment=s["id"], requirements=[],
                    hypothesis=result.get("summary") or "", workspace="",
                    runtime_path=s["runtime_path"],
                )
        if not ready:
            errors = [r.get("error") for r in (result.get("candidates") or {}).values()]
            self.complete_iteration(
                s, db, it, "; ".join(e for e in errors if e) or result.get("error")
                or "方案尝试没有产出模型",
            )
            return
        it["status"] = "evaluating"
        b["status"] = "evaluating"
        it["evaluated"] = [it["incumbent"], *ready]
        self.queue_stage(s, it, 0)

    def queue_stage(self, s, it, stage):
        """Queue one stage of the paired evaluation; stages add new blocks only.

        Stage k brings the games per side to evaluation_games * 2**k. All
        blocks are drawn up front from the iteration's frozen opponents, so a
        block keeps its opponents, wall and seats whichever stage plays it.
        """
        cfg = it["config"]
        key = read_json(Path(s["root"]) / "private/random.json")["key"]
        final_blocks = cfg["evaluation_games"] * 2 ** (cfg["evaluation_stages"] - 1) // 4
        groups = reference_groups(
            it["opponents"], digest([key, "opponents", it["id"]]), final_blocks
        )
        first = 0 if stage == 0 else cfg["evaluation_games"] * 2 ** (stage - 1) // 4
        last = cfg["evaluation_games"] * 2**stage // 4
        it.update(stage=stage, stage_games=last * 4)
        candidates = it["evaluated"]
        # 分片只改任务边界，不改 block 与种子，所以用实时的 match_workers。
        host = cfg.get("evaluation_host")
        workers = (
            s["config"]["hosts"][host]["workers"] if host else s["config"]["match_workers"]
        )
        shards = evaluation_shard_count(
            dict(cfg, match_workers=workers), last - first, len(candidates)
        )
        for cid in candidates:
            blocks = []
            for block in range(first, last):
                seats = []
                for seat in range(4):
                    players = list(groups[block])
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
                    cycle=it["cycle"],
                    candidate=cid,
                    stage=stage,
                    shard=shard,
                    shards=shards,
                    host=host,
                    assignments=part,
                )

    def stage_decisions(self, it, cfg=None):
        """Per-candidate decision at the current stage's confidence level."""
        cfg = cfg or it.get("config") or {}
        stages = cfg.get("evaluation_stages", 1)
        last = it.get("stage", 0) + 1 >= stages
        confidence = cfg.get(
            "final_confidence" if last else "early_confidence", 0.95
        )
        reference = it["results"].get(it["incumbent"])
        decisions = {}
        for cid in it["candidates"]:
            result = it["results"].get(cid)
            if result is None:
                continue
            for metric, field in (
                ("score", "paired"),
                ("field", "utility_paired"),
                ("hand", "hand_paired"),
            ):
                paired = paired_readout(reference, result, metric, confidence)
                if paired:
                    result[field] = paired
            games = it.get("stage_games") or cfg["evaluation_games"]
            result["promotion"] = dict(
                promotion_decision(reference, result, games, last),
                stage=it.get("stage", 0),
                games=games,
                confidence=confidence,
            )
            decisions[cid] = result["promotion"]["reason"]
        return decisions

    def complete_iteration(self, s, db, it, error=None, cancelled=False):
        b = s["branches"][it["branch"]]
        if it["status"] == "completed":
            return
        success = bool(it["results"]) and not cancelled
        improved = False
        if success:
            choices = [it["incumbent"], *sorted(it["candidates"])]
            # 晋级只使用完整四座轮换组的场口径配对区间；逐局读数不参与。
            decisions = self.stage_decisions(
                it, it.get("config") or s["cycles"][it["cycle"] - 1]["config"]
            )
            qualified = [cid for cid, d in decisions.items() if d == "eligible"]
            winner = it["incumbent"]
            if qualified:
                winner = max(
                    [it["incumbent"], *qualified],
                    key=lambda cid: (
                        _primary_mean(it["results"][cid]),
                        cid == it["incumbent"],
                        -choices.index(cid),
                    ),
                )
            it["selection_rule"] = SELECTION_RULE
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
        if improved:
            s["promotions_pending"] = s.get("promotions_pending", 0) + 1
        b["failed"] += int(not success)
        b["status"] = "ready"
        it.pop("config", None)
        it.update(
            status="completed",
            representative=b["representative"],
            error=error,
            improved=improved,
            completed_at=time.time(),
        )
        self.compact_jobs(s)
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
        if result["status"] == "host_fault":
            # Not the candidate's failure: the work is redone when possible.
            job["status"] = "queued"
            job.pop("stop_requested", None)
            job["host_faults"] = job.get("host_faults", 0) + 1
            if job.get("overflow"):
                job.update(host=None, overflow=False)
            self.event(db, s, "host_fault", {"job": job["id"], "host": job.get("host"),
                                             "error": result.get("error")})
            return
        if not (job["kind"] == "match" and result["status"] != "completed"):
            self.store.receipt(
                db, s["id"], job["id"], result, int(job.get("attempt", 0))
            )
        job.update(
            status="completed", error=result.get("error"), usage=result.get("usage", {})
        )
        it = s["iterations"].get(job.get("iteration"))
        if it and result.get("error"):
            failure = {
                "stage": job["kind"],
                "candidate_id": job.get("candidate"),
                "reason": result["error"],
            }
            failures = it.setdefault("execution_failures", [])
            if failure not in failures:
                failures.append(failure)
        if it and job["kind"] in ("agent", "prepare"):
            it["spent"] += result.get("usage", {}).get("wall_seconds", 0.0)
        if job["kind"] == "agent":
            b = s["branches"][job["branch"]]
            if result.get("thread_id"):
                b["thread_id"] = result["thread_id"]
            if it["status"] == "researching":
                if (
                    result["status"] == "completed"
                    and not result.get("error")
                    and not result.get("interrupted")
                    and result.get("thread_id")
                    and job.get("continuations", 0) < 2
                    and it["config"]["iteration_seconds"] - it["spent"] >= 30
                ):
                    # A clean model turn is not a completed research iteration.
                    # Resume its session and retain the original time budget.
                    job["continuations"] = job.get("continuations", 0) + 1
                    job["spent"] += result.get("usage", {}).get("wall_seconds", 0.0)
                    job["status"] = "queued"
                    self.event(
                        db,
                        s,
                        "agent_continuation",
                        {
                            "job": job["id"],
                            "iteration": it["id"],
                            "continuation": job["continuations"],
                        },
                    )
                    return
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
                it["evaluated"] = [it["incumbent"], *ready]
                self.queue_stage(s, it, 0)
        elif job["kind"] == "trial":
            self.accept_trial(s, db, job, result)
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
                job["diagnostics"] = result.get("diagnostics", [])
                job["diagnostics_truncated"] = result.get("diagnostics_truncated", 0)
                job["focal"] = result["focal"]
                jobs = [
                    j
                    for j in s["jobs"].values()
                    if j.get("iteration") == it["id"] and j["kind"] == "match"
                ]
                if any(j["status"] != "completed" for j in jobs):
                    return
                # A stage is complete for every evaluated version: results are
                # cumulative over all stages so far, since blocks never repeat.
                for cid in sorted({j["candidate"] for j in jobs}):
                    merged = self.merge_focal(
                        [j["focal"] for j in jobs if j["candidate"] == cid]
                    )
                    it["results"][cid] = merged
                    s["candidates"][cid]["last_result"] = merged
                decisions = self.stage_decisions(it)
                last = it["stage"] + 1 >= it["config"]["evaluation_stages"]
                if not last and "continue" in decisions.values() and not any(
                    d == "eligible" for d in decisions.values()
                ):
                    self.queue_stage(s, it, it["stage"] + 1)
                    return
                self.complete_iteration(s, db, it)

    @staticmethod
    def compact_jobs(s):
        """Drop per-job detail once its iteration or round has been settled.

        The whole state is rewritten on every edit, and these fields grew it
        to hundreds of megabytes over a long experiment. Their contents are
        already folded into iteration results and round rankings, and every
        job result stays in the immutable receipts table.
        """
        iterations = s.get("iterations", {})
        # The latest round's competition detail still feeds the progress panel.
        settled = {c["id"] for c in s.get("cycles", [])[:-1] if c["status"] == "completed"}
        for job in s["jobs"].values():
            if job["status"] != "completed":
                continue
            if job.get("scope") == "competition":
                if job.get("cycle") in settled:
                    job.pop("assignments", None)
                    job.pop("summary", None)
            elif iterations.get(job.get("iteration"), {}).get("status") == "completed":
                for key in ("assignments", "summary", "focal"):
                    job.pop(key, None)
        # Per-group arrays only serve the promotion decision; receipts keep them.
        for it in iterations.values():
            if it.get("status") != "completed" or it.get("compacted"):
                continue
            for result in it.get("results", {}).values():
                strip_groups(result)
            for cid in it.get("candidates", []) + [it.get("incumbent")]:
                strip_groups((s["candidates"].get(cid) or {}).get("last_result"))
            it["compacted"] = True

    @staticmethod
    def merge_focal(parts):
        """把同一候选的评估分片合成一份读数；口径等于一次跑完整批。"""
        merged = {
            key: sum(part.get(key, 0) for part in parts)
            for key in (
                "appearances",
                "score_appearances",
                "score_sum",
                "place_sum",
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
        merged["mean_score"] = (
            merged["score_sum"] / merged["score_appearances"]
            if merged["score_appearances"]
            and merged["score_appearances"] == merged["appearances"]
            else None
        )
        merged["mean_place_points"] = (
            merged["place_sum"] / merged["appearances"]
            if merged["appearances"]
            else None
        )
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
            ("field_score_", "field_score_group_means", "field_score_group_means"),
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
        groups, score_field_groups, hand_groups, score_groups = {}, {}, {}, {}
        for j in s["jobs"].values():
            if (
                j.get("scope") != "competition"
                or j["cycle"] != cycle["id"]
                or j["status"] != "completed"
            ):
                continue
            for r in j.get("summary", []):
                # A competition job is exactly one complete four-seat block.
                field_groups = r.get("group_means")
                if not field_groups:
                    if r["appearances"] != 4:
                        raise ValueError("共同竞技回执必须是完整的四座轮换组")
                    field_groups = [r["utility_sum"] / 4]
                groups.setdefault(r["candidate_id"], []).extend(field_groups)
                field_score_groups = r.get("field_score_group_means")
                if not field_score_groups and r.get("score_appearances"):
                    field_score_groups = [r["score_sum"] / 4]
                if field_score_groups:
                    score_field_groups.setdefault(r["candidate_id"], []).extend(
                        field_score_groups
                    )
                hand_mean = r.get("hand_mean_utility")
                score_mean = r.get("hand_mean_score")
                hand_group_means = r.get("hand_group_means")
                score_group_means = r.get("score_group_means")
                if hand_group_means:
                    hand_groups.setdefault(r["candidate_id"], []).extend(
                        hand_group_means
                    )
                    score_groups.setdefault(r["candidate_id"], []).extend(
                        score_group_means
                    )
                elif hand_mean is not None:
                    hand_groups.setdefault(r["candidate_id"], []).append(hand_mean)
                    score_groups.setdefault(r["candidate_id"], []).append(score_mean)
                dst = rows.setdefault(
                    r["candidate_id"],
                    {
                        k: 0
                        for k in (
                            "appearances",
                            "score_appearances",
                            "score_sum",
                            "place_sum",
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
                            score_field_groups.get(cid) or [], prefix="field_score_"
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
                        mean_score=(
                            r["score_sum"] / r["score_appearances"]
                            if r["score_appearances"]
                            and r["score_appearances"] == r["appearances"]
                            else None
                        ),
                        mean_place_points=r["place_sum"] / r["appearances"],
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
                # 平台口径：总分优先、场名次分次之；旧实验回退名次效用。
                -(_primary_mean(r) if _primary_mean(r) is not None else -1e18),
                -(
                    r["mean_place_points"]
                    if r.get("mean_score") is not None
                    else (r.get("hand_mean_utility") or -1e18)
                ),
                -r["appearances"],
                r.get("branch") or r.get("entrant_id") or r["candidate_id"],
            ),
        )

    @staticmethod
    def handoff(s, branch):
        """Compact state transfer that replaces the previous iteration's conversation."""
        b = s["branches"][branch]
        incumbent = s["candidates"].get(b["representative"]) or {}
        ranking = [
            dict(
                rank=index + 1,
                branch=row["branch"],
                mean_score=(
                    None if row.get("mean_score") is None else round(row["mean_score"], 2)
                ),
                mean_place_points=(
                    None
                    if row.get("mean_place_points") is None
                    else round(row["mean_place_points"], 3)
                ),
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
            "下面是累计研究结论与执行证据；先整理未复盘记录，再提出新计划。",
            "incumbent": {
                "candidate_id": b["representative"],
                "hypothesis": (incumbent.get("hypothesis") or "")[:1200],
            },
            "research_conclusions": distinct_conditions(ledger(s, branch)),
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
            # A recipe branch's candidate exists only once its model is handed back.
            candidate = s["candidates"].get(cid) or {"status": "training"}
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

    def snapshot(self, experiment=None, branch=None, after=0, summary=False):
        """Research state for the MCP and the workbench.

        summary=True is the workbench's polling view: iteration records,
        research conclusions and candidates' last results grow with every
        round and are served per branch (branch=...) when a detail opens.
        """
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
            if cycle["status"] == "competing":
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
        hidden = ("workspace", "spec", "training", "runtime_path")
        current = {b.get("representative") for b in s["branches"].values()}
        candidates = [
            {
                **{
                    k: v
                    for k, v in c.items()
                    if k not in hidden
                    and not (
                        summary
                        and (k == "last_result" or (
                            k in ("hypothesis", "checks", "preflight")
                            and c["id"] not in current
                        ))
                    )
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
            if not summary and (not branch or it["branch"] == branch)
        ]
        ranking = (
            cycle["ranking"]
            if cycle and cycle["status"] == "competing"
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
                    "promotions_pending": s.get("promotions_pending", 0),
                    "brawl_promotions": s["config"].get("brawl_promotions"),
                    "drain": s.get("drain"),
                    "config_error": s.get("config_error"),
                }
            )
            if current
            else None
        )
        opponents = self.default_opponents(s, s["config"]["opponent_limit"])
        from mahjong_ai.research.progress import branch_progress

        return dict(
            experiment=public_experiment,
            research_conclusions=[] if summary else ledger(s, branch),
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
            hosts={
                name: {"online": True, **s.get("hosts", {}).get(name, {})}
                for name in s["config"].get("hosts", {})
            },
            brawl_host=s["config"].get("brawl_host"),
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
