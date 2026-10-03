"""Remote harness for one recipe trial: budgeted training, paired evaluation, brawl.

The harness owns everything a recipe must not change: the Actor architecture and
observation encoding, the game environment and its table-score reward, the frozen
rule opponents and teacher, the time budget, the evaluation and the brawl. A
recipe is a package named ``recipe`` whose ``train(ctx)`` returns the trained
Actor (or its state dict) before ``ctx.deadline``; see recipes/API.md.

    python -m guided.trial train|smoke|evaluate|brawl|curve --root <dir>

Each command reads ``<root>/plan.json`` and writes ``<root>/status.json`` plus
its outputs; nothing here decides promotion beyond the fixed interval rule.
"""

import argparse
import hashlib
import importlib
import json
import os
import signal
from pathlib import Path
import sys
import threading
import time
import traceback

from . import bootstrap
from .run import atomic_json

GRACE_SECONDS = {"train": 120, "smoke": 60}
DEFAULT_LIMITS = {"gpu_memory_fraction": 0.85, "ram_fraction": 0.75}
# What a harness may fix per host through plan["execution"]: throughput only.
OPENING_TABLES = 8
EXECUTION_OPTIONS = {"workers", "tables", "gather_seconds", "inference_history_band", "fine_graph_batches"}


def _meminfo_total():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal missing from /proc/meminfo")


def _group_usage(pgid):
    """Resident memory and number of processes of the training process group."""
    page = os.sysconf("SC_PAGE_SIZE")
    total = count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.getpgid(int(entry)) != pgid:
                continue
            with open(f"/proc/{entry}/statm") as stream:
                total += int(stream.read().split()[1]) * page
            count += 1
        except (OSError, ValueError, IndexError):
            continue
    return total, count


def memory_limits(plan):
    """Hard caps the recipe cannot lift: the GPU allocator fraction and group RSS.

    Exceeding the GPU cap raises CUDA out-of-memory inside the process instead of
    spilling into shared memory; exceeding the RAM cap kills the process group.
    Either way the trial fails cleanly and the host keeps running.
    """
    import torch

    limits = DEFAULT_LIMITS | (plan.get("limits") or {})
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(limits["gpu_memory_fraction"], 0)
    caps = {
        "gpu_limit_bytes": int(total * limits["gpu_memory_fraction"]),
        "ram_limit_bytes": int(_meminfo_total() * limits["ram_fraction"]),
    }
    if "cpu_capacity" in limits:
        # A shared host (e.g. a desktop) keeps cores for itself; recipes size workers from it.
        caps["cpu_capacity"] = limits["cpu_capacity"]
    return caps


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_actor(weights, config, output, source=None):
    """Write an Actor artifact after a strict load into the fixed architecture."""
    import torch
    from .deployment import FORMAT, load_actor
    from .encoding import ENCODING
    from .networks import Actor

    actor = Actor(config["dim"], config["layers"], config["heads"])
    actor.load_state_dict(weights, strict=True)
    for value in actor.state_dict().values():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite Actor weights")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), output / "actor.pt")
    meta = {
        "format": FORMAT,
        "encoding": ENCODING,
        "config": {k: config[k] for k in ("dim", "layers", "heads", "lookahead")},
        "weights_sha256": _sha256(output / "actor.pt"),
        "source_checkpoint_sha256": source or _sha256(output / "actor.pt"),
        "backend": "pytorch",
        "actions": 231,
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
    load_actor(output, "cpu")
    return meta


class Context:
    """What a recipe may use. Everything else in the harness is off limits."""

    def __init__(self, plan, root, deadline, smoke, limits):
        import torch
        from .artifacts import read_bundle
        from .deployment import load_actor
        from .inference import prepare_frozen
        from .resources import evaluation_resources

        self.plan, self.root, self.deadline, self.smoke = plan, root, deadline, smoke
        self.device = torch.device("cuda")
        self.seed = int(plan["seed"])
        bundle = read_bundle(plan["bundle"])
        self.bundle = bundle["manifest_path"]
        self.teacher = bundle["teacher"]
        self.rules = [entry["id"] for entry in bundle["entries"]]
        # Frozen evaluation opponents of this trial (the default training
        # opponents) and the further frozen versions it offers to train
        # against or, for rule versions, learn from: rule versions run in
        # their own runtimes, model versions are loaded here on the GPU.
        specs = (plan.get("opponents") or {}) | (plan.get("catalog") or {})
        self.opponents = list(plan.get("opponents") or {})
        self.catalog = list(plan.get("catalog") or {})
        self._rule_specs = {k: v for k, v in specs.items() if v.get("family") != "model"}
        self._opponent_models = {
            k: prepare_frozen(load_actor(v["path"], "cuda")[0].eval(), "cuda")
            for k, v in specs.items()
            if v.get("family") == "model"
        }
        # Policies whose actions can label the learning seat's decisions: rule
        # versions (in the game workers) and the catalog's Actors (their argmax
        # over the legal actions, in the collection parent), except those the
        # harness offers as opponents only.
        only = set(plan.get("opponents_only") or ())
        self.teachers = (
            self.rules
            + [k for k in self._rule_specs if k not in self.rules]
            + [k for k in self.catalog if k in self._opponent_models and k not in only]
        )
        self.actor, meta = load_actor(plan["start_actor"], "cuda")
        self.actor_config = meta["config"]
        self.resources = (
            evaluation_resources()
            | {
                "cpu_count": os.cpu_count(),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
            | limits
        )
        self.line = plan["line"]
        self._metrics = open(root / "metrics.jsonl", "a")
        # Optional timed snapshots (plan["snapshot_seconds"]) for training-time studies.
        self._started = time.time()
        self._nodes = sorted(int(x) for x in plan.get("snapshot_seconds") or [])
        self._snapshots = []
        # Optional budget of training hands (plan["hand_budget"]); the time
        # budget still caps the trial.
        self.hand_budget = None if smoke else plan.get("hand_budget")
        self.hands = 0
        # Optional cap on the hands played with teacher labels (imitation), inside the hand budget.
        self.label_hand_budget = None if smoke else plan.get("label_hand_budget")
        self.labeled_hands = 0
        self._rows = 0
        # With a fixed execution, the recipe keeps torch's thread counts
        # (checked on its own thread whenever it calls ctx).
        self._threads = (torch.get_num_threads(), torch.get_num_interop_threads()) if plan.get("execution") else None

    def _guard(self):
        import torch

        if self._threads and (torch.get_num_threads(), torch.get_num_interop_threads()) != self._threads:
            raise ValueError(f"方案改了 torch 线程数（应为 {self._threads}），执行参数由 harness 固定")

    def time_left(self):
        return self.deadline - time.time()

    def hands_left(self):
        """Training hands left in the budget, or None without one."""
        return None if self.hand_budget is None else self.hand_budget - self.hands

    def labeled_hands_left(self):
        """Teacher-labelled hands left in their own cap, or None without one."""
        return None if self.label_hand_budget is None else self.label_hand_budget - self.labeled_hands

    def new_actor(self):
        """A fresh Actor with the fixed architecture (e.g. for opponent snapshots)."""
        from .networks import Actor

        c = self.actor_config
        return Actor(c["dim"], c["layers"], c["heads"]).to(self.device)

    def starts(self, name):
        """A frozen set of openings shipped in the recipe package (recipe/starts/<name>.json.gz)."""
        from .starts import load

        return load(Path(self.plan["recipe"]) / "starts" / f"{name}.json.gz")

    def collect(
        self, actor, cfg, *, samples, iteration, labels=False, history=None, hooks=None,
        inference_session=None, starts=None
    ):
        """Play training games with ``actor`` in the learning seat.

        cfg holds recipe choices (workers, tables, history_probability,
        critic_features, finish_game, opponents, teacher ...). Opponent seats
        draw uniformly from ``cfg["opponents"]`` (IDs from ctx.opponents,
        ctx.catalog or ctx.rules), by default all of ctx.opponents, or the
        bundle's rules when the trial has no frozen opponents. With labels,
        ``cfg["teacher"]`` (one of ctx.teachers, by default ctx.teacher) labels
        the learning seat's decisions: a rule version in the game workers, a
        catalog Actor by its argmax in the collection parent (the workers then
        run no teacher); ``cfg["teacher_plays"]`` plays the teacher's move in
        the learning seat (behaviour cloning). Returns (rows, records, resources);
        rows carry the fixed Actor inputs, action/logprob/value, per-hand reward on
        the last decision of each hand, ``hand``, teacher labels when requested,
        and whatever critic_features added. With ``starts`` (openings, e.g. from
        ctx.starts), every game is one hand from a randomly drawn opening, the
        learner in its ``focal`` seat; such a game counts one hand of the budget.
        """
        from .collection import collect
        from .config import configuration

        self._guard()
        if self.hand_budget is not None and self.hands_left() <= 0:
            raise ValueError("hand budget exhausted")
        chosen = cfg.get("opponents")
        chosen = list(self.opponents if chosen is None else chosen)
        unknown = set(chosen) - set(self.opponents) - set(self.catalog) - set(self.rules)
        if unknown:
            raise ValueError(f"unknown opponents: {sorted(unknown)}")
        teacher = cfg.get("teacher") or self.teacher
        if labels and teacher not in self.teachers:
            raise ValueError(f"unknown teacher: {teacher} (choose from ctx.teachers)")
        if labels and self.label_hand_budget is not None and self.labeled_hands_left() <= 0:
            raise ValueError("teacher-labelled hand budget exhausted")
        # A model teacher sits in the collection's models under its own slot.
        taught = "model:teacher:" + teacher if labels and teacher in self._opponent_models else None
        # Behaviour cloning: the teacher's move is played in the learning seat.
        teacher_plays = bool(cfg.get("teacher_plays"))
        if teacher_plays and not labels:
            raise ValueError("teacher_plays needs labels=True")
        cfg = {k: v for k, v in cfg.items() if k not in ("opponents", "teacher", "teacher_plays")}
        known = {k: v for k, v in cfg.items() if k in configuration()}
        extra = {k: v for k, v in cfg.items() if k not in known}
        run_cfg = configuration(**known) | extra
        # A harness may fix the execution per host (workers, tables, batching,
        # inference scheduling); the recipe's own choices of these are then ignored.
        execution = self.plan.get("execution") or {}
        if set(execution) - EXECUTION_OPTIONS:
            raise ValueError("unknown collection execution option")
        run_cfg |= execution
        if starts:
            # Openings play on the Python engine, whose rule seats fill their
            # caches and cost a core each: a few tables per worker suffice.
            run_cfg["tables"] = min(int(run_cfg["tables"]), OPENING_TABLES)
        samples = int(samples)
        if self.hand_budget is not None:
            # Near the end of the budget, ask only for the rows the hands left
            # should yield (at the rows per hand seen so far), so one call
            # cannot play far past the budget.
            per_hand = self._rows / self.hands if self.hands else 12.0
            samples = max(1, min(samples, int(self.hands_left() * per_hand) + 1))
            if labels and self.label_hand_budget is not None:
                samples = max(1, min(samples, int(self.labeled_hands_left() * per_hand) + 1))
        run_cfg |= {
            # More worker processes than cores only adds memory and contention.
            "workers": max(1, min(int(run_cfg["workers"]), os.cpu_count())),
            "rounds": 16,
            "lookahead": self.actor_config["lookahead"],
            "samples": samples,
            "seed": self.seed,
            "bc_only": teacher_plays,
            "deadline": self.deadline,
        }
        pool = [
            "model:opponent:" + k if k in self._opponent_models else k for k in chosen
        ]
        command = {
            "kind": "collect",
            "iteration": int(iteration),
            "deadline": self.deadline,
            "labels": bool(labels),
            "teacher": taught or teacher,
            "runtime": self.plan.get("runtime"),
            # history_probability only draws the recipe's own snapshots.
            "history": list(history or {}),
            "pool": pool,
            "specs": {k: v for k, v in self._rule_specs.items() if k in chosen or (labels and k == teacher)},
        }
        if starts:
            from .starts import check

            command["starts"] = [check(o) for o in starts]
        models = dict(history or {}) | {
            "model:opponent:" + k: m for k, m in self._opponent_models.items() if k in chosen
        }
        if taught:
            models[taught] = self._opponent_models[teacher]
        rows, records, resources = collect(
            actor,
            None,
            run_cfg,
            self.bundle,
            self.bundle,
            command,
            models,
            hooks,
            inference_session=inference_session,
        )
        # Every game started is played out, so the last collection may overrun the budget.
        played = sum(r.get("hands", run_cfg["rounds"]) for r in records)
        self.hands += played
        if labels:
            self.labeled_hands += played
        self._rows += len(rows)
        with open(self.root / "collection.jsonl", "a") as stream:
            stream.write(json.dumps({"time": time.time(), "iteration": int(iteration), "workers": run_cfg["workers"],
                                     "hands": played, "openings": bool(starts),
                                     "tables": run_cfg["tables"], **{k: v for k, v in resources.items() if k != "workers"}},
                                    default=float) + "\n")
        return rows, records, resources

    def log(self, **metrics):
        self._guard()
        row = {"time": time.time(), "left": round(self.time_left(), 1), "hands": self.hands, **metrics}
        self._metrics.write(json.dumps(row, default=float) + "\n")
        self._metrics.flush()

    def save_actor(self, actor):
        """Checkpoint the current Actor; used if training overruns the deadline."""
        import torch

        state = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        tmp = self.root / "actor-checkpoint.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.root / "actor-checkpoint.pt")
        elapsed = time.time() - self._started
        while self._nodes and elapsed >= self._nodes[0]:
            node = self._nodes.pop(0)
            path = self.root / "snapshots" / f"{node:05d}"
            export_actor(state, self.actor_config, path)
            self._snapshots.append(
                {"node": node, "elapsed": round(elapsed, 1), "path": str(path)}
            )
            atomic_json(self.root / "snapshots.json", self._snapshots)


def train(root, smoke=False):
    import torch

    plan = json.loads((root / "plan.json").read_text())
    if bootstrap.verify_source() != plan["source_id"]:
        raise ValueError("unexpected frozen harness source")
    budget = plan["smoke_seconds"] if smoke else plan["budget_seconds"]
    started = time.time()
    deadline = started + budget
    status = {
        "state": "running",
        "started": started,
        "deadline": deadline,
        "smoke": smoke,
    }
    atomic_json(root / "status.json", status)

    limits = memory_limits(plan)
    status["limits"] = limits
    pgid = os.getpgid(0)
    peak = {"ram": 0, "processes": 0}
    # A harness that fixes the execution also caps what the recipe may add:
    # the parent, the collection workers and multiprocessing's helper are the
    # whole process group, and the GPU allocator stays under its cap.
    execution = plan.get("execution") or {}
    max_processes = int(execution["workers"]) + 3 if execution.get("workers") else None

    def breach():
        rss, processes = _group_usage(pgid)
        peak["ram"] = max(peak["ram"], rss)
        peak["processes"] = max(peak["processes"], processes)
        if rss > limits["ram_limit_bytes"]:
            return "ram_limit", (f"训练进程组内存 {rss / 2**30:.1f} GB 超过上限 "
                                 f"{limits['ram_limit_bytes'] / 2**30:.1f} GB，已强制结束")
        if max_processes and processes > max_processes:
            return "process_limit", (f"训练进程组有 {processes} 个进程，超过执行参数允许的 "
                                     f"{max_processes} 个，已强制结束")
        reserved = torch.cuda.memory_reserved(0)
        if execution and reserved > limits["gpu_limit_bytes"]:
            return "gpu_limit", (f"显存占用 {reserved / 2**30:.1f} GB 超过上限 "
                                 f"{limits['gpu_limit_bytes'] / 2**30:.1f} GB，已强制结束")
        return None

    def watchdog():
        grace = GRACE_SECONDS["smoke" if smoke else "train"]
        while time.time() < deadline + grace:
            found = breach()
            if found:
                atomic_json(
                    root / "status.json",
                    status
                    | {
                        "state": "failed",
                        "reason": found[0],
                        "error": found[1],
                        "ram_peak_bytes": peak["ram"],
                        "processes_peak": peak["processes"],
                        "finished": time.time(),
                    },
                )
                os.killpg(pgid, signal.SIGKILL)
            time.sleep(1)
        atomic_json(
            root / "status.json",
            status
            | {"state": "timeout", "time": time.time(), "ram_peak_bytes": peak["ram"]},
        )
        os.killpg(pgid, signal.SIGKILL)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        torch.manual_seed(plan["seed"])
        # plan["recipe"] is the recipe package directory, imported as ``recipe``.
        package = Path(plan["recipe"]).resolve()
        if package.name != "recipe" or not (package / "__init__.py").is_file():
            raise ValueError("a recipe is a package directory named 'recipe'")
        sys.path.insert(0, str(package.parent))
        recipe = importlib.import_module("recipe")
        ctx = Context(plan, root, deadline, smoke, limits)
        result = recipe.train(ctx)
        if result is None:
            checkpoint = root / "actor-checkpoint.pt"
            if not checkpoint.exists():
                raise ValueError("recipe returned no Actor and saved no checkpoint")
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        else:
            weights = result.state_dict() if hasattr(result, "state_dict") else result
            weights = {k: v.detach().cpu() for k, v in weights.items()}
        meta = export_actor(weights, ctx.actor_config, root / "actor")
        status.update(
            state="complete",
            finished=time.time(),
            training_seconds=time.time() - started,
            overrun_seconds=max(0.0, time.time() - deadline),
            hands=ctx.hands,
            labeled_hands=ctx.labeled_hands,
            weights_sha256=meta["weights_sha256"],
        )
    except BaseException as exc:
        status.update(
            state="failed",
            reason="gpu_memory"
            if isinstance(exc, torch.cuda.OutOfMemoryError)
            else "recipe_error",
            error=traceback.format_exc()[-6000:],
            finished=time.time(),
        )
        raise
    finally:
        status.update(
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ram_peak_bytes=max(peak["ram"], _group_usage(pgid)[0]),
            processes_peak=peak["processes"],
        )
        atomic_json(root / "status.json", status)


def _models(entries):
    from .deployment import load_actor

    return {name: load_actor(path, "cuda")[0] for name, path in entries.items()}


def _execution(plan):
    from .config import configuration
    from .resources import evaluation_resources

    resources = evaluation_resources() | plan.get("execution", {})
    cfg = configuration(
        workers=resources["workers"], tables=resources["tables"], rounds=16
    )
    cfg["inference_batch_limit"] = resources["inference_batch_limit"]
    return cfg


def evaluate(root):
    """Challenger and incumbent play the same seats and walls; interval rule decides."""
    import numpy as np
    from .artifacts import read_bundle
    from .collection import collect
    from .evaluation import schedule, summarize
    from .networks import Critic
    from .statistics import interval

    plan = json.loads((root / "plan.json").read_text())
    if bootstrap.verify_source() != plan["source_id"]:
        raise ValueError("unexpected frozen harness source")
    atomic_json(root / "status.json", {"state": "running", "started": time.time()})
    try:
        bundle = read_bundle(plan["bundle"])
        models = _models(
            {
                "model:challenger": plan["challenger"],
                "model:incumbent": plan["incumbent"],
            }
        )
        targets = list(models)
        games = int(plan["games"])
        jobs = schedule(
            [e["id"] for e in bundle["entries"]],
            targets,
            games,
            plan["seed"],
            plan["namespace"],
        )
        cfg = _execution(plan)
        _, records, stats = collect(
            models["model:challenger"],
            Critic().to("cuda").eval(),
            cfg,
            bundle["manifest_path"],
            bundle["manifest_path"],
            {
                "kind": "evaluate",
                "iteration": 0,
                "jobs": jobs,
                "deadline": time.time() + 7200,
                "labels": False,
            },
            models,
        )
        summary = summarize(records, targets, games)
        diff = interval(
            np.asarray(summary["model:challenger"]["group_scores"])
            - np.asarray(summary["model:incumbent"]["group_scores"])
        )
        result = {
            "games": games,
            "challenger": {
                k: summary["model:challenger"][k] for k in ("mean", "lower", "upper")
            },
            "incumbent": {
                k: summary["model:incumbent"][k] for k in ("mean", "lower", "upper")
            },
            "difference": diff,
            "promoted": diff["lower"] > 0,
            "seconds": stats["seconds"],
        }
        atomic_json(root / "result.json", result)
        atomic_json(
            root / "status.json", {"state": "complete", "finished": time.time()}
        )
    except BaseException:
        atomic_json(
            root / "status.json",
            {"state": "failed", "error": traceback.format_exc()[-6000:]},
        )
        raise


def curve(root):
    """Several Actors and one reference on the same walls, seats and opponents.

    Every target's paired difference to the reference uses the reference's own
    games, so a training-time curve costs one extra target per snapshot.
    """
    import numpy as np
    from .artifacts import read_bundle
    from .collection import collect
    from .evaluation import schedule, summarize
    from .networks import Critic
    from .statistics import interval

    plan = json.loads((root / "plan.json").read_text())
    if bootstrap.verify_source() != plan["source_id"]:
        raise ValueError("unexpected frozen harness source")
    atomic_json(root / "status.json", {"state": "running", "started": time.time()})
    try:
        bundle = read_bundle(plan["bundle"])
        entries = {"model:reference": plan["reference"]} | {
            f"model:{name}": path for name, path in plan["targets"].items()
        }
        models = _models(entries)
        targets = list(models)
        games = int(plan["games"])
        jobs = schedule(
            [e["id"] for e in bundle["entries"]],
            targets,
            games,
            plan["seed"],
            plan["namespace"],
        )
        _, records, stats = collect(
            models["model:reference"],
            Critic().to("cuda").eval(),
            _execution(plan),
            bundle["manifest_path"],
            bundle["manifest_path"],
            {
                "kind": "evaluate",
                "iteration": 0,
                "jobs": jobs,
                "deadline": time.time() + 6 * 3600,
                "labels": False,
            },
            models,
        )
        summary = summarize(records, targets, games)
        reference = np.asarray(summary["model:reference"]["group_scores"])
        result = {
            "games": games,
            "seconds": stats["seconds"],
            "targets": {
                name.removeprefix("model:"): {
                    k: summary[name][k] for k in ("mean", "lower", "upper")
                }
                | {"group_scores": summary[name]["group_scores"]}
                for name in targets
            },
            "differences": {
                name.removeprefix("model:"): interval(
                    np.asarray(summary[name]["group_scores"]) - reference
                )
                for name in targets
                if name != "model:reference"
            },
        }
        atomic_json(root / "result.json", result)
        atomic_json(
            root / "status.json", {"state": "complete", "finished": time.time()}
        )
    except BaseException:
        atomic_json(
            root / "status.json",
            {"state": "failed", "error": traceback.format_exc()[-6000:]},
        )
        raise


def brawl(root):
    """The incumbent and the four frozen rules in a balanced five-way brawl."""
    from .artifacts import read_bundle
    from .brawl import schedule, summarize, validate_records
    from .collection import collect
    from .networks import Critic

    plan = json.loads((root / "plan.json").read_text())
    if bootstrap.verify_source() != plan["source_id"]:
        raise ValueError("unexpected frozen harness source")
    atomic_json(root / "status.json", {"state": "running", "started": time.time()})
    try:
        bundle = read_bundle(plan["bundle"])
        models = _models({"model:incumbent": plan["incumbent"]})
        entrants = list(models) + [e["id"] for e in bundle["entries"]]
        blocks = int(plan["appearances"]) // 16
        jobs = schedule(entrants, blocks, plan["seed"])
        cfg = _execution(plan)
        _, records, stats = collect(
            models["model:incumbent"],
            Critic().to("cuda").eval(),
            cfg,
            bundle["manifest_path"],
            bundle["manifest_path"],
            {
                "kind": "evaluate",
                "iteration": 0,
                "jobs": jobs,
                "deadline": time.time() + 7200,
                "labels": False,
            },
            models,
        )
        validate_records(records, jobs, entrants)
        summary = summarize(records, entrants)
        summary.pop("records", None)
        atomic_json(
            root / "result.json",
            summary | {"seconds": stats["seconds"], "games": len(records)},
        )
        atomic_json(
            root / "status.json", {"state": "complete", "finished": time.time()}
        )
    except BaseException:
        atomic_json(
            root / "status.json",
            {"state": "failed", "error": traceback.format_exc()[-6000:]},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("train", "smoke", "evaluate", "brawl", "curve")
    )
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command in ("train", "smoke"):
        train(root, smoke=args.command == "smoke")
    elif args.command == "evaluate":
        evaluate(root)
    elif args.command == "curve":
        curve(root)
    else:
        brawl(root)


if __name__ == "__main__":
    main()
