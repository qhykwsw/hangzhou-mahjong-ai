"""Remote harness for one recipe trial: budgeted training, paired evaluation, brawl.

The harness owns everything a recipe must not change: the Actor architecture and
observation encoding, the game environment and its table-score reward, the frozen
rule opponents and teacher, the time budget, the evaluation and the brawl. A
recipe is a package named ``recipe`` whose ``train(ctx)`` returns the trained
Actor (or its state dict) before ``ctx.deadline``; see recipes/API.md.

    python -m guided.trial train|smoke|evaluate|brawl --root <dir>

Each command reads ``<root>/plan.json`` and writes ``<root>/status.json`` plus
its outputs; nothing here decides promotion beyond the fixed interval rule.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from . import bootstrap
from .run import atomic_json

GRACE_SECONDS = {"train": 120, "smoke": 60}


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

    def __init__(self, plan, root, deadline, smoke):
        import torch
        from .artifacts import read_bundle
        from .deployment import load_actor
        from .resources import evaluation_resources

        self.plan, self.root, self.deadline, self.smoke = plan, root, deadline, smoke
        self.device = torch.device("cuda")
        self.seed = int(plan["seed"])
        bundle = read_bundle(plan["bundle"])
        self.bundle = bundle["manifest_path"]
        self.teacher = bundle["teacher"]
        self.rules = [entry["id"] for entry in bundle["entries"]]
        self.actor, meta = load_actor(plan["start_actor"], "cuda")
        self.actor_config = meta["config"]
        self.resources = evaluation_resources() | {
            "cpu_count": os.cpu_count(),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
        self.line = plan["line"]
        self._metrics = open(root / "metrics.jsonl", "a")

    def time_left(self):
        return self.deadline - time.time()

    def new_actor(self):
        """A fresh Actor with the fixed architecture (e.g. for opponent snapshots)."""
        from .networks import Actor

        c = self.actor_config
        return Actor(c["dim"], c["layers"], c["heads"]).to(self.device)

    def collect(
        self, actor, cfg, *, samples, iteration, labels=False, history=None, hooks=None
    ):
        """Play training games with ``actor`` in the learning seat.

        cfg holds recipe choices (workers, tables, history_probability,
        critic_features, finish_game ...). Returns (rows, records, resources);
        rows carry the fixed Actor inputs, action/logprob/value, per-hand reward on
        the last decision of each hand, ``hand``, teacher labels when requested,
        and whatever critic_features added.
        """
        from .collection import collect
        from .config import configuration

        known = {k: v for k, v in cfg.items() if k in configuration()}
        extra = {k: v for k, v in cfg.items() if k not in known}
        run_cfg = configuration(**known) | extra
        run_cfg |= {
            "rounds": 16,
            "lookahead": self.actor_config["lookahead"],
            "samples": int(samples),
            "seed": self.seed,
            "bc_only": False,
            "deadline": self.deadline,
        }
        command = {
            "kind": "collect",
            "iteration": int(iteration),
            "deadline": self.deadline,
            "labels": bool(labels),
            "history": list(history or {}),
        }
        return collect(
            actor,
            None,
            run_cfg,
            self.bundle,
            self.bundle,
            command,
            history or {},
            hooks,
        )

    def log(self, **metrics):
        row = {"time": time.time(), "left": round(self.time_left(), 1), **metrics}
        self._metrics.write(json.dumps(row, default=float) + "\n")
        self._metrics.flush()

    def save_actor(self, actor):
        """Checkpoint the current Actor; used if training overruns the deadline."""
        import torch

        tmp = self.root / "actor-checkpoint.tmp"
        torch.save({k: v.detach().cpu() for k, v in actor.state_dict().items()}, tmp)
        os.replace(tmp, self.root / "actor-checkpoint.pt")


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

    def watchdog():
        while time.time() < deadline + GRACE_SECONDS["smoke" if smoke else "train"]:
            time.sleep(1)
        atomic_json(
            root / "status.json", status | {"state": "timeout", "time": time.time()}
        )
        os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        torch.manual_seed(plan["seed"])
        # plan["recipe"] is the recipe package directory, imported as ``recipe``.
        package = Path(plan["recipe"]).resolve()
        if package.name != "recipe" or not (package / "__init__.py").is_file():
            raise ValueError("a recipe is a package directory named 'recipe'")
        sys.path.insert(0, str(package.parent))
        recipe = importlib.import_module("recipe")
        ctx = Context(plan, root, deadline, smoke)
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
            weights_sha256=meta["weights_sha256"],
        )
    except BaseException:
        status.update(
            state="failed", error=traceback.format_exc()[-6000:], finished=time.time()
        )
        atomic_json(root / "status.json", status)
        raise
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
    parser.add_argument("command", choices=("train", "smoke", "evaluate", "brawl"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command in ("train", "smoke"):
        train(root, smoke=args.command == "smoke")
    elif args.command == "evaluate":
        evaluate(root)
    else:
        brawl(root)


if __name__ == "__main__":
    main()
