"""One finite remote task, with a durable result for the Mac coordinator."""

import argparse
import json
from pathlib import Path
import time
import traceback

from . import bootstrap  # noqa: F401
from .run import atomic_json


def models(checkpoint, device="cuda"):
    import torch
    from .networks import Actor, Critic

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    actor = Actor(cfg["dim"], cfg["layers"], cfg["heads"]).to(device)
    critic = Critic().to(device)
    actor.load_state_dict(state["actor"])
    if "critic" in state:
        critic.load_state_dict(state["critic"])
    return actor.eval(), critic.eval(), cfg


def perform(task):
    import torch

    torch.set_num_threads(task.get("threads", 2))
    kind = task["kind"]
    if kind == "initialize":
        from .config import configuration
        from .networks import Actor, Critic

        cfg = configuration(**task["config"])
        torch.set_num_threads(cfg["threads"])
        torch.manual_seed(cfg["seed"])
        actor = Actor(cfg["dim"], cfg["layers"], cfg["heads"])
        critic = Critic()
        output = Path(task["output"])
        if output.exists():
            raise FileExistsError("initial parameters are immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        torch.save(
            {
                "format": "guided-v1",
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "config": cfg,
            },
            temporary,
        )
        temporary.replace(output)
        return {"checkpoint": str(output)}
    if kind == "train":
        from .run import train

        checkpoint = train(
            task["run"],
            task["config"],
            task["bundle"],
            task["panel"],
            deadline=task["deadline"],
            initial=task.get("initial"),
            resume=task.get("resume", False),
            carry=task.get("carry", False),
            updates=task.get("updates"),
            bc_seconds=task.get("bc_seconds", 0),
            eval_enabled=task.get("eval_enabled", True),
        )
        return {"checkpoint": str(checkpoint)}
    if kind == "evaluate":
        from .evaluation import evaluate

        actor, critic, cfg = models(task["checkpoint"])
        cfg = cfg | {"evaluation_games": task["games"], "seed": task["eval_seed"]}
        torch.set_num_threads(cfg["threads"])
        result = evaluate(
            actor,
            critic,
            cfg,
            task["bundle"],
            task["panel"],
            0,
            task["deadline"],
            namespace=task["namespace"],
        )
        atomic_json(task["output"], result)
        return {"output": task["output"], "results": result["results"]}
    if kind == "export":
        from .deployment import export_actor

        path = export_actor(task["checkpoint"], task["output"])
        return {"artifact": str(path)}
    if kind == "benchmark":
        from .deployment import benchmark

        result = benchmark(
            task["source"], count=task.get("count", 200), device="cuda", concurrency=10
        )
        atomic_json(task["output"], result)
        return result
    raise ValueError(f"unknown task: {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    args = parser.parse_args()
    path = Path(args.task)
    task = json.loads(path.read_text())
    status = path.with_suffix(".status.json")
    if status.exists():
        prior = json.loads(status.read_text())
        if prior["state"] == "complete":
            raise FileExistsError("completed task is immutable; create a new task ID")
    atomic_json(status, {"state": "running", "started": time.time(), "task": str(path)})
    try:
        bootstrap.verify_source()
        result = perform(task)
        atomic_json(
            status, {"state": "complete", "finished": time.time(), "result": result}
        )
    except BaseException as exc:
        atomic_json(
            status,
            {
                "state": "failed",
                "time": time.time(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
