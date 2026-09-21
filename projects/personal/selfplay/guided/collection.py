"""Synchronous GPU inference over independent CPU game workers."""

import multiprocessing as mp
from multiprocessing.connection import wait
import time

from .environment import worker


def collect(actor, critic, cfg, bundle, panel, command, history=None, hooks=None):
    """Play games in CPU workers; the parent runs every network.

    Optional training-recipe hooks apply only to the learning seat during
    collection: hooks["sample"](logits, mask) -> (actions, behaviour logprobs)
    replaces multinomial sampling, and hooks["values"](rows) -> values replaces
    the built-in privileged Critic. Evaluation never uses hooks.
    """
    import torch
    from .learning import make_batch
    from .inference import evaluate_requests

    history = history or {}
    hooks = hooks or {}
    actor.eval()
    if critic is not None:
        critic.eval()
    ctx = mp.get_context("spawn")
    pairs = [ctx.Pipe() for _ in range(cfg["workers"])]
    procs = []
    remaining = set()
    started = time.monotonic()
    all_rows, records, reports = [], [], []
    inference_sizes = []
    device = next(actor.parameters()).device
    try:
        for i, (parent, child) in enumerate(pairs):
            task = dict(command)
            task["samples"] = max(1, cfg["samples"] // cfg["workers"])
            if "jobs" in command:
                task["jobs"] = command["jobs"][i :: cfg["workers"]]
            proc = ctx.Process(
                target=worker,
                args=(child, cfg, str(bundle), str(panel), i, task),
                daemon=True,
            )
            proc.start()
            child.close()
            procs.append(proc)
            remaining.add(parent)
        while remaining:
            ready = wait(list(remaining), timeout=1)
            if not ready:
                for proc, (parent, _) in zip(procs, pairs):
                    if parent in remaining and not proc.is_alive():
                        raise RuntimeError(f"worker died: {proc.exitcode}")
                if time.time() > command["deadline"] + 30:
                    raise TimeoutError("workers did not finish within deadline grace")
                continue
            pending = []
            replies = {}
            for conn in ready:
                msg = conn.recv()
                if msg[0] == "error":
                    raise RuntimeError(msg[1])
                if msg[0] == "done":
                    reports.append(msg[1])
                    remaining.remove(conn)
                    continue
                _, requests, finished = msg
                replies[conn] = []
                for record in finished:
                    all_rows.extend(record.pop("rows"))
                    records.append(record)
                pending.extend((conn, ident, request) for ident, request in requests)
            if command["kind"] == "evaluate":
                evaluated, sizes = evaluate_requests(
                    pending,
                    actor,
                    history,
                    device,
                    cfg.get("inference_batch_limit", 128),
                )
                inference_sizes.extend(sizes)
                for conn, ident, response in evaluated:
                    replies[conn].append((ident, response))
                for conn, response in replies.items():
                    conn.send(response)
                continue
            for slot in sorted({item[2]["slot"] for item in pending}):
                group = [item for item in pending if item[2]["slot"] == slot]
                batch = make_batch([item[2]["row"] for item in group], device)
                model = actor if slot == "model:current" else history[slot]
                with torch.no_grad():
                    logits = model(
                        batch["planes"], batch["scalars"], batch["events"]
                    ).float()
                    logp = logits.masked_fill(~batch["mask"], -1e9).log_softmax(-1)
                    learning = (
                        slot == "model:current"
                        and command["kind"] != "evaluate"
                        and not cfg["bc_only"]
                    )
                    selected = None
                    if command["kind"] == "evaluate":
                        actions = logp.argmax(-1)
                    elif cfg["bc_only"] and slot == "model:current":
                        actions = batch["teacher"]
                    elif learning and "sample" in hooks:
                        actions, selected = hooks["sample"](logits, batch["mask"])
                    else:
                        actions = torch.multinomial(logp.exp(), 1).squeeze(1)
                    values = torch.zeros(len(group), device=device)
                    if learning and "values" in hooks:
                        values = torch.as_tensor(
                            hooks["values"]([item[2]["row"] for item in group]),
                            dtype=torch.float32,
                            device=device,
                        ).reshape(-1)
                    elif learning:
                        values = critic(
                            batch["planes"],
                            batch["scalars"],
                            batch["events"],
                            batch["oracle_planes"],
                            batch["oracle_scalars"],
                        )
                    if selected is None:
                        selected = logp.gather(1, actions[:, None]).squeeze(1)
                    output = (
                        torch.stack((actions.float(), selected, values), -1)
                        .cpu()
                        .numpy()
                    )
                for (conn, ident, _), (action, prob, value) in zip(group, output):
                    replies[conn].append(
                        (ident, (int(action), float(prob), float(value)))
                    )
            for conn, response in replies.items():
                conn.send(response)
        return (
            all_rows,
            records,
            {
                "seconds": time.monotonic() - started,
                "workers": reports,
                "samples": len(all_rows),
                "games": len(records),
                "inference_batches": len(inference_sizes),
                "inference_requests": sum(inference_sizes),
                "inference_batch_max": max(inference_sizes, default=0),
            },
        )
    finally:
        for parent, child in pairs:
            parent.close()
            child.close()
        for proc in procs:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
