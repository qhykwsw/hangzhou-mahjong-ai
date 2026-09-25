"""Batched GPU inference over independent CPU game workers, two rounds in flight."""

from collections import deque
import multiprocessing as mp
from multiprocessing.connection import wait
import time

import numpy as np

from .environment import worker


def request_groups(pending, history_band=0):
    """Group independent Actor requests without trimming or copying their rows."""
    if not isinstance(history_band, int) or history_band < 0:
        raise ValueError("history band must be a nonnegative integer")
    groups = {}
    for item in pending:
        request = item[2]
        length = max(1, len(request["row"]["events"]))
        band = (length + history_band - 1) // history_band if history_band else 0
        groups.setdefault((request["slot"], band), []).append(item)
    return [(slot, group) for (slot, _), group in sorted(groups.items())]


def collect(actor, critic, cfg, bundle, panel, command, history=None, hooks=None, *, inference_session=None):
    """Play games in CPU workers; the parent runs every network.

    Optional training-recipe hooks apply only to the learning seat during
    collection: hooks["sample"](logits, mask) -> (actions, behaviour logprobs)
    replaces multinomial sampling, and hooks["values"](rows) -> values replaces
    the built-in privileged Critic. Evaluation never uses hooks.
    """
    import torch
    from .learning import make_batch
    from .inference import ActorGraphs, Decided, evaluate_requests, policy_logits

    history = history or {}
    hooks = hooks or {}
    actor.eval()
    if critic is not None:
        critic.eval()
    device = next(actor.parameters()).device
    def prepare_inference():
        # The built-in Critic needs the full batch and a sampling hook its own
        # ops; every other path is the fixed decision, which the graphs replay
        # straight from the rows.
        graphs = (
            ActorGraphs(
                fine_batches=cfg.get("fine_graph_batches", False),
                history_band=cfg.get("inference_history_band", 0) if not hooks else 0,
            )
            if device.type == "cuda" and cfg.get("cuda_graphs", True) and not cfg["bc_only"]
            and "sample" not in hooks and (critic is None or "values" in hooks) else None
        )
        inference_actor = actor
        if graphs is not None and command["kind"] != "evaluate" and cfg.get("precast_collection", True):
            import copy
            from .inference import prepare_frozen

            # The policy is immutable throughout this on-policy rollout. Its
            # fp32 learner remains untouched; the snapshot uses the same fp16
            # coefficients autocast otherwise rebuilds on every graph replay.
            inference_actor = prepare_frozen(copy.deepcopy(actor).requires_grad_(False), device)
        return graphs, inference_actor

    if inference_session is None:
        graphs, inference_actor = prepare_inference()
    else:
        options = (
            command["kind"], id(critic), cfg["bc_only"],
            cfg.get("cuda_graphs", True), cfg.get("fine_graph_batches", False),
            cfg.get("inference_history_band", 0), cfg.get("precast_collection", True),
            tuple((key, id(value)) for key, value in sorted(hooks.items())),
        )
        graphs, inference_actor = inference_session.resolve(actor, history, options, prepare_inference)
    # Custom hooks keep their original batches; they may couple their rows.
    history_band = graphs.history_band if graphs else 0
    timers = {"wait": 0.0, "receive": 0.0, "batch": 0.0, "forward": 0.0, "reply": 0.0}
    batch_sizes = []
    ctx = mp.get_context("spawn")
    pairs = [ctx.Pipe() for _ in range(cfg["workers"])]
    procs = []
    remaining = set()
    started = time.monotonic()
    all_rows, records, reports = [], [], []
    inference_sizes = []
    try:
        for i, (parent, child) in enumerate(pairs):
            task = dict(command)
            task["policy_inputs_only"] = command["kind"] == "evaluate" or (
                critic is None and "values" not in hooks
            )
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
        gather = float(cfg.get("gather_seconds", 0.0))
        inflight = deque()  # rounds launched on the device, oldest first

        def finish(groups, replies):
            for group, rows_, learning, decided, complete in groups:
                mark = time.perf_counter()
                output = decided.result()
                if not complete:
                    values = np.zeros(len(group), dtype=np.float32)
                    if learning and "values" in hooks:
                        values = np.asarray(hooks["values"](rows_), dtype=np.float32).reshape(-1)
                    output = np.concatenate((output, values[:, None]), axis=1)
                timers["forward"] += time.perf_counter() - mark
                mark = time.perf_counter()
                for (conn, ident, _), (action, prob, value) in zip(group, output):
                    replies[conn].append((ident, (int(action), float(prob), float(value))))
                timers["reply"] += time.perf_counter() - mark
            mark = time.perf_counter()
            for conn, response in replies.items():
                conn.send(response)
            timers["reply"] += time.perf_counter() - mark

        while remaining or inflight:
            mark = time.perf_counter()
            # With a round on the device, gather the requests that arrive
            # while it runs (gather_seconds; the device is busy anyway, and
            # one larger batch replays fewer graphs than several small ones);
            # otherwise wait for the next ones.
            ready = wait(list(remaining), timeout=gather if inflight else 1) if remaining else []
            timers["wait"] += time.perf_counter() - mark
            mark = time.perf_counter()
            if not ready:
                if inflight:
                    finish(*inflight.popleft())
                    continue
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
            timers["receive"] += time.perf_counter() - mark
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
            groups = []
            for slot, group in request_groups(pending, history_band):
                batch_sizes.append(len(group))
                mark = time.perf_counter()
                model = inference_actor if slot == "model:current" else history[slot]
                learning = (
                    slot == "model:current"
                    and command["kind"] != "evaluate"
                    and not cfg["bc_only"]
                )
                rows_ = [item[2]["row"] for item in group]
                with torch.no_grad():
                    if graphs is not None:
                        # Launch only: every model's batch of this round goes
                        # to the device before any answer is read, and the
                        # round is read after the next one has been launched.
                        decided = graphs.decide(slot, model, rows_, sample=command["kind"] != "evaluate")
                        groups.append((group, rows_, learning, decided, False))
                        timers["batch"] += time.perf_counter() - mark
                        continue
                    batch = make_batch(rows_, device)
                    logits = policy_logits(model,
                        batch["planes"], batch["scalars"], batch["events"]
                    ).float()
                    timers["batch"] += time.perf_counter() - mark
                    mark = time.perf_counter()
                    logp = logits.masked_fill(~batch["mask"], -1e9).log_softmax(-1)
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
                            hooks["values"](rows_),
                            dtype=torch.float32,
                            device=device,
                        ).reshape(-1)
                    elif learning and critic is not None:
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
                    timers["forward"] += time.perf_counter() - mark
                    groups.append((group, rows_, learning, Decided(output), True))
            inflight.append((groups, replies))
            if graphs is None or len(inflight) > 1:
                finish(*inflight.popleft())
        return (
            all_rows,
            records,
            {
                "seconds": time.monotonic() - started,
                "workers": reports,
                "samples": len(all_rows),
                "games": len(records),
                "inference_batches": len(inference_sizes) or len(batch_sizes),
                "inference_requests": sum(inference_sizes) or sum(batch_sizes),
                "inference_batch_max": max(inference_sizes or batch_sizes, default=0),
                "inference_history_band": history_band,
                "inference_batch_buckets": None if graphs is None else graphs.batch_buckets,
                "parent_seconds": {k: round(v, 2) for k, v in timers.items()},
                "graphs": None if graphs is None else len(graphs.graphs),
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
