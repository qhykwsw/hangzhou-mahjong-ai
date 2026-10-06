"""Batched GPU inference over independent CPU game workers, two rounds in flight."""

from collections import deque
import multiprocessing as mp
from multiprocessing.connection import wait
import time

import numpy as np

from .environment import GREEDY, worker


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
    the built-in privileged Critic. Evaluation never uses hooks. With CUDA
    graphs the sampling hook gets the logits of the rollout's fp16 snapshot
    (like every other graph decision) and its batch's rows only; it must stay
    on the device (no host synchronization) to keep the launches overlapped.

    A model teacher (with labels, command["teacher"] names its "model:" slot in
    history) labels the learner's decisions here with its argmax over the
    legal actions, before the learner's move is chosen, so pure imitation
    plays it; the label goes back with the answer. The workers then run no
    teacher of their own.
    """
    import torch
    from .learning import make_batch
    from .inference import ActorGraphs, Decided, evaluate_requests, policy_logits

    history = history or {}
    hooks = hooks or {}
    teacher = command.get("teacher") if command.get("labels") and command["kind"] != "evaluate" else None
    teacher_model = history[teacher] if teacher and teacher.startswith("model:") else None
    actor.eval()
    if critic is not None:
        critic.eval()
    device = next(actor.parameters()).device
    # A sampling hook runs after a graph of the logits (ActorGraphs.decide);
    # cfg["graph_sample_hooks"] False keeps its decisions on the eager path.
    graph_hooks = cfg.get("graph_sample_hooks", True)
    def prepare_inference():
        # The built-in Critic needs the full batch; every other path is the
        # fixed decision (or a sampling hook's, on the graph's logits), which
        # the graphs replay straight from the rows. Behaviour cloning of a rule
        # teacher reads its move from the row; a model teacher's move is its
        # argmax in a graph.
        graphs = (
            ActorGraphs(
                fine_batches=cfg.get("fine_graph_batches", False),
                history_band=cfg.get("inference_history_band", 0) if not hooks else 0,
            )
            if device.type == "cuda" and cfg.get("cuda_graphs", True)
            and (not cfg["bc_only"] or teacher_model is not None)
            and ("sample" not in hooks or graph_hooks)
            and (critic is None or "values" in hooks) else None
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

    options = (
        command["kind"], id(critic), cfg["bc_only"], teacher_model is not None,
        cfg.get("cuda_graphs", True), cfg.get("fine_graph_batches", False),
        cfg.get("inference_history_band", 0), cfg.get("precast_collection", True), graph_hooks,
        tuple((key, id(value)) for key, value in sorted(hooks.items())),
    )
    if inference_session is None:
        from .collection_session import persistent_inference

        graphs, inference_actor = persistent_inference(actor, history, options, prepare_inference)
    else:
        graphs, inference_actor = inference_session.resolve(actor, history, options, prepare_inference)
    # Custom hooks keep their original batches; they may couple their rows.
    history_band = graphs.history_band if graphs else 0
    # The graphs persist across rollouts: this rollout's share of their timers.
    graph_before = (dict(graphs.seconds), dict(graphs.counts)) if hasattr(graphs, "seconds") else None
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
            for group, rows_, learning, decided, complete, taught in groups:
                mark = time.perf_counter()
                # (action, logprob, argmax), with the value (and the teacher's label) when complete
                output = decided.result()
                if not complete:
                    values = np.zeros(len(group), dtype=np.float32)
                    if learning and "values" in hooks:
                        values = np.asarray(hooks["values"](rows_), dtype=np.float32).reshape(-1)
                    output = np.concatenate((output[:, :2], values[:, None], output[:, 2:]), axis=1)
                    if taught is not None:
                        output = np.concatenate((output, taught.result()[:, 2:]), axis=1)
                        if cfg["bc_only"]:
                            # Behaviour cloning: the teacher's argmax is the move
                            # (the logprob stays the learner's own sample's).
                            output[:, 0] = output[:, 4]
                timers["forward"] += time.perf_counter() - mark
                mark = time.perf_counter()
                # (action, logprob, value, argmax), then a model teacher's label on the learner's rows.
                for (conn, ident, _), (action, prob, value, *indices) in zip(group, output):
                    replies[conn].append((ident, (int(action), float(prob), float(value), *map(int, indices))))
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
                model = inference_actor if slot in ("model:current", GREEDY) else history[slot]
                learning = (
                    slot == "model:current"
                    and command["kind"] != "evaluate"
                    and not cfg["bc_only"]
                )
                # Only the learner samples; its greedy continuations (GREEDY) take
                # the argmax, and with the playout baseline so do model opponents,
                # which keeps a deviation's continuation reproducible.
                sample = command["kind"] != "evaluate" and slot != GREEDY and (
                    slot == "model:current" or not cfg.get("playout_baseline"))
                labelled = teacher_model is not None and slot == "model:current"
                rows_ = [item[2]["row"] for item in group]
                with torch.no_grad():
                    if graphs is not None:
                        # Launch only: every model's batch of this round goes
                        # to the device before any answer is read, and the
                        # round is read after the next one has been launched.
                        # The learner's sampling hook chooses from the graph's logits.
                        hook = {"hook": hooks["sample"]} if learning and "sample" in hooks else {}
                        decided = graphs.decide(slot, model, rows_, sample=sample, **hook)
                        # The teacher's argmax on the same rows, under its own
                        # slot (no seat's), so no graph buffer ring is shared.
                        taught = graphs.decide(teacher, teacher_model, rows_, sample=False) if labelled else None
                        groups.append((group, rows_, learning, decided, False, taught))
                        timers["batch"] += time.perf_counter() - mark
                        continue
                    batch = make_batch(rows_, device)
                    if labelled:
                        teacher_logits = policy_logits(
                            teacher_model, batch["planes"], batch["scalars"], batch["events"]
                        ).float()
                        batch["teacher"] = teacher_logits.masked_fill(~batch["mask"], -1e9).log_softmax(-1).argmax(-1)
                    logits = policy_logits(model,
                        batch["planes"], batch["scalars"], batch["events"]
                    ).float()
                    timers["batch"] += time.perf_counter() - mark
                    mark = time.perf_counter()
                    logp = logits.masked_fill(~batch["mask"], -1e9).log_softmax(-1)
                    selected = None
                    if not sample:
                        actions = logp.argmax(-1)
                    elif cfg["bc_only"] and slot == "model:current":
                        # The teacher's move; where the teacher failed (label -1)
                        # the student's own sample keeps the game going.
                        sampled = torch.multinomial(logp.exp(), 1).squeeze(1)
                        actions = torch.where(batch["teacher"] >= 0, batch["teacher"], sampled)
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
                    columns = (actions.float(), selected, values, logp.argmax(-1).float())
                    if labelled:
                        columns += (batch["teacher"].float(),)
                    output = torch.stack(columns, -1).cpu().numpy()
                    timers["forward"] += time.perf_counter() - mark
                    groups.append((group, rows_, learning, Decided(output), True, None))
            inflight.append((groups, replies))
            if graphs is None or len(inflight) > 1:
                finish(*inflight.popleft())
        return (
            all_rows,
            records,
            {
                "seconds": time.monotonic() - started,
                "workers": reports,
                # Rule seats of the finished training games by where they decided (environment.rule_seat_kinds).
                "rule_seats": {kind: sum(r.get("rule_seats", {}).get(kind, 0) for r in reports)
                               for kind in ("c", "callback", "python_engine")},
                "samples": len(all_rows),
                "games": len(records),
                "inference_batches": len(inference_sizes) or len(batch_sizes),
                "inference_requests": sum(inference_sizes) or sum(batch_sizes),
                "inference_batch_max": max(inference_sizes or batch_sizes, default=0),
                "inference_history_band": history_band,
                "inference_batch_buckets": None if graphs is None else graphs.batch_buckets,
                "parent_seconds": {k: round(v, 2) for k, v in timers.items()},
                "graphs": None if graphs is None else len(graphs.graphs),
                # The parent's "batch" seconds, by part of the graph launch (guided.inference.ActorGraphs).
                "graph_seconds": None if graph_before is None else {
                    k: round(v - graph_before[0][k], 2) for k, v in graphs.seconds.items()},
                "graph_counts": None if graph_before is None else {
                    k: v - graph_before[1][k] for k, v in graphs.counts.items()},
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
