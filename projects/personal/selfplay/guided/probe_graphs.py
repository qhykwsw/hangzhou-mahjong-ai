"""Where the collection parent's "batch" seconds go on a CUDA host (about a minute of GPU).

    python -m guided.probe_graphs [--actor <exported Actor dir>] [--batch 46] [--rounds 400]

The parent launches every model's batch through ActorGraphs.decide, two rounds
in flight. Its CPU part (filling the pinned buffers) is about a microsecond a
row; anything much larger is the CUDA calls blocking. This measures, on the
host it runs on:

  pinned       whether the graph's host buffers are really page-locked
  busy_upload  one batch's four non-blocking uploads, issued while the GPU spins
  busy_replay  one graph replay, issued while the GPU spins
  rounds       ActorGraphs' own per-part seconds (graphs.seconds) over rounds
               launched the way collection.collect launches them

A non-blocking call that takes about as long as the spin is synchronous on
this host (e.g. pinned memory not honoured under WSL), and the parent then
waits for the device inside "batch".
"""

import argparse
import json
import time

import numpy as np
import torch

from . import bootstrap  # noqa: F401
from .inference import ActorGraphs, prepare_frozen


def rows_for(rng, count, planes, scalars, actions):
    rows = []
    for _ in range(count):
        length = int(rng.integers(8, 160))
        events = np.zeros((length, 5), dtype=np.int16)
        events[:, 0] = 1
        mask = np.zeros(actions, dtype=bool)
        mask[rng.choice(actions, 6, replace=False)] = True
        rows.append(dict(planes=rng.normal(size=(planes, 34)).astype(np.float16),
                         scalars=rng.normal(size=scalars).astype(np.float16), events=events, mask=mask))
    return rows


def spin():
    """Keep the device busy for about 50 ms (calibrate() sets the cycle count)."""
    torch.cuda._sleep(spin.cycles)


def calibrate():
    torch.cuda.synchronize()
    cycles = 10_000_000
    start = time.perf_counter()
    torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    per = (time.perf_counter() - start) / cycles
    spin.cycles = int(0.05 / per)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor")
    parser.add_argument("--batch", type=int, default=46)
    parser.add_argument("--rounds", type=int, default=400)
    args = parser.parse_args()
    from selfplay_core.actions import N_ACTIONS

    if args.actor:
        from .deployment import load_actor

        actor, _ = load_actor(args.actor, "cuda")
    else:
        from .networks import Actor

        actor = Actor(128, 4, 4).cuda().eval()
    model = prepare_frozen(actor.requires_grad_(False), "cuda")
    planes, scalars = model.encoder.tile_in.in_features, model.encoder.state_in.in_features
    rng = np.random.default_rng(7)
    graphs = ActorGraphs()
    rows = rows_for(rng, args.batch, planes, scalars, N_ACTIONS)
    with torch.no_grad():
        graphs.decide("probe", model, rows, sample=True).result()  # capture
        calibrate()
        report = {"device": torch.cuda.get_device_name(0), "batch": args.batch}
        tensors = next(iter(graphs.hosts.values()))[0][0]
        report["pinned"] = all(t.is_pinned() for t in tensors)
        static, out, graph = next(iter(graphs.graphs.values()))
        # Uploads and replay issued while the device spins for about 50 ms.
        spin()
        start = time.perf_counter()
        for target, source in zip(static, tensors):
            target.copy_(source, non_blocking=True)
        report["busy_upload_ms"] = round(1000 * (time.perf_counter() - start), 3)
        torch.cuda.synchronize()
        spin()
        start = time.perf_counter()
        graph.replay()
        report["busy_replay_ms"] = round(1000 * (time.perf_counter() - start), 3)
        torch.cuda.synchronize()
        report["spin_ms"] = 50
        # Rounds as collection.collect launches them: three models' batches a
        # round, a round read after the next is launched.
        before = dict(graphs.seconds)
        batches = [rows_for(rng, args.batch, planes, scalars, N_ACTIONS) for _ in range(8)]
        inflight, waited = [], 0.0
        start = time.perf_counter()
        for r in range(args.rounds):
            launched = [graphs.decide(key, model, batches[(r + i) % 8], sample=(key == "current"))
                        for i, key in enumerate(("current", "greedy", "opponent"))]
            inflight.append(launched)
            if len(inflight) > 1:
                mark = time.perf_counter()
                for decided in inflight.pop(0):
                    decided.result()
                waited += time.perf_counter() - mark
        for decided in inflight.pop(0):
            decided.result()
        total = time.perf_counter() - start
        launches = 3 * args.rounds
        report["per_batch_ms"] = {k: round(1000 * (v - before[k]) / launches, 4) for k, v in graphs.seconds.items()}
        report["per_batch_wait_ms"] = round(1000 * waited / launches, 4)
        report["per_batch_total_ms"] = round(1000 * total / launches, 4)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
