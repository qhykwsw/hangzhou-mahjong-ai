"""Execution-cost sweep over microbatches; all profile weights are discarded.

Runs after the learning/inference probes so GPU contention does not contaminate
their latency. The effective optimizer batch remains 512 on both hosts. Frozen
teacher data is used for profiling only, never claimed as on-policy RL.
"""

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

from guided import bootstrap
from guided.config import configuration
from guided.run import atomic_json
from guided.scale_probe import PROBE_SCALES


def growth_budget(previous, free_bytes, total_bytes):
    """Leave room for allocator fragmentation, Windows use and longer histories.

    The WSL probe observed severe slowdown near VRAM capacity without an OOM.
    Reserve estimates are deliberately conservative; this is a probe, not a
    search for the largest allocation the driver will accept.
    """
    estimate = 1.8 * previous["peak_reserved_bytes"]
    limit = min(0.8 * free_bytes, 0.7 * total_bytes)
    return {
        "allowed": estimate <= limit,
        "estimated_bytes": estimate,
        "limit_bytes": limit,
    }


def measure(data, scale, micro):
    import numpy as np
    import torch
    from guided.learning import make_batch, masked_log_probs, update
    from guided.networks import Actor, Critic

    dim, layers, heads = PROBE_SCALES[scale]
    cfg = configuration(
        dim=dim,
        layers=layers,
        heads=heads,
        minibatch=micro,
        effective_minibatch=512,
        epochs=1,
        target_kl=1e6,
    )
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    rows = [
        dict(data["train"][int(i)]) for i in rng.permutation(len(data["train"]))[:2048]
    ]
    actor, critic = Actor(dim, layers, heads).cuda(), Critic().cuda()
    ao = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    co = torch.optim.Adam(critic.parameters(), lr=cfg["critic_learning_rate"], eps=1e-5)
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(rows), micro):
            group = rows[start : start + micro]
            b = make_batch(group, "cuda")
            logp = masked_log_probs(
                actor(b["planes"], b["scalars"], b["events"]), b["mask"]
            )
            values = logp.gather(1, b["action"][:, None]).squeeze(1).cpu().tolist()
            for row, value in zip(group, values):
                row["logprob"] = value
    del b, logp
    update(actor, critic, ao, co, rows[:512], cfg, 1.0, rows, rng)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = update(actor, critic, ao, co, rows, cfg, 1.0, rows, rng)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    if metrics["optimized_samples"] != len(rows):
        raise RuntimeError("incomplete cost profile")
    return {
        "microbatch": micro,
        "effective_batch": 512,
        "seconds": duration,
        "samples": len(rows),
        "samples_per_second": len(rows) / duration,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_total_bytes_after": torch.cuda.mem_get_info(),
        "metrics": metrics,
    }


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    with gzip.open(args.data, "rb") as stream:
        data = pickle.load(stream)
    result = {
        "code_id": bootstrap.verify_source(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "precision": "float32",
        "purpose": "offline execution-cost only, not on-policy training",
        "scales": {},
    }
    for scale in PROBE_SCALES:
        result["scales"][scale] = []
        for micro in (32, 64, 128, 256):
            completed = result["scales"][scale]
            if completed:
                budget = growth_budget(completed[-1], *torch.cuda.mem_get_info())
                if not budget["allowed"]:
                    completed.append(
                        {
                            "microbatch": micro,
                            "status": "skipped_memory_margin",
                            **budget,
                        }
                    )
                    atomic_json(args.output, result)
                    break
            try:
                entry = measure(data, scale, micro)
                entry["status"] = "complete"
            except torch.OutOfMemoryError:
                entry = {"microbatch": micro, "status": "out_of_memory"}
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            result["scales"][scale].append(entry)
            atomic_json(args.output, result)
            print(json.dumps({"scale": scale, **entry}), flush=True)
            if entry["status"] == "out_of_memory":
                break
            if (
                len(completed) >= 2
                and entry["samples_per_second"]
                < 0.5 * completed[-2]["samples_per_second"]
            ):
                entry["resource_cliff"] = True
                atomic_json(args.output, result)
                break
    result["status"] = "complete"
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
