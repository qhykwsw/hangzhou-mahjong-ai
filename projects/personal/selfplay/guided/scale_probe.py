"""Finite hardware/learning probes; never starts the formal training schedule.

Both hosts consume identical frozen teacher trajectories, with separate game
seeds for validation. Offline joint updates below measure execution cost only;
they are NOT valid on-policy training. Their parameters are discarded before
the equal-time BC learning comparison. BC validation is not a playing-strength
evaluation, and neither it nor this probe selects a production champion.
"""

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

from . import bootstrap
from .config import SCALES, configuration
from .run import atomic_json

PROBE_SCALES = SCALES | {"xlarge": (512, 12, 8)}


def dataset(
    output, bundle, train_samples=8192, validation_samples=4096, seed=2026092101
):
    import numpy as np
    import torch
    from .artifacts import bind_runtime, read_bundle

    output = Path(output)
    if min(train_samples, validation_samples) < 1:
        raise ValueError("positive dataset sizes required")
    if output.exists():
        raise FileExistsError(output)
    manifest = read_bundle(bundle)
    bind_runtime(manifest)
    from .collection import collect
    from .deployment import sample_observations
    from .networks import Actor, Critic

    cfg = configuration(
        dim=32,
        layers=1,
        heads=4,
        workers=8,
        tables=4,
        samples=train_samples,
        bc_only=True,
    )
    torch.manual_seed(cfg["seed"])
    actor, critic = Actor(32, 1, 4).cuda(), Critic().cuda()
    result = {
        "bundle_id": manifest["id"],
        "runtime_hash": manifest["runtime_hash"],
        "code_id": bootstrap.verify_source(),
        "config": cfg,
        "data_seed": seed,
    }
    for name, split_seed, samples in (
        ("train", seed, train_samples),
        ("validation", seed + 1, validation_samples),
    ):
        command = {
            "kind": "collect",
            "iteration": 1,
            "deadline": time.time() + max(300, samples / 100),
            "labels": True,
            "history": [],
        }
        rows, records, resources = collect(
            actor,
            critic,
            cfg | {"seed": split_seed, "samples": samples},
            bundle,
            bundle,
            command,
        )
        if len(rows) < samples:
            raise RuntimeError(f"incomplete {name} dataset: {len(rows)}")
        result[name] = rows
        result[name + "_records"] = records
        result[name + "_resources"] = resources
        print(json.dumps({"dataset": name, **resources}), flush=True)
    # Distinct whole-game random namespaces, never a random row split.
    train_seeds = {r["seed"] for r in result["train_records"]}
    assert not train_seeds.intersection(r["seed"] for r in result["validation_records"])
    result["observations"] = sample_observations(600, seed=seed + 2)
    result["lengths"] = {
        name: dict(
            zip(
                ("p50", "p95", "p99", "max"),
                map(
                    float,
                    np.percentile(
                        [len(r["events"]) for r in result[name]], [50, 95, 99, 100]
                    ),
                ),
            )
        )
        for name in ("train", "validation")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb") as stream:
        pickle.dump(result, stream, protocol=5)
    atomic_json(
        output.with_suffix(".json"),
        {
            key: result[key]
            for key in (
                "bundle_id",
                "runtime_hash",
                "code_id",
                "config",
                "data_seed",
                "lengths",
                "train_resources",
                "validation_resources",
            )
        },
    )


def validation(actor, rows, micro=32):
    import torch
    from .learning import make_batch, masked_log_probs

    total_loss = total_correct = 0.0
    actor.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), micro):
            batch = make_batch(rows[start : start + micro], "cuda")
            logp = masked_log_probs(
                actor(batch["planes"], batch["scalars"], batch["events"]), batch["mask"]
            )
            total_loss -= float(logp.gather(1, batch["teacher"][:, None]).sum())
            total_correct += int((logp.argmax(-1) == batch["teacher"]).sum())
    return {
        "samples": len(rows),
        "cross_entropy": total_loss / len(rows),
        "accuracy": total_correct / len(rows),
    }


def probe(output, data_path, scale, seconds):
    import numpy as np
    import torch
    from .deployment import benchmark, export_actor
    from .learning import make_batch, masked_log_probs, update
    from .networks import Actor, Critic

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(data_path, "rb") as stream:
        data = pickle.load(stream)  # Only our explicitly transferred local artifact.
    cfg = configuration(
        dim=PROBE_SCALES[scale][0],
        layers=PROBE_SCALES[scale][1],
        heads=PROBE_SCALES[scale][2],
        minibatch=32,
        effective_minibatch=512,
        epochs=1,
    )
    torch.manual_seed(cfg["seed"])
    actor, critic = (
        Actor(cfg["dim"], cfg["layers"], cfg["heads"]).cuda(),
        Critic().cuda(),
    )
    initial = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=cfg["critic_learning_rate"], eps=1e-5
    )
    rng = np.random.default_rng(cfg["seed"])
    report = {
        "scale": scale,
        "config": cfg,
        "code_id": bootstrap.verify_source(),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": "float32",
        "parameters": sum(p.numel() for p in actor.parameters()),
        "lengths": data["lengths"],
        "learning_seconds_requested": seconds,
        "formal_training_started": False,
    }

    def persist(stage):
        report["stage"] = stage
        atomic_json(output / "result.json", report)
        print(json.dumps({"stage": stage, "scale": scale}), flush=True)

    persist("joint_profile")
    # Fixed input order across all models/hosts, copied to keep the dataset immutable.
    rows = [
        dict(data["train"][int(i)]) for i in rng.permutation(len(data["train"]))[:1024]
    ]
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(rows), cfg["minibatch"]):
            group = rows[start : start + cfg["minibatch"]]
            batch = make_batch(group, "cuda")
            logp = masked_log_probs(
                actor(batch["planes"], batch["scalars"], batch["events"]), batch["mask"]
            )
            old = logp.gather(1, batch["action"][:, None]).squeeze(1).cpu().tolist()
            for row, value in zip(group, old):
                row["logprob"] = value
    # Deliberately disable KL stopping for cost profiling, not algorithm evaluation.
    profile_cfg = cfg | {"target_kl": 1e6}
    update(
        actor, critic, actor_opt, critic_opt, rows[:512], profile_cfg, 1.0, rows, rng
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    metrics = update(
        actor, critic, actor_opt, critic_opt, rows, profile_cfg, 1.0, rows, rng
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    report["joint_profile"] = {
        "seconds": elapsed,
        "samples_per_second": len(rows) / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "metrics": metrics,
        "purpose": "offline execution-cost only, not on-policy training",
    }
    # Fresh Actor and optimizer: none of the artificial profile updates survive.
    actor.load_state_dict(initial)
    del initial, actor_opt, critic_opt, rows, batch, logp
    gc.collect()
    torch.cuda.empty_cache()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=cfg["critic_learning_rate"], eps=1e-5
    )
    report["before"] = validation(actor, data["validation"])
    persist("short_bc")
    torch.cuda.synchronize()
    started = time.perf_counter()
    deadline = time.time() + seconds
    accepted = steps = epochs = 0
    rng = np.random.default_rng(cfg["seed"])
    while time.time() < deadline:
        metrics = update(
            actor,
            critic,
            actor_opt,
            critic_opt,
            data["train"],
            cfg | {"bc_only": True, "deadline": deadline},
            1.0,
            [],
            rng,
        )
        accepted += metrics["optimized_samples"]
        steps += metrics["sgd_steps"]
        epochs += 1
    torch.cuda.synchronize()
    report["short_bc"] = {
        "seconds": time.perf_counter() - started,
        "optimized_samples": accepted,
        "steps": steps,
        "passes_started": epochs,
    }
    report["after"] = validation(actor, data["validation"])
    torch.save(
        {
            "format": "guided-v1",
            "config": cfg,
            "actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
        },
        output / "actor-checkpoint.pt",
    )
    del actor_opt, critic_opt, actor, critic
    gc.collect()
    torch.cuda.empty_cache()
    artifact = export_actor(output / "actor-checkpoint.pt", output / "actor")
    persist("http_benchmark")
    report["deployment"] = benchmark(
        artifact, data["observations"], count=600, concurrency=10
    )
    persist("complete")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("dataset")
    make.add_argument("--output", required=True)
    make.add_argument("--bundle", required=True)
    make.add_argument("--train-samples", type=int, default=8192)
    make.add_argument("--validation-samples", type=int, default=4096)
    make.add_argument("--seed", type=int, default=2026092101)
    test = sub.add_parser("probe")
    test.add_argument("--output", required=True)
    test.add_argument("--data", required=True)
    test.add_argument("--scale", required=True, choices=PROBE_SCALES)
    test.add_argument("--seconds", type=float, default=90)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    bootstrap.verify_source()
    if args.command == "dataset":
        dataset(
            args.output,
            args.bundle,
            args.train_samples,
            args.validation_samples,
            args.seed,
        )
    else:
        if args.seconds <= 0:
            raise ValueError("positive probe duration required")
        probe(args.output, args.data, args.scale, args.seconds)


if __name__ == "__main__":
    main()
