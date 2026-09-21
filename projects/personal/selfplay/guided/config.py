"""Explicit, serializable defaults for a new experiment."""

DEFAULTS = {
    "seed": 20260920,
    "dim": 128,
    "layers": 4,
    "heads": 4,
    "workers": 8,
    "tables": 8,
    "samples": 65536,
    "rounds": 16,
    "minibatch": 128,
    "effective_minibatch": 512,
    "epochs": 2,
    "learning_rate": 1e-4,
    "critic_learning_rate": 3e-4,
    "clip": 0.1,
    "target_kl": 0.01,
    "entropy": 0.001,
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "lambda_max": 1.0,
    "guidance_scale": 10.0,
    "replay_capacity": 100000,
    "history_probability": 0.25,
    "history_capacity": 16,
    "evaluation_games": 1024,
    "evaluation_seconds": 7200,
    "lookahead": True,
    "device": "cuda",
    "threads": 2,
    "checkpoint_seconds": 600,
    "bc_only": False,
    "critic_warmup_seconds": 0,
}

SCALES = {"small": (128, 4, 4), "medium": (256, 6, 8), "large": (384, 8, 8)}


def configuration(**overrides):
    unknown = set(overrides) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown configuration: {sorted(unknown)}")
    cfg = DEFAULTS | overrides
    if cfg["samples"] < 1 or cfg["workers"] < 1 or cfg["tables"] < 1:
        raise ValueError("positive sampling budget required")
    if cfg["history_capacity"] < 2 or cfg["replay_capacity"] < 0:
        raise ValueError(
            "history capacity >=2 and nonnegative replay capacity required"
        )
    if cfg["evaluation_games"] < 8 or cfg["evaluation_games"] % 4:
        raise ValueError("evaluation requires at least two complete four-seat groups")
    return cfg
