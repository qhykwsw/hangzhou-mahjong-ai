"""Validated research settings and configuration activation rules."""

from copy import deepcopy
from pathlib import Path
import re
import tomllib

COMBINATIONS = {
    "heuristic-balanced": ("heuristic", "均衡启发式", {"style": "balanced"}),
    "heuristic-adventurous": ("heuristic", "冒险启发式", {"style": "adventurous"}),
    "heuristic-conservative": ("heuristic", "保守启发式", {"style": "conservative"}),
    "heuristic-unrestricted": ("heuristic", "自由规则策略", {"style": "unrestricted"}),
    "value-lightgbm": ("value", "LightGBM 收益预测", {"algorithm": "lightgbm"}),
    "value-mlp": ("value", "MLP 收益预测", {"algorithm": "mlp"}),
    "ppo": ("ppo", "前馈 PPO", {"algorithm": "ppo"}),
    "search-rollout": ("search", "采样续打", {"algorithm": "rollout"}),
    "search-mcts": ("search", "蒙特卡洛树搜索", {"algorithm": "mcts"}),
}
MODELS = {
    "astra": {
        "model": "gpt-6-astra",
        "profile": None,
        "effort": "medium",
        "efforts": ["low", "medium", "high", "xhigh", "max"],
    },
    "deepseek": {
        "profile": "netease-deepseek-flash",
        "effort": "max",
        "efforts": ["low", "high", "max"],
    },
    "deepseek-pro": {
        "profile": "netease-deepseek-pro",
        "effort": "max",
        "efforts": ["low", "high", "max"],
    },
    "qwen": {"profile": "netease-qwen-flash", "effort": "max", "efforts": ["max"]},
}
DEFAULTS = {
    "total_rounds": 10,
    "iterations_per_round": 10,
    "opponent_limit": 7,
    "branch_iterations": {},
    "training_caps": {"heuristic": 0, "search": 0, "value": 256, "ppo": 256},
    "iteration_seconds": 3600,
    "agent_concurrency": 14,
    "training_workers": 12,
    "match_workers": 12,
    "candidate_limit": 1,
    "evaluation_games": 256,
    "competition_games_per_branch": 512,
    "training_games": 128,
    "initialization_seconds": 3600,
    "combinations": [
        {
            "id": key,
            "branches": 2,
            "branch_ids": ["02", "03"],
            "models": ["deepseek", "qwen"],
            "parameters": params,
        }
        for key, (_, _, params) in COMBINATIONS.items()
        if not key.startswith("search-")
    ],
    "models": MODELS,
    "game": {"rounds": 8, "games_per_round": 1, "stage": "qualifier", "base_score": 1},
    "timeouts": {"discard": 3.0, "response": 1.0},
    "resources": {"memory_mb": 4096, "threads": 1, "device": "cpu"},
}
IMMEDIATE = {
    "agent_concurrency",
    "training_workers",
    "match_workers",
    "total_rounds",
}


def merge(base, patch):
    value = deepcopy(base)
    for k, v in patch.items():
        value[k] = (
            merge(value[k], v)
            if isinstance(v, dict) and isinstance(value.get(k), dict)
            else deepcopy(v)
        )
    return value


def validate(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown settings: " + ", ".join(sorted(unknown)))
    for key in (
        "total_rounds",
        "iterations_per_round",
        "opponent_limit",
        "iteration_seconds",
        "agent_concurrency",
        "training_workers",
        "match_workers",
        "candidate_limit",
        "evaluation_games",
        "competition_games_per_branch",
        "training_games",
        "initialization_seconds",
    ):
        v = config[key]
        if type(v) is not int or not 1 <= v <= 1000000:
            raise ValueError(key + " must be a positive integer")
    if config["candidate_limit"] > 4:
        raise ValueError("candidate_limit must be 1..4")
    for k in ("evaluation_games", "competition_games_per_branch"):
        if config[k] % 4:
            raise ValueError(k + " must be a multiple of four")
    if not isinstance(config["branch_iterations"], dict) or any(
        not isinstance(k, str) or type(v) is not int or not 1 <= v <= 1000000
        for k, v in config["branch_iterations"].items()
    ):
        raise ValueError("branch_iterations requires positive integer allowances")
    if not isinstance(config["training_caps"], dict) or any(
        k not in ("heuristic", "value", "ppo", "search")
        or type(v) is not int
        or not 0 <= v <= 1000000
        for k, v in config["training_caps"].items()
    ):
        raise ValueError(
            "training_caps requires non-negative integer limits per family"
        )
    ids = set()
    for c in config["combinations"]:
        if (
            not isinstance(c.get("id"), str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", c["id"])
            or c["id"] in ids
        ):
            raise ValueError("select distinct valid combination IDs")
        ids.add(c["id"])
        default = COMBINATIONS.get(c["id"])
        c.setdefault("family", default[0] if default else None)
        c.setdefault("name", default[1] if default else c["id"])
        if c["family"] not in ("heuristic", "value", "ppo", "search"):
            raise ValueError("combination requires a supported algorithm family")
        params = c.setdefault("parameters", deepcopy(default[2]) if default else {})
        required = {
            "heuristic": (
                "style",
                ("balanced", "adventurous", "conservative", "unrestricted"),
            ),
            "value": ("algorithm", ("lightgbm", "mlp")),
            "ppo": ("algorithm", ("ppo",)),
            "search": ("algorithm", ("rollout", "mcts")),
        }
        key, allowed = required[c["family"]]
        if params.get(key) not in allowed:
            raise ValueError("combination algorithm/style does not match its family")
        if type(c["branches"]) is not int or c["branches"] < 1:
            raise ValueError("positive branch count required")
        if len(c["models"]) != c["branches"]:
            raise ValueError("one model per branch required")
        branch_ids = c.get("branch_ids", [f"{i + 1:02d}" for i in range(c["branches"])])
        if (
            not isinstance(branch_ids, list)
            or len(branch_ids) != c["branches"]
            or any(
                not isinstance(b, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", b)
                for b in branch_ids
            )
            or len(set(branch_ids)) != len(branch_ids)
        ):
            raise ValueError("distinct stable branch_ids required for every branch")
        if any(m not in config["models"] for m in c["models"]):
            raise ValueError("unknown research model")
        if (
            type(c.get("training_games", config["training_games"])) is not int
            or c.get("training_games", config["training_games"]) < 1
        ):
            raise ValueError("positive training_games required")
    if not config["combinations"]:
        raise ValueError("register at least one combination")
    for model in config["models"].values():
        if model["effort"] not in model["efforts"]:
            raise ValueError("unsupported reasoning effort")
    from mahjong_ai.domain.types import RoundConfig

    RoundConfig(**config["game"])
    if any(
        not isinstance(v, (int, float)) or not 0 < v <= 10
        for v in config["timeouts"].values()
    ):
        raise ValueError("invalid action window")
    r = config["resources"]
    if (
        type(r["memory_mb"]) is not int
        or r["memory_mb"] < 128
        or type(r["threads"]) is not int
        or r["threads"] < 1
    ):
        raise ValueError("invalid resource allocation")
    if r["device"] != "cpu":
        raise ValueError("this executor supports CPU; no silent device substitution")
    return config


def configuration(project, overrides=None):
    path = Path(project) / "research.toml"
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    return validate(merge(merge(DEFAULTS, raw), overrides or {}))
