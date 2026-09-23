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
    "heuristic-efficiency": ("heuristic", "牌效优先", {"style": "efficiency"}),
    "heuristic-fan": ("heuristic", "番型经营", {"style": "fan"}),
    "heuristic-adaptive": ("heuristic", "局势自适应", {"style": "adaptive"}),
    # A recipe branch: candidates are training recipes run by the recipe
    # iteration service on one of its lines (ADR-0013).
    "rl-recipe": ("recipe", "强化学习训练方案", {"line": "rl"}),
}
# Framework fallback roster; projects select their own combinations explicitly.
DEFAULT_COMBINATIONS = (
    "heuristic-balanced",
    "heuristic-adventurous",
    "heuristic-conservative",
    "heuristic-unrestricted",
    "value-lightgbm",
    "value-mlp",
    "ppo",
)
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
    "qwen": {"profile": "netease-qwen-flash", "effort": "max", "efforts": ["max"]},
    "astra-high": {
        "model": "gpt-6-astra",
        "profile": "cliproxyapi-gcp",
        "effort": "high",
        "efforts": ["low", "medium", "high", "xhigh", "max"],
    },
}
DEFAULTS = {
    # Evaluation opponents: the top of the latest common brawl, excluding anchors.
    "opponent_limit": 3,
    "training_caps": {"heuristic": 0, "search": 0, "value": 256, "ppo": 256},
    "iteration_seconds": 3600,
    "agent_concurrency": 14,
    "training_workers": 12,
    "match_workers": 12,
    "candidate_limit": 1,
    # Staged paired evaluation: games per side double for each further stage;
    # earlier stages decide only at early_confidence, the last at final_confidence.
    "evaluation_games": 256,
    "evaluation_stages": 3,
    "early_confidence": 0.99,
    "final_confidence": 0.95,
    "evaluation_shards": 4,
    # Common brawl: appearances per entrant; starts after brawl_promotions
    # promotions, or with any promotion pending after brawl_fallback_seconds.
    "brawl_appearances": 512,
    "brawl_promotions": 4,
    "brawl_fallback_seconds": 43200,
    "brawl_host": None,
    # Fixed brawl entrants besides branch representatives: anchors only
    # measure drift; guests (e.g. an RL model standing in for a branch that
    # has no version yet) may become evaluation opponents.
    "anchors": [],
    "guests": [],
    # Remote match hosts: {name: {ssh, distro, user, python, root, workers}}.
    "hosts": {},
    # Hosts that take local paired evaluations waiting beyond local capacity,
    # but only while no training runs on them.
    "overflow_hosts": [],
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
        if key in DEFAULT_COMBINATIONS
    ],
    "models": MODELS,
    "game": {"rounds": 8, "games_per_round": 1, "stage": "qualifier", "base_score": 1},
    "timeouts": {"discard": 3.0, "response": 1.0},
    "resources": {"memory_mb": 4096, "threads": 1, "device": "cpu"},
}
MACHINE = {"agent_concurrency", "training_workers", "match_workers"}
IMMEDIATE = MACHINE


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
        "opponent_limit",
        "iteration_seconds",
        "agent_concurrency",
        "training_workers",
        "match_workers",
        "candidate_limit",
        "evaluation_games",
        "evaluation_shards",
        "evaluation_stages",
        "brawl_appearances",
        "brawl_promotions",
        "brawl_fallback_seconds",
        "training_games",
        "initialization_seconds",
    ):
        v = config[key]
        if type(v) is not int or not 1 <= v <= 1000000:
            raise ValueError(key + " must be a positive integer")
    if config["candidate_limit"] > 4:
        raise ValueError("candidate_limit must be 1..4")
    for k in ("evaluation_games", "brawl_appearances"):
        if config[k] % 4:
            raise ValueError(k + " must be a multiple of four")
    if not 0.5 < config["final_confidence"] <= config["early_confidence"] < 1:
        raise ValueError("confidence levels must satisfy 0.5 < final <= early < 1")
    if config["brawl_host"] is not None and not isinstance(config["brawl_host"], str):
        raise ValueError("brawl_host must be a host name or null")
    if not isinstance(config["hosts"], dict) or any(
        not isinstance(h, dict)
        or {"ssh", "distro", "user", "python", "root", "workers"} - set(h)
        or type(h["workers"]) is not int
        or h["workers"] < 1
        for h in config["hosts"].values()
    ):
        raise ValueError("hosts entries need ssh, distro, user, python, root and workers")
    if not isinstance(config["overflow_hosts"], list) or any(
        h not in config["hosts"] for h in config["overflow_hosts"]
    ):
        raise ValueError("overflow_hosts must name configured hosts")
    for key in ("anchors", "guests"):
        if not isinstance(config[key], list) or any(
            not isinstance(a, str) for a in config[key]
        ):
            raise ValueError(key + " must list registered version IDs")
    if not isinstance(config["training_caps"], dict) or any(
        k not in ("heuristic", "value", "ppo", "search", "recipe")
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
        if c["family"] not in ("heuristic", "value", "ppo", "search", "recipe"):
            raise ValueError("combination requires a supported algorithm family")
        params = c.setdefault("parameters", deepcopy(default[2]) if default else {})
        required = {
            "heuristic": (
                "style",
                (
                    "balanced",
                    "adventurous",
                    "conservative",
                    "unrestricted",
                    "efficiency",
                    "fan",
                    "adaptive",
                ),
            ),
            "value": ("algorithm", ("lightgbm", "mlp")),
            "ppo": ("algorithm", ("ppo",)),
            "search": ("algorithm", ("rollout", "mcts")),
            "recipe": ("line", None),
        }
        key, allowed = required[c["family"]]
        if (params.get(key) not in allowed) if allowed else not isinstance(
            params.get(key), str
        ):
            raise ValueError("combination algorithm/style does not match its family")
        # Per-combination evaluation: recipe branches play more games per
        # stage and on the host that trains them.
        if "evaluation_games" in c and (
            type(c["evaluation_games"]) is not int or c["evaluation_games"] % 4
        ):
            raise ValueError("combination evaluation_games must be a multiple of four")
        if "evaluation_host" in c and c["evaluation_host"] not in config["hosts"]:
            raise ValueError("combination evaluation_host must be a configured host")
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
    # 本机并发按 CPU 调整，放在不进 Git 的 .local/ 里；只允许并发键，评估条件各机一致。
    local = Path(project) / ".local/machine.toml"
    machine = tomllib.loads(local.read_text()) if local.exists() else {}
    if set(machine) - MACHINE:
        raise ValueError(f"{local} only accepts " + ", ".join(sorted(MACHINE)))
    return validate(merge(merge(merge(DEFAULTS, raw), machine), overrides or {}))
