"""Runnable baseline packages and executable combination capability checks."""

import json
from pathlib import Path
from mahjong_ai.research.config import COMBINATIONS


def baseline_source(combination, parameters=None, family=None):
    default = COMBINATIONS.get(combination)
    family = family or default[0]
    defaults = default[2] if default else {}
    settings = {**defaults, **(parameters or {})}
    selector = settings.pop("algorithm", None)
    if family == "heuristic":
        body = "from mahjong_ai.strategies.heuristic import HeuristicPolicy\ndef build(model_dir): return HeuristicPolicy(**PARAMETERS)\n"
    elif family == "search":
        settings["algorithm"] = selector
        body = "from mahjong_ai.strategies.search import SearchPolicy\ndef build(model_dir): return SearchPolicy(**PARAMETERS)\n"
    else:
        body = (
            "from mahjong_ai.strategies.learning import LearnedPolicy, PPOPolicy, train as train_model\n"
            f"ALGORITHM = {selector!r}\n"
            "def build(model_dir):\n"
            ' cls = PPOPolicy if ALGORITHM == "ppo" else LearnedPolicy\n'
            " return cls(ALGORITHM, model_dir, **PARAMETERS)\n"
            "def train(ctx): return train_model(ctx, ALGORITHM, PARAMETERS)\n"
        )
    return (
        f"COMBINATION = {combination!r}\nFAMILY = {family!r}\nPARAMETERS = {settings!r}\n"
        + body
    )


def validate_capability(module, policy, combination, package, definition=None):
    if definition:
        family = definition["family"]
        defaults = definition["parameters"]
    else:
        family, _, defaults = COMBINATIONS[combination]
    if getattr(module, "COMBINATION", None) != combination:
        raise ValueError("combination identity changed")
    if family == "heuristic":
        if getattr(policy, "style", None) != defaults["style"]:
            raise ValueError(
                f"policy.style must equal {defaults['style']!r}; set it on the object returned by build(model_dir), including wrapper classes"
            )
        if not callable(getattr(policy, "utilities", None)):
            raise ValueError(
                "policy.utilities(observation) must be callable for a heuristic combination"
            )
    if family in ("value", "ppo"):
        expected = defaults["algorithm"]
        if (
            not callable(getattr(module, "train", None))
            or getattr(policy, "algorithm", None) != expected
        ):
            raise ValueError(
                f"strategy.train(ctx) must be callable and policy.algorithm must equal {expected!r} on the object returned by build(model_dir)"
            )
        evidence = Path(package) / "model/training.json"
        if not evidence.exists():
            raise ValueError("missing actual training evidence")
        row = json.loads(evidence.read_text())
        if row.get("algorithm") != expected or row.get("updates", 0) < 1:
            raise ValueError("no completed algorithm update")
    if (
        family == "search"
        and getattr(policy, "algorithm", None) != defaults["algorithm"]
    ):
        raise ValueError(
            f"policy.algorithm must equal {defaults['algorithm']!r} on the object returned by build(model_dir)"
        )
    return {"combination": combination, "family": family}
