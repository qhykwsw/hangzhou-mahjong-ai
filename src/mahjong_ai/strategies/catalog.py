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
        if settings.get("style") == "unrestricted":
            settings.pop("style")
            body = "from mahjong_ai.strategies.common import RulePolicy\ndef build(model_dir): return RulePolicy(**PARAMETERS)\n"
        else:
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
        raise ValueError("COMBINATION 与组合身份不一致")
    if family == "heuristic":
        if (
            defaults["style"] != "unrestricted"
            and getattr(policy, "style", None) != defaults["style"]
        ):
            raise ValueError(
                f"policy.style 必须等于 {defaults['style']!r}；请设置在 build(model_dir) "
                "返回的对象上，包装类也一样"
            )
        if not callable(getattr(policy, "utilities", None)):
            raise ValueError(
                "heuristic 组合的 policy.utilities(observation) 必须可调用"
            )
    if family in ("value", "ppo"):
        expected = defaults["algorithm"]
        if (
            not callable(getattr(module, "train", None))
            or getattr(policy, "algorithm", None) != expected
        ):
            raise ValueError(
                f"strategy.train(ctx) 必须可调用，且 build(model_dir) 返回对象的 "
                f"policy.algorithm 必须等于 {expected!r}"
            )
        evidence = Path(package) / "model/training.json"
        if not evidence.exists():
            raise ValueError("缺少实际训练的证据")
        row = json.loads(evidence.read_text())
        if row.get("algorithm") != expected or row.get("updates", 0) < 1:
            raise ValueError("没有完成任何算法更新")
    if (
        family == "search"
        and getattr(policy, "algorithm", None) != defaults["algorithm"]
    ):
        raise ValueError(
            "build(model_dir) 返回对象的 policy.algorithm 必须等于 "
            f"{defaults['algorithm']!r}"
        )
    return {"combination": combination, "family": family}
