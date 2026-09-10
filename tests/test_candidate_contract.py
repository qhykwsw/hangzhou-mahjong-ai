from types import SimpleNamespace
import pytest
from mahjong_ai.strategies.catalog import baseline_source, validate_capability


def test_missing_style_is_reported_as_exact_required_policy_attribute(tmp_path):
    module = SimpleNamespace(
        COMBINATION="heuristic-adventurous", PARAMETERS={"style": "adventurous"}
    )
    policy = SimpleNamespace(
        base=SimpleNamespace(style="adventurous"), utilities=lambda obs: []
    )
    with pytest.raises(ValueError, match=r"policy\.style.*adventurous"):
        validate_capability(module, policy, "heuristic-adventurous", tmp_path)


def test_wrapper_with_explicit_style_can_use_its_own_implementation(tmp_path):
    module = SimpleNamespace(COMBINATION="heuristic-adventurous")
    policy = SimpleNamespace(style="adventurous", utilities=lambda obs: [])
    assert (
        validate_capability(module, policy, "heuristic-adventurous", tmp_path)["family"]
        == "heuristic"
    )
    policy.style = "conservative"
    with pytest.raises(ValueError):
        validate_capability(module, policy, "heuristic-adventurous", tmp_path)


def test_unrestricted_rule_has_no_fixed_style_but_requires_utilities(tmp_path):
    module = SimpleNamespace(COMBINATION="heuristic-unrestricted")
    policy = SimpleNamespace(utilities=lambda obs: [])
    for style in (None, "balanced", "adventurous", "conservative", "dynamic"):
        policy.style = style
        assert (
            validate_capability(module, policy, module.COMBINATION, tmp_path)["family"]
            == "heuristic"
        )
    del policy.style
    assert validate_capability(module, policy, module.COMBINATION, tmp_path)
    del policy.utilities
    with pytest.raises(ValueError, match="utilities"):
        validate_capability(module, policy, module.COMBINATION, tmp_path)


def test_unrestricted_baseline_builds_and_plays_legal_action(tmp_path):
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig

    namespace = {}
    exec(baseline_source("heuristic-unrestricted"), namespace)
    policy = namespace["build"](tmp_path)
    validate_capability(
        SimpleNamespace(**namespace), policy, "heuristic-unrestricted", tmp_path
    )
    observation = HandState(RoundConfig(), 42).observation()
    assert policy.act(observation) in observation.legal_actions
