from types import SimpleNamespace
import pytest
from mahjong_ai.strategies.catalog import validate_capability


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
