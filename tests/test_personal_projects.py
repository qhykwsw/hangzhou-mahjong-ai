import json
from pathlib import Path

from mahjong_ai.competition.releases import load_policy, verify_release


ROOT = Path(__file__).parents[1]


def test_rules_release_is_immutable_and_loadable():
    release = ROOT / "projects/personal/rules/releases/v0.1.0"
    manifest = verify_release(release)
    policy = load_policy(release)
    assert manifest["family"] == "rules"
    assert policy.identity


def test_ppo_release_contains_real_training_evidence():
    release = ROOT / "projects/personal/ppo/releases/v0.1.0"
    manifest = verify_release(release)
    training = json.loads((release / "candidate/model/training.json").read_text())
    policy = load_policy(release)
    assert manifest["family"] == "ppo"
    assert training["algorithm"] == "ppo"
    assert training["updates"] > 0
    assert policy.identity
