import json
from pathlib import Path
import shutil
import time

import pytest

from mahjong_ai.competition.releases import build_release, load_policy, verify_release
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.execution.candidate_runtime import train_package
from mahjong_ai.strategies.common import PolicySpec


ROOT = Path(__file__).parents[1]


def project_copy(tmp_path, family):
    source = ROOT / "projects/personal" / family
    shutil.copy(source / "project.toml", tmp_path / "project.toml")
    shutil.copytree(source / "src", tmp_path / "src")
    return tmp_path


def test_rules_release_is_immutable_and_loadable(tmp_path):
    project_copy(tmp_path, "rules")
    release = Path(build_release(tmp_path, "v0.1.0")["path"])
    manifest = verify_release(release)
    policy = load_policy(release)
    assert manifest["family"] == "rules"
    assert policy.identity
    with pytest.raises(ValueError, match="cannot be overwritten"):
        build_release(tmp_path, "v0.1.0")
    (release / "candidate/strategy.py").write_text("changed")
    with pytest.raises(ValueError, match="has changed"):
        verify_release(release)


def test_ppo_release_contains_real_training_evidence(tmp_path):
    project_copy(tmp_path, "ppo")
    train_package(
        tmp_path / "src",
        tmp_path / "models",
        RoundConfig(rounds=1),
        [PolicySpec()] * 3,
        1,
        17,
        time.monotonic() + 60,
    )
    release = Path(build_release(tmp_path, "v0.1.0")["path"])
    manifest = verify_release(release)
    training = json.loads((release / "candidate/model/training.json").read_text())
    policy = load_policy(release)
    assert manifest["family"] == "ppo"
    assert training["algorithm"] == "ppo"
    assert training["updates"] > 0
    assert policy.identity
