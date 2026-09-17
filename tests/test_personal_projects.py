from pathlib import Path
import shutil

import pytest

from mahjong_ai.competition.releases import build_release, load_policy, verify_release


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
