import pytest

from mahjong_ai.competition import CompetitionRegistry
from mahjong_ai.project import ProjectDescriptor


def test_project_descriptor_reads_manifest(tmp_path):
    (tmp_path / "project.toml").write_text(
        '[project]\nid = "personal-ppo"\nfamily = "ppo"\nentrypoint = "src/strategy.py"\n'
    )
    descriptor = ProjectDescriptor.load(tmp_path)
    assert descriptor.id == "personal-ppo"
    assert descriptor.family == "ppo"
    assert descriptor.research == tmp_path / ".local/research"


def test_legacy_project_keeps_local_layout(tmp_path):
    descriptor = ProjectDescriptor.load(tmp_path)
    assert descriptor.family == "legacy"
    assert descriptor.strategies == tmp_path / ".local/strategies"


def test_competition_registry_immutable_versions_and_residency(tmp_path):
    registry = CompetitionRegistry(tmp_path)
    manifest = {
        "entrant_id": "personal-ppo@v1#abc",
        "project_id": "personal-ppo",
        "version": "v1",
        "content_hash": "abc",
        "algorithm": "ppo",
    }
    row = registry.register(manifest, tmp_path / "release", resident=True)
    assert row["resident"] is True
    assert registry.list(resident_only=True)[0]["entrant_id"] == manifest["entrant_id"]
    with pytest.raises(ValueError):
        registry.register(manifest, tmp_path / "other")
    registry.set_resident(manifest["entrant_id"], False)
    registry.set_status(manifest["entrant_id"], "revoked")
    assert registry.get(manifest["entrant_id"])["status"] == "revoked"
