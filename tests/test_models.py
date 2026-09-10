import json
from mahjong_ai.execution.models import resolve_model
from mahjong_ai.execution.agent import codex_command


def test_profile_catalog_is_copied_for_direct_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        'model="base"\nmodel_provider="gateway"\n'
        "model_context_window=372000\nmodel_auto_compact_token_limit=330000\n"
        '[model_providers.gateway]\nname="Gateway"\nwire_api="responses"\n'
    )
    catalog = tmp_path / "catalog.json"
    original = {"models": [{"slug": "test", "supports_search_tool": True}]}
    catalog.write_text(json.dumps(original))
    (tmp_path / "flash.config.toml").write_text(
        'model="test"\nmodel_catalog_json=' + json.dumps(str(catalog)) + "\n"
    )
    choice = {"profile": "flash", "effort": "max", "efforts": ["max"]}
    settings = resolve_model(choice, tmp_path / "job")
    assert settings["model"] == "test" and settings["model_reasoning_effort"] == "max"
    assert json.loads(catalog.read_text()) == original
    assert not json.loads(open(settings["model_catalog_json"]).read())["models"][0][
        "supports_search_tool"
    ]
    cmd = codex_command("/project", "run", "branch", "thread", choice, tmp_path / "job")
    assert cmd[-3:] == ["resume", "thread", "-"]
    assert "features.multi_agent=false" in cmd and "features.shell_tool=false" in cmd
    # Compaction limits survive the isolated worker, otherwise a long branch
    # conversation grows without any client-side ceiling.
    assert "model_context_window=372000" in cmd
    assert "model_auto_compact_token_limit=330000" in cmd
    copied = json.loads(open(settings["model_catalog_json"]).read())["models"][0]
    assert copied["context_window"] == 372000
    assert copied["auto_compact_token_limit"] == 330000
