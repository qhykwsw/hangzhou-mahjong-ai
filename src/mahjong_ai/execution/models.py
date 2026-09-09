"""Resolve branch model profiles without inheriting unrelated tools or plugins."""

import os
import json
from pathlib import Path
import tomllib
from mahjong_ai.research.config import merge


def resolve_model(selection, catalog_dir=None):
    root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    base = root / "config.toml"
    config = tomllib.loads(base.read_text()) if base.exists() else {}
    profile = selection.get("profile")
    if profile:
        if Path(profile).name != profile:
            raise ValueError("model profile must be a name")
        path = root / (profile + ".config.toml")
        if not path.exists():
            raise ValueError("model profile not found: " + profile)
        config = merge(config, tomllib.loads(path.read_text()))
    if selection.get("model"):
        config["model"] = selection["model"]
    if selection["effort"] not in selection["efforts"]:
        raise ValueError("unsupported reasoning effort")
    config["model_reasoning_effort"] = selection["effort"]
    settings = {
        k: config[k]
        for k in (
            "model",
            "model_provider",
            "model_reasoning_effort",
            "model_catalog_json",
        )
        if k in config
    }
    if catalog_dir is not None and settings.get("model_catalog_json"):
        source = Path(settings["model_catalog_json"]).expanduser()
        catalog = json.loads(source.read_text())
        rows = catalog.get("models", []) if isinstance(catalog, dict) else catalog
        for row in rows:
            # This isolated worker has only eight known research tools. Expose
            # them eagerly; third-party gateways may not implement tool search.
            row["supports_search_tool"] = False
        target = Path(catalog_dir) / "research-model-catalog.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(catalog))
        settings["model_catalog_json"] = str(target.resolve())
    provider = config.get("model_provider")
    if provider and provider in config.get("model_providers", {}):
        settings["model_providers." + provider] = config["model_providers"][provider]
    return settings
