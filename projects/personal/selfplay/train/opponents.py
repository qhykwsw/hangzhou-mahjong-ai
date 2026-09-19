"""Fixed league opponents: rule styles, exact-shape heuristics and auto-research
representatives, all loaded in-process (design decision Q13a)."""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

from mahjong_ai.strategies.common import RulePolicy
from mahjong_ai.strategies.heuristic import HeuristicPolicy

BUILTIN = [
    {"id": "rule-baseline", "kind": "rule", "params": {}},
    {"id": "heuristic-conservative", "kind": "heuristic", "style": "conservative"},
    {"id": "heuristic-balanced", "kind": "heuristic", "style": "balanced"},
    {"id": "heuristic-adventurous", "kind": "heuristic", "style": "adventurous"},
]


def auto_research_representatives(project_root) -> list[dict]:
    """Current representative of every active auto-research branch."""
    db = Path(project_root) / ".local/service/research.sqlite"
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute("select state from registry").fetchone()
    finally:
        con.close()
    if not row:
        return []
    state = json.loads(row[0])
    out = []
    for bid, branch in state.get("branches", {}).items():
        rep = branch.get("representative")
        if not rep or branch.get("archived"):
            continue
        cand = state.get("candidates", {}).get(rep)
        if not cand:
            continue
        workspace = Path(cand["workspace"])
        if (workspace / "strategy.py").exists():
            out.append({"id": f"auto:{rep}", "kind": "package", "path": str(workspace)})
    return out


class PackageLoader:
    """Import a candidate package once per process; build fresh policies on demand."""

    def __init__(self):
        self.modules = {}

    def module(self, path):
        path = str(Path(path).resolve())
        if path not in self.modules:
            if path not in sys.path:
                sys.path.insert(0, path)
            spec = importlib.util.spec_from_file_location(
                f"league_pkg_{abs(hash(path))}", Path(path) / "strategy.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules[path] = module
        return self.modules[path]

    def build(self, path):
        return self.module(path).build(Path(path) / "model")


class OpponentFactory:
    def __init__(self, entries):
        self.entries = {e["id"]: e for e in entries}
        self.loader = PackageLoader()
        self.shared = {}

    def ids(self):
        return list(self.entries)

    def build(self, opponent_id):
        entry = self.entries[opponent_id]
        kind = entry["kind"]
        if kind == "rule":
            return self.shared.setdefault(opponent_id, RulePolicy(**entry.get("params", {})))
        if kind == "heuristic":
            return self.shared.setdefault(opponent_id, HeuristicPolicy(style=entry["style"]))
        if kind == "package":
            return self.loader.build(entry["path"])
        raise ValueError(f"unknown opponent kind {kind}")


def default_entries(cfg, repo_root):
    if cfg.opponent_entries:
        source = Path(cfg.opponent_entries)
        source = source if source.is_absolute() else Path(repo_root) / source
        entries = json.loads(source.read_text())
        for entry in entries:
            if entry["kind"] == "package":
                path = Path(entry["path"])
                entry["path"] = str(path if path.is_absolute() else (source.parent / path).resolve())
                if not (Path(entry["path"]) / "strategy.py").exists():
                    raise FileNotFoundError(f"opponent package without strategy.py: {entry['path']}")
        return entries
    entries = list(BUILTIN)
    if cfg.include_auto_research:
        entries.extend(auto_research_representatives(Path(repo_root) / cfg.auto_research_project))
    return entries
