"""Paired performance-difference measurement (harness/pdl.py): a version against itself, and reproducibility."""
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICY = '''
class P:
    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass

    def act(self, obs):
        legal = obs.legal_actions
        return {pick}


def build(model_dir):
    return P()
'''


def package(tmp_path, name, pick):
    path = tmp_path / "packages" / name
    path.mkdir(parents=True)
    (path / "strategy.py").write_text(POLICY.format(pick=pick))
    from mahjong_ai.execution.candidate_runtime import tree_hash

    return dict(family="code", version=name, parameters={}, package_path=str(path), package_hash=tree_hash(path))


@pytest.fixture
def pdl(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "projects/personal/firstprinciples"))
    E = importlib.import_module("harness.evaluate")
    P = importlib.import_module("harness.pdl")
    monkeypatch.setattr(E, "LOCAL", tmp_path)
    monkeypatch.setitem(E.EVAL_WORKERS, "local", 2)
    members = {"m1": package(tmp_path, "m1", "legal[0]"), "m2": package(tmp_path, "m2", "legal[-1]"),
               "m3": package(tmp_path, "m3", "next((a for a in legal if a.action == 'hu'), legal[0])")}
    pool = dict(id="pool-t", members=members, runtime_path=str(ROOT / "src"), timeouts={"discard": 2.0, "response": 0.6},
                config={"rounds": 2, "games_per_round": 1, "stage": "qualifier", "base_score": 1},
                resources={"memory_mb": 2048, "threads": 1, "device": "cpu"})
    (tmp_path / "pool.json").write_text(json.dumps(pool))
    rows = [dict(id="x", pool="pool-t", line="rule", status="keep", spec=package(tmp_path, "x", "legal[0]")),
            dict(id="y", pool="pool-t", line="rule", status="keep",
                 spec=package(tmp_path, "y", "next((a for a in legal if a.action in ('hu', 'peng')), legal[-1])"))]
    (tmp_path / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return P


def test_a_version_against_itself_differs_nowhere(pdl):
    e = pdl.measure("x", "x", hands=16, seed=3, k=2)
    assert (e["diff"], e["se"], e["snapshots"]) == (0.0, 0.0, 0)
    assert e["games"] == 8


def test_the_same_seed_gives_the_same_measurement(pdl):
    first = pdl.measure("x", "y", hands=16, seed=7, k=2)
    again = pdl.measure("x", "y", hands=16, seed=7, k=2)
    assert first["snapshots"] > 0
    keep = ("diff", "se", "parts", "snapshots", "dealer_edge", "draw_rate")
    assert {k: first[k] for k in keep} == {k: again[k] for k in keep}
    rows = [json.loads(line) for line in pdl.results_path().read_text().splitlines()]
    assert [r["b"] for r in rows] == ["y", "y"]
