"""Iteration-line loop (ADR-0016): seal, leases, host queue, recipe rules, platform flag."""
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "projects/personal/firstprinciples"))
    E = importlib.import_module("harness.evaluate")
    rl = importlib.import_module("harness.rl")
    line = importlib.import_module("harness.line")
    monkeypatch.setattr(E, "LOCAL", tmp_path)
    monkeypatch.setattr(line, "RUNNING", tmp_path / "running")
    monkeypatch.setattr(rl, "RL", tmp_path / "rl")
    (tmp_path / "rl").mkdir()
    (tmp_path / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}}))
    monkeypatch.setattr(line, "online", lambda hosts, probe=True: {h: True for h in hosts})
    return E, rl, line


def record(E, **values):
    row = dict(pool="pool-a", line="rl", status="keep", score=0.0, se=0.1, description="x", dev_blocks={}) | values
    with E.records_path().open("a") as f:
        f.write(json.dumps(row) + "\n")


def test_seal_covers_pool_and_host_setup(harness):
    E, rl, line = harness
    with pytest.raises(E.Refused):
        E.check_seal()
    E.seal()
    E.check_seal()
    (rl.RL / "setup-4090.json").write_text(json.dumps({"source_id": "a"}))
    with pytest.raises(E.Refused):
        E.check_seal()
    E.seal()
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-b", "members": {}}))
    with pytest.raises(E.Refused):
        E.check_seal()


def test_shortened_settings_need_a_separate_record(harness, monkeypatch):
    E, _, _ = harness
    monkeypatch.setenv("FP_CONFIRM", "8")
    monkeypatch.setattr(E, "TEST", False)
    with pytest.raises(SystemExit):
        E.override("FP_CONFIRM", 2048)
    monkeypatch.setattr(E, "TEST", True)
    assert E.override("FP_CONFIRM", 2048) == 8


def test_recipe_may_not_play_games_or_lift_limits(harness, tmp_path):
    _, rl, _ = harness
    assert rl.check_recipe(ROOT / "projects/personal/firstprinciples/rl/recipe") == []
    package = tmp_path / "recipe"
    package.mkdir()
    (package / "a.py").write_text(
        "from guided import collection\nimport os\nimport torch\n"
        "torch.cuda.set_per_process_memory_fraction(1.0)\nos.system('x')\nfrom . import b\n")
    problems = rl.check_recipe(package)
    assert len(problems) == 3 and "guided.collection" in problems[0]


def test_ids_are_never_reused(harness):
    E, rl, _ = harness
    (rl.RL / "attempts" / "fp-rl-004").mkdir(parents=True)
    record(E, id="fp-rl-002", pool="old-pool")
    with E.attempts_lock():
        assert E.next_attempt_id("rl") == "fp-rl-005"
        assert E.next_attempt_id("rule") == "fp-rule-000"


def test_requeued_attempts_take_free_hosts_before_new_designs(harness):
    E, rl, line = harness
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    record(E, id="fp-rl-000")
    E.reserve_attempt("rl", "4070", "busy", stage="training")
    E.reserve_attempt("rl", None, "lost on a restart", stage="queued", next="training")
    assert line.design_slot("rl") == ("5070ti", None)
    E.reserve_attempt("rl", None, "lost too", stage="queued", next="training")
    host, why = line.design_slot("rl")
    assert host is None and "排队" in why
    line.hosts("off", "4090")
    assert line.free_hosts("rl") == ["5070ti"]


def test_rule_line_starts_with_v0_alone(harness):
    E, _, line = harness
    assert line.design_slot("rule") == ("local", None)
    E.reserve_attempt("rule", "local", "v0", stage="evaluating")
    assert line.design_slot("rule")[0] is None


def test_rl_hosts_fork_from_the_start_at_once(harness):
    E, rl, line = harness
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    E.seal()
    with pytest.raises(E.Refused):
        line.preflight("rl")  # no start record yet
    record(E, id=rl.start_id(E.load_pool()))
    line.preflight("rl")
    E.reserve_attempt("rl", "4070", "one direction", stage="training")
    assert line.design_slot("rl") == ("4090", None)


def test_source_rules_hold_for_rule_strategies(harness, tmp_path):
    E, _, _ = harness
    package = tmp_path / "rule"
    package.mkdir()
    (package / "strategy.py").write_text(
        "import ctypes\nimport torch\nfrom concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor\n"
        "torch.set_num_threads(8)\npool = ProcessPoolExecutor()\nnice = 3\neval(\"1\")\nx = re.compile\n")
    problems = E.check_source(package)
    assert len(problems) == 4, problems


def test_lease_refuses_a_second_live_session(harness, monkeypatch):
    E, _, line = harness
    me = {"pid": 1, "start": "a", "name": "claude"}
    other = {"pid": 2, "start": "b", "name": "codex"}
    monkeypatch.setattr(line, "alive", lambda owner: True)
    monkeypatch.setattr(line, "agent_identity", lambda: me)
    line.claim("rl")
    monkeypatch.setattr(line, "agent_identity", lambda: other)
    with pytest.raises(E.Refused):
        line.claim("rl")
    line.claim("rl", force=True)
    monkeypatch.setattr(line, "agent_identity", lambda: me)
    with pytest.raises(E.Refused):
        line.claim("rl")
    monkeypatch.setattr(line, "alive", lambda owner: False)
    line.claim("rl")


def test_new_best_is_compared_with_the_free_play_version(harness, monkeypatch):
    E, _, _ = harness
    monkeypatch.setattr(E, "platform_entrant", lambda: "piao")
    record(E, id="ref-piao", line="reference", status="reference",
           dev_blocks={str(k): float(k % 3) for k in range(40)})
    mine = {str(k): float(k % 3) + 2.0 + 0.1 * (k % 2) for k in range(80)}
    flag = E.against_platform({"id": "fp-rl-001", "pool": "pool-a", "dev_blocks": mine})
    assert flag["games"] == 160 and flag["above"]
    worse = {k: v - 3.0 for k, v in mine.items()}
    assert not E.against_platform({"id": "fp-rl-002", "pool": "pool-a", "dev_blocks": worse})["above"]


def test_one_version_one_record(harness):
    E, _, _ = harness
    model = {"family": "model", "parameters": {"weights": "w"}}
    rows = [
        dict(id="ref-small", line="reference", games=4096, finished_at=1, spec=model),
        dict(id="fp-rl-start-x", line="rl", games=16384, finished_at=2, spec=model),
        dict(id="ref-piao", line="reference", games=4096, finished_at=3, spec={"package_hash": "p"}),
        dict(id="ref-piao", line="reference", games=16384, finished_at=4, spec={"package_hash": "p"}),
        dict(id="fp-rl-003", line="rl", games=0, finished_at=5, spec=None),
    ]
    shown = E.distinct(rows)
    assert [(r["id"], r["games"]) for r in shown] == [("fp-rl-start-x", 16384), ("ref-piao", 16384), ("fp-rl-003", 0)]
