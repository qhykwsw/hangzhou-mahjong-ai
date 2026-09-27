"""Iteration-line loop (ADR-0016): seal, leases, host queue, recipe rules, platform flag."""
import importlib
import json
import time
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
    assert line.design_hosts("rl") == (["5070ti"], None)
    E.reserve_attempt("rl", None, "lost too", stage="queued", next="training")
    hosts, why = line.design_hosts("rl")
    assert hosts == [] and "排队" in why
    line.hosts("off", "4090")
    assert line.free_hosts("rl") == ["5070ti"]


def test_rule_line_starts_with_v0_alone(harness):
    E, _, line = harness
    assert line.design_hosts("rule") == (["local"], None)
    E.reserve_attempt("rule", "local", "v0", stage="evaluating")
    assert line.design_hosts("rule")[0] == []


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
    assert line.design_hosts("rl") == (["4090", "5070ti"], None)


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


@pytest.mark.parametrize("new_best_dev, verdict, replayed", [(2.0, "discard", []), (0.5, "keep", ["fp-rl-008"])])
def test_new_best_during_confirmation_is_first_compared_on_dev(harness, monkeypatch, new_best_dev, verdict, replayed):
    E, _, _ = harness
    monkeypatch.setattr(E, "DEV_BLOCKS", 4)
    monkeypatch.setattr(E, "CONFIRM_BLOCKS", 2)
    monkeypatch.setattr(E, "check_seal", lambda: None)
    monkeypatch.setattr(E, "platform_entrant", lambda: None)
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}, "dev_seed": 1}))
    model = {"family": "model", "parameters": {"weights": "old"}}
    record(E, id="fp-rl-007", finished_at=1, games=16, spec=model, dev_blocks={str(b): 0.0 for b in range(4)})
    calls = []

    def play(pool, attempt_id, spec, blocks, seed, label, host):
        calls.append((attempt_id, label))
        if label == "confirm":  # another host's attempt becomes the best meanwhile
            record(E, id="fp-rl-008", finished_at=2, games=16, spec={"parameters": {"weights": "new"}},
                   dev_blocks={str(b): new_best_dev for b in range(4)})
        score = 1.0 if attempt_id == "fp-rl-009" else 0.0
        return {b: score for b in blocks}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}

    monkeypatch.setattr(E, "play", play)
    result = E.evaluate("rl", {"parameters": {"weights": "mine"}}, "x", "fp-rl-009", host="4070", confirm_seed=5)
    assert result["status"] == verdict
    assert [a for a, label in calls if label.startswith("confirm-for-")] == replayed


def test_catalog_offers_every_version_that_became_a_line_best(harness):
    E, rl, _ = harness
    pool = {"id": "pool-a", "members": {"piao": {"family": "code"}}, "runtime_path": "/pool-runtime"}
    (E.LOCAL / "pool.json").write_text(json.dumps(pool))
    rule = {"family": "code", "version": "fp-rule-002", "package_hash": "p2"}
    model = {"family": "model", "version": "fp-rl-001", "parameters": {"weights": "w1"}}
    record(E, id="fp-rule-002", line="rule", spec=rule, games=1, finished_at=1)
    record(E, id="fp-rule-003", line="rule", status="discard", spec=dict(rule, package_hash="p3"), games=1, finished_at=2)
    record(E, id="fp-rl-001", spec=model, games=1, finished_at=3)
    record(E, id="fp-rl-002", status="unconfirmed", spec=dict(model, parameters={"weights": "w2"}), games=1,
           finished_at=4)
    record(E, id="ref-piao", line="reference", status="reference", spec={"package_hash": "pp"}, games=1, finished_at=5)
    offered = rl.catalog()
    assert set(offered) == {"fp-rule-002", "fp-rl-001"}
    assert offered["fp-rule-002"]["spec"]["runtime_path"] == "/pool-runtime"  # rule versions play under the pool's runtime
    assert "runtime_path" not in offered["fp-rl-001"]["spec"]


@pytest.mark.parametrize("mismatches, best_after", [(0, "fp-rule-003"), (2, "fp-rule-002")])
def test_equivalent_code_takes_over_the_best_versions_scores(harness, monkeypatch, mismatches, best_after):
    E, _, line = harness
    equivalence = importlib.import_module("harness.equivalence")
    monkeypatch.setattr(E, "platform_entrant", lambda: None)
    monkeypatch.setattr(line, "decision_ms", lambda state: {"mine": 2.0, "base": 8.0})
    record(E, id="fp-rule-002", line="rule", games=16, finished_at=1, stats={"timeouts": 0},
           spec={"package_path": "/best"}, dev_blocks={"0": 1.5, "1": -0.5}, score=0.5)
    seen = []

    def check(best, new, seed, root):
        seen.append((str(best), new))
        return {"hands": 64, "decisions": 700, "mismatches": mismatches, "seconds": [8.0, 2.0],
                "examples": ["第 1 局 draw：最好版本 a，新代码 b"] * mismatches}

    monkeypatch.setattr(equivalence, "check", check)
    state = dict(id="fp-rule-003", line="rule", host="local", description="剪枝", why="更快", base="fp-rule-002",
                 started_at=2, design_seconds=60, stage="equivalence", confirm_seed=7,
                 spec={"package_path": "/new", "package_hash": "n"})
    line.save(state)
    line.equivalent(state)
    assert seen == [("/best", "/new")]
    assert E.best("pool-a", "rule")["id"] == best_after
    mine = E.attempts("pool-a", "rule")[-1]
    if mismatches:
        assert mine["status"] == "inequivalent" and "2 个与 fp-rule-002 不同" in mine["error"]
    else:
        assert (mine["dev_blocks"], mine["score"], mine["equivalent"]["time_ratio"]) == ({"0": 1.5, "1": -0.5}, 0.5, 0.25)
    assert not E.running("rule")


def test_rl_attempts_choose_their_hands_and_may_continue_the_best(harness, monkeypatch):
    import shutil

    E, rl, line = harness
    monkeypatch.setattr(line, "spawn", lambda attempt_id: None)
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text(json.dumps({"start_actor": "/start"}))
    E.seal()
    record(E, id=rl.start_id(E.load_pool()), finished_at=1, games=16)
    record(E, id="fp-rl-003", finished_at=2, games=16, spec={"parameters": {"weights": "w3"}},
           training={"hands": 1_000_000, "budget": 1_000_000, "lineage_hands": 1_000_000})
    shutil.copytree(E.SOURCES["rl"], rl.RL / "attempts" / "fp-rl-003" / "recipe")
    E.atomic_json(E.LOCAL / "workdir-rl.json", {"base": "fp-rl-003"})
    start = lambda host, **kw: line.start("rl", "x", "y", 0, host, **kw)  # noqa: E731
    with pytest.raises(E.Refused, match="完全相同"):
        start("4070")
    with pytest.raises(E.Refused, match="训练局数"):
        start("4070", hands=rl.MAX_HANDS + 1)
    start("4070", start_from="best")  # the same recipe, trained on from the best version
    start("4090", hands=3_000_000)  # the same recipe for more hands
    states = {s["host"]: s for s in E.running("rl")}
    assert (states["4070"]["from"], states["4070"]["hands"]) == ("best", rl.HANDS)
    assert (states["4090"]["from"], states["4090"]["hands"]) == ("start", 3_000_000)

    class Host:
        def sync(self, paths):
            self.synced = paths

        def remote_path(self, path):
            return "/host" + str(path)

    monkeypatch.setattr(E, "remote_host", lambda name: Host())
    path, behind = rl.start_actor(states["4070"], "4070", {"start_actor": "/start"})
    assert path.endswith("/w3") and behind == 1_000_000  # training adds up along the line
    assert rl.start_actor(states["4090"], "4090", {"start_actor": "/start"}) == ("/start", 0)


def test_a_host_lent_until_a_time_closes_then_and_its_attempt_moves(harness, monkeypatch):
    E, rl, line = harness
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    record(E, id="fp-rl-000")
    monkeypatch.setattr(line, "interrupt", lambda state: None)
    with pytest.raises(E.Refused):
        line.hosts("off", "5070ti", at="2000-01-01 09:30")
    line.hosts("off", "5070ti", at=time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 3600)))
    E.reserve_attempt("rl", "5070ti", "a direction", stage="training")
    assert line.enabled("5070ti") and line.closing("5070ti") and line.close_due_hosts() == []
    values = json.loads((E.LOCAL / "hosts.json").read_text())
    values["5070ti"]["off_at"] = time.time() - 1  # the lending time is over
    (E.LOCAL / "hosts.json").write_text(json.dumps(values))
    notices = line.close_due_hosts()
    assert len(notices) == 1 and "排队到其他空闲主机重做训练" in notices[0]
    assert not line.enabled("5070ti") and line.close_due_hosts() == []
    assert E.running("rl")[0]["stage"] == "queued"  # the direction goes on elsewhere
    assert line.design_hosts("rl") == (["4090"], None)  # the redo takes the first free host
    line.hosts("on", "5070ti")
    assert line.enabled("5070ti") and not line.closing("5070ti")


def test_a_pool_member_runs_its_registered_equivalent_implementation(harness, monkeypatch, tmp_path):
    E, _, _ = harness
    equivalence = importlib.import_module("harness.equivalence")
    member = {"family": "code", "package_path": "/frozen", "package_hash": "f"}
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {"piao": member, "m": {"family": "model"}}}))
    package = tmp_path / "fast"
    package.mkdir()
    (package / "strategy.py").write_text("x = 1\n")
    result = {"hands": 64, "decisions": 700, "mismatches": 1, "seconds": [4.0, 1.0], "examples": ["a"]}
    monkeypatch.setattr(equivalence, "check", lambda *args, **kwargs: result)
    with pytest.raises(E.Refused, match="不同"):
        E.implement("piao", package)
    assert E.playing(E.load_pool())["piao"] == member
    result["mismatches"] = 0
    E.implement("piao", package)
    pool = E.load_pool()
    assert pool["id"] == "pool-a" and pool["members"]["piao"] == member  # the pool's identity stays
    running = E.playing(pool)["piao"]
    assert running["package_path"] != "/frozen" and running["package_hash"] == E.tree_hash(running["package_path"])
    with pytest.raises(E.Refused):
        E.implement("m", package)


def test_a_new_pool_takes_the_replayed_versions_and_their_best(harness, monkeypatch):
    E, rl, line = harness
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    record(E, id="fp-rule-005", line="rule", pool="old", score=4.0, games=16, finished_at=1, spec={"package_hash": "a"})
    record(E, id="fp-rule-009", line="rule", pool="old", score=5.0, games=16, finished_at=2, spec={"package_hash": "b"})
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-b", "members": {}, "dev_seed": 3}))
    E.seal()
    E.set_drain("换对手池")
    assert line.design_hosts("rule") == ([], "换对手池：在途尝试照常跑完，之后暂不开新尝试，等维护者完成")
    monkeypatch.setattr(E, "DEV_BLOCKS", 2)
    new = {"fp-rule-005": 1.0, "fp-rule-009": 0.5}  # the old order need not hold against new opponents
    monkeypatch.setattr(E, "play", lambda pool, cid, spec, blocks, seed, label, host: (
        {b: new[cid] + b for b in blocks}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}))
    assert E.rebase(["fp-rule-005", "fp-rule-009"]) == ["fp-rule-005", "fp-rule-009"]
    assert E.rebase(["fp-rule-005"]) == []  # resumable
    E.rebase_finish("rule")
    best = E.best("pool-b", "rule")
    assert best["id"] == "fp-rule-005" and best["rebased"]["old_score"] == 4.0
    assert {"fp-rule-005", "fp-rule-009"} <= E.reported()
    E.set_drain("换对手池", ["rl"])  # the rule line's versions are recorded: only the RL line waits
    assert line.design_hosts("rule") == (["local"], None)
    assert line.design_hosts("rl", probe=False)[0] == []
    E.set_drain(None)
    assert line.design_hosts("rule") == (["local"], None)
