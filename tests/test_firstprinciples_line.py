"""Iteration-line loop (ADR-0016): seal, leases, host queue, recipe rules, platform flag."""
import importlib
import hashlib
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
    monkeypatch.setattr(E, "MODELS", tmp_path / "models")
    (tmp_path / "rl").mkdir()
    (tmp_path / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}}))
    monkeypatch.setattr(line, "online", lambda hosts, probe=True, grace=False: {h: True for h in hosts})
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


def test_requeued_attempts_take_free_hosts_before_new_designs(harness, monkeypatch):
    E, rl, line = harness
    monkeypatch.setitem(rl.HOST_SIZE, "4090", "medium")  # hosts of two sizes, as on 2026-10-04
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    record(E, id="fp-rl-000")
    E.reserve_attempt("rl", "4070", "busy", stage="training")
    E.reserve_attempt("rl", None, "small, lost", stage="queued", next="training")
    # A small attempt is redone only on a small host: the free 4090 (another size) stays open.
    assert line.design_hosts("rl") == (["4090"], None)
    E.reserve_attempt("rl", None, "lost on a restart", stage="queued", next="training", size=rl.size("4090"))
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


@pytest.mark.parametrize("new_best_dev, verdict, replayed, pdl", [
    (2.0, "discard", [], []), (0.4, "keep", ["fp-rl-008"], []), (0.7, "keep", [], ["fp-rl-008"])])
def test_new_best_during_confirmation_is_first_compared_on_dev(harness, monkeypatch, new_best_dev, verdict, replayed,
                                                                 pdl):
    E, _, _ = harness
    P = importlib.import_module("harness.pdl")
    measured = []
    # Both ways round: forward (the attempt sits) +0.5, reverse (the best sits) -0.5.
    monkeypatch.setattr(P, "measure", lambda a, b, **kw: measured.append(a) or dict(
        seed=kw["seed"], games=8, k=2, diff=0.5 if b == "fp-rl-009" else -0.5, se=0.1, lower90=0.37, parts={},
        snapshots=3, seconds=1.0))
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
    assert measured == [x for a in pdl for x in (a, "fp-rl-009")]


@pytest.mark.parametrize("dev, pdl_diff, verdict", [
    (0.2, 0.3, "keep"), (-0.3, 0.05, "unconfirmed"), (0.4, -0.1, "discard"), (-0.6, None, "discard")])
def test_a_close_development_difference_is_judged_by_pdl(harness, monkeypatch, dev, pdl_diff, verdict):
    E, _, _ = harness
    P = importlib.import_module("harness.pdl")
    monkeypatch.setattr(E, "DEV_BLOCKS", 4)
    monkeypatch.setattr(E, "CONFIRM_BLOCKS", 2)
    monkeypatch.setattr(E, "check_seal", lambda: None)
    monkeypatch.setattr(E, "platform_entrant", lambda: None)
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}, "dev_seed": 1}))
    record(E, id="fp-rl-007", finished_at=1, games=16, spec={"weights": "old"}, dev_blocks={str(b): 0.0 for b in range(4)})
    calls, measured = [], []

    def play(pool, attempt_id, spec, blocks, seed, label, host):
        calls.append(label)
        return {b: dev if attempt_id == "fp-rl-009" else 0.0 for b in blocks}, {"timeouts": 0, "fallback_actions": 0,
                                                                                "errors": 0}

    def measure(a, b, **kw):
        measured.append((a, b, kw["a_spec"], kw["b_spec"], kw["seed"], kw["host"]))
        d = pdl_diff if b == "fp-rl-009" else -pdl_diff  # V(b) - V(a)
        return dict(seed=kw["seed"], games=8, k=2, diff=d, se=0.1, lower90=d - E.CONFIRM_Z * 0.1,
                    parts={"hu": {"diff": d, "se": 0.1}}, snapshots=3, seconds=1.0)

    monkeypatch.setattr(E, "play", play)
    monkeypatch.setattr(P, "measure", measure)
    r = E.evaluate("rl", {"weights": "mine"}, "x", "fp-rl-009", host="4070", confirm_seed=5)
    assert r["status"] == verdict and calls == ["dev"] and r["confirm"] is None
    if pdl_diff is None:
        assert measured == [] and r["pdl"] is None
    else:
        # Both ways round on the attempt's host; the reverse on its own seed.
        assert measured == [("fp-rl-007", "fp-rl-009", {"weights": "old"}, {"weights": "mine"}, 5, "4070"),
                            ("fp-rl-009", "fp-rl-007", {"weights": "mine"}, {"weights": "old"}, 4, "4070")]
        assert r["pdl"]["against"] == "fp-rl-007" and r["pdl"]["diff"] == pytest.approx(pdl_diff)
        assert r["pdl"]["se"] == pytest.approx(0.1 / 2 ** 0.5)
        assert f"差 {pdl_diff:+.2f} ± 0.07（正向 {pdl_diff:+.2f}、反向 {pdl_diff:+.2f} 的平均）" in E.report(r)


def test_an_attempt_judged_on_the_mac_lets_its_host_go(harness, monkeypatch):
    E, rl, line = harness
    (rl.RL / "setup-4070.json").write_text(json.dumps({"source_id": "a"}))
    attempt_id = E.reserve_attempt("rl", "4070", "x", stage="evaluating")
    assert "4070" not in line.free_hosts("rl")
    E.free_host(attempt_id)
    assert "4070" in line.free_hosts("rl") and E.running()[0]["host"] == "4070"


def test_rejudge_extends_an_earlier_attempt_and_judges_it_now(harness, monkeypatch):
    E, _, _ = harness
    P = importlib.import_module("harness.pdl")
    monkeypatch.setattr(E, "DEV_BLOCKS", 4)
    monkeypatch.setattr(E, "check_seal", lambda: None)
    monkeypatch.setattr(E, "platform_entrant", lambda: None)
    monkeypatch.setattr(E, "proxy_tables", lambda *a: {"error": "none"})
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}, "dev_seed": 1}))
    record(E, id="fp-rl-007", finished_at=1, games=16, spec={"weights": "old"}, dev_blocks={str(b): 0.0 for b in range(4)})
    record(E, id="fp-rl-009", status="discard", host="5070ti", finished_at=2, games=8, score=-0.1, diff=-0.1,
           spec={"weights": "mine"}, dev_blocks={"0": -0.1, "1": -0.1}, stats={"timeouts": 0}, training={"hands": 5})
    calls, measured = [], []

    def play(pool, attempt_id, spec, blocks, seed, label, host):
        calls.append((attempt_id, list(blocks), label, host))
        return {b: 0.3 for b in blocks}, {"timeouts": 1}

    monkeypatch.setattr(E, "play", play)
    monkeypatch.setattr(P, "measure", lambda a, b, **kw: measured.append((a, b, kw["seed"])) or dict(
        seed=kw["seed"], games=8, k=2, diff=0.3 if b == "fp-rl-009" else -0.3, se=0.1, lower90=0.17, parts={},
        snapshots=3, seconds=1.0))
    r = E.rejudge("fp-rl-009", seed=11)
    assert calls == [("fp-rl-009", [2, 3], "dev-extend", "5070ti")]
    assert measured == [("fp-rl-007", "fp-rl-009", 11), ("fp-rl-009", "fp-rl-007", 10)]
    assert r["status"] == "keep" and r["games"] == 16 and r["stats"] == {"timeouts": 1} and r["training"] == {"hands": 5}
    assert r["rejudged"]["old_status"] == "discard" and r["pdl"]["against"] == "fp-rl-007"
    assert E.best("pool-a", "rl")["id"] == "fp-rl-009" and "按 10-07 规则重判" in E.history("rl")
    with pytest.raises(E.Refused):
        E.rejudge("fp-rl-009")


def test_a_version_judged_again_counts_by_its_latest_record(harness):
    E, _, _ = harness
    for i, (aid, status) in enumerate((("fp-rl-007", "keep"), ("fp-rl-008", "keep"), ("fp-rl-008", "unconfirmed"))):
        record(E, id=aid, status=status, finished_at=i)
    assert E.best("pool-a", "rl")["id"] == "fp-rl-007"
    record(E, id="fp-rl-007", status="keep", finished_at=9, games=32)  # extended: still the best
    assert E.best("pool-a", "rl")["games"] == 32


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


def test_registered_training_rules_are_offered_but_never_evaluated_against(harness, tmp_path):
    E, rl, _ = harness
    pool = {"id": "pool-a", "members": {"piao": {"family": "code"}}, "runtime_path": "/pool-runtime"}
    (E.LOCAL / "pool.json").write_text(json.dumps(pool))
    package = tmp_path / "fp68-classic"
    package.mkdir()
    (package / "strategy.py").write_text('"""Header.\n\nClassic: fp-rule-068 as it is.\n"""\n')
    assert rl.register_training_rules([package], {"fp68-classic": {"score": 5.7, "se": 0.6}}) == 1
    offered = rl.catalog()
    spec, entry = offered["fp68-classic"]["spec"], offered["fp68-classic"]["record"]
    assert spec["runtime_path"] == "/pool-runtime" and spec["family"] == "code"
    assert Path(spec["package_path"]).parent.parent == E.LOCAL / "training-rules"  # a frozen copy
    assert (entry["line"], entry["score"], entry["description"]) == ("training", 5.7, "Classic: fp-rule-068 as it is.")
    assert "fp68-classic" not in E.load_pool()["members"]  # evaluation stays on the fixed pool


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
    monkeypatch.setitem(rl.HOST_SIZE, "4090", "medium")  # hosts of two sizes, as on 2026-10-04
    monkeypatch.setattr(line, "spawn", lambda attempt_id: None)
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text(json.dumps({"start_actor": "/start"}))
    E.seal()
    record(E, id=rl.start_id(E.load_pool()), finished_at=1, games=16)
    for weights, size in (("w3", "small"), ("w2", "small")):
        (E.MODELS / weights).mkdir(parents=True)
        (E.MODELS / weights / "meta.json").write_text(json.dumps({"config": rl.SIZES[size]}))
    record(E, id="fp-rl-003", finished_at=2, games=16, spec={"parameters": {"weights": "w3"}},
           training={"hands": 1_000_000, "budget": 1_000_000, "lineage_hands": 1_000_000})
    shutil.copytree(E.SOURCES["rl"], rl.RL / "attempts" / "fp-rl-003" / "recipe")
    E.atomic_json(E.LOCAL / "workdir-rl.json", {"base": "fp-rl-003"})
    start = lambda host, **kw: line.start("rl", "x", "y", 0, host, **kw)  # noqa: E731
    with pytest.raises(E.Refused, match="完全相同"):
        start("4070")
    with pytest.raises(E.Refused, match="每次尝试最多"):
        start("4070", hands=rl.MAX_HANDS + 1)
    with pytest.raises(E.Refused, match="同规格"):
        start("4090", start_from="best")  # the best is a small Actor; the 4090 trains the large size
    record(E, id="fp-rl-002", status="discard", finished_at=1.5, games=16, spec={"family": "model", "parameters": {"weights": "w2"}},
           training={"hands": 500_000, "budget": 500_000, "lineage_hands": 500_000})
    with pytest.raises(E.Refused, match="不是评估过的 RL 版本"):
        start("5070ti", start_from="fp-rl-999")
    start("4070", start_from="best")  # the same recipe, trained on from the best version
    start("4090", hands=3_000_000)  # the same recipe for more hands
    start("5070ti", start_from="fp-rl-002")  # from any evaluated version, kept or not (e.g. an imitation)
    states = {s["host"]: s for s in E.running("rl")}
    assert (states["4070"]["from"], states["4070"]["hands"]) == ("best", rl.HANDS)
    assert (states["4090"]["from"], states["4090"]["hands"]) == ("start", 3_000_000)
    assert states["5070ti"]["from"] == "fp-rl-002"

    class Host:
        def sync(self, paths):
            self.synced = paths

        def remote_path(self, path):
            return "/host" + str(path)

    monkeypatch.setattr(E, "remote_host", lambda name: Host())
    path, behind = rl.start_actor(states["4070"], "4070", {"start_actor": "/start"})
    assert path.endswith("/w3") and behind == 1_000_000  # training adds up along the line
    assert rl.start_actor(states["4090"], "4090", {"start_actor": "/start"}) == ("/start", 0)
    path, behind = rl.start_actor(states["5070ti"], "5070ti", {"start_actor": "/start"})
    assert path.endswith("/w2") and behind == 500_000


def test_any_two_scored_versions_compare_on_the_development_set(harness, monkeypatch):
    E, rl, line = harness
    monkeypatch.setattr(E, "DEV_BLOCKS", 3)
    record(E, id="fp-rl-001", dev_blocks={"0": 1.0, "1": 2.0, "2": 3.0})
    record(E, id="fp-rl-002", dev_blocks={"0": 0.0, "1": 2.0, "2": 1.0})
    record(E, id="fp-rl-003", dev_blocks={"0": 0.0})
    diff, se, games = E.compare("fp-rl-001", "fp-rl-002")
    assert (round(diff, 6), games) == (1.0, 12) and se > 0
    with pytest.raises(E.Refused, match="fp-rl-003"):
        E.compare("fp-rl-001", "fp-rl-003")


def test_a_host_lent_until_a_time_closes_then_and_its_attempt_moves(harness, monkeypatch):
    E, rl, line = harness
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    record(E, id="fp-rl-000")
    monkeypatch.setattr(line, "stop_driver", lambda state: None)
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
    # A new framework's version the rule line grows from, not a best version.
    record(E, id="fp-rule-011", line="rule", pool="old", status="discard", score=-9.0, games=16, finished_at=3,
           spec={"package_hash": "c"})
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-b", "members": {}, "dev_seed": 3}))
    E.seal()
    E.set_drain("换对手池")
    assert line.design_hosts("rule") == ([], "换对手池：在途尝试照常跑完，之后暂不开新尝试，等维护者完成")
    monkeypatch.setattr(E, "DEV_BLOCKS", 2)
    new = {"fp-rule-005": 1.0, "fp-rule-009": 0.5, "fp-rule-011": 2.0}  # the old order need not hold
    monkeypatch.setattr(E, "play", lambda pool, cid, spec, blocks, seed, label, host: (
        {b: new[cid] + b for b in blocks}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}))
    assert E.rebase(["fp-rule-005", "fp-rule-009", "fp-rule-011"]) == ["fp-rule-005", "fp-rule-009", "fp-rule-011"]
    assert E.rebase(["fp-rule-005"]) == []  # resumable
    E.rebase_finish("rule")
    best = E.best("pool-b", "rule")
    assert best["id"] == "fp-rule-005" and best["rebased"]["old_score"] == 4.0
    # The replayed framework version keeps its status: scored on the new pool, never the best.
    assert [r["status"] for r in E.attempts("pool-b", "rule") if r["id"] == "fp-rule-011"] == ["discard"]
    assert {E.report_key(r) for r in E.attempts("pool-b", "rule")} <= E.reported()
    E.set_drain("换对手池", ["rl"])  # the rule line's versions are recorded: only the RL line waits
    assert line.design_hosts("rule") == (["local"], None)
    assert line.design_hosts("rl", probe=False)[0] == []
    E.set_drain(None)
    assert line.design_hosts("rule") == (["local"], None)


def test_equivalent_code_replays_a_development_set_its_best_version_timed_out_on(harness, monkeypatch):
    E, _, line = harness
    equivalence = importlib.import_module("harness.equivalence")
    monkeypatch.setattr(E, "platform_entrant", lambda: None)
    monkeypatch.setattr(E, "DEV_BLOCKS", 2)
    monkeypatch.setattr(line, "decision_ms", lambda state: {"mine": 2.0, "base": 8.0})
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}, "dev_seed": 3}))
    record(E, id="fp-rule-002", line="rule", games=8, finished_at=1, stats={"timeouts": 3, "fallback_actions": 3},
           spec={"package_path": "/best"}, dev_blocks={"0": 1.5, "1": -0.5}, score=0.5)
    monkeypatch.setattr(equivalence, "check", lambda *args: {"hands": 64, "decisions": 700, "mismatches": 0,
                                                             "seconds": [8.0, 2.0], "examples": []})
    played = []
    monkeypatch.setattr(E, "play", lambda pool, cid, spec, blocks, seed, label, host: (
        played.append((cid, label)) or {b: 2.0 for b in blocks}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}))
    state = dict(id="fp-rule-003", line="rule", host="local", description="剪枝", why="更快", base="fp-rule-002",
                 started_at=2, design_seconds=60, stage="equivalence", confirm_seed=7,
                 spec={"package_path": "/new", "package_hash": "n"})
    line.save(state)
    line.equivalent(state)
    mine = E.best("pool-a", "rule")
    assert played == [("fp-rule-003", "dev")] and mine["id"] == "fp-rule-003"
    assert (mine["dev_blocks"], mine["score"], mine["diff"], mine["stats"]["timeouts"]) == ({"0": 2.0, "1": 2.0}, 2.0, 1.5, 0)
    assert not mine["equivalent"]["carried"]


def test_a_best_version_replays_the_development_set_under_the_current_conditions(harness, monkeypatch):
    E, _, _ = harness
    monkeypatch.setattr(E, "DEV_BLOCKS", 2)
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}, "dev_seed": 3}))
    E.seal()
    record(E, id="fp-rule-002", line="rule", games=8, finished_at=5, host="local", stats={"timeouts": 3},
           spec={"package_hash": "b"}, dev_blocks={"0": 1.0, "1": 0.0}, score=0.5)
    record(E, id="fp-rule-003", line="rule", status="unconfirmed", games=8, finished_at=6, dev_blocks={"0": 2.0, "1": 1.0})
    labels = []
    monkeypatch.setattr(E, "play", lambda pool, cid, spec, blocks, seed, label, host: (
        labels.append((cid, label, host)) or {b: 1.5 + b for b in blocks}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}))
    r = E.refresh("rule")
    assert labels == [("fp-rule-002", "dev-refresh1", "local")]
    best = E.best("pool-a", "rule")
    assert best["id"] == "fp-rule-002" and best["finished_at"] == 5 and best["dev_blocks"] == {"0": 1.5, "1": 2.5}
    assert (r["refreshed"]["old_score"], r["stats"]["timeouts"], r["score"]) == (0.5, 0, 2.0)
    assert round(E.compare("fp-rule-003", "fp-rule-002")[0], 6) == -0.5  # was +1.0 against the stale blocks
    E.refresh("rule")
    assert labels[-1][1] == "dev-refresh2"  # a later refresh never reuses the earlier one's jobs


def test_a_packages_own_c_may_not_start_processes_threads_or_python_code(harness, tmp_path):
    E, _, _ = harness
    (tmp_path / "strategy.py").write_text("from mahjong_ai import _fast\nnative = _fast.load_native(__file__)\n")
    (tmp_path / "native.c").write_text('#include "_fast.c"\nstatic int f(void) { return shanten_raw(0, 0); }\n')
    assert E.check_source(tmp_path) == []
    (tmp_path / "util.h").write_text("static void g(void) { pthread_create(0, 0, 0, 0); system(\"x\"); }\n"
                                     "#pragma omp parallel\nstatic void h(void) { PyRun_SimpleString(\"x\"); }\n")
    assert E.check_source(tmp_path) == ["util.h:1 用了 pthread_create", "util.h:1 用了 system", "util.h:2 用了 #pragma omp",
                                        "util.h:3 用了 PyRun_SimpleString"]


def test_only_the_race_section_of_the_pool_runtime_may_change(harness):
    E, _, _ = harness
    banner, table = E.RACE_SECTION
    old = f"a\n{banner}\nr1\nr2\n{table}\nm\n"
    assert E.runtime_changes(old, old) is None
    assert E.runtime_changes(old, old.replace("a\n", "a\nnew\n")) == ([], [])
    assert E.runtime_changes(old, old.replace("r2", "R2")) == ([], [("replace", 4, 4)])
    assert E.runtime_changes(old, old.replace("m\n", "")) == ([("delete", 6, 6)], [])
    assert E.runtime_changes(old, old.replace(banner, "x"))[0]  # the banner itself stays
    assert E.runtime_changes("a\nb\n", "a\nB\n") == ([("replace", 2, 2)], [])  # no race section: nothing may change


@pytest.mark.parametrize("new, refused", [("a\nB\nc\n", "竞速部分以外的原有代码"), ("a\nb\nc\n", "没有可升级的内容")])
def test_the_pool_runtime_takes_only_added_functions(harness, monkeypatch, tmp_path, new, refused):
    E, _, _ = harness
    old = tmp_path / "old-runtime"
    (old / "mahjong_ai").mkdir(parents=True)
    (old / "mahjong_ai" / "_fast.c").write_text("a\nb\nc\n")
    repo = tmp_path / "repo"
    (repo / "src" / "mahjong_ai").mkdir(parents=True)
    (repo / "src" / "mahjong_ai" / "_fast.c").write_text(new)
    monkeypatch.setattr(E, "REPO", repo)
    (E.LOCAL / "pool.json").write_text(json.dumps({"id": "pool-a", "members": {}, "runtime_path": str(old)}))
    E.seal()
    with pytest.raises(E.Refused, match=refused):
        E.upgrade_runtime()


def test_a_new_rule_framework_grows_from_its_own_versions(harness, monkeypatch, tmp_path):
    E, _, line = harness
    monkeypatch.setattr(E, "SOURCES", dict(E.SOURCES, rule=tmp_path / "rule-src"))
    monkeypatch.setattr(line, "commit", lambda rel, message: None)
    (tmp_path / "rule-src").mkdir()
    for number, (status, text) in enumerate((("keep", "best"), ("discard", "new framework")), 1):
        package = tmp_path / f"package-{number}"
        package.mkdir()
        (package / "strategy.py").write_text(text)
        record(E, id=f"fp-rule-00{number}", line="rule", status=status, finished_at=number, games=16,
               spec={"package_path": str(package)})
    E.seal()
    monkeypatch.setattr(E, "REPO", tmp_path)  # rule/ lives in the test directory
    line.restore("rule")
    start = lambda **kw: line.start("rule", "x", "y", 0, **kw)  # noqa: E731
    with pytest.raises(E.Refused, match="checkout rule fp-rule-002"):
        start(start_from="fp-rule-002")  # rule/ holds the best version, not the new framework
    line.checkout("rule", "fp-rule-002")  # discarded, still a version to grow from
    assert (tmp_path / "rule-src" / "strategy.py").read_text() == "new framework"
    with pytest.raises(E.Refused, match="不是从它恢复来的"):
        start()  # without --from an attempt is one change from the best version
    monkeypatch.setattr(line, "git", lambda *args: type("Status", (), {"stdout": " M strategy.py"})())
    monkeypatch.setattr(E, "check_source", lambda source: ["reached the source checks"])
    with pytest.raises(E.Refused, match="reached the source checks"):
        start(start_from="fp-rule-002")  # a changed rule/ checked out from a version passes the base checks
    with pytest.raises(E.Refused, match="只用于规则线"):
        line.checkout("rl", "fp-rl-001")


def test_a_scale_run_chains_its_segments_on_a_host_it_holds(harness, monkeypatch, tmp_path):
    import shutil

    E, rl, line = harness
    scale = importlib.import_module("harness.scale")
    recipe = tmp_path / "scale-recipe"
    shutil.copytree(scale.RECIPE, recipe, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(scale, "RECIPE", recipe)
    monkeypatch.setattr(scale, "SCALE", E.LOCAL / "scale")
    monkeypatch.setattr(line, "spawn", lambda attempt_id: None)
    monkeypatch.setitem(rl.HOST_SIZE, "4090", "large")  # the scale run (ended 2026-10-03) ran the large size there
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    E.seal()
    record(E, id=rl.start_id(E.load_pool()), finished_at=1, games=16)
    with pytest.raises(E.Refused, match="大规格"):
        scale.begin("4070")
    first = scale.begin("4090")
    state = E.running("rl")[0]
    assert (state["id"], state["host"], state["from"], state["stage"]) == (first, "4090", "start", "training")
    assert state["hands"] == 2_500_000 and state["scale"] == {"index": 0, "host": "4090"}
    assert json.loads((rl.RL / "attempts" / first / "recipe" / "segment.json").read_text()) == {"index": 0}
    assert "第 1/12 段" in state["description"] and "教师代打" in state["description"]
    # The segment is recorded with its Actor: the host stays the run's, switched on or not.
    (E.LOCAL / "running" / f"{first}.json").unlink()
    record(E, id=first, status="discard", finished_at=2, games=16, spec={"parameters": {"weights": "w1"}})
    assert "4090" not in line.free_hosts("rl") and line.host_states("rl")[0]["4090"] == "规模训练专用"
    second = scale.start(1)
    state = E.running("rl")[0]
    assert (state["id"], state["from"], state["scale"]["index"]) == (second, first, 1)
    # A segment the host lost is redone only on the run's host, even switched off for the line.
    line.hosts("off", "4090")
    line.requeue(state, "training")
    assert line.design_hosts("rl") == (["4070", "5070ti"], None)
    assert line.claim_host(line.load(second))["host"] == "4090"
    # The plan is frozen: a changed recipe starts no further segment.
    (E.LOCAL / "running" / f"{second}.json").unlink()
    record(E, id=second, status="crash", finished_at=3, games=0)
    (recipe / "config.json").write_text((recipe / "config.json").read_text().replace("0.015", "0.02"))
    with pytest.raises(E.Refused, match="改动"):
        scale.start(2)
    # A segment without an Actor stops the chain, the run keeping its host; redone, it takes its place.
    scale.chain()
    assert not E.running("rl") and "4090" in scale.reserved_hosts()
    (recipe / "config.json").write_text((recipe / "config.json").read_text().replace("0.02", "0.015"))
    third = scale.start(1)
    assert scale.run_state()["segments"] == [first, third] and E.running("rl")[0]["from"] == first


def test_a_lent_host_runs_the_rule_line_after_the_attempt_it_has(harness):
    E, rl, line = harness
    for host in rl.HOSTS:
        (rl.RL / f"setup-{host}.json").write_text("{}")
    record(E, id="fp-rl-000")
    record(E, id="fp-rule-000", line="rule")
    E.reserve_attempt("rl", "4070", "still training", stage="training")
    line.hosts("lend", "4070", line="rule")
    assert line.line_hosts("rule") == ("local", "4070")
    assert "4070" not in line.line_hosts("rl")
    # The RL attempt keeps the host until it finishes; meanwhile the rule line has the Mac.
    assert line.free_hosts("rule") == ["local"]
    E.release_attempt(next(s["id"] for s in E.running("rl")))
    assert line.free_hosts("rule") == ["local", "4070"]
    line.hosts("off", "4070")
    assert line.free_hosts("rule") == ["local"] and line.line_hosts("rule") == ("local", "4070")
    line.hosts("on", "4070")
    line.hosts("lend", "4070", line="rl")
    assert line.line_hosts("rule") == ("local",) and "4070" in line.line_hosts("rl")


def proxy_panel(E, proxy, tmp_path, monkeypatch, reference="fp-rule-131"):
    monkeypatch.setattr(proxy, "BLOCKS", 40)
    monkeypatch.setattr(E, "DEV_BLOCKS", 2)
    monkeypatch.setattr(E, "CONFIRM_BLOCKS", 2)
    monkeypatch.setattr(E, "platform_entrant", lambda: None)
    pool = {"id": "pool-a", "members": {}, "dev_seed": 3, "runtime_path": "/rt", "config": {"rounds": 16},
            "timeouts": {}, "resources": {}}
    (E.LOCAL / "pool.json").write_text(json.dumps(pool))
    record(E, id="fp-rule-131", line="rule", games=8, finished_at=1, spec={"family": "code", "package_hash": "r"},
           dev_blocks={"0": 0.0, "1": 1.0})
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"name": "v2", "members": {
        "p01": {"weights": "w1", "share": 0.3}, "p02": {"weights": "w2", "share": 0.2},
        "tail": {"weights": "wt", "share": 0.5}}}))
    E.seal()
    proxy.activate(panel, reference)
    with pytest.raises(E.Refused):
        E.check_seal()  # the active panel is a sealed condition
    E.seal()
    return proxy.load()


def test_a_new_best_version_plays_the_proxy_tables_for_the_record_only(harness, monkeypatch, tmp_path):
    E, _, _ = harness
    proxy = importlib.import_module("harness.proxy")
    tables = proxy_panel(E, proxy, tmp_path, monkeypatch)
    for seats in tables["lineups"]:
        named = [s for s in seats if not s.startswith("tail")]
        assert len(seats) == 3 and len(set(named)) == len(named)
        assert [s for s in seats if s.startswith("tail")] == list(proxy.TAIL_SEATS[:3 - len(named)])
    pool_score = {"fp-rule-131": 0.0, "fp-rule-132": 1.0, "fp-rule-133": 0.5, "fp-rule-134": 2.0,
                  "fp-rl-140": 0.0, "fp-rl-141": 1.0}
    proxy_score = {"fp-rule-131": 0.0, "fp-rule-132": 1.0, "fp-rule-134": -3.0, "fp-rl-140": 0.5}
    calls = []

    def play(pool, attempt_id, spec, blocks, seed, label, host):
        calls.append((pool["id"], attempt_id, label))
        if label.startswith("proxy"):
            assert E.lineups(pool) == tables["lineups"] and pool["members"]["tail-2"]["parameters"] == {"weights": "wt"}
        score = (proxy_score if label.startswith("proxy") else pool_score)[attempt_id]
        return {b: score + (b % 2) for b in blocks}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}

    def proxied():
        return [c[1:] for c in calls if c[0] == tables["id"]]

    monkeypatch.setattr(E, "play", play)
    # A new best version (strong opponents decide) plays the proxy tables, paired with the one it replaces.
    r = E.evaluate("rule", {"family": "code", "package_hash": "n"}, "x", "fp-rule-132", host="local", confirm_seed=5)
    assert r["status"] == "keep" and r["proxy"]["best"] == {"version": "fp-rule-131", "diff": 1.0, "se": 0.0}
    assert r["proxy"]["against"]["diff"] == 1.0 and "combined" not in r["confirm"]
    assert proxied() == [("fp-rule-131", "proxy-local"), ("fp-rule-132", "proxy-local")]  # the best is the reference
    assert "对 fp-rule-131 配对差 +1.00" in E.report(r)
    # Below the best on the development set: no proxy tables at all.
    calls.clear()
    r = E.evaluate("rule", {"family": "code", "package_hash": "o"}, "x", "fp-rule-133", host="local", confirm_seed=5)
    assert r["status"] == "discard" and "proxy" not in r and proxied() == []
    # Far behind at the proxy tables does not undo a confirmed win against the pool.
    r = E.evaluate("rule", {"family": "code", "package_hash": "p"}, "x", "fp-rule-134", host="local", confirm_seed=5)
    assert r["status"] == "keep" and r["proxy"]["best"]["diff"] == -4.0
    assert proxied() == [("fp-rule-134", "proxy-local"), ("fp-rule-132", "proxy-local")]
    # A line's first version is on record at the proxy tables; another host plays the reference again.
    calls.clear()
    r = E.evaluate("rl", {"family": "model", "parameters": {"weights": "m"}}, "x", "fp-rl-140", host="4070",
                   confirm_seed=5)
    assert r["status"] == "keep" and r["proxy"]["against"]["diff"] == 0.5 and "best" not in r["proxy"]
    assert proxied() == [("fp-rule-131", "proxy-4070"), ("fp-rl-140", "proxy-4070")]
    rows = E.read_json(E.LOCAL / "proxy-results.json")
    assert [(x["version"], x["host"]) for x in rows["results"]] == [("fp-rule-131", "local"), ("fp-rule-131", "4070")]
    assert rows["panels"][tables["panel"]]["proxies"] == 3
    # A failure at the proxy tables is recorded in their place.
    monkeypatch.setattr(E, "play", lambda pool, cid, spec, blocks, seed, label, host: (_ for _ in ()).throw(
        RuntimeError("down")) if label.startswith("proxy") else play(pool, cid, spec, blocks, seed, label, host))
    r = E.evaluate("rl", {"family": "model", "parameters": {"weights": "m2"}}, "x", "fp-rl-141", host="5070ti",
                   confirm_seed=5)
    assert "down" in r["proxy"]["error"] and r["status"] == "keep"


def test_the_qualifier_version_may_lose_a_little_to_strong_opponents_to_win_more_against_the_crowd(
        harness, monkeypatch, tmp_path):
    E, _, _ = harness
    proxy = importlib.import_module("harness.proxy")
    Q = importlib.import_module("harness.qualifier")
    tables = proxy_panel(E, proxy, tmp_path, monkeypatch)
    record(E, id="fp-rl-172", line="rl", games=8, finished_at=2, score=1.5, spec={"family": "model"},
           dev_blocks={"0": 1.0, "1": 2.0})
    # Per candidate: fixed proxy tables, development set, fresh proxy walls.
    scores = {"fp-rule-131": (0.0, None, 0.0),
              "fp-q1-001": (0.0, None, None),   # not ahead at the proxy tables
              "fp-q1-002": (2.0, -1.5, None),   # 2.5 below the lines' best (fp-rl-172) against strong opponents
              "fp-q1-003": (2.0, 0.0, 2.0),     # 1 below it, clearly ahead on fresh walls: the qualifier version
              "fp-q1-004": (3.0, 1.0, 2.0),     # ahead of fp-q1-003 at the fixed tables only
              "fp-q2-001": (1.0, 1.0, 1.0)}     # the other qualifier line starts from the reference too
    walls = []

    def play(pool, attempt_id, spec, blocks, seed, label, host):
        assert host == "4070"
        if pool["id"] == tables["id"]:
            score = scores[attempt_id][0]
        elif pool["id"].startswith(tables["id"] + "-fresh-"):
            assert pool["seed"] == seed and len(pool["lineups"]) == tables["blocks"]
            assert pool["lineups"] != tables["lineups"]
            walls.append((attempt_id, label, seed))
            score = scores[attempt_id][2]
        else:
            assert pool["id"] == "pool-a" and label == "dev"
            score = scores[attempt_id][1]
        return {b: score + (b % 2) for b in blocks}, {"timeouts": 0, "fallback_actions": 0, "errors": 0}

    monkeypatch.setattr(E, "play", play)
    statuses = []
    for n, line in enumerate(("q1", "q1", "q1", "q1", "q2"), 1):
        actor = tmp_path / f"actor-{n}"
        actor.mkdir()
        (actor / "actor.pt").write_bytes(bytes([n]))
        (actor / "meta.json").write_text(json.dumps({"weights_sha256": hashlib.sha256(bytes([n])).hexdigest()}))
        r = Q.submit(actor, line, "4070", f"候选 {n}")
        statuses.append((r["id"], r["status"], r["against"]))
        assert (E.MODELS / r["spec"]["parameters"]["weights"]).is_dir() and r["spec"]["version"] == r["id"]
    assert statuses == [("fp-q1-001", "discard", "fp-rule-131"), ("fp-q1-002", "pool", "fp-rule-131"),
                        ("fp-q1-003", "keep", "fp-rule-131"), ("fp-q1-004", "unconfirmed", "fp-q1-003"),
                        ("fp-q2-001", "keep", "fp-rule-131")]
    q3, q4 = Q.records("q1")[2:]
    assert q3["pool"]["against"] == "fp-rl-172" and q3["pool"]["diff"] == -1.0 and q3["confirm"]["lower90"] == 2.0
    assert q4["proxy"]["diff"] == 1.0 and q4["confirm"]["diff"] == 0.0
    # Each candidate's fresh walls are new, and the version it is judged against replays them.
    assert [w[:2] for w in walls][:4] == [("fp-q1-003", "fresh"), ("fp-rule-131", "fresh-for-fp-q1-003"),
                                          ("fp-q1-004", "fresh"), ("fp-q1-003", "fresh-for-fp-q1-004")]
    assert walls[0][2] == walls[1][2] != walls[2][2]
    assert "成功（新的海选版本）" in Q.report(q3) and "对 fp-rl-172 -1.00" in Q.report(q3)
    rows = E.read_json(E.LOCAL / "proxy-results.json")["results"]
    assert [x["version"] for x in rows if x["batch"] == "海选候选"] == [s[0] for s in statuses]


def test_each_tail_proxy_may_fill_several_seats_and_named_ones_one(harness):
    proxy = importlib.import_module("harness.proxy")
    shares = {"p01": 0.1, "p02": 0.1, "p03": 0.1, "tailsh": 0.35, "tailwt": 0.35}
    rows = proxy.lineups(shares, 400, 7)
    assert all(len(set(seats)) == 3 for seats in rows)
    assert any(seats.count("tailsh") + seats.count("tailsh-2") == 2 for seats in rows)
    assert {s for seats in rows for s in seats} <= {"p01", "p02", "p03", *proxy.seats_of("tailsh"), *proxy.seats_of("tailwt")}
