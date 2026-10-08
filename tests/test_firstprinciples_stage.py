"""Simulated stages (harness.stage): a recorded version, or a model policy from a spec file under its own name."""
import importlib
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN, ALT, OTHER = "a" * 64, "b" * 64, "c" * 64


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "projects/personal/firstprinciples"))
    E = importlib.import_module("harness.evaluate")
    stage = importlib.import_module("harness.stage")
    monkeypatch.setattr(E, "LOCAL", tmp_path)
    monkeypatch.setattr(E, "MODELS", tmp_path / "models")
    for sha in (MAIN, ALT, OTHER):
        (tmp_path / "models" / sha).mkdir(parents=True)
    pool = {"id": "pool-a", "members": {"m1": {}, "m2": {}, "m3": {}}, "config": {"rounds": 16, "stage": "qualifier"},
            "dev_seed": 5, "runtime_path": str(tmp_path / "runtime")}
    (tmp_path / "pool.json").write_text(json.dumps(pool))
    records = [{"id": v, "pool": "pool-a", "line": "rl", "status": "keep",
                "spec": {"family": "model", "version": v, "parameters": {"weights": w}}}
               for v, w in (("fp-rl-227", MAIN), ("fp-rl-200", ALT))]
    (tmp_path / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    E.seal()
    calls = []

    def play(t, version, spec, blocks, seed, label, host):
        calls.append(dict(t=t, version=version, spec=spec, label=label, host=host))
        stats = {"timeouts": 0, "fallback_actions": 0, "errors": 0,
                 "places": {b: 1 + (b + len(version)) % 4 for b in blocks}}
        if spec["parameters"].get("switch"):
            stats.update(actions=400, alt_actions=100)
        return {b: float(b % 3) for b in blocks}, stats

    monkeypatch.setattr(E, "play", play)
    monkeypatch.setattr(stage, "seat_runtime", lambda t: str(tmp_path / "seat-runtime"))
    return E, stage, calls, tmp_path


def spec_file(tmp_path, name, switch):
    path = tmp_path / f"{name}.json"
    parameters = {"weights": MAIN, **({"switch": switch} if switch else {})}
    path.write_text(json.dumps({"family": "model", "version": "anything", "parameters": parameters}))
    return str(path)


def run(stage, *args):
    stage.main(["run", *args, "--kind", "final", "--host", "4070", "--stages", "8"])


def test_a_spec_plays_under_its_name_against_a_version_and_keeps_its_own_jobs(harness):
    E, stage, calls, tmp = harness
    tail = spec_file(tmp, "tail", {"rule": "tail", "weights": ALT, "hands_left": 4})
    run(stage, "s2-final", "--spec", tail, "--against", "fp-rl-227")
    mine, theirs = calls
    assert mine["version"] == "s2-final" and mine["spec"]["version"] == "s2-final"
    assert mine["spec"]["parameters"]["switch"] == {"rule": "tail", "weights": ALT, "hands_left": 4}
    # A switching seat plays the pool's runtime with the repository's model seat; the version it pairs with, the pool's.
    assert mine["t"]["runtime_path"] == str(tmp / "seat-runtime") and theirs["t"]["runtime_path"] == str(tmp / "runtime")
    assert mine["t"]["id"] == theirs["t"]["id"] == "pool-a-stage-final" and mine["t"]["config"]["stage"] == "final"
    assert theirs["version"] == "fp-rl-227" and theirs["label"] == "stage8-4070"
    assert mine["label"].startswith("stage8-4070-spec") and mine["label"] != theirs["label"]
    record = json.loads(stage.results_path().read_text().splitlines()[-1])
    assert record["version"] == "s2-final" and record["against"] == "fp-rl-227" and record["spec"] == mine["spec"]
    assert record["alt_share"] == 0.25 and record["runtime"] == "seat-runtime" and record["stats"]["alt_actions"] == 100
    assert "第二个模型做了 25.0% 的决策" in stage.report(record)
    # Another spec under the same name, or the same spec under another name, never meets these jobs.
    calls.clear()
    contest = spec_file(tmp, "contest", {"rule": "contest", "weights": ALT})
    run(stage, "s2-final", "--spec", contest, "--against", "fp-rl-227")
    run(stage, "s4-final", "--spec", tail, "--against", "fp-rl-227")
    labels = {mine["label"]} | {c["label"] for c in calls if c["version"] != "fp-rl-227"}
    assert len(labels) == 3
    # The same spec and name again: its jobs (and so its games) are the ones played before.
    calls.clear()
    run(stage, "s2-final", "--spec", tail, "--against", "fp-rl-227")
    assert calls[0]["label"] == mine["label"]


def test_a_spec_without_a_switch_plays_the_pools_runtime(harness):
    E, stage, calls, tmp = harness
    run(stage, "plain-227", "--spec", spec_file(tmp, "plain", None), "--against", "fp-rl-200")
    mine = calls[0]
    assert mine["t"]["runtime_path"] == str(tmp / "runtime") and mine["label"].startswith("stage8-4070-spec")
    record = json.loads(stage.results_path().read_text())
    assert "alt_share" not in record and record["runtime"] == "runtime"


def test_a_recorded_version_runs_as_before(harness):
    E, stage, calls, tmp = harness
    run(stage, "fp-rl-200", "--against", "fp-rl-227")
    assert [(c["version"], c["label"], c["t"]["runtime_path"]) for c in calls] == [
        ("fp-rl-200", "stage8-4070", str(tmp / "runtime")), ("fp-rl-227", "stage8-4070", str(tmp / "runtime"))]
    assert calls[0]["spec"] == {"family": "model", "version": "fp-rl-200", "parameters": {"weights": ALT}}
    record = json.loads(stage.results_path().read_text())
    assert not {"spec", "runtime", "alt_share"} & set(record)


@pytest.mark.parametrize("name, switch, against", [
    ("s1", {"rule": "always", "weights": ALT}, None),             # no version to pair with
    ("fp-rl-200", {"rule": "always", "weights": ALT}, "fp-rl-227"),  # a version's id
    ("m1", {"rule": "always", "weights": ALT}, "fp-rl-227"),      # a pool member's
    ("a/b", {"rule": "always", "weights": ALT}, "fp-rl-227"),
    ("s1", {"rule": "always", "weights": "d" * 64}, "fp-rl-227"),  # weights not in the model store
    ("s1", {"rule": "tail", "weights": ALT, "margin": 3}, "fp-rl-227"),
])
def test_a_spec_that_cannot_play_is_refused(harness, capsys, name, switch, against):
    E, stage, calls, tmp = harness
    with pytest.raises(SystemExit):
        run(stage, name, "--spec", spec_file(tmp, "s", switch), *(["--against", against] if against else []))
    assert capsys.readouterr().out.startswith("拒绝：") and not calls and not stage.results_path().exists()


def test_a_switching_seat_plays_the_pools_runtime_with_the_repositorys_seat(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "projects/personal/firstprinciples"))
    E = importlib.import_module("harness.evaluate")
    stage = importlib.import_module("harness.stage")
    store = tmp_path / "research"
    monkeypatch.setattr(E, "RESEARCH", store)
    src = ROOT / "src" / "mahjong_ai"
    runtime = store / "old"
    shutil.copytree(src, runtime / "mahjong_ai", ignore=shutil.ignore_patterns("__pycache__"))
    pool = {"runtime_path": str(runtime)}
    assert stage.seat_runtime(pool) == str(runtime)  # the seat is already the repository's
    seat = runtime / "mahjong_ai" / "execution" / "model_seat.py"
    seat.write_text(seat.read_text() + "\n# the pool's older seat\n")
    (runtime / "mahjong_ai" / "domain" / "marker.py").write_text("# the pool's own\n")
    new = Path(stage.seat_runtime(pool))
    assert new.parent == store / ".local/research/runtimes" and new != runtime
    for f in stage.SEAT_FILES:
        assert (new / "mahjong_ai" / f).read_bytes() == (src / f).read_bytes()
    assert (new / "mahjong_ai" / "domain" / "marker.py").read_text() == "# the pool's own\n"
    assert stage.seat_runtime(pool) == str(new) and sorted(p.name for p in new.parent.iterdir()) == [new.name]
