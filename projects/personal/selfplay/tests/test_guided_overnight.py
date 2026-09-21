"""The finite startup experiment must preserve budgets and route isolation."""

import json

import pytest

from guided import overnight


def prepare(tmp_path):
    plan = {
        "host": "4070",
        "config": overnight.host_config("4070"),
        "bundle": "/frozen/teacher.json",
        "training_seconds_per_route": 14400,
        "bc_seconds": 3600,
        "evaluation_timeout_seconds": 7200,
    }
    overnight.atomic_json(tmp_path / "plan.json", plan)
    overnight.atomic_json(
        tmp_path / "preflight/receipt.json",
        {"state": "passed", "source_id": overnight.bootstrap.verify_source()},
    )
    return plan


def test_routes_share_initialization_and_have_separate_full_budgets(
    tmp_path, monkeypatch
):
    prepare(tmp_path)
    monkeypatch.setattr(
        overnight, "initialization", lambda *args: tmp_path / "initial.pt"
    )
    calls = []

    def child(argv, log, timeout, on_start):
        calls.append((argv, timeout))
        on_start(123)
        if "evaluate" in argv:
            route = argv[argv.index("--route") + 1]
            overnight.atomic_json(
                tmp_path / "evaluation" / f"{route}.json",
                {
                    "results": {
                        "model:current": {
                            "group_scores": [0, 1] if route == "A" else [2, 3]
                        }
                    }
                },
            )

    monkeypatch.setattr(overnight, "run_child", child)
    overnight.suite(tmp_path)
    assert len(calls) == 4
    first, second = (c[0] for c in calls[:2])
    assert first[first.index("--initial") + 1] == second[second.index("--initial") + 1]
    assert first[first.index("--bc-seconds") + 1] == "0"
    assert second[second.index("--bc-seconds") + 1] == "3600"
    assert all(c[1] == 14700 for c in calls[:2])
    result = json.loads((tmp_path / "selection.json").read_text())
    assert result["route"] == "B"
    assert result["further_training_started"] is False
    with pytest.raises(FileExistsError):
        overnight.suite(tmp_path)


def test_failure_stops_host_without_launching_next_route(tmp_path, monkeypatch):
    prepare(tmp_path)
    monkeypatch.setattr(
        overnight, "initialization", lambda *args: tmp_path / "initial.pt"
    )

    def fail(*args):
        raise RuntimeError("GPU failed")

    monkeypatch.setattr(overnight, "run_child", fail)
    with pytest.raises(RuntimeError, match="GPU failed"):
        overnight.suite(tmp_path)
    state = json.loads((tmp_path / "suite.json").read_text())
    assert state["stage"] == "failed"
    assert set(state["tasks"]) == {"train-A"}
