"""A host reset must not leave a recorded training PID reported as running."""

import contextlib
import io
import json

from guided.scale_monitor import SNAPSHOT


def snapshot(root, suite, stages):
    (root / "suite.json").write_text(json.dumps(suite))
    for name, stage in stages.items():
        folder = root / name
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps({"stage": stage}))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(SNAPSHOT.replace("ROOT", repr(str(root))), {})
    return json.loads(output.getvalue())


def test_host_reset_reconciles_missing_process_without_changing_recorded_file(tmp_path):
    suite = {
        "stage": "running",
        "tasks": {
            "small": {
                "state": "running",
                "pid": 999999999,
                "argv": ["python", "-m", "guided.scale_study", "train"],
            }
        },
    }
    result = snapshot(tmp_path, suite, {"small": "supervised_updates"})
    assert result["suite"]["stage"] == "interrupted"
    assert result["models"]["small"]["stage"] == "interrupted"
    assert result["models"]["small"]["recorded_stage"] == "supervised_updates"
    assert json.loads((tmp_path / "suite.json").read_text()) == suite


def test_completed_receipts_survive_parent_exit_before_final_status_write(tmp_path):
    names = ("small", "medium", "large", "xlarge")
    suite = {
        "stage": "running",
        "tasks": {name: {"state": "complete"} for name in names},
    }
    suite["tasks"]["xlarge"] = {
        "state": "running",
        "pid": 999999999,
        "argv": ["python", "-m", "guided.scale_study", "train"],
    }
    result = snapshot(tmp_path, suite, dict.fromkeys(names, "complete"))
    assert result["suite"]["stage"] == "complete"
    assert all(m["stage"] == "complete" for m in result["models"].values())
