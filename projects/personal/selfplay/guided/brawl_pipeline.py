"""Finite calibration, mixed-policy smoke test, and frozen tournament pipeline."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import bootstrap
from .overnight import run_child
from .run import atomic_json


def run(root):
    plan = json.loads((root / "plan.json").read_text())
    if plan["source_id"] != bootstrap.verify_source():
        raise ValueError("source mismatch")
    if (root / "pipeline.json").exists():
        raise FileExistsError("pipeline already started")
    if len(plan["models"]) != 6 or len(plan["rules"]) != 4:
        raise ValueError("ten-strategy roster required")
    for host in ["4090", "4070"]:
        if not plan["bc_controls"][host]["pure_bc_verified"]:
            raise ValueError("pure imitation control not verified")
    state = {
        "stage": "starting",
        "pid": os.getpid(),
        "tasks": {},
        "started": time.time(),
    }

    def save():
        state["updated"] = time.time()
        atomic_json(root / "pipeline.json", state)

    save()
    try:
        for name, module, extra, timeout in [
            ("calibration", "guided.tune_evaluation", [], 7200),
            ("smoke", "guided.brawl", ["--smoke"], 900),
            ("tournament", "guided.brawl", [], plan["timeout_seconds"] + 60),
        ]:
            state["stage"] = name
            argv = [sys.executable, "-m", module, "--root", str(root), *extra]
            task = {"state": "running", "argv": argv, "started": time.time()}
            state["tasks"][name] = task
            save()

            def launched(pid):
                task["pid"] = pid
                save()

            run_child(argv, root / (name + ".log"), timeout, launched)
            task.update(state="complete", finished=time.time())
            save()
        state["stage"] = "complete"
    except BaseException as exc:
        state.update(stage="failed", error=str(exc))
        raise
    finally:
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    run(Path(p.parse_args().root).resolve())


if __name__ == "__main__":
    main()
