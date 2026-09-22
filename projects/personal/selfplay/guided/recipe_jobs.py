"""Mac-side jobs for recipe iteration; torch-free, driven by the research service.

    python -m guided.recipe_jobs <job-dir>

``task.json`` names the kind and its inputs; ``result.json`` is written exactly
once. Remote work runs detached on the line's host and is polled, so a service
restart can re-attach by running the same job directory again.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from . import remote
from .run import atomic_json

REPO = Path(__file__).resolve().parents[4]
HOST_FILES = (
    "src/",
    "tests/test_guided_",
    "projects/personal/selfplay/guided/",
    "projects/personal/selfplay/src/",
    "projects/personal/selfplay/tests/",
)
EXTRA_FILES = ("pyproject.toml",)
POLL_SECONDS = 15
# Consecutive failed status probes tolerated before a running job is given up.
PROBE_RETRIES = 8


def absolute(path):
    """Remote paths in task files may be relative to the remote repository root."""
    return path if path.startswith("/") else f"{remote.ROOT}/{path}"


def env(code, extra=None):
    paths = [code + "/projects/personal/selfplay", code + "/src", *(extra or [])]
    return {
        "PYTHONPATH": ":".join(paths),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }


def freeze_source(output):
    """Copy the harness source and its tests (no recipes) with a checksum manifest."""
    output = Path(output)
    if not (output / "source-manifest.json").exists():
        files = [
            f
            for f in remote.source_files(REPO)
            if f.startswith(HOST_FILES) or f in EXTRA_FILES
        ]
        for name in files:
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, output / name)
        hashes = {
            f: hashlib.sha256((output / f).read_bytes()).hexdigest()
            for f in sorted(files)
        }
        identity = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        atomic_json(output / "source-manifest.json", {"id": identity, "files": hashes})
    return json.loads((output / "source-manifest.json").read_text())


NATIVE_BUILD = r"""
import hashlib,json,shutil,sysconfig,tempfile
from pathlib import Path
from setuptools import Extension,Distribution
src=Path('src/mahjong_ai');target=src/('_fast'+sysconfig.get_config_var('EXT_SUFFIX'))
if not target.exists():
 with tempfile.TemporaryDirectory(prefix='guided-native-') as folder:
  dist=Distribution({'ext_modules':[Extension('mahjong_ai._fast',sources=[str(src/'_fast.c')],extra_compile_args=['-O3'])]})
  cmd=dist.get_command_obj('build_ext');cmd.build_lib=folder+'/lib';cmd.build_temp=folder+'/temp';dist.run_command('build_ext')
  shutil.copyfile(cmd.get_ext_fullpath('mahjong_ai._fast'),target)
print(json.dumps({'native':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
"""

# The start is a training checkpoint ({"actor", "config"}) or an exported Actor
# directory (actor.pt + meta.json), e.g. a snapshot of an earlier run.
START_EXPORT = r"""
import json,sys,torch
from pathlib import Path
from guided.trial import export_actor
source,output=Path(sys.argv[1]),sys.argv[2]
if source.is_dir():
    weights=torch.load(source/'actor.pt',map_location='cpu',weights_only=True)
    cfg=json.loads((source/'meta.json').read_text())['config']
else:
    state=torch.load(source,map_location='cpu',weights_only=False)
    weights,cfg=state['actor'],state['config']
cfg={k:cfg[k] for k in ('dim','layers','heads','lookahead')}
meta=export_actor(weights,cfg,output)
print(json.dumps(meta))
"""


def prepare(task, progress):
    """Deploy the frozen harness, publish the frozen rules, export each line's start."""
    from .sync import sync_once

    from .artifacts import read_bundle

    root = Path(task["root"])
    manifest = freeze_source(root / "source")
    code = remote.ROOT + "/.local/guided-code/" + manifest["id"]
    remote_root = absolute(task["remote_root"])
    lines = {}
    for line, spec in task["lines"].items():
        host = spec["host"]
        progress(f"{line}: deploying harness")
        remote.upload(
            host, root / "source", [*manifest["files"], "source-manifest.json"], code
        )
        remote.command(
            host,
            [remote.PYTHON, "-c", NATIVE_BUILD],
            cwd=code,
            environment=env(code),
            timeout=600,
        )
        start = f"{remote_root}/{line}/start/actor"
        progress(f"{line}: exporting start Actor")
        out = remote.command(
            host,
            [
                remote.PYTHON,
                "-c",
                START_EXPORT,
                absolute(spec["start_checkpoint"]),
                start,
            ],
            cwd=code,
            environment=env(code),
            timeout=300,
        )
        lines[line] = {
            "start_actor": start,
            "start_meta": json.loads(out.strip().splitlines()[-1]),
        }
    progress("publishing frozen rules")
    hosts = tuple(sorted({spec["host"] for spec in task["lines"].values()}))
    reports = sync_once(
        task["research_project"],
        task["teacher_store"],
        remote.ROOT + "/.local/guided-teachers",
        hosts=hosts,
        code_path=code,
    )
    failed = [r for r in reports if r.get("status") not in ("published", "unchanged")]
    if failed:
        raise RuntimeError(f"teacher publication failed: {failed}")
    bundle_id = reports[0]["id"]
    bundle = read_bundle(Path(task["teacher_store"]) / bundle_id / "manifest.json")
    return {
        "source_id": manifest["id"],
        "code": code,
        "remote_root": remote_root,
        "bundle": f"{remote.ROOT}/.local/guided-teachers/{bundle_id}/manifest.json",
        "bundle_id": bundle_id,
        "teacher": bundle["teacher"],
        "rules": [entry["id"] for entry in bundle["entries"]],
        "lines": lines,
    }


def _recipe_files(package):
    package = Path(package)
    files = sorted(
        p.relative_to(package.parent).as_posix()
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    if "recipe/__init__.py" not in files:
        raise ValueError("recipe package has no __init__.py")
    return files


def _run_remote(task, progress, command, plan, timeout):
    """Upload the plan, run one trial.py command detached, poll until it finishes."""
    host, root = task["host"], task["remote_dir"]
    local = Path(task["job_dir"]) / "plan"
    local.mkdir(exist_ok=True)
    atomic_json(local / "plan.json", plan)
    remote.upload(host, local, ["plan.json"], root)
    extra = [str(Path(plan["recipe"]).parent)] if plan.get("recipe") else []
    launch_file = Path(task["job_dir"]) / f"launch-{command}.json"
    if not launch_file.exists():
        info = remote.command(
            host,
            [remote.PYTHON, "-m", "guided.trial", command, "--root", root],
            cwd=task["code"],
            environment=env(task["code"], extra),
            background_log=root + f"/{command}.log",
        )
        atomic_json(
            launch_file,
            json.loads(info.strip().splitlines()[-1]) | {"time": time.time()},
        )
    launched = json.loads(launch_file.read_text())
    probe = f"""import json,os
from pathlib import Path
r=Path({root!r});s=r/'status.json'
status=json.loads(s.read_text()) if s.exists() else {{}}
try:
    os.kill({launched["pid"]},0);alive=True
except OSError:
    alive=False
print(json.dumps({{'status':status,'alive':alive}}))
"""
    started = time.time()
    failures = 0
    while True:
        try:
            reply = json.loads(
                remote.execute(host, probe, timeout=90).strip().splitlines()[-1]
            )
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired):
            # The detached job keeps running; a relayed SSH hop may drop briefly.
            failures += 1
            if failures > PROBE_RETRIES:
                raise
            time.sleep(POLL_SECONDS)
            continue
        state = reply["status"].get("state")
        if state in ("complete", "failed", "timeout"):
            return reply["status"]
        if not reply["alive"] and time.time() - launched["time"] > 30:
            log = remote.execute(
                host,
                f"print(open({root + '/' + command + '.log'!r}).read()[-4000:])",
                timeout=60,
            )
            return {"state": "failed", "error": "process exited without status\n" + log}
        if time.time() - started > timeout:
            remote.execute(
                host,
                f"import os,signal\ntry: os.killpg({launched['pid']},signal.SIGTERM)\nexcept OSError: pass",
                timeout=60,
            )
            return {"state": "timeout", "error": "harness job exceeded its timeout"}
        progress(f"{command}: {state or 'starting'}")
        time.sleep(POLL_SECONDS)


def _fetch(task, name):
    try:
        path = Path(task["job_dir"]) / name
        remote.download(task["host"], task["remote_dir"] + "/" + name, path)
        return path
    except Exception:
        return None


def trial_train(task, progress, smoke=False):
    package = Path(task["recipe"])
    remote_recipe = task["remote_dir"] + "/recipe"
    remote.upload(
        task["host"], package.parent, _recipe_files(package), task["remote_dir"]
    )
    plan = {
        "source_id": task["source_id"],
        "recipe": remote_recipe,
        "bundle": task["bundle"],
        "start_actor": task["start_actor"],
        "budget_seconds": task["budget_seconds"],
        "smoke_seconds": task.get("smoke_seconds", 90),
        "seed": task["seed"],
        "line": task["line"],
        "limits": task.get("limits") or {},
        "snapshot_seconds": task.get("snapshot_seconds") or [],
    }
    budget = plan["smoke_seconds"] if smoke else plan["budget_seconds"]
    status = _run_remote(
        task, progress, "smoke" if smoke else "train", plan, budget + 900
    )
    snapshots = _fetch(task, "snapshots.json") if task.get("snapshot_seconds") else None
    metrics = _fetch(task, "metrics.jsonl")
    rows = []
    if metrics:
        rows = [
            json.loads(line)
            for line in metrics.read_text().splitlines()
            if line.strip()
        ]
    return {
        "status": status,
        "ok": status.get("state") == "complete",
        "actor": task["remote_dir"] + "/actor"
        if status.get("state") == "complete"
        else None,
        "metrics": rows[-400:],
        "snapshots": json.loads(snapshots.read_text()) if snapshots else [],
    }


def trial_evaluate(task, progress):
    plan = {
        "source_id": task["source_id"],
        "bundle": task["bundle"],
        "challenger": task["challenger"],
        "incumbent": task["incumbent"],
        "games": task["games"],
        "seed": task["seed"],
        "namespace": task["namespace"],
        "execution": task["execution"],
    }
    status = _run_remote(task, progress, "evaluate", plan, 5400)
    path = _fetch(task, "result.json") if status.get("state") == "complete" else None
    return {
        "status": status,
        "ok": path is not None,
        "result": json.loads(path.read_text()) if path else None,
    }


def trial_brawl(task, progress):
    plan = {
        "source_id": task["source_id"],
        "bundle": task["bundle"],
        "incumbent": task["incumbent"],
        "appearances": task["appearances"],
        "seed": task["seed"],
        "execution": task["execution"],
    }
    status = _run_remote(task, progress, "brawl", plan, 5400)
    path = _fetch(task, "result.json") if status.get("state") == "complete" else None
    return {
        "status": status,
        "ok": path is not None,
        "result": json.loads(path.read_text()) if path else None,
    }


def trial_curve(task, progress):
    plan = {
        k: task[k]
        for k in ("source_id", "bundle", "reference", "targets", "games", "seed")
    } | {"namespace": task["namespace"], "execution": task["execution"]}
    status = _run_remote(task, progress, "curve", plan, 6 * 3600)
    path = _fetch(task, "result.json") if status.get("state") == "complete" else None
    return {
        "status": status,
        "ok": path is not None,
        "result": json.loads(path.read_text()) if path else None,
    }


KINDS = {
    "prepare": prepare,
    "smoke": lambda task, progress: trial_train(task, progress, smoke=True),
    "train": trial_train,
    "evaluate": trial_evaluate,
    "brawl": trial_brawl,
    "curve": trial_curve,
}


def kill(job_dir):
    """Terminate the detached remote process group of a discarded job."""
    job_dir = Path(job_dir)
    task = json.loads((job_dir / "task.json").read_text())
    for launch in job_dir.glob("launch-*.json"):
        pid = json.loads(launch.read_text())["pid"]
        remote.execute(
            task["host"],
            f"import os,signal\ntry: os.killpg({int(pid)},signal.SIGTERM)\nexcept OSError: pass",
            timeout=60,
        )


def run(job_dir):
    job_dir = Path(job_dir)
    task = json.loads((job_dir / "task.json").read_text()) | {"job_dir": str(job_dir)}
    result_path = job_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    def progress(message):
        atomic_json(
            job_dir / "progress.json",
            {"message": message, "time": time.time(), "pid": os.getpid()},
        )

    try:
        result = {"status": "completed", "output": KINDS[task["kind"]](task, progress)}
    except BaseException as exc:
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-6000:],
        }
    atomic_json(result_path, result | {"finished": time.time()})
    return result


if __name__ == "__main__":
    if sys.argv[1] == "--kill":
        kill(sys.argv[2])
    else:
        print(json.dumps(run(sys.argv[1]), ensure_ascii=False)[:2000])
