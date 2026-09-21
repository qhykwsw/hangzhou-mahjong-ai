"""Durable Mac coordinator for one bounded, two-host guided experiment.

Construction never starts training. start()/run() begin the single wall-clock
budget; resuming always preserves its original deadline. Teacher publication is
an external service: only main training consumes its current.json pointer.
"""

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import uuid

from . import remote
from .config import SCALES, configuration
from .statistics import interval

HOUR = 3600

# The lock spans the entire node process, closing the dispatch/save crash window.
# A retry can launch another wrapper, but can never launch a duplicate node.
WRAPPER = """import fcntl,json,pathlib,runpy,sys
path=pathlib.Path(sys.argv[1])
lock=path.with_suffix('.lock').open('a')
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 raise SystemExit(0)
status=path.with_suffix('.status.json')
if status.exists():
 raise SystemExit(0)
sys.argv=['guided.node',str(path)]
runpy.run_module('guided.node',run_name='__main__')
"""


class NodeFailure(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class Coordinator:
    def __init__(
        self,
        local_run,
        code_path,
        bundle_path,
        panel_path,
        *,
        clock=time.time,
        remote_module=remote,
        updates_path=None,
    ):
        self.local = Path(local_run).resolve()
        self.local.mkdir(parents=True, exist_ok=True)
        self.path = self.local / "state.json"
        self.clock, self.remote = clock, remote_module
        bundle = PurePosixPath(bundle_path)
        updates = str(updates_path or bundle.parent.parent / "current.json")
        inputs = dict(
            code=str(code_path),
            bundle=str(bundle_path),
            panel=str(panel_path),
            updates=updates,
        )
        with (self.local / "state-init.lock").open("a") as init_lock:
            fcntl.flock(init_lock, fcntl.LOCK_EX)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if (
                    self.state.get("format") != "guided-experiment-v1"
                    or self.state["inputs"] != inputs
                ):
                    raise ValueError(
                        "coordinator inputs differ from the persisted experiment"
                    )
            else:
                self.state = {
                    "format": "guided-experiment-v1",
                    "status": "ready",
                    "stage": "scale",
                    "run_id": self.local.name + "-" + uuid.uuid4().hex[:8],
                    "inputs": inputs,
                    "started": None,
                    "deadline": None,
                    "stage_started": {},
                    "stage_deadlines": {},
                    "tasks": {},
                    "transfers": {},
                    "candidates": {},
                    "selection": {},
                }
                self._save()
        self.root = (
            PurePosixPath(self.remote.ROOT)
            / ".local/guided-experiments"
            / self.state["run_id"]
        )

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, self.path)

    def start(self, hours=168):
        if self.state["started"] is not None:
            return self.state
        if not math.isfinite(hours) or hours <= 48:
            raise ValueError(
                "experiment must leave time beyond 12h scale + 12h startup + 24h final"
            )
        now = self.clock()
        self.state.update(status="running", started=now, deadline=now + hours * HOUR)
        self.state["stage_started"]["scale"] = now
        self.state["stage_deadlines"]["scale"] = now + 12 * HOUR
        self.state["stage_deadlines"]["main"] = self.state["deadline"] - 24 * HOUR
        self.state["stage_deadlines"]["final"] = self.state["deadline"]
        self._save()
        return self.state

    def _remote_json(self, host, operation, **values):
        # This single structured request also makes remote state transitions easy
        # to test without SSH. Binary actor payloads are captured, never logged.
        request = {"operation": operation, **values}
        script = (
            "# guided-coordinator\nimport json,pathlib,base64,fcntl,sys,os,signal\nrequest="
            + repr(request)
            + "\n"
        )
        script += """op=request['operation']
if op=='status':
 path=pathlib.Path(request['path']);status=path.with_suffix('.status.json');lock=path.with_suffix('.lock')
 result=json.loads(status.read_text()) if status.exists() else None
 if result is None or result.get('state')=='running':
  if lock.exists():
   with lock.open('a') as stream:
    try:
     fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
     result=json.loads(status.read_text()) if status.exists() else None
     if result and result.get('state')=='running':
      result={'state':'failed','error':'node exited without a terminal status'}
    except BlockingIOError:
     result=result or {'state':'starting'}
 print(json.dumps(result))
elif op=='download_actor':
 root=pathlib.Path(request['path'])
 print(json.dumps({name:base64.b64encode((root/name).read_bytes()).decode() for name in ('actor.pt','meta.json')}))
elif op=='select_checkpoint':
 root=pathlib.Path(request['run']);best=root/'best.pt';latest=root/'latest.pt'
 chosen=best if best.is_file() else latest
 if not chosen.is_file(): raise FileNotFoundError(str(chosen))
 print(json.dumps({'checkpoint':str(chosen),'selection':'best' if chosen==best else 'latest'}))
elif op=='freeze_bundle':
 sys.path.insert(0,str(pathlib.Path(request['code'])/'projects/personal/selfplay'))
 from guided.artifacts import read_bundle
 pointer=pathlib.Path(request['pointer'])
 source=pointer if pointer.is_file() else pathlib.Path(request['fallback'])
 bundle=read_bundle(source)
 print(json.dumps({'manifest':bundle['manifest_path'],'id':bundle['id'],'teacher':bundle['teacher']}))
elif op=='verify_bundle':
 sys.path.insert(0,str(pathlib.Path(request['code'])/'projects/personal/selfplay'))
 from guided.artifacts import read_bundle
 bundle=read_bundle(request['manifest'])
 print(json.dumps({'id':bundle['id'],'teacher':bundle['teacher']}))
elif op=='terminate':
 pid=request.get('pid');path=request['path'];stopped=False
 if pid:
  try:
   command=pathlib.Path('/proc')/str(pid)/'cmdline'
   if path.encode() in command.read_bytes() and os.getpgid(pid)==pid:
    os.killpg(pid,signal.SIGTERM);stopped=True
  except ProcessLookupError: pass
  except FileNotFoundError: pass
 print(json.dumps({'stopped':stopped}))
else: raise ValueError(op)
"""
        return json.loads(self.remote.execute(host, script))

    def _fail(self, message):
        self.state.update(status="failed", error=str(message), failed_at=self.clock())
        self._save()
        # Stop only this coordinator's still-running process groups, with PID and
        # command-line identity checks. Never touch another experiment's jobs.
        for task in self.state["tasks"].values():
            if task.get("state") in {"submitted", "running", "starting"}:
                try:
                    result = self._remote_json(
                        task["host"],
                        "terminate",
                        pid=task.get("pid"),
                        path=task["path"],
                    )
                    task["cancellation"] = result
                except Exception as exc:
                    task["cancellation_error"] = str(exc)
        self._save()
        raise NodeFailure(str(message))

    def _task(self, name, host, payload_factory, *, allow_oom=False):
        """Return the immutable result, or None while the named node is running."""
        tasks = self.state["tasks"]
        if name not in tasks:
            if self.clock() >= self.state["stage_deadlines"].get(
                self.state["stage"], self.state["deadline"]
            ):
                raise BudgetExceeded(
                    f"{self.state['stage']} budget exhausted before {name}"
                )
            payload = payload_factory()
            path = str(self.root / "tasks" / (name + ".json"))
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            tasks[name] = dict(
                host=host,
                path=path,
                payload=payload,
                sha256=digest,
                state="pending",
                created=self.clock(),
                attempts=0,
            )
            self._save()
        task = tasks[name]
        if task["host"] != host:
            raise ValueError("an existing task cannot change hosts")
        if task["state"] == "complete":
            return task["result"]
        if task["state"] == "ineligible":
            return {"ineligible": task["reason"]}
        status = self._remote_json(host, "status", path=task["path"])
        if status:
            task["remote_status"] = status
            state = status["state"]
            task["state"] = state
            if state == "complete":
                task["result"] = status["result"]
                self._save()
                return task["result"]
            if state == "failed":
                error = status.get("error", "remote node failed")
                if (
                    allow_oom
                    and "out of memory" in error.lower()
                    and "cuda" in error.lower()
                ):
                    task.update(state="ineligible", reason=error)
                    self._save()
                    return {"ineligible": error}
                self._save()
                self._fail(f"{name} on {host}: {error}")
            self._save()
            cap = self.state["stage_deadlines"].get(
                self.state["stage"], self.state["deadline"]
            )
            # Main training enforces its own deadline; allow only checkpoint
            # finalization to consume at most five minutes of the final window.
            grace = 300 if self.state["stage"] == "main" else 0
            if self.clock() >= cap + grace:
                raise BudgetExceeded(
                    f"{self.state['stage']} deadline expired with {name} still running"
                )
            return None
        if self.clock() >= self.state["stage_deadlines"].get(
            self.state["stage"], self.state["deadline"]
        ):
            raise BudgetExceeded(
                f"{self.state['stage']} budget exhausted waiting for {name}"
            )
        if task["attempts"] and self.clock() - task["last_dispatch"] < 60:
            return None
        # Persist intent before any remote side effect. The wrapper lock prevents
        # duplicates even if Mac crashes after Popen but before saving its PID.
        task.update(
            state="submitted", attempts=task["attempts"] + 1, last_dispatch=self.clock()
        )
        self._save()
        directory = self.local / "tasks" / host
        directory.mkdir(parents=True, exist_ok=True)
        local = directory / (name + ".json")
        local.write_text(json.dumps(task["payload"], allow_nan=False) + "\n")
        self.remote.upload(host, directory, [local.name], str(self.root / "tasks"))
        code = self.state["inputs"]["code"]
        env = {
            "PYTHONPATH": ":".join(
                [
                    code + "/src",
                    code + "/projects/personal/selfplay",
                    code + "/projects/personal/selfplay/src",
                ]
            )
        }
        launched = json.loads(
            self.remote.command(
                host,
                [self.remote.PYTHON, "-c", WRAPPER, task["path"]],
                cwd=code,
                environment=env,
                background_log=str(self.root / "logs" / (name + ".log")),
            )
        )
        task["pid"] = launched["pid"]
        self._save()
        return None

    def _transfer(self, name, origin, artifact, destination="4070"):
        if name in self.state["transfers"]:
            return self.state["transfers"][name]["destination"]
        target = str(self.root / "artifacts" / name)
        if origin == destination:
            target = artifact
        else:
            encoded = self._remote_json(origin, "download_actor", path=artifact)
            if set(encoded) != {"actor.pt", "meta.json"}:
                raise ValueError(
                    "cross-host transfer must contain Actor weights and metadata only"
                )
            with tempfile.TemporaryDirectory(
                prefix=".transfer-", dir=self.local
            ) as directory:
                root = Path(directory)
                for filename, data in encoded.items():
                    (root / filename).write_bytes(base64.b64decode(data, validate=True))
                meta = json.loads((root / "meta.json").read_text())
                actual = hashlib.sha256((root / "actor.pt").read_bytes()).hexdigest()
                if (
                    actual != meta.get("weights_sha256")
                    or meta.get("format") != "guided-actor-v1"
                ):
                    raise ValueError("invalid Actor transfer checksum or format")
                self.remote.upload(destination, root, ["actor.pt", "meta.json"], target)
        self.state["transfers"][name] = dict(
            origin=origin, source=artifact, target_host=destination, destination=target
        )
        self._save()
        return target

    def _config(self, scale, seed=20260920):
        dim, layers, heads = SCALES[scale]
        return configuration(
            dim=dim,
            layers=layers,
            heads=heads,
            seed=seed,
            workers=8,
            tables=8,
            minibatch=32,
            effective_minibatch=512,
        )

    def _train(
        self,
        name,
        cfg,
        duration=None,
        *,
        initial=None,
        carry=False,
        host="4090",
        bc_seconds=0,
        main=False,
    ):
        def payload():
            cap = self.state["stage_deadlines"][self.state["stage"]]
            deadline = cap if duration is None else self.clock() + duration
            if deadline > cap:
                raise BudgetExceeded(
                    f"not enough {self.state['stage']} budget for full {name} duration"
                )
            return dict(
                kind="train",
                run=str(self.root / "runs" / name),
                config=cfg,
                bundle=self.state["inputs"]["bundle"],
                panel=self.state["inputs"]["panel"],
                deadline=deadline,
                initial=initial,
                carry=carry,
                bc_seconds=bc_seconds,
                eval_enabled=True,
                updates=self.state["inputs"]["updates"] if main else None,
            )

        return self._task(name, host, payload, allow_oom=self.state["stage"] == "scale")

    def _evaluate(self, name, host, checkpoint, games, namespace, bundle=None):
        return self._task(
            name,
            host,
            lambda: dict(
                kind="evaluate",
                checkpoint=checkpoint,
                bundle=bundle or self.state["inputs"]["bundle"],
                panel=self.state["inputs"]["panel"],
                deadline=self.state["stage_deadlines"][self.state["stage"]],
                games=games,
                eval_seed=17320260920,
                namespace=namespace,
                output=str(self.root / "evaluations" / (name + ".json")),
            ),
            allow_oom=self.state["stage"] == "scale",
        )

    @staticmethod
    def _deployable(result):
        if "ineligible" in result:
            return False, result["ineligible"]
        loads = result.get("loads", {})
        if set(loads) != {"sustained", "synchronized_burst"} or not result.get(
            "passed_legality"
        ):
            raise ValueError("deployment benchmark did not pass legal action checks")
        if any(
            load.get("errors") or load.get("illegal_actions") for load in loads.values()
        ):
            raise ValueError("deployment benchmark reported inference errors")
        worst = max(float(load["p95_ms"]) for load in loads.values())
        if not math.isfinite(worst):
            raise ValueError("nonfinite deployment latency")
        if worst > 100:
            return False, f"4070 ten-request P95 {worst:.3f}ms exceeds 100ms"
        gpu = result.get("gpu")
        if gpu and gpu["peak_allocated_bytes"] > 0.8 * gpu["total_bytes"]:
            return False, "4070 benchmark exceeds 80% GPU memory budget"
        return True, None

    @staticmethod
    def _groups(result):
        groups = result["results"]["model:current"]["group_scores"]
        interval(groups)  # validates completeness of finite group-level evidence
        return groups

    @staticmethod
    def _difference(new, old):
        if len(new) != len(old):
            raise ValueError("paired selection requires identical group counts")
        return interval([float(a) - float(b) for a, b in zip(new, old)])

    def _advance(self, stage):
        self.state["stage"] = stage
        now = self.clock()
        self.state["stage_started"][stage] = now
        if stage == "startup":
            self.state["stage_deadlines"][stage] = min(
                now + 12 * HOUR, self.state["stage_deadlines"]["main"]
            )
        if stage == "publish":
            self.state["stage_deadlines"][stage] = self.state["deadline"]
        self._save()

    def _scale(self):
        for scale in SCALES:
            candidate = self.state["candidates"].setdefault(scale, {})
            if candidate.get("state") in {"eligible", "ineligible"}:
                continue
            prefix = "scale-" + scale
            initial = self._task(
                prefix + "-initialize",
                "4090",
                lambda: dict(
                    kind="initialize",
                    config=self._config(scale),
                    output=str(self.root / "initial" / (scale + ".pt")),
                ),
                allow_oom=True,
            )
            if initial is None:
                return
            if "ineligible" in initial:
                candidate.update(state="ineligible", reason=initial["ineligible"])
                self._save()
                return
            candidate["initial"] = initial["checkpoint"]
            exported = self._task(
                prefix + "-export",
                "4090",
                lambda: dict(
                    kind="export",
                    checkpoint=initial["checkpoint"],
                    output=str(self.root / "artifacts" / (prefix + "-initial")),
                ),
                allow_oom=True,
            )
            if exported is None:
                return
            if "ineligible" in exported:
                candidate.update(state="ineligible", reason=exported["ineligible"])
                self._save()
                return
            artifact = self._transfer(prefix + "-4070", "4090", exported["artifact"])
            bench = self._task(
                prefix + "-benchmark",
                "4070",
                lambda: dict(
                    kind="benchmark",
                    source=artifact,
                    count=200,
                    output=str(self.root / "benchmarks" / (prefix + ".json")),
                ),
                allow_oom=True,
            )
            if bench is None:
                return
            candidate["benchmark"] = bench
            deployable, reason = self._deployable(bench)
            if not deployable:
                candidate.update(state="ineligible", reason=reason)
                self._save()
                return
            trained = self._train(
                prefix + "-train",
                self._config(scale),
                3 * HOUR,
                initial=initial["checkpoint"],
            )
            if trained is None:
                return
            if "ineligible" in trained:
                candidate.update(state="ineligible", reason=trained["ineligible"])
                self._save()
                return
            evaluated = self._evaluate(
                prefix + "-evaluate",
                "4090",
                trained["checkpoint"],
                1024,
                "scale-comparison",
            )
            if evaluated is None:
                return
            if "ineligible" in evaluated:
                candidate.update(state="ineligible", reason=evaluated["ineligible"])
                self._save()
                return
            candidate.update(
                state="eligible", checkpoint=trained["checkpoint"], evaluation=evaluated
            )
            self._save()
            return
        eligible = [
            name
            for name in SCALES
            if self.state["candidates"][name]["state"] == "eligible"
        ]
        if not eligible:
            raise ValueError("no model scale passed the 4070 deployment constraint")
        selected, comparisons = eligible[0], []
        for name in eligible[1:]:
            difference = self._difference(
                self._groups(self.state["candidates"][name]["evaluation"]),
                self._groups(self.state["candidates"][selected]["evaluation"]),
            )
            comparisons.append(
                dict(candidate=name, incumbent=selected, difference=difference)
            )
            if difference["lower"] > 0:
                selected = name
        self.state["selection"].update(scale=selected, scale_comparisons=comparisons)
        self._advance("startup")

    def _startup(self):
        scale = self.state["selection"]["scale"]
        initial = self.state["candidates"][scale]["initial"]
        results = {}
        for route in ("A", "B"):
            result = self._train(
                "startup-" + route,
                self._config(scale),
                4 * HOUR,
                initial=initial,
                bc_seconds=HOUR if route == "B" else 0,
            )
            if result is None:
                return
            results[route] = result
        evaluations = {}
        for route in ("A", "B"):
            result = self._evaluate(
                "startup-" + route + "-evaluate",
                "4090",
                results[route]["checkpoint"],
                1024,
                "startup-comparison",
            )
            if result is None:
                return
            evaluations[route] = result
        difference = self._difference(
            self._groups(evaluations["B"]), self._groups(evaluations["A"])
        )
        route = "B" if difference["lower"] > 0 else "A"
        self.state["selection"].update(
            route=route,
            startup_difference=difference,
            carry_checkpoint=results[route]["checkpoint"],
        )
        self._advance("main")

    def _main(self):
        scale, route = (
            self.state["selection"]["scale"],
            self.state["selection"]["route"],
        )
        results = []
        # Call both before returning, so the independent hosts launch concurrently.
        for host in ("4090", "4070"):
            result = self._train(
                "main-" + host,
                self._config(scale, 20260920 if host == "4090" else 20260921),
                host=host,
                initial=self.state["selection"]["carry_checkpoint"]
                if host == "4090"
                else None,
                carry=host == "4090",
                bc_seconds=HOUR if host == "4070" and route == "B" else 0,
                main=True,
            )
            results.append(result)
        if all(result is not None for result in results):
            self._advance("final")

    def _final(self):
        if "final_inputs" not in self.state:
            teacher = self._remote_json(
                "4090",
                "freeze_bundle",
                code=self.state["inputs"]["code"],
                pointer=self.state["inputs"]["updates"],
                fallback=self.state["inputs"]["bundle"],
            )
            matched = self._remote_json(
                "4070",
                "verify_bundle",
                code=self.state["inputs"]["code"],
                manifest=teacher["manifest"],
            )
            if (
                matched["id"] != teacher["id"]
                or matched["teacher"] != teacher["teacher"]
            ):
                raise ValueError("final teacher is not identical on both hosts")
            checkpoints = {
                host: self._remote_json(
                    host,
                    "select_checkpoint",
                    run=str(self.root / "runs" / ("main-" + host)),
                )
                for host in ("4090", "4070")
            }
            self.state["final_inputs"] = dict(
                teacher=teacher,
                checkpoints=checkpoints,
                frozen_at=self.clock(),
                games=8192,
            )
            self._save()
        frozen = self.state["final_inputs"]
        evaluations = {}
        for host in ("4090", "4070"):
            result = self._evaluate(
                "final-" + host,
                host,
                frozen["checkpoints"][host]["checkpoint"],
                frozen["games"],
                "final-test",
                frozen["teacher"]["manifest"],
            )
            if result is not None:
                evaluations[host] = result
        if len(evaluations) != 2:
            return
        teacher = frozen["teacher"]["teacher"]
        first_teacher = evaluations["4090"]["results"][teacher]["group_scores"]
        second_teacher = evaluations["4070"]["results"][teacher]["group_scores"]
        if first_teacher != second_teacher:
            raise ValueError(
                "fixed-teacher final scores disagree across hosts; paired conditions are not verified"
            )
        groups = {host: self._groups(value) for host, value in evaluations.items()}
        winner = max(
            ("4090", "4070"), key=lambda host: sum(groups[host]) / len(groups[host])
        )
        self.state["selection"].update(
            winner=winner,
            kind="actor"
            if sum(groups[winner]) / len(groups[winner])
            > sum(first_teacher) / len(first_teacher)
            else "teacher",
            teacher_bundle=frozen["teacher"]["manifest"],
            teacher_bundle_id=frozen["teacher"]["id"],
            final_difference_4070_minus_4090=self._difference(
                groups["4070"], groups["4090"]
            ),
            final_student_minus_teacher=self._difference(groups[winner], first_teacher),
            final_selection_note="This test also selects the winner; its interval is not selection-adjusted.",
        )
        self._advance("publish")

    def _publish(self):
        winner = self.state["selection"]["winner"]
        checkpoint = self.state["final_inputs"]["checkpoints"][winner]["checkpoint"]
        exported = self._task(
            "final-export",
            winner,
            lambda: dict(
                kind="export",
                checkpoint=checkpoint,
                output=str(self.root / "artifacts/final-actor"),
            ),
        )
        if exported is None:
            return
        artifact = self._transfer("final-actor-4070", winner, exported["artifact"])
        result = self._task(
            "final-deployment",
            "4070",
            lambda: dict(
                kind="benchmark",
                source=artifact,
                count=200,
                output=str(self.root / "benchmarks/final-deployment.json"),
            ),
        )
        if result is None:
            return
        eligible, reason = self._deployable(result)
        if not eligible and self.state["selection"]["kind"] == "actor":
            raise ValueError(f"final Actor failed deployment acceptance: {reason}")
        self.state["selection"]["recommendation_4070"] = (
            self.state["selection"]["teacher_bundle"]
            if self.state["selection"]["kind"] == "teacher"
            else artifact
        )
        self.state.update(
            status="complete",
            finished=self.clock(),
            artifact_4070=artifact,
            actor_artifact_role="research"
            if self.state["selection"]["kind"] == "teacher"
            else "recommended",
            deployment=result,
            actor_deployable=eligible,
            deployment_reason=reason,
            platform_started=False,
        )
        self._save()

    def tick(self):
        if self.state["started"] is None:
            raise RuntimeError("call start() before advancing the experiment")
        if self.state["status"] == "failed":
            raise NodeFailure(self.state["error"])
        if self.state["status"] == "complete":
            return self.state
        try:
            if self.clock() >= self.state["deadline"]:
                raise BudgetExceeded(
                    "the original total experiment deadline has expired"
                )
            getattr(self, "_" + self.state["stage"])()
        except (BudgetExceeded, ValueError, KeyError) as exc:
            self._fail(exc)
        return self.state

    def run(self, hours=168, poll_seconds=10):
        # Only one local process may coordinate this state; remote locks provide
        # the independent protection against ambiguous SSH dispatch outcomes.
        with (self.local / "coordinator.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another coordinator is already running") from exc
            # A second instance may have loaded this object before the first
            # coordinator wrote its budget. Reload only after winning the lock.
            self.state = json.loads(self.path.read_text())
            self.root = (
                PurePosixPath(self.remote.ROOT)
                / ".local/guided-experiments"
                / self.state["run_id"]
            )
            self.start(hours)
            while self.state["status"] != "complete":
                try:
                    self.tick()
                except NodeFailure:
                    raise
                except (OSError, RuntimeError) as exc:
                    # Transport failures are visible and retryable, not a new
                    # experiment, a new budget, or a reason to rerun completed jobs.
                    self.state["last_transport_error"] = {
                        "time": self.clock(),
                        "error": str(exc),
                    }
                    self._save()
                if self.state["status"] != "complete":
                    time.sleep(min(60, max(0.1, poll_seconds)))
        return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--updates")
    parser.add_argument("--hours", type=float, default=168)
    args = parser.parse_args()
    coordinator = Coordinator(
        args.run, args.code, args.bundle, args.panel, updates_path=args.updates
    )
    result = coordinator.run(args.hours)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "started",
                    "deadline",
                    "selection",
                    "artifact_4070",
                )
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
