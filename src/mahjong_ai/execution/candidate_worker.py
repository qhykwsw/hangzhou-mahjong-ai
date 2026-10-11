"""Worker side of the public-observation protocol. stdout is reserved for RPC."""

import importlib.util
import json
import sys
import os
import threading
import time
import traceback
from pathlib import Path

from mahjong_ai.domain.types import (
    Action,
    Meld,
    Observation,
    PublicEvent,
    RoundResult,
    StageContext,
)


def observation(data, events=None):
    d = dict(data)
    d["stage"] = StageContext(**d["stage"])
    d["legal_actions"] = tuple(
        Action(**(a | {"tiles": tuple(a.get("tiles", []))})) for a in d["legal_actions"]
    )
    if events is None:
        d["public_events"] = tuple(PublicEvent(**e) for e in d["public_events"])
    else:
        d["public_events"] = events
    d["melds"] = tuple(tuple(Meld(**m) for m in row) for row in d["melds"])
    for key in ("hand", "hand_sizes"):
        d[key] = tuple(d[key])
    d["discards"] = tuple(tuple(row) for row in d["discards"])
    return Observation(**d)


def main():
    parent = os.getppid()

    def parent_watch():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=parent_watch, daemon=True).start()
    package, output = map(Path, sys.argv[1:3])
    channel = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, str(package))

    def send(value):
        channel.write(json.dumps(value, allow_nan=False) + "\n")
        channel.flush()

    def read():
        line = sys.stdin.readline()
        if not line:
            raise EOFError()
        return json.loads(line)

    # 公开事件在一局内只追加：宿主只发增量，worker 缓存已重建的对象。
    # 这条路径占了快策略近四成时间（全量事件平均 26KB/动作）。
    state = {"events": [], "objects": []}

    def dispatch(policy, request):
        op = request["op"]
        if op == "act":
            data = dict(request["observation"])
            if "events_from" in data:
                base = data.pop("events_from")
                fresh = data.pop("events")
                if base != len(state["events"]):
                    raise ValueError("公开事件增量与缓存不匹配")
                state["events"].extend(fresh)
                state["objects"].extend(PublicEvent(**e) for e in fresh)
            else:
                events = data.pop("public_events")
                state["events"] = list(events)
                state["objects"] = [PublicEvent(**e) for e in events]
            obs = observation(data, tuple(state["objects"]))
            started = time.perf_counter()
            action = policy.act(obs)
            elapsed = time.perf_counter() - started
            return {
                "index": obs.legal_actions.index(action),
                "compute_seconds": elapsed,
            }
        if op == "reset":
            state["events"] = []
            state["objects"] = []
            policy.reset(request["seed"])
        elif op == "finish":
            policy.finish_round(RoundResult(**request["result"]), request["seat"])
        else:
            raise ValueError("unknown policy operation")
        return {"ok": True}

    try:
        module_spec = importlib.util.spec_from_file_location(
            "candidate", package / "strategy.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        policy = None
        while True:
            request = read()
            if request["op"] == "build":
                policy = module.build(package / "model")
                if not all(
                    callable(getattr(policy, name, None))
                    for name in ("reset", "act", "finish_round")
                ):
                    raise TypeError(
                        "build must return reset, act and finish_round methods"
                    )
                # Existing candidates are expected to make reset() a complete
                # game boundary. A candidate may opt out when it deliberately
                # keeps process-global state that reset cannot clear.
                send(
                    {
                        "ok": True,
                        "reuse_process": bool(getattr(module, "PROCESS_REUSE", True)),
                    }
                )
            else:
                send(dispatch(policy, request))
    except EOFError:
        pass
    except Exception as exc:
        frames = [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(exc.__traceback__)[-12:]
        ]
        send({"error": f"{type(exc).__name__}: {exc}", "traceback": frames})


if __name__ == "__main__":
    main()
