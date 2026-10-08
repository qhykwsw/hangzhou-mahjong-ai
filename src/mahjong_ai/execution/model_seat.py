"""A match seat played by an exported Actor through the host's inference service.

The service (``guided.model_server``) owns the device and batches requests from
every match worker; this client only forwards the public observation. Models
are addressed by the SHA-256 of their weights, which is also the artifact
directory name below the service root on every host.

A seat may hold a second model and a rule for when to play it instead
(``parameters["switch"]`` of a model policy, see Switch); the rule reads the
observation's stage and nothing else. Without one a seat sends exactly what it
did before switching existed.
"""

import http.client
import json
import math
import os
import re
import sys
import threading
import time
from functools import lru_cache
from urllib.parse import urlparse

from mahjong_ai.domain.types import StageKind
from mahjong_ai.execution.candidate_runtime import ActionUnavailable, observation_payload

SERVER_ENV = "MAHJONG_MODEL_SERVER"
# The self-play project, whose encoder turns an observation into Actor input.
# Encoding here, in every match worker, keeps the shared service to batching
# and inference (it was about half of the service's time).
ENCODER_ENV = "MAHJONG_SELFPLAY"
_ENCODER = []
_ENCODER_LOCK = threading.Lock()


def encoder():
    with _ENCODER_LOCK:  # many tables start at once; import once, not in a race
        return _encoder()


def _encoder():
    if not _ENCODER:
        path = os.environ.get(ENCODER_ENV)
        found = None
        if path:
            # Appended after the frozen runtime, which keeps providing mahjong_ai.
            for entry in (path, os.path.join(path, "src")):
                if entry not in sys.path:
                    sys.path.append(entry)
            from guided.encoding import encode
            from selfplay_core.actions import action_index

            found = (encode, action_index)
        _ENCODER.append(found)
    return _ENCODER[0]


# Places that advance from a stage: a semifinal's top two, a final's winner
# (guided.stage_value.PLACES). Elsewhere a switching seat plays its own model.
STAGE_PLACES = {StageKind.ADVANCEMENT: 2, StageKind.FINAL: 1}
# The switch rules and their parameters' defaults (Switch).
SWITCH_RULES = {"always": {}, "tail": {"hands_left": 4}, "beta": {"threshold": 0.01},
                "contest": {"hands_left": 6, "margin": 40}}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def check_switch(switch):
    """Refuse a malformed ``parameters["switch"]`` of a model policy (ValueError)."""
    if not isinstance(switch, dict) or switch.get("rule") not in SWITCH_RULES:
        raise ValueError(f"a model switch needs a rule, one of {sorted(SWITCH_RULES)}")
    if not (isinstance(switch.get("weights"), str) and _SHA256.fullmatch(switch["weights"])):
        raise ValueError("a model switch needs its model's weights SHA-256")
    extra = set(switch) - {"rule", "weights"} - set(SWITCH_RULES[switch["rule"]])
    if extra:
        raise ValueError(f"the {switch['rule']} switch takes no {sorted(extra)}")
    if "hands_left" in switch and not (type(switch["hands_left"]) is int and switch["hands_left"] >= 1):
        raise ValueError("hands_left must be a positive integer")
    for key in ("threshold", "margin"):
        value = switch.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")


def served_weights(spec):
    """The models a model policy's seat asks the service for: its own and its switch's."""
    parameters = spec["parameters"]
    return {parameters["weights"], *([parameters["switch"]["weights"]] if parameters.get("switch") else ())}


@lru_cache(maxsize=4096)
def stage_beta(kind, scores, seat, hands_left):
    """guided.stage_value.beta as its risk_of calls it for the v4 encoding (the same throughout a hand)."""
    from guided.stage_value import PLACES, beta

    return float(beta(scores, seat, hands_left, PLACES[kind]))


class Switch:
    """When a model seat plays its second model: a rule on the observation's stage alone.

    The rule reads the public StageContext (obs.stage) and the base score: the
    stage kind, the seat, the four stage totals by seat (stage_scores, taken
    in base scores) and the stage horizon (StageContext.effective_horizon,
    this hand's stage_round_no of stage_rounds_total). Hands left count this
    one, as guided.stage_value does: total - current + 1, at least 1. Outside
    a semifinal or a final, or without stage totals, the seat's own model
    plays whatever the rule.

      always   every decision
      tail     hands left <= hands_left
      beta     |beta| >= threshold, beta the risk coefficient of the v4
               encoding (guided.stage_value.beta, per base score, clipped
               to +-0.05)
      contest  hands left <= hands_left and the seat's total within margin
               base scores of the line: the best other total in a final (the
               leader's, or the second's for the leader), the second best
               other total in a semifinal (between second and third place)
    """

    def __init__(self, switch):
        check_switch(switch)
        self.weights, self.rule = switch["weights"], switch["rule"]
        self.parameters = dict(SWITCH_RULES[self.rule],
                               **{k: v for k, v in switch.items() if k not in ("rule", "weights")})
        if self.rule == "beta" and not encoder():
            raise RuntimeError(f"the beta switch reads guided.stage_value: {ENCODER_ENV} is not set")

    def fires(self, observation):
        ctx = observation.stage
        places = STAGE_PLACES.get(ctx.kind)
        if places is None or ctx.stage_scores is None:
            return False
        if self.rule == "always":
            return True
        current, total = ctx.effective_horizon
        left = max(1, total - current + 1)
        if self.rule == "tail":
            return left <= self.parameters["hands_left"]
        scores = tuple(s / observation.base_score for s in ctx.stage_scores)
        if self.rule == "beta":
            return abs(stage_beta(ctx.kind, scores, ctx.seat, left)) >= self.parameters["threshold"]
        line = sorted((s for seat, s in enumerate(scores) if seat != ctx.seat), reverse=True)[places - 1]
        return left <= self.parameters["hands_left"] and abs(scores[ctx.seat] - line) <= self.parameters["margin"]


class ModelPolicy:
    def __init__(self, weights, version, action_seconds=10.0, switch=None):
        self.weights, self.version, self.action_seconds = weights, version, action_seconds
        self.stats = {"actions": 0, "timeouts": 0, "errors": 0, "elapsed_seconds": 0.0,
                      "compute_seconds": 0.0, "ipc_seconds": 0.0, "fallback_actions": 0}
        # A second model and when to play it; its decisions are counted apart.
        self.switch = None if switch is None else Switch(switch)
        if self.switch is not None:
            self.stats["alt_actions"] = 0
        self.last_error = None
        self.latencies = []
        self.connection = None
        encoder()  # import the encoder while building, not inside a first action window

    def _connect(self):
        url = os.environ.get(SERVER_ENV)
        if not url:
            raise RuntimeError(f"{SERVER_ENV} is not set; no inference service for model seats")
        parsed = urlparse(url)
        if encoder():
            from selfplay_core.wire import SocketClient

            self.connection = SocketClient(parsed.hostname, parsed.port, self.action_seconds)
        else:
            self.connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=self.action_seconds)

    def start(self):
        if self.connection is None:
            self._connect()

    def reset(self, seed):
        self.start()

    def act(self, observation):
        started = time.monotonic()
        self.stats["actions"] += 1
        try:
            weights = self.weights
            if self.switch is not None and self.switch.fires(observation):
                weights = self.switch.weights
                self.stats["alt_actions"] += 1
            if self.connection is None:
                self._connect()
            codec = encoder()
            if codec:
                encode, action_index = codec
                row = encode(observation, lookahead=True)
                from selfplay_core.wire import pack as pack_inputs

                body = pack_inputs(weights, row, lookahead=True)
                reply = self.connection.decide(body)
            else:
                data, _ = observation_payload(observation, None)
                body = json.dumps({"model": weights, "observation": data})
                self.connection.request("POST", "/act", body, {"Content-Type": "application/json"})
                response = self.connection.getresponse()
                reply = json.loads(response.read())
                if response.status != 200:
                    raise RuntimeError(reply.get("error", f"HTTP {response.status}"))
            if codec:
                legal = [action_index(a) for a in observation.legal_actions]
                if reply["index"] not in legal:
                    raise ValueError("model chose an illegal action")
                index = legal.index(reply["index"])
            else:
                index = reply["legal_index"]
            if type(index) is not int or not 0 <= index < len(observation.legal_actions):
                raise ValueError("model returned an invalid action index")
            return observation.legal_actions[index]
        except TimeoutError as exc:
            self.last_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            self.stats["timeouts"] += 1
            self.close()
        except Exception as exc:
            self.last_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            self.stats["errors"] += 1
            self.close()
        finally:
            elapsed = time.monotonic() - started
            self.stats["elapsed_seconds"] += elapsed
            self.stats["ipc_seconds"] += elapsed
            self.latencies.append(elapsed)
            if len(self.latencies) > 4096:
                del self.latencies[:2048]
        self.stats["fallback_actions"] += 1
        raise ActionUnavailable("模型座位未给出合法动作")

    def finish_round(self, result, seat):
        pass

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
