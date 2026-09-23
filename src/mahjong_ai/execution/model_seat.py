"""A match seat played by an exported Actor through the host's inference service.

The service (``guided.model_server``) owns the device and batches requests from
every match worker; this client only forwards the public observation. Models
are addressed by the SHA-256 of their weights, which is also the artifact
directory name below the service root on every host.
"""

import http.client
import json
import os
import time
from urllib.parse import urlparse

from mahjong_ai.execution.candidate_runtime import ActionUnavailable, observation_payload

SERVER_ENV = "MAHJONG_MODEL_SERVER"


class ModelPolicy:
    def __init__(self, weights, version, action_seconds=10.0):
        self.weights, self.version, self.action_seconds = weights, version, action_seconds
        self.stats = {"actions": 0, "timeouts": 0, "errors": 0, "elapsed_seconds": 0.0,
                      "compute_seconds": 0.0, "ipc_seconds": 0.0, "fallback_actions": 0}
        self.last_error = None
        self.latencies = []
        self.connection = None

    def _connect(self):
        url = os.environ.get(SERVER_ENV)
        if not url:
            raise RuntimeError(f"{SERVER_ENV} is not set; no inference service for model seats")
        parsed = urlparse(url)
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
            if self.connection is None:
                self._connect()
            data, _ = observation_payload(observation, None)
            body = json.dumps({"model": self.weights, "observation": data})
            self.connection.request("POST", "/act", body, {"Content-Type": "application/json"})
            response = self.connection.getresponse()
            reply = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(reply.get("error", f"HTTP {response.status}"))
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
