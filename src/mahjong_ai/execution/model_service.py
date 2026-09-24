"""The control plane's single inference service for model seats.

Started on first use and shared by every local match worker, so requests are
batched on one device. The service code lives in the self-play project
(``guided.model_server``) because it needs torch and the Actor encoding.
"""

import atexit
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from mahjong_ai.execution.model_seat import SERVER_ENV

__all__ = ["ModelService", "SERVER_ENV"]


class ModelService:
    def __init__(self, research=None, *, root=None, log=None):
        """Serve research's models, or any directory of exported Actors (root)."""
        self.root = Path(root) if root else research.layout.research / "models"
        # The control plane runs from the repository checkout, next to the
        # self-play project that provides the service.
        self.project = Path(__file__).resolve().parents[3] / "projects/personal/selfplay"
        self.log = Path(log) if log else research.layout.research / "model-server.log"
        self.process = None
        self.port = None
        self.warmed = set()

    def warm(self, weights):
        """Load models before play: a cold load must not fall inside an action window."""
        url = self.url()
        for sha in sorted(set(weights) - self.warmed):
            request = urllib.request.Request(
                url + "/load", json.dumps({"model": sha}).encode(),
                {"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as reply:
                    if not json.loads(reply.read()).get("ok"):
                        raise RuntimeError("模型预热失败：" + sha)
            except urllib.error.HTTPError as exc:
                # The service answers a failed load with its reason in the body.
                raise RuntimeError(f"模型预热失败：{sha}（{exc.code}：{exc.read()[:500].decode(errors='replace')}）") from exc
            self.warmed.add(sha)

    def _healthy(self):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/health", timeout=2
            ) as reply:
                return json.loads(reply.read()).get("ok") is True
        except OSError:
            return False

    def url(self):
        if self.process is None or self.process.poll() is not None:
            self.warmed = set()
            self._start()
        return f"http://127.0.0.1:{self.port}"

    def _start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        env = dict(os.environ)
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(key, None)
        with self.log.open("ab") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "guided.model_server", "--root", str(self.root),
                 "--port", str(self.port)],
                cwd=self.project,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        # A one-shot evaluation must not leave the service behind when it exits.
        atexit.register(self.close)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"模型推理服务启动失败，见 {self.log}")
            if self._healthy():
                return
            time.sleep(0.5)
        self.close()
        raise RuntimeError("模型推理服务启动超时")

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
