"""A frozen service must not import a different checkout via editable paths."""

import io
import json
import subprocess
import sys

import pytest

from mahjong_ai.execution.model_service import ModelService
from mahjong_ai.execution.remote import RemoteHost


@pytest.mark.parametrize("remote", [False, True])
def test_service_uses_its_own_sdk_before_inherited_checkout(tmp_path, monkeypatch, remote):
    code = tmp_path / "frozen"
    project = code / "projects/personal/selfplay"
    project.mkdir(parents=True)
    for root, marker in [(code / "src", "frozen"), (tmp_path / "editable", "old")]:
        package = root / "mahjong_ai"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"ORIGIN = {marker!r}\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "editable"))
    real_popen = subprocess.Popen
    origins = []

    class Process:
        def poll(self):
            return None

    def start(args, **kwargs):
        # Exercise Python's real import resolution under the launcher's env.
        with real_popen(
            [sys.executable, "-c", "import mahjong_ai; print(mahjong_ai.ORIGIN)"],
            cwd=kwargs["cwd"], env=kwargs["env"], stdout=subprocess.PIPE,
        ) as process:
            origins.append(process.communicate()[0].decode().strip())
        return Process()

    monkeypatch.setattr(subprocess, "Popen", start)
    if remote:
        host = RemoteHost("test", {"root": str(tmp_path), "python": sys.executable}, tmp_path)
        host.service_code = str(code)

        def urlopen(*args, **kwargs):
            return io.BytesIO(json.dumps({"ok": bool(origins), "code": str(code)}).encode())

        monkeypatch.setattr("urllib.request.urlopen", urlopen)
        monkeypatch.setattr("time.sleep", lambda _: None)

        def execute(script, **kwargs):
            exec(script, {})
            return bool(origins)

        monkeypatch.setattr(host, "python", execute)
        host.ensure_service(tmp_path / "models", [])
    else:
        service = ModelService(root=tmp_path / "models", log=tmp_path / "service.log")
        service.project = project
        monkeypatch.setattr(service, "_healthy", lambda: True)
        monkeypatch.setattr("atexit.register", lambda _: None)
        service._start()
    assert origins == ["frozen"]
