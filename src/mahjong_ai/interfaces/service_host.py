"""Host-specific lifecycle of the web page service (launchd or systemd)."""

import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time


def open_url(url):
    commands = ["open"] if sys.platform == "darwin" else ["wslview", "xdg-open"]
    for name in commands:
        executable = shutil.which(name)
        if executable:
            try:
                subprocess.run(
                    [executable, url], check=True, capture_output=True, timeout=10
                )
                return
            except (OSError, subprocess.SubprocessError):
                continue
    print("浏览器未能自动打开，请访问：" + url, file=sys.stderr)


def start_service(root, label, project, argv, env):
    if sys.platform == "darwin":
        _start_launchd(root, label, project, argv, env)
    elif sys.platform == "linux":
        _start_systemd(root, label, project, argv, env)
    else:
        raise RuntimeError("网页服务支持 macOS 和 Linux/WSL2")


def _start_launchd(root, label, project, argv, env):
    plist = root / "service.plist"
    payload = dict(
        Label=label,
        ProgramArguments=argv,
        WorkingDirectory=str(Path(project).resolve()),
        EnvironmentVariables=env,
        RunAtLoad=True,
        KeepAlive=False,
        ExitTimeOut=15,
        ProcessType="Interactive",
        StandardOutPath=str(root / "service.log"),
        StandardErrorPath=str(root / "service.log"),
    )
    plist.write_bytes(plistlib.dumps(payload))
    plist.chmod(0o600)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain + "/" + label], capture_output=True)
    for attempt in range(50):
        launched = subprocess.run(
            ["launchctl", "bootstrap", domain, str(plist)],
            capture_output=True,
            text=True,
        )
        if launched.returncode == 0:
            return
        if launched.returncode != 5 or attempt == 49:
            raise RuntimeError(
                "launchd could not start service: " + launched.stderr.strip()
            )
        time.sleep(0.1)


def _start_systemd(root, label, project, argv, env):
    if not shutil.which("systemd-run") or not shutil.which("systemctl"):
        raise RuntimeError("Linux/WSL2 后台需要启用 systemd 用户服务")
    environment = root / "service.env"
    with open(
        environment, "w", opener=lambda path, flags: os.open(path, flags, 0o600)
    ) as stream:
        for key, value in env.items():
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            stream.write(f'{key}="{escaped}"\n')
    environment.chmod(0o600)
    log = root / "service.log"
    subprocess.run(
        ["systemctl", "--user", "stop", label + ".service"], capture_output=True
    )
    launched = subprocess.run(
        [
            "systemd-run",
            "--user",
            "--quiet",
            "--collect",
            "--service-type=exec",
            "--unit=" + label,
            "--property=WorkingDirectory=" + str(Path(project).resolve()),
            "--property=EnvironmentFile=" + str(environment),
            "--property=StandardOutput=append:" + str(log),
            "--property=StandardError=append:" + str(log),
            "--property=KillMode=mixed",
            "--property=TimeoutStopSec=15",
            "--",
            *[argument.replace("$", "$$") for argument in argv],
        ],
        capture_output=True,
        text=True,
    )
    if launched.returncode:
        raise RuntimeError(
            "systemd 用户服务启动失败："
            + (
                launched.stderr.strip()
                or f"请查看 journalctl --user -u {label}.service"
            )
        )


def stop_service(root, label, previous=None):
    if sys.platform == "linux":
        present = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                label + ".service",
                "--property=LoadState",
                "--value",
            ],
            capture_output=True,
            text=True,
        )
        if present.returncode:
            raise RuntimeError("无法查询 systemd 用户服务：" + present.stderr.strip())
        if present.stdout.strip() == "not-found":
            return False
        stopped = subprocess.run(
            ["systemctl", "--user", "stop", label + ".service"],
            capture_output=True,
            text=True,
        )
        if stopped.returncode:
            raise RuntimeError("systemd 用户服务停止失败：" + stopped.stderr.strip())
        return True
    if sys.platform != "darwin":
        raise RuntimeError("网页服务支持 macOS 和 Linux/WSL2")
    domain = f"gui/{os.getuid()}/{label}"
    stopped = subprocess.run(["launchctl", "bootout", domain], capture_output=True)
    if stopped.returncode:
        return False
    for _ in range(100):
        present = subprocess.run(["launchctl", "print", domain], capture_output=True)
        alive = (
            subprocess.run(
                ["/bin/ps", "-p", str(previous["pid"]), "-o", "command="],
                capture_output=True,
                text=True,
            ).stdout
            if previous
            else ""
        )
        if present.returncode != 0 and "mahjong_ai.interfaces.app" not in alive:
            return True
        time.sleep(0.1)
    raise RuntimeError("网页服务尚未停止，请检查服务日志")
