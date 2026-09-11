"""Exact wheel dependencies, prepared outside candidate and private data access."""

import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from mahjong_ai.execution.candidate_runtime import sandbox_command
from mahjong_ai.persistence.artifacts import atomic_json, file_hash


def clean_env(temp):
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).parents[2]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(temp),
        "TMPDIR": str(temp),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def command(
    argv,
    cwd,
    readable,
    writable,
    root,
    seconds,
    memory_bytes=2 * 1024**3,
    *,
    network=False,
    no_fork=True,
    file_bytes=64 * 1024**2,
):
    """Meter one sandboxed process; RSS is monitored on macOS (not RLIMIT_AS)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    seconds = max(0.01, seconds)
    started = time.monotonic()
    limit = 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="mahjong-workspace-") as temp:
        env = clean_env(temp)
        wrapped = [
            sys.executable,
            "-B",
            "-m",
            "mahjong_ai.execution.limited_exec",
            str(seconds),
            str(file_bytes),
            *argv,
        ]
        out = (root / "stdout.txt").open("wb")
        err = (root / "stderr.txt").open("wb")
        process = subprocess.Popen(
            sandbox_command(
                wrapped,
                [
                    *readable,
                    *mimetypes.knownfiles,
                    "/private/etc/resolv.conf",
                    "/private/etc/hosts",
                ],
                [*writable, temp],
                no_fork=no_fork,
                network=network,
            ),
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=err,
        )
        cause = None
        peak = 0
        usage = None
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                rss = subprocess.run(
                    ["/bin/ps", "-p", str(process.pid), "-o", "rss="],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if rss.isdigit():
                    peak = max(peak, int(rss) * 1024)
                if time.monotonic() - started >= seconds:
                    cause = "wall_limit"
                elif peak > memory_bytes:
                    cause = "memory_limit"
                elif any(
                    (root / name).stat().st_size > limit
                    for name in ("stdout.txt", "stderr.txt")
                ):
                    cause = "output_limit"
                if cause:
                    process.kill()
                    _, status, usage = os.wait4(process.pid, 0)
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                time.sleep(0.05)
        finally:
            if process.returncode is None:
                process.kill()
                process.wait()
            out.close()
            err.close()
    peak = max(peak, usage.ru_maxrss if usage and sys.platform == "darwin" else 0)
    if peak > memory_bytes:
        cause = cause or "memory_limit"
    truncated = False
    for name in ("stdout.txt", "stderr.txt"):
        path = root / name
        if path.stat().st_size > limit:
            with path.open("rb") as stream:
                stream.seek(-limit, 2)
                tail = stream.read()
            path.write_bytes(tail)
            truncated = True
            cause = cause or "output_limit"
    return {
        "status": "completed" if process.returncode == 0 and not cause else "failed",
        "exit_code": process.returncode,
        "reason": cause,
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": usage.ru_utime + usage.ru_stime if usage else 0,
        "peak_memory_bytes": peak,
        "memory_enforcement": "RSS monitor, 50ms interval",
        "output_truncated": truncated,
        "file_limit_bytes": file_bytes,
        "stdout": (root / "stdout.txt").read_text(errors="replace")[-16000:],
        "stderr": (root / "stderr.txt").read_text(errors="replace")[-16000:],
        "stdout_hash": file_hash(root / "stdout.txt"),
        "stderr_hash": file_hash(root / "stderr.txt"),
    }


def pip_command(python, *arguments):
    # Editable-install .pth files may point outside the frozen runtime. Avoid
    # executing them or making pip scan unrelated project source directories.
    prefix = Path(python).parent.parent
    site = (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    bootstrap = (
        f"import sys,runpy; sys.prefix=sys.exec_prefix={str(prefix)!r}; "
        f'sys.path.insert(0,{str(site)!r}); sys.argv=["pip",*sys.argv[1:]]; '
        'runpy.run_module("pip",run_name="__main__")'
    )
    return [str(python), "-S", "-c", bootstrap, *arguments]


def _prepare_environment(requirements, root, deadline):
    if not requirements:
        return sys.executable, {}
    import numpy as np

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    wheels = root / "wheels"
    wheels.mkdir(exist_ok=True)
    reqs = [*requirements]
    if not any(x.lower().startswith("numpy==") for x in reqs):
        reqs.append("numpy==" + np.__version__)
    download = command(
        pip_command(
            sys.executable,
            "download",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--dest",
            str(wheels),
            *reqs,
        ),
        root,
        [sys.prefix, root],
        [root],
        root / "download",
        deadline - time.monotonic(),
        network=True,
        file_bytes=256 * 1024**2,
    )
    if download["status"] != "completed":
        raise RuntimeError("wheel 下载失败：" + download["stderr"][-2000:])
    venv = root / "venv"
    # venv/ensurepip may spawn trusted Python subprocesses; no private files are mounted.
    created = command(
        [sys.executable, "-m", "venv", str(venv)],
        root,
        [root],
        [root],
        root / "venv-log",
        deadline - time.monotonic(),
        no_fork=False,
    )
    if created["status"] != "completed":
        raise RuntimeError("隔离环境创建失败")
    hashes = {p.name: file_hash(p) for p in wheels.glob("*.whl")}
    if not hashes:
        raise ValueError("没有下载到任何 wheel")
    python = str(venv / "bin/python")
    installed = command(
        pip_command(
            python,
            "install",
            "--no-index",
            "--no-deps",
            *[str(p) for p in sorted(wheels.glob("*.whl"))],
        ),
        root,
        [root],
        [root],
        root / "install",
        deadline - time.monotonic(),
        file_bytes=256 * 1024**2,
    )
    if installed["status"] != "completed":
        raise RuntimeError("隔离环境安装 wheel 失败：" + installed["stderr"][-2000:])
    atomic_json(root / "dependencies.json", {"requirements": reqs, "wheels": hashes})
    return python, hashes


def prepare_environment(requirements, root, deadline):
    """Reuse a completed exact-wheel environment; candidates can only read it."""
    if not requirements:
        return sys.executable, {}
    from mahjong_ai.persistence.artifacts import exclusive_lock, read_json

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / "prepare.lock", blocking=True):
        manifest = root / "dependencies.json"
        python = root / "venv/bin/python"
        if manifest.exists() and python.exists():
            record = read_json(manifest)
            if not set(requirements).issubset(record["requirements"]):
                raise ValueError("依赖缓存键不一致")
            if any(
                file_hash(root / "wheels" / name) != value
                for name, value in record["wheels"].items()
            ):
                raise ValueError("冻结的依赖 wheel 发生变化")
            return str(python), record["wheels"]
        return _prepare_environment(requirements, root, deadline)
