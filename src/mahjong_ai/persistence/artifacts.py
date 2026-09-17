from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def exclusive_lock(path, blocking=False):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f"another writer owns {path}") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_info():
    from importlib.metadata import PackageNotFoundError, version
    import platform
    import sys
    import numpy as np

    try:
        scipy_version = version("scipy")
    except PackageNotFoundError:
        scipy_version = None
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def runtime_fingerprint():
    # 冻结运行时按整棵树复制（只跳过 __pycache__），指纹必须覆盖同一集合：
    # 只认 *.py 会让重建的 C 加速器复用到旧目录。
    source = {
        str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(Path(__file__).parents[1].rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    return digest({"source": source, "environment": environment_info()})
