"""Linux filesystem isolation and a thread-compatible no-fork policy."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import errno
import os
from pathlib import Path
import shutil
import subprocess
import sys

_LAUNCHER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mahjong-sandbox")
_SECCOMP_ALLOW = 0x7FFF0000
_SECCOMP_ERRNO = 0x00050000
_MASKED_EQUAL = 7
_CLONE_THREAD = 0x00010000


def spawn(argv, **kwargs):
    """Keep the launcher thread alive for Bubblewrap's parent-death signal.

    Linux ties that signal to the creating thread, not the whole process.
    Match startup pools may exit while their candidates are still in use.
    Only Popen is serialized here; worker initialization remains parallel.
    """
    return _LAUNCHER.submit(subprocess.Popen, argv, **kwargs).result()


def command(argv, readable, writable, *, no_fork=True, network=False):
    executable = shutil.which("bwrap")
    if not executable:
        raise RuntimeError("Linux/WSL 隔离候选进程需要 bubblewrap 和 libseccomp2")
    options = [
        executable,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    if not network:
        options.append("--unshare-net")
    base = [
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/ssl",
        "/etc/nsswitch.conf",
        sys.base_prefix,
        sys.prefix,
        Path(__file__).parents[2],
    ]
    if network:
        base += ["/etc/resolv.conf", "/etc/hosts"]
    for flag, paths in (("--ro-bind", [*base, *readable]), ("--bind", writable)):
        for path in sorted({str(Path(value).absolute()) for value in paths}):
            if Path(path).exists():
                options += [flag, path, path]
    options += ["--remount-ro", "/"]
    if no_fork:
        argv = [sys.executable, "-B", "-m", __name__, *argv]
    return [*options, "--", *argv]


class ArgumentComparison(ctypes.Structure):
    _fields_ = [
        ("argument", ctypes.c_uint),
        ("operation", ctypes.c_int),
        ("mask", ctypes.c_uint64),
        ("value", ctypes.c_uint64),
    ]


def restrict_fork():
    try:
        library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise RuntimeError(
            "Linux/WSL 候选进程需要 libseccomp2，不能禁用进程隔离"
        ) from exc
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ArgumentComparison),
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(_SECCOMP_ALLOW)
    if not context:
        raise RuntimeError("无法创建候选进程 seccomp 过滤器")
    try:
        for name, error in (
            ("fork", errno.EPERM),
            ("vfork", errno.EPERM),
            ("clone", errno.EPERM),
            ("clone3", errno.ENOSYS),
        ):
            syscall = library.seccomp_syscall_resolve_name(name.encode())
            if syscall == -1:
                raise RuntimeError(f"libseccomp 不支持 {name}，请更新 libseccomp2")
            comparison = ArgumentComparison(0, _MASKED_EQUAL, _CLONE_THREAD, 0)
            result = library.seccomp_rule_add_array(
                context,
                _SECCOMP_ERRNO | error,
                syscall,
                1 if name == "clone" else 0,
                ctypes.byref(comparison) if name == "clone" else None,
            )
            if result < 0:
                raise OSError(-result, f"seccomp rule: {name}")
        result = library.seccomp_load(context)
        if result < 0:
            raise OSError(-result, "seccomp_load")
    finally:
        library.seccomp_release(context)


if __name__ == "__main__":
    restrict_fork()
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
