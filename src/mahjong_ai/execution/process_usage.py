"""RSS accounting for sandbox launchers and their payload processes."""

from pathlib import Path
import subprocess
import sys


def memory_kb(pids):
    if sys.platform == "linux":
        return {pid: _linux_memory_kb(pid, set()) for pid in pids}
    rows = subprocess.run(
        ["/bin/ps", "-o", "pid=,rss=", "-p", ",".join(str(pid) for pid in pids)],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    result = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            result[int(fields[0])] = int(fields[1])
    return result


def _linux_memory_kb(pid, visited):
    if pid in visited:
        return 0
    visited.add(pid)
    root = Path("/proc") / str(pid)
    total = 0
    try:
        for line in (root / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                total = int(line.split()[1])
                break
        children = (root / "task" / str(pid) / "children").read_text().split()
    except (OSError, ValueError):
        return total
    return total + sum(_linux_memory_kb(int(child), visited) for child in children)


def peak_memory_bytes(usage):
    if usage is None:
        return 0
    return usage.ru_maxrss * (1024 if sys.platform == "linux" else 1)
