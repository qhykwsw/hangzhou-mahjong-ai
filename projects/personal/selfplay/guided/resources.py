"""Portable evaluation concurrency based on available CPU affinity and GPU memory."""

import math
import os
from pathlib import Path


def effective_cpus():
    count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(
            line.split(":", 2)[2] for line in lines if line.startswith("0::")
        )
        root = Path("/sys/fs/cgroup")
        current = root / relative.lstrip("/")
        while current == root or root in current.parents:
            p = current / "cpu.max"
            if p.is_file():
                quota, period = p.read_text().split()
                if quota != "max":
                    count = min(count, max(1, math.floor(int(quota) / int(period))))
            if current == root:
                break
            current = current.parent
    except (OSError, ValueError, StopIteration):
        pass
    return max(1, count)


def choose_resources(cpus, free_gpu_bytes):
    if cpus < 1 or free_gpu_bytes < 0:
        raise ValueError("invalid detected resources")
    workers = max(1, min(32, math.floor(cpus * 0.8)))
    # Reserve ample device memory; this is a starting point, followed by a
    # finite same-schedule calibration rather than an assumption of peak speed.
    budget_tables = max(16, min(2048, int(free_gpu_bytes / 2**30) * 64))
    tables = max(4, min(64, budget_tables // workers))
    tables = max(4, 2 ** int(math.log2(tables)))
    return {
        "cpu_capacity": cpus,
        "free_gpu_bytes": free_gpu_bytes,
        "workers": workers,
        "tables": tables,
        "inference_batch_limit": 128,
    }


def evaluation_resources():
    import torch

    free, _ = torch.cuda.mem_get_info()
    return choose_resources(effective_cpus(), free)
