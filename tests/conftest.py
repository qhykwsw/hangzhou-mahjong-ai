import os


def pytest_xdist_auto_num_workers(config):
    """`-n auto` (the default in pyproject) takes the cores other work leaves free.

    Timing tests (live-window clients, checkpoint deadlines) need their share
    of a core; a worker per core on a machine already running evaluations
    starves them. At least four workers, at most one per core.
    """
    cores = os.cpu_count() or 1
    try:
        busy = os.getloadavg()[0]
    except OSError:
        busy = 0.0
    return max(min(4, cores), min(cores, int(cores - busy)))
