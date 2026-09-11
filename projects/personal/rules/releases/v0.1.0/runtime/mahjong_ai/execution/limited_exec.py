"""Set inherited OS limits and replace this process; never imports workspace code."""

import math
import os
import resource
import sys


def main():
    cpu, output = float(sys.argv[1]), int(sys.argv[2])
    resource.setrlimit(
        resource.RLIMIT_CPU, (max(1, math.ceil(cpu)), max(1, math.ceil(cpu)) + 1)
    )
    resource.setrlimit(resource.RLIMIT_FSIZE, (output, output))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    os.execvp(sys.argv[3], sys.argv[3:])


if __name__ == "__main__":
    main()
