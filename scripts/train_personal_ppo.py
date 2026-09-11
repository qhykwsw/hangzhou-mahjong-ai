#!/usr/bin/env python3
"""Run a small real rollout for the hand-maintained PPO project."""

import argparse
from pathlib import Path
import time

from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.execution.candidate_runtime import train_package
from mahjong_ai.strategies.common import PolicySpec


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="projects/personal/ppo")
    parser.add_argument("--output", default="models")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    project = Path(args.project).resolve()
    package = project / "src"
    output = (project / args.output).resolve()
    config = RoundConfig(rounds=8)
    opponents = [PolicySpec() for _ in range(3)]
    result = train_package(
        package,
        output,
        config,
        opponents,
        args.episodes,
        args.seed,
        time.monotonic() + 900,
        runtime=str(Path(__file__).parents[1] / "src"),
    )
    print(result)


if __name__ == "__main__":
    main()
