"""Root rollout and information-set tree search over independent sampled worlds."""

import math
import numpy as np
import time
from collections import defaultdict
from mahjong_ai.sdk.simulation import sample_world, terminal_value
from mahjong_ai.strategies.common import RulePolicy


class SearchPolicy(RulePolicy):
    def __init__(
        self, algorithm="rollout", simulations=32, depth=4, exploration=1.4, seconds=0.7
    ):
        if (
            algorithm not in ("rollout", "mcts")
            or type(simulations) is not int
            or simulations < 1
            or type(depth) is not int
            or depth < 1
        ):
            raise ValueError("invalid search dimensions")
        if not 0 < seconds < 10 or not 0 <= exploration < 100:
            raise ValueError("invalid search budget")
        super().__init__()
        self.algorithm = algorithm
        self.simulations = simulations
        self.depth = depth
        self.exploration = exploration
        self.seconds = seconds
        self.last_search = {}

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        actions = obs.legal_actions
        if len(actions) == 1:
            return actions[0]
        hu = next((a for a in actions if a.action == "hu"), None)
        if hu:
            return hu
        start = time.monotonic()
        seat = obs.stage.seat
        tree = defaultdict(lambda: [0, 0.0])
        completed = 0
        for simulation in range(max(len(actions), self.simulations)):
            if simulation and time.monotonic() - start >= self.seconds:
                break
            world = sample_world(obs, int(self.rng.integers(0, 2**63)))
            path = []
            decisions = 0
            while not world.settled and world.steps < 1000:
                current = world.observation()
                legal = current.legal_actions
                if current.stage.seat == seat and (
                    decisions == 0
                    or self.algorithm == "mcts"
                    and decisions < self.depth
                ):
                    # Information-set key excludes sampled opponent hands and wall.
                    key = (
                        decisions,
                        current.hand,
                        current.phase,
                        tuple(tuple(r) for r in current.discards),
                    )
                    total = sum(tree[(key, a)][0] for a in legal)
                    chosen = max(
                        legal,
                        key=lambda a: (
                            float("inf")
                            if not tree[(key, a)][0]
                            else tree[(key, a)][1] / tree[(key, a)][0]
                            + self.exploration
                            * math.sqrt(math.log(total + 1) / tree[(key, a)][0])
                        ),
                    )
                    path.append((key, chosen))
                    decisions += 1
                else:
                    chosen = super().act(current)
                world.step(chosen)
            if not world.settled:
                continue
            value = terminal_value(world, seat)
            for key in path:
                tree[key][0] += 1
                tree[key][1] += value
            completed += 1
        key = (0, obs.hand, obs.phase, tuple(tuple(r) for r in obs.discards))
        self.last_search = {
            "algorithm": self.algorithm,
            "simulations": completed,
            "nodes": len(tree),
        }
        if not completed:
            return super().act(obs)
        return max(
            actions,
            key=lambda a: (
                tree[(key, a)][1] / tree[(key, a)][0] if tree[(key, a)][0] else -1
            ),
        )
