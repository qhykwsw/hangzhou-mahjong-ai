"""Small, human-maintained PPO project.

The learning implementation is shared with the public strategy SDK.  This
module only fixes the project identity and training recipe, so it can be
frozen as an ordinary candidate package without entering agent research.
"""
from pathlib import Path

from mahjong_ai.strategies.learning import PPOPolicy, train as _train

COMBINATION = "personal-ppo"
FAMILY = "ppo"
PARAMETERS = {
    "hidden": 16,
    "learning_rate": 0.01,
    "clip": 0.2,
    "epochs": 2,
    "batch_size": 32,
    "entropy": 0.01,
    "rollout_games": 1,
}


def build(model_dir):
    """Build an inference policy from a frozen model directory."""
    return PPOPolicy("ppo", Path(model_dir), training=False, **PARAMETERS)


def train(ctx):
    """Run at least one real rollout and persist weights/training evidence."""
    return _train(ctx, "ppo", PARAMETERS)
