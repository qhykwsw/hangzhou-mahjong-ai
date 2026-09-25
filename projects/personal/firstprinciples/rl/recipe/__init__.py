"""Baseline recipe: repeated small, exactly on-policy policy-gradient steps.

Each step plays a large fresh rollout with the current Actor, scores every
learner decision by its hand's remaining score plus a dealer premium, subtracts
a cross-fitted Critic, and moves the Actor along the full-batch policy gradient
until KL(current || new) reaches a small fixed target. See loop.py.
"""

def train(ctx):
    # Workers import recipe.features and recipe.returns, which need NumPy only.
    # Load the learner and torch solely in the training parent.
    from .loop import train as run

    return run(ctx)

__all__ = ["train"]
