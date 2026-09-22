"""Baseline recipe: repeated small, exactly on-policy policy-gradient steps.

Each step plays a large fresh rollout with the current Actor, scores every
learner decision by its hand's remaining score plus a dealer premium, subtracts
a cross-fitted Critic, and moves the Actor along the full-batch policy gradient
until KL(current || new) reaches a small fixed target. See loop.py.
"""

from .loop import train

__all__ = ["train"]
