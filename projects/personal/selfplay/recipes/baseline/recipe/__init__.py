"""Baseline recipe: teacher-assisted PPO whose Actor step size tracks a per-rollout KL target."""

from .loop import train

__all__ = ["train"]
