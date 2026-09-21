"""Baseline training recipe: joint PPO with a fixed teacher cross-entropy term."""

from .loop import train

__all__ = ["train"]
