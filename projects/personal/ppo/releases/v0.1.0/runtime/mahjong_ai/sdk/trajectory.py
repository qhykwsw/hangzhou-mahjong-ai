"""Policy-owned action features and terminal-return labels, without hidden state."""

from dataclasses import dataclass
import numpy as np
from mahjong_ai.strategies.common import feature_matrix


@dataclass
class Decision:
    features: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float = 0.0
    source: str = "executed-terminal-return"


def features(observation):
    return feature_matrix(observation)


def softmax(logits):
    x = np.asarray(logits, dtype=float)
    p = np.exp(x - x.max())
    return p / p.sum()
