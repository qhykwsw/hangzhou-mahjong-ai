"""Training-only Critic. It may read anything critic_features provides."""

import torch
from torch import nn

from guided.networks import HistoryEncoder, PopArt


class Critic(nn.Module):
    def __init__(self, dim=128, layers=2, heads=4):
        super().__init__()
        self.encoder = HistoryEncoder(dim, layers, heads)
        self.oracle = nn.Sequential(
            nn.Linear(16 * 34 + 4, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.merge = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.head = PopArt(dim)

    def features(self, planes, scalars, events, oracle_planes, oracle_scalars):
        state, _ = self.encoder(planes, scalars, events)
        oracle = self.oracle(
            torch.cat(
                [oracle_planes.float().flatten(1), oracle_scalars.float()], dim=-1
            )
        )
        return self.merge(torch.cat([state, oracle], dim=-1))

    def normalized_value(self, planes, scalars, events, oracle_planes, oracle_scalars):
        return self.head.normalized(
            self.features(planes, scalars, events, oracle_planes, oracle_scalars)
        )

    def forward(self, planes, scalars, events, oracle_planes, oracle_scalars):
        return self.head(
            self.features(planes, scalars, events, oracle_planes, oracle_scalars)
        )
