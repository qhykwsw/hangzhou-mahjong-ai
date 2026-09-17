"""Shared pieces: event embedding, oracle encoder, two-hot symlog value head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import K_EVENTS
from env import N_ORACLE_PLANES, N_ORACLE_SCALARS


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


class EventEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kind = nn.Embedding(7, dim)
        self.seat = nn.Embedding(5, dim)
        self.tile = nn.Embedding(35, dim)
        self.aux = nn.Embedding(35, dim)
        self.hand = nn.Embedding(4, dim)
        self.pos = nn.Parameter(torch.zeros(K_EVENTS, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, events):
        return (
            self.kind(events[..., 0])
            + self.seat(events[..., 1])
            + self.tile(events[..., 2])
            + self.aux(events[..., 3])
            + self.hand(events[..., 4])
            + self.pos
        )


class OracleEncoder(nn.Module):
    out_dim = 128

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_ORACLE_PLANES * 34 + N_ORACLE_SCALARS, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_dim),
            nn.ReLU(),
        )

    def forward(self, planes, scalars):
        return self.net(torch.cat([planes.flatten(1), scalars], -1))


class TwoHot:
    """Categorical value distribution over symlog-spaced bins."""

    def __init__(self, n_bins, value_range, device):
        self.bins = torch.linspace(-value_range, value_range, n_bins, device=device)
        self.n = n_bins
        self.range = value_range

    def targets(self, y):
        z = symlog(y).clamp(-self.range, self.range)
        pos = (z + self.range) / (2 * self.range) * (self.n - 1)
        lo = pos.floor().long().clamp(0, self.n - 1)
        hi = (lo + 1).clamp(max=self.n - 1)
        w_hi = (pos - lo.float()).clamp(0, 1)
        target = torch.zeros(y.shape[0], self.n, device=y.device)
        target.scatter_add_(1, lo[:, None], (1 - w_hi)[:, None])
        target.scatter_add_(1, hi[:, None], w_hi[:, None])
        return target

    def expectation(self, logits):
        return symexp(torch.softmax(logits, -1) @ self.bins)


def mlp(sizes, last_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """Trunk-specific subclasses implement `features` and `policy_logits`."""

    feature_dim: int

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_bins = cfg.value_bins

    def build_heads(self):
        d = self.feature_dim + OracleEncoder.out_dim
        self.oracle = OracleEncoder()
        self.value_head = mlp([d, 256, self.n_bins])
        self.aux_head = mlp([d, 128, 2])

    def features(self, planes, scalars, events):
        raise NotImplementedError

    def policy_logits(self, feats):
        raise NotImplementedError

    def forward(self, planes, scalars, events, oracle_planes=None, oracle_scalars=None):
        feats = self.features(planes, scalars, events)
        logits = self.policy_logits(feats)
        if oracle_planes is None:
            return logits, None, None
        z = torch.cat([feats[0], self.oracle(oracle_planes, oracle_scalars)], -1)
        return logits, self.value_head(z), self.aux_head(z)

    def actor_logits(self, planes, scalars, events):
        return self.policy_logits(self.features(planes, scalars, events))


class ActorOnly(nn.Module):
    """Export wrapper: (planes, scalars, events) -> logits."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, planes, scalars, events):
        return self.net.actor_logits(planes, scalars, events)
