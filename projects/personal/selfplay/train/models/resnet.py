import torch
import torch.nn as nn
import torch.nn.functional as F

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS
from .common import EventEmbedding, PolicyNet


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv1d(c, c, 3, padding=1)
        self.b = nn.Conv1d(c, c, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.a(x))
        y = self.b(y)
        return F.relu(x + y)


class ResNet(PolicyNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        c = cfg.resnet_channels
        self.events = EventEmbedding(cfg.event_embed)
        self.event_proj = nn.Linear(cfg.event_embed, 32)
        in_ch = N_PLANES + N_SCALARS + 32
        self.stem = nn.Conv1d(in_ch, c, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(c) for _ in range(cfg.resnet_blocks)])
        self.feature_dim = 256
        self.vec = nn.Linear(c * 34, self.feature_dim)
        self.policy = nn.Linear(c * 34, N_ACTIONS)
        self.build_heads()

    def features(self, planes, scalars, events):
        ev = F.relu(self.event_proj(self.events(events).mean(1)))
        x = torch.cat(
            [planes, scalars[:, :, None].expand(-1, -1, 34), ev[:, :, None].expand(-1, -1, 34)], 1
        )
        h = self.blocks(F.relu(self.stem(x)))
        flat = h.flatten(1)
        return (F.relu(self.vec(flat)), flat)

    def policy_logits(self, feats):
        return self.policy(feats[1])
