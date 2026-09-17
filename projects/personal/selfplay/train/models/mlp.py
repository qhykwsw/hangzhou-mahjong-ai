import torch
import torch.nn as nn

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import K_EVENTS, N_PLANES, N_SCALARS
from .common import EventEmbedding, PolicyNet, mlp


class MLPNet(PolicyNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        h = cfg.mlp_hidden
        self.events = EventEmbedding(cfg.event_embed)
        in_dim = N_PLANES * 34 + N_SCALARS + K_EVENTS * cfg.event_embed
        self.trunk = mlp([in_dim] + [h] * cfg.mlp_layers, last_activation=True)
        self.feature_dim = h
        self.policy = nn.Linear(h, N_ACTIONS)
        self.build_heads()

    def features(self, planes, scalars, events):
        x = torch.cat([planes.flatten(1), scalars, self.events(events).flatten(1)], -1)
        return (self.trunk(x), None)

    def policy_logits(self, feats):
        return self.policy(feats[0])
