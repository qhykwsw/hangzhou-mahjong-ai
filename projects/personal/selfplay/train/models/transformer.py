import torch
import torch.nn as nn

from selfplay_core.actions import CHI, N_ACTIONS
from selfplay_core.encoding import K_EVENTS, N_PLANES, N_SCALARS
from .common import EventEmbedding, PolicyNet


class TransformerNet(PolicyNet):
    """Tokens: 34 tiles + 1 state token + K events.  Discard / peng / kong logits
    come from the tile tokens, chi / hu / pass from the state token."""

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.tf_dim
        self.tile_in = nn.Linear(N_PLANES, d)
        self.tile_emb = nn.Parameter(torch.zeros(34, d))
        self.state_in = nn.Linear(N_SCALARS, d)
        self.events = EventEmbedding(d)
        self.n_events = cfg.tf_events
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        nn.init.normal_(self.tile_emb, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, cfg.tf_heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.tf_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.feature_dim = d
        self.discard_head = nn.Linear(d, 1)
        self.claim_head = nn.Linear(d, 4)  # peng, gang exposed, concealed, added
        self.state_head = nn.Linear(d, 63 + 2)  # chi..., hu, pass
        self.build_heads()

    def features(self, planes, scalars, events):
        tiles = self.tile_in(planes.transpose(1, 2)) + self.tile_emb + self.type_emb[0]
        state = (self.state_in(scalars) + self.type_emb[1])[:, None]
        ev = self.events(events)[:, -self.n_events :] + self.type_emb[2]
        out = self.norm(self.encoder(torch.cat([tiles, state, ev], 1)))
        return (out[:, 34], out[:, :34])

    def policy_logits(self, feats):
        state, tiles = feats
        discard = self.discard_head(tiles).squeeze(-1)  # (B,34)
        claims = self.claim_head(tiles[:, :33])  # (B,33,4)
        rest = self.state_head(state)  # (B,65)
        return torch.cat(
            [
                discard,
                rest[:, :63],
                claims[:, :, 0],
                claims[:, :, 1],
                claims[:, :, 2],
                claims[:, :, 3],
                rest[:, 63:65],
            ],
            -1,
        )
