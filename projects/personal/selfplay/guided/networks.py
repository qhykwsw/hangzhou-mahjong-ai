"""Training-only legal Actor and independently parameterized privileged Critic.

The Actor has no oracle inputs. Padding is ignored even when histories are
left-padded or mixed in a batch. Positional indices count real events, not pads.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


class BatchFirstEncoderLayer(nn.TransformerEncoderLayer):
    """The fixed pre-norm block without MultiheadAttention's sequence transpose.

    Project contiguous [batch, token, feature] rows with one GEMM. The stock
    batch-first MHA first transposes them and its projection becomes a batch
    of small matrix products. Parameter names and the attention equation are
    unchanged. The final block only needs the 34 tile queries and state query:
    history keys/values still participate, but their output rows are unused.
    """

    output_tokens = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if src_mask is not None or is_causal:
            raise ValueError("the policy encoder uses only a padding mask")
        x = self.norm1(src)
        batch, length, dim = x.shape
        attention = self.self_attn
        qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        qkv = qkv.reshape(
            batch, length, 3, attention.num_heads, dim // attention.num_heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.output_tokens is not None:
            selected = self.output_tokens if isinstance(self.output_tokens, slice) else slice(self.output_tokens)
            query = query[:, :, selected]
            src = src[:, selected]
        mask = src_key_padding_mask
        if mask is not None:
            mask = (~mask if mask.dtype == torch.bool else mask)[:, None, None, :]
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        x = x.transpose(1, 2).reshape(batch, src.shape[1], dim)
        src = src + F.linear(x, attention.out_proj.weight, attention.out_proj.bias)
        return src + self.linear2(F.gelu(self.linear1(self.norm2(src))))


class HistoryEncoder(nn.Module):
    """Tile, state and event tokens through a pre-norm transformer.

    ``planes`` and ``scalars`` are the input sizes of the encoding it is built
    for (guided.encoding.INPUTS). Later encodings append planes and scalars,
    so it reads the first ``planes`` and ``scalars`` of whatever it is given.
    ``event_tail`` rows embed the events' last field (guided.encoding.EVENT_TAILS):
    before v3 that field is always 0 and an encoder built for 4 rows reads it as
    0 whatever it is given.
    """

    def __init__(self, dim, layers, heads, max_events=4096, *, state_only=False,
                 planes=N_PLANES, scalars=N_SCALARS, event_tail=4):
        super().__init__()
        if dim % heads or dim < 2:
            raise ValueError("dim must be divisible by heads and at least 2")
        self.dim, self.layers, self.heads = dim, layers, heads
        self.max_events = max_events
        self.state_only = state_only
        self.tile_in = nn.Linear(planes, dim)
        self.tile_embedding = nn.Parameter(torch.empty(34, dim))
        self.state_in = nn.Linear(scalars, dim)
        self.event_embeddings = nn.ModuleList(
            nn.Embedding(size, dim) for size in (7, 5, 35, 35, event_tail)
        )
        self.timed_events = event_tail > 4
        self.type_embedding = nn.Parameter(torch.empty(3, dim))
        nn.init.normal_(self.tile_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        positions = torch.arange(max_events + 1, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        positional = torch.zeros(max_events + 1, dim)
        positional[:, 0::2] = torch.sin(positions * frequencies)
        positional[:, 1::2] = torch.cos(positions * frequencies[: dim // 2])
        self.register_buffer("positions", positional, persistent=False)
        layer = BatchFirstEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            layers,
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones its template; initialize each layer separately.
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(block.linear1.weight)
            nn.init.xavier_uniform_(block.linear2.weight)
        self.transformer.layers[-1].output_tokens = slice(34, 35) if state_only else 35
        self.norm = nn.LayerNorm(dim)

    def forward(self, planes, scalars, events):
        valid = events[..., 0] != 0
        # Ignore every field of padding, including arbitrary sentinel values.
        events = events.long().masked_fill(~valid[..., None], 0)
        if events.shape[1] > self.max_events:
            raise ValueError(
                f"history exceeds {self.max_events} events; refusing truncation"
            )
        tail = events[..., 4] if self.timed_events else torch.zeros_like(events[..., 4])
        event_tokens = sum(
            embedding(events[..., field])
            for field, embedding in enumerate(self.event_embeddings[:4])
        ) + self.event_embeddings[4](tail)
        event_tokens = event_tokens + self.positions[valid.long().cumsum(1)]
        planes = planes[:, : self.tile_in.in_features]
        scalars = scalars[:, : self.state_in.in_features]
        tiles = self.tile_in(planes.float().transpose(1, 2)) + self.tile_embedding
        state = self.state_in(scalars.float())[:, None]
        tokens = torch.cat(
            [
                tiles + self.type_embedding[0],
                state + self.type_embedding[1],
                event_tokens + self.type_embedding[2],
            ],
            dim=1,
        )
        padding = torch.cat(
            [
                torch.zeros(
                    (events.shape[0], 35), dtype=torch.bool, device=events.device
                ),
                ~valid,
            ],
            dim=1,
        )
        encoded = self.norm(self.transformer(tokens, src_key_padding_mask=padding))
        if self.state_only:
            return encoded[:, 0], encoded[:, :0]
        return encoded[:, 34], encoded[:, :34]


def actor_of(encoding, dim, layers, heads):
    """An Actor built for ``encoding``'s inputs (guided.encoding.INPUTS and EVENT_TAILS)."""
    from .encoding import EVENT_TAILS, inputs

    planes, scalars = inputs(encoding)
    return Actor(dim, layers, heads, planes=planes, scalars=scalars, event_tail=EVENT_TAILS[encoding])


class Actor(nn.Module):
    def __init__(self, dim=128, layers=4, heads=4, *, planes=N_PLANES, scalars=N_SCALARS, event_tail=4):
        super().__init__()
        self.dim, self.layers, self.heads = dim, layers, heads
        self.encoder = HistoryEncoder(dim, layers, heads, planes=planes, scalars=scalars, event_tail=event_tail)
        self.discard_head = nn.Linear(dim, 1)
        self.claim_head = nn.Linear(dim, 4)
        self.state_head = nn.Linear(dim, 65)

    def forward(self, planes, scalars, events):
        state, tiles = self.encoder(planes, scalars, events)
        discard = self.discard_head(tiles).squeeze(-1)
        claims = self.claim_head(tiles[:, :33])
        rest = self.state_head(state)
        # Canonical selfplay_core.actions order; WHITE cannot be claimed.
        logits = torch.cat(
            [
                discard,
                rest[:, :63],
                claims[..., 0],
                claims[..., 1],
                claims[..., 2],
                claims[..., 3],
                rest[:, 63:65],
            ],
            dim=-1,
        )
        assert logits.shape[-1] == N_ACTIONS
        return logits


class PopArt(nn.Module):
    """Scalar MSE target normalization that preserves unnormalized predictions.

    Statistics use EMA moments (first batch initializes them). update_stats applies
    the exact affine last-layer correction. Optimizer moments for this last layer
    are discarded after a coordinate change: Adam's old moments are not valid in
    the new normalized coordinates. All backbone optimizer state is preserved.
    Normalization and squared loss are affine in the raw return, so the optimum
    remains its arithmetic conditional mean, unlike symlog regression.
    """

    def __init__(self, dim, beta=0.001, min_std=1.0):
        super().__init__()
        if not 0 < beta <= 1 or min_std <= 0:
            raise ValueError("invalid PopArt beta or minimum standard deviation")
        self.linear = nn.Linear(dim, 1)
        self.beta, self.min_std = beta, min_std
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("second", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("std", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("initialized", torch.tensor(False))

    def normalized(self, features):
        return self.linear(features).squeeze(-1)

    def normalize(self, target):
        return (target - self.mean.to(target.dtype)) / self.std.to(target.dtype)

    def forward(self, features):
        normalized = self.normalized(features)
        return normalized * self.std.to(normalized.dtype) + self.mean.to(
            normalized.dtype
        )

    @torch.no_grad()
    def update_stats(self, targets, optimizer=None):
        targets = targets.detach().to(device=self.mean.device, dtype=torch.float64)
        if targets.numel() == 0 or not torch.isfinite(targets).all():
            raise ValueError("PopArt requires nonempty finite targets")
        old_mean, old_std = self.mean.clone(), self.std.clone()
        rate = self.beta if bool(self.initialized) else 1.0
        mean = (1 - rate) * self.mean + rate * targets.mean()
        second = (1 - rate) * self.second + rate * targets.square().mean()
        std = (second - mean.square()).clamp(min=self.min_std**2).sqrt()
        if not torch.isfinite(torch.stack([mean, second, std])).all():
            raise FloatingPointError("nonfinite PopArt statistics")
        self.linear.weight.mul_((old_std / std).to(self.linear.weight.dtype))
        self.linear.bias.copy_(
            ((old_std * self.linear.bias.double() + old_mean - mean) / std).to(
                self.linear.bias.dtype
            )
        )
        self.mean.copy_(mean)
        self.second.copy_(second)
        self.std.copy_(std)
        self.initialized.fill_(True)
        if optimizer is not None:
            for parameter in self.linear.parameters():
                optimizer.state.pop(parameter, None)


class Critic(nn.Module):
    def __init__(self, dim=128, layers=2, heads=4):
        super().__init__()
        self.dim, self.layers, self.heads = dim, layers, heads
        self.encoder = HistoryEncoder(dim, layers, heads)
        self.oracle = nn.Sequential(
            nn.Linear(16 * 34 + 4, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.merge = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.head = PopArt(dim)

    def features(self, planes, scalars, events, oracle_planes, oracle_scalars):
        state, _ = self.encoder(planes, scalars, events)
        oracle = self.oracle(
            torch.cat(
                [
                    oracle_planes.float().flatten(1),
                    oracle_scalars.float(),
                ],
                dim=-1,
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
