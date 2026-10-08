"""Apple Metal execution of a frozen Actor with the same weights and equations.

Linear layers use fp16; embeddings, normalization and residuals stay fp32.
Different Metal kernels can round near ties differently from torch MPS.
Changing this backend requires a new evaluation runtime and reference scores.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Actor:
    def __init__(self, actor):
        import torch

        if getattr(actor.encoder, "oracle_inputs", False):
            # Inference reads legal observations only (ADR-0018).
            raise ValueError("refusing an Actor with oracle inputs (ADR-0018: training only)")
        half = set()
        for name, module in actor.named_modules():
            if isinstance(module, torch.nn.Linear):
                half.update([name + ".weight", name + ".bias"])
            if isinstance(module, torch.nn.MultiheadAttention):
                half.update([name + ".in_proj_weight", name + ".in_proj_bias"])
        weights = actor.state_dict()
        encoder = actor.encoder
        if encoder.suit_embedding:
            # Frozen weights: each shared-suit table (guided.networks.SuitEmbedding)
            # is built once, as torch builds it, and read under its per-tile name.
            from .networks import TILE_FIELDS

            shared = ("encoder.tile_embedding.", *(f"encoder.event_embeddings.{i}." for i in TILE_FIELDS))
            weights = {k: v for k, v in weights.items() if not k.startswith(shared)}
            with torch.no_grad():
                weights["encoder.tile_embedding"] = encoder.tile_embedding.weight
                for i in TILE_FIELDS:
                    weights[f"encoder.event_embeddings.{i}.weight"] = encoder.event_embeddings[i].weight
        self.w = {
            k: mx.array(
                v.detach().cpu().numpy().astype(np.float16 if k in half else np.float32)
            )
            for k, v in weights.items()
        }
        self.positions = mx.array(actor.encoder.positions.detach().cpu().numpy())
        # The prefixes of wider rows this Actor reads, and whether it embeds
        # the events' last field (guided.networks.HistoryEncoder).
        self.planes = actor.encoder.tile_in.in_features
        self.scalars = actor.encoder.state_in.in_features
        self.timed_events = actor.encoder.timed_events
        self.dim, self.heads, self.layers = actor.dim, actor.heads, actor.layers
        self.max_events = actor.encoder.max_events
        # The block before which the belief head feeds the tile tokens (None without one).
        self.belief_layer = actor.encoder.belief_layer
        self.belief_version = actor.encoder.belief_version
        if self.belief_version == 2:
            # As torch computes them (networks.HistoryEncoder._belief_v2): v1's leading
            # channels of the tile head's output layer and belief_in by their own matmuls.
            from .networks import BELIEF_V1_TILES as k

            for name, parts in (("encoder.belief_head.3.weight", (slice(None, k), slice(k, None))),
                                ("encoder.belief_head.3.bias", (slice(None, k), slice(k, None))),
                                ("encoder.belief_in.weight", ((slice(None), slice(None, k)),
                                                              (slice(None), slice(k, None))))):
                whole = np.asarray(self.w.pop(name))
                for i, part in enumerate(parts):
                    self.w[f"{name}.{i}"] = mx.array(np.ascontiguousarray(whole[part]))
        mx.eval(self.w, self.positions)
        self.logits = mx.compile(self.forward)
        self.decide = mx.compile(self.decision)

    def linear(self, x, name):
        return mx.addmm(
            self.w[name + ".bias"], x.astype(mx.float16), self.w[name + ".weight"].T
        )

    def norm(self, x, name):
        return mx.fast.layer_norm(
            x, self.w[name + ".weight"], self.w[name + ".bias"], 1e-5
        )

    def forward(self, planes, scalars, events):
        return self.outputs(planes, scalars, events)[0]

    def believe(self, tiles):
        """The belief head's logits [batch, 34, 3] from the tile tokens (guided.networks.Actor)."""
        h = self.linear(self.norm(tiles, "encoder.belief_head.0"), "encoder.belief_head.1")
        h = nn.gelu(h.astype(mx.float32)).astype(mx.float16)
        return self.linear(h, "encoder.belief_head.3")

    def believe_v2(self, x):
        """v2's beliefs at the belief block and the tokens with them added (networks.HistoryEncoder._belief_v2).

        Returns (tokens, tile logits [batch, 34, 7], state logits [batch, 6]).
        """
        w = self.w
        tiles, state = x[:, :34], x[:, 34]
        h = self.linear(self.norm(tiles, "encoder.belief_head.0"), "encoder.belief_head.1")
        h = nn.gelu(h.astype(mx.float32)).astype(mx.float16)
        out = "encoder.belief_head.3"
        hands = mx.addmm(w[out + ".bias.0"], h, w[out + ".weight.0"].T)
        rest = mx.addmm(w[out + ".bias.1"], h, w[out + ".weight.1"].T)
        p = self.linear(self.norm(state, "encoder.belief_state_head.0"), "encoder.belief_state_head.1")
        p = nn.gelu(p.astype(mx.float32)).astype(mx.float16)
        progress = self.linear(p, "encoder.belief_state_head.3")
        injected = mx.addmm(
            w["encoder.belief_in.bias"], mx.sigmoid(hands.astype(mx.float32)).astype(mx.float16),
            w["encoder.belief_in.weight.0"].T,
        ) + mx.sigmoid(rest.astype(mx.float32)).astype(mx.float16) @ w["encoder.belief_in.weight.1"].T
        state = state + self.linear(mx.sigmoid(progress.astype(mx.float32)), "encoder.belief_state_in")
        x = mx.concatenate([tiles + injected, state[:, None], x[:, 35:]], axis=1)
        return x, mx.concatenate([hands, rest], axis=-1), progress

    def outputs(self, planes, scalars, events):
        """(logits, belief logits: [batch, 3, 34] for v1, (tiles [batch, 7, 34], state [batch, 6]) for v2,
        None without a belief head)."""
        if events.shape[1] > self.max_events:
            raise ValueError(
                f"history exceeds {self.max_events} events; refusing truncation"
            )
        w = self.w
        valid = events[..., 0] != 0
        events = mx.where(valid[..., None], events, 0)
        tail = events[..., 4] if self.timed_events else mx.zeros_like(events[..., 4])
        history = sum(
            w[f"encoder.event_embeddings.{i}.weight"][events[..., i]] for i in range(4)
        ) + w["encoder.event_embeddings.4.weight"][tail]
        planes = planes[:, : self.planes]
        scalars = scalars[:, : self.scalars]
        history = history + self.positions[valid.astype(mx.int32).cumsum(1)]
        tiles = (
            self.linear(planes.transpose(0, 2, 1), "encoder.tile_in")
            + w["encoder.tile_embedding"]
        )
        state = self.linear(scalars, "encoder.state_in")[:, None]
        x = mx.concatenate(
            [
                tiles + w["encoder.type_embedding"][0],
                state + w["encoder.type_embedding"][1],
                history + w["encoder.type_embedding"][2],
            ],
            axis=1,
        )
        mask = mx.concatenate(
            [mx.ones((events.shape[0], 35), mx.bool_), valid], axis=1
        )[:, None, None, :]
        belief = None
        for layer in range(self.layers):
            if layer == self.belief_layer and self.belief_version == 2:
                x, belief, progress = self.believe_v2(x)
            elif layer == self.belief_layer:
                belief = self.believe(x[:, :34])
                injected = self.linear(mx.sigmoid(belief.astype(mx.float32)), "encoder.belief_in")
                x = mx.concatenate([x[:, :34] + injected, x[:, 34:]], axis=1)
            prefix = f"encoder.transformer.layers.{layer}."
            normed = self.norm(x, prefix + "norm1")
            batch, length, dim = normed.shape
            qkv = mx.addmm(
                w[prefix + "self_attn.in_proj_bias"],
                normed.astype(mx.float16),
                w[prefix + "self_attn.in_proj_weight"].T,
            )
            qkv = qkv.reshape(
                batch, length, 3, self.heads, dim // self.heads
            ).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if layer == self.layers - 1:
                q = q[:, :, :35]
                x = x[:, :35]
            head = dim // self.heads
            if head % 64:
                padding = [(0, 0), (0, 0), (0, 0), (0, 64 - head % 64)]
                q, k, v = [mx.pad(a, padding) for a in (q, k, v)]
            attention = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=head**-0.5, mask=mask, force_fused=True
            )
            attention = attention[..., :head]
            attention = attention.transpose(0, 2, 1, 3).reshape(batch, x.shape[1], dim)
            x = x + self.linear(attention, prefix + "self_attn.out_proj")
            z = self.linear(self.norm(x, prefix + "norm2"), prefix + "linear1")
            z = nn.gelu(z.astype(mx.float32)).astype(mx.float16)
            x = x + self.linear(z, prefix + "linear2")
        x = self.norm(x, "encoder.norm")
        state, tiles = x[:, 34], x[:, :34]
        discard = self.linear(tiles, "discard_head").squeeze(-1)
        claims = self.linear(tiles[:, :33], "claim_head")
        rest = self.linear(state, "state_head")
        logits = mx.concatenate(
            [
                discard,
                rest[:, :63],
                claims[..., 0],
                claims[..., 1],
                claims[..., 2],
                claims[..., 3],
                rest[:, 63:65],
            ],
            axis=-1,
        )
        if belief is None:
            return logits, None
        if self.belief_version == 2:
            return logits, (belief.transpose(0, 2, 1), progress)
        return logits, belief.transpose(0, 2, 1)

    def decision(self, planes, scalars, events, mask):
        logits = self.forward(planes, scalars, events).astype(mx.float32)
        masked = mx.where(mask, logits, -1e9)
        choices = mx.argmax(
            masked - mx.logsumexp(masked, axis=-1, keepdims=True), axis=-1
        )
        return choices, mx.all(mx.any(mask, axis=-1)) & mx.all(mx.isfinite(logits))
