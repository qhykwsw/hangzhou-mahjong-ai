"""Training-only legal Actor and independently parameterized privileged Critic.

The Actor reads legal inputs only, except for one training-only option: the
oracle inputs (``oracle_inputs``, ADR-0018, guided.oracle), the hidden state
fed in during training collection and stripped from every export, so whatever
is evaluated, served or deployed reads legal inputs only. Padding is ignored
even when histories are left-padded or mixed in a batch. Positional indices
count real events, not pads.
"""

import math
from typing import NamedTuple

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


# The event fields that hold a tile (34: none): tile and chi's lowest tile.
TILE_FIELDS = (2, 3)

# The oracle inputs (guided.encoding.oracle): per tile 16 planes, the three
# opponents' concealed counts >= 1..4 then the wall's, and 4 scalars, the
# opponents' shanten progress and the wall's size. Their parameters are the
# encoder's only ones under this prefix (strip_oracle).
ORACLE_PLANES, ORACLE_SCALARS = 16, 4
ORACLE_PREFIX = "encoder.oracle_"

# The v2 belief head's channels (Actor, belief_target_v2). Per tile: each
# opponent's concealed copies / 4 (v1's three), the wall's copies / 4, and
# whether each opponent, when ready, wins on drawing the tile. Per hand: each
# opponent's progress (8 - shanten) / 9 and whether it is ready (shanten <= 0).
# Opponents are in turn order after the actor.
BELIEF_TILES = ("opponent1_copies", "opponent2_copies", "opponent3_copies", "wall_copies",
                "opponent1_waits", "opponent2_waits", "opponent3_waits")
BELIEF_STATE = ("opponent1_progress", "opponent2_progress", "opponent3_progress",
                "opponent1_ready", "opponent2_ready", "opponent3_ready")
# The leading tile channels a v1 head has: v2 computes them by their own matmuls (HistoryEncoder).
BELIEF_V1_TILES = 3


class BeliefV2(NamedTuple):
    """A v2 belief head's logits, or its targets (belief_target_v2): tiles [n, 7, 34], state [n, 6].

    Channels as BELIEF_TILES and BELIEF_STATE; a sigmoid gives the predicted values.
    """

    tiles: torch.Tensor
    state: torch.Tensor


class SuitEmbedding(nn.Module):
    """A tile table whose three suits share their ranks: row 9s + r is ranks[r] + suits[s].

    Rows 27..33 (the seven honours) and the ``extra`` rows after them (an
    event's "no tile", 34) are free (``others``). ``weight`` is the whole
    table; lookups read it as an nn.Embedding reads its own.
    """

    def __init__(self, dim, extra=0, std=1.0):
        super().__init__()
        self.ranks = nn.Parameter(torch.empty(9, dim))
        self.suits = nn.Parameter(torch.empty(3, dim))
        self.others = nn.Parameter(torch.empty(7 + extra, dim))
        # A suited row adds two draws: the same spread as a free row.
        nn.init.normal_(self.ranks, std=std / math.sqrt(2))
        nn.init.normal_(self.suits, std=std / math.sqrt(2))
        nn.init.normal_(self.others, std=std)

    @property
    def weight(self):
        suited = (self.suits[:, None] + self.ranks).flatten(0, 1)
        return torch.cat([suited, self.others])

    def forward(self, index):
        return F.embedding(index, self.weight)

    @staticmethod
    def fit(table):
        """Parameters whose table is the least-squares additive fit of a per-tile ``table``.

        ranks[r] is rank r's mean over the three suits, suits[s] the suit's
        mean over its ranks less the mean of all 27; honours and extra rows
        are copied.
        """
        suited = table[:27].unflatten(0, (3, 9))
        ranks = suited.mean(0)
        return {"ranks": ranks, "suits": suited.mean(1) - ranks.mean(0), "others": table[27:].clone()}


class HistoryEncoder(nn.Module):
    """Tile, state and event tokens through a pre-norm transformer.

    ``planes`` and ``scalars`` are the input sizes of the encoding it is built
    for (guided.encoding.INPUTS). Later encodings append planes and scalars,
    so it reads the first ``planes`` and ``scalars`` of whatever it is given.
    ``event_tail`` rows embed the events' last field (guided.encoding.EVENT_TAILS):
    before v3 that field is always 0 and an encoder built for 4 rows reads it as
    0 whatever it is given.

    With ``belief_head`` (an Actor option, see Actor) ``belief_head`` reads the
    34 tile tokens after ``belief_layer = layers // 2`` blocks and ``belief_in``
    adds its sigmoid, without gradient, back to them before the next block.
    ``belief_head="v2"`` widens both to BELIEF_TILES channels and adds
    ``belief_state_head``, reading the state token there, whose sigmoid
    ``belief_state_in`` adds back to it (``belief_version`` is 1 or 2, None
    without a head).
    With ``suit_embedding="shared"`` the tile tokens' table and the events'
    tile tables are SuitEmbeddings instead of free rows per tile.
    With ``oracle_inputs`` (an Actor option, see Actor) the bias-free
    ``oracle_tile_in`` and ``oracle_state_in``, zero at the start, add the
    oracle planes to the tile tokens and the oracle scalars to the state token.
    """

    def __init__(self, dim, layers, heads, max_events=4096, *, state_only=False,
                 planes=N_PLANES, scalars=N_SCALARS, event_tail=4, belief_head=False, suit_embedding=None,
                 oracle_inputs=False):
        super().__init__()
        if dim % heads or dim < 2:
            raise ValueError("dim must be divisible by heads and at least 2")
        if suit_embedding not in (None, "shared"):
            raise ValueError(f"unknown suit_embedding: {suit_embedding!r}")
        self.dim, self.layers, self.heads = dim, layers, heads
        self.max_events = max_events
        self.state_only = state_only
        self.suit_embedding = suit_embedding
        shared = suit_embedding == "shared"
        self.tile_in = nn.Linear(planes, dim)
        self.tile_embedding = SuitEmbedding(dim, std=0.02) if shared else nn.Parameter(torch.empty(34, dim))
        self.state_in = nn.Linear(scalars, dim)
        self.event_embeddings = nn.ModuleList(
            SuitEmbedding(dim, extra=1) if shared and field in TILE_FIELDS else nn.Embedding(size, dim)
            for field, size in enumerate((7, 5, 35, 35, event_tail))
        )
        self.timed_events = event_tail > 4
        self.type_embedding = nn.Parameter(torch.empty(3, dim))
        if not shared:
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
        self.belief_layer = self.belief_version = None
        if type(belief_head) is bool or belief_head is None:
            version = 1 if belief_head else None
        elif belief_head == "v2":
            version = 2
        else:
            raise ValueError(f'belief_head must be true, false or "v2", not {belief_head!r}')
        if version:
            if layers < 2:
                raise ValueError("a belief head needs at least two layers")
            self.belief_layer, self.belief_version = layers // 2, version
            tiles = len(BELIEF_TILES) if self.belief_version == 2 else BELIEF_V1_TILES
            self.belief_head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, tiles)
            )
            # Starts at zero: the blocks after it first see exactly what they saw without it.
            self.belief_in = nn.Linear(tiles, dim)
            nn.init.zeros_(self.belief_in.weight)
            nn.init.zeros_(self.belief_in.bias)
            if self.belief_version == 2:
                self.belief_state_head = nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, len(BELIEF_STATE))
                )
                self.belief_state_in = nn.Linear(len(BELIEF_STATE), dim)
                nn.init.zeros_(self.belief_state_in.weight)
                nn.init.zeros_(self.belief_state_in.bias)
        self.oracle_inputs = bool(oracle_inputs)
        if oracle_inputs:
            # Training only (ADR-0018). Bias-free, so an all-zero oracle input
            # adds exact zeros: no input, a zero input and the stripped Actor
            # compute the same logits. Zero at the start, so the Actor first
            # decides as without them; skip_init draws nothing from the random
            # stream, so every other parameter of a new Actor is the one it
            # gets without the option.
            self.oracle_tile_in = nn.utils.skip_init(nn.Linear, ORACLE_PLANES, dim, bias=False)
            self.oracle_state_in = nn.utils.skip_init(nn.Linear, ORACLE_SCALARS, dim, bias=False)
            nn.init.zeros_(self.oracle_tile_in.weight)
            nn.init.zeros_(self.oracle_state_in.weight)

    def _transform_with_belief(self, x, padding):
        """The transformer's blocks with the belief head after block belief_layer.

        Runs TransformerEncoder.forward's path for these blocks (no nested
        tensors, an additive padding mask), so with belief_in at zero every
        output is the plain transformer's, bit for bit.
        """
        padding = torch.zeros_like(padding, dtype=x.dtype).masked_fill_(padding, float("-inf"))
        for index, block in enumerate(self.transformer.layers):
            if index == self.belief_layer:
                if self.belief_version == 2:
                    x, belief = self._belief_v2(x)
                else:
                    tiles = x[:, :34]
                    belief = self.belief_head(tiles)
                    # Stop-gradient: the policy reads the belief but never trains the head.
                    injected = self.belief_in(torch.sigmoid(belief).detach())
                    x = torch.cat([tiles + injected, x[:, 34:]], dim=1)
                    belief = belief.transpose(1, 2)
            x = block(x, src_key_padding_mask=padding)
        return x, belief

    def _belief_v2(self, x):
        """The tokens at block belief_layer with v2's beliefs added, and those beliefs' logits (BeliefV2).

        The tile head's output layer and ``belief_in`` are one Linear(dim, 7)
        and one Linear(7, dim), but each computes v1's leading channels (the
        opponents' copies) by its own matmul on the very operands a v1 head
        uses, and the other channels by a second one. So a v1 head widened to
        v2 with zeros in the new rows and columns (guided.expand) reproduces
        v1's logits bit for bit; a single matmul over all seven would not, as
        the kernel's summation order may depend on the output width.
        """
        head, inject, k = self.belief_head, self.belief_in, BELIEF_V1_TILES
        tiles, state = x[:, :34], x[:, 34]
        hidden = head[2](head[1](head[0](tiles)))
        out = head[3]
        hands = F.linear(hidden, out.weight[:k], out.bias[:k])
        rest = F.linear(hidden, out.weight[k:], out.bias[k:])
        progress = self.belief_state_head(state)
        # Stop-gradient, as v1: the policy reads the beliefs but never trains the heads.
        injected = (F.linear(torch.sigmoid(hands).detach(), inject.weight[:, :k].contiguous(), inject.bias)
                    + F.linear(torch.sigmoid(rest).detach(), inject.weight[:, k:]))
        state = state + self.belief_state_in(torch.sigmoid(progress).detach())
        x = torch.cat([tiles + injected, state[:, None], x[:, 35:]], dim=1)
        return x, BeliefV2(torch.cat([hands, rest], -1).transpose(1, 2), progress)

    def forward(self, planes, scalars, events, return_belief=False, oracle=None):
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
        table = self.tile_embedding.weight if self.suit_embedding else self.tile_embedding
        tiles = self.tile_in(planes.float().transpose(1, 2)) + table
        state = self.state_in(scalars.float())[:, None]
        if oracle is not None:
            if not self.oracle_inputs:
                raise ValueError("this Actor has no oracle inputs (ADR-0018: only a training Actor may have them)")
            oracle_planes, oracle_scalars = oracle
            tiles = tiles + self.oracle_tile_in(oracle_planes.float().transpose(1, 2))
            state = state + self.oracle_state_in(oracle_scalars.float())[:, None]
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
        if self.belief_layer is None:
            if return_belief:
                raise ValueError("this encoder has no belief head")
            encoded = self.norm(self.transformer(tokens, src_key_padding_mask=padding))
        else:
            encoded, belief = self._transform_with_belief(tokens, padding)
            encoded = self.norm(encoded)
        if self.state_only:
            result = encoded[:, 0], encoded[:, :0]
        else:
            result = encoded[:, 34], encoded[:, :34]
        return (*result, belief) if return_belief else result


def actor_options(config):
    """The optional Actor structures ``config`` turns on, as Actor keyword arguments.

    ``"belief_head": true`` adds the opponents' tile belief, ``"belief_head":
    "v2"`` the wider v2 belief, ``"suit_embedding": "shared"`` shares the
    suits' rank embeddings, ``"oracle_inputs": true`` adds the training-only
    oracle inputs (Actor). Options at their defaults are dropped, so an Actor
    without any keeps its original config, weights and exported files bit for bit.
    """
    options = {}
    belief = config.get("belief_head", False)
    if type(belief) is not bool and belief != "v2":
        raise ValueError('belief_head must be true, false or "v2"')
    if belief:
        options["belief_head"] = belief
    suits = config.get("suit_embedding")
    if suits not in (None, "shared"):
        raise ValueError(f"unknown suit_embedding: {suits!r} (only \"shared\")")
    if suits:
        options["suit_embedding"] = suits
    oracle = config.get("oracle_inputs", False)
    if type(oracle) is not bool:
        raise ValueError("oracle_inputs must be true or false")
    if oracle:
        options["oracle_inputs"] = True
    return options


def has_oracle(config):
    """Whether an Actor config (an artifact's meta.json "config") has the oracle inputs."""
    return config.get("oracle_inputs", False) is not False


def strip_oracle(weights, config):
    """(weights, config) of the same Actor without its oracle inputs (ADR-0018).

    Drops every parameter under ORACLE_PREFIX and the ``oracle_inputs``
    option, whatever the config says. What is left computes exactly what the
    Actor with them computes when it is given no oracle (or an all-zero one):
    the oracle path only ever adds its projections, which are bias-free.
    """
    weights = {k: v for k, v in weights.items() if not k.startswith(ORACLE_PREFIX)}
    return weights, {k: v for k, v in config.items() if k != "oracle_inputs"}


def with_oracle_inputs(actor):
    """A copy of ``actor`` with the oracle inputs added at zero: every logit as ``actor``'s, bit for bit.

    The training-time step expand --oracle-inputs takes on an artifact, here
    on a live Actor (e.g. a recipe's ctx.actor), on the same device.
    """
    if actor.encoder.oracle_inputs:
        raise ValueError("the Actor already has oracle inputs")
    encoder = actor.encoder
    # Every tensor but the zero oracle projections is then copied: the caller's random stream stays untouched.
    with torch.random.fork_rng(devices=[]):
        oracle = Actor(actor.dim, actor.layers, actor.heads, planes=encoder.tile_in.in_features,
                       scalars=encoder.state_in.in_features, event_tail=encoder.event_embeddings[4].num_embeddings,
                       belief_head={1: True, 2: "v2"}.get(encoder.belief_version, False),
                       suit_embedding=encoder.suit_embedding,
                       oracle_inputs=True)
    missing, unexpected = oracle.load_state_dict(actor.state_dict(), strict=False)
    if unexpected or any(not k.startswith(ORACLE_PREFIX) for k in missing):
        raise ValueError(f"cannot add oracle inputs: missing {missing}, unexpected {unexpected}")
    parameter = next(actor.parameters())
    return oracle.to(device=parameter.device).train(actor.training)


def actor_of(encoding, dim, layers, heads, **options):
    """An Actor built for ``encoding``'s inputs (guided.encoding.INPUTS and EVENT_TAILS)."""
    from .encoding import EVENT_TAILS, inputs

    planes, scalars = inputs(encoding)
    return Actor(dim, layers, heads, planes=planes, scalars=scalars, event_tail=EVENT_TAILS[encoding], **options)


def belief_target(oracle_planes):
    """The belief head's training target from rows' oracle_planes: [batch, 3, 34] copies / 4.

    Planes 4i..4i+3 hold the concealed counts >= 1..4 of the opponent i + 1
    seats after the actor (guided.encoding.oracle, a recipe's critic_features),
    so their sum is that count. A loss: binary_cross_entropy_with_logits(belief, target).
    """
    return oracle_planes[:, :12].float().unflatten(1, (3, 4)).sum(2) / 4


def belief_target_v2(oracle_planes, oracle_scalars, opp_waits):
    """A v2 belief head's training targets from rows' fields, as BeliefV2(tiles [n, 7, 34], state [n, 6]) in [0, 1].

    ``oracle_planes`` [n, 16, 34] and ``oracle_scalars`` [n, 4] as
    guided.encoding.oracle defines them (a recipe's critic_features keeps that
    definition), ``opp_waits`` [n, 3, 34] as guided.encoding.opponent_waits;
    all are hidden state, a training label only (ADR-0009). Tiles: planes
    4i..4i+3 summed / 4 for the three opponents and the wall (planes 12..15),
    then opp_waits. State: oracle_scalars[:, :3], each opponent's (8 - shanten)
    / 9, then whether it is ready (shanten <= 0, a scalar of 8/9 or more). A
    loss: sum(binary_cross_entropy_with_logits(p.float(), t) for p, t in zip(belief, target)).
    """
    counts = oracle_planes[:, :16].float().unflatten(1, (4, 4)).sum(2) / 4
    progress = oracle_scalars[:, :3].float()
    ready = (progress * 9 > 7.5).float()
    return BeliefV2(torch.cat([counts, opp_waits.float()], 1), torch.cat([progress, ready], 1))


def belief_parameters(actor):
    """The parameters of an Actor's belief heads (v2: the tile head and the state head), not their injections."""
    encoder = actor.encoder
    heads = [encoder.belief_head] if encoder.belief_version else []
    if encoder.belief_version == 2:
        heads.append(encoder.belief_state_head)
    return [p for head in heads for p in head.parameters()]


class Actor(nn.Module):
    """Logits over the 231 actions from legal inputs only.

    Optional structure (actor_options; the artifact's meta.json config):

    ``belief_head``: the encoder predicts, from the tile tokens after
    layers // 2 blocks, which tiles each opponent holds concealed: logits
    [batch, 3, 34], opponents in turn order after the actor, whose sigmoid is
    the expected copies / 4 (belief_target). The sigmoid, without gradient, is
    projected by ``encoder.belief_in`` and added to the tile tokens before the
    later blocks; ``forward(..., return_belief=True)`` returns (logits, belief
    logits) for a training loss on hidden state (ADR-0009), and the policy loss
    never reaches the head. The head reads only legal inputs, so inference is
    unchanged: one forward pass, logits only.

    ``belief_head="v2"``: the same place and the same stop-gradient injection,
    predicting more. The tile head has the 7 channels of BELIEF_TILES (v1's
    three opponents' copies / 4, the wall's copies / 4, each opponent's waits
    when ready) and ``encoder.belief_in`` is Linear(7, dim); a second head,
    ``encoder.belief_state_head``, reads the state token and predicts the 6
    channels of BELIEF_STATE (each opponent's progress (8 - shanten) / 9 and
    whether it is ready), whose sigmoid ``encoder.belief_state_in`` (Linear(6,
    dim), zero at the start) adds back to the state token. ``forward(...,
    return_belief=True)`` returns (logits, BeliefV2(tiles [batch, 7, 34], state
    [batch, 6])); belief_target_v2 builds the targets from rows' oracle_planes,
    oracle_scalars and opp_waits (guided.encoding.opponent_waits), and
    belief_parameters lists both heads' parameters. expand --belief-head v2
    widens a v1 head with every logit unchanged.

    ``suit_embedding="shared"``: every per-tile embedding, the tile tokens'
    ``encoder.tile_embedding`` and the events' tile and chi tables
    (``encoder.event_embeddings.2`` and ``.3``), is a SuitEmbedding: rank plus
    suit for the three suits, free rows for the honours (and "no tile"). The
    rest of the Actor is already tile-shared (tile_in, discard_head and
    claim_head act on every tile token alike; the planes carry a suit one-hot
    and the rank), so these tables were its only per-tile parameters on the
    input side; the 63 chi logits of state_head stay one row per suit.
    expand --share-suits starts one from an existing Actor (SuitEmbedding.fit).

    ``oracle_inputs`` (training only, ADR-0018; guided.oracle): ``forward(...,
    oracle=(planes [batch, 16, 34], scalars [batch, 4]))`` adds the hidden
    state (guided.encoding.oracle: the opponents' concealed tiles, the wall)
    at the input, ``encoder.oracle_tile_in`` (bias-free 16 -> dim) to each
    tile token and ``encoder.oracle_state_in`` (bias-free 4 -> dim) to the
    state token, both zero at the start. ``oracle=None`` skips them: the Actor
    then computes exactly what it computes stripped of them (strip_oracle),
    and so it does on an all-zero oracle. A recipe anneals the share of oracle
    values kept from 1 to 0 so that the policy stops depending on them; every
    export strips them (guided.trial.export_actor) and every inference loader
    refuses an Actor that has them (guided.deployment.load_actor).
    """

    def __init__(self, dim=128, layers=4, heads=4, *, planes=N_PLANES, scalars=N_SCALARS, event_tail=4,
                 belief_head=False, suit_embedding=None, oracle_inputs=False):
        super().__init__()
        self.dim, self.layers, self.heads = dim, layers, heads
        self.encoder = HistoryEncoder(dim, layers, heads, planes=planes, scalars=scalars, event_tail=event_tail,
                                      belief_head=belief_head, suit_embedding=suit_embedding,
                                      oracle_inputs=oracle_inputs)
        self.discard_head = nn.Linear(dim, 1)
        self.claim_head = nn.Linear(dim, 4)
        self.state_head = nn.Linear(dim, 65)

    def forward(self, planes, scalars, events, return_belief=False, oracle=None):
        if return_belief:
            state, tiles, belief = self.encoder(planes, scalars, events, return_belief=True, oracle=oracle)
        else:
            state, tiles = self.encoder(planes, scalars, events, oracle=oracle)
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
        return (logits, belief) if return_belief else logits


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
