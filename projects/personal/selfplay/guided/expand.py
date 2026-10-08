"""Extend an exported Actor: a wider encoding, a belief head or oracle inputs keep every decision;
shared suits start from a fit.

    python -m guided.expand <Actor artifact dir or guided-v1 checkpoint> <models root> \
        [--encoding guided-current-hand-v3 | guided-current-hand-v4] [--belief-head [v1 | v2]] [--share-suits] \
        [--oracle-inputs]

A wider encoding appends planes and scalars and may give the events' last
field values (guided.encoding). The extended Actor's new input columns of
encoder.tile_in and encoder.state_in start at zero (v4's risk coefficient
among them), every row of the last event embedding of a v1 or v2 Actor becomes
its row 0 (the only one those encodings use; a v3 Actor's table is already as
wide), and every other tensor is copied, so on a row of the wider encoding it computes
exactly what the source Actor computes on that row's prefixes; training then
learns to use the new inputs. ``--belief-head`` and ``--share-suits`` keep
the source's encoding unless ``--encoding`` is given. ``--belief-head`` adds the opponents' tile belief (networks.Actor): its output layer
and the projection that feeds it back start at zero, so the logits stay the
source's bit for bit; its hidden layer is drawn from a fixed seed (a dead
zero layer would never learn). ``--belief-head v2`` adds the wider v2 belief
(networks.Actor) the same way, or widens a v1 head to it: the v1 head's hidden
layers, its three output channels and belief_in's three input columns become
v2's leading ones (computed by their own matmuls, so bit for bit as before),
every new output row, injection column and the state head's output layer and
injection start at zero, the state head's hidden layer from the fixed seed.
``--share-suits`` is the one step that does
change decisions: the Actor's per-tile embeddings become shared-suit tables
(networks.SuitEmbedding), started at the least-squares additive fit of the
source's tables (rank: the mean over the three suits; suit: its mean over the
ranks less the overall mean; honours and "no tile" copied), so only what the
source's tables held beyond rank plus suit is lost. ``--oracle-inputs`` makes a
training start with the training-only oracle inputs (ADR-0018, guided.oracle):
their bias-free projections start at zero, so the logits stay the source's bit
for bit with or without oracle input. Such an artifact is a trial's start and
nothing else: deployment.load_actor refuses it everywhere else (evaluation,
serving, opponents) and every export of the training strips the inputs again.
Writes the artifact to <models root>/<weights sha>/ (the model store's layout)
and prints that path.
"""

import argparse
from pathlib import Path
import tempfile

from . import bootstrap  # noqa: F401


def _belief_weights(config, source):
    """The belief head's tensors for an Actor of ``config``: outputs and injections zero.

    Hidden layers come from a fixed seed, except that a v1 head in the
    ``source`` weights going to v2 keeps its hidden layers, and its output
    channels and belief_in's input columns (and bias) as v2's leading ones.
    """
    import torch
    from .networks import BELIEF_V1_TILES, actor_of, actor_options

    # A fixed seed: the same source always gives the same artifact.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        template = actor_of(config["encoding"], config["dim"], config["layers"], config["heads"],
                            **actor_options(config))
    weights = {k: v.detach().clone() for k, v in template.state_dict().items() if k.startswith("encoder.belief_")}
    for name, value in weights.items():
        if name.startswith(("encoder.belief_head.3.", "encoder.belief_in.",
                            "encoder.belief_state_head.3.", "encoder.belief_state_in.")):
            value.zero_()
    if config["belief_head"] == "v2" and "encoder.belief_head.3.weight" in source:
        k = BELIEF_V1_TILES
        for name in ("0.weight", "0.bias", "1.weight", "1.bias"):
            weights["encoder.belief_head." + name] = source["encoder.belief_head." + name].clone()
        weights["encoder.belief_head.3.weight"][:k] = source["encoder.belief_head.3.weight"]
        weights["encoder.belief_head.3.bias"][:k] = source["encoder.belief_head.3.bias"]
        weights["encoder.belief_in.weight"][:, :k] = source["encoder.belief_in.weight"]
        weights["encoder.belief_in.bias"][:] = source["encoder.belief_in.bias"]
    return weights


def _shared_suit_weights(weights):
    """``weights`` with each per-tile table replaced by its shared-suit fit (networks.SuitEmbedding.fit)."""
    from .networks import TILE_FIELDS, SuitEmbedding

    result = dict(weights)
    for name in ("encoder.tile_embedding", *(f"encoder.event_embeddings.{i}.weight" for i in TILE_FIELDS)):
        prefix = name.removesuffix("weight").rstrip(".") + "."
        result |= {prefix + k: v for k, v in SuitEmbedding.fit(result.pop(name)).items()}
    return result


def expand(source, root, encoding=None, *, belief_head=False, suit_embedding=None, oracle_inputs=False):
    import torch
    from .deployment import load_actor
    from .encoding import EVENT_TAILS, INPUTS, inputs
    from .networks import ORACLE_PLANES, ORACLE_SCALARS, has_oracle
    from .trial import export_actor

    # A start with oracle inputs may be expanded further; it stays a training start.
    actor, meta = load_actor(source, "cpu", oracle=True)
    order = list(INPUTS)
    encoding = encoding or meta["encoding"]
    have, want = INPUTS[meta["encoding"]], inputs(encoding)
    if order.index(encoding) < order.index(meta["encoding"]) or (
        encoding == meta["encoding"] and not belief_head and not suit_embedding and not oracle_inputs
    ):
        raise ValueError(f"{encoding} does not extend {meta['encoding']}")
    config = dict(meta["config"], encoding=encoding)
    if belief_head:
        # True (or "v1") is v1, "v2" is v2; v2 can widen a v1 head.
        version, present = "v2" if belief_head == "v2" else True, config.get("belief_head", False)
        if present == "v2":
            raise ValueError(f"the Actor already has a v2 belief head (asked for {belief_head!r})")
        if present and version is True:
            raise ValueError("the Actor already has a belief head")
        config["belief_head"] = version
    if suit_embedding:
        if suit_embedding != "shared" or config.get("suit_embedding"):
            raise ValueError(f"cannot change suit_embedding {config.get('suit_embedding')!r} to {suit_embedding!r}")
        config["suit_embedding"] = suit_embedding
    if oracle_inputs:
        if has_oracle(config):
            raise ValueError("the Actor already has oracle inputs")
        config["oracle_inputs"] = True
    weights = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    if encoding != meta["encoding"]:
        for name, size in (("encoder.tile_in.weight", want[0]), ("encoder.state_in.weight", want[1])):
            narrow = weights[name]
            wide = torch.zeros((narrow.shape[0], size), dtype=narrow.dtype)
            wide[:, : narrow.shape[1]] = narrow
            weights[name] = wide
        # Earlier encodings only ever look up row 0 (their other rows never trained),
        # so every row of the wider table starts as that row.
        tail = weights["encoder.event_embeddings.4.weight"]
        if len(tail) != EVENT_TAILS[encoding]:
            weights["encoder.event_embeddings.4.weight"] = tail[:1].expand(EVENT_TAILS[encoding], -1).contiguous()
    if suit_embedding:
        weights = _shared_suit_weights(weights)
    if belief_head:
        weights |= _belief_weights(config, weights)
    if oracle_inputs:
        # Bias-free and zero: no oracle, a zero one and any other give the source's logits until training moves them.
        dim = config["dim"]
        weights["encoder.oracle_tile_in.weight"] = torch.zeros((dim, ORACLE_PLANES))
        weights["encoder.oracle_state_in.weight"] = torch.zeros((dim, ORACLE_SCALARS))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expand-", dir=root) as tmp:
        staged = Path(tmp) / "artifact"
        written = export_actor(weights, config, staged, source=meta.get("weights_sha256"),
                               oracle_start=has_oracle(config))
        target = root / written["weights_sha256"]
        if target.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {target}")
        staged.rename(target)
    return target


def main():
    from .encoding import ENCODING_V3

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="an exported Actor directory or a guided-v1 training checkpoint")
    parser.add_argument("root", help="models root; the artifact goes to <root>/<weights sha>/")
    parser.add_argument("--encoding", help=f"default {ENCODING_V3}, or the source's with another option")
    parser.add_argument("--belief-head", nargs="?", const="v1", choices=("v1", "v2"),
                        help="add a zero-initialized belief head (v1, the default), or the v2 belief (widening a v1 head)")
    parser.add_argument("--share-suits", action="store_true",
                        help="share rank embeddings across the suits (changes decisions slightly)")
    parser.add_argument("--oracle-inputs", action="store_true",
                        help="a training start with zero oracle inputs (ADR-0018; refused outside training)")
    args = parser.parse_args()
    encoding = args.encoding or (None if args.belief_head or args.share_suits or args.oracle_inputs else ENCODING_V3)
    belief_head = {None: False, "v1": True, "v2": "v2"}[args.belief_head]
    print(expand(args.source, args.root, encoding, belief_head=belief_head,
                 suit_embedding="shared" if args.share_suits else None, oracle_inputs=args.oracle_inputs))


if __name__ == "__main__":
    main()
