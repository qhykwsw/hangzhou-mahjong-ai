"""Extend an exported Actor to a wider observation encoding or a belief head without changing a decision.

    python -m guided.expand <Actor artifact dir or guided-v1 checkpoint> <models root> \
        [--encoding guided-current-hand-v3 | guided-current-hand-v4] [--belief-head]

A wider encoding appends planes and scalars and may give the events' last
field values (guided.encoding). The extended Actor's new input columns of
encoder.tile_in and encoder.state_in start at zero (v4's risk coefficient
among them), every row of the last event embedding of a v1 or v2 Actor becomes
its row 0 (the only one those encodings use; a v3 Actor's table is already as
wide), and every other tensor is copied, so on a row of the wider encoding it computes
exactly what the source Actor computes on that row's prefixes; training then
learns to use the new inputs. ``--belief-head`` (alone it keeps the source's
encoding) adds the opponents' tile belief (networks.Actor): its output layer
and the projection that feeds it back start at zero, so the logits stay the
source's bit for bit; its hidden layer is drawn from a fixed seed (a dead
zero layer would never learn). Writes the artifact to <models root>/<weights
sha>/ (the model store's layout) and prints that path.
"""

import argparse
from pathlib import Path
import tempfile

from . import bootstrap  # noqa: F401


def _belief_weights(config):
    """The belief head's tensors for an Actor of ``config``: output layer and belief_in zero."""
    import torch
    from .networks import actor_of, actor_options

    # A fixed seed: the same source always gives the same artifact.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        template = actor_of(config["encoding"], config["dim"], config["layers"], config["heads"],
                            **actor_options(config))
    weights = {
        k: v.detach().clone()
        for k, v in template.state_dict().items()
        if k.startswith(("encoder.belief_head.", "encoder.belief_in."))
    }
    for name in ("encoder.belief_head.3.weight", "encoder.belief_head.3.bias",
                 "encoder.belief_in.weight", "encoder.belief_in.bias"):
        weights[name].zero_()
    return weights


def expand(source, root, encoding=None, *, belief_head=False):
    import torch
    from .deployment import load_actor
    from .encoding import EVENT_TAILS, INPUTS, inputs
    from .trial import export_actor

    actor, meta = load_actor(source, "cpu")
    order = list(INPUTS)
    encoding = encoding or meta["encoding"]
    have, want = INPUTS[meta["encoding"]], inputs(encoding)
    if order.index(encoding) < order.index(meta["encoding"]) or (
        encoding == meta["encoding"] and not belief_head
    ):
        raise ValueError(f"{encoding} does not extend {meta['encoding']}")
    config = dict(meta["config"], encoding=encoding)
    if belief_head:
        if config.get("belief_head"):
            raise ValueError("the Actor already has a belief head")
        config["belief_head"] = True
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
    if belief_head:
        weights |= _belief_weights(config)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expand-", dir=root) as tmp:
        staged = Path(tmp) / "artifact"
        written = export_actor(weights, config, staged, source=meta.get("weights_sha256"))
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
    parser.add_argument("--encoding", help=f"default {ENCODING_V3}, or the source's with --belief-head")
    parser.add_argument("--belief-head", action="store_true", help="add a zero-initialized belief head")
    args = parser.parse_args()
    encoding = args.encoding or (None if args.belief_head else ENCODING_V3)
    print(expand(args.source, args.root, encoding, belief_head=args.belief_head))


if __name__ == "__main__":
    main()
