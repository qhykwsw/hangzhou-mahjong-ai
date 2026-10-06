"""Extend an exported Actor to a wider observation encoding without changing a decision.

    python -m guided.expand <Actor artifact dir or guided-v1 checkpoint> <models root> \
        [--encoding guided-current-hand-v3]

A wider encoding appends planes and scalars and may give the events' last
field values (guided.encoding). The extended Actor's new input columns of
encoder.tile_in and encoder.state_in start at zero, every row of the last
event embedding becomes its row 0 (the only one an earlier encoding uses), and
every other tensor is copied, so on a row of the wider encoding it computes
exactly what the source Actor computes on that row's prefixes; training then
learns to use the new inputs. Writes the artifact to <models root>/<weights
sha>/ (the model store's layout) and prints that path.
"""

import argparse
from pathlib import Path
import tempfile

from . import bootstrap  # noqa: F401


def expand(source, root, encoding):
    import torch
    from .deployment import load_actor
    from .encoding import EVENT_TAILS, INPUTS, inputs
    from .trial import export_actor

    actor, meta = load_actor(source, "cpu")
    order = list(INPUTS)
    have, want = INPUTS[meta["encoding"]], inputs(encoding)
    if order.index(encoding) <= order.index(meta["encoding"]):
        raise ValueError(f"{encoding} does not extend {meta['encoding']}")
    weights = {k: v.detach().clone() for k, v in actor.state_dict().items()}
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
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expand-", dir=root) as tmp:
        staged = Path(tmp) / "artifact"
        written = export_actor(
            weights, dict(meta["config"], encoding=encoding), staged, source=meta.get("weights_sha256")
        )
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
    parser.add_argument("--encoding", default=ENCODING_V3)
    args = parser.parse_args()
    print(expand(args.source, args.root, args.encoding))


if __name__ == "__main__":
    main()
