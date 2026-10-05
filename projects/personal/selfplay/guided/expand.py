"""Extend an exported Actor to a wider observation encoding without changing a decision.

    python -m guided.expand <Actor artifact dir or guided-v1 checkpoint> <models root> \
        [--encoding guided-current-hand-v2]

A wider encoding appends planes (guided.encoding). The extended Actor's new
input columns of encoder.tile_in start at zero and every other tensor is
copied, so on a row of the wider encoding it computes exactly what the source
Actor computes on that row's prefix; training then learns to use the new
planes. Writes the artifact to <models root>/<weights sha>/ (the model store's
layout) and prints that path.
"""

import argparse
from pathlib import Path
import tempfile

from . import bootstrap  # noqa: F401


def expand(source, root, encoding):
    import torch
    from .deployment import load_actor
    from .encoding import INPUTS, inputs
    from .trial import export_actor

    actor, meta = load_actor(source, "cpu")
    have, want = INPUTS[meta["encoding"]], inputs(encoding)
    if want[0] <= have[0] or want[1] != have[1]:
        raise ValueError(f"{encoding} does not extend {meta['encoding']}")
    weights = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    narrow = weights["encoder.tile_in.weight"]
    wide = torch.zeros((narrow.shape[0], want[0]), dtype=narrow.dtype)
    wide[:, : narrow.shape[1]] = narrow
    weights["encoder.tile_in.weight"] = wide
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
    from .encoding import ENCODING_V2

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="an exported Actor directory or a guided-v1 training checkpoint")
    parser.add_argument("root", help="models root; the artifact goes to <root>/<weights sha>/")
    parser.add_argument("--encoding", default=ENCODING_V2)
    args = parser.parse_args()
    print(expand(args.source, args.root, args.encoding))


if __name__ == "__main__":
    main()
