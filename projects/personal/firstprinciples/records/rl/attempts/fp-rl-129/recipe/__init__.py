"""Scale run: the large Actor's whole training under one fixed recipe (see loop.py)."""


def train(ctx):
    # Workers import recipe.features and recipe.returns, which need no torch.
    # Load the learner and torch solely in the training parent.
    from .loop import train as run

    return run(ctx)


__all__ = ["train"]
