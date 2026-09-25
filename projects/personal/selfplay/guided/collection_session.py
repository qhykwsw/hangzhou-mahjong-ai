"""Scoped inference state for consecutive rollouts with unchanged policies."""


def _stamp(model):
    return id(model), tuple(
        (name, id(value), value._version)
        for name, value in (*model.named_parameters(), *model.named_buffers())
    )


class InferenceSession:
    """Own CUDA captures only until the current on-policy learning step ends.

    Parameters and buffers must stay unchanged throughout the scope. Closing
    releases the frozen Actor and its graphs before Critic/Actor training needs
    device memory. Rule policy instances and games never cross rollout jobs.
    """

    def __init__(self, actor):
        self.actor = actor
        self.actor_stamp = _stamp(actor)
        self.signature = None
        self.state = None
        self.closed = False

    def resolve(self, actor, history, options, factory):
        if self.closed:
            raise RuntimeError("inference session is closed")
        if actor is not self.actor or _stamp(actor) != self.actor_stamp:
            raise RuntimeError("Actor changed within an inference session")
        signature = (
            options,
            tuple((key, _stamp(value)) for key, value in sorted(history.items())),
        )
        if self.signature is not None and signature != self.signature:
            raise RuntimeError(
                "opponents or inference options changed within a session"
            )
        if self.state is None:
            self.state = factory()
            self.signature = signature
        return self.state

    def close(self):
        if self.state is not None and self.state[0] is not None:
            import torch

            torch.cuda.synchronize(next(self.actor.parameters()).device)
        self.state = None
        self.actor = None
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("inference session is closed")
        return self

    def __exit__(self, *_):
        self.close()
