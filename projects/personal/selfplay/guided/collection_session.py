"""Scoped inference state for consecutive rollouts with unchanged policies."""

import weakref

# Frozen inference snapshots (with their CUDA graphs) kept across learning
# steps, keyed weakly by the learning Actor so they go when it goes.
_PERSISTENT = weakref.WeakKeyDictionary()


def _tensors(model):
    return dict((*model.named_parameters(), *model.named_buffers()))


def persistent_inference(actor, history, options, factory):
    """The frozen snapshot of an earlier rollout, refreshed to the current weights.

    Capturing CUDA graphs costs about a second per rollout. Between drained
    rollouts the snapshot keeps its tensor addresses, so the captured graphs
    stay valid: every parameter and buffer (nonpersistent ones included) is
    copied in place, with the same cast prepare_frozen applies. The copy takes
    about a millisecond and runs on every reuse, so no way of updating the
    weights goes unseen. Other opponents, options or architecture rebuild.
    """
    source = _tensors(actor)
    signature = (
        options,
        repr(actor),
        tuple((key, _stamp(value)) for key, value in sorted(history.items())),
        tuple((key, tuple(t.shape), t.dtype, t.device) for key, t in source.items()),
    )
    cached = _PERSISTENT.get(actor)
    if cached is None or cached["signature"] != signature:
        _PERSISTENT.pop(actor, None)
        state = factory()
        # Only frozen snapshots with graphs are worth keeping, and the cache
        # must not hold the Actor itself (it is the weak key).
        if (state[0] is not None and state[1] is not actor
                and all(model is not actor for model in history.values())):
            _PERSISTENT[actor] = dict(state=state, signature=signature)
        return state
    import torch

    state = cached["state"]
    device = next(state[1].parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    target = _tensors(state[1])
    with torch.no_grad():
        for key, value in source.items():
            target[key].copy_(value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return state


def _stamp(model):
    return id(model), tuple(
        (name, id(value), value._version)
        for name, value in (*model.named_parameters(), *model.named_buffers())
    )


class InferenceSession:
    """Pin one inference snapshot for the rollouts of one on-policy learning step.

    Parameters and buffers must stay unchanged throughout the scope. The frozen
    Actor and its graphs outlive it (persistent_inference) and take the updated
    weights in the next scope. Rule policy instances and games never cross
    rollout jobs.
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
            self.state = persistent_inference(actor, history, options, factory)
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
