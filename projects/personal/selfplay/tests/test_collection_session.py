"""A rollout cache must never silently serve weights from an earlier update."""

import copy
import pytest
import torch
from guided.collection_session import InferenceSession


@pytest.mark.parametrize(
    "changed", ["actor", "opponent", "buffer", "replacement", "options"]
)
def test_session_rejects_changed_policy_state(changed):
    actor = torch.nn.Linear(3, 2)
    opponent = torch.nn.Linear(3, 2)
    opponent.register_buffer("context", torch.zeros(1))
    models = {"opponent": opponent}
    with InferenceSession(actor) as session:
        state = session.resolve(
            actor, models, ("collect",), lambda: (None, copy.deepcopy(actor))
        )
        assert (
            session.resolve(actor, models, ("collect",), lambda: pytest.fail("rebuilt"))
            is state
        )
        options = ("collect",)
        with torch.no_grad():
            if changed == "actor":
                actor.weight.add_(1)
            elif changed == "opponent":
                opponent.weight.add_(1)
            elif changed == "buffer":
                opponent.context.add_(1)
            elif changed == "replacement":
                models["opponent"] = copy.deepcopy(opponent)
            else:
                options = ("evaluate",)
        with pytest.raises(RuntimeError, match="changed"):
            session.resolve(
                actor, models, options, lambda: pytest.fail("served stale weights")
            )
    assert session.state is None and session.actor is None
    with pytest.raises(RuntimeError, match="closed"):
        session.resolve(actor, models, options, lambda: None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph validation")
def test_session_replays_current_rows_and_next_session_uses_updated_weights():
    import numpy as np
    from guided.inference import ActorGraphs, prepare_frozen
    from guided.networks import Actor
    from selfplay_core.actions import N_ACTIONS
    from selfplay_core.encoding import N_PLANES, N_SCALARS

    torch.set_num_threads(1)
    torch.manual_seed(791)
    actor = Actor(32, 1, 4).cuda().eval()
    rng = np.random.default_rng(811)
    rows = []
    for length in (0, 1, 13, 70):
        events = np.zeros((length, 5), dtype=np.int16)
        events[:, 0] = 1
        rows.append(
            dict(
                planes=rng.normal(size=(N_PLANES, 34)).astype(np.float16),
                scalars=rng.normal(size=N_SCALARS).astype(np.float16),
                events=events,
                mask=np.ones(N_ACTIONS, dtype=bool),
            )
        )

    def factory():
        return ActorGraphs(), prepare_frozen(
            copy.deepcopy(actor).requires_grad_(False), "cuda"
        )

    for _ in range(2):
        with InferenceSession(actor) as session, torch.no_grad():
            graphs, frozen = session.resolve(actor, {}, (), factory)
            for inputs in (rows[:2], rows[2:], rows[:2]):
                actual = (
                    graphs.decide("current", frozen, inputs, sample=False)
                    .result()
                    .copy()
                )
                fresh_graphs, fresh_actor = factory()
                expected = (
                    fresh_graphs.decide("current", fresh_actor, inputs, sample=False)
                    .result()
                    .copy()
                )
                assert np.array_equal(actual, expected)
                assert session.resolve(actor, {}, (), factory) == (graphs, frozen)
            assert all(p.dtype == torch.float32 for p in actor.parameters())
        with torch.no_grad():
            next(actor.parameters()).add_(0.01)
