"""The Metal backend preserves the Actor equation and asynchronous batch ownership."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from guided.mlx_actor import Actor as MetalActor
from guided.model_server import Service, blank
from guided.networks import Actor


def test_actor_matches_reference_with_real_and_padded_histories():
    torch.set_num_threads(1)
    torch.manual_seed(593)
    actor = Actor(32, 2, 4).eval()
    metal = MetalActor(actor)
    rng = np.random.default_rng(456)
    row = blank()
    planes = rng.normal(size=(3, *row["planes"].shape)).astype(np.float16)
    scalars = rng.normal(size=(3, *row["scalars"].shape)).astype(np.float16)
    events = np.zeros((3, 32, 5), np.int32)
    for field, size in enumerate([7, 5, 35, 35, 4]):
        events[:, 8:24, field] = rng.integers(1, size, size=(3, 16))
    events[:, :8, 1:] = -999  # padding fields must be ignored
    args = [planes, scalars, events]
    with torch.inference_mode():
        expected = actor(*(torch.from_numpy(a) for a in args)).numpy()
    actual = np.asarray(metal.logits(*(mx.array(a) for a in args)))
    np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.005)
    compact = np.asarray(
        metal.logits(mx.array(planes), mx.array(scalars), mx.array(events[:, 8:24]))
    )
    np.testing.assert_allclose(actual, compact, rtol=0.01, atol=0.005)


def test_service_two_inflight_batches_reuse_slots_without_stale_answers():
    torch.manual_seed(139)
    actor = MetalActor(Actor(32, 1, 4).eval())
    service = Service(SimpleNamespace(device="mlx", get=lambda name: (actor, {})))
    runner = service.runner
    for turn in range(3):
        slots = []
        for legal in [turn + 1, turn + 10]:
            row = blank()
            row["mask"][:] = False
            row["mask"][legal] = True
            slot = runner.slot(1, 1)
            assert slot is not None
            slot.fill([row])
            runner.launch("test", actor, slot)
            slots.append((slot, legal))
        assert runner.slot(1, 1) is None
        for slot, legal in slots:
            while not runner.ready(slot):
                service.wait()
            assert runner.take(slot) == ([legal], True)


def test_metal_rejects_invalid_event_indices():
    from guided.mlx_runner import Slot

    slot = Slot(1, 32)
    row = blank()
    row["events"][0] = [1, 0, 35, 0, 0]
    with pytest.raises(ValueError, match="embedding index"):
        slot.fill([row])
