"""CUDA replay preserves every rollout row and resets gradient accumulation."""

import copy
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph validation")
def test_resident_actor_gradient_matches_full_rollout(monkeypatch):
    from guided.networks import Actor

    # A private package name keeps this independent of other recipe fixtures.
    path = Path(__file__).resolve().parents[2] / "firstprinciples/rl/recipe"
    name = "_actor_gradient_test_recipe"
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    batching = importlib.import_module(name + ".batch")
    loop = importlib.import_module(name + ".loop")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(293)
        rows = []
        # Each common shape gets eight replays; the final three rows exercise
        # the rare-shape path and incomplete final batch. Empty histories stay empty.
        for i, length in enumerate([0] * 64 + [1] * 64 + [17] * 64 + [70] * 3):
            events = np.zeros((length, 5), dtype=np.int16)
            events[:, 0] = 1
            rows.append(
                dict(
                    planes=rng.standard_normal((N_PLANES, 34)).astype(np.float16),
                    scalars=rng.standard_normal(N_SCALARS).astype(np.float16),
                    mask=np.ones(N_ACTIONS, dtype=bool),
                    action=i % N_ACTIONS,
                    events=events,
                    # Privileged data must never enter the Actor computation.
                    oracle_planes=np.full((16, 34), np.nan, dtype=np.float32),
                    oracle_scalars=np.full(4, np.nan, dtype=np.float32),
                )
            )
        weights = rng.normal(size=len(rows)).astype(np.float32)
        weights[-3:] = [3, -7, 11]
        source = batching.ResidentRows(rows, "cuda")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(374)
            actor = Actor(32, 1, 4).cuda()
            frozen = copy.deepcopy(actor.state_dict())
            _, expected = loop.gradient(
                actor, rows, weights, 8, "cuda", batch_source=source
            )
            for _ in range(2):
                _, actual = loop.gradient(
                    actor, rows, weights, 8, "cuda", batch_source=source, capture=True
                )
                assert len(actual) == len(expected)
                for reference, value in zip(expected, actual, strict=True):
                    assert torch.isfinite(value).all()
                    assert torch.equal(reference, value)
                assert all(p.grad is None for p in actor.parameters())
            assert all(
                torch.equal(value, frozen[key])
                for key, value in actor.state_dict().items()
            )
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)
        for key in list(sys.modules):
            if key.startswith(name + "."):
                del sys.modules[key]
