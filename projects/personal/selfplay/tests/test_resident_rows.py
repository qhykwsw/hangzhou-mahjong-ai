"""Learner residency preserves the established ragged-row batch contract."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS

path = Path(__file__).resolve().parents[2] / "firstprinciples/rl/recipe/batch.py"
spec = importlib.util.spec_from_file_location("resident_recipe_batch", path)
batching = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batching)


@pytest.mark.parametrize("lengths", [[0, 1, 4, 17], [0, 0, 0, 0]])
@pytest.mark.parametrize("oracle", [False, True])
def test_resident_gather_matches_row_collator(lengths, oracle):
    rng = np.random.default_rng(731)
    rows = [dict(
        planes=rng.standard_normal((N_PLANES, 34)).astype(np.float32),
        scalars=rng.standard_normal(N_SCALARS).astype(np.float32),
        mask=rng.integers(0, 2, N_ACTIONS).astype(bool),
        action=i,
        events=rng.integers(0, 30, (length, 5), dtype=np.int64),
        oracle_planes=rng.standard_normal((16, 34)).astype(np.float64),
        oracle_scalars=rng.standard_normal(4).astype(np.float64),
    ) for i, length in enumerate(lengths)]
    resident = batching.ResidentRows(rows, "cpu")
    for indices in ([3, 0, 3, 1], [0], [2, 1]):
        expected = batching.make_batch([rows[i] for i in indices], "cpu", oracle=oracle)
        actual = resident.batch(indices, oracle=oracle)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert actual[key].dtype == expected[key].dtype
            assert actual[key].stride() == expected[key].stride()
            assert torch.equal(actual[key], expected[key]), key
        # A learner may reuse its gathered buffers; the rollout stays immutable.
        actual["planes"].zero_()
        assert torch.equal(resident.batch(indices)["planes"], expected["planes"])
