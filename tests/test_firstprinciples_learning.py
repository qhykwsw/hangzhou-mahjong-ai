"""Actor batches must neither depend on nor transfer privileged features."""
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'projects/personal/firstprinciples'))
sys.path.insert(0, str(ROOT / 'projects/personal/selfplay'))
sys.path.insert(0, str(ROOT / 'projects/personal/selfplay/src'))
from rl.recipe.batch import make_batch
from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


def test_actor_batch_accepts_only_legal_inputs_and_preserves_ragged_events():
    rows = [dict(planes=np.ones((N_PLANES, 34), np.float16),
                 scalars=np.zeros(N_SCALARS, np.float16), mask=np.ones(N_ACTIONS, bool),
                 events=np.tile([2, 1, 17, 34, 0], (n, 1)), action=17) for n in (1, 63)]
    actual = make_batch(rows, 'cpu')
    assert 'oracle_planes' not in actual and 'oracle_scalars' not in actual
    assert actual['events'].dtype == torch.int16
    torch.testing.assert_close(actual['events'][0, 1:], torch.zeros(62, 5, dtype=torch.int16))
    for i, row in enumerate(rows):
        np.testing.assert_array_equal(actual['events'][i, :len(row['events'])].numpy(), row['events'])
    for row in rows:
        row.update(oracle_planes=np.ones((16, 34), np.float32), oracle_scalars=np.ones(4, np.float32))
    privileged = make_batch(rows, 'cpu', oracle=True)
    for name, value in actual.items():
        torch.testing.assert_close(privileged[name], value)
    assert privileged['oracle_planes'].shape == (2, 16, 34)
