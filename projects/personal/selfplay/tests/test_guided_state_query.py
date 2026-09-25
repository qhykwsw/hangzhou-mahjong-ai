"""Pruning unused final queries preserves the value and its full gradient."""
import copy

import pytest

torch = pytest.importorskip('torch')

from guided.networks import HistoryEncoder
from selfplay_core.encoding import N_PLANES, N_SCALARS


@pytest.mark.parametrize('layers', [1, 2])
def test_state_query_matches_full_value_and_parameter_gradients(layers):
    torch.manual_seed(913)
    full = HistoryEncoder(16, layers, 4)
    state = HistoryEncoder(16, layers, 4, state_only=True)
    state.load_state_dict(copy.deepcopy(full.state_dict()), strict=True)
    planes = torch.randn(3, N_PLANES, 34)
    scalars = torch.randn(3, N_SCALARS)
    events = torch.zeros(3, 19, 5, dtype=torch.int16)
    events[0, :7] = torch.tensor([2, 1, 3, 34, 0])
    events[1, :19] = torch.tensor([3, 3, 17, 15, 0])
    events[2, :1] = torch.tensor([1, 4, 34, 34, 0])
    target = torch.randn(3, 16)
    expected = full(planes, scalars, events)[0]
    actual, unused = state(planes, scalars, events)
    assert unused.shape == (3, 0, 16)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    (expected * target).sum().backward()
    (actual * target).sum().backward()
    for (name, p), (other, q) in zip(full.named_parameters(), state.named_parameters()):
        assert name == other
        torch.testing.assert_close(q.grad, p.grad, rtol=2e-4, atol=3e-6, msg=name)
