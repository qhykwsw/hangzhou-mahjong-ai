"""Fused public lookahead preserves full encoded observations byte for byte."""
from dataclasses import replace
import random

import numpy as np
import pytest

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import RoundConfig
from selfplay_core import encoding


@pytest.mark.parametrize('lookahead', [False, True])
def test_fused_encoding_matches_python_assembly(monkeypatch, lookahead):
    native = encoding._fast
    assert native is not None
    rng = random.Random(250925)
    observations = []
    for seed in range(12):
        state = HandState(RoundConfig(), 1500 + seed)
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) > 1:
                observations.append(state.observation(legal))
            state.step(rng.choice(legal), legal)
    # Negative raw-distance boundaries must retain the reference-aware path.
    observations += [replace(observations[0], hand=hand) for hand in [
        (0,0,0,1,1,1,2,2,2,3,33,33,33,33),
        (0,0,0,1,1,1,2,2,2,33,33,33,33),
    ]]
    for obs in observations:
        monkeypatch.setattr(encoding, '_fast', native)
        actual = encoding.encode(obs, lookahead=lookahead)
        monkeypatch.setattr(encoding, '_fast', None)
        expected = encoding.encode(obs, lookahead=lookahead)
        for field in ['planes','scalars','events','mask','legal']:
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        assert actual.shanten == expected.shanten
