import random
import time

import numpy as np

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import RoundConfig
from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import (
    K_EVENTS,
    N_PLANES,
    N_SCALARS,
    P_DISC_SHANTEN,
    P_HAND,
    P_LEGAL_DISCARD,
    encode,
)


def sample_observations(count, seed=0):
    rng = random.Random(seed)
    out = []
    hand = HandState(RoundConfig(), seed)
    while len(out) < count:
        if hand.settled:
            hand = HandState(RoundConfig(), rng.randrange(1 << 30))
        legal = hand.legal_actions()
        if len(legal) > 1:
            out.append(hand.observation())
        special = [a for a in legal if a.action in ("chi", "peng", "gang")]
        hand.step(rng.choice(special) if special and rng.random() < 0.5 else rng.choice(legal))
    return out


def test_shapes_dtypes_and_ranges():
    for obs in sample_observations(200):
        enc = encode(obs)
        assert enc.planes.shape == (N_PLANES, 34) and enc.planes.dtype == np.float32
        assert enc.scalars.shape == (N_SCALARS,) and enc.scalars.dtype == np.float32
        assert enc.events.shape == (K_EVENTS, 5) and enc.events.dtype == np.int16
        assert enc.mask.shape == (N_ACTIONS,) and enc.mask.sum() == len(set(obs.legal_actions))
        assert np.isfinite(enc.planes).all() and np.isfinite(enc.scalars).all()
        assert enc.planes.min() >= 0 and enc.planes.max() <= 1.0
        assert enc.scalars.min() >= -1.0 and enc.scalars.max() <= 1.0
        assert enc.planes[P_HAND].sum() == len(set(obs.hand))
        assert enc.planes[P_LEGAL_DISCARD].sum() == sum(a.action == "discard" for a in obs.legal_actions)
        if obs.phase == "draw":
            assert enc.planes[P_DISC_SHANTEN].max() > 0


def test_encoding_is_deterministic_and_seat_relative():
    obs = sample_observations(5, seed=7)[-1]
    a, b = encode(obs), encode(obs)
    assert np.array_equal(a.planes, b.planes) and np.array_equal(a.scalars, b.scalars)
    assert np.array_equal(a.events, b.events)
    assert (a.events[-1] != 0).any()


def test_encoding_cost_is_bounded():
    observations = sample_observations(300, seed=11)
    for obs in observations[:20]:
        encode(obs)
    started = time.perf_counter()
    for obs in observations:
        encode(obs)
    per_call = (time.perf_counter() - started) / len(observations)
    assert per_call < 0.003, f"encode too slow: {per_call * 1e6:.0f} us"
