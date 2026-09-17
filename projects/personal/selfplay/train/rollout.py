"""Shared-memory sample store, per-worker writer and GAE at iteration cutoffs."""

from multiprocessing import shared_memory

import numpy as np

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import K_EVENTS, N_PLANES, N_SCALARS, EVENT_FIELDS
from env import N_ORACLE_PLANES, N_ORACLE_SCALARS

FIELDS = {
    # name: (per-sample shape, dtype)
    "planes": ((N_PLANES, 34), np.uint8),
    "scalars": ((N_SCALARS,), np.float16),
    "events": ((K_EVENTS, EVENT_FIELDS), np.int8),
    "oracle_planes": ((N_ORACLE_PLANES, 34), np.uint8),
    "oracle_scalars": ((N_ORACLE_SCALARS,), np.float16),
    "mask": ((N_ACTIONS,), np.bool_),
    "action": ((), np.int16),
    "logprob": ((), np.float32),
    "value": ((), np.float32),
    "reward": ((), np.float32),
    "done": ((), np.uint8),
    "hand_no": ((), np.int16),
    "aux_hand": ((), np.float32),
    "aux_hand_known": ((), np.uint8),
    "aux_game": ((), np.float32),
    "aux_game_known": ((), np.uint8),
    "adv": ((), np.float32),
    "ret": ((), np.float32),
}
REQUEST_FIELDS = {
    "planes": ((N_PLANES, 34), np.float32),
    "scalars": ((N_SCALARS,), np.float32),
    "events": ((K_EVENTS, EVENT_FIELDS), np.int16),
    "mask": ((N_ACTIONS,), np.bool_),
    "oracle_planes": ((N_ORACLE_PLANES, 34), np.uint8),
    "oracle_scalars": ((N_ORACLE_SCALARS,), np.float16),
    "slot": ((), np.int16),
    "greedy": ((), np.uint8),
    "table": ((), np.int16),
    "out_action": ((), np.int16),
    "out_logprob": ((), np.float32),
    "out_value": ((), np.float32),
}


class SharedBlock:
    """A dict of numpy arrays backed by one SharedMemory segment each."""

    def __init__(self, spec, lead_shape, create, names=None):
        self.spec, self.lead = spec, tuple(lead_shape)
        self.segments, self.arrays, self.names = {}, {}, {}
        for key, (shape, dtype) in spec.items():
            full = self.lead + shape
            nbytes = int(np.prod(full)) * np.dtype(dtype).itemsize
            if create:
                seg = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
            else:
                seg = shared_memory.SharedMemory(name=names[key])
            self.segments[key] = seg
            self.names[key] = seg.name
            arr = np.ndarray(full, dtype=dtype, buffer=seg.buf)
            if create:
                arr[...] = 0
            self.arrays[key] = arr

    def __getitem__(self, key):
        return self.arrays[key]

    def close(self, unlink=False):
        self.arrays.clear()
        for seg in self.segments.values():
            if unlink:
                try:
                    seg.unlink()
                except FileNotFoundError:
                    pass
            try:
                seg.close()
            except BufferError:
                pass  # live views elsewhere; the mapping goes away with them


class SampleWriter:
    """Writes closed transitions of one worker into its slice of the store."""

    def __init__(self, block: SharedBlock, worker: int, capacity: int):
        self.b, self.w, self.cap = block, worker, capacity
        self.count = 0

    def reset(self):
        self.count = 0

    @property
    def full(self):
        return self.count >= self.cap

    def write(self, p, next_value, done):
        i = self.count
        if i >= self.cap:
            raise RuntimeError("sample store overflow")
        b, w = self.b, self.w
        b["planes"][w, i] = p.planes
        b["scalars"][w, i] = p.scalars
        b["events"][w, i] = p.events
        b["oracle_planes"][w, i] = p.oracle_planes
        b["oracle_scalars"][w, i] = p.oracle_scalars
        b["mask"][w, i] = p.mask
        b["action"][w, i] = p.action
        b["logprob"][w, i] = p.logprob
        b["value"][w, i] = p.value
        b["reward"][w, i] = p.reward
        b["done"][w, i] = 1 if done else 0
        b["hand_no"][w, i] = p.hand_no
        b["aux_hand"][w, i] = p.aux_hand
        b["aux_hand_known"][w, i] = 1 if p.aux_hand_known else 0
        b["aux_game"][w, i] = 0.0
        b["aux_game_known"][w, i] = 0
        self.count = i + 1
        return i

    def fill_hand_aux(self, idx, hand_no, value):
        b, w = self.b, self.w
        if b["hand_no"][w, idx] == hand_no and not b["aux_hand_known"][w, idx]:
            b["aux_hand"][w, idx] = value
            b["aux_hand_known"][w, idx] = 1

    def fill_game_aux(self, idx, value):
        b, w = self.b, self.w
        b["aux_game"][w, idx] = value
        b["aux_game_known"][w, idx] = 1

    def compute_gae(self, chains, gamma, lam):
        """chains: iterable of (closed indices in time order, bootstrap value)."""
        b, w = self.b, self.w
        reward, value, done = b["reward"][w], b["value"][w], b["done"][w]
        adv, ret = b["adv"][w], b["ret"][w]
        for indices, boot in chains:
            last_adv = 0.0
            next_value = boot
            for idx in reversed(indices):
                nonterminal = 0.0 if done[idx] else 1.0
                delta = reward[idx] + gamma * next_value * nonterminal - value[idx]
                last_adv = delta + gamma * lam * nonterminal * last_adv
                adv[idx] = last_adv
                ret[idx] = last_adv + value[idx]
                next_value = value[idx]
