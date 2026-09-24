"""Two asynchronous MLX batches behind the model service's staging interface."""

import os
import queue
import threading

import mlx.core as mx
import numpy as np

from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


class Slot:
    def __init__(self, size, width):
        self.views = (
            np.zeros((size, N_PLANES, 34), np.float16),
            np.zeros((size, N_SCALARS), np.float16),
            np.zeros((size, width, 5), np.int32),
            np.zeros((size, N_ACTIONS), bool),
        )
        self.views[3][:, 0] = True
        self.rows = 0
        self.event, self.done, self.error = None, False, None
        self.choice, self.valid = None, False

    def fill(self, rows):
        planes, scalars, events, mask = self.views
        for i, row in enumerate(rows):
            planes[i], scalars[i], mask[i] = row["planes"], row["scalars"], row["mask"]
            history = np.asarray(row["events"])
            real = history[history[:, 0] != 0]
            if ((real < 0) | (real >= np.array([7, 5, 35, 35, 4]))).any():
                raise ValueError("invalid event embedding index")
            events[i, : len(history)] = history
            events[i, len(history) :] = 0
        self.rows = len(rows)


class Runner:
    polls = False

    def __init__(self):
        self.slots = {}
        self.wake, self._signal = os.pipe()
        os.set_blocking(self.wake, False)
        self.launched = queue.Queue()
        threading.Thread(target=self._wait, name="mlx-waiter", daemon=True).start()

    @staticmethod
    def new_slot(size, width):
        return Slot(size, width)

    def slot(self, count, length):
        from .model_server import IN_FLIGHT, bucket

        shape = bucket(count, length)
        if shape not in self.slots:
            self.slots[shape] = [Slot(*shape) for _ in range(IN_FLIGHT)]
        return next((s for s in self.slots[shape] if s.event is None), None)

    def launch(self, name, actor, slot):
        # mx.array copies numpy storage: reusing the slot after take cannot
        # change inputs retained by the compiled graph or another batch.
        outputs = actor.decide(*(mx.array(a) for a in slot.views))
        mx.async_eval(outputs)
        slot.event, slot.done, slot.error = outputs, False, None
        self.launched.put(slot)

    def _wait(self):
        while True:
            slot = self.launched.get()
            try:
                choice, valid = slot.event
                mx.eval(choice, valid)
                slot.choice = choice.tolist()
                slot.valid = bool(valid.item())
            except Exception as exc:
                slot.error = exc
            slot.done = True
            os.write(self._signal, b"\0")

    def drain(self):
        try:
            os.read(self.wake, 4096)
        except BlockingIOError:
            pass

    @staticmethod
    def ready(slot):
        return slot.done

    @staticmethod
    def take(slot):
        slot.event, slot.done = None, False
        if slot.error is not None:
            raise slot.error
        return slot.choice[: slot.rows], slot.valid
