"""Length grouping must preserve decisions, request order and error isolation."""

import base64
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from guided.model_server import Service, blank, decide
from guided.inference import policy_batch
from guided.networks import Actor


def payload(length):
    row = blank()
    row["events"] = np.zeros((length, 5), dtype=np.int16)
    row["events"][:, 0] = 1
    row["mask"][:34] = True
    return row, {
        "lookahead": True,
        "encoded": {k: base64.b64encode(v.tobytes()).decode() for k, v in row.items()},
    }


def test_length_groups_keep_decisions_and_oldest_request_priority():
    torch.set_num_threads(1)
    torch.manual_seed(925)
    actor = Actor(16, 1, 4).eval()
    models = SimpleNamespace(
        device="cpu", get=lambda name: (actor, {"config": {"lookahead": True}})
    )
    service = Service(models)
    service.history_band = 64  # exercise the MPS scheduler on a CPU test runner
    rows, bodies = zip(*(payload(n) for n in [129, 2, 64, 65, 130, 3]))
    for i, body in enumerate(bodies):
        service.enqueue(i, dict(body, model="one"))
    launched = []
    original = service._launch

    def launch(group, items):
        launched.append([context for context, _ in items])
        original(group, items)

    service._launch = launch
    answers = {}
    while service.busy():
        service.launch()
        for items, replies in service.finished():
            answers.update(
                (context, reply["index"]) for (context, _), reply in zip(items, replies)
            )
    assert launched == [[0, 4], [1, 2, 5], [3]]
    batch = policy_batch(rows, "cpu")
    with torch.inference_mode():
        expected, valid = decide(
            actor, *(batch[k] for k in ("planes", "scalars", "events", "mask"))
        )
    assert valid and [answers[i] for i in range(len(rows))] == expected.tolist()


@pytest.mark.parametrize("encoded", [{}, {"events": 12}, {"events": "="}, None])
def test_bad_encoding_is_a_request_error_not_a_scheduler_failure(encoded):
    actor = Actor(16, 1, 4).eval()
    models = SimpleNamespace(
        device="cpu", get=lambda name: (actor, {"config": {"lookahead": True}})
    )
    service = Service(models)
    answers = service.decide("one", [{"lookahead": True, "encoded": encoded}])
    assert "error" in answers[0]
    _, body = payload(3)
    assert "index" in service.decide("one", [body])[0]
