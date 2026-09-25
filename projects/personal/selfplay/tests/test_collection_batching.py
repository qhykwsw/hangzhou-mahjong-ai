import numpy as np
import pytest

from guided.collection import request_groups


def test_history_groups_preserve_every_connection_request_and_full_history():
    pending = []
    for i, length in enumerate((0, 64, 65, 513, 1, 128, 129)):
        row = {"events": np.zeros((length, 5), dtype=np.int16), "teacher": -1}
        pending.append((i % 2, i // 2, {"slot": f"model:{i % 2}", "row": row}))

    groups = request_groups(pending, 64)
    seen = [item for _, group in groups for item in group]
    assert sorted(map(id, seen)) == sorted(map(id, pending))
    assert [
        (slot, [len(item[2]["row"]["events"]) for item in group])
        for slot, group in groups
    ] == [
        ("model:0", [0, 1]),
        ("model:0", [65]),
        ("model:0", [129]),
        ("model:1", [64]),
        ("model:1", [128]),
        ("model:1", [513]),
    ]
    assert [[item[1] for item in group] for _, group in request_groups(pending)] == [
        [0, 1, 2, 3],
        [0, 1, 2],
    ]


def test_fine_graph_shapes_bound_padding_without_truncating_large_inputs():
    pytest.importorskip("torch")
    from guided.inference import ActorGraphs

    rows = [{"events": np.zeros((65, 5), dtype=np.int16)} for _ in range(17)]
    assert ActorGraphs().shape(rows) == (32, 96)
    assert ActorGraphs(fine_batches=True).shape(rows) == (24, 96)
    assert ActorGraphs(fine_batches=True).shape(rows * 8) is None
    rows[0]["events"] = np.zeros((513, 5), dtype=np.int16)
    assert ActorGraphs(fine_batches=True).shape(rows) is None


@pytest.mark.parametrize("band", [16, 33, -32, "64"])
def test_history_groups_cannot_alias_one_graph_buffer_ring(band):
    pytest.importorskip("torch")
    from guided.inference import ActorGraphs

    with pytest.raises(ValueError, match="multiple of 32"):
        ActorGraphs(history_band=band)
