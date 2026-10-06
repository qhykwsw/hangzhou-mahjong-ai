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


def test_a_sample_hook_decides_batches_beyond_the_graphs_as_the_eager_parent_does():
    # On CPU every batch is beyond the graphs; on CUDA a graphed batch runs the
    # same hooked_decision on the replayed logits of its own rows.
    torch = pytest.importorskip("torch")
    from guided.inference import ActorGraphs, policy_batch, policy_logits
    from guided.networks import Actor
    from selfplay_core.actions import N_ACTIONS
    from selfplay_core.encoding import N_PLANES, N_SCALARS

    torch.manual_seed(0)
    model = Actor(16, 1, 2).eval()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(130):  # beyond the largest batch bucket
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[rng.choice(N_ACTIONS, 1 + i % 5, replace=False)] = True
        rows.append({"planes": rng.random((N_PLANES, 34)).astype(np.float16),
                     "scalars": rng.random(N_SCALARS).astype(np.float16),
                     "events": rng.integers(1, 5, (1 + i % 7, 5)).astype(np.int16), "mask": mask})

    def boosted(logits, mask):  # like a recipe's: its own draws and a log weight in place of the logprob
        probs = logits.float().masked_fill(~mask, -1e9).log_softmax(-1).exp()
        actions = torch.multinomial(probs, 1).squeeze(1)
        return actions, (1.0 + probs.gather(1, actions[:, None]).squeeze(1)).log()

    def plain(logits, mask):  # no logprob: the parent takes the policy's
        return logits.masked_fill(~mask, -1e9).argmin(-1), None

    batch = policy_batch(rows, "cpu")
    with torch.no_grad():
        logits = policy_logits(model, batch["planes"], batch["scalars"], batch["events"]).float()
        logp = logits.masked_fill(~batch["mask"], -1e9).log_softmax(-1)
        for hook in (boosted, plain):
            graphs = ActorGraphs()
            torch.manual_seed(1)
            got = graphs.decide("model:current", model, rows, sample=True, hook=hook).result()
            torch.manual_seed(1)
            actions, selected = hook(logits, batch["mask"])
            if selected is None:
                selected = logp.gather(1, actions[:, None]).squeeze(1)
            want = torch.stack((actions.float(), selected, logp.argmax(-1).float()), -1).numpy()
            assert np.array_equal(got, want)
            assert graphs.counts["eager"] == 1
        # Without sampling (a greedy continuation) the hook is not asked.
        greedy = ActorGraphs().decide("greedy", model, rows, sample=False, hook=plain).result()
        assert np.array_equal(greedy[:, 0], logp.argmax(-1).float().numpy())
