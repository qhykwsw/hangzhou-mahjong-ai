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


def test_a_capturable_hook_is_one_graph_with_the_forward_and_answers_as_out_of_the_graph():
    # CPU stand-ins for the pinned buffers and the CUDA graph: replaying runs
    # the captured computation again into its output, as a graph does. A
    # capturable hook is captured with the forward (one capture per shape and
    # hook) and sees the shape's padding rows; its answers on the caller's rows
    # are hooked_decision's. An unmarked hook still runs outside the graph.
    torch = pytest.importorskip("torch")
    from guided.inference import ActorGraphs, hooked_decision, policy_batch, policy_logits
    from guided.networks import Actor
    from selfplay_core.actions import N_ACTIONS
    from selfplay_core.encoding import N_PLANES, N_SCALARS

    class Event:
        def record(self):
            pass

        def synchronize(self):
            pass

    class Replay:
        def __init__(self, compute, model, static, out):
            self.args, self.out = (compute, model, static), out

        def replay(self):
            compute, model, static = self.args
            self.out.copy_(compute(model, *static))

    class CpuGraphs(ActorGraphs):
        def _host(self, key, size, width, inputs):
            planes, scalars = inputs
            tensors = (torch.zeros((size, planes, 34), dtype=torch.float16),
                       torch.zeros((size, scalars), dtype=torch.float16),
                       torch.zeros((size, width, 5), dtype=torch.int64),
                       torch.zeros((size, N_ACTIONS), dtype=torch.bool))
            views = [t.numpy() for t in tensors]
            views[3][:, 0] = True
            return tensors, views, torch.zeros((size, 3), dtype=torch.float32), Event()

        def _capture(self, compute, model, static):
            out = compute(model, *static)
            return out, Replay(compute, model, static, out)

    torch.manual_seed(0)
    model = Actor(16, 1, 2).eval()
    rng = np.random.default_rng(1)
    rows = []
    for i in range(7):  # graphed in the bucket of 8: one padding row
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[rng.choice(N_ACTIONS, 2 + i % 4, replace=False)] = True
        rows.append({"planes": rng.random((N_PLANES, 34)).astype(np.float16),
                     "scalars": rng.random(N_SCALARS).astype(np.float16),
                     "events": rng.integers(1, 5, (1 + i % 3, 5)).astype(np.int16), "mask": mask})

    def worst(logits, mask):  # deterministic, row by row, with a log weight
        masked = logits.masked_fill(~mask, 1e9)
        return masked.argmin(-1), masked.min(-1).values.clamp(-30, 30)

    def drawn(logits, mask):  # random, as a recipe's deviations are
        probs = logits.float().masked_fill(~mask, -1e9).softmax(-1)
        u = torch.rand(len(probs), 1)
        actions = (probs.cumsum(-1) < u).sum(-1).clamp(max=probs.shape[1] - 1)
        return actions, None

    batch = policy_batch(rows, "cpu")
    with torch.no_grad():
        logits = policy_logits(model, batch["planes"], batch["scalars"], batch["events"]).float()
        eager = hooked_decision(logits, batch["mask"], worst).numpy()
        # Out of the graph: the hook on the replayed logits of the caller's rows.
        plain = CpuGraphs()
        want = plain.decide("model:current", model, rows, sample=True, hook=worst).result().copy()
        assert plain.counts["in_graph"] == 0 and plain.counts["hooked"] == 1
        # The padded shape's forward rounds the last bits differently from the
        # unpadded eager batch; the decisions are the same.
        assert np.array_equal(want[:, [0, 2]], eager[:, [0, 2]])
        assert np.allclose(want[:, 1], eager[:, 1], rtol=1e-5, atol=1e-6)
        worst.capturable = True
        graphs = CpuGraphs()
        for _ in range(3):
            got = graphs.decide("model:current", model, rows, sample=True, hook=worst).result()
            assert np.array_equal(got, want)  # row by row, the same arithmetic as out of the graph
        assert graphs.counts["captures"] == 1
        assert graphs.counts["in_graph"] == graphs.counts["hooked"] == graphs.counts["graphed"] == 3
        assert graphs.seconds["hook"] == 0.0
        # Another capturable hook is a graph of its own; its draws are legal
        # actions and differ between replays.
        drawn.capturable = True
        seen = []
        for _ in range(20):
            got = graphs.decide("model:current", model, rows, sample=True, hook=drawn).result().copy()
            assert all(rows[i]["mask"][int(a)] for i, a in enumerate(got[:, 0]))
            seen.append(tuple(got[:, 0]))
        assert graphs.counts["captures"] == 2 and len(set(seen)) > 1
