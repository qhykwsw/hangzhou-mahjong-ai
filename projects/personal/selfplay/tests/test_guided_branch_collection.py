"""The collection parent's side of cfg["branches"]: the recipe's hook names each learning row's candidates."""

import multiprocessing as mp
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from guided import artifacts, collection  # noqa: E402
from guided.config import configuration  # noqa: E402
from guided.inference import ActorGraphs, branched_decision, policy_batch, policy_logits  # noqa: E402
from guided.networks import Actor  # noqa: E402
from selfplay_core.actions import N_ACTIONS  # noqa: E402
from selfplay_core.encoding import N_PLANES, N_SCALARS  # noqa: E402


def lowest_others(logits, mask, actions):
    """Candidates: the two lowest legal action indices other than the chosen one (-1 where there are none)."""
    width = mask.shape[1]
    key = torch.where(mask, torch.arange(width), width + torch.arange(width))
    key = key.scatter(1, actions[:, None], 3 * width)
    picked = key.argsort(-1)[:, :2]
    return torch.where(mask.gather(1, picked) & (picked != actions[:, None]), picked, -1)


def worst(logits, mask):
    masked = logits.masked_fill(~mask, 1e9)
    return masked.argmin(-1), None


def random_rows(count, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[rng.choice(N_ACTIONS, 1 + i % 5, replace=False)] = True
        rows.append({"planes": rng.random((N_PLANES, 34)).astype(np.float16),
                     "scalars": rng.random(N_SCALARS).astype(np.float16),
                     "events": rng.integers(1, 5, (1 + i % 7, 5)).astype(np.int16), "mask": mask})
    return rows


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
    """ActorGraphs with CPU stand-ins for the pinned buffers and the CUDA graph (replay recomputes)."""

    def _host(self, key, size, width, inputs):
        planes, scalars = inputs
        tensors = (torch.zeros((size, planes, 34), dtype=torch.float16),
                   torch.zeros((size, scalars), dtype=torch.float16),
                   torch.zeros((size, width, 5), dtype=torch.int64),
                   torch.zeros((size, N_ACTIONS), dtype=torch.bool))
        views = [t.numpy() for t in tensors]
        views[3][:, 0] = True
        return tensors, views, torch.zeros((size, 3), dtype=torch.float32), Event()

    def _answers(self, key, size, width, inputs, columns):
        return torch.zeros((size, columns), dtype=torch.float32)

    def _capture(self, compute, model, static):
        out = compute(model, *static)
        return out, Replay(compute, model, static, out)


def test_a_branched_decision_is_the_plain_one_with_the_candidates_after_it():
    torch.manual_seed(0)
    model = Actor(16, 1, 2).eval()
    with torch.no_grad():
        for rows, graphs in ((random_rows(130, 0), ActorGraphs()), (random_rows(7, 1), CpuGraphs())):
            batch = policy_batch(rows, "cpu")
            logits = policy_logits(model, batch["planes"], batch["scalars"], batch["events"]).float()
            # Without a sample hook the decision is the default multinomial draw, as unbranched.
            torch.manual_seed(5)
            got = graphs.decide("model:current", model, rows, sample=True, branches=lowest_others).result()
            torch.manual_seed(5)
            plain = ActorGraphs._decide(model, batch["planes"], batch["scalars"], batch["events"], batch["mask"], True)
            assert got.shape == (len(rows), 5)
            assert np.array_equal(got[:, [0, 2]], plain.numpy()[:, [0, 2]])
            assert np.allclose(got[:, 1], plain.numpy()[:, 1], rtol=1e-5, atol=1e-6)
            want = lowest_others(logits, batch["mask"], torch.as_tensor(got[:, 0]).long())
            assert np.array_equal(got[:, 3:], want.float().numpy())
            # With a sample hook, the hook's decision; a capturable one still runs out of the graph.
            worst.capturable = True
            got = graphs.decide("model:current", model, rows, sample=True, hook=worst, branches=lowest_others).result()
            eager = branched_decision(logits, batch["mask"], worst, lowest_others).numpy()
            assert np.array_equal(got[:, [0, 2, 3, 4]], eager[:, [0, 2, 3, 4]])
            assert graphs.counts["in_graph"] == 0
            del worst.capturable
            # A greedy decision (a continuation) is not branched.
            greedy = graphs.decide("greedy", model, rows, sample=False, branches=lowest_others).result()
            assert greedy.shape == (len(rows), 3)


def test_the_hook_is_made_k_wide():
    hook = collection.fixed_width(lambda logits, mask, actions: actions[:, None], 3)
    out = hook(torch.zeros(4, 5), torch.ones(4, 5, dtype=torch.bool), torch.tensor([1, 2, 3, 4]))
    assert out.tolist() == [[1, -1, -1], [2, -1, -1], [3, -1, -1], [4, -1, -1]]
    wide = collection.fixed_width(lambda logits, mask, actions: torch.zeros(4, 4, dtype=torch.long), 3)
    with pytest.raises(ValueError, match="at most 3"):
        wide(torch.zeros(4, 5), torch.ones(4, 5, dtype=torch.bool), torch.zeros(4, dtype=torch.long))


@pytest.fixture
def rules(monkeypatch):
    """Heuristic rule seats; the forked workers inherit the patches."""
    from mahjong_ai.strategies.heuristic import HeuristicPolicy

    bundle = {"teacher": "teacher", "entries": [{"id": "teacher"}, {"id": "rule"}]}
    monkeypatch.setattr(artifacts, "read_bundle", lambda _: bundle)
    monkeypatch.setattr(artifacts, "bind_runtime", lambda _: None)
    monkeypatch.setattr(artifacts, "build_policy", lambda b, cid: HeuristicPolicy(style="balanced"))
    fork = mp.get_context("fork")
    monkeypatch.setattr(collection.mp, "get_context", lambda _: fork)


def run(hooks, session=None, **extra):
    cfg = configuration(rounds=2, tables=2, workers=1, samples=1, lookahead=False) | {
        "cuda_graphs": False, "playout_baseline": True, "vine": True, "branches": 2} | extra
    command = dict(kind="collect", iteration=1, deadline=time.time() + 60, labels=False, pool=["rule"], samples=1)
    torch.manual_seed(3)
    rows, records, _ = collection.collect(Actor(16, 1, 1), None, cfg, "fixture", "fixture", command, {}, hooks,
                                          inference_session=session)
    assert rows and records
    return rows


def check(rows):
    branched = 0
    for row in rows:
        legal = [int(i) for i in np.flatnonzero(row["mask"]) if i != row["action"]][:2]
        if not legal:
            assert "branches" not in row
            continue
        assert [b["action"] for b in row["branches"]] == legal
        assert all("playout" in b and b["truncated"] is False for b in row["branches"])
        branched += 1
    assert branched > 5


def test_collection_plays_the_hooks_candidates(rules):
    check(run({"branches": lowest_others}))
    with pytest.raises(ValueError, match="needs hooks"):
        run({})


def test_graph_path_plays_the_hooks_candidates(rules):
    """The CUDA-graph path's bookkeeping (two rounds in flight, answers read later), on the CPU."""
    graphs = CpuGraphs()
    session = SimpleNamespace(resolve=lambda actor, history, options, factory: (graphs, actor))
    check(run({"branches": lowest_others}, session=session))
    assert graphs.counts["hooked"] > 0


def test_the_trial_gives_the_workers_a_continuations_seat_specs(monkeypatch, tmp_path):
    from guided import deployment, inference, resources, trial

    frozen = {"rl.pt": Actor(16, 1, 1), "pool-rl.pt": Actor(16, 1, 1), "start.pt": Actor(16, 1, 1)}
    config = {"dim": 16, "layers": 1, "heads": 1, "lookahead": False}
    monkeypatch.setattr(artifacts, "read_bundle", lambda _: {
        "manifest_path": "bundle.json", "teacher": "champion", "entries": [{"id": "champion"}]})
    monkeypatch.setattr(deployment, "load_actor", lambda path, device, **_: (frozen[path], {"config": config}))
    monkeypatch.setattr(inference, "prepare_frozen", lambda actor, device: actor)
    monkeypatch.setattr(resources, "evaluation_resources", lambda: {"workers": 1, "tables": 1})
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=1))
    plan = {
        "seed": 1, "bundle": "bundle.json", "start_actor": "start.pt", "line": "rl",
        "opponents": {"pool-rule": {"family": "rule"}, "pool-rl": {"family": "model", "path": "pool-rl.pt"}},
        "catalog": {"fp-rule-131": {"family": "rule"}, "rl": {"family": "model", "path": "rl.pt"}},
    }
    ctx = trial.Context(plan, tmp_path, time.time() + 600, False, {})
    calls = []

    def fake_collect(actor, critic, cfg, bundle, panel, command, models, hooks, inference_session=None):
        calls.append((command, models, cfg))
        return [], [{}], {"seconds": 0.0}

    monkeypatch.setattr(collection, "collect", fake_collect)
    continuation = {"learner": "fp-rule-131", "opponents": {"pool-rl": "champion", "pool-rule": "rl"}}
    ctx.collect(ctx.actor, {"branches": 2, "playout_baseline": True, "branch_continuation": continuation},
                samples=10, iteration=1)
    command, models, cfg = calls[-1]
    assert cfg["branch_continuation"] == {"learner": "fp-rule-131",
                                          "opponents": {"model:opponent:pool-rl": "champion",
                                                        "pool-rule": "model:opponent:rl"}}
    assert "fp-rule-131" in command["specs"] and "champion" not in command["specs"]
    assert models["model:opponent:rl"] is frozen["rl.pt"]
    assert continuation["learner"] == "fp-rule-131"  # the recipe's cfg is left as it was
    ctx.collect(ctx.actor, {"branches": 2, "playout_baseline": True,
                            "branch_continuation": {"learner": "greedy", "opponents": None}}, samples=10, iteration=2)
    command, models, cfg = calls[-1]
    assert cfg["branch_continuation"] == {"learner": "greedy", "opponents": None}
    assert "fp-rule-131" not in command["specs"] and "model:opponent:rl" not in models
    with pytest.raises(ValueError, match="unknown branch continuation"):
        ctx.collect(ctx.actor, {"branches": 2, "playout_baseline": True,
                                "branch_continuation": {"learner": "nowhere"}}, samples=10, iteration=3)
