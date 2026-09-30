"""A frozen Actor as the teacher: the collection parent labels the learner's rows with its argmax."""

import multiprocessing as mp
import time
from types import SimpleNamespace

import pytest
import torch

from guided import artifacts, collection
from guided.config import configuration
from guided.inference import Decided, policy_batch, policy_logits
from guided.learning import make_batch
from guided.networks import Actor
from mahjong_ai.strategies.heuristic import HeuristicPolicy

SLOT = "model:teacher:frozen"


@pytest.fixture
def rules(monkeypatch):
    """Heuristic rule seats and rule teacher; the forked workers inherit the patches."""
    bundle = {"teacher": "teacher", "entries": [{"id": "teacher"}, {"id": "rule"}]}
    monkeypatch.setattr(artifacts, "read_bundle", lambda _: bundle)
    monkeypatch.setattr(artifacts, "bind_runtime", lambda _: None)
    monkeypatch.setattr(artifacts, "build_policy", lambda b, cid: HeuristicPolicy(style="balanced"))
    fork = mp.get_context("fork")
    monkeypatch.setattr(collection.mp, "get_context", lambda _: fork)


def run(teacher, history, *, bc_only=False, session=None):
    cfg = configuration(rounds=2, tables=2, workers=1, samples=1, lookahead=False, bc_only=bc_only)
    command = dict(kind="collect", iteration=1, deadline=time.time() + 60, labels=True, teacher=teacher,
                   pool=["rule"], samples=1)
    torch.manual_seed(3)
    learner = Actor(16, 1, 1)
    rows, records, _ = collection.collect(learner, None, cfg | {"cuda_graphs": False}, "fixture", "fixture",
                                          command, history, inference_session=session)
    assert rows and records
    return learner, rows


def argmax(model, rows):
    batch = make_batch(rows, "cpu")
    with torch.no_grad():
        logits = model(batch["planes"], batch["scalars"], batch["events"]).float()
    return logits.masked_fill(~batch["mask"], -1e9).argmax(-1).tolist()


def frozen_teacher():
    torch.manual_seed(17)
    return Actor(16, 1, 1).eval()


def test_every_learner_row_carries_the_teacher_actors_argmax(rules):
    teacher = frozen_teacher()
    learner, rows = run(SLOT, {SLOT: teacher})
    labels = [row["teacher"] for row in rows]
    assert labels == argmax(teacher, rows)
    assert all(row["mask"][row["teacher"]] and row["label_version"] == SLOT for row in rows)
    assert labels != argmax(learner, rows)  # the teacher's choices, not the learner's


def test_pure_imitation_plays_the_teacher_actors_move(rules):
    teacher = frozen_teacher()
    _, rows = run(SLOT, {SLOT: teacher}, bc_only=True)
    assert [row["action"] for row in rows] == [row["teacher"] for row in rows] == argmax(teacher, rows)


class EagerGraphs:
    """ActorGraphs' decision (forward, mask, sample or argmax) run eagerly on the CPU."""

    history_band, batch_buckets, graphs = 0, None, {}

    def __init__(self):
        self.calls = []

    def decide(self, key, model, rows, sample):
        self.calls.append((key, sample))
        batch = policy_batch(rows, "cpu")
        logits = policy_logits(model, batch["planes"], batch["scalars"], batch["events"]).float()
        logp = logits.masked_fill(~batch["mask"], -1e9).log_softmax(-1)
        greedy = logp.argmax(-1)
        actions = torch.multinomial(logp.exp(), 1).squeeze(1) if sample else greedy
        chosen = logp.gather(1, actions[:, None]).squeeze(1)
        return Decided(torch.stack((actions.float(), chosen, greedy.float()), -1))


def test_graph_path_labels_under_the_teachers_own_slot(rules):
    """The CUDA-graph path's bookkeeping (two rounds in flight, answers read later), on the CPU."""
    teacher = frozen_teacher()
    graphs = EagerGraphs()
    session = SimpleNamespace(resolve=lambda actor, history, options, factory: (graphs, actor))
    _, rows = run(SLOT, {SLOT: teacher}, session=session)
    assert [row["teacher"] for row in rows] == argmax(teacher, rows)
    assert set(graphs.calls) == {("model:current", True), (SLOT, False)}
    assert graphs.calls.count((SLOT, False)) == graphs.calls.count(("model:current", True))


def test_graph_path_behaviour_cloning_plays_the_teachers_argmax(rules):
    teacher = frozen_teacher()
    graphs = EagerGraphs()
    session = SimpleNamespace(resolve=lambda actor, history, options, factory: (graphs, actor))
    _, rows = run(SLOT, {SLOT: teacher}, bc_only=True, session=session)
    assert [row["action"] for row in rows] == [row["teacher"] for row in rows] == argmax(teacher, rows)


def test_rule_teachers_still_label_in_the_workers(rules):
    _, rows = run("teacher", {}, bc_only=True)
    assert all(row["teacher"] >= 0 and row["label_version"] == "teacher" for row in rows)
    assert [row["action"] for row in rows] == [row["teacher"] for row in rows]


def test_trial_offers_catalog_actors_as_teachers_and_labels_with_them(monkeypatch, tmp_path):
    from guided import deployment, inference, resources, trial

    frozen = {"rl.pt": frozen_teacher(), "pool-rl.pt": Actor(16, 1, 1), "start.pt": Actor(16, 1, 1),
              "proxy.pt": Actor(16, 1, 1)}
    config = {"dim": 16, "layers": 1, "heads": 1, "lookahead": False}
    monkeypatch.setattr(artifacts, "read_bundle", lambda _: {
        "manifest_path": "bundle.json", "teacher": "champion", "entries": [{"id": "champion"}]})
    monkeypatch.setattr(deployment, "load_actor", lambda path, device: (frozen[path], {"config": config}))
    monkeypatch.setattr(inference, "prepare_frozen", lambda actor, device: actor)
    monkeypatch.setattr(resources, "evaluation_resources", lambda: {"workers": 1, "tables": 1})
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=1))
    plan = {
        "seed": 1, "bundle": "bundle.json", "start_actor": "start.pt", "line": "rl",
        "opponents": {"pool-rule": {"family": "rule"}, "pool-rl": {"family": "model", "path": "pool-rl.pt"}},
        "catalog": {"old-rule": {"family": "rule"}, "rl": {"family": "model", "path": "rl.pt"},
                    "proxy": {"family": "model", "path": "proxy.pt"}},
        "opponents_only": ["proxy"],
        "label_hand_budget": 64,
    }
    ctx = trial.Context(plan, tmp_path, time.time() + 600, False, {})
    # Catalog Actors, not the pool's nor those offered as opponents only.
    assert ctx.teachers == ["champion", "pool-rule", "old-rule", "rl"]

    calls = []

    def fake_collect(actor, critic, cfg, bundle, panel, command, models, hooks, inference_session=None):
        calls.append((command, models, cfg))
        return [], [{}, {}], {"seconds": 0.0}

    monkeypatch.setattr(collection, "collect", fake_collect)
    with pytest.raises(ValueError, match="teacher_plays needs labels"):
        ctx.collect(ctx.actor, {"teacher": "rl", "teacher_plays": True}, samples=10, iteration=1)
    ctx.collect(ctx.actor, {"teacher": "rl", "opponents": ["pool-rule"], "teacher_plays": True},
                samples=10, iteration=1, labels=True)
    command, models, cfg = calls[-1]
    assert cfg["bc_only"] and "teacher_plays" not in cfg
    assert command["teacher"] == "model:teacher:rl" and models["model:teacher:rl"] is frozen["rl.pt"]
    assert "rl" not in command["specs"] and command["pool"] == ["pool-rule"]
    assert ctx.labeled_hands == ctx.hands == 32 and ctx.labeled_hands_left() == 32

    ctx.collect(ctx.actor, {"teacher": "old-rule"}, samples=10, iteration=2, labels=True)
    command, models, cfg = calls[-1]
    assert command["teacher"] == "old-rule" and "old-rule" in command["specs"] and not cfg["bc_only"]
    assert not any(key.startswith("model:teacher:") for key in models)
    assert ctx.labeled_hands_left() == 0
    with pytest.raises(ValueError, match="teacher-labelled hand budget exhausted"):
        ctx.collect(ctx.actor, {"teacher": "rl"}, samples=10, iteration=3, labels=True)
    with pytest.raises(ValueError, match="unknown teacher"):
        ctx.collect(ctx.actor, {"teacher": "pool-rl"}, samples=10, iteration=3, labels=True)
    # Unlabelled collections leave the teacher out of the models.
    ctx.collect(ctx.actor, {"teacher": "rl"}, samples=10, iteration=4)
    assert not any(key.startswith("model:teacher:") for key in calls[-1][1])
