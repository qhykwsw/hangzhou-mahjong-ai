"""Oracle inputs (ADR-0018, guided.oracle): off by default, zero at the start, stripped on export,
refused by every inference loader, fed to the learner's own decisions only."""

import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided import artifacts, collection, oracle as O  # noqa: E402
from guided.config import configuration  # noqa: E402
from guided.encoding import ENCODING, ENCODING_V4, augment, encode  # noqa: E402
from guided.environment import GREEDY  # noqa: E402
from guided.inference import ActorGraphs, policy_batch  # noqa: E402
from guided.learning import make_batch  # noqa: E402
from guided.networks import Actor, actor_of, actor_options, strip_oracle, with_oracle_inputs  # noqa: E402

CONFIG = {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def rows():
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig

    from guided.encoding import oracle

    rng, kept, game = random.Random(20261008), [], 0
    while len(kept) < 40:
        state = HandState(RoundConfig(rounds=16, allow_tiebreak=False), 20261008 + game)
        while not state.settled and len(kept) < 40:
            legal = state.legal_actions()
            if len(legal) > 1:
                row = encode(state.observation())
                row.update(O.masked(oracle(state, state.turn), 0.5, len(kept)))
                kept.append(row)
            state.step(rng.choice(legal))
        game += 1
    return kept


def inputs(rows, **extra):
    batch = policy_batch(rows, "cpu", **extra)
    return batch, (batch["planes"], batch["scalars"], batch["events"])


def logits_of(actor, rows, oracle=None):
    _, args = inputs(rows)
    with torch.no_grad():
        return actor(*args) if oracle is None else actor(*args, oracle=oracle)


def oracle_of(rows):
    return tuple(torch.as_tensor(x) for x in O.stack(rows))


def zeros(n):
    return torch.zeros((n, 16, 34), dtype=torch.float16), torch.zeros((n, 4), dtype=torch.float16)


def trained(actor):
    """``actor`` with the oracle inputs, their projections moved off zero as training would."""
    actor = with_oracle_inputs(actor)
    with torch.no_grad():
        torch.nn.init.normal_(actor.encoder.oracle_tile_in.weight, std=0.5)
        torch.nn.init.normal_(actor.encoder.oracle_state_in.weight, std=0.5)
    return actor


# -- the Actor -------------------------------------------------------------------------------------------


def test_off_by_default_and_the_option_draws_nothing_from_the_random_stream():
    assert actor_options(CONFIG) == actor_options(CONFIG | {"oracle_inputs": False}) == {}
    assert actor_options(CONFIG | {"oracle_inputs": True}) == {"oracle_inputs": True}
    with pytest.raises(ValueError):
        actor_options({"oracle_inputs": "yes"})
    torch.manual_seed(5)
    plain = Actor(32, 2, 4)
    torch.manual_seed(5)
    built = actor_of(ENCODING, 32, 2, 4, **actor_options(CONFIG | {"oracle_inputs": False}))
    assert not plain.encoder.oracle_inputs
    assert list(plain.state_dict()) == list(built.state_dict())
    assert not any("oracle" in name for name in plain.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), built.state_dict().values()))
    torch.manual_seed(5)
    wide = Actor(32, 2, 4, oracle_inputs=True)
    extra = {k: v for k, v in wide.state_dict().items() if k not in plain.state_dict()}
    assert {k: tuple(v.shape) for k, v in extra.items()} == {
        "encoder.oracle_tile_in.weight": (32, 16), "encoder.oracle_state_in.weight": (32, 4)}
    assert not any(v.any() for v in extra.values())
    assert all(torch.equal(v, wide.state_dict()[k]) for k, v in plain.state_dict().items())


def test_no_oracle_a_zero_oracle_and_the_stripped_actor_agree_bit_for_bit(rows):
    torch.manual_seed(11)
    base = Actor(32, 2, 4, belief_head=True)
    reference = logits_of(base, rows)
    expanded = with_oracle_inputs(base)
    # Zero at the start: any oracle input leaves every logit as it was.
    for oracle in (None, zeros(len(rows)), oracle_of(rows)):
        assert torch.equal(logits_of(expanded, rows, oracle), reference)
    learned = trained(base)
    # Trained: no input and a zero input are still the stripped Actor (keep 0), a real input is not.
    weights, config = strip_oracle(learned.state_dict(), CONFIG | {"oracle_inputs": True, "belief_head": True})
    assert config == CONFIG | {"belief_head": True}
    stripped = actor_of(ENCODING, 32, 2, 4, **actor_options(config))
    stripped.load_state_dict(weights, strict=True)
    assert torch.equal(logits_of(learned, rows), reference)
    assert torch.equal(logits_of(learned, rows, zeros(len(rows))), reference)
    assert torch.equal(logits_of(stripped, rows), reference)
    assert not torch.equal(logits_of(learned, rows, oracle_of(rows)), reference)
    # The return_belief path takes it too.
    _, args = inputs(rows)
    logits, belief = learned(*args, return_belief=True, oracle=oracle_of(rows))
    assert torch.equal(logits, logits_of(learned, rows, oracle_of(rows))) and belief.shape == (len(rows), 3, 34)


def test_the_policy_gradient_trains_the_oracle_projections(rows):
    torch.manual_seed(12)
    actor = with_oracle_inputs(Actor(32, 2, 4))
    batch, args = inputs(rows)
    logits = actor(*args, oracle=oracle_of(rows))
    logits.masked_fill(~batch["mask"], 0).square().sum().backward()
    for module in (actor.encoder.oracle_tile_in, actor.encoder.oracle_state_in):
        assert module.bias is None and module.weight.grad.abs().sum() > 0


def test_an_actor_without_the_option_refuses_oracle_input(rows):
    with pytest.raises(ValueError, match="no oracle inputs"):
        logits_of(Actor(16, 1, 4), rows[:2], zeros(2))
    with pytest.raises(ValueError, match="already has oracle inputs"):
        with_oracle_inputs(with_oracle_inputs(Actor(16, 1, 4)))


# -- export and the loaders ------------------------------------------------------------------------------


def test_export_strips_a_trained_oracle_and_keeps_its_logits_without_one(rows, tmp_path, monkeypatch):
    from guided import networks
    from guided.deployment import load_actor
    from guided.trial import export_actor

    torch.manual_seed(13)
    learned = trained(Actor(32, 2, 4))
    want = logits_of(learned, rows)
    # The option in the config (an oracle start's trial) or only the parameters (with_oracle_inputs).
    for name, config in (("configured", CONFIG | {"oracle_inputs": True}), ("parameters", CONFIG)):
        meta = export_actor(learned.state_dict(), config, tmp_path / name)
        assert meta["config"] == CONFIG
        saved = torch.load(tmp_path / name / "actor.pt", weights_only=True)
        assert not any(k.startswith(networks.ORACLE_PREFIX) for k in saved)
        exported, _ = load_actor(tmp_path / name, "cpu")
        assert not exported.encoder.oracle_inputs
        assert torch.equal(logits_of(exported, rows), want)
        assert torch.equal(logits_of(exported, rows), logits_of(learned, rows, zeros(len(rows))))
    # The check before writing: a stripped Actor that differs is refused.
    strip = networks.strip_oracle

    def broken(weights, config):
        weights, config = strip(weights, config)
        return dict(weights, **{"state_head.bias": weights["state_head.bias"] + 1e-3}), config

    monkeypatch.setattr(networks, "strip_oracle", broken)
    with pytest.raises(RuntimeError, match="stripped Actor differs"):
        export_actor(learned.state_dict(), CONFIG | {"oracle_inputs": True}, tmp_path / "broken")


def test_an_oracle_start_keeps_every_logit_and_only_a_trial_start_may_load_it(rows, tmp_path):
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.model_server import Models
    from guided.trial import export_actor

    torch.manual_seed(14)
    source = Actor(32, 2, 4)
    export_actor(source.state_dict(), CONFIG, tmp_path / "v1")
    target = expand(tmp_path / "v1", tmp_path / "models", oracle_inputs=True)
    meta = json.loads((target / "meta.json").read_text())
    assert meta["config"] == CONFIG | {"oracle_inputs": True} and meta["encoding"] == ENCODING
    # Refused by every inference loader: load_actor, the model service, the MLX backend.
    with pytest.raises(ValueError, match="oracle inputs"):
        load_actor(target, "cpu")
    with pytest.raises(ValueError, match="oracle inputs"):
        Models(tmp_path / "models", "cpu").get(target.name)
    start, _ = load_actor(target, "cpu", oracle=True)
    reference = logits_of(source, rows)
    for oracle in (None, zeros(len(rows)), oracle_of(rows)):
        assert torch.equal(logits_of(start, rows, oracle), reference)
    with pytest.raises(ValueError, match="already has oracle inputs"):
        expand(target, tmp_path / "again", oracle_inputs=True)
    # With a wider encoding and the belief head in one step, still a start that decides as the source.
    both = expand(tmp_path / "v1", tmp_path / "both", ENCODING_V4, belief_head=True, oracle_inputs=True)
    wide, meta = load_actor(both, "cpu", oracle=True)
    assert meta["config"] == CONFIG | {"belief_head": True, "oracle_inputs": True}
    assert torch.equal(logits_of(wide, [augment(r, ENCODING_V4) for r in rows], oracle_of(rows)), reference)
    try:
        from guided.mlx_actor import Actor as MlxActor
    except ImportError:
        return
    with pytest.raises(ValueError, match="oracle inputs"):
        MlxActor(start)


def test_loaders_refuse_oracle_parameters_without_the_option(tmp_path):
    import hashlib

    from guided.deployment import load_actor

    torch.manual_seed(15)
    folder = tmp_path / "smuggled"
    folder.mkdir()
    torch.save(with_oracle_inputs(Actor(32, 2, 4)).state_dict(), folder / "actor.pt")
    sha = hashlib.sha256((folder / "actor.pt").read_bytes()).hexdigest()
    (folder / "meta.json").write_text(json.dumps({"format": "guided-actor-v1", "encoding": ENCODING,
                                                  "config": CONFIG, "weights_sha256": sha}))
    with pytest.raises(ValueError, match="oracle parameters"):
        load_actor(folder, "cpu")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"format": "guided-v1", "config": CONFIG | {"oracle_inputs": True},
                "actor": with_oracle_inputs(Actor(32, 2, 4)).state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="oracle inputs"):
        load_actor(checkpoint, "cpu")
    assert load_actor(checkpoint, "cpu", oracle=True)[0].encoder.oracle_inputs


# -- the collection's pieces -----------------------------------------------------------------------------


def test_options_and_masks():
    assert O.options({}) is None and O.options({"oracle_inputs": None}) is None
    assert O.options({"oracle_inputs": {"keep": 0}}) is None
    assert O.options({"oracle_inputs": {"keep": 0.5}}, train=False) is None
    assert O.options({"oracle_inputs": {"keep": 1}}) == (1.0, True)
    assert O.options({"oracle_inputs": {"keep": 0.3, "continuations": False}}) == (0.3, False)
    for bad in (0.5, {"keep": 1.5}, {"keep": True}, {}, {"keep": 0.5, "other": 1}, {"keep": 0.5, "continuations": 1}):
        with pytest.raises(ValueError):
            O.options({"oracle_inputs": bad})
    rng = np.random.default_rng(0)
    features = {"oracle_planes": (rng.random((16, 34)) < 0.5).astype(np.float32),
                "oracle_scalars": rng.random(4).astype(np.float32)}
    whole = O.masked(features, 1.0, 7)
    assert np.array_equal(whole[O.PLANES], features["oracle_planes"].astype(np.float16))
    assert np.array_equal(whole[O.SCALARS], features["oracle_scalars"].astype(np.float16))
    ones = {"oracle_planes": np.ones((16, 34), np.float32), "oracle_scalars": np.ones(4, np.float32)}
    kept = [np.concatenate([m[O.PLANES].ravel(), m[O.SCALARS]]) for m in (O.masked(ones, 0.3, s) for s in range(50))]
    assert abs(np.mean(kept) - 0.3) < 0.01
    assert np.array_equal(O.masked(ones, 0.3, 4)[O.PLANES], O.masked(ones, 0.3, 4)[O.PLANES])
    assert not np.array_equal(kept[0], kept[1])


def test_batches_carry_the_oracle_only_when_rows_do(rows):
    plain = [{k: v for k, v in row.items() if k not in O.FIELDS} for row in rows[:4]]
    assert O.stack(plain) is None and O.argument(make_batch(plain, "cpu")) is None
    assert not set(O.FIELDS) & set(make_batch(plain, "cpu"))
    mixed = rows[:3] + plain[3:]
    planes, scalars = O.argument(make_batch(mixed, "cpu"))
    assert planes.dtype == torch.float16 and planes.shape == (4, 16, 34) and scalars.shape == (4, 4)
    assert torch.equal(planes[:3], torch.as_tensor(np.stack([r[O.PLANES] for r in rows[:3]])))
    assert not planes[3].any() and not scalars[3].any()
    assert torch.equal(policy_batch(mixed, "cpu", oracle=True)["oracle"][0], planes)
    assert policy_batch(plain, "cpu", oracle=True)["oracle"] is None


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
        planes, scalars, *oracle = inputs
        tensors = (torch.zeros((size, planes, 34), dtype=torch.float16),
                   torch.zeros((size, scalars), dtype=torch.float16),
                   torch.zeros((size, width, 5), dtype=torch.int64),
                   torch.zeros((size, 231), dtype=torch.bool))
        if oracle:
            tensors += (torch.full((size, oracle[0], 34), 7.0, dtype=torch.float16),
                        torch.full((size, 4), 7.0, dtype=torch.float16))
        views = [t.numpy() for t in tensors]
        views[3][:, 0] = True
        return tensors, views, torch.zeros((size, 3), dtype=torch.float32), Event()

    def _answers(self, key, size, width, inputs, columns):
        return torch.zeros((size, columns), dtype=torch.float32)

    def _capture(self, compute, model, static):
        out = compute(model, *static)
        return out, Replay(compute, model, static, out)


def same(got, want):
    """Decisions alike: the actions and argmaxes exactly; the logprobs up to the padded shape's rounding."""
    return np.array_equal(got[:, [0, 2]], want[:, [0, 2]]) and np.allclose(got[:, 1], want[:, 1], rtol=1e-5, atol=1e-6)


def test_the_graph_path_feeds_the_oracle_only_when_asked(rows):
    torch.manual_seed(16)
    model = trained(Actor(16, 1, 2)).eval()
    graphed = CpuGraphs()
    with torch.no_grad():
        # Graphed (the CPU stand-in), and eager past the largest graph batch.
        for copies, graphs in ((1, graphed), (4, ActorGraphs())):
            plain = [{k: v for k, v in row.items() if k not in O.FIELDS} for row in rows] * copies
            mixed = (rows[:20] + plain[20:len(rows)]) * copies
            batch, args = inputs(mixed)
            fed = oracle_of(mixed)  # zero for the rows without
            got = graphs.decide("model:current", model, mixed, sample=False, oracle=True).result()
            want = ActorGraphs._decide(model, *args, batch["mask"], False, oracle=fed).numpy()
            assert same(got, want)
            # Not asked (e.g. a teacher labelling the same rows): the rows' oracle inputs never reach the model.
            got = graphs.decide("teacher", model, mixed, sample=False).result()
            assert same(got, ActorGraphs._decide(model, *args, batch["mask"], False).numpy())
            # Asked, but no row has them: the plain decision and the plain graph.
            got = graphs.decide("model:current", model, plain, sample=False, oracle=True).result()
            assert same(got, ActorGraphs._decide(model, *args, batch["mask"], False).numpy())
            assert not same(got, want)  # the oracle changes these decisions
            # A sample hook decides from the oracle-fed logits too.
            hooked = graphs.decide("model:current", model, mixed, sample=True, oracle=True,
                                   hook=lambda logits, mask: (logits.masked_fill(~mask, -1e9).argmax(-1), None))
            assert np.array_equal(hooked.result()[:, 0], want[:, 0])
    # The oracle's graphs are their own: its inputs count names its 16 planes.
    assert {key[3] for key in graphed.graphs} == {(rows[0]["planes"].shape[0], len(rows[0]["scalars"])),
                                                  (rows[0]["planes"].shape[0], len(rows[0]["scalars"]), 16)}


# -- collection ------------------------------------------------------------------------------------------


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


class Spy(Actor):
    """An Actor that records, per forward, whether it was given oracle input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fed = []

    def forward(self, planes, scalars, events, return_belief=False, oracle=None):
        self.fed.append(oracle is not None)
        return super().forward(planes, scalars, events, return_belief=return_belief, oracle=oracle)


def collect(actor, extra=None, *, history=None, command=None, session=None):
    cfg = configuration(rounds=2, tables=2, workers=1, samples=1, lookahead=False) | {
        "cuda_graphs": False, "playout_baseline": True, "vine": True} | (extra or {})
    base = dict(kind="collect", iteration=1, deadline=time.time() + 60, labels=False, pool=["rule"], samples=1)
    torch.manual_seed(3)
    got, records, _ = collection.collect(actor, None, cfg, "fixture", "fixture", base | (command or {}),
                                         history or {}, inference_session=session)
    assert got and records
    return got


def same_rows(a, b):
    def plain(value):
        if hasattr(value, "tobytes"):
            return (value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(plain(v) for v in value)
        return value

    return plain(a) == plain(b)


def test_off_and_keep_zero_collect_exactly_what_the_stripped_actor_does(rules):
    torch.manual_seed(21)
    base = Actor(16, 1, 1)
    reference = collect(base)
    learned = trained(base)
    assert same_rows(collect(learned), reference)
    assert same_rows(collect(learned, {"oracle_inputs": {"keep": 0}}), reference)
    assert not any(set(O.FIELDS) & set(row) for row in reference)
    with pytest.raises(ValueError, match="oracle_inputs option"):
        collect(base, {"oracle_inputs": {"keep": 0.5}})


def check_oracle_collection(monkeypatch, *, session=None, continuations=True):
    torch.manual_seed(22)
    learner = Spy(16, 1, 1, oracle_inputs=True)
    with torch.no_grad():
        torch.nn.init.normal_(learner.encoder.oracle_tile_in.weight, std=2.0)
        torch.nn.init.normal_(learner.encoder.oracle_state_in.weight, std=2.0)
    opponent, teacher = Spy(16, 1, 1), Spy(16, 1, 1)
    teacher_slot = "model:teacher:frozen"
    seen = []
    original = collection.request_groups

    def groups(pending, band=0):
        seen.extend((request["slot"], O.PLANES in request["row"]) for _, _, request in pending)
        return original(pending, band)

    monkeypatch.setattr(collection, "request_groups", groups)
    spec = {"keep": 0.5} | ({} if continuations else {"continuations": False})
    got = collect(learner, {"oracle_inputs": spec, "history_probability": 0.5},
                  history={"model:snapshot": opponent, teacher_slot: teacher},
                  command={"history": ["model:snapshot"], "labels": True, "teacher": teacher_slot}, session=session)
    slots = {slot for slot, _ in seen}
    assert {"model:current", GREEDY, "model:snapshot"} <= slots
    # The learner's requests carry the oracle; its continuations unless turned off; nobody else's.
    assert all(fed == (slot == "model:current" or (slot == GREEDY and continuations)) for slot, fed in seen)
    assert any(learner.fed) and not any(opponent.fed) and not any(teacher.fed)
    # Every row keeps what its decision saw: its logprob is the learner's given exactly that input.
    assert all(set(O.FIELDS) <= set(row) for row in got)
    batch = make_batch(got, "cpu")
    with torch.no_grad():
        args = (batch["planes"], batch["scalars"], batch["events"])
        logp = torch.log_softmax(learner(*args, oracle=O.argument(batch)).masked_fill(~batch["mask"], -1e9), -1)
        blind = torch.log_softmax(learner(*args).masked_fill(~batch["mask"], -1e9), -1)
    chosen = logp.gather(1, batch["action"][:, None]).squeeze(1)
    torch.testing.assert_close(chosen, batch["logprob"], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(blind.gather(1, batch["action"][:, None]).squeeze(1), batch["logprob"], atol=1e-3)


def test_the_learner_and_only_the_learner_decides_with_the_oracle(rules, monkeypatch):
    check_oracle_collection(monkeypatch)


def test_continuations_can_decide_without_it(rules, monkeypatch):
    check_oracle_collection(monkeypatch, continuations=False)


def test_the_graph_path_feeds_the_learner_the_rows_oracle(rules, monkeypatch):
    """The CUDA-graph path's bookkeeping (oracle buffers, two rounds in flight), on the CPU."""
    graphs = CpuGraphs()
    session = SimpleNamespace(resolve=lambda actor, history, options, factory: (graphs, actor))
    check_oracle_collection(monkeypatch, session=session)
    assert any(len(key[3]) == 3 for key in graphs.graphs) and any(len(key[3]) == 2 for key in graphs.graphs)


def test_the_gpu_probe_passes_on_the_cpu_stand_in():
    """guided.probe_oracle_graphs (the CUDA check for the maintainer) on CpuGraphs, so its logic stays sound."""
    from guided.probe_oracle_graphs import check, passed

    torch.manual_seed(23)
    actor = trained(Actor(32, 2, 4)).eval()
    report = check(actor, CpuGraphs(), "cpu", 8)
    assert passed(report) and report["argmax_agree"] == report["refreshed"] == report["plain_agree"] == 1.0
