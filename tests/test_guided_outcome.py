"""The outcome head and the stage re-rank: off by default, the head never moves a policy logit, expand adds it
bit for bit, the re-rank prefers variance behind and safety ahead inside a stage only, and every serving path
(model service, Metal, the HTTP deployment, match seats over the wire, a whole stage match) applies the switch."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")
torch = pytest.importorskip("torch")

from guided import outcome  # noqa: E402
from guided.encoding import ENCODING, encode  # noqa: E402
from guided.inference import policy_batch  # noqa: E402
from guided.networks import Actor, actor_of, actor_options, outcome_parameters, with_oracle_inputs  # noqa: E402
from guided.outcome import (  # noqa: E402
    DEFAULT_BINS,
    StageState,
    bin_index,
    outcome_loss,
    place_chance,
    rerank,
    rerank_options,
    stage_of_row,
)
from mahjong_ai.domain.types import StageKind  # noqa: E402
from test_guided_encoding_v4 import main_body, observations  # noqa: E402

BASE = {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}
LOOSE = {"top_k": 231, "mass": 1.0, "margin": 1e9}  # every legal action is a candidate: the re-rank shows


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def sample():
    return observations(hands=10, seed=20261008)


@pytest.fixture(scope="module")
def rows(sample):
    """Rows of the same decisions outside a stage, in a semifinal and in a final."""
    return [encode(obs) for plain, semifinal, final, _ in sample[:24] for obs in (plain, semifinal, final)]


def inputs(rows):
    batch = policy_batch(rows, "cpu")
    return batch, [batch[k] for k in ("planes", "scalars", "events")]


def logits_of(actor, rows, **kwargs):
    with torch.no_grad():
        return actor(*inputs(rows)[1], **kwargs)


def trained_head(actor, seed=0, scale=1.0):
    """Give the head's output layers trained-looking (random) weights."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (actor.outcome_head.discard, actor.outcome_head.claim, actor.outcome_head.state):
            p.weight.copy_(torch.randn(p.weight.shape, generator=generator) * scale)
            p.bias.copy_(torch.randn(p.bias.shape, generator=generator) * scale)
    return actor


def export(actor, config, path):
    from guided.trial import export_actor

    export_actor(actor.state_dict(), config, path)
    return path


# -- the option ------------------------------------------------------------------


def test_off_by_default_and_the_switches_validate(tmp_path):
    from guided.deployment import load_actor

    off = {"outcome_head": False}
    assert actor_options(BASE) == actor_options(BASE | off) == actor_options(BASE | {"outcome_head": None}) == {}
    assert actor_options(BASE | {"outcome_head": True}) == {"outcome_head": {"bins": list(DEFAULT_BINS)}}
    assert actor_options(BASE | {"outcome_head": {"bins": [-1, 0, 2]}}) == {"outcome_head": {"bins": [-1.0, 0.0, 2.0]}}
    for bad in ({"bins": [0]}, {"bins": [1, 0]}, {"bins": [0, 0]}, {"bins": [0, float("nan")]}, {"edges": [0, 1]},
                {"bins": "01"}, {"bins": [0, True]}, "yes", 1):
        with pytest.raises(ValueError):
            actor_options({"outcome_head": bad})
    torch.manual_seed(5)
    plain = Actor(32, 2, 4)
    torch.manual_seed(5)
    built = actor_of(ENCODING, 32, 2, 4, **actor_options(BASE | off))
    assert list(plain.state_dict()) == list(built.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), built.state_dict().values()))
    assert not any("outcome" in name for name in plain.state_dict()) and outcome_parameters(plain) == []
    # Exports without the option and with it switched off are the same files.
    a = json.loads((export(plain, BASE, tmp_path / "a") / "meta.json").read_text())
    b = json.loads((export(plain, BASE | off, tmp_path / "b") / "meta.json").read_text())
    assert a == b and a["config"] == BASE
    with pytest.raises(ValueError, match="no outcome head"):
        plain(*inputs([blank_row()])[1], return_outcome=True)
    # The meta.json switch: normalized, needs the head.
    assert rerank_options(None) is None and rerank_options(False) is None
    assert rerank_options(True) == outcome.RERANK_DEFAULTS
    assert rerank_options({"top_k": 2})["top_k"] == 2
    for bad in ({"top_k": 0}, {"top_k": 1.5}, {"mass": 0}, {"mass": 1.5}, {"margin": -1}, {"min_gain": -1},
                {"per_hand_sd": 0}, {"temperature": 1}, {"margin": float("inf")}, "on"):
        with pytest.raises(ValueError):
            rerank_options(bad)
    meta = json.loads((tmp_path / "a/meta.json").read_text())
    (tmp_path / "a/meta.json").write_text(json.dumps(meta | {"outcome_rerank": True}))
    with pytest.raises(ValueError, match="outcome head"):
        load_actor(tmp_path / "a", "cpu")
    (tmp_path / "a/meta.json").write_text(json.dumps(meta | {"outcome_rerank": False}))
    assert load_actor(tmp_path / "a", "cpu")[1]["outcome_rerank"] is None


def blank_row():
    from guided.model_server import blank

    return blank()


def test_the_head_never_moves_a_policy_logit(rows):
    torch.manual_seed(7)
    plain = Actor(32, 2, 4)
    after_plain = torch.rand(4)
    torch.manual_seed(7)
    headed = Actor(32, 2, 4, outcome_head=True)
    # The head draws from its own seed: every other parameter, and the caller's stream after, are as without it.
    assert torch.equal(torch.rand(4), after_plain)
    old, new = plain.state_dict(), headed.state_dict()
    assert set(new) - set(old) == {f"outcome_head.{m}.{p}" for m in ("tile_hidden", "state_hidden", "discard",
                                                                       "claim", "state") for p in ("weight", "bias")}
    assert all(torch.equal(old[k], new[k]) for k in old)
    reference = logits_of(plain, rows)
    assert torch.equal(logits_of(headed, rows), reference)
    logits, out = logits_of(headed, rows, return_outcome=True)
    assert torch.equal(logits, reference) and out.shape == (len(rows), 231, len(DEFAULT_BINS))
    assert not out.any()  # the output layers start at zero: every action's distribution uniform
    assert headed.outcome_head.tile_hidden.weight.abs().sum() > 0  # its hidden layers can learn
    # A trained head still leaves the policy alone, with a belief head and shared suits too.
    trained_head(headed)
    assert torch.equal(logits_of(headed, rows), reference)
    torch.manual_seed(8)
    both = Actor(32, 2, 4, belief_head="v2", suit_embedding="shared", outcome_head={"bins": [-8, 0, 10]})
    torch.manual_seed(8)
    alone = Actor(32, 2, 4, belief_head="v2", suit_embedding="shared")
    logits, belief, out = logits_of(both, rows, return_belief=True, return_outcome=True)
    want, want_belief = logits_of(alone, rows, return_belief=True)
    assert torch.equal(logits, want) and torch.equal(belief.tiles, want_belief.tiles)
    assert out.shape == (len(rows), 231, 3)
    assert torch.equal(logits_of(with_oracle_inputs(both), rows), want)
    assert with_oracle_inputs(both).outcome_spec == {"bins": [-8.0, 0.0, 10.0]}


def test_the_head_follows_the_action_order(rows):
    """Bin 0 of each action reproduces the policy's logit of that action when the head is wired to copy it."""
    torch.manual_seed(9)
    actor = Actor(32, 2, 4, outcome_head={"bins": [0, 1]}).eval()
    head, shift = actor.outcome_head, 1000.0
    with torch.no_grad():
        # gelu(x + 1000) = x + 1000 in fp32: the hidden layers pass the tokens through, shifted.
        for hidden in (head.tile_hidden, head.state_hidden):
            hidden.weight.copy_(torch.eye(32))
            hidden.bias.fill_(shift)
        for out, policy, width in ((head.discard, actor.discard_head, 1), (head.claim, actor.claim_head, 4),
                                   (head.state, actor.state_head, 65)):
            out.weight.zero_()
            out.bias.zero_()
            for k in range(width):
                out.weight[2 * k] = policy.weight[k]
                out.bias[2 * k] = policy.bias[k] - shift * policy.weight[k].sum()
    logits, out = logits_of(actor, rows, return_outcome=True)
    torch.testing.assert_close(out[..., 0], logits, rtol=0, atol=0.02)
    assert not out[..., 1].any()


def test_detached_outcome_training_keeps_the_policy_and_the_loss_is_the_taken_actions_cross_entropy(rows):
    torch.manual_seed(10)
    actor = Actor(32, 2, 4, outcome_head=True)
    batch, args = inputs(rows)
    reference = logits_of(actor, rows)
    actions = batch["mask"].float().argmax(-1)
    scores = torch.tensor([(-8.0, 0.0, 24.0, 10.0, -1.0)[i % 5] for i in range(len(rows))])
    optimizer = torch.optim.SGD(outcome_parameters(actor), lr=0.5)
    _, out = actor(*args, return_outcome="detached")
    loss = outcome_loss(out, actions, scores)
    manual = torch.nn.functional.cross_entropy(out[torch.arange(len(rows)), actions], bin_index(scores))
    assert torch.allclose(loss, manual)
    weights = torch.rand(len(rows))
    picked = out[torch.arange(len(rows)), actions]
    each = torch.nn.functional.cross_entropy(picked, bin_index(scores), reduction="none")
    assert torch.allclose(outcome_loss(out, actions, scores, weights=weights), (each * weights).sum() / weights.sum())
    loss.backward()
    assert all(p.grad is None for name, p in actor.named_parameters() if not name.startswith("outcome_head."))
    optimizer.step()
    assert torch.equal(logits_of(actor, rows), reference)
    assert outcome_loss(logits_of(actor, rows, return_outcome=True)[1], actions, scores) < loss
    # Without detaching, the loss reaches the trunk (a recipe chooses).
    actor.zero_grad()
    outcome_loss(actor(*args, return_outcome=True)[1], actions, scores).backward()
    assert actor.encoder.tile_in.weight.grad is not None and actor.encoder.tile_in.weight.grad.abs().sum() > 0


def test_every_settlement_score_has_its_own_bin():
    assert [bin_index(v) for v in DEFAULT_BINS] == list(range(len(DEFAULT_BINS)))
    scores = {0} | {-f for f in (1, 2, 4, 8, 16, 32, 64)} | {-8 * f for f in (1, 2, 4, 8, 16, 32, 64)} | {
        10 * f for f in (1, 2, 4, 8, 16, 32, 64)} | {24 * f for f in (1, 2, 4, 8, 16, 32)}
    assert all(DEFAULT_BINS[bin_index(s)] == s for s in scores)
    # Nearest value, a midpoint going up; beyond the ends, the end bins.
    assert bin_index(22) == DEFAULT_BINS.index(24) and bin_index(21.99) == DEFAULT_BINS.index(20)
    assert bin_index(-0.5) == DEFAULT_BINS.index(0) and bin_index(5) == DEFAULT_BINS.index(10)
    assert bin_index(-1e6) == 0 and bin_index(1e6) == len(DEFAULT_BINS) - 1
    values = np.array([-600, -9, -8, 0, 3, 7, 22, 1000, 1536])
    assert bin_index(values).tolist() == bin_index(torch.tensor(values)).tolist() == [bin_index(v) for v in values]
    assert bin_index(torch.tensor(values)).dtype == torch.long
    assert bin_index(4, [-8, 0, 10]) == 1 and bin_index(5, [-8, 0, 10]) == 2


def test_expand_adds_the_head_to_fp_rl_227s_shape_and_the_v2_belief_bit_for_bit(rows, tmp_path):
    """fp-rl-227's shape (a trained v1 belief, shared suits), its v2 widening, and both in one step."""
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.model_server import Models, decide

    config = BASE | {"layers": 4, "belief_head": True, "suit_embedding": "shared"}
    torch.manual_seed(227)
    source = actor_of(ENCODING, 32, 4, 4, belief_head=True, suit_embedding="shared")
    encoder = source.encoder
    with torch.no_grad():  # a trained-looking head, injection and suit tables
        for p in (encoder.belief_in.weight, encoder.belief_in.bias, encoder.belief_head[3].weight,
                  encoder.belief_head[3].bias, encoder.tile_embedding.ranks):
            torch.nn.init.normal_(p, std=0.5)
    export(source, config, tmp_path / "v1")
    narrow, _ = load_actor(tmp_path / "v1", "cpu")
    target = expand(tmp_path / "v1", tmp_path / "models", outcome_head=True)
    wide, meta = load_actor(target, "cpu")
    assert meta["config"] == config | {"outcome_head": {"bins": list(DEFAULT_BINS)}}
    assert meta["encoding"] == ENCODING and target.name == meta["weights_sha256"]
    for part in (rows, rows[:1], rows[1:3], rows[5:12]):
        logits, belief = logits_of(narrow, part, return_belief=True)
        assert torch.equal(logits_of(wide, part), logits)
        wide_logits, wide_belief, out = logits_of(wide, part, return_belief=True, return_outcome=True)
        assert torch.equal(wide_logits, logits) and torch.equal(wide_belief, belief) and not out.any()
    old, new = narrow.state_dict(), wide.state_dict()
    assert all(torch.equal(old[k], new[k]) for k in old) and set(new) - set(old) == {
        k for k in new if k.startswith("outcome_head.")}
    assert new["outcome_head.tile_hidden.weight"].abs().sum() > 0
    served, _ = Models(target.parent, "cpu").get(target.name)
    batch, args = inputs(rows)
    with torch.inference_mode():
        assert torch.equal(decide(served, *args, batch["mask"])[0], decide(narrow, *args, batch["mask"])[0])
    with pytest.raises(ValueError, match="already has an outcome head"):
        expand(target, tmp_path / "again", outcome_head=True)
    # The v2 belief variant: widened from the expanded Actor, or both options in one step, custom bins too.
    for start, options in ((target, {"belief_head": "v2"}),
                           (tmp_path / "v1", {"belief_head": "v2", "outcome_head": {"bins": [-8, 0, 10]}})):
        made = expand(start, tmp_path / f"v2-{len(options)}", **options)
        v2, meta2 = load_actor(made, "cpu")
        assert meta2["config"]["belief_head"] == "v2" and meta2["config"]["outcome_head"]
        assert torch.equal(logits_of(v2, rows), logits_of(narrow, rows))
        assert not logits_of(v2, rows, return_outcome=True)[1].any()
    assert load_actor(made, "cpu")[1]["config"]["outcome_head"] == {"bins": [-8.0, 0.0, 10.0]}


# -- the stage ---------------------------------------------------------------------


def test_the_rows_carry_their_stage_and_the_dealer(sample):
    for plain, semifinal, final, qualifier in sample[:30]:
        assert "stage_state" not in encode(plain) and "stage_state" not in encode(qualifier)
        for obs, places in ((semifinal, 2), (final, 1)):
            row = encode(obs)
            current, total = obs.stage.effective_horizon
            seat = obs.stage.seat
            assert row["stage_state"] == (places, total - current + 1,
                                          *(float(obs.stage.stage_scores[(seat + k) % 4]) for k in range(4)))
            state = stage_of_row(row)
            assert state.dealer == (obs.stage.dealer - seat) % 4 and state.scores == tuple(row["stage_state"][2:])


def test_the_chance_of_each_outcome_follows_the_stage():
    from guided.outcome import _scenarios

    for value in (-512, -16, -8, -4, -1, 0, 10, 24, 768):
        for dealer in (0, 1, 2, 3, None):
            cases = _scenarios(float(value), dealer)
            assert sum(w for w, _ in cases) == pytest.approx(1.0)
            assert all(sum(shares) == pytest.approx(-value) for _, shares in cases)  # zero-sum
    # A final's last hand is exact: from -30 behind a dealer at 0, a win of 10 leaves the learner at -20 and the
    # dealer at -8, one of 20 at -10 and -16; every payment loses.
    chance = place_chance(StageState(1, 1, (-30.0, 0.0, -100.0, -100.0), 1))
    assert chance[bin_index(10)] == 0 and chance[bin_index(20)] == 1
    assert not chance[: bin_index(0) + 1].any() and chance[bin_index(20):].min() == 1
    # Even, with many hands left: more is better, and a semifinal starts at a half.
    even = place_chance(StageState(2, 80, (0.0, 0.0, 0.0, 0.0), 0))
    assert (np.diff(even) >= 0).all() and even[bin_index(0)] == pytest.approx(0.5, abs=1e-4)
    assert place_chance(StageState(1, 80, (0.0, 0.0, 0.0, 0.0), 0))[bin_index(0)] == pytest.approx(0.25, abs=1e-3)
    # Far ahead it hardly matters; the order of the other three does not either.
    ahead = place_chance(StageState(1, 2, (500.0, 0.0, -10.0, -20.0), 0))
    assert ahead[bin_index(-64):].min() > 0.999
    np.testing.assert_allclose(place_chance(StageState(1, 9, (5.0, 40.0, -10.0, -35.0), None)),
                               place_chance(StageState(1, 9, (5.0, -35.0, 40.0, -10.0), None)), atol=1e-6)
    assert outcome.utilities([None, StageState(1, 1, (-30.0, 0.0, -100.0, -100.0), 1)]).shape == (2, 24)
    assert not outcome.utilities([None])[0].any()


def synthetic(policy):
    """Rows with legal actions 0 (safe: always 0), 1 (a gamble: -16 or +20) and 2 (+768), the policy's logits."""
    n = len(policy)
    logits = torch.full((n, 231), -3.0)
    logits[:, :3] = torch.tensor(policy)
    mask = torch.zeros((n, 231), dtype=torch.bool)
    mask[:, :3] = True
    out = torch.zeros((n, 231, len(DEFAULT_BINS)))
    out[:, :, :] = -30.0
    out[:, 0, bin_index(0)] = 30.0
    out[:, 1, bin_index(-16)] = out[:, 1, bin_index(20)] = 30.0
    out[:, 2, bin_index(768)] = 30.0
    return logits, out, mask


def test_the_rerank_wants_variance_behind_and_safety_ahead_and_nothing_without_a_stage():
    behind = StageState(1, 2, (-60.0, 0.0, -20.0, -30.0), 2)  # a final, far behind, two hands left
    ahead = StageState(2, 1, (10.0, 40.0, 0.0, -50.0), 0)  # a semifinal's last hand, second, dealing
    logits, out, mask = synthetic([(1.0, 0.8, -5.0), (0.8, 1.0, -5.0), (1.0, 0.8, -5.0), (1.0, -2.0, -5.0)])
    stages = [behind, ahead, None, behind]
    # Behind, the gamble (a near-tie); ahead, the safe play over the policy's gamble; no stage, the argmax;
    # the gamble far below the argmax is no near-tie.
    assert rerank(logits, out, mask, stages).tolist() == [1, 0, 0, 0]
    assert rerank(logits, out, mask, [None] * 4).tolist() == [0, 1, 0, 0]
    # The jackpot is far from the policy's choice: only loose options reach it (ahead, the safe play already
    # advances for sure: a tie with it goes to the more probable action).
    assert rerank(logits, out, mask, stages, **LOOSE).tolist() == [2, 0, 0, 2]
    assert rerank(logits, out, mask, stages, top_k=1).tolist() == [0, 1, 0, 0]
    assert rerank(logits, out, mask, stages, min_gain=0.99).tolist() == [0, 1, 0, 0]
    # A head that knows nothing (zero: every action uniform) never moves a decision.
    assert rerank(logits, torch.zeros_like(out), mask, stages, **LOOSE).tolist() == [0, 1, 0, 0]
    # choose on its own: graph-safe tensors in, the same choices out.
    utility = torch.as_tensor(outcome.utilities(stages))
    log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
    chosen = outcome.choose(log_probs, out, mask, utility, log_probs.argmax(-1),
                            **outcome.choice_options(rerank_options(True)))
    assert chosen.tolist() == [1, 0, 0, 0]


# -- serving -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A model with a trained-looking outcome head, and its re-rank variants (default and loose)."""
    root = tmp_path_factory.mktemp("models")
    torch.manual_seed(11)
    actor = trained_head(Actor(32, 2, 4, outcome_head=True), scale=2.0)
    from guided.trial import export_actor

    meta = export_actor(actor.state_dict(), BASE | {"outcome_head": True}, root / "staging")
    (root / "staging").rename(root / meta["weights_sha256"])
    plain = root / meta["weights_sha256"]
    return {"root": root, "plain": plain.name, "default": outcome.variant(plain, root).name,
            "loose": outcome.variant(plain, root, LOOSE).name}


def reference(models, name, rows):
    """What a model decides on these rows, computed directly: argmax, re-ranked where a row has a stage."""
    from guided.deployment import load_actor

    actor, meta = load_actor(models["root"] / name, "cpu")
    batch, args = inputs(rows)
    with torch.no_grad():
        logits, out = actor(*args, return_outcome=True)
    switch = meta.get("outcome_rerank")
    if not switch:
        return torch.log_softmax(logits.masked_fill(~batch["mask"], -1e9), -1).argmax(-1).tolist()
    return rerank(logits, out, batch["mask"], [stage_of_row(r) for r in rows], **switch).tolist()


def test_the_variant_is_the_same_weights_with_the_switch(models):
    from guided.deployment import load_actor

    root = models["root"]
    plain, plain_meta = load_actor(root / models["plain"], "cpu")
    loose, loose_meta = load_actor(root / models["loose"], "cpu")
    assert plain_meta.get("outcome_rerank") is None
    assert loose_meta["outcome_rerank"] == outcome.RERANK_DEFAULTS | LOOSE
    assert loose_meta["weights_sha256"] == plain_meta["weights_sha256"] == models["plain"] != models["loose"]
    assert (root / models["plain"] / "actor.pt").read_bytes() == (root / models["loose"] / "actor.pt").read_bytes()
    assert outcome.variant(root / models["plain"], root, LOOSE).name == models["loose"]  # the same switch, the same id
    with pytest.raises(ValueError, match="outcome head"):
        torch.manual_seed(1)
        bare = export(Actor(32, 2, 4), BASE, root.parent / "bare")
        outcome.variant(bare, root)


def test_the_model_service_reranks_stage_rows_only(models, rows):
    from guided.model_server import Models, Service
    from selfplay_core.wire import pack, unpack

    service = Service(Models(models["root"], "cpu"))
    plain_rows = rows[0::3]
    for name in ("plain", "default", "loose"):
        model = models[name]
        bodies = [pack(model, r, lookahead=True) for r in rows]
        got = [a["index"] for a in service.decide(model, [unpack(b + b.stage) for b in bodies])]
        assert got == reference(models, model, rows)
        assert service.inputs(model)["stage"] is (name != "plain")
    argmax = reference(models, models["plain"], rows)
    loose = reference(models, models["loose"], rows)
    assert sum(a != b for a, b in zip(argmax, loose)) > len(rows) // 4  # the loose switch re-ranks often
    assert all(argmax[i] == loose[i] for i in range(0, len(rows), 3))  # outside a stage, never
    # Without its stage a request decides as the plain model: a stage row without the trailer, a plain row.
    model = models["loose"]
    bodies = [pack(model, r, lookahead=True) for r in rows]
    assert [a["index"] for a in service.decide(model, [unpack(b) for b in bodies])] == argmax
    assert [a["index"] for a in service.decide(model, [unpack(pack(model, r, lookahead=True)) for r in plain_rows])] \
        == argmax[0::3]
    for bad in (bodies[1] + bodies[1].stage[:-1], bodies[1] + b"MJS2" + bodies[1].stage[4:],
                bodies[1] + bodies[1].stage[:4] + bytes([3]) + bodies[1].stage[5:]):
        with pytest.raises(ValueError):
            unpack(bad)
    # The JSON endpoint's base64 rows re-rank the same.
    import base64

    payloads = [{"lookahead": True, "encoded": {
        k: base64.b64encode(np.ascontiguousarray(np.float16(v) if k == "risk_beta" else v).tobytes()).decode()
        for k, v in r.items()}} for r in rows]
    assert [a["index"] for a in service.decide(model, payloads)] == loose


def test_complete_observations_rerank_in_both_services(models, sample):
    from guided.deployment import BatchService, load_actor, parse_observation
    from guided.model_server import Models, Service
    from mahjong_ai.execution.candidate_runtime import observation_payload
    from selfplay_core.actions import action_index

    staged = [obs for plain, semifinal, final, _ in sample[:12] for obs in (plain, semifinal, final)]
    payloads = [observation_payload(obs)[0] for obs in staged]
    rows = [encode(parse_observation(p)) for p in payloads]
    assert sum("stage_state" in r for r in rows) == 2 * len(sample[:12])
    want = reference(models, models["loose"], rows)
    service = Service(Models(models["root"], "cpu"))
    answers = service.decide(models["loose"], [{"observation": p} for p in payloads])
    assert [[action_index(a) for a in obs.legal_actions][x["legal_index"]] for obs, x in zip(staged, answers)] == want
    actor, metadata = load_actor(models["root"] / models["loose"], "cpu")
    http = BatchService(actor, metadata, max_batch=10)
    try:
        assert [http.submit(p).result(timeout=30)["index"] for p in payloads] == want
    finally:
        http.close()


def test_match_seats_send_the_stage_only_to_a_reranking_model(models, sample, monkeypatch):
    """Match seats (mahjong_ai's ModelPolicy, unchanged) against the service: inside a stage, only a re-ranking
    model's requests carry the stage, after the plain body; every other request is the bytes they always were."""
    import socket
    import threading
    import time
    import urllib.request

    from guided.model_server import Models, Service, serve
    from mahjong_ai.execution import model_seat
    from selfplay_core import wire

    service = Service(Models(models["root"], "cpu"))
    received = []
    unpack = wire.unpack
    monkeypatch.setattr(wire, "unpack", lambda body: (received.append(bytes(body)), unpack(body))[1])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=serve, args=(service, port), daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            with urllib.request.urlopen(url + "/health", timeout=1):
                break
        except OSError:
            time.sleep(0.02)
    monkeypatch.setenv(model_seat.SERVER_ENV, url)
    monkeypatch.setenv(model_seat.ENCODER_ENV, str(PROJECT))
    monkeypatch.setattr(model_seat, "_ENCODER", [])

    def play(sha, observations):
        received.clear()
        policy = model_seat.ModelPolicy(sha, "v")
        chosen = [policy.act(obs) for obs in observations]
        policy.close()
        assert policy.stats["errors"] == policy.stats["fallback_actions"] == 0
        return list(received), chosen

    staged = [obs for plain, semifinal, final, _ in sample[:10] for obs in (plain, semifinal, final)]
    rows = [encode(obs) for obs in staged]
    plain, _ = play(models["plain"], staged)
    assert plain == [main_body(models["plain"], r) for r in rows]
    sent, chosen = play(models["loose"], staged)
    trailers = [wire.STAGE.pack(b"MJS1", *r["stage_state"][:2], *r["stage_state"][2:]) if "stage_state" in r else b""
                for r in rows]
    assert sent == [main_body(models["loose"], r) + t for r, t in zip(rows, trailers)]
    from selfplay_core.actions import action_index

    assert [action_index(a) for a in chosen] == reference(models, models["loose"], rows)


def test_a_stage_match_plays_through_the_rerank(models, monkeypatch):
    """A whole semifinal of two short games: a re-ranking model seat over the wire, three rule seats."""
    import socket
    import threading
    import time
    import urllib.request

    from guided import model_server
    from guided.model_server import Models, Service, serve
    from mahjong_ai.domain.engine import MahjongEngine
    from mahjong_ai.domain.types import RoundConfig
    from mahjong_ai.execution import model_seat
    from mahjong_ai.strategies.common import RulePolicy

    service = Service(Models(models["root"], "cpu"))
    staged, moved = [], []
    utility = model_server.staged_utility
    monkeypatch.setattr(model_server, "staged_utility",
                        lambda rows, meta: (lambda u: (staged.append(u is not None), u)[1])(utility(rows, meta)))
    choose = outcome.choose
    monkeypatch.setattr(outcome, "choose", lambda log_probs, out, mask, u, greedy, **kw: (
        lambda c: (moved.append(int((c != greedy).sum())), c)[1])(choose(log_probs, out, mask, u, greedy, **kw)))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=serve, args=(service, port), daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            with urllib.request.urlopen(url + "/health", timeout=1):
                break
        except OSError:
            time.sleep(0.02)
    monkeypatch.setenv(model_seat.SERVER_ENV, url)
    monkeypatch.setenv(model_seat.ENCODER_ENV, str(PROJECT))
    monkeypatch.setattr(model_seat, "_ENCODER", [])
    seat = model_seat.ModelPolicy(models["loose"], "v")
    config = RoundConfig(rounds=2, stage=StageKind.ADVANCEMENT, games_per_round=2)
    result = MahjongEngine(record_events=False).run_round([seat, RulePolicy({}), RulePolicy({}), RulePolicy({})],
                                                          config, 20261008)
    seat.close()
    assert result.completed and result.rounds_played == 4 and sum(result.total_score_by_seat) == 0
    assert seat.stats["actions"] > 10 and seat.stats["errors"] == seat.stats["fallback_actions"] == 0
    assert staged and all(staged) and sum(moved) > 0


# -- Metal ---------------------------------------------------------------------------


def test_metal_computes_the_head_and_the_rerank_as_torch(models, rows):
    mx = pytest.importorskip("mlx.core")
    from guided.deployment import load_actor
    from guided.mlx_actor import Actor as MetalActor
    from guided.model_server import Models, Service

    actor, meta = load_actor(models["root"] / models["loose"], "cpu")
    metal = MetalActor(actor, rerank=meta["outcome_rerank"])
    batch, args = inputs(rows)
    with torch.no_grad():
        want_logits, want_out = actor(*args, return_outcome=True)
    arrays = [mx.array(a.numpy().astype(np.int32 if a.dtype == torch.int64 else a.numpy().dtype)) for a in args]
    logits, belief, out = metal.outputs(*arrays, outcome=True)
    assert belief is None
    np.testing.assert_allclose(np.asarray(logits), want_logits.numpy(), rtol=0.01, atol=0.005)
    np.testing.assert_allclose(np.asarray(out), want_out.numpy(), rtol=0.01, atol=0.02)
    # The re-ranked decision: by choose's equation on Metal's own numbers, up to near-ties.
    states = [stage_of_row(r) for r in rows]
    utility = outcome.utilities(states)
    choices, valid = metal.decide_reranked(*arrays, mx.array(batch["mask"].numpy()), mx.array(utility))
    assert bool(valid.item())
    logits, out = (torch.from_numpy(np.asarray(a.astype(mx.float32))) for a in (logits, out))
    log_probs = torch.log_softmax(logits.masked_fill(~batch["mask"], -1e9), -1)
    dist = out.softmax(-1)
    value = (dist * torch.from_numpy(utility)[:, None, :]).sum(-1).masked_fill(~batch["mask"], -1.0)
    picked = value.gather(1, torch.tensor(np.asarray(choices).tolist())[:, None])[:, 0]
    greedy = log_probs.argmax(-1)
    best = torch.maximum(value.max(-1).values, value.gather(1, greedy[:, None])[:, 0])
    assert (picked >= best - 1e-3).all()
    assert all(c == g for c, g, s in zip(np.asarray(choices).tolist(), greedy.tolist(), states) if s is None)
    # Served on Metal: stage rows go through the re-rank, the plain model keeps its argmax.
    from selfplay_core.wire import pack, unpack

    service = Service(Models(models["root"], "mlx"))
    for name in ("plain", "loose"):
        bodies = [pack(models[name], r, lookahead=True) for r in rows]
        got = [a["index"] for a in service.decide(models[name], [unpack(b + b.stage) for b in bodies])]
        want = reference(models, models[name], rows)
        assert sum(a == b for a, b in zip(got, want)) >= 0.9 * len(rows)
    with pytest.raises(ValueError, match="outcome head"):
        torch.manual_seed(2)
        MetalActor(Actor(32, 1, 4), rerank=rerank_options(True))


# -- CUDA (run on a GPU host: pytest tests/test_guided_outcome.py -k cuda) ------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graphs_rerank_as_eager_and_the_reference(models, rows):
    from guided.model_server import Models, Service
    from selfplay_core.wire import pack, unpack

    graphed = Service(Models(models["root"], "cuda"), graphs=True)
    eager = Service(Models(models["root"], "cuda"), graphs=False)
    for name in ("plain", "loose"):
        model = models[name]
        graphed.load(model)  # captures the plain and the re-ranking graphs of every warm shape
        requests = [unpack(b + b.stage) for b in (pack(model, r, lookahead=True) for r in rows)]
        got = [a["index"] for a in graphed.decide(model, requests)]
        assert got == [a["index"] for a in eager.decide(model, requests)]
        want = reference(models, model, rows)  # fp32 on the CPU: fp16 may break a near-tie differently
        assert sum(a == b for a, b in zip(got, want)) >= 0.9 * len(rows)
