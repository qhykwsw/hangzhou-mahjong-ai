"""Optional Actor structures: off by default, the belief head expands without changing a decision."""

from pathlib import Path
import random
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided.encoding import ENCODING, ENCODING_V4, augment, encode
from guided.inference import policy_batch
from guided.networks import Actor, actor_of, actor_options, belief_target


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

    rng, kept, game = random.Random(20261007), [], 0
    while len(kept) < 60:
        state = HandState(RoundConfig(rounds=16, allow_tiebreak=False), 20261007 + game)
        while not state.settled and len(kept) < 60:
            legal = state.legal_actions()
            if len(legal) > 1:
                kept.append(encode(state.observation()))
            state.step(rng.choice(legal))
        game += 1
    return kept


def logits_of(actor, rows, **kwargs):
    batch = policy_batch(rows, "cpu")
    with torch.no_grad():
        return actor(batch["planes"], batch["scalars"], batch["events"], **kwargs)


def test_options_default_to_the_original_actor(tmp_path):
    from guided.trial import export_actor
    import json

    base = {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}
    assert actor_options(base) == actor_options(base | {"belief_head": False}) == {}
    with pytest.raises(ValueError):
        actor_options({"belief_head": "yes"})
    torch.manual_seed(5)
    plain = Actor(32, 2, 4)
    torch.manual_seed(5)
    built = actor_of(ENCODING, 32, 2, 4, **actor_options(base | {"belief_head": False}))
    assert list(plain.state_dict()) == list(built.state_dict())
    assert not any("belief" in name for name in plain.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), built.state_dict().values()))
    export_actor(plain.state_dict(), base | {"belief_head": False}, tmp_path / "plain")
    assert json.loads((tmp_path / "plain/meta.json").read_text())["config"] == base


def test_expanded_belief_head_keeps_every_logit(rows, tmp_path):
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.trial import export_actor

    torch.manual_seed(31)
    source = Actor(32, 2, 4)
    export_actor(source.state_dict(), {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}, tmp_path / "v1")
    target = expand(tmp_path / "v1", tmp_path / "models", belief_head=True)
    narrow, _ = load_actor(tmp_path / "v1", "cpu")
    wide, meta = load_actor(target, "cpu")
    assert meta["encoding"] == ENCODING and meta["config"]["belief_head"] is True
    assert wide.encoder.belief_layer == 1
    reference = logits_of(narrow, rows)
    assert torch.equal(logits_of(wide, rows), reference)
    logits, belief = logits_of(wide, rows, return_belief=True)
    assert torch.equal(logits, reference) and belief.shape == (len(rows), 3, 34)
    assert not belief.any()  # the head's output layer starts at zero
    assert wide.encoder.belief_head[1].weight.abs().sum() > 0  # its hidden layer can learn
    with pytest.raises(ValueError, match="already has a belief head"):
        expand(target, tmp_path / "again", belief_head=True)
    # A wider encoding and the belief head in one step keep every decision too.
    both = expand(tmp_path / "v1", tmp_path / "both", ENCODING_V4, belief_head=True)
    wide4, meta4 = load_actor(both, "cpu")
    assert meta4["encoding"] == ENCODING_V4 and meta4["config"]["belief_head"] is True
    assert torch.equal(logits_of(wide4, [augment(r, ENCODING_V4) for r in rows]), reference)


def test_belief_is_returned_with_its_shape_and_the_policy_never_trains_the_head(rows):
    torch.manual_seed(8)
    actor = Actor(32, 4, 4, belief_head=True)
    torch.nn.init.normal_(actor.encoder.belief_in.weight)  # the policy reads the belief
    assert actor.encoder.belief_layer == 2
    batch = policy_batch(rows, "cpu")
    inputs = [batch[k] for k in ("planes", "scalars", "events")]
    logits, belief = actor(*inputs, return_belief=True)
    assert logits.shape == (len(rows), 231) and belief.shape == (len(rows), 3, 34)
    assert torch.equal(actor(*inputs), logits)

    def grads():
        result = {n: p.grad for n, p in actor.named_parameters()}
        actor.zero_grad(set_to_none=True)
        return result

    def touched(grad):
        return grad is not None and bool(grad.abs().sum() > 0)

    (logits.masked_fill(~batch["mask"], 0) * torch.randn_like(logits)).sum().backward()
    policy = grads()
    assert not any(touched(g) for n, g in policy.items() if n.startswith("encoder.belief_head."))
    assert touched(policy["encoder.belief_in.weight"])
    assert touched(policy["encoder.transformer.layers.0.linear1.weight"])

    counts = torch.randint(0, 5, (len(rows), 3, 34))
    oracle = torch.zeros(len(rows), 16, 34)
    for k in range(4):
        oracle[:, k:12:4] = (counts > k).float()
    target = belief_target(oracle)
    assert torch.equal(target, counts / 4)
    _, belief = actor(*inputs, return_belief=True)
    torch.nn.functional.binary_cross_entropy_with_logits(belief, target).backward()
    learned = grads()
    assert all(touched(learned[n]) for n in learned if n.startswith("encoder.belief_head."))
    # Below the head the trunk learns from the belief; the injection and everything after it do not.
    assert touched(learned["encoder.transformer.layers.1.linear1.weight"])
    assert touched(learned["encoder.tile_in.weight"])
    after = [n for n in learned if n.startswith(("encoder.belief_in.", "encoder.transformer.layers.2.",
                                                 "encoder.transformer.layers.3.", "encoder.norm.",
                                                 "discard_head.", "claim_head.", "state_head."))]
    assert after and not any(touched(learned[n]) for n in after)


def test_belief_head_needs_two_layers_and_plain_actors_return_no_belief(rows):
    with pytest.raises(ValueError, match="two layers"):
        Actor(16, 1, 4, belief_head=True)
    with pytest.raises(ValueError, match="no belief head"):
        logits_of(Actor(16, 1, 4), rows[:2], return_belief=True)
