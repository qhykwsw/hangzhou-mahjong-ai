"""Optional Actor structures: off by default, the belief heads (v1, v2) expand without changing a decision,
shared suits start from the source's tables; the v2 belief's targets and worker-side labels."""

from pathlib import Path
import random
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided.encoding import ENCODING, ENCODING_V4, augment, encode
from guided.inference import policy_batch
from guided.networks import (BELIEF_STATE, BELIEF_TILES, Actor, BeliefV2, actor_of, actor_options,
                             belief_parameters, belief_target, belief_target_v2, with_oracle_inputs)


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
    off = {"belief_head": False, "suit_embedding": None}
    assert actor_options(base) == actor_options(base | off) == {}
    assert actor_options(base | {"belief_head": "v2"}) == {"belief_head": "v2"}
    for bad in ({"belief_head": "yes"}, {"belief_head": "v3"}, {"belief_head": 1}, {"suit_embedding": "per-suit"}):
        with pytest.raises(ValueError):
            actor_options(bad)
    with pytest.raises(ValueError, match="belief_head"):
        Actor(32, 2, 4, belief_head="v1")
    torch.manual_seed(5)
    plain = Actor(32, 2, 4)
    torch.manual_seed(5)
    built = actor_of(ENCODING, 32, 2, 4, **actor_options(base | off))
    assert list(plain.state_dict()) == list(built.state_dict())
    assert not any(k in name for name in plain.state_dict() for k in ("belief", "ranks", "suits", "others"))
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), built.state_dict().values()))
    export_actor(plain.state_dict(), base | off, tmp_path / "plain")
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


def belief_block(actor, rows):
    """The tokens entering block belief_layer - 1's output (before the head) and block belief_layer (after it)."""
    seen = {}
    layers = actor.encoder.transformer.layers
    hooks = [layers[actor.encoder.belief_layer - 1].register_forward_hook(
                 lambda module, args, output: seen.__setitem__("before", output)),
             layers[actor.encoder.belief_layer].register_forward_pre_hook(
                 lambda module, args, kwargs: seen.__setitem__("after", args[0]), with_kwargs=True)]
    try:
        out = logits_of(actor, rows, return_belief=True)
    finally:
        for hook in hooks:
            hook.remove()
    return out, seen["before"], seen["after"]


def test_v1_belief_keeps_its_parameters_and_equation(rows):
    torch.manual_seed(12)
    actor = Actor(32, 4, 4, belief_head=True).eval()
    torch.nn.init.normal_(actor.encoder.belief_in.weight, std=0.5)
    encoder = actor.encoder
    assert encoder.belief_version == 1 and encoder.belief_layer == 2
    assert {k: tuple(v.shape) for k, v in actor.state_dict().items() if "belief" in k} == {
        "encoder.belief_head.0.weight": (32,), "encoder.belief_head.0.bias": (32,),
        "encoder.belief_head.1.weight": (32, 32), "encoder.belief_head.1.bias": (32,),
        "encoder.belief_head.3.weight": (3, 32), "encoder.belief_head.3.bias": (3,),
        "encoder.belief_in.weight": (32, 3), "encoder.belief_in.bias": (32,)}
    (_, belief), before, after = belief_block(actor, rows)
    with torch.no_grad():
        want = encoder.belief_head(before[:, :34])
        injected = encoder.belief_in(torch.sigmoid(want))
    assert torch.equal(belief, want.transpose(1, 2))
    assert torch.equal(after, torch.cat([before[:, :34] + injected, before[:, 34:]], 1))


def test_v2_beliefs_their_equation_and_the_policy_never_trains_the_heads(rows):
    torch.manual_seed(8)
    actor = Actor(32, 4, 4, belief_head="v2")
    encoder = actor.encoder
    assert encoder.belief_version == 2 and encoder.belief_layer == 2
    for layer in (encoder.belief_in, encoder.belief_state_in):  # the policy reads the beliefs
        assert not layer.weight.any() and not layer.bias.any()
        torch.nn.init.normal_(layer.weight)
    batch = policy_batch(rows, "cpu")
    inputs = [batch[k] for k in ("planes", "scalars", "events")]
    logits, belief = actor(*inputs, return_belief=True)
    assert isinstance(belief, BeliefV2) and len(BELIEF_TILES) == 7 and len(BELIEF_STATE) == 6
    assert logits.shape == (len(rows), 231)
    assert belief.tiles.shape == (len(rows), 7, 34) and belief.state.shape == (len(rows), 6)
    assert torch.equal(actor(*inputs), logits)
    # The equation: both heads read the tokens after belief_layer blocks, their sigmoids go back in.
    actor.eval()
    (_, belief), before, after = belief_block(actor, rows)
    with torch.no_grad():
        tiles, state = encoder.belief_head(before[:, :34]), encoder.belief_state_head(before[:, 34])
        want = torch.cat([before[:, :34] + encoder.belief_in(torch.sigmoid(tiles)),
                          (before[:, 34] + encoder.belief_state_in(torch.sigmoid(state)))[:, None], before[:, 35:]], 1)
    torch.testing.assert_close(belief.tiles, tiles.transpose(1, 2), rtol=1e-6, atol=1e-6)
    assert torch.equal(belief.state, state)
    torch.testing.assert_close(after, want, rtol=1e-6, atol=1e-6)
    actor.train()

    def grads():
        result = {n: p.grad for n, p in actor.named_parameters()}
        actor.zero_grad(set_to_none=True)
        return result

    def touched(grad):
        return grad is not None and bool(grad.abs().sum() > 0)

    heads = ("encoder.belief_head.", "encoder.belief_state_head.")
    names = {id(p): n for n, p in actor.named_parameters()}
    assert sorted(names[id(p)] for p in belief_parameters(actor)) == sorted(
        n for n in names.values() if n.startswith(heads))
    logits, _ = actor(*inputs, return_belief=True)
    (logits.masked_fill(~batch["mask"], 0) * torch.randn_like(logits)).sum().backward()
    policy = grads()
    assert not any(touched(g) for n, g in policy.items() if n.startswith(heads))
    assert all(touched(policy[n]) for n in ("encoder.belief_in.weight", "encoder.belief_state_in.weight",
                                             "encoder.transformer.layers.0.linear1.weight"))
    counts = torch.randint(0, 5, (len(rows), 4, 34))
    oracle = torch.zeros(len(rows), 16, 34)
    for k in range(4):
        oracle[:, k::4] = (counts > k).float()
    shanten = torch.randint(-1, 9, (len(rows), 3))
    scalars = torch.cat([((8 - shanten).double() / 9).float(), torch.rand(len(rows), 1)], 1)
    waits = (torch.rand(len(rows), 3, 34) < 0.1).float() * (shanten <= 0).float()[..., None]
    target = belief_target_v2(oracle, scalars, waits)
    _, belief = actor(*inputs, return_belief=True)
    sum(torch.nn.functional.binary_cross_entropy_with_logits(p, t) for p, t in zip(belief, target)).backward()
    learned = grads()
    assert all(touched(learned[n]) for n in learned if n.startswith(heads))
    # Below the heads the trunk learns from the beliefs; the injections and everything after them do not.
    assert touched(learned["encoder.transformer.layers.1.linear1.weight"])
    after = [n for n in learned if n.startswith(("encoder.belief_in.", "encoder.belief_state_in.",
                                                 "encoder.transformer.layers.2.", "encoder.transformer.layers.3.",
                                                 "encoder.norm.", "discard_head.", "claim_head.", "state_head."))]
    assert after and not any(touched(learned[n]) for n in after)


def test_belief_target_v2_reads_the_rows_fields():
    rows = 50
    counts = torch.randint(0, 5, (rows, 4, 34))
    oracle = torch.zeros(rows, 16, 34)
    for k in range(4):
        oracle[:, k::4] = (counts > k).float()
    # oracle_scalars as guided.encoding.oracle stores them: (8 - shanten) / 9 in float32.
    shanten = torch.arange(-1, 9).repeat(15)[: rows * 3].reshape(rows, 3)
    scalars = torch.cat([((8 - shanten).double() / 9).float(), torch.rand(rows, 1)], 1)
    waits = (torch.rand(rows, 3, 34) < 0.2).float() * (shanten <= 0).float()[..., None]
    target = belief_target_v2(oracle.half(), scalars, waits)
    assert isinstance(target, BeliefV2) and target.tiles.dtype == target.state.dtype == torch.float32
    assert torch.equal(target.tiles[:, :4], counts / 4) and torch.equal(target.tiles[:, :3], belief_target(oracle))
    assert torch.equal(target.tiles[:, 4:], waits)
    assert torch.equal(target.state[:, :3], scalars[:, :3])
    assert torch.equal(target.state[:, 3:], (shanten <= 0).float())


def test_a_v1_belief_widens_to_v2_with_every_logit_unchanged(rows, tmp_path):
    """fp-rl-227's shape (a trained v1 head, shared suits) widened to v2: bit for bit the same decisions."""
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.model_server import Models, decide
    from guided.trial import export_actor

    config = {"dim": 32, "layers": 4, "heads": 4, "lookahead": True, "belief_head": True, "suit_embedding": "shared"}
    torch.manual_seed(227)
    source = actor_of(ENCODING, 32, 4, 4, belief_head=True, suit_embedding="shared")
    encoder = source.encoder
    with torch.no_grad():  # a trained-looking head, injection and suit tables
        for p in (encoder.belief_in.weight, encoder.belief_in.bias, encoder.belief_head[3].weight,
                  encoder.belief_head[3].bias, encoder.tile_embedding.ranks):
            torch.nn.init.normal_(p, std=0.5)
    export_actor(source.state_dict(), config, tmp_path / "v1")
    target = expand(tmp_path / "v1", tmp_path / "models", belief_head="v2")
    narrow, _ = load_actor(tmp_path / "v1", "cpu")
    wide, meta = load_actor(target, "cpu")
    assert meta["config"] == config | {"belief_head": "v2"} and wide.encoder.belief_version == 2
    for part in (rows, rows[:1], rows[1:3], rows[5:12]):
        logits, belief = logits_of(narrow, part, return_belief=True)
        assert belief.abs().sum() > 0
        wide_logits, wide_belief = logits_of(wide, part, return_belief=True)
        assert torch.equal(wide_logits, logits) and torch.equal(logits_of(wide, part), logits)
        assert torch.equal(wide_belief.tiles[:, :3], belief)
        assert not wide_belief.tiles[:, 3:].any() and not wide_belief.state.any()
    old, new = narrow.state_dict(), wide.state_dict()
    for name in ("0.weight", "0.bias", "1.weight", "1.bias"):
        assert torch.equal(new["encoder.belief_head." + name], old["encoder.belief_head." + name])
    assert torch.equal(new["encoder.belief_head.3.weight"][:3], old["encoder.belief_head.3.weight"])
    assert torch.equal(new["encoder.belief_head.3.bias"][:3], old["encoder.belief_head.3.bias"])
    assert torch.equal(new["encoder.belief_in.weight"][:, :3], old["encoder.belief_in.weight"])
    assert torch.equal(new["encoder.belief_in.bias"], old["encoder.belief_in.bias"])
    new_parts = [new["encoder.belief_head.3.weight"][3:], new["encoder.belief_head.3.bias"][3:],
                 new["encoder.belief_in.weight"][:, 3:]] + [
        v for k, v in new.items() if k.startswith(("encoder.belief_state_head.3.", "encoder.belief_state_in."))]
    assert len(new_parts) == 7 and not any(v.any() for v in new_parts)
    assert new["encoder.belief_state_head.1.weight"].abs().sum() > 0  # its hidden layer can learn
    # Served (model_server's loader, the CPU backend) it decides as the source.
    served, _ = Models(target.parent, "cpu").get(target.name)
    batch = policy_batch(rows, "cpu")
    args = [batch[k] for k in ("planes", "scalars", "events", "mask")]
    with torch.inference_mode():
        assert torch.equal(decide(served, *args)[0], decide(narrow, *args)[0])
    assert torch.equal(logits_of(with_oracle_inputs(wide), rows), logits_of(narrow, rows))
    assert with_oracle_inputs(wide).encoder.belief_version == 2
    with pytest.raises(ValueError, match="already has a v2 belief head"):
        expand(target, tmp_path / "again", belief_head="v2")
    with pytest.raises(ValueError, match="already has a v2 belief head"):
        expand(target, tmp_path / "again", belief_head=True)
    with pytest.raises(ValueError, match="already has a belief head"):
        expand(tmp_path / "v1", tmp_path / "again", belief_head=True)


def test_v2_belief_added_to_a_plain_actor_with_a_wider_encoding_and_oracle_inputs(rows, tmp_path):
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.trial import export_actor

    config = {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}
    torch.manual_seed(32)
    source = Actor(32, 2, 4)
    export_actor(source.state_dict(), config, tmp_path / "plain")
    reference = logits_of(source, rows)
    target = expand(tmp_path / "plain", tmp_path / "models", ENCODING_V4, belief_head="v2", oracle_inputs=True)
    start, meta = load_actor(target, "cpu", oracle=True)
    assert meta["config"] == config | {"belief_head": "v2", "oracle_inputs": True}
    wide = [augment(r, ENCODING_V4) for r in rows]
    logits, belief = logits_of(start, wide, return_belief=True)
    assert torch.equal(logits, reference) and not belief.tiles.any() and not belief.state.any()
    assert start.encoder.belief_head[1].weight.abs().sum() > 0


# -- the v2 belief's labels (worker side, training only) ---------------------------------------------


def waits_of(hands, melds=(0, 0, 0, 0), you_cai_bi_kao=False, seat=0):
    from types import SimpleNamespace

    from guided.encoding import opponent_waits
    from mahjong_ai.domain.rules import tile_id
    from mahjong_ai.domain.types import RoundConfig

    state = SimpleNamespace(hands=[[tile_id(c) for c in h.split()] for h in hands],
                            melds=[[object()] * n for n in melds], wall=[],
                            config=RoundConfig(rounds=16, allow_tiebreak=False, you_cai_bi_kao=you_cai_bi_kao))
    waits = opponent_waits(state, seat)
    assert waits.dtype == np.float32 and waits.shape == (3, 34)
    return waits


def tiles_of(row):
    from mahjong_ai.domain.rules import tile_code

    return {tile_code(int(t)) for t in np.flatnonzero(row)}


def test_opponent_waits_are_exactly_the_winning_draws_of_ready_opponents():
    learner = "1w 4w 7w 1b 4b 7b 1t 4t 7t 东 南 西 北"
    waits = waits_of([learner,
                      "1w 2w 3w 4b 5b 6b 7t 8t 9t 东 东 东 9w",   # a single wait
                      "1w 2w 3w 4w 5w 6b 6b 6b 7t 8t 9t 东 东",   # two-sided
                      "2b 3b 4b 5t 5t 5t 7w 8w 南 南"],           # one meld
                     melds=(0, 0, 0, 1))
    # The white tile is wild: drawing it completes every ready hand.
    assert [tiles_of(row) for row in waits] == [{"9w", "白"}, {"3w", "6w", "白"}, {"6w", "9w", "白"}]
    # The learner's own hand is never read: from seat 2 the opponents are 3, 0, 1.
    from_two = waits_of([learner, "1w 2w 3w 4b 5b 6b 7t 8t 9t 东 东 东 9w", learner, learner], seat=2)
    assert [tiles_of(row) for row in from_two] == [set(), set(), {"9w", "白"}]


def test_opponent_waits_are_zero_when_not_ready_and_never_a_fifth_copy():
    far = "1w 4w 7w 1b 4b 7b 1t 4t 7t 东 南 西 北"
    one_away = "1w 2w 3w 4b 5b 6b 7t 8t 东 东 南 西 北"
    fifth = "1w 1w 1w 1w 东 东 东 南 南 南 西 西 西"  # a single wait on a fifth 1w, or on the wild white
    waits = waits_of([far, far, one_away, fifth])
    assert not waits[:2].any() and tiles_of(waits[2]) == {"白"}
    # A drawn tile in hand (14 tiles less 3 per meld) is not a waiting hand.
    assert not waits_of([far, "1w 2w 3w 4b 5b 6b 7t 8t 9t 东 东 东 9w 9w", far, far]).any()


def test_opponent_waits_follow_the_white_tile_rules():
    from mahjong_ai.domain.rules import can_hu, tile_id

    far = "1w 4w 7w 1b 4b 7b 1t 4t 7t 东 南 西 北"
    baotou = "1w 2w 3w 4b 5b 6b 7t 8t 9t 东 东 东 白"   # the white pairs with any draw
    plain = "1w 1w 2b 3b 4t 5t 6t 7t 8t 9t 东 东 白"    # ready with a white, not baotou
    waits = waits_of([far, baotou, plain, far])
    assert waits[0].all()
    hand = sorted(tile_id(c) for c in plain.split())
    brute = {t for t in range(34) if hand.count(t) < 4 and can_hu(tuple(sorted(hand + [t])), 0)}
    assert {tile_id(c) for c in tiles_of(waits[1])} == brute and {"1w", "东", "1b", "4b"} <= tiles_of(waits[1])
    assert len(brute) < 34 and not waits[2].any()
    # 有财必拷响: a hand holding a white wins only on baotou.
    strict = waits_of([far, baotou, plain, far], you_cai_bi_kao=True)
    assert strict[0].all() and not strict[1].any()


def test_belief_features_add_the_waits_to_the_oracle_fields():
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig
    from guided.encoding import belief_features, opponent_waits, oracle

    state = HandState(RoundConfig(rounds=16, allow_tiebreak=False), 20261008)
    features = belief_features(state, 1)
    assert set(features) == {"oracle_planes", "oracle_scalars", "opp_waits"}
    assert all(np.array_equal(features[k], v) for k, v in oracle(state, 1).items())
    assert np.array_equal(features["opp_waits"], opponent_waits(state, 1))


def test_shared_suits_make_every_tile_table_rank_plus_suit(rows):
    torch.manual_seed(9)
    actor = Actor(32, 2, 4, suit_embedding="shared")
    names = dict(actor.named_parameters())
    assert "encoder.tile_embedding" not in names and "encoder.event_embeddings.2.weight" not in names
    for prefix, rest in (("encoder.tile_embedding.", 7), ("encoder.event_embeddings.2.", 8),
                         ("encoder.event_embeddings.3.", 8)):
        assert [tuple(names[prefix + k].shape) for k in ("ranks", "suits", "others")] == [(9, 32), (3, 32), (rest, 32)]
    for table in (actor.encoder.tile_embedding, actor.encoder.event_embeddings[2], actor.encoder.event_embeddings[3]):
        weight = table.weight.detach()
        suited = weight[:27].unflatten(0, (3, 9))
        # Two suits differ by one vector at every rank.
        torch.testing.assert_close(suited - suited[:1], (table.suits - table.suits[:1]).detach()[:, None].expand(3, 9, 32))
        assert torch.equal(weight[27:], table.others.detach())
    batch = policy_batch(rows, "cpu")
    logits = actor(batch["planes"], batch["scalars"], batch["events"])
    logits.masked_fill(~batch["mask"], 0).square().sum().backward()
    assert all(names[k].grad.abs().sum() > 0 for k in names if k.endswith(("ranks", "suits", "others")))


def test_share_suits_starts_from_the_least_squares_fit_of_the_source(rows, tmp_path):
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.trial import export_actor

    config = {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}
    torch.manual_seed(41)
    source = Actor(32, 2, 4)
    export_actor(source.state_dict(), config, tmp_path / "v1")
    target = expand(tmp_path / "v1", tmp_path / "models", suit_embedding="shared")
    shared, meta = load_actor(target, "cpu")
    assert meta["encoding"] == ENCODING and meta["config"]["suit_embedding"] == "shared"
    for name, table in (("encoder.tile_embedding", shared.encoder.tile_embedding),
                        ("encoder.event_embeddings.2.weight", shared.encoder.event_embeddings[2]),
                        ("encoder.event_embeddings.3.weight", shared.encoder.event_embeddings[3])):
        original = source.state_dict()[name]
        suited = original[:27].unflatten(0, (3, 9))
        torch.testing.assert_close(table.ranks.detach(), suited.mean(0))
        fitted = table.weight.detach()
        assert torch.equal(fitted[27:], original[27:])
        # The residual of a least-squares additive fit sums to zero over suits and over ranks.
        residual = (original - fitted)[:27].unflatten(0, (3, 9))
        assert residual.sum(0).abs().max() < 1e-5 and residual.sum(1).abs().max() < 1e-5
    assert logits_of(shared, rows).isfinite().all()
    with pytest.raises(ValueError, match="cannot change suit_embedding"):
        expand(target, tmp_path / "again", suit_embedding="shared")
    # Tables that already are rank plus suit come through unchanged.
    with torch.no_grad():
        for table in (source.encoder.tile_embedding, source.encoder.event_embeddings[2].weight,
                      source.encoder.event_embeddings[3].weight):
            table[:27] = (torch.randn(3, 1, 32) + torch.randn(1, 9, 32)).flatten(0, 1)
    export_actor(source.state_dict(), config, tmp_path / "additive")
    exact, _ = load_actor(expand(tmp_path / "additive", tmp_path / "models", suit_embedding="shared"), "cpu")
    torch.testing.assert_close(logits_of(exact, rows), logits_of(source, rows), atol=1e-5, rtol=1e-5)
    # Every option and a wider encoding at once.
    both = expand(tmp_path / "v1", tmp_path / "all", ENCODING_V4, belief_head=True, suit_embedding="shared")
    _, meta = load_actor(both, "cpu")
    assert meta["config"] == config | {"belief_head": True, "suit_embedding": "shared"}
