"""Encoding v2 extends v1 rows exactly; an extended Actor and the model service keep every v1 decision."""

import base64
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided.encoding import (
    ENCODING,
    ENCODING_V2,
    INPUTS,
    N_PLANES,
    augment,
    call_planes,
    encode,
    encode_v2,
    encoder,
)


def observations(hands=80, seed=20261006):
    """Decision states of random play that takes calls often, so open hands and kongs come up."""
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig

    rng, kept = random.Random(seed), []
    for i in range(hands):
        state = HandState(RoundConfig(rounds=16), seed + i, i % 4, 1 + i % 16, [0] * 4, [], False)
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) == 1:
                state.step(legal[0], legal)
                continue
            kinds = {a.action for a in legal}
            if kinds & {"peng", "chi", "gang"} or rng.random() < 0.05:
                kept.append(state.observation(legal))
            calls = [a for a in legal if a.action in ("peng", "chi", "gang")]
            if calls and rng.random() < 0.6:
                action = rng.choice(calls)
            else:
                action = rng.choice([a for a in legal if a.action in ("discard", "pass")] or legal)
            state.step(action, legal)
    return kept


@pytest.fixture(scope="module")
def sample():
    return observations()


def expected_call_planes(obs):
    """The v2 planes from the observation itself, through the Python analysis."""
    from mahjong_ai.sdk.shape import analyze, remaining_tiles

    def best(hand, opened):
        if len(hand) == 13 - 3 * opened:
            return analyze(tuple(sorted(hand)), opened, remaining, chain, piao, you_cai)
        rows = []
        for tile in sorted(set(hand)):
            rest = list(hand)
            rest.remove(tile)
            rows.append(analyze(tuple(sorted(rest)), opened, remaining, chain, piao, you_cai))
        return min(rows, key=lambda r: (r["shanten"], -r["ukeire"], -r["fan_value"]))

    opened = len(obs.melds[obs.stage.seat])
    chain, piao, you_cai = min(obs.chain_count, 8), min(obs.piao_count, 4), int(obs.you_cai_bi_kao)
    remaining = remaining_tiles(obs)
    out, seen = np.zeros((16, 34), np.float32), set()
    for action in obs.legal_actions:
        hand = list(obs.hand)
        if action.action == "peng":
            group, tile, melds, leave = 0, action.tile, opened + 1, [action.tile] * 2
        elif action.action == "chi":
            group, tile, melds, leave = 1, min(action.tiles + (action.tile,)), opened + 1, list(action.tiles)
        elif action.action == "gang":
            group, tile, melds, leave = {
                "exposed": (2, action.tile, opened + 1, [action.tile] * 3),
                "concealed": (3, action.tile, opened + 1, [action.tile] * 4),
                "added": (3, action.tile, opened, [action.tile]),
            }[action.kind]
        else:
            continue
        for t in leave:
            hand.remove(t)
        r = best(hand, melds)
        seen.add(action.kind if action.action == "gang" else action.action)
        out[4 * group : 4 * group + 4, tile] = (
            (8 - r["shanten"]) / 9.0,
            min(r["ukeire"], 40) / 40.0,
            min(r["winning_copies"], 12) / 12.0,
            math.log1p(r["fan_value"]) / 8.0,
        )
    return out, seen


def test_v2_rows_are_v1_rows_and_each_moves_lookahead(sample):
    kinds = set()
    for obs in sample:
        v1, v2 = encode(obs), encode_v2(obs)
        assert v2["planes"].shape == (INPUTS[ENCODING_V2][0], 34) and v2["planes"].dtype == np.float16
        assert np.array_equal(v2["planes"][:N_PLANES], v1["planes"])
        for key in ("scalars", "events", "mask"):
            assert np.array_equal(v2[key], v1[key])
        expected, seen = expected_call_planes(obs)
        kinds |= seen
        assert np.array_equal(call_planes(v1), expected)
        assert np.array_equal(v2["planes"][N_PLANES:], expected.astype(np.float16))
        assert augment(v2, ENCODING_V2) is v2 and augment(v1, ENCODING) is v1
    assert kinds == {"peng", "chi", "exposed", "concealed", "added"}
    assert encoder(ENCODING) is encode and encoder(ENCODING_V2) is encode_v2
    with pytest.raises(ValueError, match="unknown observation encoding"):
        encoder("guided-current-hand-v0")


def batch_logits(actor, rows):
    import torch
    from guided.inference import policy_batch

    batch = policy_batch(rows, "cpu")
    with torch.no_grad():
        return actor(batch["planes"], batch["scalars"], batch["events"]).numpy()


def test_expanded_actor_keeps_every_decision_and_v1_actors_read_the_prefix(sample, tmp_path):
    torch = pytest.importorskip("torch")
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.networks import Actor
    from guided.trial import export_actor

    torch.set_num_threads(1)
    torch.manual_seed(31)
    source = Actor(16, 1, 4)
    config = {"dim": 16, "layers": 1, "heads": 4, "lookahead": True}
    export_actor(source.state_dict(), config, tmp_path / "v1")
    target = expand(tmp_path / "v1", tmp_path / "models", ENCODING_V2)
    narrow, narrow_meta = load_actor(tmp_path / "v1", "cpu")
    wide, wide_meta = load_actor(target, "cpu")
    assert (narrow_meta["encoding"], wide_meta["encoding"]) == (ENCODING, ENCODING_V2)
    assert target.name == wide_meta["weights_sha256"]
    assert wide.encoder.tile_in.in_features == INPUTS[ENCODING_V2][0]
    rows1, rows2 = [encode(o) for o in sample], [encode_v2(o) for o in sample]
    reference = batch_logits(narrow, rows1)
    assert np.array_equal(batch_logits(wide, rows2), reference)
    assert np.array_equal(batch_logits(narrow, rows2), reference)
    with pytest.raises(ValueError, match="does not extend"):
        expand(target, tmp_path / "again", ENCODING_V2)


def test_model_service_extends_v1_requests_for_a_v2_model(sample):
    torch = pytest.importorskip("torch")
    from guided.model_server import Service, decide
    from guided.inference import policy_batch
    from guided.networks import Actor
    from selfplay_core.wire import pack, unpack

    torch.set_num_threads(1)
    torch.manual_seed(77)
    planes, scalars = INPUTS[ENCODING_V2]
    actor = Actor(16, 1, 4, planes=planes, scalars=scalars).eval()
    meta = {"config": {"lookahead": True}, "encoding": ENCODING_V2}
    service = Service(SimpleNamespace(device="cpu", get=lambda name: (actor, meta)))
    rows = [encode(o) for o in sample[:40]]
    payloads = [
        {"lookahead": True, "encoded": {k: base64.b64encode(np.ascontiguousarray(v).tobytes()).decode()
                                        for k, v in r.items()}}
        for r in rows
    ]
    batch = policy_batch([augment(r, ENCODING_V2) for r in rows], "cpu")
    with torch.inference_mode():
        expected, valid = decide(actor, *(batch[k] for k in ("planes", "scalars", "events", "mask")))
    assert valid and [a["index"] for a in service.decide("wide", payloads)] == expected.tolist()
    # Match workers' binary requests (selfplay_core.wire) carry v1 rows too.
    binary = [unpack(pack("0" * 64, r, lookahead=True)) for r in rows]
    assert [a["index"] for a in service.decide("wide", binary)] == expected.tolist()
