"""Encoding v3 extends v2 rows; its additions match a slow reference from the observation, and an extended Actor keeps every decision."""

import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided.encoding import (  # noqa: E402
    ENCODING,
    ENCODING_V2,
    ENCODING_V3,
    EVENT_TAILS,
    INPUTS,
    N_PLANES,
    N_SCALARS,
    V3_PLANES,
    V3_SCALARS,
    augment,
    encode,
    encode_v2,
    encode_v3,
    encoder,
    v3_inputs,
    widest,
)

WHITE = 33


def positions(hands=60, seed=20261007, keep=0.15):
    """Decision states of greedy shanten play (ready hands, kongs and wins come up), every call and kong window kept."""
    from mahjong_ai import _fast
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import RoundConfig
    from mahjong_ai.sdk.shape import remaining_tiles

    def discard(obs, allowed):
        c = list(_fast.counts(tuple(obs.hand)))
        opened = len(obs.melds[obs.stage.seat])
        avail = list(remaining_tiles(obs))
        best, key = None, None
        for t in sorted(allowed):
            c[t] -= 1
            d = _fast.shanten_raw(c, opened)
            ukeire = _fast.analyze_counts(c, opened, avail, 0, 0, 0)[4] if d >= 0 else 99
            c[t] += 1
            k = (d, -ukeire, t == WHITE, t)
            if key is None or k < key:
                best, key = t, k
        return best

    rng, kept = random.Random(seed), []
    for i in range(hands):
        state = HandState(RoundConfig(rounds=16), seed + i, i % 4, 1 + i % 16, [0] * 4, [], False)
        while not state.settled:
            legal = state.legal_actions()
            if len(legal) == 1:
                state.step(legal[0], legal)
                continue
            obs = state.observation(legal)
            kinds = {a.action for a in legal}
            if kinds & {"peng", "chi", "gang", "hu"} or rng.random() < keep:
                kept.append(obs)
            by = {a.action: a for a in legal}
            if "hu" in by and rng.random() < 0.85:
                action = by["hu"]
            elif "gang" in by and rng.random() < 0.5:
                action = next(a for a in legal if a.action == "gang")
            elif kinds & {"peng", "chi"} and rng.random() < 0.3:
                action = rng.choice([a for a in legal if a.action in ("peng", "chi")])
            elif "discard" in by:
                # Now and then a white goes, so catch rounds come up.
                allowed = {a.tile for a in legal if a.action == "discard"}
                tile = WHITE if WHITE in allowed and rng.random() < 0.1 else discard(obs, allowed)
                action = next(a for a in legal if a.action == "discard" and a.tile == tile)
            else:
                action = next(a for a in legal if a.action == "pass")
            state.step(action, legal)
    return kept


@pytest.fixture(scope="module")
def sample():
    return positions()


def reference_v3(obs):
    """The v3 additions recomputed from the observation with the pure-Python shape and rules.

    Everything but the lookahead planes 8-27 and the scalars that read them
    (0-19, 23, 24), which the Monte Carlo study checks instead. Returns
    (planes, scalars, tails, known) with known the plane and scalar indices it fills.
    """
    from mahjong_ai.domain import rules
    from mahjong_ai.sdk import shape

    seat = obs.stage.seat
    hand = list(obs.hand)
    opened = len(obs.melds[seat])
    chain, piao, you_cai = min(obs.chain_count, 8), min(obs.piao_count, 4), bool(obs.you_cai_bi_kao)
    remaining = list(shape.remaining_tiles(obs))
    pool = sum(remaining)
    planes = np.zeros((V3_PLANES, 34), np.float32)
    scalars = np.zeros(V3_SCALARS, np.float32)
    discarding = len(hand) == 14 - 3 * opened

    def counts(tiles):
        c = [0] * 34
        for t in tiles:
            c[t] += 1
        return c

    def plain(c):
        return shape._normal(tuple(c[:WHITE]) + (0,), opened, 0, 0) - c[WHITE]

    def seven(c):
        if opened:
            return 8
        pairs = sum(n // 2 for n in c[:WHITE])
        singles = sum(n % 2 for n in c[:WHITE])
        return 14 - (2 * pairs + min(singles, 7 - pairs)) - c[WHITE] - 1

    saved = rules._fast
    rules._fast = None  # the pure-Python shape, baotou and fan
    try:
        rules._calculate_fan_cached.cache_clear()
        rules._baotou.cache_clear()
        if discarding:
            for t in sorted(set(hand)):
                rest = list(hand)
                rest.remove(t)
                c = counts(rest)
                p, s = plain(c), seven(c)
                pu = pw = su = sw = 0
                for x in range(34):
                    if remaining[x] <= 0 or c[x] >= 4:
                        continue
                    c[x] += 1
                    px, sx = plain(c), seven(c)
                    pu += remaining[x] * (px < p)
                    su += remaining[x] * (not opened and sx < s)
                    if min(px, sx) < 0:
                        fan = rules.calculate_fan(tuple(sorted(rest + [x])), x, opened, chain, piao, you_cai)
                        if fan is not None and fan.fan > 0:
                            pw += remaining[x] * (px < 0)
                            sw += remaining[x] * (not opened and sx < 0)
                    c[x] -= 1
                planes[0:6, t] = ((8 - p) / 9, min(pu, 40) / 40, min(pw, 12) / 12,
                                  (8 - s) / 9, min(su, 40) / 40, min(sw, 12) / 12)
                planes[6, t] = float(WHITE in rest and rules.is_baotou(rest, opened))
                outs = 0
                if min(p, s) <= 1:
                    for x in range(34):
                        if remaining[x] <= 0 or rest.count(x) >= 4 or (WHITE not in rest and x != WHITE):
                            continue
                        drawn = rest + [x]
                        if any(
                            WHITE in (left := [u for j, u in enumerate(drawn) if j != drawn.index(y)])
                            and min(plain(counts(left)), seven(counts(left))) <= 0
                            and rules.is_baotou(left, opened)
                            for y in sorted(set(drawn))
                        ):
                            outs += remaining[x]
                planes[7, t] = min(outs, 40) / 40
                planes[28, t] = (4 - min(4, rest.count(WHITE) + piao)) / 4
    finally:
        rules._fast = saved

    # Call risk.
    nxt = (seat + 1) % 4
    if sum(m.kind == "chi" for m in obs.melds[nxt]) < 2:
        for t in range(27):
            r, total = t % 9, 0
            for a, b in ((-2, -1), (-1, 1), (1, 2)):
                if 0 <= r + a < 9 and 0 <= r + b < 9:
                    total += remaining[t + a] * remaining[t + b]
            planes[29, t] = min(total / 16, 1.0)
    for t in range(WHITE):
        planes[30, t] = max(0, remaining[t] - 1) / 3

    # Kongs.
    pengs = {m.tiles[0] for m in obs.melds[seat] if m.kind == "peng"}
    for a in obs.legal_actions:
        if a.action != "gang":
            continue
        rest = list(hand)
        for _ in range({"exposed": 3, "concealed": 4, "added": 1}[a.kind]):
            rest.remove(a.tile)
        melds = opened + (a.kind != "added")
        r = shape.analyze(tuple(sorted(rest)), melds, tuple(remaining), chain, piao, you_cai)
        again = sum(remaining[t] for t in range(WHITE) if t != a.tile and (rest.count(t) == 3 or t in pengs))
        planes[31, a.tile] = r["winning_copies"] / pool if pool else 0
        planes[32, a.tile] = min(chain + 1, 8) / 8
        planes[33, a.tile] = min(again / pool, 1) if pool else 0

    # This hand's events: tails, rivers, calls, catch rounds.
    events = [e for e in obs.public_events if e.hand_no == obs.stage.round_no]
    rel = [None if e.seat is None else (e.seat - seat) % 4 for e in events]
    discard_index = [i for i, e in enumerate(events) if e.kind == "discard"]
    tails = np.zeros(max(1, len(events)), np.int16)
    owner, previous, last_call, rivers = None, {}, {}, [[] for _ in range(4)]
    for i, e in enumerate(events):
        after = sum(j > i for j in discard_index)
        forced = 0
        if e.kind == "discard":
            if owner is not None and rel[i] != owner:
                forced = 1
            elif previous.get(rel[i]) in ("chi", "peng"):
                forced = 2
            if e.tile == WHITE:
                owner = rel[i]
            elif rel[i] == owner:
                owner = None
            claimed = i + 1 < len(events) and events[i + 1].kind in ("chi", "peng", "gang")
            if not claimed:
                planes[34 + rel[i], e.tile] += math.exp(-0.2 * (after / 4))
                rivers[rel[i]].append(e.tile)
        elif e.kind in ("chi", "peng", "gang"):
            last_call[rel[i]] = after
        previous[rel[i]] = e.kind
        turns = after // 4
        tails[i] = forced * 8 + (turns if turns < 5 else 5 if turns < 7 else 6 if turns < 10 else 7)
    # The replayed rivers and catch round are the observation's own.
    for r in range(4):
        assert sorted(rivers[r]) == sorted(obs.discards[(seat + r) % 4]), "replayed river differs"
    assert (owner is not None) == bool(obs.catch_play)
    if owner is not None:
        assert (seat + owner) % 4 == obs.god_discarder_seat

    # Scalars.
    left = max(0, obs.wall_remaining - 20)
    if discarding:
        draws = left // 4
    else:
        last = rel[discard_index[-1]] if discard_index else 0
        first = (-(last + 1)) % 4 + 1
        draws = 0 if left < first else (left - first) // 4 + 1
    scalars[20] = min(draws, 20) / 20
    if any(a.action == "hu" for a in obs.legal_actions):
        scalars[21] = 1.0
        scalars[22] = math.log2(max(1, obs.hu_fan)) / 9
    if discarding and WHITE in hand:
        s_white = shape.shanten(tuple(sorted(u for j, u in enumerate(hand) if j != hand.index(WHITE))), opened)
        others = [
            shape.shanten(tuple(sorted(u for j, u in enumerate(hand) if j != hand.index(t))), opened)
            for t in set(hand) if t != WHITE
        ]
        if others:
            scalars[25] = float(np.clip((max(0, s_white) - max(0, min(others))) / 2, -1, 1))
    if WHITE in hand:
        rest = list(hand)
        rest.remove(WHITE)
        now = rules.is_baotou(rest, opened)
        if now and obs.drawn_tile is not None:
            before = list(hand)
            before.remove(obs.drawn_tile)
            now = rules.is_baotou(before, opened)
        if now:
            scalars[26:29] = (1.0, min(chain + 1, 8) / 8, min(piao + 1, 4) / 4)
    scalars[29] = min(len(discard_index) / 4 / 20, 1)
    for r in range(4):
        scalars[30 + r] = min(last_call[r] / 4 / 20, 1) if r in last_call else 1.0
    scalars[34] = float(bool(events) and rel[-1] == 0 and events[-1].kind in ("chi", "peng"))
    known_planes = [i for i in range(V3_PLANES) if not 8 <= i < 28]
    known_scalars = [i for i in range(V3_SCALARS) if i >= 20 and i not in (23, 24)]
    return planes, scalars, tails, (known_planes, known_scalars)


def test_v3_rows_extend_v2_rows(sample):
    for obs in sample[:300]:
        v1, v2, v3 = encode(obs), encode_v2(obs), encode_v3(obs)
        assert v3["planes"].shape == (INPUTS[ENCODING_V3][0], 34) and v3["planes"].dtype == np.float16
        assert v3["scalars"].shape == (INPUTS[ENCODING_V3][1],) and v3["scalars"].dtype == np.float16
        assert np.array_equal(v3["planes"][: len(v2["planes"])], v2["planes"])
        assert np.array_equal(v3["scalars"][:N_SCALARS], v1["scalars"])
        assert np.array_equal(v3["events"][:, :4], v1["events"][:, :4]) and not v1["events"][:, 4].any()
        assert np.array_equal(v3["mask"], v1["mask"])
        assert int(v3["events"][:, 4].max()) < EVENT_TAILS[ENCODING_V3]
        # A v2 row extends to the same v3 row; a v3 row is left as it is for any encoding.
        again = augment(v2, ENCODING_V3)
        for key in ("planes", "scalars", "events", "mask"):
            assert np.array_equal(again[key], v3[key])
        assert augment(v3, ENCODING_V3) is v3 and augment(v3, ENCODING) is v3
    assert encoder(ENCODING_V3) is encode_v3
    assert widest([ENCODING, ENCODING_V3, ENCODING_V2]) == ENCODING_V3


def test_v3_additions_match_the_reference(sample):
    covered = {"kong": 0, "forced": 0, "after_call": 0, "baotou": 0, "piao": 0, "hu": 0}
    for obs in sample:
        planes, scalars, tails = v3_inputs(encode(obs))
        expected, expected_scalars, expected_tails, (known_planes, known_scalars) = reference_v3(obs)
        np.testing.assert_allclose(planes[known_planes], expected[known_planes], atol=2e-6)
        np.testing.assert_allclose(scalars[known_scalars], expected_scalars[known_scalars], atol=2e-6)
        assert np.array_equal(tails, expected_tails)
        covered["kong"] += bool(planes[32].any())
        covered["forced"] += bool((tails // 8 == 1).any())
        covered["after_call"] += bool((tails // 8 == 2).any())
        covered["baotou"] += bool(planes[7].any())
        covered["piao"] += bool(scalars[26])
        covered["hu"] += bool(scalars[21])
    # A piao window is rare in a short sample; it is compared whenever one comes up.
    assert all(v for k, v in covered.items() if k != "piao"), covered


def test_lookahead_is_consistent(sample):
    """Probabilities in [0, 1], monotone in k, ready certain once ready, no win before ready."""
    for obs in sample[:400]:
        planes, scalars, _ = v3_inputs(encode(obs))
        look = np.concatenate([planes[8:28].T, scalars[None, :20]])  # per tile, then the reference
        for row in look:
            ready, baotou, win = row[0::4], row[1::4], row[2::4]
            for p in (ready, baotou, win):
                assert ((p >= -1e-6) & (p <= 1 + 1e-6)).all()
                assert (np.diff(p) >= -1e-6).all()
            assert (win <= ready + 1e-6).all()


def test_extended_actor_keeps_every_decision(sample, tmp_path):
    torch = pytest.importorskip("torch")
    from guided.deployment import load_actor
    from guided.expand import expand
    from guided.inference import policy_batch
    from guided.networks import Actor
    from guided.trial import export_actor

    torch.manual_seed(0)
    source = Actor(32, 2, 4)
    with torch.no_grad():
        source.encoder.event_embeddings[4].weight.normal_()  # rows a v1 Actor never uses
    v1_dir = tmp_path / "v1"
    export_actor(source.state_dict(), {"dim": 32, "layers": 2, "heads": 4, "lookahead": True}, v1_dir)
    v2_dir = expand(v1_dir, tmp_path / "models", ENCODING_V2)
    for start in (v1_dir, v2_dir):
        wide_dir = expand(start, tmp_path / ("models-" + start.name), ENCODING_V3)
        narrow, _ = load_actor(start, "cpu")
        wide, meta = load_actor(wide_dir, "cpu")
        assert meta["encoding"] == ENCODING_V3
        rows = [encode(o) for o in sample[:200]]
        narrow_encoding = ENCODING if start == v1_dir else ENCODING_V2
        with torch.no_grad():
            a = policy_batch([augment(r, narrow_encoding) for r in rows], "cpu")
            b = policy_batch([augment(r, ENCODING_V3) for r in rows], "cpu")
            assert torch.equal(narrow(a["planes"], a["scalars"], a["events"]), wide(b["planes"], b["scalars"], b["events"]))
            # The narrow Actor reads a v3 row's prefixes and ignores its event tails.
            assert torch.equal(narrow(a["planes"], a["scalars"], a["events"]), narrow(b["planes"], b["scalars"], b["events"]))


def test_model_service_extends_v1_requests_for_a_v3_model(sample):
    import base64
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from guided.inference import policy_batch
    from guided.model_server import Service, decide
    from guided.networks import actor_of
    from selfplay_core.wire import pack, unpack

    torch.set_num_threads(1)
    torch.manual_seed(78)
    actor = actor_of(ENCODING_V3, 16, 1, 4).eval()
    with torch.no_grad():
        for p in actor.parameters():
            p.normal_()  # every new input column and tail row reaches the logits
    meta = {"config": {"lookahead": True}, "encoding": ENCODING_V3}
    service = Service(SimpleNamespace(device="cpu", get=lambda name: (actor, meta)))
    rows = [encode(o) for o in sample[:40]]
    payloads = [
        {"lookahead": True, "encoded": {k: base64.b64encode(np.ascontiguousarray(v).tobytes()).decode()
                                        for k, v in r.items()}}
        for r in rows
    ]
    batch = policy_batch([augment(r, ENCODING_V3) for r in rows], "cpu")
    with torch.inference_mode():
        expected, valid = decide(actor, *(batch[k] for k in ("planes", "scalars", "events", "mask")))
    assert valid and [a["index"] for a in service.decide("wide", payloads)] == expected.tolist()
    # Match workers' binary requests (selfplay_core.wire) carry v1 rows too.
    binary = [unpack(pack("0" * 64, r, lookahead=True)) for r in rows]
    assert [a["index"] for a in service.decide("wide", binary)] == expected.tolist()
