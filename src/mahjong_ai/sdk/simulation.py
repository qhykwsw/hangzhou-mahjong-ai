"""Hypothetical worlds generated solely from a legal observation and private RNG."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import random
from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.domain.rules import WHITE


def sample_world(obs, seed):
    rng = random.Random(seed)
    seat = obs.stage.seat
    known = Counter(obs.hand)
    for row in obs.discards:
        known.update(row)
    for row in obs.melds:
        for m in row:
            known.update(m.tiles)
    if any(n > 4 for n in known.values()):
        raise ValueError("inconsistent visible tile counts")
    remaining = [t for t in range(34) for _ in range(4 - known[t])]
    melds = [list(row) for row in obs.melds]
    for s, row in enumerate(melds):
        for i, m in enumerate(row):
            if m.concealed and not m.tiles:
                choices = [t for t in range(33) if remaining.count(t) == 4]
                if not choices:
                    raise ValueError("concealed kong has no consistent assignment")
                tile = rng.choice(choices)
                for _ in range(4):
                    remaining.remove(tile)
                row[i] = replace(m, tiles=(tile,) * 4)
    rng.shuffle(remaining)
    hands = []
    for s, n in enumerate(obs.hand_sizes):
        if s == seat:
            hands.append(list(obs.hand))
        else:
            hands.append(remaining[:n])
            del remaining[:n]
    if len(remaining) != obs.wall_remaining:
        raise ValueError("public observation violates tile conservation")
    state = object.__new__(HandState)
    state.config = RoundConfig(
        rounds=obs.stage.rounds_total,
        stage=obs.stage.kind,
        you_cai_bi_kao=obs.you_cai_bi_kao,
        base_score=obs.base_score,
    )
    state.dealer = obs.stage.dealer
    state.hand_no = obs.stage.round_no
    state.rounds_total = obs.stage.rounds_total
    state.scores = tuple(obs.stage.effective_scores)
    state.in_tiebreak = obs.stage.in_tiebreak
    state.wall = remaining
    state.hands = hands
    state.initial_hands = deepcopy(hands)
    state.melds = melds
    state.discards = [list(r) for r in obs.discards]
    # Existing private draw/deal payloads belong to the observer, never to opponents.
    state.events = [
        (
            e,
            e.seat if e.kind in ("draw", "deal") or e.concealed else None,
        )
        for e in obs.public_events
    ]
    state.hand_start = next(
        (i for i, (e, _) in enumerate(state.events) if e.hand_no == state.hand_no),
        len(state.events),
    )
    state.phase = obs.phase
    state.turn = seat
    state.drawn = obs.drawn_tile
    # Public hands are sorted. A known draw is the last inserted tile used by
    # the official timeout fallback; sorting must not turn it into another tile.
    if state.drawn is not None:
        state.hands[seat].remove(state.drawn)
        state.hands[seat].append(state.drawn)
    state.kong_draw = False
    state.chain = [0] * 4
    state.piao = [0] * 4
    state.chain[seat] = obs.chain_count
    state.piao[seat] = obs.piao_count
    state.catch_owner = obs.god_discarder_seat if obs.catch_play else None
    current = [e for e in obs.public_events if e.hand_no == state.hand_no]
    discard = next((e for e in reversed(current) if e.kind == "discard"), None)
    state.discarder = discard.seat if discard else None
    state.discarded = discard.tile if discard else None
    state.response_queue = []
    if obs.phase == "response_peng":
        if discard is None:
            raise ValueError("response requires a public discard")
        order = (
            [(discard.seat + i) % 4 for i in range(1, 4)]
            if state.catch_owner is None
            else [state.catch_owner]
        )
        state.response_queue = order[order.index(seat) + 1 :]
    if obs.phase == "draw" and obs.drawn_tile is not None:
        # A private kong is still a public event with redacted tile payload.
        before = [e for e in current if e.kind != "draw"]
        state.kong_draw = bool(
            before and before[-1].kind == "gang" and before[-1].seat == seat
        )
    state.white_acquired = [h.count(WHITE) for h in hands]
    state.winner = None
    state.fan = None
    state.delta = (0, 0, 0, 0)
    state.settled = False
    state.steps = 0
    state.assert_conservation()
    if set(state.legal_actions()) != set(obs.legal_actions):
        raise ValueError("sample does not reproduce legal actions")
    return state


def clone(world):
    return deepcopy(world)


def terminal_value(world, seat):
    scores = [s + d for s, d in zip(world.scores, world.delta)]
    return sum(
        1.0 if scores[seat] > v else 0.5 if scores[seat] == v else 0.0
        for i, v in enumerate(scores)
        if i != seat
    )
