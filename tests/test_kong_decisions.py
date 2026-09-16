"""Synthetic v32/v33 rule contracts, not captured official game snapshots."""

from collections import Counter

import pytest

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.rules import WHITE, is_baotou, tile_code, tile_id
from mahjong_ai.domain.types import Action, Meld, RoundConfig
from mahjong_ai.platform.official import OfficialAdapter


KINDS = {"concealed": "an", "added": "bu", "exposed": "ming"}


def kong_state(kind, draw="白", baotou=True):
    waiting = (
        "东 东 东 南 南 南 西 西 西 白" if baotou else "东 东 东 南 南 南 西 西 北 北"
    )
    target = tile_id("1w")
    state = HandState(RoundConfig(), 1)
    own = [target] * {"concealed": 4, "added": 1, "exposed": 3}[kind]
    own += [tile_id(tile) for tile in waiting.split()]
    state.melds = [[Meld("peng", (target,) * 3)] if kind == "added" else [], [], [], []]
    state.discards = [[], [], [], [target] if kind == "exposed" else []]
    replacements = [tile_id(draw), WHITE]
    used = Counter(own + replacements)
    used.update(tile for row in state.melds for meld in row for tile in meld.tiles)
    used.update(tile for row in state.discards for tile in row)
    assert max(used.values()) <= 4
    deck = list((Counter({tile: 4 for tile in range(34)}) - used).elements())
    state.hands = [own, deck[:13], deck[13:26], deck[26:39]]
    state.wall = replacements + deck[39:]
    state.events = []
    state.initial_hands = [list(hand) for hand in state.hands]
    state.white_acquired = [hand.count(WHITE) for hand in state.hands]
    state.drawn = tile_id("西") if kind != "exposed" else None
    state.phase = "response_peng" if kind == "exposed" else "draw"
    state.discarder = 3 if kind == "exposed" else None
    state.discarded = target if kind == "exposed" else None
    state.assert_conservation()
    return state, Action("gang", target, kind=kind)


def official_document(state, first_kind):
    observation = state.observation()
    events = []
    for index, (event, _) in enumerate(state.events, 1):
        kind = first_kind if event.tile == tile_id("1w") else "concealed"
        events.append(
            {
                "seq": index,
                "type": {"draw": "tile_drawn", "discard": "tile_discarded"}.get(
                    event.kind, event.kind
                ),
                "seat": event.seat,
                "tile": tile_code(event.tile) if event.tile is not None else "",
                "_round_no": state.hand_no,
                "data": {"kind": KINDS[kind]}
                if event.kind == "gang"
                else {"gang_replenish": True},
            }
        )
    return {
        "snapshot": {
            "seat": state.turn,
            "turn": state.turn,
            "waited_seat": state.turn,
            "phase": state.phase,
            "responding_seats": [],
            "round_no": state.hand_no,
            "dealer": state.dealer,
            "my_hand": [tile_code(tile) for tile in observation.hand],
            "drawn_tile": tile_code(state.drawn),
            "melds": [
                [
                    {
                        "kind": "gang_"
                        + KINDS[
                            first_kind
                            if meld.tiles[0] == tile_id("1w")
                            else "concealed"
                        ],
                        "tiles": [tile_code(tile) for tile in meld.tiles],
                        "concealed": meld.concealed,
                    }
                    for meld in row
                ]
                for row in state.melds
            ],
            "discards": [[tile_code(tile) for tile in row] for row in state.discards],
            "hand_counts": [len(hand) for hand in state.hands],
            "scores": list(state.scores),
            "wall_remaining": len(state.wall),
            "last_discard": "0w",
            "god": {
                "baotou": state.current_fan().baotou,
                "chain_count": state.chain[state.turn],
                "catch_play": False,
                "god_discarder_seat": -1,
            },
        },
        "wire_events": events,
        "tournament": {
            "config": {
                "Kind": "test",
                "Rounds": 8,
                "BaseScore": 1,
                "YouCaiBiKao": False,
            }
        },
    }


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("baotou", [True, False])
def test_v32_kong_fan_uses_post_kong_waiting_hand(kind, baotou):
    state, action = kong_state(kind, "白" if baotou else "西", baotou)
    before_draw = list(state.hands[0])
    if state.drawn is not None:
        before_draw.remove(state.drawn)
    assert is_baotou(before_draw, len(state.melds[0])) is (baotou and kind == "exposed")
    state.step(action)
    state.assert_conservation()
    assert state.current_fan().baotou is baotou
    assert state.current_fan().fan == (4 if baotou else 2)
    observation = OfficialAdapter()(official_document(state, kind))
    assert observation.legal_actions == state.legal_actions()
    assert observation.hu_fan == state.current_fan().fan


@pytest.mark.parametrize("kind", KINDS)
def test_v33_kong_win_waits_for_decision_and_timeout_can_hu(kind):
    state, action = kong_state(kind)
    state.step(action)
    assert state.phase == "draw" and state.turn == 0 and not state.settled
    assert state.drawn == WHITE and state.kong_draw
    assert Action("hu") in state.legal_actions()
    assert Action("discard", WHITE) in state.legal_actions()
    assert state.timeout_action() == Action("hu")
    state.step(state.timeout_action())
    assert state.settled and state.winner == 0
    assert state.delta == (96, -32, -32, -32)
    state.assert_conservation()


@pytest.mark.parametrize("kind", KINDS)
def test_v33_declining_kong_win_can_continue_piao(kind):
    state, action = kong_state(kind)
    state.step(action)
    state.step(Action("discard", WHITE))
    assert not state.settled
    assert state.chain[0] == 2 and state.piao[0] == 1
    assert state.catch_owner == 0
    state.assert_conservation()


@pytest.mark.parametrize("kind", KINDS)
def test_v33_declining_kong_win_can_kong_again(kind):
    state, action = kong_state(kind, "东")
    state.step(action)
    next_kong = Action("gang", tile_id("东"), kind="concealed")
    observation = OfficialAdapter()(official_document(state, kind))
    assert (
        Action("hu") in observation.legal_actions
        and next_kong in observation.legal_actions
    )
    state.step(next_kong)
    assert not state.settled and state.chain[0] == 2
    assert state.current_fan().fan == 8
    observation = OfficialAdapter()(official_document(state, kind))
    assert observation.hu_fan == 8 and Action("hu") in observation.legal_actions
    state.step(Action("discard", WHITE))
    assert state.chain[0] == 3 and state.piao[0] == 1
    state.assert_conservation()
