"""Synthetic v26 contract scenarios, separate from captured v25 platform evidence."""

from collections import Counter
from dataclasses import asdict

import pytest

from mahjong_ai.execution.candidate_worker import observation as decode_observation
from mahjong_ai.domain.engine import HandState, IllegalAction
from mahjong_ai.platform.official import OfficialAdapter
from mahjong_ai.platform.online import SnapshotError
from mahjong_ai.strategies.common import action_summary
from mahjong_ai.platform.replay_audit import audit_replay
from mahjong_ai.domain.rules import WHITE, tile_code, tile_id
from mahjong_ai.domain.types import Action, Meld, RoundConfig


def hand(text):
    return [tile_id(t) for t in text.split()]


def circle_state(claim="peng", discarder=1):
    own = hand("1w 1w 2w 2w 2w 3w 3w 3w 4w 4w 4w 白 白 白")
    target = tile_id("1w")
    if claim == "none":
        target = tile_id("东")
    if claim == "chi":
        own = hand("1w 2w 4w 4w 4w 5w 5w 5w 6w 6w 6w 白 白 白")
        target = tile_id("3w")
    # Reserve scripted draws before distributing all remaining physical tiles.
    draws = [tile_id("9t")] * (discarder - 1) + [target]
    deck = list(
        (Counter({t: 4 for t in range(34)}) - Counter(own) - Counter(draws)).elements()
    )
    state = HandState(RoundConfig(), 1)
    state.hands = [own, deck[:13], deck[13:26], deck[26:39]]
    state.wall = deck[39:][::-1] + draws[::-1]
    state.drawn = WHITE
    state.events = []
    state.initial_hands = [list(h) for h in state.hands]
    state.step(Action("discard", WHITE))
    for _ in range(discarder - 1):
        state.step(Action("discard", state.drawn))
        while state.phase.startswith("response_"):
            state.step(Action("pass"))
    state.step(Action("discard", target))
    state.assert_conservation()
    return state, target


def official_document(state):
    obs = state.observation()
    events = []
    names = {"draw": "tile_drawn", "discard": "tile_discarded"}
    for i, (event, private) in enumerate(state.events, 1):
        events.append(
            {
                "seq": i,
                "type": names.get(event.kind, event.kind),
                "seat": event.seat,
                "tile": tile_code(event.tile)
                if event.tile is not None and private in (None, state.turn)
                else "",
                "_round_no": state.hand_no,
                "data": {"tiles": [tile_code(t) for t in event.tiles]},
            }
        )
    snap = {
        "seat": state.turn,
        "phase": state.phase,
        "turn": state.turn,
        "responding_seats": [state.turn],
        "round_no": state.hand_no,
        "dealer": state.dealer,
        "my_hand": [tile_code(t) for t in obs.hand],
        "drawn_tile": tile_code(obs.drawn_tile) if obs.drawn_tile is not None else "",
        "melds": [
            [
                {
                    "kind": m.kind,
                    "tiles": [tile_code(t) for t in m.tiles],
                    "concealed": m.concealed,
                }
                for m in row
            ]
            for row in obs.melds
        ],
        "discards": [[tile_code(t) for t in row] for row in obs.discards],
        "hand_counts": list(obs.hand_sizes),
        "scores": [0] * 4,
        "wall_remaining": len(state.wall),
        "last_discard": tile_code(state.discarded)
        if state.discarded is not None
        else "0w",
        "god": {
            "catch_play": obs.catch_play,
            "god_discarder_seat": obs.god_discarder_seat,
            "chain_count": obs.chain_count,
        },
    }
    return {
        "snapshot": snap,
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


@pytest.mark.parametrize("discarder", [1, 2, 3])
@pytest.mark.parametrize("can_claim", [True, False])
def test_circle_only_opens_windows_for_owner_even_without_claim(discarder, can_claim):
    state, target = circle_state("peng" if can_claim else "none", discarder)
    assert state.phase == "response_peng" and state.turn == 0
    assert state.response_queue == [] and state.catch_owner == 0
    assert (Action("peng", target) in state.legal_actions()) is can_claim
    state.step(Action("pass"))
    if discarder == 3:
        assert state.phase == "response_chi" and state.turn == 0
        if not can_claim:
            assert state.legal_actions() == (Action("pass"),)
        state.step(Action("pass"))
    assert state.phase == "draw" and state.turn == (discarder + 1) % 4
    assert state.catch_owner == (None if discarder == 3 else 0)


@pytest.mark.parametrize("claim,discarder", [("peng", 1), ("chi", 3)])
@pytest.mark.parametrize("continue_piao", [True, False])
def test_exempt_call_preserves_chain_until_discard(claim, discarder, continue_piao):
    state, target = circle_state(claim, discarder)
    if claim == "chi":
        state.step(Action("pass"))
    action = Action(
        claim, target, (tile_id("1w"), tile_id("2w")) if claim == "chi" else ()
    )
    assert action in OfficialAdapter()(official_document(state)).legal_actions
    state.step(action)
    assert state.chain[0] == state.piao[0] == 1
    assert state.drawn is None and state.catch_owner == 0
    local = state.observation()
    remote = OfficialAdapter()(official_document(state))
    assert local.legal_actions == remote.legal_actions
    assert (
        remote.catch_play
        and remote.god_discarder_seat == 0
        and not remote.catch_restricted
    )
    assert remote.piao_count == 1 and Action("hu") not in remote.legal_actions
    assert action_summary(remote, Action("discard", WHITE))[2]
    assert decode_observation(asdict(remote)).god_discarder_seat == 0
    state.step(Action("discard", WHITE if continue_piao else tile_id("4w")))
    if continue_piao:
        assert state.chain[0] == state.piao[0] == 2
        assert state.catch_owner == 0 and state.phase == "draw" and state.turn == 1
        # Chain recovery must include the piao on both sides of the call.
        state.step(Action("discard", state.drawn))
        assert state.turn == 0 and state.phase == "response_peng"
        adapted = OfficialAdapter()(official_document(state))
        assert adapted.chain_count == adapted.piao_count == 2
    else:
        assert state.chain[0] == state.piao[0] == 0
        assert state.catch_owner is None and state.phase == "response_peng"
        assert state.turn == 1 and state.response_queue == [2, 3]
    state.assert_conservation()


@pytest.mark.parametrize("seat,restricted", [(0, False), (1, True)])
def test_exemption_controls_discard_added_kong_and_response(seat, restricted):
    state = HandState(RoundConfig(), 1)
    state.catch_owner = 0
    state.turn = seat
    state.hands[seat] = hand("1w 1w 1w 1w 2w 3w 4w 5w 6w 东 东")
    state.melds[seat] = [Meld("peng", (tile_id("2w"),) * 3)]
    state.drawn = tile_id("东")
    obs = state.observation()
    assert obs.catch_restricted is restricted and obs.god_discarder_seat == 0
    assert (Action("discard", tile_id("3w")) in obs.legal_actions) is not restricted
    assert (
        Action("gang", tile_id("2w"), kind="added") in obs.legal_actions
    ) is not restricted
    assert Action("gang", tile_id("1w"), kind="concealed") in obs.legal_actions
    assert (
        OfficialAdapter()(official_document(state)).legal_actions == obs.legal_actions
    )
    state.phase = "response_peng"
    state.discarded = tile_id("1w")
    state.hands[seat].remove(tile_id("1w"))
    legal = state.legal_actions()
    assert (Action("gang", tile_id("1w"), kind="exposed") in legal) is not restricted
    assert (Action("peng", tile_id("1w")) in legal) is not restricted
    if restricted:
        with pytest.raises(IllegalAction):
            state.step(Action("peng", tile_id("1w")))
    assert OfficialAdapter()(official_document(state)).legal_actions == legal


def test_restricted_player_can_still_self_draw_hu():
    state = HandState(RoundConfig(), 1)
    state.catch_owner = 1
    state.hands[0] = hand("1w 1w 1w 2w 2w 2w 3w 3w 3w 4w 4w 4w 东 白")
    state.drawn = tile_id("东")
    assert state.observation().catch_restricted
    assert Action("hu") in state.legal_actions()


def test_exposed_kong_preserves_circle_during_replenishment():
    state, target = circle_state()
    state.hands[0].append(target)  # isolated action scenario, no conservation claim
    state.step(Action("gang", target, kind="exposed"))
    assert state.turn == 0 and state.kong_draw and state.catch_owner == 0
    assert state.chain[0] == 2 and state.piao[0] == 1
    assert not state.observation().catch_restricted
    state.step(
        next(
            a
            for a in state.legal_actions()
            if a.action == "discard" and a.tile != WHITE
        )
    )
    assert state.catch_owner is None and state.chain[0] == state.piao[0] == 0


@pytest.mark.parametrize("owner", [None, -1, 4, True, "0"])
def test_active_circle_rejects_missing_or_invalid_owner(owner):
    state, _ = circle_state()
    document = official_document(state)
    if owner is None:
        del document["snapshot"]["god"]["god_discarder_seat"]
    else:
        document["snapshot"]["god"]["god_discarder_seat"] = owner
    with pytest.raises(SnapshotError, match="god_discarder_seat"):
        OfficialAdapter()(document)


def test_replay_audit_preserves_piao_across_exempt_peng():
    state, target = circle_state()
    state.step(Action("peng", target))
    state.step(Action("discard", WHITE))
    # Let the circle finish, then self-draw with the accumulated two-piao chain.
    while state.turn != 0 or state.phase != "draw":
        state.step(
            Action("pass")
            if state.phase.startswith("response_")
            else Action("discard", state.drawn)
        )
    state.step(Action("hu"))
    events = []
    names = {
        "draw": "tile_drawn",
        "discard": "tile_discarded",
        "hand_ended": "round_ended",
    }
    for i, (event, _) in enumerate(state.events, 1):
        data = {"tiles": [tile_code(t) for t in event.tiles]}
        if event.kind == "hand_ended":
            data.update(draw=False, dealer=0, fan=state.fan.fan)
        events.append(
            {
                "seq": i,
                "type": names.get(event.kind, event.kind),
                "seat": event.seat,
                "tile": tile_code(event.tile) if event.tile is not None else "",
                "data": data,
            }
        )
    replay = {
        "blocks": [
            {
                "round_no": 1,
                "seq_start": 1,
                "start_hands": [[tile_code(t) for t in h] for h in state.initial_hands],
                "events": events,
            }
        ]
    }
    result = audit_replay(replay)
    assert result["status"] == "passed", result["failures"]
    assert state.chain[0] == state.piao[0] == 2
