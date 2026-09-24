"""Rebuild finished platform games as legal decisions for imitation (ADR-0015).

A platform replay carries every seat's starting hand and every draw, discard,
claim, pass and timeout. Each hand is replayed in the local engine with a wall
arranged to deal exactly those tiles, so every decision point yields the same
legal observation the local engine would give that seat, labelled with the
action the platform player actually took. The replay has to agree with the
engine throughout: each drawn tile, each action's legality, every hand's
settlement and the game's final scores. A game that disagrees anywhere raises
``ReplayMismatch`` and contributes nothing.

Only the observation of the deciding seat is recorded; hidden tiles shape the
replay but never enter a decision's input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.rules import WHITE, tile_id
from mahjong_ai.domain.types import Action, PublicEvent, RoundConfig

RESPONSE_PHASES = ("response_peng", "response_chi")
GANG_KINDS = {"an": "concealed", "bu": "added", "ming": "exposed"}


class ReplayMismatch(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    game_id: str
    hand: int
    seat: int
    user_id: str
    observation: object
    legal: tuple
    action: Action
    timeout: bool  # the platform acted for the player; not the player's choice


def config_for(replay_config=None):
    """Local round configuration matching a platform room configuration."""
    cfg = replay_config or {}
    return RoundConfig(
        rounds=int(cfg.get("Rounds", 8)),
        base_score=int(cfg.get("BaseScore", 1)),
        you_cai_bi_kao=bool(cfg.get("YouCaiBiKao", False)),
        allow_tiebreak=False,
    )


def _scripted_hand(config, block, hand_no, scores, history, views):
    """A hand whose deal and wall reproduce the platform's tiles."""
    dealer = block["dealer"]
    start = [[tile_id(t) for t in hand] for hand in block["start_hands"]]
    if sorted(len(h) for h in start) != [13, 13, 13, 14] or len(start[dealer]) != 14:
        raise ReplayMismatch("starting hands do not match the dealer")
    normal, kong = [], []
    for e in block["events"]:
        if e["type"] == "tile_drawn":
            (kong if (e.get("data") or {}).get("gang_replenish") else normal).append(
                tile_id(e["tile"])
            )
    remaining = Counter({t: 4 for t in range(34)})
    remaining.subtract(t for h in start for t in h)
    remaining.subtract(normal + kong)
    if min(remaining.values()) < 0:
        raise ReplayMismatch("replay deals more copies of a tile than exist")
    filler = [t for t, n in sorted(remaining.items()) for _ in range(n)]
    state = HandState(config, 0, dealer, hand_no, scores, history, False, previous_views=views)
    state.rounds_total = config.rounds
    dealt = [h[:13] for h in start]
    # The engine pops ordinary draws from the end and kong replacements from
    # the front; the dealer's fourteenth tile is its opening draw.
    state.wall = kong + filler + normal[::-1]
    state.hands = [list(h) for h in dealt]
    state.initial_hands = [list(h) for h in dealt]
    for seat in range(4):
        state.events[state.hand_start + seat] = (
            PublicEvent("deal", seat, None, tuple(dealt[seat]), hand_no, ()),
            seat,
        )
    state.drawn = start[dealer][13]
    state.hands[dealer].append(state.drawn)
    state.white_acquired = [h.count(WHITE) for h in state.hands]
    state.turn, state.phase, state.kong_draw = dealer, "draw", False
    return state


def _hands(replay):
    """One block per hand: the platform pages long hands into several blocks."""
    hands = {}
    for block in sorted(replay["blocks"], key=lambda b: b["seq_start"]):
        if block.get("truncated"):
            raise ReplayMismatch("hand events are truncated")
        hand = hands.setdefault(block["round_no"], dict(block, events=[]))
        # Only a hand's first page carries the starting hands.
        if hand["dealer"] != block["dealer"] or (
            any(block.get("start_hands") or ()) and hand["start_hands"] != block["start_hands"]
        ):
            raise ReplayMismatch("pages of one hand disagree")
        hand["events"].extend(block["events"])
    if sorted(hands) != list(range(1, len(hands) + 1)):
        raise ReplayMismatch("hands are missing")
    for hand in hands.values():
        seqs = [e["seq"] for e in hand["events"]]
        if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
            raise ReplayMismatch("hand events have gaps")
    return [hands[n] for n in sorted(hands)]


def _discard_timeouts(events):
    """Indices of discards the platform made for a player whose time ran out."""
    marked, last = set(), {}
    for i, e in enumerate(events):
        if e["type"] == "tile_discarded":
            last[e["seat"]] = i
        elif e["type"] == "timeout" and (e.get("data") or {}).get("kind") == "discard":
            if e["seat"] in last:
                marked.add(last[e["seat"]])
    return marked


def _claim(e):
    """The engine action for a claiming platform event, or None."""
    kind = e["type"]
    tile = tile_id(e["tile"]) if e.get("tile") else None
    if kind == "peng":
        return Action("peng", tile)
    if kind == "chi":
        used = list(tile_id(t) for t in e["data"]["tiles"])
        used.remove(tile)
        return Action("chi", tile, tuple(sorted(used)))
    if kind == "gang" and e["data"]["kind"] == "ming":
        return Action("gang", tile, kind="exposed")
    return None


def decisions(replay, replay_config=None):
    """Every multi-choice decision of a finished platform game, in order."""
    if replay.get("status") != "finished":
        raise ReplayMismatch("game is not finished")
    config = config_for(replay_config)
    users = [s["user_id"] for s in replay["seats"]]
    scores, history, views = [0, 0, 0, 0], [], None
    out = []
    for block in _hands(replay):
        hand_no = block["round_no"]
        state = _scripted_hand(config, block, hand_no, scores, history, views)
        events = block["events"]
        timed_out = _discard_timeouts(events)
        responses = {}  # seat -> timed-out windows since the last discard

        def record(action, timeout):
            legal = state.legal_actions()
            if action not in legal:
                raise ReplayMismatch(
                    f"hand {hand_no}: {action} is not legal for seat {state.turn} in {state.phase}"
                )
            if len(legal) > 1:
                seat = state.turn
                out.append(
                    Decision(
                        replay["game_id"], hand_no, seat, users[seat],
                        state.observation(), legal, action, timeout,
                    )
                )
            state.step(action)

        def resolve(claimant=None, claim=None):
            """Close the open response window up to the event that ended it."""
            while state.phase in RESPONSE_PHASES:
                window = "peng" if state.phase == "response_peng" else "chi"
                if claim is not None and state.turn == claimant and (
                    (claim.action == "chi") == (window == "chi")
                ):
                    record(claim, False)
                    return True
                record(Action("pass"), window in responses.get(state.turn, ()))
            return False

        for i, e in enumerate(events):
            kind, seat = e["type"], e["seat"]
            if kind == "pass":
                continue
            if kind == "timeout":
                data = e.get("data") or {}
                if data.get("kind") == "response":
                    responses.setdefault(seat, set()).add(data.get("window"))
                continue
            claim = _claim(e)
            if claim is not None:
                if not resolve(seat, claim):
                    raise ReplayMismatch(f"hand {hand_no}: claim by seat {seat} had no window")
                continue
            resolve()
            if state.settled and kind not in ("round_ended", "game_ended"):
                raise ReplayMismatch(f"hand {hand_no}: events continue after settlement")
            if kind == "tile_drawn":
                if state.phase != "draw" or state.turn != seat or state.drawn != tile_id(e["tile"]):
                    raise ReplayMismatch(f"hand {hand_no}: draw disagrees at seq {e['seq']}")
            elif kind == "tile_discarded":
                if state.turn != seat or state.phase != "draw":
                    raise ReplayMismatch(f"hand {hand_no}: discard out of turn at seq {e['seq']}")
                responses = {}
                record(Action("discard", tile_id(e["tile"])), i in timed_out)
            elif kind == "gang":
                if state.turn != seat or state.phase != "draw":
                    raise ReplayMismatch(f"hand {hand_no}: kong out of turn at seq {e['seq']}")
                record(Action("gang", tile_id(e["tile"]), kind=GANG_KINDS[e["data"]["kind"]]), False)
            elif kind == "round_ended":
                data = e["data"]
                if not data.get("draw"):
                    if state.settled or state.turn != seat or state.phase != "draw":
                        raise ReplayMismatch(f"hand {hand_no}: win out of turn")
                    record(Action("hu"), False)
                if not state.settled:
                    raise ReplayMismatch(f"hand {hand_no}: the local hand is still open")
                if tuple(state.delta) != tuple(data["scores"]):
                    raise ReplayMismatch(
                        f"hand {hand_no}: settlement {state.delta} != platform {data['scores']}"
                    )
            elif kind != "game_ended":
                raise ReplayMismatch(f"unknown event {kind}")
        if not state.settled:
            raise ReplayMismatch(f"hand {hand_no}: no settlement")
        state.assert_conservation()
        scores = [s + d for s, d in zip(scores, state.delta)]
        history, views = state.events, state.public_views()
    final = [sum(r["scores"][i] for r in replay.get("rounds", [])) for i in range(4)]
    if replay.get("rounds") and final != scores:
        raise ReplayMismatch(f"final scores {scores} != platform {final}")
    return out
