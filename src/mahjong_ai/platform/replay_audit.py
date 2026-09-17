"""Replay completed official games through the real local state machine.

Recorded draws constrain a hypothetical wall; unobserved tiles fill its middle.
Sequential response passes only advance the local queue, not certify timing.
Replay data is post-game audit evidence and never a live policy observation.
"""

from collections import Counter

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.rules import tile_id
from mahjong_ai.domain.types import Action, RoundConfig


def _initial_state(blocks, events, config):
    hands = [[tile_id(t) for t in h] for h in blocks[0]["start_hands"]]
    if len(hands) != 4 or sorted(map(len, hands)) != [13, 13, 13, 14]:
        raise ValueError("four initial hands with one 14-tile dealer required")
    dealer = next(s for s, hand in enumerate(hands) if len(hand) == 14)
    available = Counter({t: 4 for t in range(34)})
    available.subtract(t for hand in hands for t in hand)
    normal, supplement = [], []
    kong_pending = False
    for event in events:
        if event["type"] == "gang":
            kong_pending = True
        elif event["type"] == "tile_drawn":
            tile = tile_id(event["tile"])
            (supplement if kong_pending else normal).append(tile)
            available[tile] -= 1
            kong_pending = False
    if min(available.values()) < 0:
        raise ValueError("recorded initial hands and draws exceed four copies")
    state = HandState(config, 0, dealer=dealer, hand_no=blocks[0]["round_no"])
    state.hands = [list(h) for h in hands]
    state.initial_hands = [list(h) for h in hands]
    state.drawn = hands[dealer][-1]
    state.wall = supplement + list(available.elements()) + normal[::-1]
    state.white_acquired = [h.count(33) for h in hands]
    state.events = []
    state.assert_conservation()
    return state


def _action(event):
    kind, data = event["type"], event.get("data") or {}
    tile = tile_id(event["tile"])
    if kind == "tile_discarded":
        return Action("discard", tile)
    if kind == "chi":
        used = [tile_id(t) for t in data["tiles"]]
        used.remove(tile)
        return Action("chi", tile, tuple(sorted(used)))
    if kind == "gang":
        return Action(
            "gang",
            tile,
            kind={"an": "concealed", "bu": "added", "ming": "exposed"}[data["kind"]],
        )
    return Action("peng", tile)


def _discard_timeout_follows(events, index):
    """White discards may auto-draw before the originating timeout is logged."""
    event = events[index]
    offset = index + 1
    while offset < len(events) and events[offset]["type"] in (
        "tile_drawn",
        "game_ended",
    ):
        offset += 1
    if offset == len(events):
        return False
    timeout = events[offset]
    return (
        timeout["type"] == "timeout"
        and timeout.get("seat") == event.get("seat")
        and (timeout.get("data") or {}).get("kind") == "discard"
    )


def audit_replay(replay, base_score=1, you_cai_bi_kao=False):
    groups = {}
    event_types = Counter()
    for block in replay["blocks"]:
        groups.setdefault(block["round_no"], []).append(block)
        event_types.update(e["type"] for e in block.get("events", []))
    failures, missing, outcomes = [], [], []
    checks = 0
    timeout_checks = 0

    def check(condition, message, number, seq=None):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(
                {"ok": False, "check": message, "round_no": number, "seq": seq}
            )
        return condition

    config = RoundConfig(base_score=base_score, you_cai_bi_kao=you_cai_bi_kao)
    previous = None
    cumulative = [0] * 4
    numbers = sorted(groups)
    complete_prefix = bool(numbers) and numbers == list(range(1, numbers[-1] + 1))
    for number in numbers:
        blocks = sorted(groups[number], key=lambda b: b["seq_start"])
        events = sorted(
            [e for b in blocks for e in b.get("events", [])], key=lambda e: e["seq"]
        )
        if not blocks[0].get("start_hands"):
            missing.append({"round_no": number, "reason": "missing initial hands"})
            complete_prefix = False
            continue
        try:
            state = _initial_state(blocks, events, config)
        except (ValueError, KeyError, AssertionError, TypeError) as exc:
            check(False, str(exc), number)
            complete_prefix = False
            continue
        if previous is not None and previous[0] == number - 1:
            check(
                state.dealer == previous[1],
                "dealer transition matches local convention",
                number,
            )
        observed_draws = 0
        ended = False
        for index, event in enumerate(events):
            seq, kind, seat = event["seq"], event["type"], event.get("seat")
            data = event.get("data") or {}
            if index:
                check(
                    seq == events[index - 1]["seq"] + 1,
                    "event sequence contiguous",
                    number,
                    seq,
                )
            try:
                if kind == "tile_drawn":
                    draws = [e for e, _ in state.events if e.kind == "draw"]
                    while len(draws) <= observed_draws and state.phase.startswith(
                        "response_"
                    ):
                        state.step(Action("pass"))
                        draws = [e for e, _ in state.events if e.kind == "draw"]
                    if len(draws) <= observed_draws or (
                        draws[observed_draws].seat,
                        draws[observed_draws].tile,
                    ) != (seat, tile_id(event["tile"])):
                        raise ValueError(
                            "official draw differs from local draw transition"
                        )
                    observed_draws += 1
                elif kind in ("tile_discarded", "chi", "peng", "gang"):
                    action = _action(event)
                    advances = 0
                    while (
                        state.turn != seat or action not in state.legal_actions()
                    ) and state.phase.startswith("response_"):
                        state.step(Action("pass"))
                        advances += 1
                        if advances > 4:
                            raise ValueError("cannot reach official response window")
                    if state.turn != seat:
                        raise ValueError("official actor differs from local turn")
                    if kind == "tile_discarded" and _discard_timeout_follows(
                        events, index
                    ):
                        timeout_checks += 1
                        check(
                            state.timeout_action() == action,
                            "official discard timeout matches local fallback",
                            number,
                            seq,
                        )
                    state.step(action)
                elif kind == "round_ended":
                    if data["draw"]:
                        while state.phase.startswith("response_"):
                            state.step(Action("pass"))
                        if not state.settled or state.winner is not None:
                            raise ValueError(
                                "official draw settlement differs from local state"
                            )
                    else:
                        if state.turn != seat:
                            raise ValueError("official winner differs from local actor")
                        state.step(Action("hu"))
                        check(
                            state.fan.fan == data["fan"],
                            "official fan matches local calculator",
                            number,
                            seq,
                        )
                    check(
                        data.get("dealer", state.dealer) == state.dealer,
                        "event dealer matches initial hand",
                        number,
                        seq,
                    )
                    if "scores" not in data:
                        missing.append(
                            {"round_no": number, "reason": "missing settlement scores"}
                        )
                    else:
                        check(
                            tuple(data["scores"]) == state.delta,
                            "official score delta matches local settlement",
                            number,
                            seq,
                        )
                    cumulative = [
                        s + delta for s, delta in zip(cumulative, state.delta)
                    ]
                    result = {
                        "round_no": number,
                        "dealer": state.dealer,
                        "winner": state.winner,
                        "local_payments": state.delta,
                    }
                    if state.fan:
                        result.update(local_fan=state.fan.fan, official_fan=data["fan"])
                    outcomes.append(result)
                    previous = (
                        number,
                        state.dealer if state.winner is None else state.winner,
                    )
                    ended = True
                elif kind == "game_ended":
                    if complete_prefix:
                        check(
                            tuple(data["final_scores"]) == tuple(cumulative),
                            "official final scores match local ledger",
                            number,
                            seq,
                        )
                    continue
                else:
                    # The actual discard/draw is logged separately from timeouts.
                    continue
                state.assert_conservation()
                check(
                    True, "official transition legal and tiles conserved", number, seq
                )
            except (ValueError, KeyError, AssertionError, TypeError, IndexError) as exc:
                check(False, f"official transition rejected: {exc}", number, seq)
                complete_prefix = False
                break
        if not ended:
            missing.append(
                {"round_no": number, "reason": "missing or unreachable round_ended"}
            )
            complete_prefix = False
    return {
        "status": "failed"
        if failures
        else "incomplete"
        if missing or not outcomes
        else "passed",
        "hands": len(outcomes),
        "checks": checks,
        "timeout_checks": timeout_checks,
        "failures": failures,
        "missing": missing,
        "outcomes": outcomes,
        "event_types": dict(event_types),
        "scope": "real engine replay: observed actions, draws, discard timeout choices, fan, score deltas, dealer and conservation; response timing and unobserved paths unverified",
    }


def main(argv=None):
    import argparse
    import json
    from mahjong_ai.persistence.artifacts import read_json, atomic_json

    p = argparse.ArgumentParser(
        description="Audit completed official replay JSON against the real local engine"
    )
    p.add_argument("replay")
    p.add_argument("--base-score", type=int, default=1)
    p.add_argument("--you-cai-bi-kao", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    result = audit_replay(read_json(args.replay), args.base_score, args.you_cai_bi_kao)
    atomic_json(args.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("outcomes", "failures")},
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
