"""Platform replays rebuild into exactly the decisions that produced them."""

import random

import pytest

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.rules import tile_code
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.strategies.common import RulePolicy

from guided.platform_replays import ReplayMismatch, decisions

GANG = {"concealed": "an", "added": "bu", "exposed": "ming"}


def play(seed, rounds=4, page=128):
    """A local game written the way the platform serves replays, and its decisions."""
    rng, rule = random.Random(seed), RulePolicy()
    config = RoundConfig(rounds=rounds, allow_tiebreak=False)
    scores, history, views, dealer = [0] * 4, [], None, 0
    blocks, results, expected, seq = [], [], [], 0

    def emit(kind, seat=-1, tile="", data=None):
        nonlocal seq
        seq += 1
        events.append(dict(seq=seq, type=kind, seat=seat, tile=tile, data=data, ts=0))

    for hand_no in range(1, rounds + 1):
        state = HandState(config, seed * 100 + hand_no, dealer, hand_no, scores, history,
                          False, previous_views=views)
        start = [[tile_code(t) for t in h] for h in state.initial_hands]
        start[dealer].append(tile_code(state.drawn))
        events, seen = [], len(state.events)
        while not state.settled:
            legal = state.legal_actions()
            # Mostly a rule player, so hands are won; random claims add melds and kongs.
            claims = [a for a in legal if a.action in ("peng", "gang", "chi")]
            action = (
                rng.choice(claims) if claims and rng.random() < 0.4
                else rule.act(state.observation()) if len(legal) > 1 else legal[0]
            )
            responding = state.phase.startswith("response_")
            timeout = action.action in ("pass", "discard") and rng.random() < 0.1
            if len(legal) > 1:
                expected.append((hand_no, state.turn, action, timeout))
            seat, window = state.turn, state.phase.split("_")[-1]
            state.step(action)
            if responding and action.action == "pass":
                if timeout:
                    emit("timeout", seat, data={"kind": "response", "window": window})
                else:
                    emit("pass", seat)
            for event, _ in state.events[seen:]:
                if event.kind == "draw":
                    emit("tile_drawn", event.seat, tile_code(event.tile),
                         {"gang_replenish": True} if state.events[seen - 1][0].kind == "gang"
                         else None)
                elif event.kind == "discard":
                    emit("tile_discarded", event.seat, tile_code(event.tile), {"catch_play": False})
                    if timeout:
                        emit("timeout", event.seat, data={"kind": "discard"})
                elif event.kind in ("chi", "peng"):
                    emit(event.kind, event.seat, tile_code(event.tile),
                         {"tiles": [tile_code(t) for t in event.tiles]} if event.kind == "chi" else None)
                elif event.kind == "gang":
                    emit("gang", event.seat, tile_code(event.tile), {"kind": GANG[action.kind]})
                elif event.kind == "hand_ended":
                    emit("round_ended", -1 if event.seat is None else event.seat,
                         data={"draw": state.winner is None, "scores": list(state.delta),
                               "round_no": hand_no, "dealer": dealer})
                seen += 1
        results.append(dict(round_no=hand_no, dealer=dealer, winner=state.winner,
                            is_draw=int(state.winner is None), multiplier=1,
                            scores=list(state.delta)))
        scores = [s + d for s, d in zip(scores, state.delta)]
        history, views = state.events, state.public_views()
        dealer = dealer if state.winner is None else state.winner
        if hand_no == rounds:
            emit("game_ended", data={"final_scores": scores})
        for i in range(0, len(events), page):
            chunk = events[i:i + page]
            blocks.append(dict(round_no=hand_no, dealer=results[-1]["dealer"],
                               seq_start=chunk[0]["seq"], seq_end=chunk[-1]["seq"],
                               start_hands=start if i == 0 else [None] * 4,
                               events=chunk, truncated=False))
    replay = dict(game_id=f"g{seed}", room_id="r", batch=0, status="finished",
                  seats=[dict(user_id=f"u{i}", name="") for i in range(4)],
                  blocks=blocks, rounds=results)
    return replay, expected


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_replays_rebuild_the_exact_decisions(seed):
    replay, expected = play(seed)
    rebuilt = decisions(replay, {"Rounds": 4})
    assert [(d.hand, d.seat, d.action, d.timeout) for d in rebuilt] == expected
    assert all(d.observation.stage.seat == d.seat for d in rebuilt)
    assert all(d.action in d.observation.legal_actions for d in rebuilt)


def test_a_replay_that_disagrees_with_the_rules_is_rejected():
    replay, _ = play(4)
    event = next(e for b in replay["blocks"] for e in b["events"] if e["type"] == "round_ended")
    event["data"]["scores"] = [s + 1 for s in event["data"]["scores"]]
    with pytest.raises(ReplayMismatch):
        decisions(replay, {"Rounds": 4})


def test_dataset_indexes_ranks_freezes_and_encodes(tmp_path):
    import json

    from guided import platform_dataset as dataset

    room = tmp_path / "competition" / "free-1"
    (room / "replays").mkdir(parents=True)
    (room / "match.json").write_text(json.dumps({"user_id": "u0", "match": {"config": {"Rounds": 4}}}))
    replay, expected = play(5)
    (room / "replays" / "g5.json").write_text(json.dumps(replay))
    data = tmp_path / "data"
    added = dataset.scan([tmp_path / "competition"], data, tmp_path)
    assert [r["status"] for r in added] == ["ok"]
    assert dataset.scan([tmp_path / "competition"], data, tmp_path) == []  # incremental
    table = dataset.players(dataset.load_index(data))
    assert sorted(s["user_id"] for s in table) == ["u1", "u2", "u3"]  # ours excluded
    manifest = dataset.freeze(dataset.load_index(data), ["u2"])
    rows = list(dataset.rows(manifest, tmp_path))
    want = [(h, a) for h, s, a, t in expected if s == 2 and not t]
    assert len(rows) == len(want) and {r["user"] for r in rows} == {"u2"}
    assert all(r["mask"][r["teacher"]] for r in rows)
