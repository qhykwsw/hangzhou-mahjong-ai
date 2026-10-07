"""Branches of a learning decision (guided.branches): the copies play the same on both engines, bit for bit."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "projects/personal/selfplay"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

from guided import artifacts, fastgame  # noqa: E402  (fastgame first: members register through it)
from guided.config import configuration  # noqa: E402
from guided.environment import GREEDY, Game  # noqa: E402
from guided.statistics import seed_for  # noqa: E402
from test_guided_fastgame import learner, members, rule_table  # noqa: E402,F401


def candidates(row, legal):
    """Scripted candidates: none at some rows, else up to three legal actions picked from the row's bytes."""
    digest = int(np.frombuffer(row["planes"].tobytes(), dtype=np.uint16).sum()) + 3 * int(row["events"].sum())
    if digest % 4 == 0 or len(legal) < 2:
        return ()
    return tuple(legal[(digest // 5 + j * (1 + digest % 3)) % len(legal)] for j in range(min(3, len(legal) - 1)))


def play(game, slots=None):
    """Drive a game and its copies as the worker does, the scripted learner answering (with candidates).

    ``slots`` collects (kind of copy, slot) of the copies' requests.
    """
    requests = 0
    while True:
        for shadow in game.shadows:
            while not shadow.done:
                request = shadow.advance()
                if request is None:
                    break
                if slots is not None:
                    slots.append((type(shadow).__name__.removeprefix("Fast"), request["slot"]))
                legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
                shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
                requests += 1
        request = game.advance()
        if request is None:
            if game.settled():
                return requests
            continue
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        choice, greedy = learner(request["row"], legal)
        game.apply((choice, -0.5, 0.125, greedy, candidates(request["row"], legal)))
        requests += 1


def plain(value):
    """A row (with nested results) as comparable values, arrays by dtype, shape and bytes."""
    if hasattr(value, "tobytes"):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(v) for v in value)
    return value


def both(cfg, members, monkeypatch, seats, seed, *, entries=None):
    """The same game on the Python and the C engine, played out; (games, the copies' request slots)."""
    if entries is None:
        by_id = {f"m{i}": m for i, m in enumerate(members)}
        monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
        entries = {k: None for k in by_id}
    focal = seats.index("model:current")
    games = [cls(cfg, {"entries": []}, entries, seats, seed, focal=focal, train=True)
             for cls in (Game, fastgame.FastGame)]
    assert games[1].seats is not None
    slots = [[], []]
    counts = [play(game, seen) for game, seen in zip(games, slots)]
    assert counts[0] == counts[1] > 50
    assert slots[0] == slots[1]
    assert games[0].scores == games[1].scores
    assert plain(games[0].rows) == plain(games[1].rows)
    return games, slots[1]


BASE = {"playout_baseline": True, "branches": 3}


@pytest.mark.parametrize("extra", [
    {},
    {"vine": True, "stable_wall": True, "luck": True},
    {"stage": {"kind": "advancement", "games": 10, "per_game_sd": 70}, "encoding": "guided-current-hand-v4"},
])
def test_branch_copies_play_the_same_on_both_engines(members, monkeypatch, extra):
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra
    games, _ = both(cfg, members, monkeypatch, ["m0", "model:current", "m2", "m5"],
                    seed_for(61, "branches", str(sorted(extra))))
    rows = games[1].rows
    branched = [row for row in rows if "branches" in row]
    assert len(branched) > 20 and len(branched) < len(rows)
    for row in branched:
        legal = np.flatnonzero(row["mask"])
        assert [b["action"] for b in row["branches"]] == list(candidates(row, list(legal)))
        for result in row["branches"]:
            assert result["truncated"] is False and result["playout_dealer"] in range(4)
            assert len(result["playout_scores"]) == 4 and result["playout_deals_next"] in (0.0, 1.0)
            assert ("playout_luck_pay" in result) == bool(extra.get("luck"))
    assert not any("branches" in row for row in rows if not candidates(row, list(np.flatnonzero(row["mask"]))))


@pytest.mark.parametrize("extra", [{}, {"vine": True, "stable_wall": True, "luck": True}])
def test_branch_copies_with_python_rule_seats_play_the_same_on_both_engines(members, monkeypatch, extra):
    entries, _ = rule_table(members, monkeypatch, {"py": 2})
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra
    games, _ = both(cfg, members, monkeypatch, ["model:current", "py", "m1", "m3"],
                    seed_for(67, "branches-python", str(sorted(extra))), entries=entries)
    assert games[1].python_seats
    assert sum("branches" in row for row in games[1].rows) > 20


def test_a_candidate_copy_is_the_hand_it_continues(members, monkeypatch):
    """Under vine the hand plays the argmax and the deviation's copy the sampled move, each continuing
    greedily: a candidate equal to the argmax replays the hand itself, one equal to the sampled move
    its deviation copy (same wall, same seats, the same answers)."""
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "vine": True, "branches": 2, "stable_wall": True, "luck": True}
    seats = ["m1", "model:current", "m3", "m4"]
    for cls in (Game, fastgame.FastGame):
        game = cls(cfg, {"entries": []}, {k: None for k in by_id}, seats, seed_for(71, "branch-identity"),
                   focal=1, train=True)
        slots = []
        while True:
            for shadow in game.shadows:
                while not shadow.done:
                    request = shadow.advance()
                    if request is None:
                        break
                    slots.append(request["slot"])
                    legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
                    shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
            request = game.advance()
            if request is None:
                if game.settled():
                    break
                continue
            legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
            choice, greedy = learner(request["row"], legal)
            game.apply((choice, -0.5, 0.125, greedy, (greedy, choice)))
        assert set(slots) == {GREEDY}
        outcome = {}
        for row in game.rows:
            outcome[row["hand"]] = outcome.get(row["hand"], 0.0) + row["reward"]
        checked = 0
        for row in game.rows:
            same, deviation = row["branches"]
            # The hand after this decision is the argmax path: the first branch is its outcome.
            assert same["playout"] == outcome[row["hand"]]
            if row["deviated"]:
                assert deviation["playout"] == row["playout"]
                assert deviation["playout_luck_pay"] == row["playout_luck_pay"]
                checked += 1
            else:
                assert deviation == same
        assert checked > 5


def tables(members, monkeypatch, python_ids, extra=None):
    """build_policy for members m0.., PythonRule seats under ``python_ids`` and ``extra`` ids; the entries."""
    entries, _ = rule_table(members, monkeypatch, python_ids)
    build = artifacts.build_policy
    extra = extra or {}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: extra[cid]() if cid in extra else build(bundle, cid))
    return entries | {k: None for k in extra}


@pytest.mark.parametrize("extra", [{}, {"vine": True, "stable_wall": True, "luck": True},
                                   {"stage": {"kind": "final", "games": 10, "per_game_sd": 70}}])
def test_a_rule_continuation_and_its_control_play_the_same_on_both_engines(members, monkeypatch, extra):
    """The learner continues as a race member (in C), one opponent becomes another member, one a rule
    deciding in Python; every row with candidates also gets its argmax under the same continuation."""
    entries = tables(members, monkeypatch, {"py": 3})
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra | {
        "branch_continuation": {"learner": "m2", "opponents": {"m1": "m5", "m4": "py"}}, "branch_control": True}
    games, slots = both(cfg, members, monkeypatch, ["m4", "model:current", "m1", "m0"],
                        seed_for(73, "branch-continuation", str(sorted(extra))), entries=entries)
    plan = games[1]._branch_seats
    assert plan.python == {0} and plan.slots == {} and -1 not in plan.profiles[1:]
    # The copies ask the parent nothing: every seat of the continuation is a rule.
    assert not any(kind == "Branch" for kind, _ in slots)
    branched = [row for row in games[1].rows if "branches" in row]
    assert len(branched) > 20
    for row in branched:
        control = row["branch_control"]
        assert control["action"] == learner(row, list(np.flatnonzero(row["mask"])))[1]
        assert control["truncated"] is False and "playout" in control
    assert not any("branch_control" in row for row in games[1].rows if "branches" not in row)
    # The continuation's Python rule saw the same Observations on both engines.
    python, fast = games[0]._branch_seats[0][1], games[1]._branch_seats.policies[0]
    assert python.seen == fast.seen and len(fast.seen) > 50


def test_a_model_continuation_asks_the_parent_under_its_slot(members, monkeypatch):
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | {
        "branch_continuation": {"learner": "model:snapshot", "opponents": {"m3": "model:cheap"}}}
    _, slots = both(cfg, members, monkeypatch, ["model:current", "m1", "m3", "m5"], seed_for(79, "branch-model"))
    assert {slot for kind, slot in slots if kind == "Branch"} == {"model:snapshot", "model:cheap"}
    assert {slot for kind, slot in slots if kind == "Shadow"} == {GREEDY}


def test_continuation_options_are_checked(members, monkeypatch):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True) | BASE
    seats = ["model:current", "m1", "m2", "m3"]
    for bad, match in [({"branch_continuation": {"learner": "model:current"}}, "cannot continue"),
                       ({"branch_continuation": {"leaner": "m1"}}, "is {'learner'"),
                       ({"branches": 0, "branch_control": True}, "need cfg"),
                       ({"branch_truncate": -1}, "positive integer"),
                       ({"branches": 0, "branch_truncate": 2}, "need cfg"),
                       ({"branches": 0, "branch_resample": {}}, "need cfg"),
                       ({"branch_resample": {"weight": None}}, "is {'weights'")]:
        with pytest.raises(ValueError, match=match):
            Game(cfg | bad, {"entries": []}, {k: None for k in by_id}, seats, 1, train=True)
    for cls in (Game, fastgame.FastGame):
        with pytest.raises(ValueError, match="unknown branch continuation"):
            cls(cfg | {"branch_continuation": {"learner": "nowhere"}}, {"entries": []}, {k: None for k in by_id},
                seats, 1, train=True)


RULE131 = Path(__import__("os").environ.get("RULE131_PACKAGE")
               or PROJECT.parents[2] / "projects/personal/firstprinciples/.local/attempts/fp-rule-131/package")
needs131 = pytest.mark.skipif(not fastgame._is_rule131(str(RULE131)), reason="fp-rule-131's frozen package is not here")


def rule131():
    from test_guided_rule131 import rule131 as load

    return load()


@needs131
@pytest.mark.parametrize("extra", [{}, {"vine": True, "stable_wall": True, "luck": True}])
def test_a_131_continuation_and_its_control_play_the_same_on_both_engines(members, monkeypatch, extra):
    entries = tables(members, monkeypatch, {}, {"r131": rule131})
    cfg = configuration(rounds=8, lookahead=True, gae_lambda=0.95) | BASE | extra | {
        "branch_continuation": {"learner": "r131", "opponents": None}, "branch_control": True}
    games, slots = both(cfg, members, monkeypatch, ["m1", "m3", "model:current", "m4"],
                        seed_for(83, "branch-131", str(sorted(extra))), entries=entries)
    plan = games[1]._branch_seats
    assert not plan.python and plan.profiles[2] == fastgame.rule131_member(plan.policies[2])[0]
    assert -1 not in plan.profiles and not any(kind == "Branch" for kind, _ in slots)
    assert sum("branch_control" in row for row in games[1].rows) > 10


@needs131
def test_131_candidates_and_their_control_continue_on_one_wall(members, monkeypatch):
    """A one-hand game, by hand: at the learner's first discard (no kong, no win) with three plain
    discards (not the white) the answer names the lowest as its argmax and the next two as candidates. Each copy starts
    from that very hand (the remaining wall the same tiles in the same order in all three), plays its
    discard, then fp-rule-131 plays the learner's seat in C and the members the others, to the end:
    each result is the hand an independent replay of that discard reaches from the same fork, so a
    candidate's playout minus the control's compares two discards on one wall and one continuation."""
    entries = tables(members, monkeypatch, {}, {"r131": rule131})
    cfg = configuration(rounds=1, lookahead=True, gae_lambda=0.95) | {
        "playout_baseline": True, "stable_wall": True, "branches": 2,
        "branch_continuation": {"learner": "r131"}, "branch_control": True}
    seats = ["model:current", "m1", "m3", "m4"]
    member = fastgame.rule131_member(rule131())
    rows = []
    for cls in (Game, fastgame.FastGame):
        game = cls(cfg, {"entries": []}, entries, seats, seed_for(89, "branch-131-by-hand", 0), focal=0, train=True)
        asked = None
        while True:
            for shadow in game.shadows:
                while not shadow.done:
                    request = shadow.advance()
                    assert request is None or request["slot"] == GREEDY  # only the deviation copies ask
                    if request is None:
                        break
                    legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
                    shadow.apply((legal[0], 0.0, 0.0, 0))
            request = game.advance()
            if request is None:
                if game.settled():
                    break
                continue
            legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
            plain_discards = [i for i in legal if i < 33]  # not the white: its discard draws at once
            if asked is None and len(plain_discards) >= 3 and max(legal) < 34:
                asked = plain_discards[:3]
                if cls is fastgame.FastGame:
                    fork = game.state.fork()
                    copies_start = len(game.shadows)
                game.apply((asked[0], -0.5, 0.0, asked[0], tuple(asked[1:])))
                if cls is fastgame.FastGame:
                    walls = [s.state.snapshot()["wall"] for s in game.shadows[copies_start:]]
                    assert len(walls) == 3 and walls[0] == walls[1] == walls[2] == fork.snapshot()["wall"]
                continue
            game.apply((legal[0], -0.5, 0.0, legal[0], ()))
        assert asked is not None
        rows.append(next(row for row in game.rows if "branches" in row))
    assert plain(rows[0]) == plain(rows[1])
    row = rows[1]
    assert [b["action"] for b in row["branches"]] == asked[1:] and row["branch_control"]["action"] == asked[0]
    # The independent replays from the same fork: 131 in the learner's seat, the members in the others'.
    plans = [member] + [fastgame.race_member(members[int(s[1:])]) for s in seats[1:]]
    profiles, fallbacks = tuple(p for p, _ in plans), tuple(f for _, f in plans)
    for result in (*row["branches"], row["branch_control"]):
        hand = fork.fork()
        hand.step(result["action"])
        assert hand.run(profiles, fallbacks) is None
        base = game.config.base_score
        assert result["playout"] == hand.delta[0] / base
        assert result["playout_scores"] == [d / base for d in hand.delta]
    # The example by hand: the argmax discards tile 5, the candidates tiles 9 and 12; on the same wall the
    # first candidate's hand ends +24 for the learner, the second's -16, the argmax's -8, so the paired
    # differences to the control are +32 and -8.
    assert asked == [5, 9, 12]
    assert [b["playout"] for b in row["branches"]] == [24.0, -16.0] and row["branch_control"]["playout"] == -8.0


def leaves(rows):
    """(row, branch) -> its truncated result, over branches and controls."""
    out = {}
    for i, row in enumerate(rows):
        for j, result in enumerate([*row.get("branches", ()), *([row["branch_control"]] if "branch_control" in row else [])]):
            out[(i, j)] = result
    return out


@pytest.mark.parametrize("k, extra", [
    (1, {}),
    (3, {"vine": True, "stable_wall": True, "luck": True}),
    (2, {"stage": {"kind": "advancement", "games": 10, "per_game_sd": 70}, "encoding": "guided-current-hand-v4"}),
])
def test_truncated_copies_stop_at_the_learners_kth_decision_on_both_engines(members, monkeypatch, k, extra):
    """A truncated copy is the untruncated one cut short: its leaf row is the k-th learner request of the
    same copy played out, its scores those before the hand; copies whose hand ends first are complete."""
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra | {"branch_control": True}
    seats = ["m0", "model:current", "m2", "m5"]
    seed = seed_for(101, "branch-truncate", k)
    games, _ = both(cfg | {"branch_truncate": k}, members, monkeypatch, seats, seed)
    cut = leaves(games[1].rows)
    # The same game without truncation, recording each copy's learner requests.
    full = fastgame.FastGame(cfg, {"entries": []}, {f"m{i}": None for i in range(6)}, seats, seed, focal=1, train=True)
    asked, before = {}, []
    while True:
        for shadow in full.shadows:
            while not shadow.done:
                request = shadow.advance()
                if request is None:
                    break
                if request["slot"] == GREEDY and isinstance(shadow, fastgame.FastBranch):
                    asked.setdefault(id(shadow.row), []).append(request["row"])
                legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
                shadow.apply((learner(request["row"], legal)[1], -0.25, 0.0, 0))
        request = full.advance()
        if request is None:
            if full.settled():
                break
            continue
        before.append([score / full.config.base_score for score in full.scores])  # the game's, before this hand
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        choice, greedy = learner(request["row"], legal)
        full.apply((choice, -0.5, 0.125, greedy, candidates(request["row"], legal)))
    assert len(before) == len(full.rows)
    whole = leaves(full.rows)
    assert set(whole) == set(cut) and len(cut) > 30
    stopped = 0
    for key, result in cut.items():
        played = whole[key]
        if not result["truncated"]:
            assert result == played and len(asked.get(id(played), ())) < k
            continue
        stopped += 1
        leaf = asked[id(played)][k - 1]
        assert set(result["leaf_row"]) == {"planes", "scalars", "events", "mask"}
        assert plain(result["leaf_row"]) == plain({key_: leaf[key_] for key_ in result["leaf_row"]})
        assert result["playout_scores"] == before[key[0]]
        assert "playout" not in result and "playout_dealer" not in result
        assert ("playout_luck_pay" in result) == bool(extra.get("luck"))
    assert stopped > 20


@needs131
def test_a_truncated_131_continuation_counts_the_rules_decisions_on_both_engines(members, monkeypatch):
    """With a rule in the learner's seat the copy stops at the rule's k-th decision: in C the seat leaves run
    at each of its turns and plays fp-rule-131 by Hand.act until then."""
    entries = tables(members, monkeypatch, {"py": 4}, {"r131": rule131})
    cfg = configuration(rounds=8, lookahead=True, gae_lambda=0.95) | BASE | {
        "branch_continuation": {"learner": "r131", "opponents": {"m1": "py"}}, "branch_control": True,
        "branch_truncate": 4, "stable_wall": True, "luck": True}
    games, slots = both(cfg, members, monkeypatch, ["m1", "m3", "model:current", "m4"],
                        seed_for(103, "branch-truncate-131"), entries=entries)
    plan = games[1]._branch_seats
    assert plan.profiles[2] == -1 and 2 in plan.members and plan.python == {0}
    assert not any(kind == "Branch" for kind, _ in slots)
    results = list(leaves(games[1].rows).values())
    assert sum(r["truncated"] for r in results) > 10 and sum(not r["truncated"] for r in results) > 0


def test_a_python_rule_learner_truncates_on_both_engines(members, monkeypatch):
    entries = tables(members, monkeypatch, {"py": 2})
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | {
        "branch_continuation": {"learner": "py"}, "branch_truncate": 2}
    games, _ = both(cfg, members, monkeypatch, ["model:current", "m1", "m3", "m4"],
                    seed_for(107, "branch-truncate-python"), entries=entries)
    assert games[1]._branch_seats.python == {0}
    assert sum(r["truncated"] for r in leaves(games[1].rows).values()) > 20


WORLDS = []


def weight(row, world):
    """A world-weight hook: records what it is given, and weighs by the others' hands and the row's inputs."""
    WORLDS.append((sorted(row), world))
    others = [h for s, h in enumerate(world["hands"]) if s != world["seat"]]
    return 1.0 + sum(sum(h) for h in others) / 1000.0 + float(row["planes"].astype(np.float32).sum()) / 1e6


@pytest.mark.parametrize("extra", [
    {"branch_control": True},
    {"branch_control": True, "vine": True, "stable_wall": True, "luck": True, "branch_truncate": 3},
    {"stage": {"kind": "advancement", "games": 10, "per_game_sd": 70}, "encoding": "guided-current-hand-v4"},
])
def test_resampled_worlds_are_the_same_on_both_engines(members, monkeypatch, extra):
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra | {
        "branch_resample": {"weights": "test_guided_branches:weight"}}
    WORLDS.clear()
    seats = ["m3", "m0", "model:current", "m5"]
    games, _ = both(cfg, members, monkeypatch, seats, seed_for(109, "branch-resample", str(sorted(extra))))
    half = len(WORLDS) // 2
    assert half > 20 and WORLDS[:half] == WORLDS[half:]  # the Python game's worlds, then the C game's
    rows = [row for row in games[1].rows if "branches" in row]
    assert len(rows) == half
    for row, (keys, world) in zip(rows, WORLDS[half:]):
        assert keys == ["events", "mask", "planes", "scalars"] and world["seat"] == 2
        results = [*row["branches"], *([row["branch_control"]] if "branch_control" in row else [])]
        assert len({r["world_weight"] for r in results}) == 1
        assert results[0]["world_weight"] == weight({k: row[k] for k in keys}, world)
    # The same game without resampling: the copies play other worlds, the game itself is the same.
    plain_cfg = {k: v for k, v in cfg.items() if k != "branch_resample"}
    again, _ = both(plain_cfg, members, monkeypatch, seats, seed_for(109, "branch-resample", str(sorted(extra))))
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("branches", "branch_control")} for r in rows]
    assert plain(strip(again[1].rows)) == plain(strip(games[1].rows))
    keys = ("playout", "playout_scores", "leaf_row")
    differ = sum(plain([{k: b.get(k) for k in keys} for b in x["branches"]])
                 != plain([{k: b.get(k) for k in keys} for b in y["branches"]])
                 for x, y in zip(again[1].rows, games[1].rows) if "branches" in x)
    assert differ > 5
    assert not any("world_weight" in b for row in again[1].rows for b in row.get("branches", ()))


@pytest.mark.parametrize("extra", [{}, {"stable_wall": True, "luck": True, "branch_truncate": 2}])
def test_resampled_worlds_with_python_seats_play_the_same_on_both_engines(members, monkeypatch, extra):
    """Python rule seats (the game's and a continuation's) decide in a re-dealt world on the C engine
    from a Mirror re-dealt the same way."""
    entries = tables(members, monkeypatch, {"py": 2, "py2": 4})
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | BASE | extra | {
        "branch_resample": {"weights": None}, "branch_control": True,
        "branch_continuation": {"learner": "greedy", "opponents": {"m1": "py2"}}}
    games, _ = both(cfg, members, monkeypatch, ["model:current", "py", "m1", "m3"],
                    seed_for(113, "branch-resample-python", str(sorted(extra))), entries=entries)
    assert games[1].python_seats == {1} and games[1]._branch_seats.python == {1, 2}
    results = [b for row in games[1].rows for b in row.get("branches", ())]
    assert len(results) > 20 and all(b["world_weight"] == 1.0 for b in results)


def test_a_world_keeps_the_learners_view_and_redeals_the_rest(members, monkeypatch):
    from guided.branches import redeal

    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True) | BASE | {"branch_resample": {}, "stable_wall": True}
    game = fastgame.FastGame(cfg, {"entries": []}, {k: None for k in by_id}, ["m1", "model:current", "m2", "m3"],
                             seed_for(127, "world"), focal=1, train=True)
    changed = 0
    for _ in range(30):
        request = game.advance()
        if request is None:
            break
        real = game.state.snapshot()
        (world, _), weight = game._branch_world(request["row"])
        got = world.snapshot()
        assert weight == 1.0
        assert got["hands"][1] == real["hands"][1]
        for key in ("melds", "discards", "events", "initial_hands", "white_acquired", "chain", "piao", "catch_owner",
                    "turn", "phase", "drawn", "scores", "steps", "log"):
            assert got[key] == real[key], key
        assert [len(h) for h in got["hands"]] == [len(h) for h in real["hands"]]
        assert len(got["wall"]) == len(real["wall"])
        hidden = lambda snap: sorted([t for s in (0, 2, 3) for t in snap["hands"][s]] + list(snap["wall"]))
        assert hidden(got) == hidden(real)
        changed += got["hands"] != real["hands"]
        # The world depends on the learner's view alone: the others' real tiles dealt otherwise give the same one.
        pool, hands = hidden(real), [list(h) for h in real["hands"]]
        for s in (0, 2, 3):
            hands[s], pool = pool[:len(hands[s])], pool[len(hands[s]):]
        assert hands != [list(h) for h in real["hands"]]
        assert redeal(hands, pool, 1, 7) == redeal(real["hands"], real["wall"], 1, 7)
        legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
        game.apply((legal[0], 0.0, 0.0, legal[0]))
    assert changed > 20
    hand = fastgame.native.Hand(5, 0, 1, (0, 0, 0, 0), 16)
    snap = hand.snapshot()
    with pytest.raises(ValueError, match="keeps the tiles"):
        hand.redeal([snap["hands"][0][:-1] + (snap["wall"][0],), *snap["hands"][1:]], snap["wall"][:-1] + (33,))
    with pytest.raises(ValueError, match="tiles expected"):
        hand.redeal(snap["hands"], snap["wall"][1:])
    assert hand.snapshot() == snap  # a refused redeal changes nothing


@needs131
def test_resampled_131_continuations_play_the_same_on_both_engines(members, monkeypatch):
    entries = tables(members, monkeypatch, {}, {"r131": rule131})
    cfg = configuration(rounds=8, lookahead=True, gae_lambda=0.95) | BASE | {
        "branch_continuation": {"learner": "r131"}, "branch_control": True, "branch_resample": {},
        "branch_truncate": 3, "stable_wall": True, "luck": True}
    games, slots = both(cfg, members, monkeypatch, ["m1", "model:current", "m3", "m4"],
                        seed_for(131, "branch-resample-131"), entries=entries)
    assert not any(kind == "Branch" for kind, _ in slots)
    assert sum("branch_control" in row for row in games[1].rows) > 10


def test_without_branches_no_row_carries_them_and_stray_candidates_fail(members, monkeypatch):
    by_id = {f"m{i}": m for i, m in enumerate(members)}
    monkeypatch.setattr(artifacts, "build_policy", lambda bundle, cid: by_id[cid])
    cfg = configuration(rounds=16, lookahead=True, gae_lambda=0.95) | {"playout_baseline": True}
    game = fastgame.FastGame(cfg, {"entries": []}, {k: None for k in by_id}, ["model:current", "m1", "m2", "m3"],
                             seed_for(3, "no-branches"), focal=0, train=True)
    assert game.branching is None
    request = game.advance()
    legal = [int(i) for i in np.flatnonzero(request["row"]["mask"])]
    with pytest.raises(ValueError, match="need cfg"):
        game.apply((legal[0], 0.0, 0.0, legal[0], (legal[0],)))
    with pytest.raises(ValueError, match="playout_baseline"):
        Game(configuration(rounds=16, lookahead=True) | {"branches": 2}, {"entries": []}, {k: None for k in by_id},
             ["model:current", "m1", "m2", "m3"], 1, train=True)
    with pytest.raises(ValueError, match="positive integer"):
        Game(cfg | {"branches": True}, {"entries": []}, {k: None for k in by_id},
             ["model:current", "m1", "m2", "m3"], 1, train=True)
    # Evaluation games never branch.
    assert Game(cfg | {"branches": 2}, {"entries": []}, {k: None for k in by_id},
                ["model:current", "m1", "m2", "m3"], 1).branching is None
