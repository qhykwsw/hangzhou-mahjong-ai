"""Openings: training hands that start from a recorded position.

An opening is a dict with ``seed``, ``dealer``, ``hand_no``, ``scores``,
``prefix`` and ``focal`` (``kind`` and other keys are kept as labels). The
wall comes from ``seed`` exactly as HandState shuffles it, and ``prefix`` is
every engine step of the hand from the deal, single-legal steps included, as
``[action, tile, tiles, kind]``. Replaying the prefix on the engine reaches the
position with the events and observations of a real game; the learning seat
``focal`` decides next. Hands before the opening are not part of its game, so
the observations carry no earlier-hand history. The maintainer's harness
(``harness.starts``) records openings from rule self-play.
"""

from collections import Counter
import gzip
import json
from pathlib import Path
import random
import time

KEYS = ("seed", "dealer", "hand_no", "scores", "prefix", "focal")


def pack(action):
    return [action.action, action.tile, list(action.tiles), action.kind]


def unpack(item):
    from mahjong_ai.domain.types import Action

    action, tile, tiles, kind = item
    return Action(action, tile, tuple(tiles), kind)


def check(opening):
    missing = [k for k in KEYS if k not in opening]
    if missing:
        raise ValueError(f"opening lacks {missing}")
    if not (0 <= opening["focal"] < 4 and 0 <= opening["dealer"] < 4 and opening["hand_no"] >= 1
            and len(opening["scores"]) == 4):
        raise ValueError("opening has an invalid seat, dealer, hand number or scores")
    return opening


def replay(state, prefix, focal):
    """Step ``state`` through ``prefix``; the learner ``focal`` must decide next."""
    for item in prefix:
        legal = state.legal_actions()
        action = unpack(item)
        if action not in legal:
            raise ValueError(f"opening step {action} is not legal at its position")
        state.step(action, legal)
    if state.settled or state.turn != focal or len(state.legal_actions()) < 2:
        raise ValueError("an opening must stop where its learning seat has a choice")


def save(path, openings):
    with gzip.open(path, "wt", encoding="utf-8") as out:
        json.dump([check(o) for o in openings], out)


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [check(o) for o in json.load(stream)]


PER_HAND = 3  # recorded decisions of one kind per hand at most (they are correlated)


def _ready(state, seat):
    """Tenpai: some tile completes the hand (after the best discard when holding the drawn tile)."""
    from mahjong_ai.domain.rules import can_hu

    hand, melds = list(state.hands[seat]), len(state.melds[seat])

    def waiting(tiles):
        return any(tiles.count(t) < 4 and can_hu(tiles + [t], melds) for t in range(34))

    if len(hand) == 14 - 3 * melds:
        return any(waiting(hand[:i] + hand[i + 1:]) for i in sorted({hand.index(t) for t in hand}))
    return waiting(hand)


def harvest(task):
    """Record openings from rule self-play (harness.starts runs it under the pool's runtime).

    The task names the runtime, the rule package for all four seats, the kinds,
    a seed, how many openings to find and a deadline. Returns (openings, hands played).
    """
    from .artifacts import bind_runtime, load_package

    bind_runtime({"runtime": task["runtime"], "runtime_hash": Path(task["runtime"]).name})
    from mahjong_ai.domain.engine import HandState, derive_seed
    from mahjong_ai.domain.rules import WHITE, is_baotou
    from mahjong_ai.domain.types import RoundConfig

    config = RoundConfig(rounds=16, allow_tiebreak=False)
    policies = [load_package(task["package"]) for _ in range(4)]
    rng = random.Random(task["seed"])
    kinds, out, played = set(task["kinds"]), [], 0
    while len(out) < task["count"] and time.time() < task["deadline"]:
        seed = derive_seed(task["seed"], "opening", played)
        dealer, hand_no = rng.randrange(4), rng.randint(1, 16)
        for seat, policy in enumerate(policies):
            policy.reset(derive_seed(seed, "policy", seat))
        state = HandState(config, seed, dealer, hand_no, (0, 0, 0, 0))
        state.rounds_total = config.rounds
        trace, kept, recorded = [], {}, Counter()

        def record(kind, seat, **labels):
            if recorded[kind] < PER_HAND:
                recorded[kind] += 1
                out.append({"seed": seed, "dealer": dealer, "hand_no": hand_no, "scores": [0, 0, 0, 0],
                            "prefix": list(trace), "focal": seat, "kind": kind,
                            "whites": state.hands[seat].count(WHITE), **labels})

        while not state.settled:
            legal = state.legal_actions()
            seat = state.turn
            action = None
            if len(legal) > 1:
                if ("fish-window" in kinds and state.drawn is not None and WHITE in state.hands[seat]
                        and any(a.action == "hu" for a in legal)):
                    waiting = list(state.hands[seat])
                    waiting.remove(state.drawn)
                    record("fish-window", seat, baotou=is_baotou(tuple(sorted(waiting)), len(state.melds[seat])))
                gangs = [a for a in legal if a.action == "gang"]
                if seat in kept:
                    if "kept-fourth" in kinds and any(a.tile == kept[seat] for a in gangs):
                        record("kept-fourth", seat, hu=any(a.action == "hu" for a in legal))
                elif gangs and WHITE in state.hands[seat] and _ready(state, seat):
                    if "kong-window" in kinds:
                        record("kong-window", seat, hu=any(a.action == "hu" for a in legal))
                    if "kept-fourth" in kinds:
                        tile = gangs[0].tile
                        kept[seat] = tile
                        allowed = [a for a in legal if a.action != "gang"]
                        action = next((a for a in allowed if a.action == "peng"), None)
                        if action is None:
                            keep = [a for a in allowed if not (a.action == "discard" and a.tile == tile)] or allowed
                            action = policies[seat].act(state.observation(tuple(keep)))
            if action is None:
                action = legal[0] if len(legal) == 1 else policies[seat].act(state.observation(legal))
            trace.append(pack(action))
            state.step(action, legal)
        played += 1
    return out, played
