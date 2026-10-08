"""Training games inside a tournament stage (cfg["stage"]), for rank objectives.

    cfg["stage"] = {"kind": "advancement" | "final", "games": 10, "per_game_sd": 70}

A training game is still one game, placed at a uniformly drawn game g of a
stage of ``games`` games: the four seats carry zero-sum stage scores into it,
each seat's standard deviation per_game_sd·√g (points), drawn from the game's
own random stream (seed_for(seed, "stage")), so walls and policies draw as
without a stage. Every observation of the game, of all four seats, teachers
and continuations alike, carries the stage the way the tournament settlement
(mahjong_ai.domain.settlement) writes it:

    stage_scores        the carried scores plus this game's before the hand, by seat
    rank_by_seat        score_ranks(stage_scores)
    stage_round_no      rounds·g + the hand's number in this game
    stage_rounds_total  rounds·games
    kind                the stage's StageKind

The final's tiebreak games are not played. The hand state carries
``stage_prior`` (the carried scores by seat), ``stage_game`` (g) and
``stage_kind`` for a recipe's critic_features; the learner's rows carry
``stage_prior`` (relative to the learner: its own first) and ``stage_game``.

The C engine (guided.fastgame) encodes its rows from the game's own scores and
hand; their stage-dependent scalars (the score context, the horizon and the
stage kind, and a v4 row's risk coefficient; the planes do not read the
stage) are rewritten here with the encoder's arithmetic, and every row gets
the encoder's ``risk_beta`` and ``stage_state``, so both engines' rows are the same bytes.
"""

from dataclasses import replace
from functools import lru_cache
import math
import random

import numpy as np

from .statistics import seed_for


def start(cfg, seed):
    """The game's place in its stage, or None without cfg["stage"]."""
    spec = cfg.get("stage")
    if not spec:
        return None
    from mahjong_ai.domain.types import StageKind

    games = int(spec.get("games", 10))
    if games < 1:
        raise ValueError("a stage has at least one game")
    rng = random.Random(seed_for(seed, "stage"))
    game = rng.randrange(games)
    prior = zero_sum(rng, float(spec.get("per_game_sd", 70)) * math.sqrt(game))
    return {"kind": StageKind(spec["kind"]), "games": games, "game": game, "prior": prior}


def zero_sum(rng, sd):
    """Four integer scores summing to zero, each with standard deviation ``sd``."""
    if sd <= 0:
        return (0, 0, 0, 0)
    # Centring four draws keeps 3/4 of their variance.
    draws = [rng.gauss(0.0, sd * math.sqrt(4 / 3)) for _ in range(4)]
    mean = sum(draws) / 4
    centred = [d - mean for d in draws]
    scores = [math.floor(c) for c in centred]
    # Round to integers that still sum to zero: the largest remainders round up.
    for i in sorted(range(4), key=lambda i: scores[i] - centred[i])[: -sum(scores)]:
        scores[i] += 1
    return tuple(scores)


def attach(state, info):
    """The stage on a hand state, for critic features (copies of the hand keep it)."""
    state.stage_prior, state.stage_game, state.stage_kind = info["prior"], info["game"], info["kind"]


def context(info, scores, hand_no, rounds):
    """The stage fields of an observation in this game's hand ``hand_no`` (``scores``: the game's before it)."""
    from mahjong_ai.domain.settlement import score_ranks

    stage_scores = tuple(p + s for p, s in zip(info["prior"], scores))
    return {"kind": info["kind"], "stage_scores": stage_scores, "rank_by_seat": score_ranks(stage_scores),
            "stage_round_no": rounds * info["game"] + hand_no, "stage_rounds_total": rounds * info["games"]}


def observe(obs, info, scores, hand_no, rounds):
    """The observation as the tournament settlement hands it out in this stage."""
    return replace(obs, stage=replace(obs.stage, **context(info, scores, hand_no, rounds)))


def row_fields(info, focal):
    """The learner's row fields: the carried scores, its own first, and the game's index in the stage."""
    prior = info["prior"]
    return {"stage_prior": [prior[(focal + k) % 4] for k in range(4)], "stage_game": info["game"]}


def rewrite(row, info, scores, hand_no, rounds, seat):
    """Rewrite a row's stage-dependent scalars (a C row encodes the game alone) as the encoder does under the stage.

    The first scalars, and a v4 row's last one (guided.encoding.RISK); the
    row's ``risk_beta`` and ``stage_state`` are the encoder's (guided.encoding.encode).
    """
    from .encoding import RISK

    values, risk, state = _scalars(info["prior"], info["game"], info["games"], info["kind"], tuple(scores),
                                   hand_no, rounds, seat)
    scalars = np.array(row["scalars"], copy=True)
    scalars[: len(values)] = values.astype(scalars.dtype)
    if len(scalars) > RISK:
        scalars[RISK] = 0.0 if risk is None else risk
    row["scalars"] = scalars
    if risk is None:
        row.pop("risk_beta", None)
    else:
        row["risk_beta"] = risk
    if state is None:
        row.pop("stage_state", None)
    else:
        row["stage_state"] = state
    return row


@lru_cache(maxsize=4096)
def _scalars(prior, game, games, kind, scores, hand_no, rounds, seat):
    """The encoder's first scalars, risk_beta and stage_state under the stage; the same for a seat throughout a hand."""
    from mahjong_ai.domain.types import StageKind
    from selfplay_core.encoding import _midranks, symlog

    from .stage_value import risk_input, stage_state

    fields = context({"prior": prior, "game": game, "games": games, "kind": kind}, scores, hand_no, rounds)
    stage_scores = fields["stage_scores"]
    rel = [(seat + i) % 4 for i in range(4)]
    own = stage_scores[seat]
    ranks = _midranks(stage_scores)
    current, total = fields["stage_round_no"], fields["stage_rounds_total"]
    # selfplay_core.encoding's first scalars, term for term: score context,
    # horizon, tiebreak (never in training), stage kind.
    values = np.asarray([
        *(symlog(stage_scores[s] / 8.0) / 4.0 for s in rel),
        *(symlog((own - stage_scores[s]) / 8.0) / 4.0 for s in rel[1:]),
        (ranks[seat] - 1) / 3.0,
        *((ranks[s] - 1) / 3.0 for s in rel[1:]),
        min(1.0, max(0.0, (current - 1) / max(1, total - 1))),
        min(1.0, max(0, total - current) / 16.0),
        min(1.0, total / 20.0),
        0.0,
        *(float(fields["kind"] == k) for k in StageKind),
    ], dtype=np.float32)
    values.flags.writeable = False
    # The game's scores are in base scores: training plays at a base score of 1.
    return (values, risk_input(fields["kind"], stage_scores, seat, current, total),
            stage_state(fields["kind"], stage_scores, seat, current, total))
