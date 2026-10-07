"""Oracle guiding (ADR-0018): the learner's Actor reads the hidden state during training collection only.

Suphx-style: an Actor with the ``oracle_inputs`` option (guided.networks.Actor)
takes the hidden state as an extra input, starting at zero weight and trained
by the recipe's policy gradient; the recipe lowers the share of it kept, from 1
to 0, until the policy no longer depends on it. ADR-0009 lets only the Critic
read hidden state; ADR-0018 extends that to this Actor input during training,
with the guarantees that make the released policy legal: every export strips
the input and its parameters (guided.trial.export_actor), an Actor fed no
oracle computes bit for bit what the stripped one computes, and every
inference loader refuses an Actor that has it (guided.deployment.load_actor,
guided.model_server), so evaluation, free play and the competition never see it.

    cfg["oracle_inputs"] = {"keep": γ}                          # 0 <= γ <= 1
    cfg["oracle_inputs"] = {"keep": γ, "continuations": False}

absent or None is off, and so is γ = 0: then nothing is computed, no row or
request carries the fields and the learner decides with oracle=None, exactly as
the stripped Actor. Otherwise, at every decision of the learner's seat in a
training game the worker computes the oracle features of that state
(guided.encoding.oracle; the C engine's Hand.oracle, bit for bit the same):

    planes [16, 34]: per tile, the three opponents' concealed counts >= 1..4
                     (in turn order after the learner), then the wall's
    scalars [4]:     each opponent's (8 - shanten) / 9, the wall's size / 136

and keeps each of the 548 values with probability γ, else 0 (an element-wise
Bernoulli mask, not rescaled), drawn from the decision's own random stream
seed_for(game seed, "oracle", hand, k), k the learner's decisions in the hand
before this one along the path. The masked values, float16, ride with the
inference request, the parent feeds them to the learner (and only to it), and
they stay on the row:

    row["oracle_input_planes"]   float16 [16, 34]
    row["oracle_input_scalars"]  float16 [4]

so a recipe's gradient sees exactly the input its behaviour policy saw.

"The learner's seat" is the sampling learner (model:current) and, unless
``continuations`` is False, its greedy continuations (GREEDY) in the playout
baseline's deviation copies and in the branches' copies whose learner
continues greedily: the same policy, so a deviation's value and the game's own
path are values of one policy. A copy's i-th learner decision draws with
k = (the forked decision's k) + i: as long as a copy follows the game's own
path it draws the game's masks (common random numbers for the vine pairing).
Opponents (model or rule), teachers (model teachers in the parent, rule
teachers in the worker) and evaluation never get it: rows of other seats
carry no oracle fields and the parent passes none to any other model.

A recipe's own batches feed them back with ``stack`` (or learning.make_batch
and ``argument``):

    oracle = stack(rows)            # (planes [n, 16, 34], scalars [n, 4]) float16 arrays, or None
    logits = actor(planes, scalars, events, oracle=None if oracle is None else tuple(
        torch.as_tensor(x, device=device) for x in oracle))
"""

import numpy as np

from .statistics import seed_for

PLANES, SCALARS = "oracle_input_planes", "oracle_input_scalars"
FIELDS = (PLANES, SCALARS)
SIZE = 16 * 34 + 4


def options(cfg, train=True):
    """A training game's oracle options (keep, continuations), or None when off (absent, None, keep 0)."""
    spec = cfg.get("oracle_inputs")
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {"keep", "continuations"} or "keep" not in spec:
        raise ValueError("cfg['oracle_inputs'] is {'keep': γ in [0, 1], 'continuations': True or False}")
    keep = spec["keep"]
    if isinstance(keep, bool) or not isinstance(keep, (int, float)) or not 0.0 <= keep <= 1.0:
        raise ValueError("cfg['oracle_inputs']['keep'] is a number in [0, 1]")
    continuations = spec.get("continuations", True)
    if type(continuations) is not bool:
        raise ValueError("cfg['oracle_inputs']['continuations'] is True or False")
    if not train or keep == 0:
        return None
    return float(keep), continuations


def masked(features, keep, seed):
    """The row fields of oracle ``features`` (oracle_planes, oracle_scalars) each kept with probability ``keep``."""
    planes = np.asarray(features["oracle_planes"], dtype=np.float32).reshape(16, 34)
    scalars = np.asarray(features["oracle_scalars"], dtype=np.float32).reshape(4)
    if keep < 1.0:
        kept = np.random.default_rng(seed).random(SIZE) < keep
        planes = np.where(kept[:544].reshape(16, 34), planes, 0.0)
        scalars = np.where(kept[544:], scalars, 0.0)
    return {PLANES: planes.astype(np.float16), SCALARS: scalars.astype(np.float16)}


def turn(rows, hand):
    """The learner's decisions so far in ``hand``: the trailing rows of that hand."""
    count = 0
    for row in reversed(rows):
        if row["hand"] != hand:
            break
        count += 1
    return count


def fields(game, features, hand, k):
    """The masked oracle fields of the learner's decision ``k`` (from 0) in ``hand`` of ``game``."""
    keep, _ = game.oracle
    return masked(features, keep, seed_for(game.seed, "oracle", hand, k))


def decision(game, row, features):
    """The game's own learner decision: its masked fields onto ``row`` (``features()`` gives the oracle's).

    Sets game.oracle_turn, the decision's k, from which the copies forked at
    this decision count on (continuation).
    """
    game.oracle_turn = turn(game.rows, game.hand_no)
    row.update(fields(game, features(), game.hand_no, game.oracle_turn))


def continuation(copy, row, hand, features):
    """A copy's greedy learner decision in ``hand``: the next k after the copy's last one."""
    copy.oracle_turn += 1
    row.update(fields(copy.game, features(), hand, copy.oracle_turn))


def stack(rows):
    """The rows' oracle inputs as float16 arrays (planes [n, 16, 34], scalars [n, 4]), zero where a row has
    none (exactly as no input), or None when no row has them."""
    if not any(PLANES in row for row in rows):
        return None
    planes = np.zeros((len(rows), 16, 34), dtype=np.float16)
    scalars = np.zeros((len(rows), 4), dtype=np.float16)
    for i, row in enumerate(rows):
        if PLANES in row:
            planes[i] = row[PLANES]
            scalars[i] = row[SCALARS]
    return planes, scalars


def argument(batch):
    """An Actor's ``oracle`` argument from a batch with the oracle fields (learning.make_batch), else None."""
    if PLANES not in batch:
        return None
    return batch[PLANES], batch[SCALARS]
