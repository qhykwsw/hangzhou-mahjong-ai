"""The hand-outcome head's bins and loss, and the stage re-rank that reads it.

Outcome head (networks.Actor option ``"outcome_head": {"bins": [...]}``):
for every action, logits over B bins of the learner's score change for the
rest of the current hand, in base-score units. A score belongs to the bin
whose value is nearest (a midpoint goes to the upper bin; scores beyond the
ends to the end bins): ``bin_index``. Hangzhou settlement (rules.payments)
makes a hand's score one of 0 (a drawn hand: nobody pays), -F or -8F
(paying a non-dealer winner, or paying when the dealer is on either side),
+10F (winning as a non-dealer: 8F + F + F) or +24F (as the dealer), F a
power of two; DEFAULT_BINS holds every such score from -512 to +768, so up
to F = 64 (32 for a dealer's win) each score is its own bin. A recipe trains
the head with ``outcome_loss`` (cross-entropy of the taken action's bin).

Stage re-rank (an artifact's meta.json ``"outcome_rerank": {...}``, read by
the model service, deployment's HTTP service and their loaders): in a
semifinal (top two advance) or a final (first only), among the policy's own
near-ties, play the action whose predicted outcome gives the better chance of
finishing in those places. A row's stage state (``StageState``: places, the
hands left with this one, the four stage totals in base scores from the
learner in turn order, the dealer) gives, for each bin value v, the chance
``place_chance`` of finishing in place if this hand ends at v. The hand's
settlement moves the other three as well, and given v and the dealer it is
one of a few cases (``_scenarios``), each zero-sum:

  - v > 0, the learner won: the payments themselves, each -v/3 when the
    learner deals, else -0.8 v for the dealer and -0.1 v for the other two;
  - v = 0: a drawn hand, nobody moves;
  - v < 0, the learner paid |v|: as the dealer (8F) to one of the three, who
    gains 10F while the other two pay F (each a third); as a non-dealer, a
    payment of 8 or more is taken as the dealer's win (24F, F = |v|/8: the
    other two also pay |v|), a smaller one as a non-dealer's (10F, F = |v|:
    the dealer pays 8|v|, the remaining non-dealer |v|; either winner, a half
    each); fans are steep, so the smaller F of the two readings is the likely one.

Then the remaining n' = hands_left - 1 hands are stage_value.advance's model
(independent normals of sd^2 n' for all four, P(first) or P(top two) by
Gauss-Hermite quadrature, exact places with no hands left, a tie straddling
the line counting half), averaged over the cases.

``sd`` is per_hand_sd (stage_value's 17.5 per hand by default). An action's
value is its predicted distribution's expectation of that chance
(softmax(outcome logits) . u). ``choose`` restricts the choice to the
policy's near-ties: among the ``top_k`` most probable legal actions, those
before which the cumulative probability is still below ``mass`` and whose
log-probability is within ``margin`` of the best; the policy's argmax stays
unless one of them is worth more than ``min_gain`` above it. Without a stage
(free play, qualifiers, an artifact without the switch) a row's values are
all zero and the decision is the policy's, bit for bit.
"""

from functools import lru_cache
import math
import sys
from typing import NamedTuple

import numpy as np

from . import bootstrap  # noqa: F401

# Every hand score of Hangzhou settlement up to F = 64 (a dealer's win up to F = 32), in base scores.
DEFAULT_BINS = (-512.0, -256.0, -128.0, -64.0, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0,
                10.0, 20.0, 24.0, 40.0, 48.0, 80.0, 96.0, 160.0, 192.0, 320.0, 384.0, 640.0, 768.0)
PER_HAND_SD = 17.5  # stage_value.advance's per-hand spread, points per base score
RERANK_DEFAULTS = {"top_k": 3, "mass": 0.9, "margin": 1.0, "min_gain": 0.0, "per_hand_sd": PER_HAND_SD}


def outcome_bins(spec):
    """The bin values of an ``outcome_head`` option (true: DEFAULT_BINS; {"bins": [...]}), as a list of floats.

    At least two finite values, strictly increasing.
    """
    if spec is True:
        return list(DEFAULT_BINS)
    if not isinstance(spec, dict) or set(spec) - {"bins"}:
        raise ValueError('outcome_head must be true or {"bins": [...]}')
    bins = spec.get("bins", DEFAULT_BINS)
    if isinstance(bins, (str, bytes)) or not hasattr(bins, "__len__"):
        raise ValueError("outcome_head bins must be a list of numbers")
    values = []
    for value in bins:
        if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("outcome_head bins must be finite numbers")
        values.append(float(value))
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("outcome_head needs at least two strictly increasing bins")
    return values


def bin_edges(bins=DEFAULT_BINS):
    """The B - 1 boundaries between the bins: midpoints of neighbouring values."""
    values = np.asarray(bins, dtype=np.float64)
    return (values[1:] + values[:-1]) / 2


def bin_index(scores, bins=DEFAULT_BINS):
    """The bin of each score (base-score units): the nearest bin value's, a midpoint going to the upper one.

    A number gives an int, a numpy array an array, a torch tensor a long
    tensor on its device (the outcome head's class targets).
    """
    edges = bin_edges(bins)
    torch = sys.modules.get("torch")  # a tensor means torch is loaded
    if torch is not None and isinstance(scores, torch.Tensor):
        boundaries = torch.as_tensor(edges, dtype=torch.float64, device=scores.device)
        return torch.bucketize(scores.to(torch.float64), boundaries, right=True)
    index = np.searchsorted(edges, np.asarray(scores, dtype=np.float64), side="right")
    return int(index) if np.ndim(index) == 0 else index


def outcome_loss(outcome_logits, actions, scores, bins=DEFAULT_BINS, weights=None):
    """Mean cross-entropy of the actions' predicted outcomes against the scores that followed them.

    ``outcome_logits`` [n, A, B] (Actor(..., return_outcome=True)),
    ``actions`` [n] (the action whose hand ``scores`` [n] tells: the taken one
    on a main-path row, a deviation on its continuation's copy), optional
    ``weights`` [n] (a weighted mean). Only the chosen action's logits get a
    gradient.
    """
    import torch.nn.functional as F

    picked = outcome_logits.gather(1, actions.long()[:, None, None].expand(-1, 1, outcome_logits.shape[-1]))
    loss = F.cross_entropy(picked[:, 0].float(), bin_index(scores, bins), reduction="none")
    if weights is None:
        return loss.mean()
    weights = weights.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-12)


# -- stage states ------------------------------------------------------------


class StageState(NamedTuple):
    """What the re-rank reads of a semifinal or final decision.

    ``places``: 2 (semifinal: the top two advance) or 1 (final); ``hands_left``:
    the stage's hands from this one on (at least 1); ``scores``: the four stage
    totals in base scores, the learner's first, then the others in turn order;
    ``dealer``: the dealer's seat relative to the learner (0: the learner), or
    None when unknown.
    """

    places: int
    hands_left: int
    scores: tuple
    dealer: int | None = None


def stage_of_row(row):
    """A row's StageState, or None: its ``stage_state`` (guided.encoding.encode, stage.rewrite,
    selfplay_core.wire) and the dealer from its scalars (selfplay_core.encoding's seats)."""
    state = row.get("stage_state")
    if state is None:
        return None
    from .encoding import _SEATS

    places, left, *scores = state
    dealer = np.asarray(row["scalars"][_SEATS:_SEATS + 4], dtype=np.float32)
    seat = int(dealer.argmax()) if dealer.max() > 0.5 else None
    return StageState(int(places), int(left), tuple(float(s) for s in scores), seat)


def _scenarios(value, dealer):
    """[(weight, (d1, d2, d3))]: the other three's changes in this hand when the learner's is ``value``.

    ``dealer`` relative to the learner (0: the learner), or None when unknown
    (then the payments' mean, a third each). See the module for the cases.
    """
    others = (1, 2, 3)
    if value == 0:
        return [(1.0, (0.0, 0.0, 0.0))]
    if value > 0:
        if dealer in others:  # a non-dealer's 10F: the dealer pays 8F, the other two F
            return [(1.0, tuple(-0.8 * value if j == dealer else -0.1 * value for j in others))]
        return [(1.0, (-value / 3.0,) * 3)]
    paid = -value
    if dealer == 0:  # the learner deals and paid 8F to one of the three (10F); the other two paid F
        return [(1.0 / 3.0, tuple(1.25 * paid if j == winner else -paid / 8.0 for j in others)) for winner in others]
    if dealer in others:
        if paid >= 8:  # the dealer won 24F, F = paid / 8: every non-dealer paid 8F
            return [(1.0, tuple(3.0 * paid if j == dealer else -paid for j in others))]
        # A non-dealer won 10F, F = paid: the dealer paid 8F, the remaining non-dealer F.
        return [(0.5, tuple(10.0 * paid if j == winner else -8.0 * paid if j == dealer else -paid for j in others))
                for winner in others if winner != dealer]
    return [(1.0, (paid / 3.0,) * 3)]


@lru_cache(maxsize=8192)
def _place_chance(places, hands_left, scores, dealer, bins, sd):
    from .stage_value import advance

    rows, weights, index = [], [], []
    for b, value in enumerate(bins):
        for weight, shares in _scenarios(value, dealer):
            rows.append([scores[0] + value, *(s + d for s, d in zip(scores[1:], shares))])
            weights.append(weight)
            index.append(b)
    chance = advance(np.asarray(rows), max(0, hands_left - 1), places, sd)
    out = np.bincount(index, weights=np.asarray(weights) * chance, minlength=len(bins)).astype(np.float32)
    out.flags.writeable = False
    return out


def place_chance(state, bins=DEFAULT_BINS, per_hand_sd=PER_HAND_SD):
    """[B] float32: the chance of finishing within ``state.places`` if this hand ends at each bin's value."""
    return _place_chance(int(state.places), int(state.hands_left), tuple(float(s) for s in state.scores),
                         None if state.dealer is None else int(state.dealer), tuple(float(b) for b in bins),
                         float(per_hand_sd))


def utilities(states, bins=DEFAULT_BINS, per_hand_sd=PER_HAND_SD):
    """[n, B] float32 of place_chance per row; a zero row for a row without a stage (None)."""
    out = np.zeros((len(states), len(bins)), dtype=np.float32)
    for i, state in enumerate(states):
        if state is not None:
            out[i] = place_chance(state, bins, per_hand_sd)
    return out


# -- the decision --------------------------------------------------------------


def rerank_options(spec):
    """The normalized ``outcome_rerank`` of an artifact's meta.json, or None when absent or false.

    true takes RERANK_DEFAULTS; a dict overrides some of them: ``top_k`` (an
    int >= 1), ``mass`` in (0, 1], ``margin`` >= 0 (nats), ``min_gain`` >= 0
    (chance of the target places), ``per_hand_sd`` > 0 (points per base score).
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict) or set(spec) - set(RERANK_DEFAULTS):
        raise ValueError(f"outcome_rerank must be true or a dict of {sorted(RERANK_DEFAULTS)}")
    options = dict(RERANK_DEFAULTS, **spec)
    for key, value in options.items():
        if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"outcome_rerank {key} must be a finite number")
    if type(options["top_k"]) is not int or options["top_k"] < 1:
        raise ValueError("outcome_rerank top_k must be an integer of at least 1")
    if not 0 < options["mass"] <= 1 or options["margin"] < 0 or options["min_gain"] < 0 or options["per_hand_sd"] <= 0:
        raise ValueError("outcome_rerank needs 0 < mass <= 1, margin >= 0, min_gain >= 0, per_hand_sd > 0")
    return {k: (int(v) if k == "top_k" else float(v)) for k, v in options.items()}


def choice_options(options):
    """The keyword arguments of ``choose`` from rerank_options."""
    return {k: options[k] for k in ("top_k", "mass", "margin", "min_gain")}


def choose(log_probs, outcome_logits, mask, utility, greedy, *, top_k, mass, margin, min_gain):
    """The re-ranked action per row [n] (torch, on the device, no host synchronization: graph-safe).

    ``log_probs`` [n, A]: the policy's masked log-softmax; ``outcome_logits``
    [n, A, B]; ``utility`` [n, B] (``utilities``: zero rows without a stage);
    ``greedy`` [n]: the policy's own choice, kept unless a near-tie candidate
    (see the module) is worth more than ``min_gain`` above it.
    """
    import torch

    k = min(int(top_k), log_probs.shape[-1])
    top, order = log_probs.topk(k, dim=-1)
    probs = top.exp()
    before = probs.cumsum(-1) - probs
    candidate = mask.gather(1, order) & (before < mass) & (top >= top[:, :1] - margin)
    bins = outcome_logits.shape[-1]
    utility = utility.float()
    dist = outcome_logits.float().gather(1, order[..., None].expand(-1, -1, bins)).softmax(-1)
    value = (dist * utility[:, None, :]).sum(-1).masked_fill(~candidate, -math.inf)
    own = outcome_logits.float().gather(1, greedy[:, None, None].expand(-1, 1, bins))[:, 0].softmax(-1)
    own = (own * utility).sum(-1)
    best = value.argmax(-1)
    gain = value.gather(1, best[:, None])[:, 0] - own
    return torch.where(gain > min_gain, order.gather(1, best[:, None])[:, 0], greedy)


def rerank(logits, outcome_logits, mask, stages, *, bins=DEFAULT_BINS, **options):
    """The chosen action per row [n]: the policy's argmax, re-ranked where a row has a stage.

    ``logits`` [n, A] and ``outcome_logits`` [n, A, B] of an Actor with the
    outcome head, ``mask`` [n, A] the legal actions, ``stages`` one StageState
    or None per row; ``options`` as rerank_options (the defaults where left
    out). Without any stage it is the masked argmax.
    """
    import torch
    import torch.nn.functional as F

    options = rerank_options(options)
    log_probs = F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
    greedy = log_probs.argmax(-1)
    if not any(s is not None for s in stages):
        return greedy
    utility = torch.as_tensor(utilities(stages, bins, options["per_hand_sd"]), device=logits.device)
    return choose(log_probs, outcome_logits, mask, utility, greedy, **choice_options(options))


# -- artifacts -------------------------------------------------------------------


def variant(source, root, options=None):
    """A copy of an Actor artifact (with the outcome head) whose meta.json turns the stage re-rank on.

    The weights are the source's (meta.json keeps their weights_sha256);
    the copy goes to <root>/<id>/, id the SHA-256 of the weights' hash and the
    normalized switch, so a match addresses it like any model and the source
    keeps deciding without it. Returns the copy's directory.
    """
    import hashlib
    import json
    from pathlib import Path
    import shutil
    import tempfile

    source, root = Path(source), Path(root)
    meta = json.loads((source / "meta.json").read_text())
    if not (meta.get("config") or {}).get("outcome_head"):
        raise ValueError("the stage re-rank needs an Actor with the outcome head (guided.expand --outcome-head)")
    if hashlib.sha256((source / "actor.pt").read_bytes()).hexdigest() != meta.get("weights_sha256"):
        raise ValueError("Actor weight checksum mismatch")
    switch = rerank_options(True if options is None else options)
    name = hashlib.sha256(json.dumps([meta["weights_sha256"], switch], sort_keys=True).encode()).hexdigest()
    target = root / name
    if target.exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rerank-", dir=root) as tmp:
        staged = Path(tmp) / "artifact"
        staged.mkdir()
        shutil.copy2(source / "actor.pt", staged / "actor.pt")
        (staged / "meta.json").write_text(json.dumps(dict(meta, outcome_rerank=switch), indent=2) + "\n")
        from .deployment import load_actor

        load_actor(staged, "cpu")  # the switch and the head load together
        staged.rename(target)
    return target


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Stage re-rank switch for an Actor with the outcome head.")
    sub = parser.add_subparsers(dest="command", required=True)
    on = sub.add_parser("variant", help="copy an artifact with the re-rank on to <root>/<id>/ and print it")
    on.add_argument("source")
    on.add_argument("root")
    for key, value in RERANK_DEFAULTS.items():
        on.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    if args.command == "variant":
        options = {key: getattr(args, key) for key in RERANK_DEFAULTS}
        target = variant(args.source, args.root, options)
        print(json.dumps({"path": str(target), "model": target.name}))


if __name__ == "__main__":
    main()
