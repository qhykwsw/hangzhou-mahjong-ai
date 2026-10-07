"""Training opponents a little off the fixed pool: jittered parameters and timeouts.

Two cfg keys change a training game's rule seats, never the learner, a model
seat or an evaluation game (Game without train). Without them, with sd 0 or
mean 0, a game is bit for bit the one before they existed. Both draw from the
game's own streams (statistics.seed_for of its seed), so a game replays the
same on either engine.

cfg["opponent_jitter"] = {"sd": 0.1}: each seat of a race member (a fixed
pool member deciding whole in race_act, fastgame.race_member) plays the game
on its own copy of the member's race profile, every float constant times
1 + sd * N(0, 1), the normals in FLOATS' order from
seed_for(seed, "opponent-jitter", seat), then clipped to where the constant
means something: the probabilities q, q_late and draw_risk to [0, 0.95],
q_early to [0, q_late], the discount far_step to [0, 1], the rest (saved,
one_step, hunt, tie) at least 0. The integer and flag fields (upgrade,
last_wall, style, complete, guard) stay; draw_risk, which the members derive
from q, moves on its own. The copy decides in native.c on both engines
(native.Hand.run on the C hand, native.race_act on the Python engine's
Observation) under an id of its own, so the race memo never mixes copies.
fp-rule-131 (rule131.h) is not jittered: it binds one of nine profiles by
role and hand and decides with constants and tables outside them
(rule131_tables.h), so scaling the profiles would not be a jitter of the
strategy as a whole; nor are other rule seats, which have no profile. Members
the worker loaded before importing fastgame are not recognised and play as
frozen; the worker imports it first.

cfg["opponent_timeout"] = {"mean": 0.15, "shape": 2.0}: each rule seat draws a
timeout rate eps ~ Beta(mean * shape, (1 - mean) * shape) from
seed_for(seed, "opponent-timeout-rate", seat): shape (the concentration,
default 2) sets the tail: at 2 and mean 0.15 the median rate is 0.05, a third
of the seats stay under 0.01 and about one in eleven times out at more than
half its decisions. Each decision of
the seat with a choice (two or more legal actions) times out with
probability eps, the k-th drawing the k-th uniform of the seat's stream in
the hand (splitmix64 of seed_for(seed, "opponent-timeout", hand, seat) +
(k + 1) * golden, 53 bits; ``uniform``). A decision out of time plays the
platform's own move, as the platform guide states it (docs/sources/
platform-guide.html, DiscardTimeoutSec: win when it can, else discard the
rightmost tile; call windows pass) and domain.engine's
HandState.timeout_action implements it: a call window passes, a turn wins
when it can, else lets go the last tile of the hand in its order, the tile
just drawn (after a chi or peng, the last tile the hand kept). The rule
does not act on it and does not see it (its act is not called). The
platform's replays agree: in the 6268 indexed games (guided.platform_dataset)
all 30458 discard timeouts right after a draw let go the drawn tile, and
replayed in the engine (3000 of them) all 529 right after a chi or peng the
last tile of the hand.
"""

import math
import random

from .statistics import seed_for

FLOATS = ("q", "saved", "one_step", "far_step", "q_early", "q_late", "hunt", "draw_risk", "tie")
PROBABILITY_CAP = 0.95
SHAPE = 2.0
MASK = (1 << 64) - 1


def settings(cfg):
    """(sd, (mean, shape)) of cfg's opponent keys, each None when off; malformed keys raise."""
    jitter, timeout = cfg.get("opponent_jitter"), cfg.get("opponent_timeout")
    sd = None
    if jitter:
        if not isinstance(jitter, dict) or set(jitter) - {"sd"}:
            raise ValueError("opponent_jitter is {'sd': float}")
        sd = float(jitter.get("sd", 0.0))
        if not (math.isfinite(sd) and sd >= 0):
            raise ValueError("opponent_jitter sd must be finite and >= 0")
        sd = sd or None
    rate = None
    if timeout:
        if not isinstance(timeout, dict) or set(timeout) - {"mean", "shape"}:
            raise ValueError("opponent_timeout is {'mean': float, 'shape': float}")
        mean, shape = float(timeout.get("mean", 0.0)), float(timeout.get("shape", SHAPE))
        if not (0 <= mean < 1 and math.isfinite(shape) and shape > 0):
            raise ValueError("opponent_timeout needs 0 <= mean < 1 and shape > 0")
        rate = (mean, shape) if mean > 0 else None
    return sd, rate


def plan(cfg, seed, specs, policies):
    """A training game's opponents: ({seat: (jittered profile id, fallback)}, four timeout rates or None)."""
    sd, rate = settings(cfg)
    jitter = {}
    if sd:
        from .fastgame import native, race_member

        for seat, policy in sorted(policies.items()):
            member = race_member(policy)
            if member is None:
                continue
            profile, fallback = member
            values = jittered(native.race_floats(profile), sd, random.Random(seed_for(seed, "opponent-jitter", seat)))
            jitter[seat] = (native.jitter_profile(profile, values), fallback)
    rates = None
    if rate:
        mean, shape = rate
        rates = tuple(
            random.Random(seed_for(seed, "opponent-timeout-rate", seat)).betavariate(mean * shape, (1 - mean) * shape)
            if not spec.startswith("model:") else 0.0
            for seat, spec in enumerate(specs)
        )
    return jitter, rates


def jittered(values, sd, rng):
    """A profile's floats (FLOATS' order) each times 1 + sd * N(0, 1), clipped (see the module docstring)."""
    q, saved, one_step, far_step, q_early, q_late, hunt, draw_risk, tie = (
        v * (1.0 + sd * rng.gauss(0.0, 1.0)) for v in values)
    q_late = min(PROBABILITY_CAP, max(0.0, q_late))
    return (
        min(PROBABILITY_CAP, max(0.0, q)), max(0.0, saved), max(0.0, one_step), min(1.0, max(0.0, far_step)),
        min(q_late, max(0.0, q_early)), q_late, max(0.0, hunt), min(PROBABILITY_CAP, max(0.0, draw_risk)),
        max(0.0, tie),
    )


def jittered_act(obs, profile, fallback):
    """A jittered member's move on an Observation (native.race_act)."""
    from .fastgame import native

    return obs.legal_actions[native.race_act(obs, profile, fallback)]


def hand_keys(seed, hand_no):
    """The four seats' timeout stream keys in one hand of a game."""
    return tuple(seed_for(seed, "opponent-timeout", hand_no, seat) for seat in range(4))


def arm(state, rates, keys):
    """A Python hand's timeouts (native.Hand.timeouts on the C engine); fork copies them with the hand."""
    state.timeout_rates, state.timeout_keys, state.timeout_counts = tuple(rates), tuple(keys), [0, 0, 0, 0]


def uniform(key, k):
    """native.c's g_uniform: the k-th uniform of a stream, splitmix64 of key + (k + 1) * golden, 53 bits."""
    z = (key + (k + 1) * 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z ^= z >> 31
    return (z >> 11) * (1.0 / 9007199254740992.0)


def timed_out(state, seat):
    """Whether the seat's decision now times out (each decision of a seat with a rate draws once)."""
    eps = state.timeout_rates[seat]
    if not eps > 0.0:
        return False
    k = state.timeout_counts[seat]
    state.timeout_counts[seat] = k + 1
    return uniform(state.timeout_keys[seat], k) < eps


def default_action(state, legal):
    """The platform's move for a player out of time (HandState.timeout_action, here for every runtime)."""
    if state.phase.startswith("response_"):
        return next(a for a in legal if a.action == "pass")
    hu = next((a for a in legal if a.action == "hu"), None)
    if hu is not None:
        return hu
    for tile in reversed(state.hands[state.turn]):
        action = next((a for a in legal if a.action == "discard" and a.tile == tile), None)
        if action is not None:
            return action
    raise ValueError("no timeout action")
