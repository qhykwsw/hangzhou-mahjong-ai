"""Branches of a learning decision (cfg["branches"]): candidates the recipe names, each played out in a copy.

    cfg["branches"] = K      # at most K candidates per learning row; needs cfg["playout_baseline"]

In the collection parent, after the learner's move is chosen, the recipe's
hook ``hooks["branches"](logits, mask, actions)`` returns each learning row's
candidates, an (n, K) integer tensor padded with -1 (fewer columns are padded,
more are an error). The worker forks the hand at that decision once per
candidate, plays the candidate in the copy and continues it to the end of the
hand: the learner's seat with the current Actor's argmax (GREEDY), every other
seat as in the game (rules deciding as they do, model opponents with their
argmax under the playout baseline), on the game's own wall. Each copy is an
independent continuation; the game itself and the playout baseline's single
deviation copy (Shadow) are unchanged. The row then carries

    branches = [{"action", "truncated": False, "playout", "playout_scores",
                 "playout_dealer", "playout_deals_next",
                 "playout_luck_pay", "playout_luck_deal" (with cfg["luck"])}, ...]

in the candidates' order, the fields as a deviation row's (Shadow): the
learner's hand score in base-score units, the four scores after the hand by
absolute seat, who deals next and whether the learner does, and the draw luck
at the copy's end. Rows without a candidate have no ``branches``. The game is
finished (finish_game) only once every copy has played out.

    cfg["branch_continuation"] = {"learner": "greedy" | "<spec>", "opponents": {"<spec>": "<spec>"} | None}
    cfg["branch_control"] = True

The continuation is how the copies play on after the candidate. ``learner``
is the learner's seat: "greedy" (the default, the current Actor's argmax) or
a seat spec, a rule version (e.g. "fp-rule-131": a race member or
fp-rule-131 then decides in C, any other rule in Python) or a "model:" slot
the parent answers with its argmax. ``opponents`` replaces an opponent seat's
spec, as the game's seats hold it, by another (an abstract or cheaper
opponent); seats it does not name play as in the game. A rule version comes
from the bundle or the trial's versions (the worker's External); every seat
of a continuation gets its own policy, reset from the game's seed. The trial
(ctx.collect) takes the recipe's IDs, as cfg["opponents"] names them, and
gives the workers seat specs. "model:current" (the sampling learner) is not a
continuation.

With branch_control, a row with candidates also gets ``branch_control``, its
argmax played the same way (same fork, same continuation), so a candidate's
difference to the control is paired: same wall, same seats.

    cfg["branch_truncate"] = k

Each copy (the control too) then stops at the learner's k-th decision after
the candidate (a decision: a turn with more than one legal action, whoever
plays the learner's seat) instead of playing the hand out, unless the hand
ends first. A stopped copy's result is

    {"action", "truncated": True, "playout_scores", "leaf_row",
     "playout_luck_pay", "playout_luck_deal" (with cfg["luck"])}

``playout_scores`` the four scores so far by absolute seat (a hand pays only
when it ends, so the scores before it), ``leaf_row`` the learner's row at the
stop (its planes, scalars, events and mask: the legal observation's inputs,
encoded as the game's rows are), the luck so far. A recipe values the leaf
with its own value network in the parent; the framework does not.

    cfg["branch_resample"] = {"weights": None | "module:function"}

Each row's copies (its candidates and its control) then start from one world
re-dealt from the learner's point of view instead of the real hand: the
other three seats' concealed tiles and the remaining wall are pooled, sorted
(so the world depends on the learner's unseen multiset alone, not on who
really holds what) and shuffled by the row's own random stream
(seed_for(game seed, "branch-world", row index)); each seat gets back as many
as it held, the rest is the wall, as long as it was. The learner's hand and
everything public stay: melds (the others' concealed kongs too, kept whole:
the pool is the learner's unseen tiles less those kongs' tiles), rivers, hand
sizes, the white tile's state (catch-play circle, chain and piao counts),
the scores and the hand's events. The others' past private events (their
deal and draws) still name the real tiles; a rule reading only its current
hand and the public state (race members, fp-rule-131) does not see them.

A stable wall (cfg["stable_wall"]) keeps its structure: the re-dealt tiles
fill the remaining positions in wall order, so from the fork on every seat
draws from its own positions as before and a row's copies stay paired with
each other; what lies at those positions is the re-dealt world's, not the
real wall's. The draw luck accumulates on as before, its expectations over
the re-dealt wall.

``weights`` (resolved in the worker) is called once per re-dealt world as
f(row, world) with the learner's row inputs (planes, scalars, events, mask)
and world = {"seat", "hands" (four tuples by absolute seat, the learner's
own included), "wall" (in draw order), "opened" (melds per seat)}; it returns
the world's weight, e.g. a belief model's likelihood of the others' hands.
Every result of the row carries it as ``world_weight`` (1.0 without
``weights``). Both engines draw the same world from the same stream, so they
stay bit for bit the same.

Both engines play the copies the same, bit for bit: Branch here on the Python
engine, fastgame.FastBranch on the C engine.
"""

import random
from types import SimpleNamespace

from .environment import GREEDY, Shadow, hook, staged
from .statistics import seed_for

LEAF = ("planes", "scalars", "events", "mask")


def options(cfg, train):
    """A training game's branch options, or None without cfg["branches"]."""
    width = cfg.get("branches")
    if not width:
        stray = [key for key in ("branch_continuation", "branch_control", "branch_truncate") if cfg.get(key)]
        stray += ["branch_resample"] if cfg.get("branch_resample") not in (None, False) else []
        if stray:
            raise ValueError(f"{stray} need cfg['branches']")
        return None
    if not train:
        return None
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("cfg['branches'] is the most candidates per row, a positive integer")
    if not cfg.get("playout_baseline"):
        raise ValueError("branches need cfg['playout_baseline'] (model opponents answer with their argmax)")
    continuation = cfg.get("branch_continuation") or {}
    if not isinstance(continuation, dict) or set(continuation) - {"learner", "opponents"}:
        raise ValueError("cfg['branch_continuation'] is {'learner': spec, 'opponents': {spec: spec} or None}")
    learner = continuation.get("learner") or "greedy"
    swaps = dict(continuation.get("opponents") or {})
    for spec in (learner, *swaps, *swaps.values()):
        if not isinstance(spec, str) or not spec:
            raise ValueError("branch continuation specs are strings")
    if "model:current" in (learner, *swaps.values()):
        raise ValueError("the sampling learner (model:current) cannot continue a branch")
    truncate = cfg.get("branch_truncate") or 0
    if isinstance(truncate, bool) or not isinstance(truncate, int) or truncate < 0:
        raise ValueError("cfg['branch_truncate'] is the learner's decision a copy stops at, a positive integer")
    resample = cfg.get("branch_resample")
    if resample is True:
        resample = {}
    if resample not in (None, False) and (not isinstance(resample, dict) or set(resample) - {"weights"}):
        raise ValueError("cfg['branch_resample'] is {'weights': None or 'module:function'}")
    return SimpleNamespace(width=width, learner=GREEDY if learner == "greedy" else learner, swaps=swaps,
                           control=bool(cfg.get("branch_control")), truncate=truncate,
                           resample=resample not in (None, False),
                           weights=(resample or {}).get("weights") if isinstance(resample, dict) else None)


def seat_plan(game):
    """How each seat acts in the game's branch copies: ("model", slot) asks the parent, ("rule", policy) decides."""
    opts = game.branching
    plan = []
    for seat, spec in enumerate(game.specs):
        target = opts.learner if seat == game.focal else opts.swaps.get(spec, spec)
        if target.startswith("model:"):
            plan.append(("model", target))
        elif seat != game.focal and target == spec:
            plan.append(("rule", game.policies[seat]))
        else:
            plan.append(("rule", policy(game, target, seat)))
    return plan


def policy(game, spec, seat):
    """A continuation's own policy of a rule version for a seat (as Game builds its seats')."""
    from .artifacts import build_policy

    entries, external = game._sources
    if external is not None and spec in external.specs:
        built = external.acquire(spec)
        game.borrowed.append((spec, built))
    elif spec in entries:
        built = build_policy(entries[spec], spec)
    else:
        raise ValueError(f"unknown branch continuation: {spec}")
    built.reset(seed_for(game.seed, "branch-policy", seat))
    return built


def start(game, row, legal, candidates, argmax):
    """The copies of a learning decision's candidates (and its control), each writing its result into the row."""
    world, weight = game._branch_world(row)
    results = row["branches"] = []
    moves = [(int(index), results) for index in candidates]
    if game.branching.control:
        moves.append((int(argmax), None))
    for index, into in moves:
        result = {"action": index}
        if weight is not None:
            result["world_weight"] = weight
        if into is None:
            row["branch_control"] = result
        else:
            into.append(result)
        game._branch_copy(world, index, legal, result)


def world_seed(game):
    """The random stream of the world a row's copies share: by the game's seed and the row's index."""
    return seed_for(game.seed, "branch-world", len(game.rows))


def redeal(hands, wall, focal, seed):
    """The hidden tiles re-dealt (see the module docstring): (four hands, wall), the learner's hand as it was."""
    others = [s for s in range(4) if s != focal]
    pool = sorted([t for s in others for t in hands[s]] + list(wall))
    random.Random(seed).shuffle(pool)
    dealt, at = [list(h) for h in hands], 0
    for s in others:
        dealt[s] = pool[at:at + len(hands[s])]
        at += len(hands[s])
    return dealt, pool[at:]


def place(state, hands, wall, focal):
    """A Python HandState's hidden tiles replaced by a redeal's (native.Hand.redeal for the C hand)."""
    for s in range(4):
        if s != focal:
            state.hands[s][:] = hands[s]
    state.wall[:] = wall


def weigh(game, row, hands, wall, opened):
    """The re-dealt world's weight: cfg["branch_resample"]["weights"](row inputs, world), else 1.0."""
    spec = game.branching.weights
    if not spec:
        return 1.0
    world = {"seat": game.focal, "hands": [tuple(h) for h in hands], "wall": tuple(wall), "opened": tuple(opened)}
    return float(hook(spec, None)({key: row[key] for key in LEAF}, world))


def finished(result, game, scores, delta, winner, dealer):
    """A copy's hand settled: the learner's hand score and the next hand's start (as Shadow.settle writes them)."""
    base = game.config.base_score
    # The winner deals next; after a draw the dealer stays.
    dealer = winner if winner is not None else dealer
    result["truncated"] = False
    result["playout"] = delta[game.focal] / base
    result["playout_deals_next"] = float(dealer == game.focal)
    result["playout_dealer"] = dealer
    result["playout_scores"] = [(s + d) / base for s, d in zip(scores, delta)]


def truncated(result, game, scores, delta, row):
    """A copy stopped at the learner's k-th later decision: the scores so far and the learner's row there."""
    base = game.config.base_score
    result["truncated"] = True
    result["playout_scores"] = [(s + d) / base for s, d in zip(scores, delta)]
    result["leaf_row"] = {key: row[key] for key in LEAF}


class Branch(Shadow):
    """One candidate's copy of the hand on the Python engine (see the module docstring).

    ``plan`` says how each seat acts (seat_plan); ``row`` is the candidate's
    result, which settle() fills.
    """

    def __init__(self, game, state, action, legal, result, plan):
        self.game, self.state, self.row, self.plan = game, state, result, plan
        self.dealer, self.pending, self.done = game.dealer, None, False
        self.decisions = 0  # the learner's decisions since the candidate, for cfg["branch_truncate"]
        state.step(action, legal)

    def advance(self):
        from .encoding import encoder

        game, state = self.game, self.state
        encode = encoder(game.encoding)
        cut = game.branching.truncate
        while not state.settled:
            if state.steps >= game.config.max_actions_per_hand:
                raise RuntimeError("hand action budget exceeded")
            legal = state.legal_actions()
            if len(legal) == 1:
                state.step(legal[0], legal)
                continue
            seat = state.turn
            obs = state.observation(legal)
            if game.stage:
                obs = staged(obs, game.stage, state, game.config.rounds)
            if cut and seat == game.focal:
                self.decisions += 1
                if self.decisions == cut:
                    self.done, self.state = True, None
                    truncated(self.row, game, state.scores, state.delta, encode(obs, game.cfg["lookahead"]))
                    if game.luck:
                        self.row["playout_luck_pay"], self.row["playout_luck_deal"] = state.luck
                    return None
            kind, who = self.plan[seat]
            if kind == "rule":
                action = who.act(obs)
                if action not in legal:
                    raise ValueError("frozen opponent returned an illegal action")
                state.step(action, legal)
                continue
            self.pending = legal
            row = encode(obs, game.cfg["lookahead"])
            row["teacher"] = -1
            return {"row": row, "slot": who, "train": False, "bc": False}
        self.done, self.state = True, None
        self.settle(state)
        return None

    def settle(self, state):
        game = self.game
        finished(self.row, game, state.scores, state.delta, state.winner, self.dealer)
        if game.luck:
            self.row["playout_luck_pay"], self.row["playout_luck_deal"] = state.luck
