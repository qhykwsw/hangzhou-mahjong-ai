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

Both engines play the copies the same, bit for bit: Branch here on the Python
engine, fastgame.FastBranch on the C engine.
"""

from types import SimpleNamespace

from .environment import GREEDY, Shadow, staged
from .statistics import seed_for

LEAF = ("planes", "scalars", "events", "mask")


def options(cfg, train):
    """A training game's branch options, or None without cfg["branches"]."""
    width = cfg.get("branches")
    if not width:
        stray = [key for key in ("branch_continuation", "branch_control") if cfg.get(key)]
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
    return SimpleNamespace(width=width, learner=GREEDY if learner == "greedy" else learner, swaps=swaps,
                           control=bool(cfg.get("branch_control")))


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
    world = game._branch_world(row)
    results = row["branches"] = []
    for index in candidates:
        result = {"action": int(index)}
        results.append(result)
        game._branch_copy(world, int(index), legal, result)
    if game.branching.control:
        result = row["branch_control"] = {"action": int(argmax)}
        game._branch_copy(world, int(argmax), legal, result)


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


class Branch(Shadow):
    """One candidate's copy of the hand on the Python engine (see the module docstring).

    ``plan`` says how each seat acts (seat_plan); ``row`` is the candidate's
    result, which settle() fills.
    """

    def __init__(self, game, state, action, legal, result, plan):
        self.game, self.state, self.row, self.plan = game, state, result, plan
        self.dealer, self.pending, self.done = game.dealer, None, False
        state.step(action, legal)

    def advance(self):
        from .encoding import encoder

        game, state = self.game, self.state
        encode = encoder(game.encoding)
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
