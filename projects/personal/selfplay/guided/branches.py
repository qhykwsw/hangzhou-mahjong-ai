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

Both engines play the copies the same, bit for bit: Branch here on the Python
engine, fastgame.FastBranch on the C engine.
"""

from types import SimpleNamespace

from .environment import GREEDY, Shadow, staged

LEAF = ("planes", "scalars", "events", "mask")


def options(cfg, train):
    """A training game's branch options, or None without cfg["branches"]."""
    width = cfg.get("branches")
    if not train or not width:
        return None
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("cfg['branches'] is the most candidates per row, a positive integer")
    if not cfg.get("playout_baseline"):
        raise ValueError("branches need cfg['playout_baseline'] (model opponents answer with their argmax)")
    return SimpleNamespace(width=width, learner=GREEDY)


def seat_plan(game):
    """How each seat acts in the game's branch copies: ("model", slot) asks the parent, ("rule", policy) decides."""
    plan = []
    for seat, spec in enumerate(game.specs):
        if seat == game.focal:
            plan.append(("model", game.branching.learner))
        elif spec.startswith("model:"):
            plan.append(("model", spec))
        else:
            plan.append(("rule", game.policies[seat]))
    return plan


def start(game, row, legal, candidates, argmax):
    """The copies of a learning decision's candidates, each writing its result into row["branches"]."""
    world = game._branch_world(row)
    results = row["branches"] = []
    for index in candidates:
        result = {"action": int(index)}
        results.append(result)
        game._branch_copy(world, int(index), legal, result)


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
