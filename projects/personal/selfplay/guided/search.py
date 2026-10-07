"""Search at the learner's key windows, for expert iteration (cfg["search"]).

At a learning decision of a configured kind (``kong``: the learner may kong;
``fish``: it may win by its own draw while holding a white), the candidates
are the Actor's argmax and the structural alternatives to it: every kong when
the argmax is not one (and, when it is, the best non-kong move), winning when
the argmax plays on (and, when it wins, the best move that does not). The
best move of a category is the Actor's argmax over that category alone,
asked from the collection parent like any greedy decision.

Each candidate is then played in ``worlds`` copies of the actual hand in
which only the wall is reshuffled (seeded per world): the opponents keep their
real hands, every candidate meets the same worlds, the learner plays its argmax
afterwards and the other seats as in the game, to the end of the hand. Like the
playout baseline (whose continuation plays the real hand), this reads hidden
state only to value the learner's choices in the training process (ADR-0009);
the Actor itself sees its legal observation alone. Uniformly re-dealt hidden
hands (sample_world) were measured to bias the search: on 300 fish windows the
opponents won 8% of the continuations instead of 34%, and playing on was
overvalued by 9.6 points. Because the values carry the real hidden hands, a
per-row choice of the best candidate partly imitates an oracle; average over
rows (e.g. use the values as advantages) rather than trusting single rows.
A world's value is the learner's hand score in base-score
units, plus the value of the next hand's start: ``continuation`` ("module:function",
resolved in the worker) is called with {"seat", "dealer" (who deals next: the
winner, or the dealer again after a draw), "hand_no" and "rounds" of the next
hand, "scores" (the game's scores after this hand, in base-score units), "last"
(this was the game's last hand)}; without it, ``deal_value`` is added when the
learner deals next.

The learning row gets ``search_kind``, ``search_actions`` (candidate action
indices, the argmax first), ``search_values`` (worlds x candidates) and
``search_q`` (the candidates' means), ``search_worlds`` (how many were played). The
game is finished (finish_game) only after its searches, like after the
playout baseline's continuations. Searches run on the Python engine.

    cfg["search"] = {"kinds": ["kong", "fish"], "worlds": 8, "max_windows": 1, "continuation": "recipe.value:after_hand"}
"""

import copy

from .environment import GREEDY, Shadow, continues

KINDS = ("kong", "fish")
WHITE = 33


def window_kind(obs, legal, kinds):
    """The window a learning decision is, or None (from the learner's own legal observation)."""
    actions = {a.action for a in legal}
    if "fish" in kinds and "hu" in actions and WHITE in obs.hand:
        return "fish"
    if "kong" in kinds and "gang" in actions:
        return "kong"
    return None


def _category(kind, action):
    """The structural category of a move in a window: the decision the search contrasts."""
    if kind == "fish":
        return "hu" if action.action == "hu" else "gang" if action.action == "gang" else "other"
    return "gang" if action.action == "gang" else "other"


class Search:
    def __init__(self, game, obs, legal, row, greedy, kind, real):
        from selfplay_core.actions import action_index

        self.game, self.obs, self.legal, self.row, self.kind = game, obs, legal, row, kind
        self.real = real  # a copy of the actual hand at the window
        self.oracle_turn = game.oracle_turn  # the window's learner decision in its hand (guided.oracle)
        self.cfg = game.search
        # A recipe's value of the next hand's start (e.g. a fitted V(dealer, hands left, scores)) replaces deal_value.
        from .environment import hook

        self.continuation = hook(self.cfg.get("continuation"), None)
        self.index = {action_index(a): a for a in legal}
        self.greedy = greedy
        categories = {}
        for i, a in self.index.items():
            categories.setdefault(_category(kind, a), []).append(i)
        mine = _category(kind, self.index[greedy])
        self.candidates = [greedy]
        self.asked = []
        for name, members in sorted(categories.items()):
            if name == mine:
                if name in ("gang", "hu"):
                    self.candidates += [i for i in members if i != greedy]  # e.g. a second kong
                continue
            if name in ("gang", "hu"):
                self.candidates += members  # few moves: each is a candidate
            else:
                self.asked.append(members)  # the Actor's best of the category
        self.values = {}

    def start(self):
        if self.asked:
            self.waiting = len(self.asked)
            for members in self.asked:
                self.game.shadows.append(Probe(self, members))
        else:
            self.play()

    def answered(self, index):
        if index not in self.candidates:
            self.candidates.append(index)
        self.waiting -= 1
        if not self.waiting:
            self.play()

    def play(self):
        import random

        from .environment import fork
        from .statistics import seed_for

        game = self.game
        self.rollouts, self.worlds = 0, int(self.cfg.get("worlds", 8))
        for m in range(self.worlds):
            world = fork(self.real)
            random.Random(seed_for(game.seed, "search", len(game.rows), m)).shuffle(world.wall)
            for index in self.candidates:
                state = fork(world)
                legal = state.legal_actions()
                move = next(a for a in legal if a == self.index[index])
                self.rollouts += 1
                game.shadows.append(WorldRollout(self, state, move, legal, m, index))
        if not self.rollouts:
            self.finish()

    def settled(self, m, index, value):
        self.values[(m, index)] = value
        self.rollouts -= 1
        if not self.rollouts:
            self.finish()

    def finish(self):
        worlds = sorted({m for m, _ in self.values})
        table = [[self.values[(m, i)] for i in self.candidates] for m in worlds]
        self.row.update(
            search_kind=self.kind,
            search_actions=list(self.candidates),
            search_values=table,
            search_q=[sum(r[k] for r in table) / len(table) if table else 0.0 for k in range(len(self.candidates))],
            search_worlds=len(table),
        )


class Probe:
    """One greedy question to the collection parent: the Actor's best move among ``members``."""

    def __init__(self, search, members):
        import numpy as np

        self.search, self.done, self.asked = search, False, False
        mask = np.zeros_like(search.row["mask"])
        mask[members] = True
        row = dict(search.row, mask=mask, teacher=-1)
        if not continues(search.game):
            # The row's masked oracle inputs (guided.oracle) are its own decision's: a greedy question
            # sees them only when the learner's greedy decisions do.
            from .oracle import FIELDS

            for key in FIELDS:
                row.pop(key, None)
        self.request = {"row": row, "slot": GREEDY, "train": False, "bc": False}

    def advance(self):
        if self.asked:
            return None
        self.asked = True
        return self.request

    def apply(self, response):
        self.done = True
        self.search.answered(int(response[0]))


class WorldRollout(Shadow):
    """A candidate's play-out in one world: the playout baseline's continuation, settled into the search."""

    def __init__(self, search, state, move, legal, m, index):
        self.search, self.m, self.index = search, m, index
        self.game, self.state, self.row = search.game, state, search.row
        self.dealer, self.pending, self.done = search.obs.stage.dealer, None, False
        self.oracle_turn = search.oracle_turn  # the searched decision's (guided.oracle)
        state.step(move, legal)

    def settle(self, final):
        game, search = self.game, self.search
        dealer = final.winner if final.winner is not None else self.dealer
        value = final.delta[game.focal] / game.config.base_score
        if search.continuation is not None:
            value += float(search.continuation(next_hand(final, dealer, game.focal, game.config)))
        else:
            value += float(search.cfg.get("deal_value", 0.0)) * (dealer == game.focal)
        search.settled(self.m, self.index, value)


def next_hand(final, dealer, seat, config):
    """What the game carries into the next hand after ``final``: the continuation's argument."""
    return {"seat": seat, "dealer": dealer, "hand_no": final.hand_no + 1, "rounds": config.rounds,
            "scores": [(s + d) / config.base_score for s, d in zip(final.scores, final.delta)],
            "last": final.hand_no >= config.rounds}
