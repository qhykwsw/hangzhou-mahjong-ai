"""The training game on the C engine (native.c): the same games, rows and requests, faster.

A training game whose rule seats are all fixed pool members deciding whole
in the runtime's race_act (and without a rule teacher; a model teacher
labels in the collection parent) plays its hands in C: the engine, every
member's turn, the learner's encoding and the default Critic features.
Python keeps the game's bookkeeping, the requests and the rows,
line for line as Game and Shadow do. Any other game stays on the Python
engine. Imported by a worker after it binds the rules runtime, before any
rule package loads.
"""

from mahjong_ai import _fast

from .environment import GREEDY, Game, hook

native = _fast.load_native(__file__)

# The fixed pool's members register their race profile with the runtime when
# their package loads; the same arguments register it here, so the C game can
# run the member's decision.
PROFILES = {}
_register = _fast.race_profile


def _race_profile(*args):
    index = _register(*args)
    PROFILES[index] = native.race_profile(*args)
    return index


_fast.race_profile = _race_profile


def _trivial(function):
    code = getattr(function, "__code__", None)
    return code is not None and code.co_names == () and code.co_consts in ((None,), ())


def race_member(policy):
    """(profile, fallback) of a member deciding whole in _fast.race_act, else None."""
    cls = type(policy)
    act = getattr(cls, "act", None)
    code = getattr(act, "__code__", None)
    if code is None or code.co_names != ("legal_actions", "_fast", "race_act", "PROFILE", "examined"):
        return None
    if not (_trivial(getattr(cls, "reset", None)) and _trivial(getattr(cls, "finish_round", None))):
        return None
    scope = act.__globals__
    if scope.get("_fast") is not _fast or scope.get("PROFILE") not in PROFILES:
        return None
    return PROFILES[scope["PROFILE"]], scope["examined"]


class FastGame(Game):
    """Game on native.Hand when every rule seat is a race member and no rule teacher labels."""

    def _next_hand(self):
        if not hasattr(self, "seats"):
            self.seats = self._plan()
            self.hands = []  # (dealer, scores, log) of every settled hand, for replays
        if self.seats is None:
            return super()._next_hand()
        from mahjong_ai.domain.engine import derive_seed

        self.hand_no += 1
        config = self.config
        self.state = native.Hand(
            derive_seed(self.seed, "wall", self.hand_no), self.dealer, self.hand_no, self.scores,
            config.rounds, config.base_score, int(config.you_cai_bi_kao), config.max_actions_per_hand,
        )
        self.start = (self.dealer, tuple(self.scores))

    def _plan(self):
        if self.teacher is not None:
            return None
        profiles, fallbacks = [], []
        for seat, spec in enumerate(self.specs):
            if spec.startswith("model:"):
                profiles.append(-1)
                fallbacks.append(None)
                continue
            member = race_member(self.policies[seat])
            if member is None:
                return None
            profiles.append(member[0])
            fallbacks.append(member[1])
        return tuple(profiles), tuple(fallbacks)

    def encode(self, state, seat):
        row = state.encode(self.cfg["lookahead"])
        if row is None:  # a hand past the winning boundary: the Python encoder's analysis
            from .encoding import encode

            row = encode(observation(state.snapshot(), seat, self.config), self.cfg["lookahead"])
        return row

    def advance(self):
        if self.seats is None:
            return super().advance()
        profiles, fallbacks = self.seats
        while not self.complete:
            state = self.state
            seat = state.run(profiles, fallbacks)
            if seat is None:
                delta = state.delta
                self.scores = [s + d for s, d in zip(self.scores, delta)]
                if self.rows and self.rows[-1]["hand"] == self.hand_no:
                    self.rows[-1]["reward"] += delta[self.focal] / self.config.base_score
                else:
                    self.predecision_reward += delta[self.focal] / self.config.base_score
                self.hands.append((*self.start, state.snapshot()["log"]))
                if state.winner is not None:
                    self.dealer = state.winner
                if self.hand_no == self.config.rounds:
                    # Race members keep no state across games: finish_round is a no-op.
                    self.complete = True
                    self.unfinished = self.train
                    self.settled()
                    return None
                self._next_hand()
                continue
            row = self.encode(state, seat)
            row.update(
                hand=self.hand_no,
                action=0,
                logprob=0.0,
                value=0.0,
                reward=0.0,
                teacher=-1,
                label_version=self.teacher_id,
                adv=0.0,
                ret=0.0,
            )
            if self.train and seat == self.focal:
                features = self.cfg.get("critic_features")
                row.update(state.oracle(seat) if not features else hook(features, None)(StateView(self, state), seat))
            self.pending = (row, state.legal(), seat)
            return {
                "row": row,
                "slot": self.specs[seat],
                "train": self.train and seat == self.focal,
                "bc": self.bc,
            }

    def apply(self, response):
        if self.seats is None:
            return super().apply(response)
        row, legal, seat = self.pending
        index, logprob, value, *more = response
        greedy = more[:1]
        if index not in legal:
            raise ValueError("neural policy returned an illegal action")
        if self.train and seat == self.focal:
            row.update(action=index, logprob=logprob, value=value)
            if len(more) > 1:  # a model teacher's label, from the collection parent
                row["teacher"] = more[1]
            if self.playout:
                if not greedy:
                    raise ValueError("the playout baseline needs the argmax with each collection answer")
                row["deviated"] = greedy[0] != index
                if row["deviated"]:
                    move = next(a for a in legal if a == greedy[0])
                    self.shadows.append(FastShadow(self, self.state.fork(), move, row))
            self.rows.append(row)
        self.pending = None
        self.state.step(index)


class FastShadow:
    """Shadow (a deviation's greedy continuation) on native.Hand."""

    def __init__(self, game, state, move, row):
        self.game, self.state, self.row = game, state, row
        self.dealer, self.pending, self.done = game.dealer, None, False
        state.step(move)

    def advance(self):
        game, state = self.game, self.state
        seat = state.run(*game.seats)
        if seat is not None:
            self.pending = state.legal()
            row = game.encode(state, seat)
            row["teacher"] = -1
            return {"row": row, "slot": GREEDY if seat == game.focal else game.specs[seat], "train": False, "bc": False}
        self.done, self.state = True, None
        # The winner deals next; after a draw the dealer stays.
        dealer = state.winner if state.winner is not None else self.dealer
        self.row["playout"] = state.delta[game.focal] / game.config.base_score
        self.row["playout_deals_next"] = float(dealer == game.focal)
        return None

    def apply(self, response):
        legal, self.pending = self.pending, None
        if response[0] not in legal:
            raise ValueError("neural policy returned an illegal action")
        self.state.step(response[0])


def observation(snap, seat, config):
    """The seat's Observation of a snapshot, with this hand's events only (all the encoder reads)."""
    state = materialize(snap, config)
    if state.turn != seat:
        raise ValueError("observations are of the seat to move")
    return state.observation()


def materialize(snap, config, history=(), views=None):
    """A HandState with a snapshot's fields (its events after ``history``)."""
    from mahjong_ai.domain.engine import HandState
    from mahjong_ai.domain.types import Meld, PublicEvent

    state = object.__new__(HandState)
    state.config = config
    state.dealer, state.hand_no = snap["dealer"], snap["hand_no"]
    state.rounds_total, state.scores, state.in_tiebreak = snap["rounds_total"], tuple(snap["scores"]), False
    state.wall = list(snap["wall"])
    state.hands = [list(h) for h in snap["hands"]]
    state.white_acquired = list(snap["white_acquired"])
    state.melds = [[Meld(k, tuple(t), f, c) for k, t, f, c in row] for row in snap["melds"]]
    state.discards = [list(r) for r in snap["discards"]]
    events = list(history)
    state.hand_start = len(events)
    for seat, tiles in enumerate(snap["initial_hands"]):
        events.append((PublicEvent("deal", seat, None, tuple(tiles), state.hand_no, (), False), seat))
    for kind, seat, tile, tiles, scores, concealed, private in snap["events"]:
        events.append((PublicEvent(kind, seat, tile, tuple(tiles), state.hand_no, tuple(scores), concealed), private))
    state.events = events
    if views is not None:
        state._views = [list(v) for v in views]
        state._views_source = state.events
        state._views_len = len(history)
    state.initial_hands = [list(h) for h in snap["initial_hands"]]
    state.phase, state.turn, state.drawn = snap["phase"], snap["turn"], snap["drawn"]
    state.kong_draw = snap["kong_draw"]
    state.chain, state.piao = list(snap["chain"]), list(snap["piao"])
    state.catch_owner, state.response_queue = snap["catch_owner"], list(snap["response_queue"])
    state.discarder, state.discarded, state.winner = snap["discarder"], snap["discarded"], snap["winner"]
    state.fan = None  # only a settled hand has one, and only the payment read it
    state.delta, state.settled, state.steps = tuple(snap["delta"]), snap["settled"], snap["steps"]
    return state


class StateView:
    """What a critic_features hook reads of the hand: its plain fields, and on
    any other attribute the Python engine's state, replayed to this point."""

    def __init__(self, game, hand):
        self._game, self._hand, self._full = game, hand, None
        snap = self._snap = hand.snapshot()
        self.dealer, self.turn, self.hand_no = snap["dealer"], snap["turn"], snap["hand_no"]
        self.hands = [list(h) for h in snap["hands"]]
        self.wall = list(snap["wall"])

    @property
    def melds(self):
        from mahjong_ai.domain.types import Meld

        return [[Meld(k, tuple(t), f, c) for k, t, f, c in row] for row in self._snap["melds"]]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._full is None:
            self._full = replay(self._game, self._snap)
        return getattr(self._full, name)


def replay(game, snap):
    """The game's Python HandState at a snapshot of its current hand, replayed from the first hand."""
    from mahjong_ai.domain.engine import HandState, derive_seed
    from selfplay_core.actions import action_index

    history, views = [], None
    config = game.config
    for number, (dealer, scores, log) in enumerate([*game.hands, (snap["dealer"], snap["scores"], snap["log"])], 1):
        if log is None:
            raise RuntimeError("hand too long to replay")
        state = HandState(config, derive_seed(game.seed, "wall", number), dealer, number, scores, history, False,
                          previous_views=views)
        state.rounds_total = config.rounds
        for index in log:
            legal = state.legal_actions()
            state.step(next(a for a in legal if action_index(a) == index), legal)
        history, views = state.events, state.public_views()
    return state
