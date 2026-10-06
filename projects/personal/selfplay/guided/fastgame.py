"""The training game on the C engine (native.c): the same games, rows and requests, faster.

A training game plays its hands in C: the engine, every fixed pool member's
turn (members deciding whole in the runtime's race_act), fp-rule-131's turn
and labels (rule131.h), the learner's encoding and the default Critic
features. Python keeps the game's bookkeeping, the requests and the rows,
line for line as Game and Shadow do. Any other rule seat (e.g. another
version from the catalog) and rule teacher decide on the Python engine's own
Observation: the C hand stops at their turn, a Python hand (Mirror) replays
the C hand's action log to that point, and the rule's action is played back
into C. A table with such a seat also replays each finished hand in Python,
for the next hand's public history. Imported by a worker after it binds the
rules runtime, before any rule package loads.
"""

from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import sys

from mahjong_ai import _fast

from .environment import GREEDY, Game, _teacher_failed, hook, staged

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


# fp-rule-131 as frozen (rule131.h is its port), by its sources' sha256.
RULE131 = {
    "strategy.py": "b18d020d42a979405a2b0a81faa6360185d8626b06eab7d07e781670d943a43c",
    "native.c": "74114e5e54d446a9aee12371b391258475b537c061f350363eca1f1607ba442f",
}


@lru_cache(maxsize=None)
def _is_rule131(package):
    try:
        return all(hashlib.sha256((Path(package) / name).read_bytes()).hexdigest() == digest
                   for name, digest in RULE131.items())
    except OSError:
        return False


def rule131_member(policy):
    """(member, fallback) of fp-rule-131's policy, which decides whole in C (rule131.h), else None."""
    cls = type(policy)
    module = sys.modules.get(cls.__module__)
    path = getattr(module, "__file__", None)
    if cls.__name__ != "Race" or not path or not _is_rule131(str(Path(path).resolve().parent)):
        return None
    scope = vars(module)
    profiles = scope["PROFILES"]  # SAVED -> the runtime's profile, registered here too as it loaded
    plain = inspect.signature(scope["piao_calls"]).parameters["saved"].default
    try:
        roles = tuple(tuple(PROFILES[profiles[scope["SAVED_BY_ROLE"][(dealer, left)]]] for left in range(4))
                      for dealer in (False, True))
        return native.rule131(PROFILES[profiles[plain]], roles), scope["analyze"]
    except KeyError:  # the package loaded before this module: its profiles are not here
        return None


class Mirror:
    """A C hand's Python HandState, kept at the C hand's position by replaying its action log.

    Built as Game._next_hand builds the hand (wall, dealer, scores, the game's
    public history before it, stable wall and luck, stage), so its
    observations are the Python engine's.
    """

    def __init__(self, game, hand_no, dealer, scores, history, views):
        from mahjong_ai.domain.engine import HandState, derive_seed

        state = HandState(game.config, derive_seed(game.seed, "wall", hand_no), dealer, hand_no, scores, history,
                          False, previous_views=views)
        state.rounds_total = game.config.rounds
        if game.stable or game.luck:
            from .walls import train_hand

            train_hand(state, stable=game.stable, luck_seat=game.focal if game.luck else None)
        if game.stage:
            from .stage import attach

            attach(state, game.stage)
        self.state, self.applied = state, 0

    def sync(self, hand):
        """The Python hand at the C hand's position."""
        from selfplay_core.actions import action_index

        log = hand.snapshot()["log"]
        if log is None:
            raise RuntimeError("hand too long to mirror")
        state = self.state
        for index in log[self.applied:]:
            legal = state.legal_actions()
            state.step(next(a for a in legal if action_index(a) == index), legal)
        self.applied = len(log)
        return state


def python_act(game, mirror, hand, seat):
    """A Python rule seat's turn: its action on the mirror's Observation, played on both hands."""
    from selfplay_core.actions import action_index

    state = mirror.sync(hand)
    legal = state.legal_actions()
    obs = state.observation(legal)
    if game.stage:
        obs = staged(obs, game.stage, state, game.config.rounds)
    action = game.policies[seat].act(obs)
    if action not in legal:
        raise ValueError("frozen opponent returned an illegal action")
    state.step(action, legal)
    mirror.applied += 1
    hand.step(action_index(action))


class FastGame(Game):
    """Game on native.Hand; race members and fp-rule-131 decide in C, other rule seats and teachers on a Mirror."""

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
        if self.stable:
            self.state.stabilize()
        if self.luck:
            self.state.track_luck(self.focal)
        self.start = (self.dealer, tuple(self.scores))
        self.mirror = None

    def _plan(self):
        profiles, fallbacks, python = [], [], set()
        for seat, spec in enumerate(self.specs):
            policy = None if spec.startswith("model:") else self.policies[seat]
            member = None if policy is None else race_member(policy) or rule131_member(policy)
            if member is None:  # the learner, a model opponent, or a rule deciding in Python
                profiles.append(-1)
                fallbacks.append(None)
                if not spec.startswith("model:"):
                    python.add(seat)
                continue
            profiles.append(member[0])
            fallbacks.append(member[1])
        self.python_seats = frozenset(python)
        # fp-rule-131 as the teacher labels on the C hand (_label).
        self.teacher_member = None if self.teacher is None else rule131_member(self.teacher)
        self.mirrored = bool(python) or (self.teacher is not None and self.teacher_member is None)
        return tuple(profiles), tuple(fallbacks)

    def _mirror(self):
        """This hand's Mirror, built on first use."""
        if self.mirror is None:
            dealer, scores = self.start
            self.mirror = Mirror(self, self.hand_no, dealer, list(scores), self.history, self.views)
        return self.mirror

    def _finish_round(self):
        """Game.advance's finish_round calls, for the rule seats and teacher deciding in Python."""
        from mahjong_ai.domain.settlement import score_ranks
        from mahjong_ai.domain.types import RoundResult

        result = RoundResult("guided", 0, self.config.stage, self.hand_no, tuple(self.scores), None, (), True,
                             {"objective": "four-player-settlement-score-v1"}, score_ranks(self.scores))
        for seat, policy in self.policies.items():
            policy.finish_round(result, seat)
        if self.teacher:
            self.teacher.finish_round(result, self.focal)

    def encode(self, state, seat):
        from .encoding import augment, encode

        row = state.encode(self.cfg["lookahead"])
        if row is None:  # a hand past the winning boundary: the Python encoder's analysis
            row = encode(observation(state.snapshot(), seat, self.config), self.cfg["lookahead"])
        # The C engine encodes v1; a wider encoding extends the row (guided.encoding).
        # Inside a stage the caller then rewrites the stage's scalars, v4's among them (guided.stage.rewrite).
        return augment(row, self.encoding)

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
                    if self.luck:
                        self.rows[-1]["end_luck_pay"], self.rows[-1]["end_luck_deal"] = state.luck
                    if self.stage:  # the game's four scores after the hand, by absolute seat
                        self.rows[-1]["end_scores"] = [s / self.config.base_score for s in self.scores]
                else:
                    self.predecision_reward += delta[self.focal] / self.config.base_score
                self.hands.append((*self.start, state.snapshot()["log"]))
                if self.mirrored:  # the next hand's public history, as Game keeps it
                    finished = self._mirror().sync(state)
                    self.history, self.views = finished.events, finished.public_views()
                if state.winner is not None:
                    self.dealer = state.winner
                if self.hand_no == self.config.rounds:
                    # Race members keep no state across games: finish_round is a no-op.
                    if self.mirrored:
                        self._finish_round()
                    self.complete = True
                    self.unfinished = self.train
                    self.settled()
                    return None
                self._next_hand()
                continue
            if seat in self.python_seats:
                python_act(self, self._mirror(), state, seat)
                continue
            row = self.encode(state, seat)
            if self.stage:
                from .stage import rewrite

                rewrite(row, self.stage, self.scores, self.hand_no, self.config.rounds, seat)
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
                if self.luck:
                    row["luck_pay"], row["luck_deal"] = state.luck
                if self.stage:
                    from .stage import row_fields

                    row.update(row_fields(self.stage, seat))
                if self.teacher:
                    self._label(row, state)
            self.pending = (row, state.legal(), seat)
            return {
                "row": row,
                "slot": self.specs[seat],
                "train": self.train and seat == self.focal,
                "bc": self.bc,
            }

    def _label(self, row, hand):
        """The rule teacher's label on a learner row, as Game.advance gives it."""
        from selfplay_core.actions import action_index

        if self.teacher_member is not None:
            try:
                row["teacher"] = hand.act(*self.teacher_member)
            except Exception:
                _teacher_failed(self.teacher_id)
                self.teacher_errors += 1
            return
        state = self._mirror().sync(hand)
        legal = state.legal_actions()
        obs = state.observation(legal)
        if self.stage:
            obs = staged(obs, self.stage, state, self.config.rounds)
        # A teacher that fails on a decision leaves that row unlabelled (Game.advance).
        try:
            action = self.teacher.act(obs)
        except Exception:
            action = None
            _teacher_failed(self.teacher_id)
        if action in legal:
            row["teacher"] = action_index(action)
        else:
            self.teacher_errors += 1
            if action is not None:
                _teacher_failed(self.teacher_id, f"illegal action {action}")

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
                    if self.vine:
                        move, index = index, move
                    self.shadows.append(FastShadow(self, self.state.fork(), move, row))
            self.rows.append(row)
        self.pending = None
        self.state.step(index)


class FastShadow:
    """Shadow (a deviation's greedy continuation) on native.Hand."""

    def __init__(self, game, state, move, row):
        self.game, self.state, self.row = game, state, row
        self.dealer, self.pending, self.done = game.dealer, None, False
        self.start_scores = list(game.scores)  # the game's scores before this hand
        self.hand_no = game.hand_no  # the game may move on to its next hand before this copy ends
        # The hand's public history, for a Mirror of the copy (built on a Python seat's first turn).
        self.history, self.views, self.mirror = game.history, game.views, None
        state.step(move)

    def advance(self):
        game, state = self.game, self.state
        seat = state.run(*game.seats)
        while seat is not None and seat in game.python_seats:
            if self.mirror is None:
                self.mirror = Mirror(game, self.hand_no, self.dealer, self.start_scores, self.history, self.views)
            python_act(game, self.mirror, state, seat)
            seat = state.run(*game.seats)
        if seat is not None:
            self.pending = state.legal()
            row = game.encode(state, seat)
            if game.stage:
                from .stage import rewrite

                rewrite(row, game.stage, self.start_scores, self.hand_no, game.config.rounds, seat)
            row["teacher"] = -1
            return {"row": row, "slot": GREEDY if seat == game.focal else game.specs[seat], "train": False, "bc": False}
        self.done, self.state = True, None
        # The winner deals next; after a draw the dealer stays.
        dealer = state.winner if state.winner is not None else self.dealer
        self.row["playout"] = state.delta[game.focal] / game.config.base_score
        self.row["playout_deals_next"] = float(dealer == game.focal)
        self.row["playout_dealer"] = dealer
        self.row["playout_scores"] = [(s + d) / game.config.base_score for s, d in zip(self.start_scores, state.delta)]
        if game.luck:
            self.row["playout_luck_pay"], self.row["playout_luck_deal"] = state.luck
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
        if game.stage:  # guided.stage's fields, as the Python hand carries them
            self.stage_prior, self.stage_game, self.stage_kind = (
                game.stage["prior"], game.stage["game"], game.stage["kind"])

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
        if game.stable or game.luck:
            from .walls import train_hand

            train_hand(state, stable=game.stable, luck_seat=game.focal if game.luck else None)
        for index in log:
            legal = state.legal_actions()
            state.step(next(a for a in legal if action_index(a) == index), legal)
        history, views = state.events, state.public_views()
    return state
