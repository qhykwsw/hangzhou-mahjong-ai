"""CPU workers own complete games; the parent owns all neural inference.

Importing this module must not load the game SDK: each spawned worker first
binds the immutable rules runtime supplied with its teachers.
"""

import copy
import gc
import importlib
from pathlib import Path
import random
import time
import traceback

from . import bootstrap  # noqa: F401
from .statistics import seed_for, advantages


# The current Actor's argmax: the learner's seat in a deviation's greedy continuation.
GREEDY = "model:greedy"


def hook(spec, default):
    """Resolve an optional "module:function" training-recipe hook.

    Resolved lazily inside workers, after the teacher runtime is bound, so a
    recipe module may import the engine at module level.
    """
    if not spec:
        return default
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


class Game:
    def __init__(
        self,
        cfg,
        bundle,
        entries,
        seat_specs,
        seed,
        *,
        focal=0,
        teacher=None,
        train=False,
        bc=False,
        group=None,
        external=None,
        opening=None,
    ):
        from mahjong_ai.domain.types import RoundConfig
        from .artifacts import build_policy
        from .encoding import ENCODING

        self.cfg, self.seed, self.focal = cfg, seed, focal
        # Model seats' inputs, for the widest encoding at the table (cfg["encoding"],
        # set by the trial): each Actor reads the prefix of the planes it was built for.
        self.encoding = cfg.get("encoding") or ENCODING
        # An opening (guided.starts) is a one-hand game from a recorded position.
        self.opening = opening
        self.train, self.bc, self.group = train, bc, group
        self.config = RoundConfig(rounds=cfg["rounds"], allow_tiebreak=False)
        self.specs = seat_specs
        self.policies = {}
        # Frozen versions outside the bundle (e.g. evaluation opponents from
        # another runtime) come from the worker's pool of isolated processes.
        self.borrowed = []
        for seat, spec in enumerate(seat_specs):
            if not spec.startswith("model:"):
                if external is not None and spec in external.specs:
                    policy = external.acquire(spec)
                    self.borrowed.append((spec, policy))
                else:
                    policy = build_policy(entries[spec], spec)
                policy.reset(seed_for(seed, "policy", seat))
                self.policies[seat] = policy
        self.teacher = None
        # A model teacher (a "model:" slot) labels in the collection parent,
        # whose answer carries the label (apply); only a rule teacher acts here.
        if teacher and not teacher.startswith("model:"):
            # A teacher outside the bundle (a frozen version the trial offers) is borrowed like an opponent.
            bundled = bundle is not None and any(e["id"] == teacher for e in bundle["entries"])
            if not bundled and external is not None and teacher in external.specs:
                self.teacher = external.acquire(teacher)
                self.borrowed.append((teacher, self.teacher))
            else:
                self.teacher = build_policy(bundle, teacher)
            self.teacher.reset(seed_for(seed, "teacher", focal))
        self.teacher_id = teacher or ""
        self.teacher_errors = 0
        self.scores, self.dealer, self.hand_no = [0] * 4, seed % 4, 0
        if opening is not None:
            self.scores, self.dealer, self.hand_no = list(opening["scores"]), opening["dealer"], opening["hand_no"] - 1
        self.history, self.views, self.rows = [], None, []
        self.complete = False
        self.predecision_reward = 0.0
        self.pending = None
        # cfg["playout_baseline"]: every learner deviation from the argmax gets a
        # greedy continuation of its hand (Shadow); the game is finished once they are.
        self.playout = bool(cfg.get("playout_baseline")) and train
        # cfg["vine"]: the hand itself always plays the learner's argmax and each
        # deviation is played out in the copy instead (see Shadow).
        self.vine = self.playout and bool(cfg.get("vine"))
        # cfg["search"] (guided.search): key windows of the learner get a search over worlds.
        self.search = cfg.get("search") if train else None
        # guided.walls: training hands on a stable wall, and the learner's draw luck on its rows.
        self.stable = bool(cfg.get("stable_wall")) and train
        self.luck = bool(cfg.get("luck")) and train
        # cfg["stage"] (guided.stage): the game is placed inside a tournament stage.
        from .stage import start

        self.stage = start(cfg, seed)
        self.searched, self.window = 0, None
        self.shadows = []
        self.unfinished = False
        self._next_hand()

    def _next_hand(self):
        from mahjong_ai.domain.engine import HandState, derive_seed

        self.hand_no += 1
        opening = self.opening
        self.state = HandState(
            self.config,
            opening["seed"] if opening else derive_seed(self.seed, "wall", self.hand_no),
            self.dealer,
            self.hand_no,
            self.scores,
            self.history,
            False,
            previous_views=self.views,
        )
        self.state.rounds_total = self.config.rounds
        if opening:
            from .starts import replay

            replay(self.state, opening["prefix"], self.focal)
            if self.train and self.cfg.get("starts_reshuffle"):
                # The recorded wall's order past the position is hidden: a fresh
                # order per game makes each replay a new hidden sample.
                random.Random(seed_for(self.seed, "starts-reshuffle")).shuffle(self.state.wall)
        if self.stable or self.luck:
            from .walls import train_hand

            train_hand(self.state, stable=self.stable, luck_seat=self.focal if self.luck else None)
        if self.stage:
            from .stage import attach

            attach(self.state, self.stage)

    def advance(self):
        from mahjong_ai.domain.types import RoundResult
        from mahjong_ai.domain.settlement import score_ranks
        from selfplay_core.actions import action_index
        from .encoding import encoder, oracle

        encode = encoder(self.encoding)
        while not self.complete:
            state = self.state
            if state.settled:
                self.scores = [s + d for s, d in zip(self.scores, state.delta)]
                if self.rows and self.rows[-1]["hand"] == self.hand_no:
                    self.rows[-1]["reward"] += (
                        state.delta[self.focal] / self.config.base_score
                    )
                    if self.luck:
                        self.rows[-1]["end_luck_pay"], self.rows[-1]["end_luck_deal"] = state.luck
                    if self.stage:  # the game's four scores after the hand, by absolute seat
                        self.rows[-1]["end_scores"] = [s / self.config.base_score for s in self.scores]
                else:
                    # A hand in which the learner had no choice (before its first,
                    # or a later hand won before its turn) credits no decision:
                    # rows carry their own hand's score, recipes build targets per hand.
                    self.predecision_reward += (
                        state.delta[self.focal] / self.config.base_score
                    )
                self.history, self.views = state.events, state.public_views()
                if state.winner is not None:
                    self.dealer = state.winner
                if self.hand_no == self.config.rounds or self.opening:
                    self.complete = True
                    result = RoundResult(
                        "guided",
                        0,
                        self.config.stage,
                        self.hand_no,
                        tuple(self.scores),
                        None,
                        (),
                        True,
                        {"objective": "four-player-settlement-score-v1"},
                        score_ranks(self.scores),
                    )
                    for seat, policy in self.policies.items():
                        policy.finish_round(result, seat)
                    if self.teacher:
                        self.teacher.finish_round(result, self.focal)
                    self.unfinished = self.train
                    self.settled()
                    return None
                self._next_hand()
                continue
            if state.steps >= self.config.max_actions_per_hand:
                raise RuntimeError("hand action budget exceeded")
            legal = state.legal_actions()
            if len(legal) == 1:
                state.step(legal[0], legal)
                continue
            seat = state.turn
            obs = state.observation(legal)
            if self.stage:
                obs = staged(obs, self.stage, state, self.config.rounds)
            if seat in self.policies:
                action = self.policies[seat].act(obs)
                if action not in legal:
                    raise ValueError("frozen opponent returned an illegal action")
                state.step(action, legal)
                continue
            row = encode(obs, self.cfg["lookahead"])
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
                row.update(hook(self.cfg.get("critic_features"), oracle)(state, seat))
                if self.luck:
                    row["luck_pay"], row["luck_deal"] = state.luck
                if self.stage:
                    from .stage import row_fields

                    row.update(row_fields(self.stage, seat))
                if self.teacher:
                    # A teacher that fails on a decision (raises, or answers
                    # illegally) leaves that row unlabelled instead of
                    # stopping the collection.
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
            window = None
            if self.search and self.train and seat == self.focal and self.searched < self.search.get("max_windows", 1):
                from .search import window_kind

                window = window_kind(obs, legal, self.search.get("kinds", ("kong", "fish")))
            self.pending = (row, legal, seat)
            self.window = (obs, window) if window else None
            return {
                "row": row,
                "slot": self.specs[seat],
                "train": self.train and seat == self.focal,
                "bc": self.bc,
            }

    def apply(self, response):
        from selfplay_core.actions import action_index

        row, legal, seat = self.pending
        obs, window = self.window or (None, None)
        self.window = None
        # Collection answers (action, logprob, value, argmax), and after them a
        # model teacher's label on a learning decision; evaluation (action, logprob, value).
        index, logprob, value, *more = response
        greedy = more[:1]
        action = next((a for a in legal if action_index(a) == index), None)
        if action is None:
            raise ValueError("neural policy returned an illegal action")
        if self.train and seat == self.focal:
            row.update(action=index, logprob=logprob, value=value)
            if len(more) > 1:
                row["teacher"] = more[1]
            if self.playout:
                if not greedy:
                    raise ValueError("the playout baseline needs the argmax with each collection answer")
                row["deviated"] = greedy[0] != index
                if row["deviated"]:
                    move = next(a for a in legal if action_index(a) == greedy[0])
                    if self.vine:
                        # The copy plays the sampled move, the hand its argmax.
                        move, action = action, move
                    self.shadows.append(Shadow(self, fork(self.state), move, legal, row))
            if window:
                if not greedy:
                    raise ValueError("a search needs the argmax with each collection answer")
                from .search import Search

                self.searched += 1
                Search(self, obs, legal, row, greedy[0], window, fork(self.state)).start()
            self.rows.append(row)
        self.pending = None
        self.state.step(action, legal)

    def settled(self):
        """Complete, with every greedy continuation played out; the rows are then finished once."""
        if not self.complete or not all(shadow.done for shadow in self.shadows):
            return False
        if self.unfinished:
            self.unfinished = False
            finish = hook(self.cfg.get("finish_game"), None)
            if finish is None:
                advantages(self.rows, self.cfg["gae_lambda"])
            else:
                finish(self.rows, self.cfg)
        return True


_TEACHER_WARNED = set()


def _teacher_failed(teacher, what=None):
    """Report a teacher's first failure in this worker on stderr (the trial's log)."""
    if teacher in _TEACHER_WARNED:
        return
    _TEACHER_WARNED.add(teacher)
    import sys

    detail = what or traceback.format_exc(limit=3)
    print(f"teacher {teacher} failed on a decision; unlabelled from now on where it fails:\n{detail}", file=sys.stderr, flush=True)


def staged(obs, info, state, rounds):
    """The observation inside the game's stage (guided.stage), from the hand's own scores and number."""
    from .stage import observe

    return observe(obs, info, state.scores, state.hand_no, rounds)


def fork(state):
    """A copy of a hand to play on.

    The game's recorded events and the per-seat views built from them are
    immutable and shared (deep-copying them cost about as much as playing
    the continuation); the lists holding them and the live hand are copied.
    """
    memo = {id(item): item for item in state.events}
    for view in getattr(state, "_views", ()):
        memo.update((id(item), item) for item in view)
    return copy.deepcopy(state, memo)


class Shadow:
    """A learner deviation's greedy continuation, for the playout baseline.

    A copy of the hand at the deviation, with the argmax played there and at
    every later learner decision and every other seat as in the game, to the
    end of the hand. Wall and seeds are fixed and the other seats decide
    deterministically (rules; model opponents answer with their argmax under
    cfg["playout_baseline"]), so until the learner's next deviation the game's
    own path is this greedy path: every greedy decision before a deviation
    shares its playout value, and after the hand's last deviation the realized
    outcome. The deviating row gets ``playout`` (the learner's hand score in
    base-score units) and ``playout_deals_next`` (1.0 when the learner deals
    the next hand).

    Under cfg["vine"] the roles swap: the hand plays the argmax at every
    learner decision, so its states are the argmax path's whatever the
    sampling, and the copy plays the sampled move there and the argmax
    afterwards; ``playout`` is then the deviation's value and the hand's own
    outcome is the argmax value at each of its deviations.
    """

    def __init__(self, game, state, action, legal, row):
        self.game, self.state, self.row = game, state, row
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
            if seat in game.policies:
                action = game.policies[seat].act(obs)
                if action not in legal:
                    raise ValueError("frozen opponent returned an illegal action")
                state.step(action, legal)
                continue
            self.pending = legal
            row = encode(obs, game.cfg["lookahead"])
            row["teacher"] = -1
            return {"row": row, "slot": GREEDY if seat == game.focal else game.specs[seat], "train": False, "bc": False}
        self.done, self.state = True, None
        self.settle(state)
        return None

    def settle(self, state):
        game = self.game
        # The winner deals next; after a draw the dealer stays.
        dealer = state.winner if state.winner is not None else self.dealer
        self.row["playout"] = state.delta[game.focal] / game.config.base_score
        self.row["playout_deals_next"] = float(dealer == game.focal)
        # The continuation's next-hand start, for a recipe's value of it.
        self.row["playout_dealer"] = dealer
        self.row["playout_scores"] = [(s + d) / game.config.base_score for s, d in zip(state.scores, state.delta)]
        if game.luck:
            self.row["playout_luck_pay"], self.row["playout_luck_deal"] = state.luck

    def apply(self, response):
        from selfplay_core.actions import action_index

        legal, self.pending = self.pending, None
        action = next((a for a in legal if action_index(a) == response[0]), None)
        if action is None:
            raise ValueError("neural policy returned an illegal action")
        self.state.step(action, legal)


class External:
    """Per-worker pool of policies for frozen versions outside the bundle.

    A version frozen with the worker's own runtime (same fingerprint, i.e.
    directory name) loads in-process like a bundle rule; any other runs in
    its own runtime in a sandboxed process (mahjong_ai's RemotePolicy),
    reused by later games in the same worker.
    """

    def __init__(self, specs):
        self.specs, self.free = specs, {k: [] for k in specs}

    def acquire(self, spec):
        from pathlib import Path
        from mahjong_ai.strategies.common import PolicySpec
        from .artifacts import bound_runtime_name, load_package

        if self.free[spec]:
            return self.free[spec].pop()
        frozen = self.specs[spec]
        # The task owner may vouch for a package (isolation "none"): it then runs
        # here under the bound runtime like a bundle rule, whatever runtime it was
        # frozen with; the rules SDK is behaviour-identical across runtimes.
        if frozen.get("isolation") == "none" or Path(frozen.get("runtime_path") or "").name == bound_runtime_name():
            return load_package(frozen["package_path"])
        policy = PolicySpec(**{k: v for k, v in frozen.items() if k != "isolation"}).build()
        if hasattr(policy, "start"):
            policy.start()
        return policy

    def release(self, borrowed):
        for spec, policy in borrowed:
            self.free[spec].append(policy)

    def close(self):
        for policies in self.free.values():
            for policy in policies:
                try:
                    policy.close()
                except Exception:
                    pass


def worker(conn, cfg, bundle_path, panel_path, worker_index, command):
    """One bounded collection/evaluation job; no policy crosses job boundaries."""
    # Automatic collections trigger every few hundred allocations and cost
    # about 5% of a game; a full one every 100 games bounds the cycles instead.
    gc.disable()
    try:
        from .artifacts import read_bundle, bind_runtime

        bundle, panel = read_bundle(bundle_path), read_bundle(panel_path)
        if command.get("runtime"):
            # The trial's own frozen runtime (its evaluation pool's) instead of
            # the bundle's copy: games run on the current engine everywhere.
            for b in (bundle, panel):
                b["runtime"], b["runtime_hash"] = command["runtime"], Path(command["runtime"]).name
        bind_runtime(bundle)
        bind_runtime(panel)
        entries = {e["id"]: panel for e in panel["entries"]}
        entries.update({e["id"]: bundle for e in bundle["entries"]})
        rules = [e["id"] for e in bundle["entries"]]
        external = External(command.get("specs") or {})
        # Opponent seats draw from the command's pool when given (frozen
        # evaluation opponents), otherwise from the bundle's rules.
        pool = list(command.get("pool") or rules)
        rng = random.Random(
            seed_for(cfg["seed"], "worker", command["iteration"], worker_index)
        )
        deadline = command["deadline"]
        todo = list(command.get("jobs", []))
        openings = command.get("starts") or []
        is_eval = command["kind"] == "evaluate"
        # Training games whose rule seats the C engine can play run there.
        if not is_eval and cfg.get("fast_game", True) and not cfg.get("search"):
            from .fastgame import FastGame as Training
        else:
            Training = Game
        completed_samples = completed_games = teacher_errors = 0
        games = {}
        next_id = 0

        def add_game():
            nonlocal next_id
            if time.time() >= deadline:
                return False
            if is_eval:
                if not todo:
                    return False
                job = todo.pop(0)
                game = Game(
                    cfg,
                    bundle,
                    entries,
                    job["seats"],
                    job["seed"],
                    focal=job["focal"],
                    group=job["group"],
                )
            else:
                if completed_samples >= command["samples"]:
                    return False
                # With openings every game is one hand from a recorded position,
                # on the Python engine (the C engine starts only from a deal).
                opening = rng.choice(openings) if openings else None
                focal = opening["focal"] if opening else rng.randrange(4)
                slots = command.get("history", [])
                seats = [
                    rng.choice(slots)
                    if slots and rng.random() < cfg["history_probability"]
                    else rng.choice(pool)
                    for _ in range(4)
                ]
                seats[focal] = "model:current"
                game = (Game if opening else Training)(
                    cfg,
                    bundle,
                    entries,
                    seats,
                    seed_for(
                        cfg["seed"],
                        "training-wall",
                        command["iteration"],
                        worker_index,
                        next_id,
                    ),
                    focal=focal,
                    teacher=(command.get("teacher") or bundle["teacher"]) if command["labels"] else None,
                    train=True,
                    bc=cfg["bc_only"],
                    external=external,
                    **({"opening": opening} if opening else {}),
                )
            games[next_id] = game
            next_id += 1
            return True

        def inputs(request):
            if not command.get("policy_inputs_only"):
                return request
            # Rollout/privileged fields stay in the worker's owned row and are
            # sent once with the completed game. Actor inference never reads
            # them; BC also needs its label.
            return dict(request, row={
                key: request["row"][key] for key in ("planes", "scalars", "events", "mask", "teacher")
            })

        for _ in range(cfg["tables"]):
            add_game()
        while games and time.time() < deadline:
            requests, finished = [], []
            for ident, game in list(games.items()):
                for n, shadow in enumerate(game.shadows):
                    if not shadow.done:
                        request = shadow.advance()
                        if request is not None:
                            requests.append(((ident, n), inputs(request)))
                request = game.advance()
                if request is None and not game.settled():
                    continue  # its greedy continuations are still being played
                if request is None:
                    external.release(game.borrowed)
                    completed_games += 1
                    if completed_games % 100 == 0:
                        gc.collect()
                    completed_samples += len(game.rows)
                    teacher_errors += game.teacher_errors
                    finished.append(
                        {
                            "rows": game.rows,
                            "score": game.scores[game.focal],
                            "scores": game.scores,
                            "hands": 1 if game.opening else game.config.rounds,
                            "seed": game.seed,
                            "focal": game.focal,
                            "group": game.group,
                            "policy": game.specs[game.focal],
                            "seats": game.specs,
                            "predecision_reward": game.predecision_reward,
                        }
                    )
                    del games[ident]
                    add_game()
                else:
                    requests.append((ident, inputs(request)))
            if requests or finished:
                conn.send(("batch", requests, finished))
                responses = conn.recv()
                if responses is None:
                    break
                for ident, response in responses:
                    if isinstance(ident, tuple):
                        games[ident[0]].shadows[ident[1]].apply(response)
                    else:
                        games[ident].apply(response)
        conn.send(
            (
                "done",
                {
                    "games": completed_games,
                    "samples": completed_samples,
                    "unfinished_games": len(games),
                    "teacher_errors": teacher_errors,
                },
            )
        )
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if "external" in locals():
            external.close()
        conn.close()
