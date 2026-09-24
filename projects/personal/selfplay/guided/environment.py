"""CPU workers own complete games; the parent owns all neural inference.

Importing this module must not load the game SDK: each spawned worker first
binds the immutable rules runtime supplied with its teachers.
"""

import importlib
from pathlib import Path
import random
import time
import traceback

from . import bootstrap  # noqa: F401
from .statistics import seed_for, advantages


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
    ):
        from mahjong_ai.domain.types import RoundConfig
        from .artifacts import build_policy

        self.cfg, self.seed, self.focal = cfg, seed, focal
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
        self.teacher = build_policy(bundle, teacher) if teacher else None
        if self.teacher:
            self.teacher.reset(seed_for(seed, "teacher", focal))
        self.teacher_id = teacher or ""
        self.scores, self.dealer, self.hand_no = [0] * 4, seed % 4, 0
        self.history, self.views, self.rows = [], None, []
        self.complete = False
        self.predecision_reward = 0.0
        self.pending = None
        self._next_hand()

    def _next_hand(self):
        from mahjong_ai.domain.engine import HandState, derive_seed

        self.hand_no += 1
        self.state = HandState(
            self.config,
            derive_seed(self.seed, "wall", self.hand_no),
            self.dealer,
            self.hand_no,
            self.scores,
            self.history,
            False,
            previous_views=self.views,
        )
        self.state.rounds_total = self.config.rounds

    def advance(self):
        from mahjong_ai.domain.types import RoundResult
        from mahjong_ai.domain.settlement import score_ranks
        from selfplay_core.actions import action_index
        from .encoding import encode, oracle

        while not self.complete:
            state = self.state
            if state.settled:
                self.scores = [s + d for s, d in zip(self.scores, state.delta)]
                if self.rows:
                    self.rows[-1]["reward"] += (
                        state.delta[self.focal] / self.config.base_score
                    )
                else:
                    # Rewards before the first choice cannot be credited to a future action.
                    self.predecision_reward += (
                        state.delta[self.focal] / self.config.base_score
                    )
                self.history, self.views = state.events, state.public_views()
                if state.winner is not None:
                    self.dealer = state.winner
                if self.hand_no == self.config.rounds:
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
                    if self.train:
                        finish = hook(self.cfg.get("finish_game"), None)
                        if finish is None:
                            advantages(self.rows, self.cfg["gae_lambda"])
                        else:
                            finish(self.rows, self.cfg)
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
                if self.teacher:
                    action = self.teacher.act(obs)
                    if action not in legal:
                        raise ValueError("teacher returned an illegal action")
                    row["teacher"] = action_index(action)
            self.pending = (row, legal, seat)
            return {
                "row": row,
                "slot": self.specs[seat],
                "train": self.train and seat == self.focal,
                "bc": self.bc,
            }

    def apply(self, response):
        from selfplay_core.actions import action_index

        row, legal, seat = self.pending
        index, logprob, value = response
        action = next((a for a in legal if action_index(a) == index), None)
        if action is None:
            raise ValueError("neural policy returned an illegal action")
        if self.train and seat == self.focal:
            row.update(action=index, logprob=logprob, value=value)
            self.rows.append(row)
        self.pending = None
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
        is_eval = command["kind"] == "evaluate"
        completed_samples = completed_games = 0
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
                focal = rng.randrange(4)
                slots = command.get("history", [])
                seats = [
                    rng.choice(slots)
                    if slots and rng.random() < cfg["history_probability"]
                    else rng.choice(pool)
                    for _ in range(4)
                ]
                seats[focal] = "model:current"
                game = Game(
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
                    teacher=bundle["teacher"] if command["labels"] else None,
                    train=True,
                    bc=cfg["bc_only"],
                    external=external,
                )
            games[next_id] = game
            next_id += 1
            return True

        for _ in range(cfg["tables"]):
            add_game()
        while games and time.time() < deadline:
            requests, finished = [], []
            for ident, game in list(games.items()):
                request = game.advance()
                if request is None:
                    external.release(game.borrowed)
                    completed_games += 1
                    completed_samples += len(game.rows)
                    finished.append(
                        {
                            "rows": game.rows,
                            "score": game.scores[game.focal],
                            "scores": game.scores,
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
                    if command.get("policy_inputs_only"):
                        # Rollout/privileged fields stay in the worker's owned
                        # row and are sent once with the completed game. Actor
                        # inference never reads them; BC also needs its label.
                        request = dict(request, row={
                            key: request["row"][key]
                            for key in ("planes", "scalars", "events", "mask", "teacher")
                        })
                    requests.append((ident, request))
            if requests or finished:
                conn.send(("batch", requests, finished))
                responses = conn.recv()
                if responses is None:
                    break
                for ident, response in responses:
                    games[ident].apply(response)
        conn.send(
            (
                "done",
                {
                    "games": completed_games,
                    "samples": completed_samples,
                    "unfinished_games": len(games),
                },
            )
        )
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if "external" in locals():
            external.close()
        conn.close()
