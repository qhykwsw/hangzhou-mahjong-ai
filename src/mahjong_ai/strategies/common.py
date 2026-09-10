from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from mahjong_ai.domain.rules import WHITE, counts, hand_potential, is_baotou
from mahjong_ai.domain.types import (
    Action,
    Observation,
    RoundResult,
    StageKind,
    RANK_OBJECTIVE,
)

FEATURE_VERSION = "public-state-action-rank-v2"
ACTIONS = ("discard", "chi", "peng", "gang", "hu", "pass")


class Policy(Protocol):
    name: str
    version: str

    def reset(self, seed: int): ...
    def act(self, observation: Observation) -> Action: ...
    def finish_round(self, result: RoundResult, seat: int): ...


def risk_preference(obs: Observation) -> float:
    """Continuous score-relative risk, not a stage-specific reward switch."""
    ctx = obs.stage
    own = ctx.own_score
    other = sorted(
        (x for i, x in enumerate(ctx.effective_scores) if i != ctx.seat), reverse=True
    )
    # Each opponent has equal rank value. The stage never switches objectives.
    spread = max(16.0, max(ctx.effective_scores) - min(ctx.effective_scores))
    return (
        sum(math.tanh((score - own) / spread) for score in other)
        / 3
        * ctx.progress**1.5
    )


def observed_ranks(obs):
    if obs.stage.rank_by_seat is not None:
        return obs.stage.rank_by_seat
    # Only visible scores; ties stay tied when official tie fields are unknown.
    from mahjong_ai.domain.settlement import score_ranks

    return score_ranks(obs.stage.effective_scores)


def state_features(obs: Observation) -> np.ndarray:
    f = [n / 4 for n in counts(obs.hand)]
    for row in obs.discards:
        f.extend(n / 4 for n in counts(row))
    for row in obs.melds:
        f.extend(n / 4 for n in counts(t for m in row for t in m.tiles))
    f.extend(n / 14 for n in obs.hand_sizes)
    ctx = obs.stage
    f.extend(math.tanh((x - ctx.own_score) / 64) for x in ctx.effective_scores)
    f.extend(
        (
            ctx.progress,
            ctx.remaining_rounds / max(1, ctx.effective_horizon[1]),
            math.log1p(ctx.effective_horizon[1]) / 5,
            float(ctx.in_tiebreak),
            obs.wall_remaining / 136,
            obs.chain_count / 8,
            obs.piao_count / 4,
            math.log2(max(1, obs.hu_fan)) / 10,
            float(obs.catch_play),
            float(obs.you_cai_bi_kao),
            risk_preference(obs),
        )
    )
    f.extend(float(ctx.kind == kind) for kind in StageKind)
    f.extend(float(ctx.dealer == (ctx.seat + i) % 4) for i in range(4))
    f.extend(float(obs.phase == p) for p in ("draw", "response_peng", "response_chi"))
    # Public chronology summary plus recent events; the complete history is also
    # available in Observation for alternative rule/model encoders.
    for event in obs.public_events[-8:]:
        f.extend(
            (
                float(event.kind == "discard"),
                float(event.kind == "draw"),
                -1.0 if event.seat is None else (event.seat - ctx.seat) % 4 / 3,
                -1.0 if event.tile is None else event.tile / 33,
            )
        )
    f.extend([0.0] * (32 - min(8, len(obs.public_events)) * 4))
    return np.asarray(f, dtype=np.float64)


def action_summary(obs: Observation, action: Action):
    h = list(obs.hand)
    opened = len(obs.melds[obs.stage.seat])
    before = hand_potential(tuple(sorted(h)), opened)
    piao = False
    if action.action == "discard":
        h.remove(action.tile)
        if action.tile == WHITE and obs.drawn_tile is not None:
            waiting = list(obs.hand)
            waiting.remove(obs.drawn_tile)
            piao = is_baotou(waiting, opened) and is_baotou(h, opened)
        elif action.tile == WHITE and obs.catch_play and not obs.catch_restricted:
            piao = is_baotou(h, opened)
    elif action.action in ("peng", "chi"):
        used = action.tiles if action.action == "chi" else (action.tile,) * 2
        for t in used:
            h.remove(t)
        opened += 1
    elif action.action == "gang":
        n = {"concealed": 4, "exposed": 3, "added": 1}[action.kind]
        for _ in range(n):
            h.remove(action.tile)
        if action.kind != "added":
            opened += 1
    after = hand_potential(tuple(sorted(h)), opened)
    return before, after, piao


def action_features(obs: Observation, action: Action):
    before, after, piao = action_summary(obs, action)
    f = [float(action.action == a) for a in ACTIONS]
    f.extend(float(action.tile == t) for t in range(34))
    f.extend(float(action.kind == k) for k in ("concealed", "exposed", "added"))
    f.extend(
        (
            after / 30,
            (after - before) / 10,
            float(piao),
            float(action.tile == WHITE),
            risk_preference(obs),
            obs.stage.progress * float(piao),
            risk_preference(obs) * float(piao),
            risk_preference(obs) * float(action.action == "hu"),
            risk_preference(obs) * float(action.action == "gang"),
        )
    )
    return np.asarray(f, dtype=np.float64)


def feature_matrix(obs: Observation):
    s = state_features(obs)
    ranks = np.asarray([(r - 1) / 3 for r in observed_ranks(obs)])
    return np.asarray(
        [np.concatenate((s, action_features(obs, a), ranks)) for a in obs.legal_actions]
    )


@dataclass
class RulePolicy:
    name: str = "rule"
    version: str = "rule-v1"
    risk_scale: float = 1.0
    call_bonus: float = 0.15
    kong_bonus: float = 0.5
    piao_bonus: float = 0.0
    hu_bonus: float = 12.0
    program: str | None = None

    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass

    def utilities(self, obs):
        risk = risk_preference(obs) * self.risk_scale
        values = []
        for action in obs.legal_actions:
            before, after, piao = action_summary(obs, action)
            if action.action == "hu":
                value = self.hu_bonus + math.log2(max(1, obs.hu_fan)) - risk * 2
            elif action.action == "discard":
                value = (after - before) * 0.7
                if piao:
                    value += self.hu_bonus - 2 + self.piao_bonus + risk * 5
            elif action.action in ("chi", "peng"):
                value = (after - before) * 0.5 + self.call_bonus
            elif action.action == "gang":
                value = (after - before) * 0.5 + self.kong_bonus + risk
            else:
                value = 0.0
            if self.program:
                from mahjong_ai.platform.program import evaluate_program

                value = evaluate_program(
                    self.program,
                    {
                        "base": float(value),
                        "risk": float(risk),
                        "progress": obs.stage.progress,
                        "rank": observed_ranks(obs)[obs.stage.seat],
                        "ranks": observed_ranks(obs),
                        "remaining": obs.stage.remaining_rounds,
                        "score": obs.stage.own_score,
                        "gap": max(obs.stage.effective_scores) - obs.stage.own_score,
                        "fan": obs.hu_fan,
                        "quality": after,
                        "quality_delta": after - before,
                        "piao": piao,
                        "discard": action.action == "discard",
                        "hu": action.action == "hu",
                        "kong": action.action == "gang",
                        "call": action.action in ("chi", "peng"),
                        "white": action.tile == WHITE,
                        "hand": counts(obs.hand),
                        "scores": obs.stage.effective_scores,
                        "state": tuple(float(v) for v in state_features(obs)),
                    },
                )
            values.append(value)
        return np.asarray(values)

    def act(self, obs):
        if not obs.legal_actions:
            raise ValueError("no legal actions")
        return obs.legal_actions[int(np.argmax(self.utilities(obs)))]


@dataclass(frozen=True)
class TrainingRecipe:
    algorithm: str = "reinforce-value-baseline"
    learning_rate: float = 0.002
    reward_scale: float = 1.0
    entropy: float = 0.01
    hidden_size: int = 16
    seed: int = 0

    def __post_init__(self):
        if self.algorithm != "reinforce-value-baseline":
            raise ValueError("unsupported training algorithm")
        if (
            not all(
                math.isfinite(v)
                for v in (self.learning_rate, self.reward_scale, self.entropy)
            )
            or not 0 < self.learning_rate <= 0.1
            or self.reward_scale <= 0
        ):
            raise ValueError("invalid training recipe")
        if (
            type(self.hidden_size) is not int
            or type(self.seed) is not int
            or self.seed < 0
            or not 4 <= self.hidden_size <= 256
            or not 0 <= self.entropy <= 1
        ):
            raise ValueError("invalid network/entropy budget")


class RLPolicy:
    """Trainable contextual softmax policy with a small NumPy MLP.

    Uses a rule prior for a useful start, then optimizes terminal settlement rank utility via
    REINFORCE and a value baseline. No stage-specific binary reward is used.
    """

    name = "rl"
    version = "rl-v1"

    def __init__(
        self, recipe=None, model_path=None, training=False, name="rl", version="rl-v1"
    ):
        self.recipe = recipe or TrainingRecipe()
        self.name, self.version = name, version
        self.training = training
        self.prior = RulePolicy()
        self.params = None
        self.trace = []
        self.last_update = {}
        self.rng = np.random.default_rng(self.recipe.seed)
        if model_path:
            with np.load(model_path, allow_pickle=False) as data:
                schema = str(data["schema"])
                if schema not in (FEATURE_VERSION, "public-state-action-v1"):
                    raise ValueError("feature schema mismatch")
                self.params = {k: data[k].copy() for k in ("w", "b", "out", "value")}
                if schema == "public-state-action-v1":
                    if self.params["w"].ndim != 2 or self.params["value"].ndim != 1:
                        raise ValueError("invalid legacy model shape")
                    self.params["w"] = np.pad(self.params["w"], ((0, 4), (0, 0)))
                    self.params["value"] = np.concatenate(
                        (
                            self.params["value"][:-1],
                            np.zeros(4),
                            self.params["value"][-1:],
                        )
                    )
                self.loaded_objective = (
                    str(data["objective"]) if "objective" in data else "legacy-score"
                )
                if training and self.loaded_objective != RANK_OBJECTIVE:
                    self.params["value"].fill(
                        0
                    )  # actor warm start; discard the old score critic
            p = self.params
            if (
                p["w"].ndim != 2
                or p["w"].shape[1] != self.recipe.hidden_size
                or p["b"].shape != (self.recipe.hidden_size,)
                or p["out"].shape != (self.recipe.hidden_size,)
                or p["value"].shape != (p["w"].shape[0] + 1,)
                or any(not np.isfinite(v).all() for v in p.values())
            ):
                raise ValueError(
                    "model shape or values do not match the training recipe"
                )

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.trace = []

    def _initialize(self, x):
        if self.params is None:
            rng = np.random.default_rng(self.recipe.seed)
            size = x.shape[1]
            hidden = self.recipe.hidden_size
            self.params = {
                "w": rng.normal(0, 0.05, (size, hidden)),
                "b": np.zeros(hidden),
                "out": rng.normal(0, 0.05, hidden),
                "value": np.zeros(size + 1),
            }

    def act(self, obs):
        if not obs.legal_actions:
            raise ValueError("no legal actions")
        x = feature_matrix(obs)
        self._initialize(x)
        p = self.params
        if x.shape[1] != p["w"].shape[0]:
            raise ValueError("invalid model input shape")
        h = np.tanh(x @ p["w"] + p["b"])
        logits = h @ p["out"] + self.prior.utilities(obs)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        index = (
            int(self.rng.choice(len(probs), p=probs))
            if self.training
            else int(np.argmax(probs))
        )
        if self.training and len(probs) > 1:
            self.trace.append((x, h, probs, index))
        return obs.legal_actions[index]

    def finish_round(self, result, seat):
        if not self.training or not result.completed or not self.trace:
            self.last_update = {"decisions": 0, "updated": False}
            self.trace = []
            return
        reward = result.rank_utility(seat) / self.recipe.reward_scale
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        for x, h, probs, chosen in self.trace:
            context = np.append(x.mean(axis=0), 1.0)
            baseline = context @ p["value"]
            advantage = np.clip(reward - baseline, -20, 20)
            d = -probs.copy()
            d[chosen] += 1
            d *= advantage
            # Entropy gradient for a categorical softmax.
            logp = np.log(np.maximum(probs, 1e-12))
            d -= self.recipe.entropy * probs * (logp - np.dot(probs, logp))
            grads["out"] += h.T @ d
            dh = d[:, None] * p["out"][None, :] * (1 - h * h)
            grads["w"] += x.T @ dh
            grads["b"] += dh.sum(axis=0)
            grads["value"] += np.clip(reward - baseline, -20, 20) * context
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = self.recipe.learning_rate / max(len(self.trace), norm, 1)
        for k in p:
            p[k] += grads[k] * scale
            if not np.isfinite(p[k]).all():
                raise FloatingPointError("nonfinite RL model")
        self.last_update = {
            "decisions": len(self.trace),
            "updated": True,
            "reward": reward,
            "gradient_norm": norm,
        }
        self.trace = []

    def save(self, path):
        if self.params is None:
            raise ValueError(
                "cannot save an uninitialized model; run actual training first"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                schema=np.array(FEATURE_VERSION),
                objective=np.array(
                    RANK_OBJECTIVE
                    if self.training
                    else getattr(self, "loaded_objective", "untrained")
                ),
                **self.params,
            )
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)


@dataclass(frozen=True)
class PolicySpec:
    family: str = "rule"
    version: str = "rule-baseline"
    parameters: dict = field(default_factory=dict)
    model_path: str | None = None
    program: str | None = None
    package_path: str | None = None
    package_hash: str | None = None
    action_seconds: float = 3.0
    python_path: str | None = None
    response_seconds: float = 1.0
    runtime_path: str | None = None

    def __post_init__(self):
        if not self.version or len(self.version) > 128:
            raise ValueError("invalid version")
        if self.package_path:
            if (
                not self.package_hash
                or not 0 < self.action_seconds <= 10
                or not 0 < self.response_seconds <= 10
            ):
                raise ValueError(
                    "code policy requires a content hash and finite action budget"
                )
            return
        if self.family not in ("rule", "rl"):
            raise ValueError("unknown builtin policy family")
        if self.program:
            from mahjong_ai.platform.program import compile_program

            if self.family != "rule":
                raise ValueError("only rule policies accept expressions")
            compile_program(self.program)
        if self.family == "rule":
            allowed = {
                "risk_scale",
                "call_bonus",
                "kong_bonus",
                "piao_bonus",
                "hu_bonus",
            }
            if set(self.parameters) - allowed:
                raise ValueError("unsupported rule parameters")
            if any(
                not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 50
                for v in self.parameters.values()
            ):
                raise ValueError("out-of-budget rule parameters")
        else:
            TrainingRecipe(**self.parameters)

    def build(self, training=False):
        if self.package_path:
            if training:
                raise ValueError(
                    "code policies train through train(ctx), not PolicySpec.build"
                )
            from mahjong_ai.execution.candidate_runtime import RemotePolicy

            return RemotePolicy(
                self.package_path,
                self.package_hash,
                self.action_seconds,
                self.python_path,
                self.version,
                self.response_seconds,
                self.runtime_path,
            )
        if self.family == "rule":
            return RulePolicy(
                version=self.version, program=self.program, **self.parameters
            )
        return RLPolicy(
            TrainingRecipe(**self.parameters),
            self.model_path,
            training,
            version=self.version,
        )

    def as_dict(self):
        data = asdict(self)
        if not self.package_path:
            for key in (
                "package_path",
                "package_hash",
                "action_seconds",
                "python_path",
                "response_seconds",
                "runtime_path",
            ):
                data.pop(key)
        return data

    @property
    def identity(self):
        d = self.as_dict()
        d.pop("version")
        if self.package_path:
            from mahjong_ai.execution.candidate_runtime import tree_hash
            import sys

            if tree_hash(self.package_path) != self.package_hash:
                raise ValueError("冻结的候选发生变化")
            d["package_path"] = self.package_hash
            # venv locations are not distinct experiments. The package content
            # includes locked wheel hashes; resolve the shared interpreter identity.
            d["python_path"] = str(Path(self.python_path or sys.executable).resolve())
        from mahjong_ai.persistence.artifacts import runtime_fingerprint

        if self.runtime_path:
            d.pop("runtime_path", None)
            d["runtime"] = tree_hash(self.runtime_path)
        else:
            d["runtime"] = runtime_fingerprint()
        if self.model_path:
            d["model_path"] = hashlib.sha256(
                Path(self.model_path).read_bytes()
            ).hexdigest()
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
