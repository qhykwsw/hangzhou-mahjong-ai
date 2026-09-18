"""Terminal-return regressors and clipped PPO, implemented with explicit trajectories."""

from pathlib import Path
import json
import numpy as np
from mahjong_ai.strategies.common import RulePolicy
from mahjong_ai.sdk.trajectory import Decision, features, softmax


class LearnedPolicy:
    name = "learned"

    def __init__(
        self,
        algorithm,
        model_dir,
        training=False,
        hidden=32,
        learning_rate=0.001,
        clip=0.2,
        epochs=4,
        batch_size=64,
        entropy=0.01,
        rollout_games=8,
        boosting_rounds=32,
        num_leaves=15,
    ):
        if algorithm not in ("ppo", "mlp", "lightgbm"):
            raise ValueError("unknown learning algorithm")
        if any(
            type(v) is not int or v < 1
            for v in (
                hidden,
                epochs,
                batch_size,
                rollout_games,
                boosting_rounds,
                num_leaves,
            )
        ):
            raise ValueError("positive model and training dimensions required")
        if not 0 < learning_rate <= 1 or not 0 < clip < 1 or not 0 <= entropy <= 1:
            raise ValueError("invalid learning parameters")
        self.rollout_games = rollout_games
        self.boosting_rounds = boosting_rounds
        self.num_leaves = num_leaves
        self.algorithm = algorithm
        self.model_dir = Path(model_dir)
        self.training = training
        self.hidden = hidden
        self.lr = learning_rate
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy = entropy
        self.prior = RulePolicy()
        self.params = None
        self.trace = []
        self.buffer = []
        self.updates = 0
        self.rng = np.random.default_rng(0)
        self.booster = None
        path = self.model_dir / "weights.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                self.params = {k: data[k].copy() for k in data.files}
        if self.params is not None and self.params["w"].shape[1] != hidden:
            if training:
                self.params = None
            else:
                raise ValueError("model hidden dimension differs from configuration")
        metadata = self.model_dir / "training.json"
        if metadata.exists():
            self.updates = json.loads(metadata.read_text()).get("updates", 0)
        if algorithm == "lightgbm" and (self.model_dir / "model.txt").exists():
            import lightgbm as lgb

            self.booster = lgb.Booster(model_file=str(self.model_dir / "model.txt"))
        if not training and self.params is None and self.booster is None:
            raise ValueError("trained model required")

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.trace = []

    def initialize(self, x):
        if self.params is None:
            d = x.shape[1]
            self.params = {
                "w": self.rng.normal(0, 0.05, (d, self.hidden)),
                "b": np.zeros(self.hidden),
                "actor": self.rng.normal(0, 0.05, self.hidden),
                "value": np.zeros(d + 1),
            }

    def forward(self, x):
        self.initialize(x)
        h = np.tanh(x @ self.params["w"] + self.params["b"])
        return h, h @ self.params["actor"]

    def act(self, obs):
        x = features(obs)
        h, pred = self.forward(x)
        if self.booster is not None:
            pred = self.booster.predict(x, num_threads=1)
        logits = pred + self.prior.utilities(obs) * (
            1.0 if self.algorithm == "ppo" else 0.1
        )
        probs = softmax(logits)
        if self.training:
            if self.algorithm != "ppo":
                probs = 0.8 * probs + 0.2 / len(probs)
            selected = int(self.rng.choice(len(probs), p=probs))
        else:
            selected = int(np.argmax(logits))
        value = float(np.append(x.mean(axis=0), 1) @ self.params["value"])
        if self.training:
            self.trace.append(
                Decision(x, selected, float(np.log(max(probs[selected], 1e-12))), value)
            )
        return obs.legal_actions[selected]

    def finish_round(self, result, seat):
        if self.training and result.completed:
            for row in self.trace:
                # 主口径是桌面分（平台按总分排序），不是名次效用。
                row.reward = result.score(seat)
            self.buffer.extend(self.trace)
        self.trace = []

    def update(self):
        if not self.buffer:
            return
        if self.algorithm == "lightgbm":
            import lightgbm as lgb

            x = np.asarray([r.features[r.action] for r in self.buffer])
            y = np.asarray([r.reward for r in self.buffer])
            self.booster = lgb.train(
                {
                    "objective": "regression",
                    "verbosity": -1,
                    "num_threads": 1,
                    "min_data_in_leaf": 5,
                    "learning_rate": 0.05,
                    "num_leaves": self.num_leaves,
                },
                lgb.Dataset(x, label=y),
                num_boost_round=self.boosting_rounds,
                init_model=self.booster,
            )
        else:
            for _ in range(self.epochs):
                order = self.rng.permutation(len(self.buffer))
                for start in range(0, len(order), self.batch_size):
                    gradients = {k: np.zeros_like(v) for k, v in self.params.items()}
                    batch = order[start : start + self.batch_size]
                    for i in batch:
                        row = self.buffer[i]
                        x = row.features
                        h, pred = self.forward(x)
                        if self.algorithm == "ppo":
                            # Store the rule prior as part of immutable behavior logits.
                            probs = softmax(
                                pred + getattr(row, "prior", np.zeros(len(pred)))
                            )
                            ratio = np.exp(
                                np.log(max(probs[row.action], 1e-12))
                                - row.log_probability
                            )
                            advantage = row.reward - row.value
                            active = (advantage >= 0 and ratio <= 1 + self.clip) or (
                                advantage < 0 and ratio >= 1 - self.clip
                            )
                            d = -probs
                            d[row.action] += 1
                            d *= advantage * ratio if active else 0
                            logs = np.log(np.maximum(probs, 1e-12))
                            d -= self.entropy * probs * (logs - np.dot(probs, logs))
                        else:
                            d = np.zeros(len(pred))
                            d[row.action] = row.reward - pred[row.action]
                        gradients["actor"] += h.T @ d
                        dh = d[:, None] * self.params["actor"] * (1 - h * h)
                        gradients["w"] += x.T @ dh
                        gradients["b"] += dh.sum(axis=0)
                        context = np.append(x.mean(axis=0), 1)
                        gradients["value"] += (
                            row.reward - context @ self.params["value"]
                        ) * context
                    norm = max(
                        1.0,
                        np.sqrt(sum(float((g * g).sum()) for g in gradients.values())),
                    )
                    for k, g in gradients.items():
                        self.params[k] += self.lr * g / max(len(batch), norm)
        self.updates += 1
        self.buffer = []

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.params is None:
            raise ValueError("no real training decisions")
        np.savez_compressed(folder / "weights.npz", **self.params)
        if self.booster is not None:
            self.booster.save_model(str(folder / "model.txt"))
        (folder / "training.json").write_text(
            json.dumps(
                {
                    "algorithm": self.algorithm,
                    "updates": self.updates,
                    "label_source": "executed-terminal-return",
                }
            )
        )


class PPOPolicy(LearnedPolicy):
    def act(self, obs):
        action = super().act(obs)
        if self.training:
            self.trace[-1].prior = self.prior.utilities(obs).copy()
        return action


def train(ctx, algorithm, parameters):
    cls = PPOPolicy if algorithm == "ppo" else LearnedPolicy
    folder = ctx.output if ctx.completed_episodes else ctx.initial
    learner = cls(algorithm, folder, True, **parameters)
    completed = ctx.completed_episodes
    while completed < ctx.episodes:
        count = min(learner.rollout_games, ctx.episodes - completed)
        ctx.play_many(learner, count)
        completed += count
        learner.update()
        learner.save(ctx.output)
        ctx.checkpoint(
            {"episode": completed, "algorithm": algorithm, "updates": learner.updates}
        )
    return {
        "algorithm": algorithm,
        "updates": learner.updates,
        "episodes": ctx.episodes,
    }
