"""Training loop: collect, update, repeat until the harness deadline."""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from .critic import Critic
from .learning import _critic_args, make_batch, update

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
REPLAY_KEYS = ("planes", "scalars", "events", "mask", "teacher", "label_version")


def train(ctx):
    cfg = dict(CONFIG)
    cfg["workers"] = max(
        1, int(ctx.resources["cpu_capacity"] * cfg.pop("worker_fraction"))
    )
    if ctx.resources["gpu_total_bytes"] < 16 * 2**30:
        cfg["minibatch"] = min(cfg["minibatch"], 64)
    actor = ctx.actor
    critic = Critic().to(ctx.device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=cfg["critic_learning_rate"], eps=1e-5
    )
    replay = deque(maxlen=cfg["replay_capacity"])
    rng = np.random.default_rng(ctx.seed)

    def values(rows):
        critic.eval()
        with torch.no_grad():
            return (
                critic(*_critic_args(make_batch(rows, ctx.device)))
                .float()
                .cpu()
                .numpy()
            )

    iteration = 0
    while ctx.time_left() > 0:
        iteration += 1
        rows, records, resources = ctx.collect(
            actor,
            cfg,
            samples=cfg["samples"],
            iteration=iteration,
            labels=cfg["lambda_max"] > 0,
            hooks={"values": values},
        )
        if not rows or ctx.time_left() <= 0:
            break
        stats = update(
            actor,
            critic,
            actor_opt,
            critic_opt,
            rows,
            cfg | {"deadline": ctx.deadline},
            1.0,
            list(replay),
            rng,
        )
        replay.extend({k: r[k] for k in REPLAY_KEYS} for r in rows if r["teacher"] >= 0)
        ctx.log(
            **stats,
            iteration=iteration,
            games=len(records),
            collect_seconds=resources["seconds"],
        )
        ctx.save_actor(actor)
    return actor
