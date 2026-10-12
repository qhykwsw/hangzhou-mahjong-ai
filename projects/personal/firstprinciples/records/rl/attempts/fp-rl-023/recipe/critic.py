"""A small observation-plus-oracle Critic, fitted fresh for every rollout."""

import numpy as np
import torch
from torch import nn

from guided.networks import HistoryEncoder

from .batch import actor_args, make_batch


class Critic(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.encoder = HistoryEncoder(dim, 2, 4, state_only=True)
        self.side = nn.Sequential(nn.Linear(16 * 34 + 4, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.GELU())
        self.out = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, mb):
        state, _ = self.encoder(*actor_args(mb))
        side = self.side(torch.cat([mb["oracle_planes"].flatten(1), mb["oracle_scalars"]], -1))
        return self.out(torch.cat([state, side], -1)).squeeze(-1)


def fit_predict(rows, target, fit_idx, pred_idx, cfg, rng, device, seed, *, batch_source=None, model=None):
    """One epoch on fit_idx, then predictions for pred_idx (in target units).

    The oracle side identifies a hand, so more epochs memorise hands and the
    held-out fit gets worse; one epoch is the measured optimum. A given model
    keeps its weights from earlier rollouts: those were other games, so it
    trains one more epoch on fresh data without seeing pred_idx's returns.
    """
    torch.manual_seed(seed)
    model = model if model is not None else Critic().to(device)
    steps = None
    if torch.device(device).type == "cuda":
        from .critic_steps import CriticSteps

        steps = CriticSteps(model, cfg["critic_learning_rate"])
    opt = None if steps else torch.optim.Adam(model.parameters(), lr=cfg["critic_learning_rate"], eps=1e-5)
    mean, sd = target[fit_idx].mean(), max(target[fit_idx].std(), 1e-6)
    size = int(cfg["critic_batch"])
    model.train()
    order = rng.permutation(fit_idx)
    for s in range(0, len(order), size):
        b = order[s : s + size]
        goal = torch.as_tensor((target[b] - mean) / sd, device=device, dtype=torch.float32)
        batch = (batch_source.batch(b, oracle=True) if batch_source is not None
                 else make_batch([rows[i] for i in b], device, oracle=True))
        if steps is not None:
            steps(batch, goal)
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(batch)
        loss = ((pred.float() - goal) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    # Captured buffers are no longer needed during held-out prediction.
    if steps is not None:
        torch.cuda.synchronize()
        del steps
    model.eval()
    # Predict in history-length order to limit padding, then restore order.
    order = pred_idx[np.argsort([len(rows[i]["events"]) for i in pred_idx], kind="stable")]
    pred = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), 512):
            b = order[s : s + 512]
            batch = (batch_source.batch(b, oracle=True) if batch_source is not None
                     else make_batch([rows[i] for i in b], device, oracle=True))
            out = model(batch).float().cpu().numpy()
            pred.update(zip(b.tolist(), out))
    return np.array([pred[i] for i in pred_idx.tolist()]) * sd + mean
