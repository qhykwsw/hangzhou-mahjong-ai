"""PPO update over the shared sample store."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from models.common import symlog


class Batch:
    """Iteration batch resident on the device; minibatches are slices of a permutation."""

    def __init__(self, store, counts, device):
        self.device = device
        ws = np.concatenate([np.full(n, w, dtype=np.int64) for w, n in enumerate(counts)])
        is_ = np.concatenate([np.arange(n, dtype=np.int64) for n in counts])
        self.n = len(ws)
        adv = store["adv"][ws, is_]
        self.adv_mean, self.adv_std = float(adv.mean()), float(adv.std() + 1e-8)
        reward = store["reward"][ws, is_]
        ret = store["ret"][ws, is_]
        self.stats = {
            "reward_mean": float(reward.mean()),
            "reward_abs_mean": float(np.abs(reward).mean()),
            "ret_mean": float(ret.mean()),
            "ret_std": float(ret.std()),
            "value_mean": float(store["value"][ws, is_].mean()),
            "done_frac": float(store["done"][ws, is_].mean()),
            "aux_hand_known": float(store["aux_hand_known"][ws, is_].mean()),
            "aux_game_known": float(store["aux_game_known"][ws, is_].mean()),
        }
        # Compact storage types on the device; converted per minibatch.
        self.t = {}
        for name in ("planes", "scalars", "events", "oracle_planes", "oracle_scalars", "mask", "action",
                     "logprob", "adv", "ret", "aux_hand", "aux_hand_known", "aux_game", "aux_game_known"):
            arr = store[name][ws, is_]
            if name == "events":
                arr = arr.astype(np.int64)
            elif name == "action":
                arr = arr.astype(np.int64)
            self.t[name] = torch.from_numpy(np.ascontiguousarray(arr)).to(device)

    def minibatch(self, idx):
        idx = torch.as_tensor(idx, device=self.device)
        t = {k: v[idx] for k, v in self.t.items()}
        return {
            "planes": t["planes"].float() / 255.0,
            "scalars": t["scalars"].float(),
            "events": t["events"],
            "oracle_planes": t["oracle_planes"].float() / 255.0,
            "oracle_scalars": t["oracle_scalars"].float(),
            "mask": t["mask"],
            "action": t["action"],
            "old_logprob": t["logprob"],
            "adv": (t["adv"] - self.adv_mean) / self.adv_std,
            "ret": t["ret"],
            "aux_hand": t["aux_hand"],
            "aux_hand_known": t["aux_hand_known"].float(),
            "aux_game": t["aux_game"],
            "aux_game_known": t["aux_game_known"].float(),
        }


def explained_variance(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / (var + 1e-8)) if var > 0 else 0.0


def ppo_update(model, optimizer, batch: Batch, cfg, twohot, clip, entropy_coef, rng):
    model.train()
    autocast = cfg.use_autocast and next(model.parameters()).is_cuda
    n = batch.n
    keys = ("loss", "policy_loss", "value_loss", "aux_loss", "entropy", "approx_kl", "clipfrac", "ev_symlog")
    agg = torch.zeros(len(keys), device=batch.device)
    steps = 0
    for epoch in range(cfg.ppo_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch]
            if len(idx) < cfg.minibatch // 4:
                continue
            mb = batch.minibatch(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                logits, value_logits, aux = model(
                    mb["planes"], mb["scalars"], mb["events"], mb["oracle_planes"], mb["oracle_scalars"]
                )
            logits, value_logits, aux = logits.float(), value_logits.float(), aux.float()
            logits = logits.masked_fill(~mb["mask"], -1e9)
            logp = torch.log_softmax(logits, -1)
            logp_a = logp.gather(1, mb["action"][:, None]).squeeze(1)
            ratio = torch.exp(logp_a - mb["old_logprob"])
            adv = mb["adv"]
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            policy_loss = -surr.mean()
            probs = torch.exp(logp)
            entropy = -(probs * logp.masked_fill(~mb["mask"], 0.0)).sum(-1).mean()
            target = twohot.targets(mb["ret"])
            value_loss = -(target * torch.log_softmax(value_logits, -1)).sum(-1).mean()
            aux_t = torch.stack([symlog(mb["aux_hand"]), symlog(mb["aux_game"])], -1)
            aux_w = torch.stack([mb["aux_hand_known"], mb["aux_game_known"]], -1)
            aux_loss = ((aux - aux_t) ** 2 * aux_w).sum() / aux_w.sum().clamp(min=1.0)
            loss = policy_loss + cfg.value_coef * value_loss + cfg.aux_coef * aux_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                clipfrac = ((ratio - 1).abs() > clip).float().mean()
                pred, tgt = symlog(twohot.expectation(value_logits)), symlog(mb["ret"])
                ev = 1.0 - (tgt - pred).var() / (tgt.var() + 1e-8)
                agg += torch.stack([loss.detach(), policy_loss.detach(), value_loss.detach(), aux_loss.detach(),
                                    entropy.detach(), approx_kl, clipfrac, ev])
            steps += 1
    model.eval()
    values = (agg / max(1, steps)).tolist()
    return dict(zip(keys, values)) | {"sgd_steps": steps, **batch.stats}
