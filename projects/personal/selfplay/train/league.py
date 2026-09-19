"""League pool: historical checkpoints (PFSP-weighted) plus fixed opponents."""

import copy
import math
import random
import time
from pathlib import Path

import torch

# 淘汰的快照不再参与抽样，但已开局、座位上仍是它的对局还会来请求动作；
# 保留这么久让它们打完（一场 16 局只要几分钟）。
RETIRED_SECONDS = 3600


class League:
    def __init__(self, cfg, fixed_ids, run_dir, device):
        self.cfg = cfg
        self.fixed_ids = list(fixed_ids)
        self.run_dir = Path(run_dir)
        self.device = device
        self.checkpoints = {}  # slot -> {"id", "path", "created", "games", "wins", "model"}
        self.retired = {}  # slot -> (retired_at, model)
        self.stats = {fid: [0, 0] for fid in fixed_ids}  # id -> [games, wins of current policy]
        self.next_slot = 1
        self.next_id = 0
        self.last_snapshot = time.time()
        (self.run_dir / "league").mkdir(parents=True, exist_ok=True)

    # ---- snapshots ------------------------------------------------------------
    def maybe_snapshot(self, model, force=False):
        if not force and time.time() - self.last_snapshot < self.cfg.league_snapshot_minutes * 60:
            return None
        self.last_snapshot = time.time()
        ident = f"ckpt-{self.next_id:03d}"
        self.next_id += 1
        path = self.run_dir / "league" / f"{ident}.pt"
        torch.save(model.state_dict(), path)
        frozen = copy.deepcopy(model).to(self.device).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        if len(self.checkpoints) >= self.cfg.league_capacity:
            self._evict()
        slot = self.next_slot
        self.next_slot += 1
        self.checkpoints[slot] = {"id": ident, "path": str(path), "created": time.time(), "model": frozen}
        self.stats[ident] = [0, 0]
        return slot

    def _evict(self):
        # Drop the easiest opponent (highest win rate of the current policy),
        # never the newest checkpoint.
        newest = max(self.checkpoints, key=lambda s: self.checkpoints[s]["created"])
        candidates = [s for s in self.checkpoints if s != newest]
        if not candidates:
            return
        victim = max(candidates, key=lambda s: self._winrate(self.checkpoints[s]["id"]))
        self.retired[victim] = (time.time(), self.checkpoints.pop(victim)["model"])

    # ---- stats ------------------------------------------------------------------
    def _winrate(self, ident):
        games, wins = self.stats.get(ident, [0, 0])
        return (wins + 1.0) / (games + 2.0)  # Laplace prior 0.5

    def record_games(self, records):
        for rec in records:
            seats, scores = rec["seats"], rec["scores"]
            mine = [i for i, (k, ident) in enumerate(seats) if k == "model" and ident == 0]
            if not mine:
                continue
            for i in mine:
                for j, (k, ident) in enumerate(seats):
                    if j == i or (k == "model" and ident == 0):
                        continue
                    if k == "model" and ident not in self.checkpoints:
                        continue  # evicted while the game was in flight
                    key = self.checkpoints[ident]["id"] if k == "model" else ident
                    row = self.stats.setdefault(key, [0, 0])
                    row[0] += 1
                    row[1] += 1 if scores[i] > scores[j] else 0

    def opponent_models(self):
        cutoff = time.time() - RETIRED_SECONDS
        for slot in [s for s, (at, _) in self.retired.items() if at < cutoff]:
            del self.retired[slot]
        models = {slot: model for slot, (_, model) in self.retired.items()}
        models.update((slot, c["model"]) for slot, c in self.checkpoints.items())
        return models

    def snapshot(self, elapsed_hours):
        cfg = self.cfg
        frac = min(1.0, elapsed_hours / max(1e-6, cfg.fixed_opponent_anneal_hours))
        fixed_prob = cfg.fixed_opponent_prob_start + frac * (cfg.fixed_opponent_prob_end - cfg.fixed_opponent_prob_start)
        slots, weights = [], []
        for slot, c in self.checkpoints.items():
            slots.append(slot)
            weights.append((1.0 - self._winrate(c["id"])) ** cfg.pfsp_power + 0.02)
        if len(slots) > cfg.league_active_slots:
            # Limit distinct opponent networks per iteration: every extra slot is
            # an extra GPU forward per batch.  Sample without replacement by weight.
            chosen, pool = [], list(zip(slots, weights))
            rng = random.Random(f"{elapsed_hours:.6f}:{self.next_id}")
            for _ in range(cfg.league_active_slots):
                total = sum(w for _, w in pool)
                r, acc = rng.random() * total, 0.0
                for k, (slot, w) in enumerate(pool):
                    acc += w
                    if acc >= r:
                        chosen.append(pool.pop(k))
                        break
            slots, weights = [s for s, _ in chosen], [w for _, w in chosen]
        return {
            "fixed": self.fixed_ids,
            "fixed_weights": None,
            "slots": slots,
            "slot_weights": weights,
            "fixed_prob": fixed_prob if slots else 1.0,
        }

    def summary(self):
        return {
            "checkpoints": [
                {"slot": s, "id": c["id"], "winrate_vs_me": round(self._winrate(c["id"]), 3), "games": self.stats[c["id"]][0]}
                for s, c in sorted(self.checkpoints.items())
            ],
            "fixed": {fid: {"games": self.stats[fid][0], "winrate_vs_me": round(self._winrate(fid), 3)} for fid in self.fixed_ids},
        }

    def state_dict(self):
        return {
            "checkpoints": {s: {k: v for k, v in c.items() if k != "model"} for s, c in self.checkpoints.items()},
            "stats": self.stats,
            "next_slot": self.next_slot,
            "next_id": self.next_id,
        }

    def load_state_dict(self, state, model_factory):
        self.stats = {k: list(v) for k, v in state["stats"].items()}
        for fid in self.fixed_ids:  # 续跑时固定对手名单可能换过
            self.stats.setdefault(fid, [0, 0])
        self.next_slot = state["next_slot"]
        self.next_id = state["next_id"]
        for slot, c in state["checkpoints"].items():
            frozen = model_factory().to(self.device).eval()
            frozen.load_state_dict(torch.load(c["path"], map_location=self.device))
            for p in frozen.parameters():
                p.requires_grad_(False)
            self.checkpoints[int(slot)] = {**c, "model": frozen}
