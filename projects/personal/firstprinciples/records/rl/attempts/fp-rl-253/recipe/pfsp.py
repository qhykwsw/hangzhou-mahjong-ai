"""Strength-labelled opponents drawn by prioritized fictitious self-play (cfg["pfsp"]; opponents item 5).

The fixed pool keeps its share of the opponent seats (`pool_weight` copies of
each member). The other seats go to frozen versions of both lines (rule
versions and RL snapshots, catalog IDs, no platform proxies), each labelled
by its development-set score on the fixed pool (`versions`, from the catalog
listing when the recipe was written). For each version the learner's
head-to-head p, the chance it finishes a game above that seat (ties half),
starts from the labels, sigmoid((learner - score) / temperature), worth
`prior_games` games, and is updated from every training game's final scores
(ctx.collect's records). Before each collection the `slots` version seats are
dealt in proportion to (1 - p) ** power (AlphaStar's PFSP "hard"), largest
remainders first: versions the learner already beats lose their seats to the
ones it does not, a curriculum that follows the learner.
"""

import math

import numpy as np

MODEL = "model:opponent:"


class Pfsp:
    def __init__(self, pool, spec):
        self.pool = list(pool)
        self.strength = {k: float(v) for k, v in spec["versions"].items()}
        self.pool_weight = int(spec.get("pool_weight", 20))
        self.slots = int(spec.get("slots", 60))
        self.power = float(spec.get("power", 2.0))
        m = float(spec.get("prior_games", 200))
        t = float(spec.get("temperature", 3.0))
        learner = float(spec["learner"])
        self.wins = {k: m / (1.0 + math.exp(-(learner - s) / t)) for k, s in self.strength.items()}
        self.games = {k: m for k in self.strength}
        self.seen = {k: 0 for k in self.strength}

    def p(self, key):
        return self.wins[key] / self.games[key]

    def counts(self):
        keys = list(self.strength)
        w = np.array([(1.0 - self.p(k)) ** self.power for k in keys])
        share = self.slots * w / max(w.sum(), 1e-12)
        n = np.floor(share).astype(int)
        for i in np.argsort(-(share - n))[: self.slots - int(n.sum())]:
            n[i] += 1
        return dict(zip(keys, n.tolist()))

    def opponents(self):
        out = self.pool * self.pool_weight
        for key, n in self.counts().items():
            out += [key] * n
        return out

    def observe(self, records):
        for record in records:
            seats, scores, focal = record.get("seats"), record.get("scores"), record.get("focal")
            if seats is None or scores is None or focal is None:
                continue
            for seat, spec in enumerate(seats):
                key = spec[len(MODEL):] if spec.startswith(MODEL) else spec
                if seat == focal or key not in self.strength:
                    continue
                gap = scores[focal] - scores[seat]
                self.wins[key] += 1.0 if gap > 0 else 0.5 if gap == 0 else 0.0
                self.games[key] += 1.0
                self.seen[key] += 1

    def stats(self):
        counts = self.counts()
        out = {"pfsp_version_seats": sum(counts.values()) / (sum(counts.values()) + len(self.pool) * self.pool_weight)}
        for key in self.strength:
            short = key.replace("fp-", "").replace("-", "_")
            out[f"pfsp_p_{short}"] = self.p(key)
            out[f"pfsp_n_{short}"] = counts[key]
        out["pfsp_seen"] = sum(self.seen.values())
        return out
