"""Gain-weighted training games (cfg["regret"]; opponents digest #7, MAESTRO).

The step's direction is mostly noise (chunk_cosine ~0.1), so where the games
are played could matter: games whose deviations show more to gain carry more
signal per hand. Each opponent (a fixed-pool member or a lineage version) and
each opening set gets a gain from the paired deviation differences A (luck
taken off) of its games (see update), and seats draw opponents in proportion
to their base share times exp(temperature * z), z the gain's distance from
the others' in standard errors, clipped to [floor, 1 / floor]; opening calls
draw their set the same way. Every game's rows carry the importance weight
prod_seats base / drawn (and base / drawn for its opening set), so the
objective stays the base distribution's mean; the clip bounds the weights.
"""

from collections import Counter

import numpy as np

MODEL = "model:opponent:"


def shrink(z):
    """z less one standard error toward 0: differences within the noise leave the draw alone."""
    return np.sign(z) * np.maximum(np.abs(z) - 1.0, 0.0)


class RegretSampler:
    def __init__(self, base, sets, spec):
        counts = Counter(base)
        total = sum(counts.values())
        self.members = sorted(counts)
        self.base = {m: counts[m] / total for m in self.members}
        self.sets = list(sets)
        self.set_base = 1.0 / max(1, len(self.sets))
        self.floor = float(spec.get("floor", 0.5))
        self.temperature = float(spec.get("temperature", 0.5))
        self.xtx = self.xty = None
        self.smooth = float(spec.get("smooth", 0.5))
        self.slots = int(spec.get("slots", 120))
        self.regret = {}
        self.drawn = dict(self.base)
        self.set_drawn = {s: self.set_base for s in self.sets}
        self.active = False

    def opponents(self):
        """The opponent list (duplicates weigh) the seats draw from."""
        share = self.drawn if self.active else self.base
        out = []
        for m in self.members:
            out += [m] * max(1, int(round(share[m] * self.slots)))
        counts = Counter(out)
        total = sum(counts.values())
        self.drawn = {m: counts[m] / total for m in self.members}
        return out

    def pick_set(self, rng):
        names = list(self.sets)
        p = np.array([self.set_drawn[s] for s in names])
        return names[int(rng.choice(len(names), p=p / p.sum()))]

    def mark(self, rows, records, opening=None):
        """Tag each game's rows with its opponents, opening set and importance weight (records in row order)."""
        groups, start = [], 0
        for i in range(1, len(rows) + 1):
            if i == len(rows) or rows[i].get("game_mark") != rows[start].get("game_mark"):
                groups.append((start, i))
                start = i
        if len(groups) != len(records):
            for r in rows:  # cannot pair rows with games: no weighting for this call
                r["regret_is"], r["regret_seats"], r["regret_set"] = 1.0, (), opening or "game"
            return False
        for (a, b), record in zip(groups, records):
            seats = [s[len(MODEL):] if s.startswith(MODEL) else s for k, s in enumerate(record["seats"])
                     if k != record["focal"]]
            weight = 1.0
            for s in seats:
                if s in self.base:
                    weight *= self.base[s] / max(self.drawn.get(s, self.base[s]), 1e-12)
            if opening is not None:
                weight *= self.set_base / max(self.set_drawn.get(opening, self.set_base), 1e-12)
            for r in rows[a:b]:
                r["regret_is"], r["regret_seats"], r["regret_set"] = weight, tuple(seats), opening or "game"
        return True

    def update(self, rows, raw, deviated):
        """Fold this rollout's deviations into the cells' gains, then set the draw for the next one.

        221's regret, the mean of max(A, 0), never moved the draw: with A's
        noise (sd ~14 base units) every cell's mean sat at the noise floor
        sigma * phi(0) ~ 2.0 (spread 1.07), and each deviation was credited
        to all three opponents at its table, which dilutes a member's own part
        threefold. Here a member's gain is its coefficient in the least
        squares fit A ~ sum over the table's three seats of theta[seat] (the
        mean of A carries no noise floor; the fit takes the other two seats
        out), an opening set's the mean A of its games. Each is compared with
        the members' (sets') mean in units of its own standard error, z, and
        the draw multiplies the base share by exp(temperature * shrink(z)), clipped
        to [floor, 1 / floor]: always active, so a cell whose deviations show
        more to gain gets more games in proportion to the evidence for it.
        The fit's sums decay by `smooth` per rollout.
        """
        index = {m: k for k, m in enumerate(self.members)}
        n = len(self.members)
        xtx, xty = np.zeros((n, n)), np.zeros(n)
        set_sum, set_sq, set_n = Counter(), Counter(), Counter()
        for i in np.flatnonzero(deviated):
            a = float(raw[i])
            x = np.zeros(n)
            for s in rows[i].get("regret_seats", ()):
                if s in index:
                    x[index[s]] += 1.0
            if x.any():
                xtx += np.outer(x, x)
                xty += x * a
            key = rows[i].get("regret_set", "game")
            set_sum[key] += a
            set_sq[key] += a * a
            set_n[key] += 1
        decay = self.smooth
        self.xtx = xtx if getattr(self, "xtx", None) is None else decay * self.xtx + xtx
        self.xty = xty if getattr(self, "xty", None) is None else decay * self.xty + xty
        # The deviations' own spread, for the standard errors.
        count = max(1, sum(set_n.values()))
        sigma2 = max(sum(set_sq.values()) / count - (sum(set_sum.values()) / count) ** 2, 1e-9)
        stats = {}
        temperature = float(getattr(self, "temperature", 0.5))
        ridge = 1e-6 * max(float(np.trace(self.xtx)), 1.0)
        try:
            cov = np.linalg.inv(self.xtx + ridge * np.eye(n))
        except np.linalg.LinAlgError:
            cov = None
        if cov is not None and np.diag(self.xtx).min() > 0:
            theta = cov @ self.xty
            se = np.sqrt(np.maximum(np.diag(cov) * sigma2, 1e-12))
            z = (theta - theta.mean()) / se
            factor = np.clip(np.exp(temperature * shrink(z)), self.floor, 1 / self.floor)
            raw_q = {m: self.base[m] * float(factor[index[m]]) for m in self.members}
            total = sum(raw_q.values())
            self.drawn = {m: raw_q[m] / total for m in self.members}
            self.active = True
            for m in self.members:
                stats["regret_" + m] = float(theta[index[m]])
                stats["regret_z_" + m] = float(z[index[m]])
            stats["regret_factor_max"] = float(factor.max())
            stats["regret_factor_min"] = float(factor.min())
        # Opening sets: their games' own mean A (one set per game, nothing to take out).
        for key in set_n:
            mean = set_sum[key] / set_n[key]
            old = self.regret.get("set:" + key)
            self.regret["set:" + key] = (mean, set_n[key]) if old is None else (
                decay * old[0] + (1 - decay) * mean, decay * old[1] + set_n[key])
        sets = [s for s in self.sets if "set:" + s in self.regret]
        if sets and len(sets) == len(self.sets):
            means = np.array([self.regret["set:" + s][0] for s in sets])
            ses = np.sqrt(sigma2 / np.maximum([self.regret["set:" + s][1] for s in sets], 1.0))
            z = (means - means.mean()) / ses
            factor = np.clip(np.exp(temperature * shrink(z)), self.floor, 1 / self.floor)
            raw_q = {s: self.set_base * float(f) for s, f in zip(sets, factor)}
            total = sum(raw_q.values())
            self.set_drawn = {s: raw_q[s] / total for s in sets}
            for s, m, zz in zip(sets, means, z):
                stats["regret_set_" + s] = float(m)
                stats["regret_set_z_" + s] = float(zz)
        stats["regret_active"] = float(self.active)
        stats["regret_sigma"] = float(np.sqrt(sigma2))
        return stats
