"""Regret-weighted training games (cfg["regret"]; opponents digest #7, MAESTRO).

The step's direction is mostly noise (chunk_cosine ~0.1), so where the games
are played could matter: games whose deviations show more to gain carry more
signal per hand. Each opponent (a fixed-pool member or a lineage version) and
each opening set gets a regret, the mean of max(A, 0) over the deviations of
the games it sat in (A the paired difference, luck taken off), smoothed over
steps. Seats then draw opponents in proportion to their base share times
regret / mean regret, clipped to [floor, 1 / floor], and opening calls draw
their set the same way. Every game's rows carry the importance weight
prod_seats base / drawn (and base / drawn for its opening set), so the
objective stays the base distribution's mean; the clip bounds the weights.
Regrets within `flat` of each other (max / min) leave the base draw as it is:
then nothing here differs from the base recipe.
"""

from collections import Counter

import numpy as np

MODEL = "model:opponent:"


class RegretSampler:
    def __init__(self, base, sets, spec):
        counts = Counter(base)
        total = sum(counts.values())
        self.members = sorted(counts)
        self.base = {m: counts[m] / total for m in self.members}
        self.sets = list(sets)
        self.set_base = 1.0 / max(1, len(self.sets))
        self.floor = float(spec.get("floor", 0.5))
        self.flat = float(spec.get("flat", 1.5))
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
        """Fold this rollout's deviations into the regrets, then set the draw for the next one."""
        sums, counts = Counter(), Counter()
        for i in np.flatnonzero(deviated):
            gain = max(float(raw[i]), 0.0)
            for s in rows[i].get("regret_seats", ()):
                sums[s] += gain
                counts[s] += 1
            key = "set:" + rows[i].get("regret_set", "game")
            sums[key] += gain
            counts[key] += 1
        for key in counts:
            value = sums[key] / counts[key]
            old = self.regret.get(key)
            self.regret[key] = value if old is None else self.smooth * old + (1 - self.smooth) * value
        stats = {}
        members = [m for m in self.members if m in self.regret]
        if len(members) >= 2:
            values = np.array([self.regret[m] for m in members])
            spread = float(values.max() / max(values.min(), 1e-12))
            stats["regret_spread"] = spread
            self.active = spread >= self.flat
            if self.active:
                mean = float(values.mean())
                factor = {m: float(np.clip(self.regret[m] / mean, self.floor, 1 / self.floor)) for m in members}
                raw_q = {m: self.base[m] * factor.get(m, 1.0) for m in self.members}
                z = sum(raw_q.values())
                self.drawn = {m: raw_q[m] / z for m in self.members}
        sets = [s for s in self.sets if "set:" + s in self.regret]
        if self.active and len(sets) == len(self.sets) and sets:
            values = np.array([self.regret["set:" + s] for s in sets])
            mean = float(values.mean())
            raw_q = {s: self.set_base * float(np.clip(self.regret["set:" + s] / mean, self.floor, 1 / self.floor))
                     for s in sets}
            z = sum(raw_q.values())
            self.set_drawn = {s: raw_q[s] / z for s in sets}
            stats["regret_set_spread"] = float(values.max() / max(values.min(), 1e-12))
        stats["regret_active"] = float(self.active)
        for m in members:
            stats["regret_" + m] = self.regret[m]
        return stats
