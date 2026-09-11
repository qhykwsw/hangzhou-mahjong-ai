"""Small, deterministic, observation-only rules strategy.

This project is intentionally independent from the automatic research state
machine.  The strategy uses only the public ``Observation`` supplied by the
engine and always returns one of its legal actions.
"""

from collections import Counter


class SimpleRules:
    name = "personal-rules"
    version = "v0.1.0"

    def reset(self, seed: int):
        self.seed = int(seed)

    def finish_round(self, result, seat: int):
        # Keep the policy stateless between rounds.  The result is deliberately
        # not used: it is not part of the observation available during play.
        return None

    def act(self, observation):
        legal = tuple(observation.legal_actions)
        if not legal:
            raise ValueError("observation has no legal actions")

        # Taking a guaranteed win is always preferred.
        for action in legal:
            if action.action == "hu":
                return action

        # Calls are useful when available, while pass remains the safe fallback
        # for response phases.  Preserve engine order for equal choices.
        priority = {"gang": 0, "peng": 1, "chi": 2, "discard": 3, "pass": 4}
        best_priority = min(priority.get(a.action, 5) for a in legal)
        candidates = [a for a in legal if priority.get(a.action, 5) == best_priority]

        if candidates and candidates[0].action == "discard":
            # Discard a tile with the smallest visible multiplicity first.  The
            # tile id is a stable tie-breaker, making matches reproducible.
            counts = Counter(observation.hand)
            return min(candidates, key=lambda a: (counts[a.tile], a.tile or -1))
        return candidates[0]


def build(model_dir):
    """Return the frozen strategy instance expected by the release contract."""
    policy = SimpleRules()
    policy.reset(0)
    return policy
