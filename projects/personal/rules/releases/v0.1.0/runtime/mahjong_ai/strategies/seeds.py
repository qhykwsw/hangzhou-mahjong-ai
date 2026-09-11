"""Small starting players; not a fixed evaluation target."""

from mahjong_ai.strategies.common import PolicySpec


def default_opponents():
    return (
        PolicySpec(version="balanced"),
        PolicySpec(version="conservative", parameters={"risk_scale": 1.8}),
        PolicySpec(version="attacking", parameters={"risk_scale": 0.4}),
    )


def seed_sources():
    source = "from mahjong_ai.strategies.common import RulePolicy\ndef build(model): return RulePolicy({})\n"
    return {
        "均衡": source.format(""),
        "稳健": source.format("risk_scale=1.8"),
        "进攻": source.format("risk_scale=.4"),
        "随机": """import random
class RandomPolicy:
    def reset(self,seed): self.rng=random.Random(seed)
    def act(self,obs): return self.rng.choice(obs.legal_actions)
    def finish_round(self,result,seat): pass
def build(model): return RandomPolicy()
""",
    }
