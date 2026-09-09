import time
import pytest
from mahjong_ai.execution.candidate_runtime import (
    ActionUnavailable,
    RemotePolicy,
    tree_hash,
    train_package,
)
from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai.strategies.seeds import default_opponents


def test_timeout_keeps_worker_and_discards_late_reply(tmp_path):
    (tmp_path / "strategy.py").write_text("""import time
class Policy:
 def finish_round(self, result, seat): pass
 def reset(self, seed): self.calls=0
 def act(self, obs):
  self.calls+=1
  if self.calls==1:
   time.sleep(.1)
   return obs.legal_actions[0]
  return obs.legal_actions[-1]
def build(model): return Policy()
""")
    p = RemotePolicy(tmp_path, tree_hash(tmp_path), action_seconds=0.02)
    hand = HandState(RoundConfig(rounds=1), 123, 0, 1, [0] * 4, [], False)
    obs = hand.observation()
    p.reset(1)
    try:
        pid = p.worker.process.pid
        with pytest.raises(ActionUnavailable):
            p.act(obs)
        time.sleep(0.12)
        assert p.act(obs) == obs.legal_actions[-1]
        assert p.worker.process.pid == pid
        assert p.stats["timeouts"] == 1 and p.stats["errors"] == 0
    finally:
        p.close()


def test_platform_fallback_discards_rightmost_in_insertion_order():
    hand = HandState(RoundConfig(rounds=1), 123, 0, 1, [0] * 4, [], False)
    legal = hand.legal_actions()
    expected = next((a for a in legal if a.action == "hu"), None)
    if expected is None:
        expected = next(
            a
            for tile in reversed(hand.hands[hand.turn])
            for a in legal
            if a.action == "discard" and a.tile == tile
        )
    assert hand.timeout_action() == expected


def test_generic_training_can_compute_artifact_without_rl_games(tmp_path):
    package = tmp_path / "source"
    package.mkdir()
    (package / "strategy.py").write_text("""def train(ctx):
 (ctx.output/'lookup.txt').write_text('computed lookup')
 ctx.checkpoint({'ready':True})
 return {'lookups':1}
""")
    model = tmp_path / "out/model"
    result = train_package(
        package,
        model,
        RoundConfig(rounds=1),
        default_opponents(),
        2,
        3,
        time.monotonic() + 10,
    )
    assert result["episodes"] == []
    assert (model / "lookup.txt").read_text() == "computed lookup"
