import json
import time
from pathlib import Path

import pytest

from mahjong_ai.candidate_runtime import CandidateProcess, RemotePolicy, tree_hash
from mahjong_ai.engine import HandState
from mahjong_ai.research_agent import codex_command
from mahjong_ai.storage import atomic_json, read_json
from mahjong_ai.types import RoundConfig


def test_candidate_cannot_read_host_files_or_network(tmp_path):
    secret=tmp_path/'private.txt';secret.write_text('secret')
    package=tmp_path/'package';package.mkdir()
    source=f'''from pathlib import Path
import socket
from mahjong_ai.policy import RulePolicy
def build(model):
    try: Path({str(secret)!r}).read_text()
    except PermissionError: pass
    else: raise AssertionError("private read allowed")
    try: socket.socket().connect(("127.0.0.1", 9))
    except PermissionError: pass
    else: raise AssertionError("network allowed")
    return RulePolicy()
'''
    (package/'strategy.py').write_text(source)
    worker=CandidateProcess(package)
    try:assert worker.call({'op':'build'})=={'ok':True}
    finally:worker.close()


def test_slow_policy_times_out_and_falls_back(tmp_path):
    (tmp_path/'strategy.py').write_text('from mahjong_ai.policy import RulePolicy\nclass Slow(RulePolicy):\n def act(self,obs):\n  while True: pass\ndef build(model):return Slow()\n')
    p=RemotePolicy(str(tmp_path),tree_hash(tmp_path),action_seconds=.02)
    hand=HandState(RoundConfig(rounds=1),123,0,1,[0]*4,[],False)
    obs=hand.observation();p.reset(1)
    started=time.monotonic()
    from mahjong_ai.candidate_runtime import ActionUnavailable
    with pytest.raises(ActionUnavailable): p.act(obs)
    assert time.monotonic()-started<1
    assert p.stats['timeouts']==1
    assert p.stats['errors']==0
    p.close()


def test_codex_only_receives_research_tools(tmp_path):
    command=codex_command(tmp_path, "run-id", "rule")
    assert '--ignore-user-config' in command
    assert 'features.shell_tool=false' in command
    assert 'features.unified_exec=false' in command
    assert not any('mcp_servers.playwright' in x for x in command)


def test_training_checkpoint_recovers_same_model(tmp_path):
    from mahjong_ai.candidate_runtime import train_package
    from mahjong_ai.seeds import default_opponents
    import numpy as np
    package=tmp_path/'package';package.mkdir()
    (package/'strategy.py').write_text('''from mahjong_ai.policy import RLPolicy,TrainingRecipe
import time
def train(ctx):
    path=ctx.output/'model.npz'
    learner=RLPolicy(TrainingRecipe(),path if ctx.completed_episodes else None,training=True)
    start=ctx.completed_episodes
    for i in range(start,ctx.episodes):
        ctx.play(learner)
        learner.save(path)
        ctx.checkpoint({'index':i})
        if start==0 and i==0:time.sleep(2)
''')
    out=tmp_path/'partial'/'model'
    with pytest.raises(TimeoutError):
        train_package(package,out,RoundConfig(rounds=1),default_opponents(),3,19,time.monotonic()+.5)
    progress=read_json(out.parent/'training-checkpoint.json')
    assert len(progress['episodes'])==1
    resumed=train_package(package,out,RoundConfig(rounds=1),default_opponents(),3,19,time.monotonic()+10)
    whole=tmp_path/'whole'/'model'
    direct=train_package(package,whole,RoundConfig(rounds=1),default_opponents(),3,19,time.monotonic()+10)
    assert resumed['episodes']==direct['episodes']
    with np.load(out/'model.npz') as a,np.load(whole/'model.npz') as b:
        for key in ('w','b','out','value'):np.testing.assert_array_equal(a[key],b[key])
