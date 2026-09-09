import json
import time
from pathlib import Path

import pytest

from mahjong_ai.candidate_runtime import CandidateProcess, RemotePolicy, tree_hash
from mahjong_ai.engine import HandState
from mahjong_ai.research import Research
from mahjong_ai.research_cli import codex_command
from mahjong_ai.storage import atomic_json, read_json
from mahjong_ai.types import RoundConfig


def wait_batch(r,seconds=30):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        s=r.status()
        if s['batches'][-1]['status']!='running':return s
        time.sleep(.05)
    raise AssertionError('batch did not finish')


def bootstrap(tmp_path,family='rule'):
    r=Research(tmp_path);r.initialize(family,RoundConfig(rounds=1),seconds=120,workers=2)
    r.begin();r.submit_batch(['initial'],pairs=2,training_episodes=3,league_samples=0)
    wait_batch(r);r.decide('batch-0000',['initial'],'seed baseline completed; no improvement claimed')
    return r


def test_code_population_barrier_history_and_ancestry(tmp_path):
    r=bootstrap(tmp_path)
    for name,source in [('a','from mahjong_ai.policy import RulePolicy\ndef build(model): return RulePolicy(risk_scale=.3)\n'),
                        ('broken','raise RuntimeError("intentional failure")\n')]:
        r.create_candidate(name,'test alternative',parent='initial')
        r.candidate_file(name,'strategy.py',source)
    r.submit_batch(['a','broken'],pairs=3,league_samples=4)
    with pytest.raises(ValueError,match='batch'):
        r.create_candidate('early','must wait')
    with pytest.raises(ValueError,match='barrier'):
        r.decide('batch-0001',['initial'],'premature')
    wait_batch(r)
    result=r.details('batch','batch-0001')
    assert result['candidates']['a']['status']=='completed'
    assert result['candidates']['broken']['status']=='failed'
    assert result['league']['samples']%len(result['league']['table'])==0
    r.decide('batch-0001',['initial','a'],'retain a; broken worker failed, not evidence of weaker play')
    r.create_candidate('child','combine old reference',parent='a',references=['initial'])
    assert r.details('candidate','child')['parent']=='a'
    assert r.history('intentional failure')['total']>=1
    trace=r.details('trace','a',limit=1)['items'][0]['observation']
    assert 'initial_hands' not in json.dumps(trace)
    assert 'seed' not in trace
    r.end()


def test_training_contract_uses_real_games_and_frozen_model(tmp_path):
    r=bootstrap(tmp_path,'rl')
    c=r.details('batch','batch-0000')['candidates']['initial']
    assert len(c['training']['episodes'])==3
    assert all(0<=e['rank_utility']<=3 for e in c['training']['episodes'])
    assert Path(c['spec']['package_path'],'model','model.npz').exists()
    r.end()


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
    assert p.act(obs)==obs.legal_actions[0]
    assert time.monotonic()-started<1
    assert p.stats['timeouts']==1
    assert p.stats['errors']==0
    p.close()


def test_budget_resume_and_extensions_do_not_reset_history(tmp_path):
    r=Research(tmp_path);r.initialize('rule',seconds=10,acceptance_slots=2)
    r.begin();time.sleep(.02);r.end()
    before=r.status()['seconds_remaining'];r.begin();r.end()
    assert r.status()['seconds_remaining']<=before<10
    run_id=r.status()['run_id'];r.add_budget(20,3)
    assert r.status()['run_id']==run_id
    assert r.status()['acceptance_remaining']==5
    s=read_json(r.state_path)
    assert sum(e['alpha']*e['slots'] for e in s['acceptance_epochs'])<.05


def test_paths_and_frozen_mutation_are_rejected(tmp_path):
    r=bootstrap(tmp_path)
    with pytest.raises(ValueError,match='relative'):r.candidate_file('initial','../run.json')
    with pytest.raises(ValueError,match='frozen'):r.candidate_file('initial','strategy.py','bad')
    c=r.details('candidate','initial')
    Path(c['spec']['package_path'],'strategy.py').write_text('changed')
    from mahjong_ai.policy import PolicySpec
    with pytest.raises(ValueError,match='changed'):PolicySpec(**c['spec']).build()
    r.end()


def test_codex_only_receives_research_tools(tmp_path):
    command=codex_command(tmp_path)
    assert '--ignore-user-config' in command
    assert 'features.shell_tool=false' in command
    assert 'features.unified_exec=false' in command
    assert not any('mcp_servers.playwright' in x for x in command)


def test_time_only_default_and_removing_cap_preserves_time(tmp_path):
    r=Research(tmp_path/'default');r.initialize('rule',seconds=3600)
    assert r.status()['acceptance_remaining'] is None
    r.add_budget(seconds=5)
    assert r.status()['acceptance_remaining'] is None
    capped=Research(tmp_path/'capped');capped.initialize('rule',seconds=3600,acceptance_slots=2)
    before=capped._read()
    capped.add_budget(unlimited=True)
    after=capped._read()
    assert after['acceptance_granted'] is None
    assert after['seconds_granted']==before['seconds_granted']
    assert after['acceptance_used']==before['acceptance_used']


def test_many_acceptances_do_not_hit_bootstrap_resolution_cap():
    from mahjong_ai.evaluator import EvaluationReport
    from mahjong_ai.types import RANK_OBJECTIVE
    protocol={'engine':'test','pairs':1000,'objective':RANK_OBJECTIVE,'config':RoundConfig().as_dict()}
    side={'score':0,'rank':2,'rank_utility':2,'first':False,'top_two':True,'rounds_played':1}
    rows=[{'completed':True,'rounds_started':2,'delta':1.,'baseline':side,'candidate':side} for _ in range(1000)]
    summary=EvaluationReport(protocol,rows,alpha=1e-12).summary()
    assert 'Hoeffding' in summary['uncertainty_method']
    assert summary['bootstrap_interval'][0]>0


def test_training_checkpoint_recovers_same_model(tmp_path):
    from mahjong_ai.candidate_runtime import train_package
    from mahjong_ai.iteration import default_opponents
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


def test_acceptance_deduplicates_and_never_returns_private_records(tmp_path,monkeypatch):
    r=bootstrap(tmp_path)
    r.create_candidate('new','candidate')
    r.candidate_file('new','strategy.py','from mahjong_ai.policy import RulePolicy\ndef build(model):return RulePolicy(risk_scale=.3)\n')
    r.submit_batch(['new'],pairs=2,league_samples=0);wait_batch(r)
    r.decide('batch-0001',['initial','new'],'development complete; submit independent evidence')
    monkeypatch.setattr(r,'_launch',lambda *args:None)
    a=r.accept('new');task=read_json(tmp_path/'acceptance'/a['id']/'task.json')
    assert task['pairs']==1000
    atomic_json(tmp_path/'acceptance'/a['id']/'public.json',{'status':'inconclusive','summary':{'mean_rank_utility_delta':0}})
    before=r.status()['acceptance_remaining']
    assert r.accept('new')['id']==a['id']
    assert r.status()['acceptance_remaining']==before
    assert str(task['seed']) not in json.dumps(r.history())
    assert 'private' not in json.dumps(r.details('acceptance',a['id']))
    r.end()


def test_operator_interrupt_stops_codex_and_preserves_run(tmp_path,monkeypatch):
    from mahjong_ai import research_cli
    r=Research(tmp_path);r.initialize('rule',seconds=10)
    class Process:
        pid=123456789
        stdin=None
        stopped=False
        def __init__(self,*args,**kwargs):
            import io
            self.stdin=io.BytesIO()
        def poll(self):return 0 if self.stopped else None
        def wait(self,timeout=None):self.stopped=True;return 0
    process=Process()
    monkeypatch.setattr(research_cli.subprocess,'Popen',lambda *a,**k:process)
    killed=[]
    monkeypatch.setattr(research_cli.os,'killpg',lambda pid,sig:killed.append(pid))
    monkeypatch.setattr(research_cli.time,'sleep',lambda _:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):research_cli.drive(r)
    assert killed==[process.pid]
    assert r._read()['lease'] is None
    assert r.status()['run_id']==r._read()['id']
