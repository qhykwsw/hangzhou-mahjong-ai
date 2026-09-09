import copy
import json
from pathlib import Path
import time
import pytest
from starlette.testclient import TestClient
from mahjong_ai.arena import Arena, configuration, schedule, standings
from mahjong_ai.arena_controller import Controller
from mahjong_ai.app import application
from mahjong_ai.candidate_runtime import ActionUnavailable, RemotePolicy, tree_hash, train_package
from mahjong_ai.engine import HandState
from mahjong_ai.seeds import default_opponents
from mahjong_ai.storage import atomic_json, digest, read_json
from mahjong_ai.types import RoundConfig


@pytest.fixture
def arena(tmp_path):
    for name in ('alpha', 'beta'):
        d = tmp_path / 'directions' / name
        d.mkdir(parents=True)
        (d / 'direction.toml').write_text('name="'+name+'"\nbrief="Explore independently"')
    (tmp_path / 'research.toml').write_text('directions=["alpha","beta"]\n')
    return Arena(tmp_path)


def test_schedule_balances_appearances_and_every_seat():
    ids = [f'p{i:02d}' for i in range(16)]
    games = schedule(ids, 128, 54)
    assert games == schedule(ids, 128, 54)
    for cid in ids:
        assert [sum(g['players'][seat] == cid for g in games) for seat in range(4)] == [8]*4
    for start in range(0,128,4):
        assert len({tuple(sorted(g['players'])) for g in games[start:start+4]}) == 1


def test_direction_scope_freezing_budget_and_atomic_batch(arena):
    run = arena.create(seconds=20)
    arena.control(run,'resume')
    arena.fork(run,'alpha','child','idea','initial')
    with pytest.raises(ValueError,match='owned'):
        arena.fork(run,'alpha','stolen','idea','beta--initial')
    with pytest.raises(ValueError,match='relative'):
        arena.candidate(run,'alpha','initial',path='../private/random.json')
    source=Path(arena.candidate(run,'alpha','child')['workspace'])/'strategy.py'
    source.unlink()
    with pytest.raises(ValueError,match='strategy.py'):
        arena.submit(run,'alpha',['initial','child'])
    assert arena.candidate(run,'alpha','initial')['status']=='draft'
    source.write_text('def build(model): pass')
    arena.submit(run,'alpha',['initial','child'])
    with pytest.raises(ValueError,match='draft'):
        arena.candidate(run,'alpha','initial',path='strategy.py',content='change')
    visible=json.dumps(arena.snapshot(run,'beta'))
    assert 'Explore independently' not in visible or 'beta' in visible
    assert 'package_path' not in visible and 'workspace' not in visible and 'thread_id' not in visible
    arena.control(run,'pause')
    with arena.edit(run) as (s,db):s['status']='paused'
    before=arena.snapshot(run)['run']['seconds_remaining']
    arena.control(run,'budget',20)
    arena.control(run,'resume')
    assert arena.snapshot(run)['run']['seconds_remaining']>before+19
    with pytest.raises(ValueError,match='pause'):
        arena.create()


def test_period_elimination_is_idempotent_and_waits_for_new_batch(arena):
    run=arena.create()
    with arena.edit(run) as (s,db):
        base=copy.deepcopy(s['candidates']['seed-0'])
        for i in range(4,16):
            cid=f'seed-{i:02d}'
            s['candidates'][cid]={**base,'id':cid,'status':'active'}
        arena.advance(s,db)
        period=s['periods'][0]
        for game in period['schedule']:
            # Lowest candidate IDs win deterministically; timeout counts don't affect ranking.
            ids=game['players']; ranks=[sorted(ids).index(x)+1 for x in ids]
            receipt={'players':ids,'ranks':ranks,'execution':[{'timeouts':999 if r==1 else 0} for r in ranks]}
            db.execute('INSERT INTO results VALUES(?,?,?,?)',(run,1,game['index'],json.dumps(receipt)))
        arena.advance(s,db)
        assert len(period['eliminated'])==4
        assert len([c for c in s['candidates'].values() if c['status']=='active'])==12
        arena.advance(s,db)
        assert len(s['periods'])==1
        assert len(period['eliminated'])==4


def test_timeout_keeps_worker_and_discards_late_reply(tmp_path):
    (tmp_path/'strategy.py').write_text('''import time
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
''')
    p=RemotePolicy(tmp_path,tree_hash(tmp_path),action_seconds=.02)
    hand=HandState(RoundConfig(rounds=1),123,0,1,[0]*4,[],False)
    obs=hand.observation();p.reset(1)
    try:
        pid=p.worker.process.pid
        with pytest.raises(ActionUnavailable):p.act(obs)
        time.sleep(.12)
        assert p.act(obs)==obs.legal_actions[-1]
        assert p.worker.process.pid==pid
        assert p.stats['timeouts']==1 and p.stats['errors']==0
    finally:p.close()


def test_platform_fallback_discards_rightmost_in_insertion_order():
    hand=HandState(RoundConfig(rounds=1),123,0,1,[0]*4,[],False)
    legal=hand.legal_actions()
    expected=next((a for a in legal if a.action=='hu'),None)
    if expected is None:
        expected=next(a for tile in reversed(hand.hands[hand.turn]) for a in legal if a.action=='discard' and a.tile==tile)
    assert hand.timeout_action()==expected


def test_generic_training_can_compute_artifact_without_rl_games(tmp_path):
    package=tmp_path/'source';package.mkdir()
    (package/'strategy.py').write_text('''def train(ctx):
 (ctx.output/'lookup.txt').write_text('computed lookup')
 ctx.checkpoint({'ready':True})
 return {'lookups':1}
''')
    model=tmp_path/'out/model'
    result=train_package(package,model,RoundConfig(rounds=1),default_opponents(),2,3,time.monotonic()+10)
    assert result['episodes']==[]
    assert (model/'lookup.txt').read_text()=='computed lookup'


def test_api_controls_and_rejects_cross_origin(arena,monkeypatch):
    monkeypatch.setattr(Controller,'tick',lambda self:None)
    with TestClient(application(arena.project)) as client:
        assert client.get('/').status_code==200
        r=client.post('/api/runs',json={'directions':['alpha'],'seconds':30})
        assert r.status_code==200
        run=r.json()['run']['id']
        assert client.post('/api/control',json={'run':run,'action':'pause'},headers={'Origin':'https://other.example'}).status_code==403
        assert client.post('/api/control',json={'run':run,'action':'pause'}).json()['run']['status']=='pausing'
        assert client.get('/api/state').json()['run']['id']==run


def test_prepare_rejects_bad_load_and_deduplicates_identical_artifact(arena,tmp_path):
    from mahjong_ai.arena_jobs import prepare
    run=arena.create();arena.control(run,'resume')
    arena.candidate(run,'alpha','initial',path='strategy.py',content='def build(model): return None')
    arena.submit(run,'alpha',['initial'])
    s=arena.read(run);job=next(j for j in s['jobs'].values() if j['kind']=='prepare')
    task=dict(packages=job['packages'],config=s['config']['game'],timeouts=s['config']['timeouts'],
              opponents=[c['spec'] for c in s['candidates'].values() if c.get('spec')],training_games=1,seed=4)
    (tmp_path/'prepare').mkdir()
    result=prepare(task,tmp_path/'prepare',time.monotonic()+10)
    assert result['candidates']['alpha--initial']['status']=='failed'
    assert 'reset, act and finish_round' in result['candidates']['alpha--initial']['error']
    # Identical immutable content is associated with the existing entrant, without new slots.
    seed=s['candidates']['seed-0']['spec']
    row={'status':'completed','candidates':{'alpha--initial':{'status':'ready','spec':seed}}}
    root=Path(s['root'])/'jobs'/job['id'];atomic_json(root/'result.json',row)
    controller=Controller(arena)
    after=arena.read(run)
    assert after['candidates']['alpha--initial']['status']=='duplicate'
    assert after['batches'][job['batch_id']]['status']=='completed'
    with arena.edit(run) as (state,db):controller.consume(state,db)
    assert len(arena.read(run)['batches'])==1


def test_recovery_consumes_finished_match_once_and_keeps_receipt(arena):
    run=arena.create();arena.control(run,'resume')
    with arena.edit(run) as (s,db):
        arena.advance(s,db)
        job=next(j for j in s['jobs'].values() if j['kind']=='match')
        job['status']='admitted'
        root=Path(s['root'])/'jobs'/job['id']
        task={'assignment':job['assignment'],'deadline':time.time()+30}
        atomic_json(root/'task.json',task)
        atomic_json(root/'result.json',dict(status='completed',players=job['assignment']['players'],
            ranks=[1,2,3,4],scores=[3,1,-1,-3],hands=8,execution=[{}]*4,diagnostics=[],task_hash=digest(task)))
    Controller(arena);Controller(arena)
    assert len(arena.receipts(run,1))==1
    assert arena.read(run)['jobs'][job['id']]['status']=='completed'
    assert arena.read(run)['status']=='paused'


def test_match_and_training_share_two_slots_and_agent_uses_own_slot(arena,monkeypatch):
    controller=Controller(arena)
    run=arena.create();arena.control(run,'resume');arena.submit(run,'alpha',['initial'])
    launched=[]
    def launch(s,job):
        launched.append(job['kind']);job['status']='running'
        atomic_json(Path(s['root'])/'jobs'/job['id']/'task.json',{'deadline':time.time()+60})
    monkeypatch.setattr(controller,'launch',launch)
    controller.tick()
    assert launched.count('prepare')==1 and launched.count('match')==1 and launched.count('agent')==1
    controller.tick()
    assert len(launched)==3
    if controller.caffeine:
        controller.caffeine.terminate();controller.caffeine.wait()


def test_budget_expiry_charges_waiting_and_does_not_auto_resume(arena,monkeypatch):
    import mahjong_ai.arena as domain
    run=arena.create(seconds=10)
    arena.control(run,'resume')
    start=arena.read(run)['accounted_at']
    monkeypatch.setattr(domain.time,'time',lambda:start+11)
    with arena.edit(run) as (s,db):
        assert s['status']=='pausing' and s['pause_reason']=='budget exhausted'
        s['status']='paused'
    spent=arena.read(run)['seconds_used']
    monkeypatch.setattr(domain.time,'time',lambda:start+100)
    arena.control(run,'budget',20)
    assert arena.read(run)['seconds_used']==spent
    assert arena.read(run)['status']=='paused'
    arena.control(run,'resume')
    assert arena.read(run)['status']=='running'


def test_codex_resume_preserves_direction_thread_and_model_settings(tmp_path,monkeypatch):
    from mahjong_ai.research_agent import codex_command
    (tmp_path/'config.toml').write_text('model="configured-model"\nmodel_reasoning_effort="high"\n')
    monkeypatch.setenv('CODEX_HOME',str(tmp_path))
    command=codex_command('/project','run-id','alpha','thread-original')
    assert command[-3:]==['resume','thread-original','-']
    assert 'model="configured-model"' in command
    assert not any('beta' in part for part in command)
