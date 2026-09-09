import pytest
from mahjong_ai.ratings import elo_changes, EloLedger
from mahjong_ai.league import run_league
from mahjong_ai.types import RoundConfig
from mahjong_ai.policy import PolicySpec
from mahjong_ai.storage import read_json


def test_pairwise_elo_zero_sum_ties_and_opponent_strength():
    ids=['a','b','c','d']
    assert elo_changes(ids,[1,2,3,4],{})=={'a':12,'b':4,'c':-4,'d':-12}
    assert elo_changes(ids,[2.5]*4,{})==dict.fromkeys(ids,0.)
    strong=elo_changes(ids,[1,2,3,4],{'a':1100,'b':1700,'c':1700,'d':1700})
    assert strong['a']>12 and sum(strong.values())==pytest.approx(0)
    same=elo_changes(['a']*4,[1,2,3,4],{})
    assert same=={'a':0.}


def test_elo_idempotency_context_and_competition_isolation(tmp_path):
    ledger=EloLedger(tmp_path/'elo.json',{'games':2})
    a=ledger.record('one',['a','b','c','d'],[1,2,3,4],completed=True,source='research-development')
    assert ledger.record('one',['a','b','c','d'],[1,2,3,4],completed=True,source='research-development')==a
    assert len(read_json(tmp_path/'elo.json')['events'])==1
    with pytest.raises(ValueError,match='changed'):
        ledger.record('one',['a','b','c','d'],[4,3,2,1],completed=True,source='research-development')
    with pytest.raises(ValueError,match='research'):
        ledger.record('two',['a','b','c','d'],[1,2,3,4],completed=True,source='competition')
    with pytest.raises(ValueError,match='incomplete'):
        ledger.record('two',['a','b','c','d'],[1,2,3,4],completed=False,source='research-league')


def test_league_resume_does_not_play_or_rate_twice(tmp_path,monkeypatch):
    policies=[PolicySpec(parameters={'risk_scale':r}) for r in (.1,.3,1.,2.)]
    first=run_league(policies,RoundConfig(rounds=2,games_per_round=2),3,8,tmp_path)
    from mahjong_ai.engine import MahjongEngine
    monkeypatch.setattr(MahjongEngine,'run_round',lambda *a:pytest.fail('must use cache'))
    assert run_league(policies,RoundConfig(rounds=2,games_per_round=2),3,8,tmp_path)==first
    assert len(read_json(tmp_path/'elo.json')['events'])==3
