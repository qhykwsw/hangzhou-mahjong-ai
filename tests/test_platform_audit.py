import importlib.util
from pathlib import Path

from mahjong_ai.storage import atomic_json,read_json,digest


def test_actual_multihand_score_deltas_sum_to_final_not_difference_of_deltas(tmp_path):
    spec = importlib.util.spec_from_file_location('platform_audit_script',Path(__file__).parents[1]/'scripts/audit_platform_run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    replay=read_json(Path(__file__).parent/'fixtures/official-v25-eight-hand-replay.json')
    atomic_json(tmp_path/'workflow.json',{'protocol':{'config':{'M':1,'Rounds':8,'BaseScore':1,'YouCaiBiKao':False}}})
    atomic_json(tmp_path/'replays'/'game.json',replay)
    final=next(e['data']['final_scores'] for b in replay['blocks'] for e in b['events'] if e['type']=='game_ended')
    for i in range(4):
        atomic_json(tmp_path/f'player-{i}'/'game-fixture.json',{'status':'finished','final':{'snapshot':{'seat':i,'scores':final}}})
    result=module.audit(tmp_path)
    assert result['status']=='passed'
    assert result['games'][0]['final_score_consistent']
    assert all(c['ok'] for c in result['games'][0]['score_checks'])
