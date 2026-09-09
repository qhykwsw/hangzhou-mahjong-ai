#!/usr/bin/env python3
"""Aggregate a pair of completed research runs without exporting private samples."""
import argparse
import json
from pathlib import Path

from mahjong_ai.storage import atomic_json, file_hash, read_json


def report(root):
    result={'version':1,'objective':'four-player-settlement-rank-v1','families':{}}
    for family in ('rule','rl'):
        r=root/family;s=read_json(r/'run.json')
        batches=[];training=0;development=0;league=0;diagnostic=0;tools=[];violations=[]
        for b in s['batches']:
            d=read_json(r/'batches'/b['id']/'result.json')
            candidates=[]
            for cid,row in d['candidates'].items():
                t=row.get('training') or {};summary=row.get('summary',{})
                training+=len(t.get('episodes',[]));development+=summary.get('actual_rounds_started',0)
                if (r/'batches'/b['id']/cid/'traces.json').exists():diagnostic+=1
                candidates.append({'id':cid,'parent':s['candidates'][cid]['parent'],
                    'references':s['candidates'][cid]['references'],'status':row['status'],
                    'hypothesis':s['candidates'][cid]['hypothesis'],'training_episodes':len(t.get('episodes',[])),
                    'training_metrics':t.get('reported_metrics'),
                    'development_mean_rank_delta':summary.get('mean_rank_utility_delta'),
                    'execution':summary.get('execution')})
            league+=d.get('league',{}).get('samples',0)
            batches.append({'id':b['id'],'status':b['status'],'candidates':candidates})
        for path in sorted((r/'agent').glob('turn-*.jsonl')):
            for line in path.read_text().splitlines():
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                item=event.get('item',{})
                if event.get('type')=='item.started':
                    if item.get('type')=='mcp_tool_call':tools.append(item['tool'])
                    if item.get('type') in ('command_execution','file_change'):violations.append(item['type'])
        before_path=root/'before-resume.json';resume={}
        if before_path.exists():
            before=read_json(before_path)[family]
            resume={'same_run':s['id']==before['run_id'],'same_thread':s['thread_id']==before['thread_id'],
                    'same_batches':s['batches']==before['batches'],'same_acceptance_count':s['acceptance_used']==before['acceptance_used'],
                    'cached_pairs_unchanged':all(file_hash(r/p)==h for p,h in before['hashes'].items())}
        comparisons=[]
        for a in s['acceptances']:
            data=read_json(r/'acceptance'/a['id']/'public.json')
            comparisons.append({'candidate':a['candidate'],'baseline':a['baseline'],**data})
        frozen=r/'runtime'/'mahjong_ai'
        changed=[p.name for p in Path(__file__).resolve().parents[1].joinpath('src/mahjong_ai').glob('*.py') if file_hash(p)!=file_hash(frozen/p.name)]
        restore=[]
        restoration=r/'agent'/'restore-check.jsonl'
        if restoration.exists():
            for line in restoration.read_text().splitlines():
                event=json.loads(line);item=event.get('item',{})
                if event.get('type')=='item.completed' and item.get('type')=='mcp_tool_call':restore.append({'tool':item['tool'],'status':item['status'],'error':item.get('error')})
        result['families'][family]={'run_id':s['id'],'baseline':s['baseline'],'config':s['config'],
            'elapsed_active_seconds':s['seconds_used'],'batches':batches,'training_settlements':training,
            'development_settlements':development,'league_settlements':league,'diagnostic_settlements':diagnostic,
            'acceptance':comparisons,'tool_calls':len(tools),'tool_names':sorted(set(tools)),
            'nonresearch_execution_events':violations,'resume':resume,'restored_thread_readonly_calls':restore,
            'source_matches_workspace':not changed,'workspace_changes_after_run':changed}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();data=report(Path(args.root));atomic_json(args.output,data)
    print(json.dumps({f:{'batches':len(d['batches']),'seconds':d['elapsed_active_seconds'],
        'acceptance':[(a['candidate'],a['status']) for a in d['acceptance']],
        'resume':d['resume'],'source_matches_workspace':d['source_matches_workspace']}
        for f,d in data['families'].items()},ensure_ascii=False,indent=2))
