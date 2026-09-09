#!/usr/bin/env python3
"""Report one bounded continuation of an existing research run."""
import argparse
import time
from pathlib import Path

from mahjong_ai.storage import atomic_json, read_json


def report(session):
    session=Path(session)
    meta=read_json(session/'session.json');before=meta['before'];root=Path(meta['root']);after=read_json(root/'run.json')
    new_batches=after['batches'][len(before['batches']):]
    candidates=[];development_rounds=0;diagnostic_rounds=0;failures=[]
    batches=[]
    for b in new_batches:
        path=root/'batches'/b['id']/'result.json'
        data=read_json(path) if path.exists() else {}
        rows=[]
        for cid in b['candidates']:
            row=data.get('candidates',{}).get(cid,{})
            if not row:
                candidate_result=root/'batches'/b['id']/cid/'result.json'
                if candidate_result.exists():row=read_json(candidate_result)
            summary=row.get('summary',{})
            development_rounds+=summary.get('actual_rounds_started',0)
            diagnostic_rounds+=int((root/'batches'/b['id']/cid/'traces.json').exists())
            c=after['candidates'][cid]
            parent=after['candidates'].get(c['parent'],{})
            unchanged=bool(c.get('spec') and parent.get('spec') and c['spec']['package_hash']==parent['spec']['package_hash'])
            entry={'id':cid,'parent':c['parent'],'references':c['references'],'hypothesis':c['hypothesis'],
                   'unchanged_control':unchanged,'population_status':c['status'],'experiment_status':row.get('status','pending'),
                   'development':summary,'error':row.get('error')}
            rows.append(entry);candidates.append(entry)
            if row.get('status')=='failed':failures.append(cid)
        batches.append({'id':b['id'],'status':b['status'],'requested_pairs_per_candidate':b['pairs'],'candidates':rows})
    acceptances=after['acceptances'][len(before['acceptances']):]
    migration=read_json(session/'budget-migration.json') if (session/'budget-migration.json').exists() else None
    remaining=max(0,after['seconds_granted']-after['seconds_used']-(time.time()-after['lease']['started'] if after['lease'] else 0))
    return {'version':1,'run_id':after['id'],'family':after['family'],'objective':after['objective'],
            'config':after['config'],'requested_seconds':meta['requested_active_seconds'],
            'running':bool(after['lease']),'remaining_seconds':remaining,
            'original_wall_end':migration['original_end'] if migration else None,
            'baseline_before':before['baseline'],'baseline_after':after['baseline'],
            'acceptance_cap':after['acceptance_granted'],
            'batch_count':len(new_batches),'decided_batches':sum(b['status']=='decided' for b in new_batches),
            'candidate_experiments':len(candidates),'unchanged_controls':sum(c['unchanged_control'] for c in candidates),
            'development_settlements':development_rounds,'diagnostic_settlements':diagnostic_rounds,
            'acceptance_settlements':sum(a.get('summary',{}).get('actual_rounds_started',0) for a in acceptances),
            'failed_candidates':failures,'acceptances':acceptances,'batches':batches,
            'decisions':after['decisions'][len(before['decisions']):],
            'note':'Each promotion compares to its stated parent baseline. Do not sum confidence intervals or claim a fresh end-to-end comparison.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=report(args.session);atomic_json(args.output,result)
    print({k:result[k] for k in ('running','remaining_seconds','batch_count','candidate_experiments','unchanged_controls','baseline_before','baseline_after','development_settlements','acceptance_settlements')})
