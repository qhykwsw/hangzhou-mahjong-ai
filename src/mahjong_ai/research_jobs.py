"""Host-owned batch jobs. Candidate code only executes through candidate_runtime."""
from __future__ import annotations

import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .candidate_runtime import CandidateProcess, train_package, tree_hash
from .engine import MahjongEngine, derive_seed
from .evaluator import Evaluator, promotion_decision
from .policy import PolicySpec
from .ratings import EloLedger
from .storage import atomic_json, digest, exclusive_lock, read_json
from .types import RoundConfig


def prepare_environment(requirements, root, deadline):
    if not requirements:return sys.executable
    env=root/'venv'
    if not env.exists():
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True,
                       timeout=max(.01,deadline-time.monotonic()),stdout=subprocess.DEVNULL)
    python=str(env/'bin'/'python')
    subprocess.run([python,'-m','pip','install','--disable-pip-version-check',*requirements],check=True,
                   timeout=max(.01,deadline-time.monotonic()),stdout=subprocess.DEVNULL)
    freeze=subprocess.check_output([python,'-m','pip','freeze'],text=True,timeout=max(.01,deadline-time.monotonic()))
    (root/'requirements.lock').write_text(freeze)
    return python


def run_candidate(task,cid,root,deadline):
    output=root/cid; final=output/'result.json'
    if final.exists():
        saved=read_json(final)
        if saved['status']=='completed':
            PolicySpec(**saved['spec']).identity
            return saved
    source=Path(task['packages'][cid]['input'])
    try:
        if tree_hash(source)!=task['packages'][cid]['source_hash']:raise ValueError('input candidate changed')
        python=prepare_environment(task['packages'][cid]['requirements'],output,deadline)
        package=output/'frozen'
        if package.exists():shutil.rmtree(package)
        shutil.copytree(source,package)
        if (output/'requirements.lock').exists():shutil.copy2(output/'requirements.lock',package/'requirements.lock')
        training=None
        if task['family']=='rl':
            train_result=output/'training.json'; model=output/'training-model'
            if train_result.exists():
                training=read_json(train_result)
                if training.get('model_hash')!=tree_hash(model):raise ValueError('training artifact changed')
            else:
                training=train_package(source,model,RoundConfig(**task['config']),
                    tuple(PolicySpec(**p) for p in task['opponents']),task['batch']['training_episodes'],
                    task['seed']+int(digest(cid)[:6],16),deadline,python)
                training['model_hash']=tree_hash(model);atomic_json(train_result,training)
            if (package/'model').exists():shutil.rmtree(package/'model')
            shutil.copytree(model,package/'model')
        elif (package/'model').exists() and any((package/'model').iterdir()):
            raise ValueError('rule route cannot carry trained model artifacts')
        spec=PolicySpec(task['family'],cid,package_path=str(package),package_hash=tree_hash(package),
                        action_seconds=task['action_seconds'],python_path=python)
        probe=CandidateProcess(package,python)
        try:probe.call({'op':'build'},min(10,max(.001,deadline-time.monotonic())))
        finally:probe.close()
        atomic_json(output/'policy.json',spec.as_dict())
        # Separate process makes evaluator caches independent across candidates and kills hung jobs.
        eval_task={'baseline':task['baseline'],'candidate':spec.as_dict(),'opponents':task['opponents'],
                   'config':task['config'],'pairs':task['batch']['pairs'],'seed':task['seed'],
                   'alpha':.05,'workers':1}
        atomic_json(output/'evaluation-task.json',eval_task)
        log=(output/'evaluation.log').open('ab')
        p=subprocess.Popen([sys.executable,'-B','-m','mahjong_ai.research_jobs','evaluate',str(output),str(time.time()+max(0,deadline-time.monotonic()))],stdout=log,stderr=log)
        try:p.wait(timeout=max(.01,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            p.terminate();p.wait();raise TimeoutError('batch time budget exhausted')
        finally:log.close()
        if p.returncode:raise RuntimeError('candidate evaluation failed; inspect recorded error')
        summary=read_json(output/'development'/'summary.json')
        if summary['attempted_pairs']!=task['batch']['pairs']:raise TimeoutError('development evaluation paused')
        if summary['failure_count']:raise ValueError('development had failed settlements')
        executions=[read_json(p)['candidate'].get('execution',{}) for p in (output/'development').glob('pair-*.json')]
        summary['execution']={key:sum(e.get(key,0) for e in executions) for key in ('actions','timeouts','errors','elapsed_seconds')}
        if summary['execution']['errors']:raise ValueError('candidate raised errors or returned illegal actions')
        # Diagnostic observations are legal at decision time, unlike full omniscient replays.
        from .policy import RulePolicy
        traces=[]; remote=spec.build(); anchor=RulePolicy()
        class Capture:
            name,version='capture',cid
            def reset(self,seed):remote.reset(seed)
            def act(self,obs):
                action=remote.act(obs)
                anchor_action=anchor.act(obs)
                if len(traces)<128:traces.append({'observation':asdict(obs),'chosen_index':obs.legal_actions.index(action),
                                                'reference_index':obs.legal_actions.index(anchor_action)})
                return anchor_action  # every candidate is probed on identical observations
            def finish_round(self,result,seat):remote.finish_round(result,seat)
        try:
            players=[Capture(),*[PolicySpec(**p).build() for p in task['opponents']]]
            MahjongEngine().run_round(players,RoundConfig(**task['config']),derive_seed(0,'diagnostic'))
        finally:remote.close()
        atomic_json(output/'traces.json',traces)
        summary['diagnostic_execution']=remote.stats
        summary['behavior_signature']=digest([t['chosen_index'] for t in traces])
        summary['behavior_changed_fraction']=sum(t['chosen_index']!=t['reference_index'] for t in traces)/max(1,len(traces))
        row={'status':'completed','candidate':cid,'spec':spec.as_dict(),'summary':summary,'training':training,
             'behavior_signature':digest([t['chosen_index'] for t in traces])}
    except TimeoutError:
        return {'status':'paused','candidate':cid}
    except Exception as exc:
        row={'status':'failed','candidate':cid,'error':f'{type(exc).__name__}: {exc}'}
    atomic_json(final,row)
    return row


def run_batch(root,deadline):
    task=read_json(root/'task.json'); batch=task['batch']
    with ThreadPoolExecutor(max_workers=task['workers']) as pool:
        futures={cid:pool.submit(run_candidate,task,cid,root,deadline) for cid in batch['candidates']}
        rows={cid:f.result() for cid,f in futures.items()}
    if any(r['status']=='paused' for r in rows.values()):
        atomic_json(root/'result.json',{'status':'paused','candidates':rows});return
    league={}
    eligible=[r['spec'] for r in rows.values() if r['status']=='completed']
    if eligible and batch['league_samples']:
        specs=[PolicySpec(**p) for p in eligible]
        baseline=PolicySpec(**task['baseline'])
        specs.append(baseline)
        specs.extend(PolicySpec(**p) for p in task['opponents'])
        # Balanced cyclic blocks: every member gets the same four seats per cycle.
        # Round up to full cycles: each member appears once at every seat per cycle.
        total=((batch['league_samples']+len(specs)-1)//len(specs))*len(specs)
        totals=[[] for _ in specs];ledger=EloLedger(root/'league-elo.json',{'config':task['config'],'kind':'batch-development'})
        instances=[p.build() for p in specs]
        try:
            for i in range(total):
                if time.monotonic()>=deadline:
                    atomic_json(root/'result.json',{'status':'paused','candidates':rows});return
                order=[(i+j)%len(specs) for j in range(4)]
                path=root/f'league-{i:05}.json'
                if path.exists():row=read_json(path)
                else:
                    result=MahjongEngine().run_round([instances[j] for j in order],RoundConfig(**task['config']),derive_seed(task['seed'],'league',i))
                    row={'order':order,'completed':result.completed,'ranks':list(result.rank_by_seat)}
                    atomic_json(path,row)
                if row['completed']:
                    ledger.record(str(i),[specs[j].identity for j in order],row['ranks'],completed=True,source='research-development')
                    for slot,j in enumerate(order):totals[j].append(row['ranks'][slot])
        finally:
            for p in instances:
                if hasattr(p,'close'):p.close()
        league={'samples':total,'elo':ledger.table(),'table':[{'version':specs[i].version,'appearances':len(v),
                'mean_rank':sum(v)/len(v) if v else None} for i,v in enumerate(totals)],
                'note':'cyclic four-player development games; use paired fixed opponents for promotion'}
    atomic_json(root/'result.json',{'status':'completed','batch_id':batch['id'],'candidates':rows,'league':league,
                                  'completed_at':time.time()})


def evaluate(root,deadline,acceptance=False):
    t=read_json(root/('task.json' if acceptance else 'evaluation-task.json'))
    report=Evaluator().evaluate(PolicySpec(**t['baseline']),PolicySpec(**t['candidate']),
        tuple(PolicySpec(**p) for p in t['opponents']),RoundConfig(**t['config']),pairs=t['pairs'],
        seed=t['seed'],alpha=t['alpha'],workers=t['workers'],output=root/('private' if acceptance else 'development'),deadline=deadline)
    if acceptance:
        data=promotion_decision(report) if len(report.pairs)==t['pairs'] else {'status':'paused','reason':'budget interrupted acceptance'}
        atomic_json(root/'public.json',data|{'summary':report.summary()})


def main():
    mode,path,end=sys.argv[1:4];root=Path(path)
    deadline=time.monotonic()+max(0,float(end)-time.time())
    # Signals interrupt work; cached completed games survive. No candidate is promoted on interruption.
    with exclusive_lock(root/'.job.lock'):
        try:
            if mode=='batch':run_batch(root,deadline)
            else:evaluate(root,deadline,mode=='acceptance')
        except Exception as exc:
            if mode=='batch':
                task=read_json(root/'task.json')
                atomic_json(root/'result.json',{'status':'completed','candidates':{
                    cid:{'status':'failed','candidate':cid,'error':f'{type(exc).__name__}: {exc}'}
                    for cid in task['batch']['candidates']},'league':{},'infrastructure_error':True})
            elif mode=='acceptance':atomic_json(root/'public.json',{'status':'failed','reason':f'{type(exc).__name__}: {exc}'})
            else:raise


if __name__=='__main__':main()
