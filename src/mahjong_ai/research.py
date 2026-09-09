"""Persistent research domain. One route, one ledger, one batch decision at a time."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .candidate_runtime import tree_hash
from .iteration import default_opponents
from .policy import PolicySpec
from .storage import atomic_json, digest, exclusive_lock, read_json, runtime_fingerprint
from .types import RANK_OBJECTIVE, RoundConfig


def child_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or not p.parts or any(x in ('.', '..') for x in p.parts):
        raise ValueError('use a relative path inside the candidate')
    root = Path(root).resolve()
    target = root / p
    if not target.resolve().is_relative_to(root): raise ValueError('path escapes candidate')
    if any(x.is_symlink() for x in [target, *target.parents] if x.is_relative_to(root)):
        raise ValueError('symlinks are not supported')
    return target


class Research:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.state_path = self.root / 'run.json'

    def initialize(self, family, config=RoundConfig(), seconds=3600, acceptance_slots=None,
                   workers=2, action_seconds=.05, seed=0):
        if family not in ('rule', 'rl') or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('valid route and positive total seconds required')
        if acceptance_slots is not None and not 1 <= acceptance_slots <= 100:
            raise ValueError('acceptance slots must be omitted or 1..100')
        if not 1 <= workers <= 16 or not 0 < action_seconds <= 1:
            raise ValueError('workers 1..16, action_seconds (0,1] required')
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            if self.state_path.exists(): raise ValueError('run already exists; resume it instead')
            self.root.mkdir(parents=True, exist_ok=True)
            self.root.chmod(0o700)
            state = {'version': 1, 'id': uuid.uuid4().hex, 'family': family, 'objective': RANK_OBJECTIVE,
                     'config': config.as_dict(), 'runtime': runtime_fingerprint(), 'seed': seed,
                     'workers': workers, 'action_seconds': action_seconds,
                     'seconds_granted': seconds, 'seconds_used': 0., 'lease': None,
                     'acceptance_granted': acceptance_slots, 'acceptance_used': 0,
                     'acceptance_epochs': ([{'slots': acceptance_slots, 'used': 0, 'alpha': .025/acceptance_slots}] if acceptance_slots else []),
                     'baseline': None, 'active_batch': None, 'candidates': {}, 'batches': [],
                     'decisions': [], 'acceptances': [], 'thread_id': None, 'events': []}
            from .research_templates import RULE, RL
            workspace = self.root / 'workspaces' / 'initial'
            workspace.mkdir(parents=True)
            (workspace / 'strategy.py').write_text(RULE if family == 'rule' else RL)
            state['candidates']['initial'] = {'id': 'initial', 'parent': None, 'references': [],
                                             'hypothesis': 'initial baseline', 'status': 'draft',
                                             'workspace': 'workspaces/initial', 'requirements': []}
            atomic_json(self.state_path, state)
        return self.status()

    def _read(self):
        s = read_json(self.state_path)
        if s['runtime'] != runtime_fingerprint():
            raise ValueError('runtime changed; use the frozen runtime recorded for this run')
        return s

    def _remaining(self, s):
        active = max(0, time.time()-s['lease']['started']) if s['lease'] else 0
        return max(0., s['seconds_granted']-s['seconds_used']-active)

    def begin(self):
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            s = self._read()
            if s['lease']:
                # The CLI holds .agent.lock; this is crash recovery of the previous lease.
                s['seconds_used'] += max(0, s['lease'].get('heartbeat', s['lease']['started'])-s['lease']['started'])
                s['lease'] = None
            if self._remaining(s) <= 0: raise ValueError('time budget exhausted; explicitly add budget')
            s['lease'] = {'started': time.time(), 'heartbeat': time.time()}
            atomic_json(self.state_path, s)

    def heartbeat(self):
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            s = self._read()
            if s['lease']:
                s['lease']['heartbeat'] = time.time()
                atomic_json(self.state_path, s)

    def end(self):
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            s = self._read()
            if s['lease']:
                s['seconds_used'] += max(0, time.time()-s['lease']['started'])
                s['lease'] = None
                atomic_json(self.state_path, s)

    def add_budget(self, seconds=0, slots=0, unlimited=False):
        if not math.isfinite(seconds) or seconds < 0 or type(slots) is not int or not 0 <= slots <= 100:
            raise ValueError('invalid budget addition')
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            s = self._read()
            s['seconds_granted'] += seconds
            if unlimited: s['acceptance_granted'] = None
            if s['acceptance_granted'] is not None: s['acceptance_granted'] += slots
            if slots and s['acceptance_granted'] is not None:
                # Geometric lifetime spending; extension never resets the error allowance.
                i = len(s['acceptance_epochs'])
                s['acceptance_epochs'].append({'slots': slots, 'used': 0, 'alpha': .025/(2**i)/slots})
            s['events'].append({'kind': 'budget_added', 'seconds': seconds, 'slots': slots, 'unlimited':unlimited, 'at': time.time()})
            atomic_json(self.state_path, s)
        return self.status()

    def _idle(self, s):
        if s['active_batch']: raise ValueError('batch is active: wait for its complete result and decide first')
        if any(a['status'] in ('running','paused') for a in s['acceptances']):
            raise ValueError('acceptance is pending; resume or cancel it before changing candidates')

    def _budget(self, s):
        if not s['lease'] or self._remaining(s) <= 0:
            raise ValueError('research is paused or time budget exhausted')

    def status(self):
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            s = self._read()
            self._refresh(s)
            atomic_json(self.state_path, s)
            return {'run_id': s['id'], 'family': s['family'], 'objective': s['objective'],
                    'rank_rewards': [3,2,1,0], 'config': s['config'], 'baseline': s['baseline'],
                    'seconds_remaining': round(self._remaining(s),3),
                    'acceptance_remaining': None if s['acceptance_granted'] is None else s['acceptance_granted']-s['acceptance_used'],
                    'active_batch': s['active_batch'],
                    'batches': [{k:b[k] for k in ('id','status','candidates')} for b in s['batches']],
                    'population': [{k:v for k,v in c.items() if k in ('id','status','parent','references','hypothesis','summary')} for c in s['candidates'].values()],
                    'acceptances': s['acceptances']}

    def _refresh(self, s):
        for b in s['batches']:
            result = self.root / 'batches' / b['id'] / 'result.json'
            if b['status'] != 'decided' and result.exists():
                data = read_json(result)
                b['status'] = data['status']
                if data['status'] == 'completed':
                    for cid, row in data['candidates'].items():
                        c = s['candidates'][cid]
                        c['status'] = 'evaluated' if row['status'] == 'completed' else 'failed'
                        if row.get('spec'): c['spec'] = row['spec']
                        c['summary'] = row.get('summary', {'error':row.get('error')})
                    # Initialization is explicitly a seed baseline, never a claimed promotion.
                    if b.get('initial') and data['candidates']['initial']['status'] == 'completed':
                        s['baseline'] = 'initial'
        for a in s['acceptances']:
            path = self.root / 'acceptance' / a['id'] / 'public.json'
            if path.exists():
                data = read_json(path)
                a.update(data)
                if data.get('status') == 'promoted' and not a.get('applied'):
                    if s['baseline'] != a['baseline']: raise ValueError('acceptance parent baseline changed')
                    s['baseline'] = a['candidate']
                    a['applied'] = True

    def create_candidate(self, candidate_id, hypothesis, parent=None, references=None):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', candidate_id): raise ValueError('invalid candidate ID')
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            s = self._read(); self._idle(s); self._budget(s)
            if candidate_id in s['candidates']: raise ValueError('ID exists; use a new ID or read existing candidate')
            parent = parent or s['baseline'] or 'initial'
            if parent not in s['candidates']: raise ValueError('unknown parent')
            refs = references or []
            if any(x not in s['candidates'] for x in refs): raise ValueError('unknown reference')
            p = s['candidates'][parent]
            source = Path(p['spec']['package_path']) if p.get('spec') else self.root / p['workspace']
            dest = self.root / 'workspaces' / candidate_id
            shutil.copytree(source, dest)
            s['candidates'][candidate_id] = {'id':candidate_id,'parent':parent,'references':refs,
                'hypothesis':hypothesis,'status':'draft','workspace':str(dest.relative_to(self.root)),
                'requirements':p.get('requirements',[])}
            atomic_json(self.state_path,s)
        return {'candidate_id':candidate_id,'files':[str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()]}

    def candidate_file(self, candidate_id, path, content=None):
        with exclusive_lock(self.root / '.state.lock', blocking=True):
            s=self._read(); c=s['candidates'][candidate_id]
            root=Path(c['spec']['package_path']) if c.get('spec') else self.root/c['workspace']
            if content is not None:
                self._idle(s); self._budget(s)
                if c['status']!='draft': raise ValueError('frozen candidate: create a descendant to edit')
                if not isinstance(content,str) or len(content)>500_000: raise ValueError('text file too large')
                target=child_path(root,path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(content)
                return {'written':path,'candidate_id':candidate_id}
            target=child_path(root,path)
            if target.stat().st_size>500_000: raise ValueError('file too large for text view')
            return {'candidate_id':candidate_id,'path':path,'content':target.read_text()}

    def dependencies(self,candidate_id,requirements):
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',r) for r in requirements):
            raise ValueError('use exact name==version requirements; URLs and pip options are not accepted')
        with exclusive_lock(self.root/'.state.lock', blocking=True):
            s=self._read(); self._idle(s); self._budget(s)
            c=s['candidates'][candidate_id]
            if c['status']!='draft': raise ValueError('candidate is frozen')
            c['requirements']=requirements
            atomic_json(self.state_path,s)
        return {'requirements':requirements,'prepared':'during batch, isolated venv with lock'}

    def submit_batch(self,candidates,pairs=20,training_episodes=64,league_samples=8):
        if not candidates or len(set(candidates))!=len(candidates) or len(candidates)>16:
            raise ValueError('submit 1..16 distinct candidates')
        if not 1<=pairs<=1000 or not 1<=training_episodes<=100000 or not 0<=league_samples<=1000:
            raise ValueError('invalid experiment size')
        with exclusive_lock(self.root/'.state.lock', blocking=True):
            s=self._read(); self._refresh(s); self._idle(s); self._budget(s)
            for cid in candidates:
                c=s['candidates'][cid]
                if c['status']!='draft': raise ValueError('use new descendants for a new experiment; history retains prior results')
            bid=f'batch-{len(s["batches"]):04d}'
            root=self.root/'batches'/bid;root.mkdir(parents=True)
            b={'id':bid,'status':'running','candidates':candidates,'initial':s['baseline'] is None,
               'pairs':pairs,'training_episodes':training_episodes,'league_samples':league_samples}
            if b['initial'] and candidates!=['initial']: raise ValueError('initialize seed baseline with batch [initial] first')
            packages={}
            for cid in candidates:
                c=s['candidates'][cid];source=self.root/c['workspace']
                if not (source/'strategy.py').exists(): raise ValueError('strategy.py required')
                tree_hash(source)
                package=root/cid/'input';shutil.copytree(source,package)
                packages[cid]={'input':str(package),'requirements':c['requirements'],'source_hash':tree_hash(package)}
                c['status']='submitted'
            descriptor={'run':str(self.root),'batch':b,'packages':packages,'family':s['family'],
                'config':s['config'],'seed':s['seed']+len(s['batches'])*100000,
                'baseline':s['candidates'][s['baseline']]['spec'] if s['baseline'] else PolicySpec().as_dict(),
                'opponents':[p.as_dict() for p in default_opponents()],
                'action_seconds':s['action_seconds'],'workers':s['workers']}
            atomic_json(root/'task.json',descriptor)
            s['batches'].append(b);s['active_batch']=bid
            atomic_json(self.state_path,s)
            self._launch('batch',root,s)
        return {'batch_id':bid,'status':'running','next':'wait for all results, then get batch details and decide'}

    def _launch(self,kind,root,s):
        deadline=time.time()+self._remaining(s)
        log=(root/'worker.log').open('ab')
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent.parent), PYTHONDONTWRITEBYTECODE='1')
        p=subprocess.Popen([sys.executable,'-B','-m','mahjong_ai.research_jobs',kind,str(root),str(deadline)],
                           stdout=log,stderr=log,env=env,start_new_session=True)
        log.close();atomic_json(root/'process.json',{'pid':p.pid,'deadline':deadline})

    def resume_jobs(self):
        with exclusive_lock(self.root/'.state.lock', blocking=True):
            s=self._read();self._refresh(s);self._budget(s)
            for b in s['batches']:
                if b['status'] in ('running','paused'):
                    root=self.root/'batches'/b['id']
                    result=root/'result.json'
                    if result.exists():result.unlink()
                    self._launch('batch',root,s);b['status']='running'
            for a in s['acceptances']:
                if a['status'] in ('running','paused'):
                    root=self.root/'acceptance'/a['id']
                    (root/'public.json').unlink(missing_ok=True)
                    self._launch('acceptance',root,s);a['status']='running'
            atomic_json(self.state_path,s)

    def decide(self,batch_id,retain,conclusion):
        with exclusive_lock(self.root/'.state.lock', blocking=True):
            s=self._read();self._refresh(s)
            if s['active_batch']!=batch_id: raise ValueError('not the active batch')
            b=next(b for b in s['batches'] if b['id']==batch_id)
            if b['status']!='completed': raise ValueError('batch barrier: all experiments must finish before deciding')
            if not conclusion.strip():raise ValueError('record a conclusion with reasons')
            if len(set(retain))!=len(retain) or any(c not in s['candidates'] or not s['candidates'][c].get('spec') for c in retain):
                raise ValueError('retain only evaluated candidates')
            for c in s['candidates'].values():
                if c.get('spec'):c['status']='active' if c['id'] in retain or c['id']==s['baseline'] else 'archived'
            b['status']='decided';s['active_batch']=None
            s['decisions'].append({'batch_id':batch_id,'retain':retain,'conclusion':conclusion,
                                   'snapshot':digest(read_json(self.root/'batches'/batch_id/'result.json'))})
            atomic_json(self.state_path,s)
        return {'status':'decided','batch_id':batch_id,'retained':retain}

    def accept(self,candidate_id):
        with exclusive_lock(self.root/'.state.lock', blocking=True):
            s=self._read();self._refresh(s);self._idle(s);self._budget(s)
            c=s['candidates'][candidate_id]
            if not c.get('spec') or candidate_id==s['baseline']:raise ValueError('submit an evaluated non-baseline candidate')
            for a in s['acceptances']:
                if a['candidate_hash']==c['spec']['package_hash'] and a['baseline']==s['baseline']:
                    return a  # exact candidate/conditions reuse, not fresh significance hunting
            if s['acceptance_granted'] is not None and s['acceptance_used']>=s['acceptance_granted']:raise ValueError('formal acceptance slots exhausted')
            if s['acceptance_granted'] is None:
                if 'unlimited_alpha' not in s:
                    reserved=sum(e['slots']*e['alpha'] for e in s['acceptance_epochs'])
                    s['unlimited_alpha']={'remaining_total':max(1e-12,.05-reserved),'used':0}
                budget=s['unlimited_alpha'];n=budget['used']+1
                epoch={'slots':1,'used':0,'alpha':budget['remaining_total']/(n*(n+1))}
                budget['used']=n
            else: epoch=next(e for e in s['acceptance_epochs'] if e['used']<e['slots'])
            epoch['used']+=1;s['acceptance_used']+=1
            aid=f'acceptance-{s["acceptance_used"]:04d}';root=self.root/'acceptance'/aid
            a={'id':aid,'candidate':candidate_id,'candidate_hash':c['spec']['package_hash'],
               'baseline':s['baseline'],'status':'running'}
            task={'baseline':s['candidates'][s['baseline']]['spec'],'candidate':c['spec'],
                  'opponents':[p.as_dict() for p in default_opponents()], 'config':s['config'],
                  'pairs':1000,'alpha':epoch['alpha'],'seed':int.from_bytes(os.urandom(8),'little'),
                  'workers':s['workers']}
            atomic_json(root/'task.json',task);s['acceptances'].append(a);atomic_json(self.state_path,s)
            self._launch('acceptance',root,s)
        return a

    def history(self,query='',offset=0,limit=10):
        if offset<0 or not 1<=limit<=50:raise ValueError('offset >=0; limit 1..50')
        self.status();s=self._read()
        rows=[]
        for c in s['candidates'].values():rows.append({'kind':'candidate',**{k:v for k,v in c.items() if k not in ('workspace','spec')}})
        rows += [{'kind':'decision',**d} for d in s['decisions']]
        rows += [{'kind':'batch',**b} for b in s['batches']]
        rows += [{'kind':'acceptance',**a} for a in s['acceptances']]
        rows=[r for r in rows if query.lower() in json.dumps(r,ensure_ascii=False).lower()]
        return {'total':len(rows),'items':rows[offset:offset+limit],'next_offset':offset+limit if offset+limit<len(rows) else None}

    def details(self,kind,object_id,offset=0,limit=10):
        if offset<0 or not 1<=limit<=50:raise ValueError('invalid pagination')
        self.status();s=self._read()
        if kind=='candidate':return s['candidates'][object_id]
        if kind=='batch':
            if object_id not in [b['id'] for b in s['batches']]:raise ValueError('unknown batch')
            b=next(b for b in s['batches'] if b['id']==object_id)
            if b['status'] not in ('completed','decided'):return {'status':b['status'],'results_withheld_until_batch_complete':True}
            return read_json(self.root/'batches'/object_id/'result.json')
        if kind=='acceptance':return next(a for a in s['acceptances'] if a['id']==object_id)
        if kind=='trace':
            # Only public decision observations, never engine event logs or hidden acceptance files.
            cid=object_id
            if cid not in s['candidates']:raise ValueError('unknown candidate')
            for b in reversed(s['batches']):
                if cid in b['candidates'] and b['status'] in ('completed','decided'):
                    path=self.root/'batches'/b['id']/cid/'traces.json'
                    rows=read_json(path) if path.exists() else []
                    return {'total':len(rows),'items':rows[offset:offset+limit]}
            return {'total':0,'items':[]}
        raise ValueError('kind must be candidate, batch, acceptance or trace')

    def sample(self,limit=4):
        s=self._read();eligible=[c for c in s['candidates'].values() if c.get('spec')]
        eligible.sort(key=lambda c:(c['id']!=s['baseline'],c['status']!='active',c.get('summary',{}).get('candidate_mean_rank',4),c['id']))
        # Measured behavior niches, not self-reported labels. No extra promotion reward.
        chosen=[];seen=set();limit=max(1,min(limit,16))
        for c in eligible:
            signature=c.get('summary',{}).get('behavior_signature',c['id'])
            if signature not in seen or c['id']==s['baseline']:
                chosen.append(c);seen.add(signature)
            if len(chosen)==limit:break
        for c in eligible:
            if len(chosen)==limit:break
            if c not in chosen:chosen.append(c)
        return {'recommendations':[{'id':c['id'],'hypothesis':c['hypothesis'],'summary':c.get('summary',{})} for c in chosen],
                'reason':'baseline and measured rank leaders plus an alternative; compare only matching protocols; history remains selectable'}

    def cancel(self,kind,object_id):
        """Terminate owned computation; cancellation is not a win/loss observation."""
        import signal
        with exclusive_lock(self.root/'.state.lock', blocking=True):
            s=self._read()
            if kind=='batch':
                item=next(b for b in s['batches'] if b['id']==object_id);base='batches'
            elif kind=='acceptance':
                item=next(a for a in s['acceptances'] if a['id']==object_id);base='acceptance'
            else:raise ValueError('kind must be batch or acceptance')
            root=self.root/base/object_id
            if item['status'] not in ('running','paused'):return {'status':item['status']}
            process=root/'process.json'
            if process.exists():
                pid=read_json(process)['pid']
                command=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
                if 'mahjong_ai.research_jobs' in command and str(root) in command:
                    try:os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for _ in range(100):
                try:
                    with exclusive_lock(root/'.job.lock'):break
                except RuntimeError:time.sleep(.02)
            else:raise RuntimeError('job has not stopped; retry cancel')
            if kind=='batch':
                rows={}
                for cid in item['candidates']:
                    result=root/cid/'result.json'
                    rows[cid]=read_json(result) if result.exists() else {'status':'cancelled','candidate':cid,'error':'cancelled by researcher'}
                atomic_json(root/'result.json',{'status':'completed','candidates':rows,'league':{},'cancelled':True})
            else:atomic_json(root/'public.json',{'status':'cancelled','reason':'cancelled by researcher; slot remains spent'})
            self._refresh(s);atomic_json(self.state_path,s)
        return {'status':'cancelled','id':object_id}
