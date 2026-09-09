"""Operator entry point. Budget additions are never exposed to the research agent."""
import argparse
from datetime import date
from itertools import count
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from .research import Research
from .research_templates import CONTRACT
from .storage import atomic_json, exclusive_lock
from .types import RoundConfig


def create_run_root(family):
    """Reserve a dated directory atomically; a resumed run keeps its original name."""
    base = Path('.local/runs').resolve()
    base.mkdir(parents=True, exist_ok=True)
    for number in count(1):
        root = base / f'{date.today():%Y%m%d}-{family}-{number:02d}'
        try:
            root.mkdir(mode=0o700)
            return root
        except FileExistsError:
            continue


def list_runs(base):
    """Inspect persisted metadata without loading or changing frozen runtimes."""
    rows = []
    for root in sorted(base.iterdir()) if base.exists() else []:
        path = root / 'run.json'
        if root.is_symlink() or not path.is_file():
            continue
        state = json.loads(path.read_text())
        engine_path = root / 'runtime/mahjong_ai/engine.py'
        engine = re.search(r"^ENGINE_VERSION\s*=\s*['\"]([^'\"]+)", engine_path.read_text(), re.M) if engine_path.exists() else None
        rows.append({'name': root.name, 'root': str(root.resolve()), 'run_id': state['id'],
                     'family': state['family'], 'baseline': state['baseline'],
                     'engine': engine[1] if engine else None,
                     'batches': len(state['batches']), 'acceptances': len(state['acceptances'])})
    return {'root': str(base.resolve()), 'runs': rows}


def toml_value(value):
    if isinstance(value,dict):return '{'+','.join(json.dumps(k)+'='+toml_value(v) for k,v in value.items())+'}'
    if isinstance(value,list):return '['+','.join(toml_value(v) for v in value)+']'
    return json.dumps(value)


def codex_command(root,thread=None):
    # Only preserve the configured model/provider, never unrelated tools or plugins.
    path=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'config.toml'
    config=tomllib.loads(path.read_text()) if path.exists() else {}
    argv=['codex','exec','--ignore-user-config','--skip-git-repo-check','--json']
    settings={'sandbox_mode':'read-only','approval_policy':'never','features.shell_tool':False,
              'features.unified_exec':False,'web_search':'disabled',
              'mcp_servers.mahjong_research':{'command':sys.executable,
                'args':['-B','-m','mahjong_ai.research_mcp',str(root)],
                'env':{'PYTHONPATH':str(Path(__file__).parent.parent)},'required':True,
                'tool_timeout_sec':45,'startup_timeout_sec':20}}
    for key in ('model','model_provider','model_reasoning_effort'):
        if key in config:settings[key]=config[key]
    provider=config.get('model_provider')
    if provider and provider in config.get('model_providers',{}):
        settings['model_providers.'+provider]=config['model_providers'][provider]
    for k,v in settings.items():argv+=['-c',k+'='+toml_value(v)]
    if thread:argv+=['resume',thread]
    argv+=['-']
    return argv


def stop_jobs(research):
    for base in ('batches','acceptance'):
        for path in (research.root/base).glob('*/process.json'):
            from .storage import read_json
            pid=read_json(path)['pid']
            command=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
            if 'mahjong_ai.research_jobs' in command and str(path.parent) in command:
                try:os.killpg(pid,signal.SIGTERM)
                except (ProcessLookupError, PermissionError):pass
    with exclusive_lock(research.root/'.state.lock', blocking=True):
        state=research._read()
        for batch in state['batches']:
            if batch['status']=='running':batch['status']='paused'
        for acceptance in state['acceptances']:
            if acceptance['status']=='running':acceptance['status']='paused'
        atomic_json(research.state_path,state)


def drive(research,brief=''):
    root=research.root
    p=None
    with exclusive_lock(root/'.agent.lock'):
        research.begin()
        try:
            research.resume_jobs()
            idle_turns=0
            while research.status()['seconds_remaining']>0:
                state=research._read()
                if state['acceptance_granted'] is not None and state['acceptance_used']>=state['acceptance_granted'] and not any(a['status']=='running' for a in state['acceptances']):break
                # One persistent thread; each turn can use arbitrary research tools.
                prompt=(CONTRACT+'\nYou are the dedicated '+state['family']+' research agent. '
                    'Use ONLY mahjong_research tools. Do not use shell, files outside candidates, external tools, or subagents. '
                    'Start by reading status and relevant persisted history. Complete a multi-candidate batch, '
                    'wait for every result, record the unified population decision and submit a promising candidate '
                    'for formal acceptance if useful. You may execute several batches per turn. '
                    'Continue until budget ends; do not stop just because a first batch ran. '
                    'If you return, the controller will continue this same research thread.\n'+brief)
                n=len(state['events']); log=root/'agent'/f'turn-{n:05d}.jsonl';log.parent.mkdir(exist_ok=True)
                cwd=root/'agent'/'workspace';cwd.mkdir(exist_ok=True)
                with log.open('ab') as stream:
                    p=subprocess.Popen(codex_command(root,state['thread_id']),stdin=subprocess.PIPE,
                                       stdout=stream,stderr=subprocess.DEVNULL,cwd=cwd,start_new_session=True)
                    p.stdin.write(prompt.encode());p.stdin.close()
                    while p.poll() is None:
                        research.heartbeat()
                        status=research.status()
                        acceptance_done=status['acceptance_remaining']==0 and not any(a['status']=='running' for a in status['acceptances'])
                        if status['seconds_remaining']<=0 or acceptance_done:
                            os.killpg(p.pid,signal.SIGTERM);break
                        time.sleep(1)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                thread=None
                for line in log.read_text().splitlines():
                    try:event=json.loads(line)
                    except json.JSONDecodeError:continue
                    if event.get('type')=='thread.started':thread=event.get('thread_id')
                with exclusive_lock(root/'.state.lock', blocking=True):
                    current=research._read()
                    if thread:current['thread_id']=thread
                    current['events'].append({'kind':'agent_turn','returncode':p.returncode,'log':str(log.relative_to(root))})
                    atomic_json(research.state_path,current)
                if p.returncode:break
                updated=research._read()
                progressed=len(updated['batches'])!=len(state['batches']) or len(updated['decisions'])!=len(state['decisions']) or updated['acceptance_used']!=state['acceptance_used']
                idle_turns=0 if progressed else idle_turns+1
                if idle_turns>=3:break  # provider/no-progress condition, resume keeps all state
        finally:
            if p is not None and p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
            try:stop_jobs(research)
            finally:research.end()
    return research.status()


def main(argv=None):
    parser=argparse.ArgumentParser(description='Persistent population research with a dedicated Codex')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('init','run','status','budget','history','list'):
        p=sub.add_parser(name)
        p.add_argument('--root', required=name not in ('init','list'),
                       help='run directory; init defaults to .local/runs/YYYYMMDD-family-NN; list uses the runs parent')
        if name=='init':
            p.add_argument('--family',choices=['rule','rl'],required=True)
            p.add_argument('--seconds',type=float,default=3600);p.add_argument('--acceptance-slots',type=int,default=None)
            p.add_argument('--rounds',type=int,default=8);p.add_argument('--games-per-round',type=int,default=1)
            p.add_argument('--workers',type=int,default=2);p.add_argument('--action-seconds',type=float,default=.05)
            p.add_argument('--seed',type=int,default=0)
        if name=='budget':
            p.add_argument('--seconds',type=float,default=0);p.add_argument('--slots',type=int,default=0)
            p.add_argument('--unlimited-acceptance',action='store_true')
        if name=='history':p.add_argument('--query',default='')
        if name=='run':p.add_argument('--brief',default='')
    args=parser.parse_args(argv)
    if args.command=='list':
        print(json.dumps(list_runs(Path(args.root or '.local/runs').resolve()),ensure_ascii=False,indent=2))
        return 0
    root=Path(args.root).resolve() if args.root else create_run_root(args.family)
    # Freeze runtime before initialization; later work in the repository cannot invalidate a long run.
    runtime=root/'runtime'
    if not os.environ.get('MAHJONG_FROZEN_RESEARCH'):
        if args.command=='init':
            if runtime.exists():raise ValueError('runtime already exists')
            shutil.copytree(Path(__file__).parent,runtime/'mahjong_ai',ignore=shutil.ignore_patterns('__pycache__'))
        if not runtime.exists():raise ValueError('run runtime missing; initialize first')
        env=dict(os.environ,PYTHONPATH=str(runtime),MAHJONG_FROZEN_RESEARCH='1')
        forwarded=list(argv if argv is not None else sys.argv[1:])
        if args.root is None:forwarded+=['--root',str(root)]
        return subprocess.call([sys.executable,'-B','-m','mahjong_ai.research_cli',*forwarded],env=env)
    r=Research(root)
    if args.command=='init':result=r.initialize(args.family,RoundConfig(rounds=args.rounds,games_per_round=args.games_per_round),args.seconds,args.acceptance_slots,args.workers,args.action_seconds,args.seed)
    elif args.command=='run':result=drive(r,args.brief)
    elif args.command=='budget':result=r.add_budget(args.seconds,args.slots,args.unlimited_acceptance)
    elif args.command=='history':result=r.history(args.query)
    else:result=r.status()
    print(json.dumps({'root':str(root),**result},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
