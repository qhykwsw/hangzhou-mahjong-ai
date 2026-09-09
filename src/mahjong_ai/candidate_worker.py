"""Worker side of the public-observation protocol. stdout is reserved for RPC."""
import importlib.util
import json
import sys
import os
import threading
import time
from pathlib import Path

from .types import Action, Meld, Observation, PublicEvent, RoundResult, StageContext


def observation(data):
    d = dict(data)
    d['stage'] = StageContext(**d['stage'])
    d['legal_actions'] = tuple(Action(**(a | {'tiles': tuple(a.get('tiles', []))})) for a in d['legal_actions'])
    d['public_events'] = tuple(PublicEvent(**e) for e in d['public_events'])
    d['melds'] = tuple(tuple(Meld(**m) for m in row) for row in d['melds'])
    for key in ('hand', 'hand_sizes'): d[key] = tuple(d[key])
    d['discards'] = tuple(tuple(row) for row in d['discards'])
    return Observation(**d)


def main():
    parent=os.getppid()
    def parent_watch():
        while True:
            time.sleep(1)
            if os.getppid()!=parent:os._exit(1)
    threading.Thread(target=parent_watch,daemon=True).start()
    package, output = map(Path, sys.argv[1:3])
    channel = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, str(package))
    def send(value):
        channel.write(json.dumps(value, allow_nan=False) + '\n')
        channel.flush()
    def read():
        line = sys.stdin.readline()
        if not line: raise EOFError()
        return json.loads(line)
    def dispatch(policy, request):
        op = request['op']
        if op == 'act':
            obs = observation(request['observation'])
            action = policy.act(obs)
            return {'index': obs.legal_actions.index(action)}
        if op == 'reset': policy.reset(request['seed'])
        elif op == 'finish': policy.finish_round(RoundResult(**request['result']), request['seat'])
        else: raise ValueError('unknown policy operation')
        return {'ok': True}
    class TrainingContext:
        objective = 'four-player-settlement-rank-v1'
        def __init__(self, request):
            self.episodes = request['episodes']
            self.config = request['config']
            self.output = output
            self.initial = package / 'model'
            self.completed_episodes=request.get('completed_episodes',0)
            self.checkpoint_metadata=request.get('checkpoint')
        def checkpoint(self,metadata=None):
            send({'request':'checkpoint','metadata':metadata})
            reply=read()
            if not reply.get('ok'):raise RuntimeError('checkpoint not committed')
        def play(self, policy):
            send({'request': 'play'})
            while True:
                request = read()
                if 'result' in request and 'op' not in request:
                    return RoundResult(**request['result']), request['seat']
                send(dispatch(policy, request))
    try:
        module_spec = importlib.util.spec_from_file_location('candidate', package / 'strategy.py')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        policy = None
        while True:
            request = read()
            if request['op'] == 'build':
                policy = module.build(package / 'model')
                send({'ok': True})
            elif request['op'] == 'train':
                metrics = module.train(TrainingContext(request))
                send({'done': True, 'metrics': metrics or {}})
            else: send(dispatch(policy, request))
    except EOFError: pass
    except Exception as exc:
        send({'error': f'{type(exc).__name__}: {exc}'})


if __name__ == '__main__': main()
