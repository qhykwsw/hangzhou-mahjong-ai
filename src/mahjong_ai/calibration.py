"""Read-only guide fetch and explicitly stateless official fan-calculator checks."""
from __future__ import annotations

import json
import random
import ssl
import time
import urllib.request
from pathlib import Path

from .rules import CODES, calculate_fan, payments, tile_id
from .storage import atomic_json, runtime_fingerprint, environment_info


def _request(base, path, body=None, insecure=False):
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    request = urllib.request.Request(base.rstrip('/') + path,
               data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
               headers={'Content-Type': 'application/json'}, method='POST' if body is not None else 'GET')
    with urllib.request.urlopen(request, timeout=15, context=context) as response:
        return json.load(response)


def samples(count=32, seed=104):
    rng = random.Random(seed)
    cases = [
        {'hand': ['1w','1w','2w','2w','3w','3w','4w','5w','6w','白','白','白','白'],
         'draw': '9t', 'chain': {'count': 0, 'piao': 0}, 'base': 1},
        {'hand': ['1w','1w','1w','2w','3w','4w','5w','6w','7w','8w','9w','东','东'],
         'draw': '东', 'chain': {'count': 0, 'piao': 0}, 'base': 1},
    ]
    while len(cases) < count:
        if len(cases) % 4 == 0:
            tiles = rng.sample(list(range(34)) * 4, 14)
        elif len(cases) % 3 == 0:
            tiles = [t for t in rng.sample(range(33), 7) for _ in range(2)]
        else:
            tiles = []
            for _ in range(4):
                if rng.random() < .7:
                    start = rng.choice([0, 9, 18]) + rng.randrange(7)
                    tiles.extend((start, start + 1, start + 2))
                else:
                    tiles.extend([rng.randrange(33)] * 3)
            tiles.extend([rng.randrange(33)] * 2)
        for i in rng.sample(range(14), rng.randrange(5)):
            tiles[i] = 33
        if any(tiles.count(t) > 4 for t in set(tiles)):
            continue
        rng.shuffle(tiles)
        cases.append({'hand': [CODES[t] for t in tiles[:13]], 'draw': CODES[tiles[-1]],
                      'chain': {'count': rng.randrange(4), 'piao': 0}, 'base': rng.choice([1, 2, 5])})
    return cases


def calibrate(base, output, count=32, insecure=False):
    meta = _request(base, '/portal/api/guide/version', insecure=insecure)
    records = []
    for request in samples(count):
        draw = tile_id(request['draw'])
        h = [tile_id(t) for t in request['hand']] + [draw]
        local = calculate_fan(h, draw, chain_count=request['chain']['count'])
        remote = _request(base, '/portal/api/tools/fan-calc', request, insecure)
        expected = {'hu': local is not None, 'baotou': local.baotou if local else False,
                    'fan': local.fan if local else 0}
        matched = bool(remote.get('hu')) == expected['hu']
        if local:
            matched = matched and remote['fan'] == local.fan and remote['baotou'] == local.baotou
            matched = matched and remote['scores']['dealer_hu']['win'] == payments(0, 0, local.fan, request['base'])[0]
        records.append({'request': request, 'local': expected, 'official': remote, 'matched': matched})
        time.sleep(.14)  # official 10/s/IP; stay below it
    result = {'runtime': runtime_fingerprint(), 'environment': environment_info(), 'guide_version': meta['version'], 'updated_at': meta['updated_at'],
              'source': base.rstrip('/') + '/portal/api/tools/fan-calc',
              'samples': len(records), 'matched': sum(r['matched'] for r in records),
              'scope': 'closed-hand hu/baotou/fan/dealer payment only; NOT full engine certification',
              'records': records}
    atomic_json(output, result)
    return result
