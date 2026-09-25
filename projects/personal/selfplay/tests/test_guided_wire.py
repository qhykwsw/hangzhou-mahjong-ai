"""Transport framing preserves full public tensors and rejects corrupt bodies."""
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from guided.model_server import Service, blank, decode
from guided.networks import Actor
from selfplay_core.wire import pack, unpack


@pytest.mark.parametrize('length', [1, 63, 129, 4096])
def test_binary_transport_matches_json_inputs(length):
    row = blank()
    rng = np.random.default_rng(914)
    row['planes'][:] = rng.uniform(size=row['planes'].shape)
    row['events'] = np.tile([2, 1, 7, 34, 0], (length, 1)).astype(np.int16)
    old = decode({k: base64.b64encode(v.tobytes()).decode() for k, v in row.items()})
    body = pack('a' * 64, row, lookahead=True)
    request = unpack(body)
    assert request['model'] == 'a' * 64 and request['lookahead'] is True
    for key in row:
        np.testing.assert_array_equal(request['encoded'][key], old[key])
    for bad in (b'', body[:60], body[:-1], body + b'x', b'BAD!' + body[4:]):
        with pytest.raises(ValueError): unpack(bad)


def test_binary_and_json_requests_share_batch_and_decision():
    torch.set_num_threads(1)
    torch.manual_seed(915)
    actor = Actor(16, 1, 4).eval()
    models = SimpleNamespace(device='cpu', get=lambda name: (actor, {'config': {'lookahead': True}}))
    service = Service(models)
    row = blank()
    row['mask'][:34] = True
    legacy = dict(model='a' * 64, lookahead=True,
                  encoded={k: base64.b64encode(v.tobytes()).decode() for k, v in row.items()})
    binary = unpack(pack('a' * 64, row, lookahead=True))
    a, b = service.decide('a' * 64, [legacy, binary])
    assert a == b and 'index' in a


def test_http_loop_replies_to_synchronous_cpu_launch_and_bad_batch(tmp_path):
    import http.client
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import time

    project = Path(__file__).resolve().parents[1]
    repo = project.parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [project, project / 'src', repo / 'src'])),
               OMP_NUM_THREADS='1')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    program = '''import sys,torch
from types import SimpleNamespace
from guided.networks import Actor
from guided.model_server import Service,serve
torch.set_num_threads(1)
actor=Actor(16,1,4).eval()
models=SimpleNamespace(device='cpu',get=lambda name:(actor,{'config':{'lookahead':True}}))
serve(Service(models),int(sys.argv[1]))
'''
    with (tmp_path / 'server.log').open('wb') as log:
        child = subprocess.Popen([sys.executable, '-c', program, str(port)], env=env, stdout=log, stderr=log)
    conn = None
    try:
        for _ in range(100):
            try:
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                conn.request('GET', '/health')
                response = conn.getresponse()
                assert response.status == 200
                response.read()
                break
            except OSError:
                if conn: conn.close()
                time.sleep(.05)
        else:
            pytest.fail('model service did not start')
        from selfplay_core.wire import SocketClient
        native = SocketClient('127.0.0.1', port, 2)
        try:
            body = pack('a' * 64, blank(), lookahead=True)
            assert native.decide(body)['index'] == 0
            with pytest.raises(RuntimeError, match='truncated'):
                native.decide(b'broken')
            assert native.decide(body)['index'] == 0
        finally:
            native.close()
        from selfplay_core.wire import FRAME_HEADER, FRAME_MAGIC
        # A peer can disappear halfway through a header or after submitting
        # a complete request whose result has not yet been sent.
        frame = FRAME_HEADER.pack(FRAME_MAGIC, len(body)) + body
        for prefix in (frame[:5], frame):
            abrupt = socket.create_connection(('127.0.0.1', port), 2)
            abrupt.sendall(prefix)
            abrupt.close()
        conn.request('GET', '/health')
        response = conn.getresponse()
        assert response.status == 200
        response.read()
        # This parses successfully and fails during batch preparation. No
        # device event exists to wake the selector, but it must reply now.
        conn.request('POST', '/act', json.dumps(dict(model='a' * 64, lookahead=True, encoded={})))
        response = conn.getresponse()
        assert response.status == 400 and 'error' in json.loads(response.read())
    finally:
        if conn: conn.close()
        child.terminate()
        child.wait(timeout=10)


def test_native_frames_survive_fragmentation_and_coalescing():
    from guided.model_server import Connection, requests
    from selfplay_core.wire import FRAME_HEADER, FRAME_MAGIC

    body = pack('a' * 64, blank(), lookahead=True)
    frame = FRAME_HEADER.pack(FRAME_MAGIC, len(body)) + body
    connection = Connection(None)
    for byte in frame[:-1]:
        connection.inbox.append(byte)
        assert list(requests(connection)) == []
    connection.inbox += frame[-1:] + frame
    assert list(requests(connection)) == [('BINARY', '/act', body)] * 2
    connection.inbox += FRAME_HEADER.pack(b'BAD!', len(body))
    with pytest.raises(ValueError):
        list(requests(connection))


def test_native_frames_reject_oversize_before_body():
    from guided.model_server import Connection, requests, MAX_BODY
    from selfplay_core.wire import FRAME_HEADER, FRAME_MAGIC

    connection = Connection(None)
    connection.inbox += FRAME_HEADER.pack(FRAME_MAGIC, MAX_BODY + 1)
    with pytest.raises(ValueError):
        list(requests(connection))
