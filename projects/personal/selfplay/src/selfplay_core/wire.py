"""Binary public Actor inputs for a local inference service; numpy only.

Requests carry v1 rows (guided-current-hand-v1); the service extends them for
a model of a wider encoding (guided.encoding.augment).

guided-current-hand-v4 also reads a semifinal or final row's ``risk_beta``,
the one input not computed from the v1 row. A request to a model whose
encoding reads it carries it as one more float16 after the mask; every other
request is the plain body, byte for byte as before v4, which a service from
before v4 accepts. pack writes the plain body and keeps the row's risk_beta
beside it (Body.risk); SocketClient appends it only for a model the service
says reads it (POST /inputs, asked once per model; no answer: the plain body).
"""
import struct

import numpy as np

from .actions import N_ACTIONS
from .encoding import N_PLANES, N_SCALARS

_HEADER = struct.Struct('<4s64sBI')
_MAGIC = b'MJE1'


class Body(bytes):
    """A packed request: the plain body, and ``risk`` (a row's risk_beta as a float16, or b'') beside it."""

    risk = b''


def _size(length):
    """The plain body's size for ``length`` events."""
    return _HEADER.size + 2 * (N_PLANES * 34 + N_SCALARS + length * 5) + N_ACTIONS


def pack(model, row, *, lookahead):
    model = model.encode('ascii')
    if len(model) != 64 or any(c not in b'0123456789abcdef' for c in model):
        raise ValueError('model must be a weights SHA-256')
    arrays = [np.asarray(row[k], dtype=d) for k, d in (
        ('planes', '<f2'), ('scalars', '<f2'), ('events', '<i2'), ('mask', '?'))]
    body = Body(_HEADER.pack(_MAGIC, model, bool(lookahead), len(arrays[2])) + b''.join(a.tobytes() for a in arrays))
    if 'risk_beta' in row:
        body.risk = np.asarray([row['risk_beta']], dtype='<f2').tobytes()
    return body


def unpack(body):
    if len(body) < _HEADER.size:
        raise ValueError('truncated Actor input header')
    magic, model, lookahead, length = _HEADER.unpack_from(body)
    if magic != _MAGIC or lookahead not in (0, 1) or not 1 <= length <= 4096:
        raise ValueError('invalid Actor input header')
    if any(c not in b'0123456789abcdef' for c in model):
        raise ValueError('invalid model SHA-256')
    layout = [('planes', '<f2', N_PLANES * 34, 2, (N_PLANES, 34)),
              ('scalars', '<f2', N_SCALARS, 2, (N_SCALARS,)),
              ('events', '<i2', length * 5, 2, (length, 5)),
              ('mask', '?', N_ACTIONS, 1, (N_ACTIONS,))]
    expected = _size(length)
    if len(body) not in (expected, expected + 2):
        raise ValueError('incorrect Actor input body size')
    row, offset = {}, _HEADER.size
    for name, dtype, count, itemsize, shape in layout:
        row[name] = np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += count * itemsize
    if len(body) > expected:
        risk = float(np.frombuffer(body, dtype='<f2', count=1, offset=expected)[0])
        if not -1.0 <= risk <= 1.0:
            raise ValueError('invalid risk coefficient')
        row['risk_beta'] = risk
    return dict(model=model.decode('ascii'), lookahead=bool(lookahead), encoded=row, _binary=True)


FRAME_MAGIC = b'MJP1'
FRAME_HEADER = struct.Struct('<4sI')
REPLY = struct.Struct('<i')
MAX_ERROR = 2000


class SocketClient:
    """One request in flight per policy seat; HTTP control traffic stays separate."""

    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.socket = None
        self.reads_risk = {}  # model -> whether its encoding reads risk_beta (the service's /inputs)

    def _read(self, count):
        data = bytearray()
        while len(data) < count:
            part = self.socket.recv(count - len(data))
            if not part:
                raise ConnectionError('inference service closed the response')
            data.extend(part)
        return data

    def decide(self, body):
        import socket

        risk = getattr(body, 'risk', b'')
        if risk and self._reads_risk(body[4:68].decode('ascii')):
            body = bytes(body) + risk
        if self.socket is None:
            self.socket = socket.create_connection((self.host, self.port), self.timeout)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.sendall(FRAME_HEADER.pack(FRAME_MAGIC, len(body)) + body)
        value = REPLY.unpack(self._read(REPLY.size))[0]
        if value < 0:
            if -value > MAX_ERROR:
                raise ValueError('oversized inference error')
            raise RuntimeError(self._read(-value).decode('utf-8', errors='replace'))
        if value >= N_ACTIONS:
            raise ValueError('invalid inference action index')
        return {'index': value}

    def _reads_risk(self, model):
        """Whether ``model`` reads risk_beta, asked of the service once (only for a row that has one).

        A service from before v4 answers 404 and one that cannot load the
        model 400: both mean no. Without an answer it is no as well, kept for
        the model: a service that does not answer must not cost a wait per decision.
        """
        if model not in self.reads_risk:
            import http.client
            import json

            connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            try:
                connection.request('POST', '/inputs', json.dumps({'model': model}),
                                   {'Content-Type': 'application/json'})
                response = connection.getresponse()
                data = response.read()
            except (OSError, http.client.HTTPException):
                self.reads_risk[model] = False
                return False
            finally:
                connection.close()
            try:
                answer = response.status == 200 and json.loads(data).get('risk_beta') is True
            except ValueError:
                answer = False
            self.reads_risk[model] = answer
        return self.reads_risk[model]

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
