"""Binary public Actor inputs for a local inference service; numpy only.

Requests carry v1 rows (guided-current-hand-v1); the service extends them for
a model of a wider encoding (guided.encoding.augment).
"""
import struct

import numpy as np

from .actions import N_ACTIONS
from .encoding import N_PLANES, N_SCALARS

_HEADER = struct.Struct('<4s64sBI')
_MAGIC = b'MJE1'


def pack(model, row, *, lookahead):
    model = model.encode('ascii')
    if len(model) != 64 or any(c not in b'0123456789abcdef' for c in model):
        raise ValueError('model must be a weights SHA-256')
    arrays = [np.asarray(row[k], dtype=d) for k, d in (
        ('planes', '<f2'), ('scalars', '<f2'), ('events', '<i2'), ('mask', '?'))]
    return _HEADER.pack(_MAGIC, model, bool(lookahead), len(arrays[2])) + b''.join(a.tobytes() for a in arrays)


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
    expected = _HEADER.size + 2 * (N_PLANES * 34 + N_SCALARS + length * 5) + N_ACTIONS
    if len(body) != expected:
        raise ValueError('incorrect Actor input body size')
    row, offset = {}, _HEADER.size
    for name, dtype, count, itemsize, shape in layout:
        row[name] = np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += count * itemsize
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

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
