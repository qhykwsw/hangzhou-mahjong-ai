"""Per-device warmup streams share cuBLAS workspaces across learner captures."""

import torch

_WARMUP_STREAMS = {}


def warmup_stream(device):
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _WARMUP_STREAMS:
        _WARMUP_STREAMS[index] = torch.cuda.Stream(device=index)
    return _WARMUP_STREAMS[index]
