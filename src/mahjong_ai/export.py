"""Portable policy bundles. Export does not certify or submit a competition release."""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from .policy import PolicySpec
from .storage import atomic_json, environment_info, read_json, runtime_fingerprint


def export_baseline(root, family, output):
    original = PolicySpec(**read_json(Path(root) / 'baselines.json')[family])
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('export destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    if original.model_path:
        shutil.copy2(original.model_path, destination / 'model.npz')
    policy = replace(original, model_path='model.npz' if original.model_path else None)
    atomic_json(destination / 'policy.json', policy.as_dict())
    atomic_json(destination / 'manifest.json', {'policy_id': original.identity,
                'runtime': runtime_fingerprint(), 'environment': environment_info(),
                'scope': 'experimental local baseline; online certification pending'})
    return {'policy': str(destination / 'policy.json'), 'family': family}
