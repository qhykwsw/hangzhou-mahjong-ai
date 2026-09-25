"""Host integration must preserve attempts across pool replacement."""
import importlib
import json
from pathlib import Path


def test_attempt_ids_do_not_reuse_an_earlier_pool(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'projects/personal/firstprinciples'))
    evaluate = importlib.import_module('harness.evaluate')
    monkeypatch.setattr(evaluate, 'LOCAL', tmp_path)
    (tmp_path / 'pool.json').write_text(json.dumps({'id': 'new-pool'}))
    previous = {'pool': 'old-pool', 'line': 'rl', 'id': 'fp-rl-007'}
    record = json.dumps(previous) + '\n'
    (tmp_path / 'attempts.jsonl').write_text(record)
    assert evaluate.reserve_attempt('rl', '5070ti', 'new host') == 'fp-rl-008'
    assert evaluate.reserve_attempt('rl', '4090', 'parallel host') == 'fp-rl-009'
    assert (tmp_path / 'attempts.jsonl').read_text() == record


def test_frozen_source_inventory_does_not_read_enclosing_git(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'projects/personal/selfplay'))
    from guided import remote as training
    from mahjong_ai.execution import remote as execution
    files = {'src/mahjong_ai/execution/model_seat.py': 'one',
             'projects/personal/selfplay/src/selfplay_core/wire.py': 'two',
             'projects/personal/firstprinciples/harness/rl.py': 'three'}
    (tmp_path / 'source-manifest.json').write_text(json.dumps({'files': files}))
    monkeypatch.setattr(execution, '__file__', str(tmp_path / 'src/mahjong_ai/execution/remote.py'))
    monkeypatch.setattr(training.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not call git')))
    assert training.source_files(tmp_path) == sorted(files)
    assert execution.service_code_files() == sorted(list(files)[:2])
