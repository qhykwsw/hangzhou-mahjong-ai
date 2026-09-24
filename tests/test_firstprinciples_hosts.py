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
