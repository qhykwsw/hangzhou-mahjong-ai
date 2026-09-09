import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from mahjong_ai.engine import MahjongEngine
from mahjong_ai.evaluator import Evaluator, EvaluationReport, promotion_decision
from mahjong_ai.generation import CommandGenerator, SearchGenerator, validate_proposal
from mahjong_ai.iteration import IterationRunner, IterationRunConfig, default_opponents
from mahjong_ai.policy import PolicySpec, RLPolicy, RulePolicy, TrainingRecipe
from mahjong_ai.program import compile_program
from mahjong_ai.storage import read_json
from mahjong_ai.training import train
from mahjong_ai.types import RANK_OBJECTIVE, RoundConfig


def test_identical_policies_pair_to_zero_and_resume_without_duplicate_samples(tmp_path):
    config = RoundConfig(rounds=2)
    spec = PolicySpec()
    evaluator = Evaluator()
    a = evaluator.evaluate(spec, spec, default_opponents(), config, pairs=4, output=tmp_path)
    b = evaluator.evaluate(spec, spec, default_opponents(), config, pairs=4, output=tmp_path)
    assert len(a.pairs) == len(b.pairs) == 4
    assert [p['delta'] for p in a.pairs] == [0] * 4
    assert a.summary()['actual_rounds_started'] == 8
    assert promotion_decision(a)['status'] == 'inconclusive'
    with pytest.raises(ValueError, match='protocol differs'):
        evaluator.evaluate(spec, spec, default_opponents(), RoundConfig(rounds=3), pairs=4, output=tmp_path)


def test_three_opponents_required():
    with pytest.raises(ValueError, match='THREE'):
        Evaluator().evaluate(PolicySpec(), PolicySpec(), default_opponents()[:2], RoundConfig())


def test_model_trains_from_terminal_rank_and_checkpoint_resume_matches_uninterrupted(tmp_path):
    recipe = TrainingRecipe(seed=33)
    cfg = RoundConfig(rounds=2)
    full = train(recipe, default_opponents(), cfg, 4, tmp_path / 'full', seed=71)
    # Re-running the completed protocol must not update weights or train again.
    before = (tmp_path / 'full/model.npz').read_bytes()
    repeat = train(recipe, default_opponents(), cfg, 4, tmp_path / 'full', seed=71)
    with np.load(tmp_path / 'full/model.npz') as saved:
        learner = RLPolicy(recipe, training=False)
        MahjongEngine().run_round([learner, *[RulePolicy() for _ in range(3)]], cfg, 71)
        assert not np.array_equal(saved['w'], learner.params['w'])
    assert full['status'] == repeat['status'] == 'completed'
    assert sum(e['updated'] for e in full['episodes']) > 0
    assert all(e['reward'] == e['rank_utility'] / recipe.reward_scale for e in full['episodes'] if e['updated'])
    model1 = RLPolicy(recipe, tmp_path / 'full/model.npz')
    model2 = RLPolicy(recipe, tmp_path / 'full/model.npz')
    engine = MahjongEngine(record_events=False)
    a = engine.run_round([model1, *[RulePolicy() for _ in range(3)]], cfg, 72)
    b = engine.run_round([model2, *[RulePolicy() for _ in range(3)]], cfg, 72)
    assert a == b
    assert len(repeat['episodes']) == 4


@pytest.mark.parametrize('source', ["__import__('os')", 'state.__class__', '[x for x in hand]', '2 ** 999999'])
def test_generated_rule_cannot_read_files_or_mutate_engine(source):
    with pytest.raises((ValueError, SyntaxError)):
        compile_program(source)


def test_real_two_family_loop_is_resumable_and_hides_validation_records(tmp_path):
    settings = IterationRunConfig(iterations=2, evaluation_rounds=8, training_rounds=2, seconds=60)
    runner = IterationRunner(tmp_path)
    result = runner.run('test', RoundConfig(rounds=2), settings, SearchGenerator())
    assert result['status'] == 'completed'
    state = read_json(tmp_path / 'runs/test/run.json')
    assert [x['family'] for x in state['iterations']] == ['rule', 'rl']
    assert all(x['status'] == 'inconclusive' for x in state['iterations'])
    registry = read_json(tmp_path / 'baselines.json')
    assert registry['rule']['version'] == 'rule-baseline'
    assert registry['rl']['model_path']
    repeated = runner.run('test', RoundConfig(rounds=2), settings, SearchGenerator())
    assert repeated == result
    for entry in state['iterations']:
        summary = entry['feedback']['validation']
        assert 'seed' not in summary and 'pairs' not in summary and 'events' not in summary
        assert summary['actual_rounds_started'] == 8
        assert entry['feedback']['development_examples']


def test_generation_rejects_unknown_parameters():
    p = {'rationale': 'bad', 'rule_expression': None, 'parameters_json': '{"allow_hu_cheat":true}'}
    with pytest.raises(ValueError):validate_proposal(p, PolicySpec())


def test_interrupted_training_resumes_same_weights(tmp_path, monkeypatch):
    import mahjong_ai.training as training
    original = training.MahjongEngine.run_round
    calls = 0
    def interrupted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3: raise KeyboardInterrupt()
        return original(self, *args, **kwargs)
    recipe, cfg = TrainingRecipe(seed=19), RoundConfig(rounds=3)
    with monkeypatch.context() as m:
        m.setattr(training.MahjongEngine, 'run_round', interrupted)
        with pytest.raises(KeyboardInterrupt):
            train(recipe, default_opponents(), cfg, 5, tmp_path / 'resumed', seed=84)
    assert len(read_json(tmp_path / 'resumed/training.json')['episodes']) == 2
    resumed = train(recipe, default_opponents(), cfg, 5, tmp_path / 'resumed', seed=84)
    full = train(recipe, default_opponents(), cfg, 5, tmp_path / 'full', seed=84)
    assert resumed['episodes'] == full['episodes']
    with np.load(resumed['model']) as a, np.load(full['model']) as b:
        for key in a: assert np.array_equal(a[key], b[key])
    damaged = Path(resumed['model'])
    damaged.write_bytes(damaged.read_bytes() + b'damaged')
    with pytest.raises(ValueError, match='content changed'):
        train(recipe, default_opponents(), cfg, 5, tmp_path / 'resumed', seed=84)


def test_interrupted_evaluation_reuses_only_committed_pairs(tmp_path, monkeypatch):
    import mahjong_ai.evaluator as ev
    original = ev._pair_job
    visited = []
    def interrupted(job):
        visited.append(job[0])
        if job[0] == 2: raise KeyboardInterrupt()
        return original(job)
    args = (PolicySpec(), PolicySpec(), default_opponents(), RoundConfig(rounds=2))
    with monkeypatch.context() as m:
        m.setattr(ev, '_pair_job', interrupted)
        with pytest.raises(KeyboardInterrupt):
            Evaluator().evaluate(*args, pairs=5, output=tmp_path)
    assert len(list(tmp_path.glob('pair-*.json'))) == 2
    visited.clear()
    def recorded(job):
        visited.append(job[0])
        return original(job)
    monkeypatch.setattr(ev, '_pair_job', recorded)
    result = Evaluator().evaluate(*args, pairs=5, output=tmp_path)
    assert visited == [2, 3, 4] and len(result.pairs) == 5
    path = tmp_path / 'pair-000000.json'
    value = read_json(path)
    value['delta'] = 10000
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='corrupt cached pair'):
        Evaluator().evaluate(*args, pairs=5, output=tmp_path)


def test_promotion_commit_recovers_without_evaluation_again(tmp_path, monkeypatch):
    # Exercise transaction recovery with real games and a forced decision only
    # for this test; score-based decision behavior is tested separately.
    import mahjong_ai.iteration as it
    atomic = it.atomic_json
    monkeypatch.setattr(it, 'promotion_decision', lambda report: {'status': 'promoted', 'reason': 'test transaction'})
    cfg, settings = RoundConfig(rounds=2), IterationRunConfig(iterations=1, evaluation_rounds=4,
                                     training_rounds=1, families=('rule',), seconds=60)
    def crash(path, data):
        if Path(path).name == 'baselines.json' and data['rule']['version'] != 'rule-baseline':
            raise KeyboardInterrupt()
        atomic(path, data)
    with monkeypatch.context() as m:
        m.setattr(it, 'atomic_json', crash)
        with pytest.raises(KeyboardInterrupt):
            IterationRunner(tmp_path).run('commit', cfg, settings, SearchGenerator())
    assert read_json(tmp_path / 'baselines.json')['rule']['version'] == 'rule-baseline'
    result_path = tmp_path / 'runs/commit/iteration-0000/result.json'
    candidate = read_json(result_path)['candidate']
    monkeypatch.setattr(it.Evaluator, 'evaluate', lambda *a, **kw: pytest.fail('must reuse committed result'))
    result = IterationRunner(tmp_path).run('commit', cfg, settings, SearchGenerator())
    assert result['status'] == 'completed'
    assert read_json(tmp_path / 'baselines.json')['rule'] == candidate


def test_training_failure_is_recorded_and_retains_baseline(tmp_path, monkeypatch):
    import mahjong_ai.iteration as it
    cfg = RoundConfig(rounds=1)
    settings = IterationRunConfig(iterations=1, evaluation_rounds=4, training_rounds=1,
                                  families=('rl',), seconds=60)
    runner = IterationRunner(tmp_path)
    runner._initialize(cfg, settings, None)
    before = read_json(tmp_path / 'baselines.json')
    def failed(*args, **kwargs): raise RuntimeError('injected training failure')
    monkeypatch.setattr(it, 'train', failed)
    result = runner.run('failure', cfg, settings, SearchGenerator())
    assert result['outcomes'] == [{'family': 'rl', 'status': 'failed'}]
    assert read_json(tmp_path / 'baselines.json') == before


def test_generator_timeout_uses_remaining_budget(monkeypatch):
    from mahjong_ai import generation as g
    monkeypatch.setattr(g.time, 'monotonic', lambda: 100)
    assert g._timeout(120, 107) == 7
    with pytest.raises(TimeoutError): g._timeout(120, 99)


def test_loaded_model_must_match_declared_architecture(tmp_path):
    x = train(TrainingRecipe(hidden_size=8), default_opponents(), RoundConfig(rounds=2), 1, tmp_path)
    with pytest.raises(ValueError, match='model shape'):
        RLPolicy(TrainingRecipe(hidden_size=16), x['model'])
    assert RLPolicy(TrainingRecipe(hidden_size=8), x['model']).params['w'].shape[1] == 8


def test_generator_sees_effective_defaults_without_model_path():
    from mahjong_ai.generation import generator_baseline
    value = generator_baseline(PolicySpec('rl', parameters={'seed': 2}))
    assert value['parameters']['entropy'] == .01
    assert value['parameters']['learning_rate'] == .002
    assert value['model_path'] is None


def test_rank_interval_decisions_and_failed_samples_cannot_promote():
    protocol = {'objective': RANK_OBJECTIVE, 'engine': 'test', 'pairs': 30, 'config': {'stage': 'qualifier', 'rounds': 8}}
    for delta, expected in ((1, 'promoted'), (-1, 'rejected'), (0, 'inconclusive')):
        pairs = [{'index': i, 'completed': True, 'delta': delta, 'rounds_started': 2,
                  'baseline': {'score': 100, 'rank': 3, 'rank_utility': 1, 'first': False, 'top_two': False, 'rounds_played':8},
                  'candidate': {'score': -100, 'rank': 3-delta, 'rank_utility': 1+delta, 'first': True, 'top_two': True, 'rounds_played':8}} for i in range(30)]
        report = EvaluationReport(protocol, pairs)
        assert promotion_decision(report)['status'] == expected
        pairs[-1]['completed'] = False
        assert promotion_decision(report)['status'] == 'inconclusive'


def test_behavior_diagnostics_distinguish_noop_from_real_action_changes():
    from mahjong_ai.diagnostics import compare_behavior
    cfg = RoundConfig(rounds=4)
    baseline = PolicySpec()
    same = compare_behavior(baseline, baseline, default_opponents(), cfg, seed=61)
    changed = compare_behavior(baseline, PolicySpec(program='-base'), default_opponents(), cfg, seed=61)
    assert same['decisions'] > 0 and same['different_actions'] == 0
    assert changed['different_actions'] > 0
