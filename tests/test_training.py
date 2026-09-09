import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from mahjong_ai.engine import MahjongEngine
from mahjong_ai.seeds import default_opponents
from mahjong_ai.policy import PolicySpec, RLPolicy, RulePolicy, TrainingRecipe
from mahjong_ai.program import compile_program
from mahjong_ai.storage import read_json
from mahjong_ai.training import train
from mahjong_ai.types import RANK_OBJECTIVE, RoundConfig


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


def test_loaded_model_must_match_declared_architecture(tmp_path):
    x = train(TrainingRecipe(hidden_size=8), default_opponents(), RoundConfig(rounds=2), 1, tmp_path)
    with pytest.raises(ValueError, match='model shape'):
        RLPolicy(TrainingRecipe(hidden_size=16), x['model'])
    assert RLPolicy(TrainingRecipe(hidden_size=8), x['model']).params['w'].shape[1] == 8
