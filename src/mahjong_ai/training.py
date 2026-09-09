from __future__ import annotations

import time
from pathlib import Path

from .types import RANK_OBJECTIVE
from .engine import ENGINE_VERSION, MahjongEngine, derive_seed
from .policy import PolicySpec, RLPolicy, TrainingRecipe
from .storage import atomic_json, file_hash, read_json, runtime_fingerprint


def train(recipe, opponents, config, rounds, output, seed=0, initial_model=None, deadline=None):
    if rounds < 1 or len(opponents) != 3:
        raise ValueError('positive training rounds and three opponents required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model = root / 'model.npz'
    state_path = root / 'training.json'
    protocol = {'objective': RANK_OBJECTIVE, 'recipe': recipe.__dict__, 'opponents': [p.identity for p in opponents],
                'config': config.as_dict(), 'rounds': rounds, 'seed': seed,
                'initial_model_hash': file_hash(initial_model) if initial_model else None,
                'engine': ENGINE_VERSION, 'runtime': runtime_fingerprint()}
    state = {'protocol': protocol, 'episodes': [], 'status': 'training'}
    if state_path.exists():
        state = read_json(state_path)
        if state['protocol'] != protocol:
            raise ValueError('training resume protocol differs')
    if state.get('checkpoint') and file_hash(root / state['checkpoint']) != state['checkpoint_hash']:
        raise ValueError('training checkpoint content changed')
    if state['status'] == 'completed':
        if file_hash(model) != state['model_hash']:
            raise ValueError('completed model content changed')
        return state
    # Checkpoint+state are committed via a generation-specific model filename.
    checkpoint = state.get('checkpoint')
    learner = RLPolicy(recipe, root / checkpoint if checkpoint else initial_model, training=True)
    try:
        for i in range(len(state['episodes']), rounds):
            if deadline is not None and time.monotonic() >= deadline:
                break
            players = [p.build() for p in opponents]
            seat = i % 4
            players.insert(seat, learner)
            result = MahjongEngine(record_events=False).run_round(players, config, derive_seed(seed, 'train', i))
            if not result.completed:
                state['status'] = 'failed'
                state['error'] = result.metadata['reason']
                atomic_json(state_path, state)
                raise RuntimeError('training round incomplete')
            entry = {'episode': i, 'seat': seat, 'score': result.score(seat), 'rank': result.rank_by_seat[seat],
                     'rank_utility': result.rank_utility(seat), 'games_completed': len(result.metadata['games']),
                     'rounds_played': result.rounds_played, **learner.last_update}
            if learner.params is None:
                raise RuntimeError('no decisions occurred; cannot create a model')
            new_checkpoint = f'checkpoint-{i + 1:06}.npz'
            learner.save(root / new_checkpoint)
            previous = state.get('checkpoint')
            state['checkpoint'] = new_checkpoint
            state['checkpoint_hash'] = file_hash(root / new_checkpoint)
            state['episodes'].append(entry)
            atomic_json(state_path, state)
            if previous and previous != new_checkpoint:
                (root / previous).unlink(missing_ok=True)
        if len(state['episodes']) == rounds:
            learner.save(model)
            state['status'] = 'completed'
            state['model_hash'] = file_hash(model)
        else:
            state['status'] = 'paused'
        state['model'] = str(model.resolve()) if state['status'] == 'completed' else None
        atomic_json(state_path, state)
    finally:
        learner.trace = []
    return state
