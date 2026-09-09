from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .engine import derive_seed
from .diagnostics import compare_behavior
from .evaluator import Evaluator, promotion_decision
from .generation import SearchGenerator, validate_proposal
from .policy import PolicySpec, TrainingRecipe
from .storage import atomic_json, digest, exclusive_lock, read_json, runtime_fingerprint, environment_info
from .training import train
from .types import RANK_OBJECTIVE, RoundConfig


@dataclass(frozen=True)
class IterationRunConfig:
    iterations: int = 2
    evaluation_rounds: int = 2000  # 1000 paired complete settlements, old/new 1000 each
    training_rounds: int = 12
    workers: int = 1
    seconds: float = 300
    seed: int = 0
    families: tuple[str, ...] = ('rule', 'rl')
    max_generator_retries: int = 1

    def __post_init__(self):
        if self.iterations < 1 or self.training_rounds < 1 or self.seconds <= 0:
            raise ValueError('invalid iteration budget')
        if self.evaluation_rounds < 2 or self.evaluation_rounds % 2:
            raise ValueError('evaluation_rounds must be a positive even number (old + new)')
        if not self.families or set(self.families) - {'rule', 'rl'}:
            raise ValueError('unsupported family schedule')
        if self.workers < 1 or not 0 <= self.max_generator_retries <= 5:
            raise ValueError('invalid worker/retry budget')


def default_opponents():
    return (PolicySpec('rule', 'rule-balanced'),
            PolicySpec('rule', 'rule-conservative', {'risk_scale': .3, 'call_bonus': -.3}),
            PolicySpec('rule', 'rule-attacking', {'risk_scale': 2., 'piao_bonus': 1.}))


class IterationRunner:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.registry_path = self.root / 'baselines.json'

    def _initialize(self, config, settings, deadline):
        if self.registry_path.exists():
            if read_json(self.registry_path).get('objective') != RANK_OBJECTIVE:
                raise ValueError('legacy baseline registry; use a new rank experiment root')
            return
        registry = {'version': 2, 'objective': RANK_OBJECTIVE, 'rule': PolicySpec().as_dict()}
        if 'rl' in settings.families:
            state = train(TrainingRecipe(seed=settings.seed), default_opponents(), config,
                          settings.training_rounds, self.root / 'initial-rl', settings.seed,
                          deadline=deadline)
            if state['status'] != 'completed':
                return
            registry['rl'] = PolicySpec('rl', 'rl-baseline', asdict(TrainingRecipe(seed=settings.seed)), state['model']).as_dict()
        atomic_json(self.registry_path, registry)

    def run(self, run_id, config=RoundConfig(), settings=IterationRunConfig(), generator=None):
        if not run_id or not all(c.isalnum() or c in '-_' for c in run_id):
            raise ValueError('run_id must contain only letters, digits, hyphens, underscores')
        generator = generator or SearchGenerator(settings.seed)
        root = self.root / 'runs' / run_id
        root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + settings.seconds
        with exclusive_lock(self.root / '.writer.lock'):
            self._initialize(config, settings, deadline)
            if not self.registry_path.exists():
                return {'status': 'paused', 'reason': 'initial RL training checkpointed'}
            registry = read_json(self.registry_path)
            if 'rl' in settings.families and 'rl' not in registry:
                state = train(TrainingRecipe(seed=settings.seed), default_opponents(), config,
                              settings.training_rounds, self.root / 'initial-rl', settings.seed, deadline=deadline)
                if state['status'] != 'completed': return {'status': 'paused', 'reason': 'initial RL training checkpointed'}
                registry['rl'] = PolicySpec('rl', 'rl-baseline', asdict(TrainingRecipe(seed=settings.seed)), state['model']).as_dict()
                atomic_json(self.registry_path, registry)
            protocol = {'objective': RANK_OBJECTIVE, 'config': config.as_dict(), 'settings': asdict(settings), 'generator': generator.configuration(), 'runtime': runtime_fingerprint()}
            protocol['settings']['families'] = list(settings.families)
            protocol['settings'].pop('seconds')  # resuming grants a new time budget, same experiment
            state_path = root / 'run.json'
            if state_path.exists():
                state = read_json(state_path)
                if state['protocol'] != protocol:
                    raise ValueError('run resume configuration changed; select another run_id')
            else:
                opponents = [PolicySpec(**registry['rule']), default_opponents()[1],
                             PolicySpec(**registry['rl']) if 'rl' in registry else default_opponents()[2]]
                prior_feedback = []
                for previous_run in sorted(self.root.glob('runs/*/run.json'), key=lambda p: p.stat().st_mtime)[-3:]:
                    prior_feedback.extend(i['feedback'] for i in read_json(previous_run).get('iterations', [])[-2:])
                state = {'prior_feedback': prior_feedback[-4:], 'protocol': protocol, 'opponents': [p.as_dict() for p in opponents],
                         'environment': environment_info(), 'opponent_ids': [p.identity for p in opponents], 'iterations': [], 'status': 'running'}
                atomic_json(state_path, state)
                import shutil
                source_dir = root / 'source'
                source_dir.mkdir(exist_ok=True)
                for source in Path(__file__).parent.glob('*.py'):
                    shutil.copy2(source, source_dir / source.name)
            opponents = tuple(PolicySpec(**p) for p in state['opponents'])
            if [p.identity for p in opponents] != state['opponent_ids']:
                raise ValueError('frozen opponent artifact changed')
            for index in range(len(state['iterations']), settings.iterations):
                if time.monotonic() >= deadline: break
                directory = root / f'iteration-{index:04}'
                directory.mkdir(exist_ok=True)
                result_path = directory / 'result.json'
                family = settings.families[index % len(settings.families)]
                current = PolicySpec(**read_json(self.registry_path)[family])
                parent_path = directory / 'parent.json'
                if parent_path.exists():
                    baseline = PolicySpec(**read_json(parent_path))
                else:
                    baseline = current
                    atomic_json(parent_path, baseline.as_dict())
                if result_path.exists():
                    outcome = read_json(result_path)
                else:
                    proposal_path = directory / 'proposal.json'
                    if proposal_path.exists():
                        proposal = read_json(proposal_path)
                    else:
                        feedback = (state.get('prior_feedback', []) + [entry['feedback'] for entry in state['iterations']])[-4:]
                        failure = None
                        for attempt in range(settings.max_generator_retries + 1):
                            if time.monotonic() >= deadline: break
                            try:
                                proposal = generator.propose(baseline, feedback, index, deadline=deadline)
                                validate_proposal(proposal, baseline)
                                atomic_json(proposal_path, proposal)
                                failure = None
                                break
                            except Exception as exc:
                                failure = f'{type(exc).__name__}: {exc}'
                                feedback = feedback + [{'proposal_error': failure}]
                                atomic_json(directory / f'generator-failure-{attempt}.json', {'error': failure})
                        if not proposal_path.exists():
                            if failure is None or time.monotonic() >= deadline: break
                            outcome = {'index': index, 'family': family, 'status': 'failed',
                                       'reason': 'generator did not produce a valid candidate',
                                       'feedback': {'family': family, 'status': 'failed', 'reason': failure}}
                            atomic_json(result_path, outcome)
                            state['iterations'].append(outcome)
                            atomic_json(state_path, state)
                            continue
                    try:
                        training_feedback = None
                        candidate = validate_proposal(proposal, baseline)
                        if family == 'rl':
                            recipe = TrainingRecipe(**candidate.parameters)
                            # Architecture changes train from scratch; other changes warm start.
                            initial = baseline.model_path
                            if recipe.hidden_size != TrainingRecipe(**baseline.parameters).hidden_size:
                                initial = None
                            training = train(recipe, opponents, config, settings.training_rounds,
                                             directory / 'training', derive_seed(settings.seed, index, 'train'),
                                             initial_model=initial, deadline=deadline)
                            if training['status'] != 'completed': break
                            candidate = replace(candidate, model_path=training['model'])
                            training_feedback = {'recipe': asdict(recipe), 'completed_rounds': len(training['episodes']),
                                                 'updates': sum(e['updated'] for e in training['episodes']),
                                                 'decisions': sum(e['decisions'] for e in training['episodes']),
                                                 'mean_rank_utility': sum(e['rank_utility'] for e in training['episodes']) / len(training['episodes']),
                                                 'mean_rank': sum(e['rank'] for e in training['episodes']) / len(training['episodes']),
                                                 'mean_round_score': sum(e['score'] for e in training['episodes']) / len(training['episodes']),
                                                 'recipe_evidence': 'single seed; reusable recipe improvement unconfirmed'}
                        atomic_json(directory / 'candidate.json', candidate.as_dict())
                        developer = Evaluator().evaluate(
                            baseline, candidate, opponents, config, pairs=2,
                            seed=derive_seed(settings.seed, index, 'development'),
                            output=directory / 'development', workers=settings.workers,
                            record_events=True, deadline=deadline)
                        if len(developer.pairs) != 2: break
                        # Separate seed domain and artifact subtree; generator never gets raw validation records.
                        report = Evaluator().evaluate(
                            baseline, candidate, opponents, config, pairs=settings.evaluation_rounds // 2,
                            seed=derive_seed(settings.seed, run_id, index, 'validation'),
                            output=directory / 'private-validation', workers=settings.workers,
                            alpha=max(.0001, .05 / settings.iterations), deadline=deadline)
                        if len(report.pairs) != settings.evaluation_rounds // 2: break
                        behavior = compare_behavior(baseline, candidate, opponents, config, seed=derive_seed(settings.seed, index, 'development-behavior'))
                        decision = promotion_decision(report)
                        # Elo consumes only public developer games, never hidden validation records.
                        from .ratings import EloLedger
                        context = {'engine': report.protocol['engine'], 'config': config.as_dict()}
                        ledger = EloLedger(self.root / 'elo' / (digest(context) + '.json'), context)
                        for pair in developer.completed:
                            for side in ('baseline','candidate'):
                                ledger.record(digest([developer.protocol, pair['index'], side]),
                                              pair[side]['policy_ids'], pair[side]['rank_by_player'],
                                              completed=True, source='research-development')
                        elo_table = ledger.table()
                        examples = []
                        if developer.completed:
                            worst = min(developer.completed, key=lambda p: p['delta'])
                            result = worst['candidate']['result']
                            hands = result['metadata']['hands']
                            if hands:
                                hand = min(hands, key=lambda h: h['delta'][h['seat_players'].index(worst['seat'])])
                                examples = [{'tested_seat': worst['seat'], 'hand': hand,
                                             'events': [e for e in result['metadata']['events']
                                                        if e['hand_no'] == hand['hand_no']]}]
                        outcome = {'index': index, 'family': family, **decision,
                                   'baseline': baseline.as_dict(), 'candidate': candidate.as_dict(),
                                   'baseline_id': baseline.identity, 'candidate_id': candidate.identity,
                                   'feedback': {'family': family, 'candidate': candidate.version,
                                                'rationale': proposal['rationale'], **decision,
                                                'training': training_feedback,
                                                'behavior': behavior,
                                                'elo': {'source':'development only', 'used_for_promotion':False, 'table':elo_table},
                                                'development': developer.summary(),
                                                'validation': report.summary(),
                                                'development_examples': examples}}
                        atomic_json(result_path, outcome)
                    except Exception as exc:
                        outcome = {'index': index, 'family': family, 'status': 'failed',
                                   'reason': 'candidate pipeline failed; baseline retained',
                                   'feedback': {'family': family, 'status': 'failed',
                                                'reason': f'{type(exc).__name__}: {exc}'}}
                        atomic_json(result_path, outcome)
                if outcome['status'] == 'promoted':
                    registry = read_json(self.registry_path)
                    registered = PolicySpec(**registry[family])
                    if registered.identity not in (outcome['baseline_id'], outcome['candidate_id']):
                        raise RuntimeError('baseline changed outside this run; refusing to overwrite')
                    registry[family] = outcome['candidate']
                    atomic_json(self.registry_path, registry)
                state['iterations'].append(outcome)
                atomic_json(state_path, state)
            state['status'] = 'completed' if len(state['iterations']) == settings.iterations else 'paused'
            atomic_json(state_path, state)
            return {'status': state['status'], 'iterations': len(state['iterations']),
                    'outcomes': [{'family': i['family'], 'status': i['status']} for i in state['iterations']],
                    'root': str(root)}
