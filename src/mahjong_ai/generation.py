from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .policy import PolicySpec, TrainingRecipe
from .storage import digest

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'rationale': {'type': 'string'},
                   'rule_expression': {'type': ['string', 'null']},
                   'parameters_json': {'type': 'string'}},
    'required': ['rationale', 'rule_expression', 'parameters_json'],
}
GUIDANCE = '''Propose ONE small strategy improvement for a Hangzhou Mahjong research bot.
Use only the supplied developer-visible context. Do not inspect files or run tools.
The ONLY improvement objective is terminal four-player relative rank utility: ranks
1/2/3/4 earn 3/2/1/0. Extra raw score at the same officially resolved rank earns no
extra reward. Complete configured multi-game settlements before assessing rank.
Adjust risk using visible ranks, score gaps and variable remaining opportunity;
no fixed-hand switch or unconditional trailing gamble. Elo is a descriptive
strength ranking, never training reward or independent proof of promotion.
For rule family: parameters_json is an object with any changed keys from
risk_scale, call_bonus, kong_bonus, piao_bonus, hu_bonus (numeric, absolute <=50).
You may also propose a numeric Python expression as rule_expression. It replaces
the action score. Available names: base (existing rule score), risk (-1 lead to
+1 behind, scaled by progress), progress (0..1), remaining, score, gap, fan,
rank (current 1..4), ranks (four public ranks), quality, quality_delta, piao,
discard, hu, kong, call, white, hand (34 counts),
scores (four public scores), state (public numeric feature vector). Allowed math:
+ - * / comparisons if/else and/or, abs/min/max/log1p and constant vector indexing.
No statements, imports, attributes, loops, filesystem or engine access.
For RL family: rule_expression MUST be null; parameters_json may change only
learning_rate, reward_scale, entropy, hidden_size, seed, algorithm. Algorithm is
reinforce-value-baseline, hidden_size 4..256, learning_rate (0,.1], reward_scale >0,
entropy [0,1]. Training occurs after your proposal. Do not invent training results.
Developer behavior diagnostics compare your candidate and baseline on identical
public observations. If previous changes produce no different actions, investigate
a different action distinction or parameter magnitude instead of repeating no-op
perturbations. An action difference is diagnostic only, never proof of improvement.
Return exactly the requested JSON with concise rationale; {} is valid parameters.
'''


def validate_proposal(proposal, baseline):
    if set(proposal) != {'rationale', 'rule_expression', 'parameters_json'}:
        raise ValueError('generator output must match the candidate schema')
    if not isinstance(proposal['rationale'], str) or len(proposal['rationale']) > 8000:
        raise ValueError('invalid rationale')
    parameters = json.loads(proposal['parameters_json'])
    if not isinstance(parameters, dict): raise ValueError('parameters_json must encode an object')
    merged = baseline.parameters | parameters
    expression = proposal['rule_expression']
    if expression is not None and not isinstance(expression, str):
        raise ValueError('rule_expression must be string or null')
    version = baseline.family + '-' + digest(proposal)[:12]
    spec = PolicySpec(baseline.family, version, merged,
                      baseline.model_path if baseline.family == 'rl' else None,
                      expression if expression is not None else baseline.program)
    return spec


def generator_baseline(baseline):
    from dataclasses import asdict
    data = baseline.as_dict()
    data['model_path'] = None
    if baseline.family == 'rl':
        data['parameters'] = asdict(TrainingRecipe(**baseline.parameters))
    else:
        data['parameters'] = {'risk_scale': 1., 'call_bonus': .15, 'kong_bonus': .5,
                              'piao_bonus': 0., 'hu_bonus': 12.} | baseline.parameters
    return data


def _execute(argv, prompt, cwd, timeout):
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=cwd, start_new_session=True)
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try: process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    if process.returncode:
        # Do not copy provider logs/credentials into artifacts or the next prompt.
        raise RuntimeError(f'generator command exited {process.returncode}')
    if len(stdout) > 2_000_000: raise ValueError('generator output too large')
    return stdout


def _timeout(configured, deadline):
    remaining = configured if deadline is None else min(configured, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("iteration time budget exhausted")
    return remaining


class CommandGenerator:
    """stdin JSON context -> stdout proposal JSON, no shell interpolation."""
    name = 'command-agent'
    def __init__(self, argv, timeout=120):
        if not argv: raise ValueError('missing generator argv')
        self.argv, self.timeout = list(argv), timeout

    def configuration(self):
        return {'name': self.name, 'argv': self.argv, 'timeout': self.timeout}

    def propose(self, baseline, feedback, iteration, deadline=None):
        context = {'instructions': GUIDANCE, 'baseline': generator_baseline(baseline),
                   'feedback': feedback, 'iteration': iteration, 'schema': SCHEMA}
        # The external generator sees only this context, never evaluation files.
        context['baseline']['model_path'] = None
        with tempfile.TemporaryDirectory(prefix='mahjong-generator-') as cwd:
            result = _execute(self.argv, json.dumps(context, ensure_ascii=False), cwd, _timeout(self.timeout, deadline))
        proposal = json.loads(result)
        validate_proposal(proposal, baseline)
        return proposal


class CodexGenerator:
    name = 'codex-agent'
    def __init__(self, executable='codex', timeout=180, model=None):
        self.executable, self.timeout, self.model = executable, timeout, model

    def configuration(self):
        return {'name': self.name, 'executable': self.executable, 'timeout': self.timeout, 'model': self.model}

    def propose(self, baseline, feedback, iteration, deadline=None):
        context = {'baseline': generator_baseline(baseline), 'feedback': feedback, 'iteration': iteration}
        context['baseline']['model_path'] = None
        with tempfile.TemporaryDirectory(prefix='mahjong-generator-') as directory:
            root = Path(directory)
            (root / 'schema.json').write_text(json.dumps(SCHEMA))
            argv = [self.executable, 'exec', '--ephemeral', '--sandbox', 'read-only',
                    '--skip-git-repo-check', '--cd', directory, '--color', 'never',
                    '--output-schema', str(root / 'schema.json'),
                    '--output-last-message', str(root / 'proposal.json')]
            if self.model: argv += ['--model', self.model]
            argv.append('-')
            _execute(argv, GUIDANCE + '\nCONTEXT\n' + json.dumps(context, ensure_ascii=False), directory, _timeout(self.timeout, deadline))
            proposal = json.loads((root / 'proposal.json').read_text())
        validate_proposal(proposal, baseline)
        return proposal


class SearchGenerator:
    """Offline reproducible search for exercising the same pipeline; NOT an LLM."""
    name = 'offline-search'
    def __init__(self, seed=0): self.seed = seed

    def configuration(self):
        return {'name': self.name, 'seed': self.seed}

    def propose(self, baseline, feedback, iteration, deadline=None):
        rng = random.Random(self.seed + iteration)
        if baseline.family == 'rule':
            key = ('call_bonus', 'risk_scale', 'kong_bonus', 'piao_bonus')[iteration % 4]
            defaults = {'call_bonus': .15, 'risk_scale': 1., 'kong_bonus': .5, 'piao_bonus': 0.}
            parameters = {key: max(-10, min(10, baseline.parameters.get(key, defaults[key]) + rng.choice([-.4, .4])))}
        else:
            parameters = {'learning_rate': min(.1, baseline.parameters.get('learning_rate', .002) * rng.choice([.7, 1.3]))}
        return {'rationale': 'Bounded offline parameter experiment; no claimed improvement before evaluation.',
                'rule_expression': None, 'parameters_json': json.dumps(parameters)}
