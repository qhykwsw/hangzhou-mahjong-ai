from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .engine import ENGINE_VERSION, MahjongEngine
from .evaluator import Evaluator, promotion_decision
from .generation import CodexGenerator, CommandGenerator, SearchGenerator
from .iteration import IterationRunner, IterationRunConfig, default_opponents
from .policy import PolicySpec, TrainingRecipe
from .storage import atomic_json, read_json
from .training import train
from .types import RoundConfig, StageKind


def spec(value):
    if value == 'rule': return PolicySpec()
    if value == 'rl': return PolicySpec('rl', 'rl-initialized')
    path = Path(value)
    data = read_json(path)
    if data.get('model_path'):
        model = Path(data['model_path'])
        data['model_path'] = str(model if model.is_absolute() else (path.parent / model).resolve())
    return PolicySpec(**data)


def common(parser):
    parser.add_argument('--rounds', type=int, default=8, help='hands per game')
    parser.add_argument('--games-per-round', type=int, default=1, help='games per complete four-player settlement')
    parser.add_argument('--rotate-seats', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--stage', choices=list(StageKind), default='qualifier')
    parser.add_argument('--base-score', type=int, default=1)
    parser.add_argument('--you-cai-bi-kao', action='store_true')
    parser.add_argument('--tiebreak', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--max-extra-hands', type=int, default=64,
                        help='execution budget; exhausted results remain INCOMPLETE')
    parser.add_argument('--seed', type=int, default=0)


def config(args):
    return RoundConfig(args.rounds, StageKind(args.stage), args.base_score,
                       args.you_cai_bi_kao, args.tiebreak, args.max_extra_hands,
                       games_per_round=args.games_per_round, rotate_seats=args.rotate_seats)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Real Hangzhou Mahjong local research and iteration')
    sub = parser.add_subparsers(dest='command', required=True)
    play = sub.add_parser('play', help='run a complete actual Mahjong round')
    common(play)
    play.add_argument('--policies', nargs=4, default=['rule'] * 4)
    play.add_argument('--output', default='.local/mahjong-ai/play.json')
    play.add_argument('--check-invariants', action='store_true')
    evaluate = sub.add_parser('evaluate', help='paired full-round evaluation, old and new vs three opponents')
    common(evaluate)
    evaluate.add_argument('--baseline', default='rule')
    evaluate.add_argument('--candidate', required=True)
    evaluate.add_argument('--evaluation-rounds', type=int, default=2000, help='total complete settlements, split equally old/new')
    evaluate.add_argument('--workers', type=int, default=1)
    evaluate.add_argument('--output', required=True)
    league = sub.add_parser('league', help='local frozen-policy league and Elo table; no training or promotion')
    common(league)
    league.add_argument('--policies', nargs=4, required=True)
    league.add_argument('--samples', type=int, default=100)
    league.add_argument('--output', required=True)
    training = sub.add_parser('train', help='train a terminal rank-reward RL policy')
    common(training)
    training.add_argument('--training-rounds', type=int, default=24)
    training.add_argument('--learning-rate', type=float, default=.002)
    training.add_argument('--hidden-size', type=int, default=16)
    training.add_argument('--initial-model')
    training.add_argument('--output', required=True)
    iteration = sub.add_parser('iterate', help='generate, train, evaluate, promote and resume')
    common(iteration)
    iteration.add_argument('--root', default='.local/mahjong-rank')
    iteration.add_argument('--run-id', required=True)
    iteration.add_argument('--iterations', type=int, default=2)
    iteration.add_argument('--evaluation-rounds', type=int, default=2000, help='total complete settlements, split equally old/new')
    iteration.add_argument('--training-rounds', type=int, default=12)
    iteration.add_argument('--seconds', type=float, default=300)
    iteration.add_argument('--workers', type=int, default=1)
    iteration.add_argument('--families', nargs='+', choices=['rule', 'rl'], default=['rule', 'rl'])
    iteration.add_argument('--generator', choices=['search', 'codex', 'command'], default='codex')
    iteration.add_argument('--generator-argv', help='JSON array of executable and arguments, no shell expansion')
    iteration.add_argument('--generator-timeout', type=float, default=180)
    iteration.add_argument('--model', help='optional generator model; otherwise preserve configured model')
    benchmark = sub.add_parser('benchmark', help='measure real rules and RL throughput locally')
    common(benchmark)
    benchmark.add_argument('--rl-policy', help='trained policy.json; default is an initialized network')
    benchmark.add_argument('--samples', type=int, default=20)
    benchmark.add_argument('--output', default='.local/mahjong-ai/benchmark.json')
    export = sub.add_parser('export', help='export a portable local baseline policy bundle')
    export.add_argument('--root', required=True)
    export.add_argument('--family', choices=['rule', 'rl'], required=True)
    export.add_argument('--output', required=True)
    calibration = sub.add_parser('calibrate', help='compare local fan results with stateless official oracle')
    calibration.add_argument('--base-url', default='https://10.240.169.190:18080')
    calibration.add_argument('--insecure', action='store_true')
    calibration.add_argument('--samples', type=int, default=32)
    calibration.add_argument('--output', default='.local/mahjong-ai/calibration.json')
    for name in ('online-probe', 'online-play'):
        online = sub.add_parser(name, help='read-only schema probe' if name == 'online-probe' else 'play using a verified snapshot mapping')
        online.add_argument('--base-url', default='https://10.240.169.190:18080')
        online.add_argument('--token-env', default='MAHJONG_TOKEN')
        online.add_argument('--tournament-id')
        online.add_argument('--insecure', action='store_true')
        online.add_argument('--output', required=True)
        if name == 'online-play':
            online.add_argument('--mapping', help='optional custom mapping; default current official adapter')
            online.add_argument('--policy', required=True)
            online.add_argument('--workers', type=int, default=16)
            online.add_argument('--seconds', type=float, default=300)
            online.add_argument('--join', action='store_true', help='also register/ready in available stages')
    args = parser.parse_args(argv)
    if args.command == 'export':
        from .export import export_baseline
        print(json.dumps(export_baseline(args.root, args.family, args.output), ensure_ascii=False))
        return 0
    if args.command == 'calibrate':
        from .calibration import calibrate
        if args.samples < 2: parser.error('at least two calibration samples required')
        result = calibrate(args.base_url, args.output, args.samples, args.insecure)
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False))
        return 0 if result['matched'] == result['samples'] else 1
    if args.command.startswith('online-'):
        from .online import PlatformClient, SnapshotAdapter, TournamentRunner, probe
        client = PlatformClient(args.base_url, os.environ.get(args.token_env), args.insecure)
        if args.command == 'online-probe':
            result = probe(client, args.output, args.tournament_id)
        else:
            from .official import OfficialAdapter
            adapter = SnapshotAdapter(read_json(args.mapping)) if args.mapping else OfficialAdapter()
            result = TournamentRunner(client, spec(args.policy), adapter,
                                      args.output, args.workers, args.join).run(args.tournament_id, args.seconds)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    cfg = config(args)
    if args.command == 'league':
        from .league import run_league
        print(json.dumps(run_league([spec(p) for p in args.policies],cfg,args.samples,args.seed,args.output),ensure_ascii=False))
    elif args.command == 'play':
        result = MahjongEngine(check_invariants=args.check_invariants).run_round(
            [spec(p).build() for p in args.policies], cfg, args.seed)
        atomic_json(args.output, result.as_dict())
        print(json.dumps({'completed': result.completed, 'scores': result.total_score_by_seat,
                          'hands': result.rounds_played, 'ranks': result.rank_by_seat, 'output': args.output}, ensure_ascii=False))
    elif args.command == 'evaluate':
        if args.evaluation_rounds < 2 or args.evaluation_rounds % 2:
            parser.error('evaluation-rounds must be positive and even')
        report = Evaluator().evaluate(spec(args.baseline), spec(args.candidate), default_opponents(), cfg,
                                     args.evaluation_rounds // 2, args.seed, args.output, args.workers)
        print(json.dumps(report.summary() | {'decision': promotion_decision(report)}, ensure_ascii=False))
    elif args.command == 'train':
        recipe = TrainingRecipe(learning_rate=args.learning_rate, hidden_size=args.hidden_size, seed=args.seed)
        state = train(recipe, default_opponents(), cfg, args.training_rounds, args.output,
                      args.seed, args.initial_model)
        policy = PolicySpec('rl', 'rl-' + str(args.seed), recipe.__dict__, 'model.npz')
        atomic_json(Path(args.output) / 'policy.json', policy.as_dict())
        print(json.dumps({'status': state['status'], 'training_rounds': len(state['episodes']),
                          'gradient_updates': sum(e['updated'] for e in state['episodes']),
                          'policy': str(Path(args.output) / 'policy.json')}, ensure_ascii=False))
    elif args.command == 'iterate':
        if args.generator == 'codex':
            generator = CodexGenerator(timeout=args.generator_timeout, model=args.model)
        elif args.generator == 'command':
            if not args.generator_argv: parser.error('--generator-argv required')
            generator = CommandGenerator(json.loads(args.generator_argv), args.generator_timeout)
        else:
            generator = SearchGenerator(args.seed)
        settings = IterationRunConfig(args.iterations, args.evaluation_rounds, args.training_rounds,
                                      args.workers, args.seconds, args.seed, tuple(args.families))
        print(json.dumps(IterationRunner(args.root).run(args.run_id, cfg, settings, generator), ensure_ascii=False))
    else:
        if args.samples < 1: parser.error('samples must be positive')
        from .storage import environment_info, runtime_fingerprint
        result = {'engine': ENGINE_VERSION, 'runtime': runtime_fingerprint(), 'environment': environment_info(),
                  'config': cfg.as_dict(), 'samples': args.samples, 'timings': {},
                  'rl_model': args.rl_policy or 'initialized, untrained network'}
        for family in ('rule', 'rl'):
            begin = time.perf_counter()
            hands, steps = 0, 0
            for i in range(args.samples):
                chosen = spec(args.rl_policy) if family == 'rl' and args.rl_policy else PolicySpec(family)
                players = [chosen.build() for _ in range(4)]
                r = MahjongEngine(record_events=False).run_round(players, cfg, args.seed + i)
                if not r.completed: raise RuntimeError('benchmark encountered incomplete round')
                hands += r.rounds_played
                steps += r.metadata['steps']
            elapsed = time.perf_counter() - begin
            result['timings'][family] = {'seconds': elapsed, 'hands': hands, 'steps': steps,
                                         'hands_per_second': hands / elapsed,
                                         'complete_rounds_per_second': args.samples / elapsed}
        atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': raise SystemExit(main())
