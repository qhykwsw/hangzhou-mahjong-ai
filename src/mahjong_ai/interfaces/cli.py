from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from mahjong_ai.domain.engine import ENGINE_VERSION, MahjongEngine
from mahjong_ai.strategies.seeds import default_opponents
from mahjong_ai.strategies.common import PolicySpec, TrainingRecipe
from mahjong_ai.persistence.artifacts import atomic_json, read_json
from mahjong_ai.strategies.training import train
from mahjong_ai.domain.types import RoundConfig, StageKind


def spec(value):
    if value == "rule":
        return PolicySpec()
    if value == "rl":
        return PolicySpec("rl", "rl-initialized")
    path = Path(value)
    data = read_json(path)
    if data.get("model_path"):
        model = Path(data["model_path"])
        data["model_path"] = str(
            model if model.is_absolute() else (path.parent / model).resolve()
        )
    for key in ("package_path", "python_path", "runtime_path"):
        if data.get(key) and not Path(data[key]).is_absolute():
            data[key] = str((path.parent / data[key]).resolve())
    return PolicySpec(**data)


def common(parser):
    parser.add_argument("--rounds", type=int, default=8, help="hands per game")
    parser.add_argument(
        "--games-per-round",
        type=int,
        default=1,
        help="games per complete four-player settlement",
    )
    parser.add_argument(
        "--rotate-seats", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stage", choices=list(StageKind), default="qualifier")
    parser.add_argument("--base-score", type=int, default=1)
    parser.add_argument("--you-cai-bi-kao", action="store_true")
    parser.add_argument(
        "--tiebreak", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--max-extra-hands",
        type=int,
        default=64,
        help="execution budget; exhausted results remain INCOMPLETE",
    )
    parser.add_argument("--seed", type=int, default=0)


def config(args):
    return RoundConfig(
        args.rounds,
        StageKind(args.stage),
        args.base_score,
        args.you_cai_bi_kao,
        args.tiebreak,
        args.max_extra_hands,
        games_per_round=args.games_per_round,
        rotate_seats=args.rotate_seats,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Real Hangzhou Mahjong local research and iteration"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    play = sub.add_parser("play", help="run a complete actual Mahjong round")
    common(play)
    play.add_argument("--policies", nargs=4, default=["rule"] * 4)
    play.add_argument("--output", default=".local/mahjong-ai/play.json")
    play.add_argument("--check-invariants", action="store_true")
    training = sub.add_parser("train", help="train a terminal rank-reward RL policy")
    common(training)
    training.add_argument("--training-rounds", type=int, default=24)
    training.add_argument("--learning-rate", type=float, default=0.002)
    training.add_argument("--hidden-size", type=int, default=16)
    training.add_argument("--initial-model")
    training.add_argument("--output", required=True)
    benchmark = sub.add_parser(
        "benchmark", help="measure real rules and RL throughput locally"
    )
    common(benchmark)
    benchmark.add_argument(
        "--rl-policy", help="trained policy.json; default is an initialized network"
    )
    benchmark.add_argument("--samples", type=int, default=20)
    benchmark.add_argument("--output", default=".local/mahjong-ai/benchmark.json")
    export = sub.add_parser("export", help="export a frozen research candidate bundle")
    export.add_argument("--project", default=".")
    export.add_argument("--candidate", required=True)
    export.add_argument("--output", required=True)
    release = sub.add_parser("release-build", help="build an immutable project release")
    release.add_argument("--project", required=True)
    release.add_argument("--version", required=True)
    release.add_argument("--parent")
    register = sub.add_parser("competition-register", help="register a frozen release")
    register.add_argument("bundle")
    register.add_argument("--root", default=".")
    register.add_argument("--resident", action="store_true")
    listed = sub.add_parser("competition-list", help="list registered releases")
    listed.add_argument("--root", default=".")
    entrant = sub.add_parser("competition-status", help="change a release status")
    entrant.add_argument("entrant_id")
    entrant.add_argument("status", choices=("registered", "paused", "revoked"))
    entrant.add_argument("--root", default=".")
    competition = sub.add_parser(
        "competition-run", help="run an independent four-seat frozen-release brawl"
    )
    common(competition)
    competition.add_argument("--policies", nargs=4, required=True)
    competition.add_argument("--games", type=int, default=16)
    competition.add_argument("--output", required=True)
    competition.add_argument("--check-invariants", action="store_true")
    calibration = sub.add_parser(
        "calibrate", help="compare local fan results with stateless official oracle"
    )
    calibration.add_argument("--base-url", default="https://10.240.169.190:18080")
    calibration.add_argument("--insecure", action="store_true")
    calibration.add_argument("--samples", type=int, default=32)
    calibration.add_argument("--output", default=".local/mahjong-ai/calibration.json")
    for name in ("online-probe", "online-play"):
        online = sub.add_parser(
            name,
            help="read-only schema probe"
            if name == "online-probe"
            else "play using a verified snapshot mapping",
        )
        online.add_argument("--base-url", default="https://10.240.169.190:18080")
        online.add_argument("--token-env", default="MAHJONG_TOKEN")
        online.add_argument("--tournament-id")
        online.add_argument("--insecure", action="store_true")
        online.add_argument("--output", required=True)
        if name == "online-play":
            online.add_argument(
                "--mapping",
                help="optional custom mapping; default current official adapter",
            )
            online.add_argument("--policy", required=True)
            online.add_argument("--workers", type=int, default=16)
            online.add_argument("--seconds", type=float, default=300)
            online.add_argument(
                "--join",
                action="store_true",
                help="also register/ready in available stages",
            )
    args = parser.parse_args(argv)
    if args.command == "export":
        from mahjong_ai.platform.export import export_candidate

        print(
            json.dumps(
                export_candidate(args.project, args.candidate, args.output),
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "release-build":
        from mahjong_ai.competition.releases import build_release

        print(
            json.dumps(
                build_release(args.project, args.version, parent=args.parent),
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "competition-register":
        from mahjong_ai.competition import CompetitionRegistry
        from mahjong_ai.competition.releases import verify_release

        manifest = verify_release(args.bundle)
        result = CompetitionRegistry(args.root).register(
            manifest, args.bundle, resident=args.resident
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "competition-list":
        from mahjong_ai.competition import CompetitionRegistry

        print(json.dumps(CompetitionRegistry(args.root).list(), ensure_ascii=False))
        return 0
    if args.command == "competition-status":
        from mahjong_ai.competition import CompetitionRegistry

        print(
            json.dumps(
                CompetitionRegistry(args.root).set_status(args.entrant_id, args.status),
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "competition-run":
        if args.games < 4 or args.games % 4:
            parser.error("competition games must be a positive multiple of four")
        from mahjong_ai.competition.releases import load_policy
        from mahjong_ai.evaluation.schedule import schedule, standings

        specs = {str(i): load_policy(value) for i, value in enumerate(args.policies)}
        assignments = schedule(list(specs), args.games, args.seed)
        receipts = []
        for assignment in assignments:
            players = [specs[p].build() for p in assignment["players"]]
            result = MahjongEngine(check_invariants=args.check_invariants).run_round(
                players, config(args), args.seed + assignment["index"]
            )
            receipts.append(
                {
                    "index": assignment["index"],
                    "players": assignment["players"],
                    "ranks": list(result.rank_by_seat),
                    "completed": result.completed,
                    "scores": list(result.total_score_by_seat),
                    "result": result.as_dict(),
                }
            )
        atomic_json(
            args.output,
            {
                "kind": "independent-competition",
                "policies": args.policies,
                "games": args.games,
                "receipts": receipts,
                "standings": standings(
                    list(specs),
                    [
                        {
                            "players": r["players"],
                            "ranks": r["ranks"],
                            "scores": r["scores"],
                            "hand_deltas": [
                                list(hand["delta"])
                                for hand in (r["result"].get("metadata") or {}).get(
                                    "hands", []
                                )
                            ],
                        }
                        for r in receipts
                        if r["completed"]
                    ],
                ),
            },
        )
        print(
            json.dumps(
                {"output": args.output, "games": len(receipts)}, ensure_ascii=False
            )
        )
        return 0
    if args.command == "calibrate":
        from mahjong_ai.platform.calibration import calibrate

        if args.samples < 2:
            parser.error("at least two calibration samples required")
        result = calibrate(args.base_url, args.output, args.samples, args.insecure)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "records"}, ensure_ascii=False
            )
        )
        return 0 if result["matched"] == result["samples"] else 1
    if args.command.startswith("online-"):
        from mahjong_ai.platform.online import (
            PlatformClient,
            SnapshotAdapter,
            TournamentRunner,
            probe,
        )

        client = PlatformClient(
            args.base_url, os.environ.get(args.token_env), args.insecure
        )
        if args.command == "online-probe":
            result = probe(client, args.output, args.tournament_id)
        else:
            from mahjong_ai.platform.official import OfficialAdapter

            adapter = (
                SnapshotAdapter(read_json(args.mapping))
                if args.mapping
                else OfficialAdapter()
            )
            result = TournamentRunner(
                client, spec(args.policy), adapter, args.output, args.workers, args.join
            ).run(args.tournament_id, args.seconds)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    cfg = config(args)
    if args.command == "play":
        result = MahjongEngine(check_invariants=args.check_invariants).run_round(
            [spec(p).build() for p in args.policies], cfg, args.seed
        )
        atomic_json(args.output, result.as_dict())
        print(
            json.dumps(
                {
                    "completed": result.completed,
                    "scores": result.total_score_by_seat,
                    "hands": result.rounds_played,
                    "ranks": result.rank_by_seat,
                    "output": args.output,
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "train":
        recipe = TrainingRecipe(
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            seed=args.seed,
        )
        state = train(
            recipe,
            default_opponents(),
            cfg,
            args.training_rounds,
            args.output,
            args.seed,
            args.initial_model,
        )
        policy = PolicySpec("rl", "rl-" + str(args.seed), recipe.__dict__, "model.npz")
        atomic_json(Path(args.output) / "policy.json", policy.as_dict())
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "training_rounds": len(state["episodes"]),
                    "gradient_updates": sum(e["updated"] for e in state["episodes"]),
                    "policy": str(Path(args.output) / "policy.json"),
                },
                ensure_ascii=False,
            )
        )
    else:
        if args.samples < 1:
            parser.error("samples must be positive")
        from mahjong_ai.persistence.artifacts import (
            environment_info,
            runtime_fingerprint,
        )

        result = {
            "engine": ENGINE_VERSION,
            "runtime": runtime_fingerprint(),
            "environment": environment_info(),
            "config": cfg.as_dict(),
            "samples": args.samples,
            "timings": {},
            "rl_model": args.rl_policy or "initialized, untrained network",
        }
        for family in ("rule", "rl"):
            begin = time.perf_counter()
            hands, steps = 0, 0
            for i in range(args.samples):
                chosen = (
                    spec(args.rl_policy)
                    if family == "rl" and args.rl_policy
                    else PolicySpec(family)
                )
                players = [chosen.build() for _ in range(4)]
                r = MahjongEngine(record_events=False).run_round(
                    players, cfg, args.seed + i
                )
                if not r.completed:
                    raise RuntimeError("benchmark encountered incomplete round")
                hands += r.rounds_played
                steps += r.metadata["steps"]
            elapsed = time.perf_counter() - begin
            result["timings"][family] = {
                "seconds": elapsed,
                "hands": hands,
                "steps": steps,
                "hands_per_second": hands / elapsed,
                "complete_rounds_per_second": args.samples / elapsed,
            }
        atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
