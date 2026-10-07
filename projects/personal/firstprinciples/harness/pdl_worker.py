"""One worker of a paired performance-difference measurement (harness/pdl.py).

Runs under the pool's frozen runtime (PYTHONPATH) and imports nothing from the
harness, whose module puts the repository's source first on sys.path. The
self-play project's walls.py gives the stable wall and the draw luck.

  python pdl_worker.py <job dir>   (task.json in, result.json out)

Games are played as the development set plays them (the settlement loop of
mahjong_ai.domain.settlement for one game: the winner deals next, the dealer
keeps the deal after a draw, observations carry the stage). Version B sits at
the block's focal seat and plays its own games; at each of its decisions where
A, reading the same observation, would act otherwise, the hand is copied and
continued K times from the true state for each of the two actions: the wall's
unseen order reshuffled (the same order for both, a stable wall from there on),
the focal seat continued by A, the others by the pool members. Each branch
reports (my hand score - my draw luck, I deal next - its luck).
"""
import copy
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path


def build(spec):
    if spec["family"] == "model":
        from mahjong_ai.execution.model_seat import ModelPolicy

        # No time pressure here: a slow answer must not turn into a fallback action.
        policy = ModelPolicy(spec["parameters"]["weights"], spec["version"], action_seconds=120.0)
    else:
        from mahjong_ai.execution.candidate_runtime import _inline_module, tree_hash

        package = Path(spec["package_path"])
        if tree_hash(package) != spec["package_hash"]:
            raise RuntimeError(f"{spec['version']}: 冻结的包发生变化")
        policy = _inline_module(package, spec["package_hash"]).build(package / "model")
    if hasattr(policy, "start"):
        policy.start()
    return policy


def act(policy, obs):
    from mahjong_ai.execution.candidate_runtime import ActionUnavailable

    for attempt in range(3):
        try:
            return policy.act(obs)
        except ActionUnavailable as exc:
            # The policies decide from the observation alone, so asking again is the same
            # decision: a model service that timed out once (a busy training host) is retried;
            # one that cannot answer at all voids the measurement.
            if attempt == 2:
                raise RuntimeError(f"策略未给出动作：{getattr(policy, 'last_error', None) or exc}") from exc
            time.sleep(10)


def main(root):
    root = Path(root)
    task = json.loads((root / "task.json").read_text())
    sys.path.append(task["guided"])  # after the frozen runtime
    from walls import train_hand
    from mahjong_ai.domain.engine import HandState, derive_seed
    from mahjong_ai.domain.settlement import score_ranks
    from mahjong_ai.domain.types import RoundConfig

    config = RoundConfig(**task["config"])
    seed, k = task["seed"], task["k"]
    pols = {name: build(spec) for name, spec in task["policies"].items()}

    def observe(hand, legal, stage):
        obs = hand.observation(legal)
        return replace(obs, stage=replace(obs.stage, **stage))

    def branch(snapshot, first, j):
        hand, focal, stage, players, key = snapshot
        h = copy.deepcopy(hand)
        random.Random(f"{seed}:{key}:{j}").shuffle(h.wall)
        train_hand(h, stable=True, luck_seat=focal)
        h.step(first)
        while not h.settled:
            legal = h.legal_actions()
            if len(legal) == 1:
                x = legal[0]
            else:
                x = act(pols["A" if h.turn == focal else players[h.turn]], observe(h, legal, stage))
            h.step(x, legal)
        deals = h.winner == focal or (h.winner is None and h.dealer == focal)
        return h.delta[focal] - h.luck[0], float(deals) - h.luck[1]

    started, snapshots, hands, games = time.time(), [], [], []
    for block in task["blocks"]:
        others = task["lineups"][block % len(task["lineups"])]
        for focal in range(4):
            players = list(others)
            players.insert(focal, "B")
            gseed = derive_seed(seed, "game", block, focal)
            for name in set(players):
                pols[name].reset(derive_seed(gseed, "policy", name))
            pols["A"].reset(derive_seed(gseed, "policy", "A"))
            scores, history, views, dealer = [0] * 4, [], None, gseed % 4
            for hand_no in range(1, config.rounds + 1):
                hand = HandState(config, derive_seed(gseed, "wall", hand_no), dealer, hand_no, scores, history, False,
                                 previous_views=views)
                hand.rounds_total = config.rounds
                stage = dict(round_no=hand_no, stage_scores=tuple(scores), rank_by_seat=tuple(score_ranks(scores)),
                             stage_round_no=hand_no, stage_rounds_total=config.rounds)
                decision = 0
                while not hand.settled:
                    legal = hand.legal_actions()
                    if len(legal) == 1:
                        a = legal[0]
                    else:
                        obs = observe(hand, legal, stage)
                        a = act(pols[players[hand.turn]], obs)
                        if hand.turn == focal:
                            decision += 1
                            b = act(pols["A"], obs)
                            if b != a:
                                key = f"{block}:{focal}:{hand_no}:{decision}"
                                snap = (copy.deepcopy(hand), focal, stage, players, key)
                                vals = [(*branch(snap, b, j), *branch(snap, a, j)) for j in range(k)]
                                kinds = {x.action for x in legal}
                                category = ("hu" if "hu" in kinds else "gang" if "gang" in kinds else "discard") \
                                    if obs.phase == "draw" else ("resp-gang" if "gang" in kinds else "resp")
                                snapshots.append(dict(block=block, focal=focal, hand_no=hand_no, category=category,
                                                      kind=f"{b.action}->{a.action}", whites=obs.hand.count(33),
                                                      dealer=hand.dealer == focal, wall=obs.wall_remaining, values=vals))
                    hand.step(a, legal)
                hands.append((hand.dealer == focal, hand.delta[focal], hand.winner is None))
                for s in range(4):
                    scores[s] += hand.delta[s]
                history, views = hand.events, hand.public_views()
                if hand.winner is not None:
                    dealer = hand.winner
            games.append((block, focal))
    (root / "result.json").write_text(json.dumps(dict(status="completed", snapshots=snapshots, hands=hands, games=games,
                                                      seconds=time.time() - started)))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as exc:
        import traceback

        Path(sys.argv[1], "result.json").write_text(json.dumps(dict(status="failed", error=traceback.format_exc()[-3000:])))
        raise SystemExit(1) from exc
