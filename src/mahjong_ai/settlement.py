"""A complete four-participant settlement; games may rotate physical seats."""
from dataclasses import asdict, replace

from .types import RANK_OBJECTIVE, RoundResult, StageKind


def score_ranks(scores):
    """Midranks for score ties; used for official per-game place points."""
    return tuple(1 + sum(v > x for v in scores) + (sum(v == x for v in scores) - 1) / 2 for x in scores)


def final_ranks(scores, place_points, whites, player_ids, stage):
    if stage == StageKind.FINAL:
        return score_ranks(scores)
    order = sorted(range(4), key=lambda i: (-scores[i], -place_points[i], -whites[i], player_ids[i]))
    return tuple(order.index(i) + 1 for i in range(4))


def run_settlement(engine, policies, config, seed):
    from .engine import HandState, derive_seed
    if len(policies) != 4 or len({id(p) for p in policies}) != 4:
        raise ValueError('exactly four independent policy instances required')
    for player, policy in enumerate(policies): policy.reset(derive_seed(seed, 'policy', player))
    scores, whites, points = [0]*4, [0]*4, [0.]*4
    hands, games, archived_events = [], [], []
    hand_no = steps = overtime = game_index = 0
    completed, reason = True, ''
    horizon = config.games_per_round * config.rounds
    while True:
        if game_index >= config.games_per_round:
            if not config.allow_tiebreak or len(set(scores)) == 4: break
            if config.max_tiebreak_rounds is not None and overtime >= config.max_tiebreak_rounds:
                completed, reason = False, 'tiebreak execution budget exhausted'
                break
            horizon += config.rounds
        # Stable player identity is retained in policy instances and the ledger.
        # Public hand history is scoped to the current game/seat mapping.
        seat_players = tuple((s + game_index) % 4 for s in range(4)) if config.rotate_seats else tuple(range(4))
        game_scores, game_whites, history = [0]*4, [0]*4, []
        dealer = seed % 4 if game_index == 0 else derive_seed(seed, 'dealer', game_index) % 4
        game_hands = 0
        for local_hand in range(1, config.rounds + 1):
            extra = game_index >= config.games_per_round
            if extra:
                if config.max_tiebreak_rounds is not None and overtime >= config.max_tiebreak_rounds:
                    completed, reason = False, 'tiebreak execution budget exhausted'
                    break
                overtime += 1
            hand_no += 1
            hand = HandState(config, derive_seed(seed, 'wall', hand_no), dealer, hand_no,
                             game_scores, history, extra)
            hand.rounds_total = config.rounds
            stage_scores = tuple(scores[i] for i in seat_players)
            # Live observations may not include rivals' acquired white counts.
            # Keep score ties tied; only the terminal settlement uses all official
            # tie-break fields. Never leak hidden tile counts through live rank.
            stage_ranks = tuple(score_ranks(scores)[i] for i in seat_players)
            while not hand.settled:
                if hand.steps >= config.max_actions_per_hand:
                    completed, reason = False, 'hand action budget exhausted'
                    break
                legal = hand.legal_actions()
                if len(legal) == 1: action = legal[0]
                else:
                    obs = hand.observation()
                    obs = replace(obs, stage=replace(obs.stage, round_no=local_hand,
                                  stage_scores=stage_scores, rank_by_seat=stage_ranks,
                                  stage_round_no=hand_no, stage_rounds_total=horizon))
                    action = policies[seat_players[hand.turn]].act(obs)
                hand.step(action)
                if engine.check_invariants: hand.assert_conservation()
            steps += hand.steps
            if not completed: break
            for seat, player in enumerate(seat_players):
                scores[player] += hand.delta[seat]
                whites[player] += hand.white_acquired[seat]
                game_scores[seat] += hand.delta[seat]
                game_whites[seat] += hand.white_acquired[seat]
            history = hand.events
            game_hands += 1
            hands.append({'hand_no': hand_no, 'game_no': game_index+1, 'game_hand_no': local_hand,
                          'seat_players': seat_players, 'dealer': dealer, 'winner': hand.winner,
                          'fan': hand.fan.fan if hand.fan else 0,
                          'detail': list(hand.fan.detail) if hand.fan else [],
                          'delta': hand.delta, 'scores': tuple(scores),
                          'game_scores': tuple(game_scores),
                          'initial_hands': hand.initial_hands if engine.record_events else None})
            if hand.winner is not None: dealer = hand.winner
        if engine.record_events:
            archived_events.extend(asdict(e) | {'private_to': private, 'game_no': game_index+1,
                                               'seat_players': seat_players} for e, private in history)
        if game_hands == config.rounds:
            ranks = score_ranks(game_scores)
            for seat, player in enumerate(seat_players): points[player] += 5 - 2*ranks[seat]
        games.append({'game_no': game_index+1, 'seat_players': seat_players,
                      'scores_by_seat': tuple(game_scores), 'white_by_seat': tuple(game_whites),
                      'stage_scores_by_player': tuple(scores), 'hands_completed': game_hands,
                      'completed': game_hands == config.rounds, 'in_tiebreak': game_index >= config.games_per_round})
        game_index += 1
        if not completed: break
    ranks = final_ranks(scores, points, whites, config.player_ids, config.stage) if completed else ()
    first = next((i for i in range(4) if ranks[i] == 1), None) if completed else None
    result = RoundResult(engine.name, seed, config.stage, hand_no, tuple(scores), first,
                         tuple(i for i in range(4) if ranks[i] <= 2) if completed else (), completed,
                         {'reason': reason, 'steps': steps, 'hands': hands, 'games': games,
                          'events': archived_events, 'white_acquired': whites, 'place_points': points,
                          'player_ids': config.player_ids, 'objective': RANK_OBJECTIVE,
                          'result_index': 'stable participant slot',
                          'calibration': 'four-player local settlement; live full-stage parity pending',
                          'config': config.as_dict()}, ranks)
    # One terminal update for the whole settlement, never one update per game.
    for player, policy in enumerate(policies): policy.finish_round(result, player)
    return result
