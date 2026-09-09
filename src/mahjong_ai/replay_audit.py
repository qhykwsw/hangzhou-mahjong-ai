"""Check official completed replay blocks against local tile and score rules.

Input uses the public portal replay contract. Replays are post-game evidence and
must never enter a live player's Observation as hidden information.
"""
from __future__ import annotations

from collections import Counter

from .rules import WHITE, calculate_fan, is_baotou, payments, tile_id


def audit_replay(replay, base_score=1, you_cai_bi_kao=False):
    groups = {}
    for block in replay['blocks']:
        groups.setdefault(block['round_no'], []).append(block)
    checks, outcomes, missing, event_types = [], [], [], Counter()
    def check(condition, message, hand, seq=None):
        checks.append({'ok': bool(condition), 'check': message, 'round_no': hand, 'seq': seq})
    previous_dealer = previous_winner = None
    previous_was_draw = None
    for number in sorted(groups):
        blocks = sorted(groups[number], key=lambda b: b['seq_start'])
        events = sorted([e for b in blocks for e in b.get('events', [])], key=lambda e: e['seq'])
        if not blocks[0].get('start_hands'):
            missing.append({'round_no': number, 'reason': 'missing initial hands'})
            continue
        hands = [[tile_id(t) for t in h] for h in blocks[0]['start_hands']]
        if len(hands) != 4: raise ValueError('replay must have four initial hands')
        initial_dealers = [s for s, h in enumerate(hands) if len(h) == 14]
        if len(initial_dealers) != 1:
            missing.append({'round_no': number, 'reason': 'cannot identify 14-tile initial dealer'})
            continue
        dealer = initial_dealers[0]
        if previous_dealer is not None:
            expected = previous_dealer if previous_was_draw else previous_winner
            check(dealer == expected, 'dealer transition matches local convention', number)
        melds, rivers = [[] for _ in range(4)], [[] for _ in range(4)]
        draws = [None] * 4
        draws[dealer] = hands[dealer][-1]
        chains, piaos, kong_draw = [0] * 4, [0] * 4, [False] * 4
        catch_owner = None
        last_discarder = None
        ended = False
        def remove(seat, tile, amount, seq):
            check(hands[seat].count(tile) >= amount, 'consumed tile exists in hand', number, seq)
            for _ in range(amount):
                if tile in hands[seat]: hands[seat].remove(tile)
        for i, event in enumerate(events):
            seq, kind, seat = event['seq'], event['type'], event.get('seat')
            data = event.get('data') or {}
            event_types[kind] += 1
            if i: check(seq == events[i - 1]['seq'] + 1, 'event sequence contiguous', number, seq)
            tile = tile_id(event['tile']) if event.get('tile') else None
            if kind == 'tile_drawn':
                if catch_owner == seat and not kong_draw[seat]:
                    catch_owner = None
                hands[seat].append(tile)
                draws[seat] = tile
            elif kind == 'tile_discarded':
                if catch_owner is not None and catch_owner != seat:
                    check(tile == draws[seat], 'catch-play player discards drawn tile', number, seq)
                waiting = list(hands[seat])
                if draws[seat] in waiting: waiting.remove(draws[seat])
                before_bt = draws[seat] is not None and is_baotou(waiting, len(melds[seat]))
                remove(seat, tile, 1, seq)
                called_in_circle = catch_owner == seat and draws[seat] is None
                piao = tile == WHITE and (before_bt or called_in_circle) and is_baotou(hands[seat], len(melds[seat]))
                chains[seat] = chains[seat] + 1 if piao else 0
                piaos[seat] = piaos[seat] + 1 if piao else 0
                rivers[seat].append(tile)
                last_discarder = seat
                draws[seat], kong_draw[seat] = None, False
                if tile == WHITE:
                    catch_owner = seat
                elif catch_owner == seat:
                    catch_owner = None
            elif kind in ('chi', 'peng', 'gang'):
                gang_kind = data.get('kind') if kind == 'gang' else None
                if kind != 'gang' or gang_kind != 'an':
                    check(catch_owner is None or catch_owner == seat,
                          'catch-play claims require exempt player', number, seq)
                if kind == 'chi':
                    tiles = [tile_id(t) for t in data['tiles']]
                    consumed = list(tiles)
                    consumed.remove(tile)
                    for t in consumed: remove(seat, t, 1, seq)
                    check(sum(m['kind'] == 'chi' for m in melds[seat]) < 2, 'at most two chi', number, seq)
                    melds[seat].append({'kind': 'chi', 'tiles': tiles})
                elif kind == 'peng':
                    remove(seat, tile, 2, seq)
                    melds[seat].append({'kind': 'peng', 'tiles': [tile] * 3})
                elif gang_kind in ('an', 'ming'):
                    remove(seat, tile, 4 if gang_kind == 'an' else 3, seq)
                    melds[seat].append({'kind': 'gang', 'tiles': [tile] * 4})
                elif gang_kind == 'bu':
                    remove(seat, tile, 1, seq)
                    old = next((m for m in melds[seat] if m['kind'] == 'peng' and m['tiles'][0] == tile), None)
                    check(old is not None, 'added kong expands an existing peng', number, seq)
                    if old: old.update(kind='gang', tiles=[tile] * 4)
                else:
                    raise ValueError('unknown official gang kind; inspect replay contract')
                if kind in ('chi', 'peng') or gang_kind == 'ming':
                    check(last_discarder is not None and tile in rivers[last_discarder], 'claim consumes river tile', number, seq)
                    if last_discarder is not None and tile in rivers[last_discarder]: rivers[last_discarder].remove(tile)
                if kind == 'gang':
                    chains[seat] += 1
                    kong_draw[seat] = True
                else:
                    if catch_owner != seat:
                        chains[seat] = piaos[seat] = 0
                    kong_draw[seat] = False
                draws[seat] = None
                last_discarder = None
            elif kind == 'round_ended':
                ended = True
                draw = data['draw']
                if 'dealer' in data: check(data['dealer'] == dealer, 'event dealer matches initial hand', number, seq)
                result = {'round_no': number, 'dealer': dealer, 'winner': None if draw else seat}
                if not draw:
                    local = calculate_fan(hands[seat], draws[seat], len(melds[seat]), chains[seat], piaos[seat],
                                          you_cai_bi_kao, kong_draw[seat])
                    check(local is not None, 'official winner is locally eligible to hu', number, seq)
                    if local:
                        check(data['fan'] == local.fan, 'official fan matches local calculator', number, seq)
                        result.update(local_fan=local.fan, official_fan=data['fan'],
                                      local_payments=payments(seat, dealer, local.fan, base_score))
                outcomes.append(result)
                previous_dealer, previous_winner, previous_was_draw = dealer, seat, draw
            if kind in ('tile_drawn', 'tile_discarded', 'chi', 'peng', 'gang'):
                visible = [t for row in hands for t in row] + [t for row in rivers for t in row]
                visible += [t for row in melds for m in row for t in m['tiles']]
                check(all(n <= 4 for n in Counter(visible).values()), 'no more than four copies of a tile', number, seq)
                check(20 <= 136 - len(visible) <= 83, 'wall remains within physical limits', number, seq)
        if not ended: missing.append({'round_no': number, 'reason': 'missing round_ended'})
    failures = [c for c in checks if not c['ok']]
    return {'status': 'failed' if failures else 'incomplete' if missing or not outcomes else 'passed',
            'hands': len(outcomes), 'checks': len(checks), 'failures': failures, 'missing': missing,
            'outcomes': outcomes, 'event_types': dict(event_types),
            'scope': 'observed replay transitions, fan and physical invariants; unobserved paths remain unverified'}


def main(argv=None):
    import argparse
    import json
    from .storage import read_json, atomic_json
    p = argparse.ArgumentParser(description='Audit completed official replay JSON against local rules')
    p.add_argument('replay')
    p.add_argument('--base-score', type=int, default=1)
    p.add_argument('--you-cai-bi-kao', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    result = audit_replay(read_json(args.replay), args.base_score, args.you_cai_bi_kao)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('outcomes', 'failures')}, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
