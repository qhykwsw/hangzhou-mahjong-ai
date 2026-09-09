"""Four-player Mahjong engine; research implementation, not the official binary."""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, replace
from typing import Protocol, Sequence

from .rules import WHITE, counts, calculate_fan, is_baotou, payments
from .types import Action, Meld, Observation, PublicEvent, RoundConfig, RoundResult, StageContext, StageKind


ENGINE_VERSION = 'hangzhou-local-v5'


def derive_seed(*parts) -> int:
    raw = ':'.join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


class IllegalAction(ValueError):
    pass


class RoundEngine(Protocol):
    name: str
    def run_round(self, policies: Sequence, config: RoundConfig, seed: int) -> RoundResult: ...


class HandState:
    def __init__(self, config, seed, dealer=0, hand_no=1, scores=(0, 0, 0, 0),
                 previous_events=(), in_tiebreak=False):
        self.config = config
        self.dealer = dealer
        self.hand_no = hand_no
        self.rounds_total = config.rounds
        self.scores = tuple(scores)
        self.in_tiebreak = in_tiebreak
        self.wall = [t for t in range(34) for _ in range(4)]
        random.Random(seed).shuffle(self.wall)
        self.hands = [[] for _ in range(4)]
        for _ in range(13):
            for seat in range(4):
                self.hands[seat].append(self.wall.pop())
        self.white_acquired = [h.count(WHITE) for h in self.hands]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.events: list[tuple[PublicEvent, int | None]] = list(previous_events)
        self.hand_start = len(self.events)
        self.initial_hands = [list(h) for h in self.hands]
        for seat in range(4):
            self._event('deal', seat, tiles=tuple(self.hands[seat]), private=seat)
        self.phase = 'draw'
        self.turn = dealer
        self.drawn = None
        self.kong_draw = False
        self.chain = [0] * 4
        self.piao = [0] * 4
        self.catch_owner = None
        self.response_queue = []
        self.discarder = None
        self.discarded = None
        self.winner = None
        self.fan = None
        self.delta = (0, 0, 0, 0)
        self.settled = False
        self.steps = 0
        self._draw(dealer)

    def _event(self, kind, seat=None, tile=None, tiles=(), private=None, scores=()):
        self.events.append((PublicEvent(kind, seat, tile, tuple(tiles), self.hand_no, tuple(scores)), private))

    def _draw(self, seat, kong=False):
        if len(self.wall) <= 20:
            self._finish(None)
            return
        if not kong and self.catch_owner == seat:
            self.catch_owner = None
        self.turn = seat
        self.drawn = self.wall.pop(0) if kong else self.wall.pop()
        self.hands[seat].append(self.drawn)
        if self.drawn == WHITE: self.white_acquired[seat] += 1
        self.kong_draw = kong
        self.phase = 'draw'
        self._event('draw', seat, self.drawn, private=seat)

    def current_fan(self):
        if self.phase != 'draw':
            return None
        s = self.turn
        return calculate_fan(self.hands[s], self.drawn, len(self.melds[s]),
                             self.chain[s], self.piao[s], self.config.you_cai_bi_kao, self.kong_draw)

    def legal_actions(self):
        if self.settled:
            return ()
        s = self.turn
        h = self.hands[s]
        c = counts(h)
        catch_restricted = self.catch_owner is not None and self.catch_owner != s
        if self.phase == 'draw':
            result = []
            if self.current_fan():
                result.append(Action('hu'))
            if catch_restricted:
                if self.drawn is not None:
                    result.append(Action('discard', self.drawn))
            else:
                result.extend(Action('discard', t) for t in sorted(set(h)))
            if self.drawn is not None and len(self.wall) > 20:
                result.extend(Action('gang', t, kind='concealed') for t in range(33) if c[t] == 4)
                if not catch_restricted:
                    result.extend(Action('gang', m.tiles[0], kind='added') for m in self.melds[s]
                                  if m.kind == 'peng' and c[m.tiles[0]])
            return tuple(result)
        result = [Action('pass')]
        t = self.discarded
        if t is None or t == WHITE or catch_restricted:
            return tuple(result)
        if self.phase == 'response_peng':
            if c[t] >= 2:
                result.append(Action('peng', t))
            if c[t] == 3 and len(self.wall) > 20:
                result.append(Action('gang', t, kind='exposed'))
        elif self.phase == 'response_chi' and t < 27 and sum(m.kind == 'chi' for m in self.melds[s]) < 2:
            suit = t // 9 * 9
            for start in range(max(suit, t - 2), min(t, suit + 6) + 1):
                used = tuple(x for x in range(start, start + 3) if x != t)
                if all(c[x] for x in used):
                    result.append(Action('chi', t, used))
        return tuple(result)

    def observation(self):
        seat = self.turn
        public = []
        for e, private in self.events:
            if private is None or private == seat:
                public.append(e)
            else:
                public.append(replace(e, tile=None, tiles=()))
        melds = tuple(tuple(m if not m.concealed or i == seat else replace(m, tiles=())
                            for m in row) for i, row in enumerate(self.melds))
        fan = self.current_fan()
        return Observation(
            StageContext(self.config.stage, seat, self.scores, self.hand_no,
                         self.rounds_total, self.dealer, in_tiebreak=self.in_tiebreak),
            tuple(sorted(self.hands[seat])), tuple(public), self.legal_actions(), melds,
            tuple(tuple(x) for x in self.discards), tuple(len(h) for h in self.hands),
            len(self.wall), self.phase, self.drawn if self.phase == 'draw' else None,
            self.catch_owner is not None, self.chain[seat], self.piao[seat], fan.fan if fan else 0,
            self.config.you_cai_bi_kao,
            god_discarder_seat=self.catch_owner if self.catch_owner is not None else -1,
        )

    def timeout_action(self):
        """Official fallback uses insertion order, not the sorted public hand."""
        legal = self.legal_actions()
        if self.phase.startswith('response_'):
            return next(a for a in legal if a.action == 'pass')
        hu = next((a for a in legal if a.action == 'hu'), None)
        if hu is not None:
            return hu
        for tile in reversed(self.hands[self.turn]):
            action = next((a for a in legal if a.action == 'discard' and a.tile == tile), None)
            if action is not None:
                return action
        raise IllegalAction('no official timeout action')

    def step(self, action):
        legal = self.legal_actions()
        if action not in legal:
            raise IllegalAction(f'{action} not legal in {self.phase}, seat {self.turn}')
        self.steps += 1
        s = self.turn
        h = self.hands[s]
        if action.action == 'hu':
            self.fan = self.current_fan()
            self._finish(s)
        elif action.action == 'discard':
            waiting = list(h)
            if self.drawn is not None:
                waiting.remove(self.drawn)
            was_bt = self.drawn is not None and is_baotou(waiting, len(self.melds[s]))
            h.remove(action.tile)
            # The exempt player can resume piao after a chi/peng without drawing.
            called_in_circle = self.catch_owner == s and self.drawn is None
            piao = action.tile == WHITE and (was_bt or called_in_circle) and is_baotou(h, len(self.melds[s]))
            if piao:
                self.chain[s] += 1
                self.piao[s] += 1
            else:
                self.chain[s] = 0
                self.piao[s] = 0
            self.discards[s].append(action.tile)
            self._event('discard', s, action.tile)
            self.discarder, self.discarded = s, action.tile
            self.drawn = None
            self.kong_draw = False
            if action.tile == WHITE:
                self.catch_owner = s
            elif self.catch_owner == s:
                self.catch_owner = None
            if action.tile == WHITE:
                self._draw((s + 1) % 4)
            else:
                self.phase = 'response_peng'
                # Open the window even when the exempt player cannot claim.
                self.response_queue = ([self.catch_owner] if self.catch_owner is not None
                                       else [(s + i) % 4 for i in range(1, 4)])
                self.turn = self.response_queue.pop(0)
        elif action.action == 'pass':
            if self.phase == 'response_peng' and self.response_queue:
                self.turn = self.response_queue.pop(0)
            elif self.phase == 'response_peng' and self.catch_owner in (None, (self.discarder + 1) % 4):
                self.phase = 'response_chi'
                self.turn = (self.discarder + 1) % 4
            else:
                self._draw((self.discarder + 1) % 4)
        elif action.action in ('chi', 'peng'):
            consumed = action.tiles if action.action == 'chi' else (action.tile,) * 2
            for t in consumed:
                h.remove(t)
            self.discards[self.discarder].pop()
            tiles = tuple(sorted((*consumed, action.tile)))
            self.melds[s].append(Meld(action.action, tiles, self.discarder))
            self._event(action.action, s, action.tile, tiles)
            self.phase = 'draw'
            self.drawn = None  # cannot hu before drawing
            self.kong_draw = False
            if self.catch_owner != s:
                self.chain[s] = self.piao[s] = 0
        elif action.action == 'gang':
            t = action.tile
            if action.kind == 'exposed':
                for _ in range(3): h.remove(t)
                self.discards[self.discarder].pop()
                self.melds[s].append(Meld('gang', (t,) * 4, self.discarder))
            elif action.kind == 'concealed':
                for _ in range(4): h.remove(t)
                self.melds[s].append(Meld('gang', (t,) * 4, concealed=True))
            else:
                h.remove(t)
                index = next(i for i, m in enumerate(self.melds[s]) if m.kind == 'peng' and m.tiles[0] == t)
                self.melds[s][index] = replace(self.melds[s][index], kind='gang', tiles=(t,) * 4)
            self.chain[s] += 1
            self._event('gang', s, t, (t,) * 4, private=s if action.kind == 'concealed' else None)
            self._draw(s, kong=True)

    def _finish(self, winner):
        self.winner = winner
        self.settled = True
        self.phase = 'settled'
        if winner is not None:
            self.delta = payments(winner, self.dealer, self.fan.fan, self.config.base_score)
        self._event('hand_ended', winner, scores=self.delta)

    def assert_conservation(self):
        all_tiles = list(self.wall)
        for hand in self.hands: all_tiles.extend(hand)
        for discards in self.discards: all_tiles.extend(discards)
        for row in self.melds:
            for m in row: all_tiles.extend(m.tiles)
        if counts(all_tiles) != (4,) * 34:
            raise AssertionError('tile conservation violated')


class MahjongEngine:
    name = ENGINE_VERSION

    def __init__(self, record_events=True, check_invariants=False):
        self.record_events = record_events
        self.check_invariants = check_invariants

    def run_round(self, policies, config, seed, *, policy_seeds=None):
        from .settlement import run_settlement
        return run_settlement(self, policies, config, seed, policy_seeds=policy_seeds)
