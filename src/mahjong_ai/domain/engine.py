"""Four-player Mahjong engine; research implementation, not the official binary."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Protocol, Sequence

from mahjong_ai.domain.rules import WHITE, counts, calculate_fan, payments
from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.types import (
    Action,
    Meld,
    Observation,
    PublicEvent,
    RoundConfig,
    RoundResult,
    StageContext,
)


ENGINE_VERSION = "hangzhou-local-v9"
POLICY_EVENT_KINDS = frozenset(("draw", "discard", "chi", "peng", "gang", "hand_ended"))


_PASS_ACTION = Action("pass")
_HU_ACTION = Action("hu")
_DISCARD_ACTIONS = tuple(Action("discard", tile) for tile in range(34))
_PENG_ACTIONS = tuple(Action("peng", tile) for tile in range(34))
_GANG_ACTIONS = {
    kind: tuple(Action("gang", tile, kind=kind) for tile in range(34))
    for kind in ("concealed", "added", "exposed")
}
_CHI_ACTIONS = tuple(
    tuple(
        (used, Action("chi", tile, used))
        for start in range(
            max(tile // 9 * 9, tile - 2), min(tile, tile // 9 * 9 + 6) + 1
        )
        for used in (tuple(x for x in range(start, start + 3) if x != tile),)
    )
    for tile in range(27)
)


def derive_seed(*parts) -> int:
    raw = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


class IllegalAction(ValueError):
    pass


class RoundEngine(Protocol):
    name: str

    def run_round(
        self, policies: Sequence, config: RoundConfig, seed: int
    ) -> RoundResult: ...


class HandState:
    def __init__(
        self,
        config,
        seed,
        dealer=0,
        hand_no=1,
        scores=(0, 0, 0, 0),
        previous_events=(),
        in_tiebreak=False,
        previous_views=None,
    ):
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
        # Per-seat policy-visible projections of ``events``; appended lazily so
        # every observation is O(new events) instead of re-masking the whole
        # multi-hand history.  A caller that already holds the projection of
        # ``previous_events`` (the settlement loop) passes it in.
        if previous_views is not None:
            if len(previous_views) != 4:
                raise ValueError("previous_views requires four seat projections")
            self._views = [list(v) for v in previous_views]
            self._views_source = self.events
            self._views_len = len(self.events)
        self.initial_hands = [list(h) for h in self.hands]
        for seat in range(4):
            self._event("deal", seat, tiles=tuple(self.hands[seat]), private=seat)
        self.phase = "draw"
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
        self._draw(dealer, initial=True)

    def _event(self, kind, seat=None, tile=None, tiles=(), private=None, scores=()):
        self.events.append(
            (
                PublicEvent(
                    kind,
                    seat,
                    tile,
                    tuple(tiles),
                    self.hand_no,
                    tuple(scores),
                    concealed=kind == "gang" and private is not None,
                ),
                private,
            )
        )

    def _draw(self, seat, kong=False, initial=False):
        if len(self.wall) <= 20:
            self._finish(None)
            return
        # Official snapshots retain the circle through its owner's next draw.
        # The owner is exempt; their non-white discard clears it, while another
        # white discard starts the next circle. Keep this policy feature aligned.
        self.turn = seat
        self.drawn = self.wall.pop(0) if kong else self.wall.pop()
        self.hands[seat].append(self.drawn)
        if self.drawn == WHITE:
            self.white_acquired[seat] += 1
        self.kong_draw = kong
        self.phase = "draw"
        if not initial:
            self._event("draw", seat, self.drawn, private=seat)

    def current_fan(self):
        if self.phase != "draw":
            return None
        s = self.turn
        return calculate_fan(
            self.hands[s],
            self.drawn,
            len(self.melds[s]),
            self.chain[s],
            self.piao[s],
            self.config.you_cai_bi_kao,
            self.kong_draw,
        )

    def legal_actions(self):
        if self.settled:
            return ()
        s = self.turn
        h = self.hands[s]
        catch_restricted = self.catch_owner is not None and self.catch_owner != s
        if self.phase == "draw":
            c = counts(h)
            result = []
            if self.current_fan():
                result.append(_HU_ACTION)
            if catch_restricted:
                if self.drawn is not None:
                    result.append(_DISCARD_ACTIONS[self.drawn])
            else:
                result.extend(_DISCARD_ACTIONS[t] for t in sorted(set(h)))
            if len(self.wall) > 20:
                result.extend(
                    _GANG_ACTIONS["concealed"][t] for t in range(33) if c[t] == 4
                )
                if not catch_restricted:
                    result.extend(
                        _GANG_ACTIONS["added"][m.tiles[0]]
                        for m in self.melds[s]
                        if m.kind == "peng" and c[m.tiles[0]]
                    )
            return tuple(result)
        result = [_PASS_ACTION]
        t = self.discarded
        if t is None or t == WHITE or catch_restricted:
            return tuple(result)
        if self.phase == "response_peng":
            copies = h.count(t)
            if copies >= 2:
                result.append(_PENG_ACTIONS[t])
            if copies == 3 and len(self.wall) > 20:
                result.append(_GANG_ACTIONS["exposed"][t])
        elif (
            self.phase == "response_chi"
            and t < 27
            and sum(m.kind == "chi" for m in self.melds[s]) < 2
        ):
            c = counts(h)
            for used, action in _CHI_ACTIONS[t]:
                if c[used[0]] and c[used[1]]:
                    result.append(action)
        return tuple(result)

    def _sync_views(self):
        events = self.events
        if getattr(self, "_views_source", None) is not events or getattr(
            self, "_views_len", 0
        ) > len(events):
            self._views = [[], [], [], []]
            self._views_source = events
            self._views_len = 0
        done = self._views_len
        if done == len(events):
            return
        views = self._views
        for event, private in events[done:]:
            kind = event.kind
            if kind not in POLICY_EVENT_KINDS:
                continue
            tile, tiles = event.tile, tuple(sorted(event.tiles))
            if kind in ("peng", "gang") and tile is not None:
                tiles = (tile,) * (3 if kind == "peng" else 4)
            scores = tuple(event.scores)
            visible = (
                event
                if tile == event.tile
                and tiles == event.tiles
                and scores == event.scores
                else replace(event, tile=tile, tiles=tiles, scores=scores)
            )
            hidden = None
            secret_draw = kind == "draw"
            for viewer in range(4):
                if (
                    (private is not None and private != viewer)
                    or (secret_draw and event.seat != viewer)
                    or (event.concealed and event.seat != viewer)
                ):
                    if hidden is None:
                        hidden = replace(event, tile=None, tiles=(), scores=scores)
                    views[viewer].append(hidden)
                else:
                    views[viewer].append(visible)
        self._views_len = len(events)

    def public_views(self):
        """Four per-seat policy event lists covering every recorded event."""
        self._sync_views()
        return self._views

    def observation(self, legal=None):
        """The policy's view; ``legal`` passes legal_actions() already computed this step."""
        seat = self.turn
        self._sync_views()
        melds = tuple(
            tuple(
                m if not m.concealed or i == seat else replace(m, tiles=()) for m in row
            )
            for i, row in enumerate(self.melds)
        )
        fan = self.current_fan()
        return Observation(
            StageContext(
                self.config.stage,
                seat,
                self.scores,
                self.hand_no,
                self.rounds_total,
                self.dealer,
                in_tiebreak=self.in_tiebreak,
            ),
            tuple(sorted(self.hands[seat])),
            tuple(self._views[seat]),
            self.legal_actions() if legal is None else legal,
            melds,
            tuple(tuple(x) for x in self.discards),
            tuple(len(h) for h in self.hands),
            len(self.wall),
            self.phase,
            self.drawn if self.phase == "draw" else None,
            self.catch_owner is not None,
            self.chain[seat],
            self.piao[seat],
            fan.fan if fan else 0,
            self.config.you_cai_bi_kao,
            god_discarder_seat=self.catch_owner if self.catch_owner is not None else -1,
            base_score=self.config.base_score,
        )

    def timeout_action(self, legal=None):
        """Official fallback uses insertion order, not the sorted public hand."""
        if legal is None:
            legal = self.legal_actions()
        if self.phase.startswith("response_"):
            return next(a for a in legal if a.action == "pass")
        hu = next((a for a in legal if a.action == "hu"), None)
        if hu is not None:
            return hu
        for tile in reversed(self.hands[self.turn]):
            action = next(
                (a for a in legal if a.action == "discard" and a.tile == tile), None
            )
            if action is not None:
                return action
        raise IllegalAction("no official timeout action")

    def step(self, action, legal=None):
        """Apply ``action``; ``legal`` is this step's legal_actions() when the caller has it."""
        if legal is None:
            legal = self.legal_actions()
        if action not in legal:
            raise IllegalAction(f"{action} not legal in {self.phase}, seat {self.turn}")
        self.steps += 1
        s = self.turn
        h = self.hands[s]
        if action.action == "hu":
            self.fan = self.current_fan()
            self._finish(s)
        elif action.action == "discard":
            # Only this hand's events can hold the caller's last move.
            piao = action.tile == WHITE and can_discard_piao(
                h,
                self.drawn,
                len(self.melds[s]),
                [e for e, _ in self.events[self.hand_start :]],
                s,
                self.hand_no,
                catch_exempt=self.catch_owner == s,
            )
            h.remove(action.tile)
            if piao:
                self.chain[s] += 1
                self.piao[s] += 1
            else:
                self.chain[s] = 0
                self.piao[s] = 0
            self.discards[s].append(action.tile)
            self._event("discard", s, action.tile)
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
                self.phase = "response_peng"
                # Open the window even when the exempt player cannot claim.
                self.response_queue = (
                    [self.catch_owner]
                    if self.catch_owner is not None
                    else [(s + i) % 4 for i in range(1, 4)]
                )
                self.turn = self.response_queue.pop(0)
        elif action.action == "pass":
            if self.phase == "response_peng" and self.response_queue:
                self.turn = self.response_queue.pop(0)
            elif self.phase == "response_peng" and self.catch_owner in (
                None,
                (self.discarder + 1) % 4,
            ):
                self.phase = "response_chi"
                self.turn = (self.discarder + 1) % 4
            else:
                self._draw((self.discarder + 1) % 4)
        elif action.action in ("chi", "peng"):
            consumed = action.tiles if action.action == "chi" else (action.tile,) * 2
            for t in consumed:
                h.remove(t)
            self.discards[self.discarder].pop()
            tiles = tuple(sorted((*consumed, action.tile)))
            self.melds[s].append(Meld(action.action, tiles, self.discarder))
            self._event(action.action, s, action.tile, tiles)
            self.phase = "draw"
            self.drawn = None  # cannot hu before drawing
            self.kong_draw = False
            if self.catch_owner != s:
                self.chain[s] = self.piao[s] = 0
        elif action.action == "gang":
            t = action.tile
            if action.kind == "exposed":
                for _ in range(3):
                    h.remove(t)
                self.discards[self.discarder].pop()
                self.melds[s].append(Meld("gang", (t,) * 4, self.discarder))
            elif action.kind == "concealed":
                for _ in range(4):
                    h.remove(t)
                self.melds[s].append(Meld("gang", (t,) * 4, concealed=True))
            else:
                h.remove(t)
                index = next(
                    i
                    for i, m in enumerate(self.melds[s])
                    if m.kind == "peng" and m.tiles[0] == t
                )
                self.melds[s][index] = replace(
                    self.melds[s][index], kind="gang", tiles=(t,) * 4
                )
            self.chain[s] += 1
            self._event(
                "gang",
                s,
                t,
                (t,) * 4,
                private=s if action.kind == "concealed" else None,
            )
            self._draw(s, kong=True)

    def _finish(self, winner):
        self.winner = winner
        self.settled = True
        self.phase = "settled"
        if winner is not None:
            self.delta = payments(
                winner, self.dealer, self.fan.fan, self.config.base_score
            )
        self._event("hand_ended", winner, scores=self.delta)

    def assert_conservation(self):
        all_tiles = list(self.wall)
        for hand in self.hands:
            all_tiles.extend(hand)
        for discards in self.discards:
            all_tiles.extend(discards)
        for row in self.melds:
            for m in row:
                all_tiles.extend(m.tiles)
        if counts(all_tiles) != (4,) * 34:
            raise AssertionError("tile conservation violated")


class MahjongEngine:
    name = ENGINE_VERSION

    def __init__(self, record_events=True, check_invariants=False):
        self.record_events = record_events
        self.check_invariants = check_invariants

    def run_round(self, policies, config, seed, *, policy_seeds=None):
        from mahjong_ai.domain.settlement import run_settlement

        return run_settlement(self, policies, config, seed, policy_seeds=policy_seeds)
