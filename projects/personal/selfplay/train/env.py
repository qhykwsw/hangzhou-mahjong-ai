"""One table = one game (场) driven directly on HandState with oracle access.

The table asks for model decisions through `pending_request` and receives them
through `apply`; fixed opponents are played inline.  Transitions for the
current policy's seats are assembled per (seat) chain with the hold-back rule:
a decision is emitted only when its successor state is known (next decision of
the same seat, or the end of the game), so bootstrapped GAE never needs a value
that does not exist yet.
"""

from dataclasses import dataclass, field
import random

import numpy as np

from mahjong_ai.domain.engine import HandState, derive_seed
from mahjong_ai.domain.rules import WHITE
from mahjong_ai.domain.types import RoundConfig, StageKind
from mahjong_ai.sdk.shape import shanten
from selfplay_core.encoding import encode

N_ORACLE_PLANES = 16  # 3 opponents x 4 count planes + 4 wall count planes
N_ORACLE_SCALARS = 4  # 3 opponent shanten + wall tiles left fraction

_THRESH = np.arange(1, 5, dtype=np.uint8)[:, None]

SLOT_CURRENT = 0


def _count_planes_u8(target, tiles):
    if tiles:
        c = np.bincount(np.fromiter(tiles, dtype=np.int64, count=len(tiles)), minlength=34)
        target[:] = (c[None, :] >= _THRESH) * 255


def encode_oracle(state: HandState, seat: int):
    planes = np.zeros((N_ORACLE_PLANES, 34), dtype=np.uint8)
    scal = np.zeros(N_ORACLE_SCALARS, dtype=np.float16)
    for i in range(1, 4):
        s = (seat + i) % 4
        _count_planes_u8(planes[4 * (i - 1) : 4 * i], state.hands[s])
        opened = len(state.melds[s])
        h = state.hands[s]
        if len(h) in (13 - 3 * opened, 14 - 3 * opened):
            try:
                scal[i - 1] = (8 - shanten(tuple(sorted(h)), opened)) / 9.0
            except ValueError:
                scal[i - 1] = 0.0
    _count_planes_u8(planes[12:16], state.wall)
    scal[3] = len(state.wall) / 136.0
    return planes, scal


@dataclass
class Pending:
    planes: np.ndarray
    scalars: np.ndarray
    events: np.ndarray
    oracle_planes: np.ndarray
    oracle_scalars: np.ndarray
    mask: np.ndarray
    action: int
    logprob: float
    value: float
    shanten: int
    hand_no: int
    reward: float = 0.0
    aux_hand: float = 0.0
    aux_hand_known: bool = False


@dataclass
class Chain:
    closed: list = field(default_factory=list)  # store indices
    pending: Pending | None = None


class Table:
    def __init__(self, cfg, rng: random.Random, factory, store, table_id, eval_mode=False):
        self.cfg = cfg
        self.rng = rng
        self.factory = factory
        self.store = store  # SampleWriter
        self.table_id = table_id
        self.eval_mode = eval_mode
        self.games = 0
        self.shaping_weight = 0.0
        self.league = None  # snapshot dict from the parent
        self.request = None
        self.state = None
        self.records = []  # finished game records since last drain
        self.finished_chains = []  # closed index lists of games completed this iteration
        self.new_game()

    # ---- game lifecycle ----------------------------------------------------
    def configure(self, league, shaping_weight):
        self.league = league
        self.shaping_weight = shaping_weight

    def new_game(self):
        cfg, rng = self.cfg, self.rng
        self.games += 1
        self.game_seed = rng.getrandbits(62)
        prob = cfg.you_cai_bi_kao_prob
        if self.eval_mode:
            rounds = cfg.eval_rounds
            # Evaluation follows the configured switch; only a mixed setting alternates.
            ycbk = bool(self.table_id % 2) if 0.0 < prob < 1.0 else prob >= 1.0
            kind = StageKind.QUALIFIER
        else:
            if cfg.rounds_choices and rng.random() < cfg.rounds_choices_prob:
                rounds = rng.choice(list(cfg.rounds_choices))
            else:
                rounds = rng.randint(cfg.rounds_min, cfg.rounds_max)
            ycbk = rng.random() < prob
            kind = rng.choices(list(StageKind), weights=cfg.stage_probs)[0]
        self.config = RoundConfig(rounds=rounds, stage=kind, you_cai_bi_kao=ycbk)
        self.seat_kinds = self._assign_seats()  # per seat: ("model", slot) | ("fixed", id)
        self.policies = {}
        for seat, (kind_, ident) in enumerate(self.seat_kinds):
            if kind_ == "fixed":
                policy = self.factory.build(ident)
                policy.reset(derive_seed(self.game_seed, "policy", seat))
                self.policies[seat] = policy
        self.chains = {
            seat: Chain()
            for seat, (kind_, ident) in enumerate(self.seat_kinds)
            if kind_ == "model" and ident == SLOT_CURRENT
        }
        self.scores = [0] * 4
        self.dealer = self.game_seed % 4
        self.hand_no = 0
        self.history, self.views = [], None
        self.fallbacks = 0
        self._next_hand()

    def _assign_seats(self):
        cfg, rng = self.cfg, self.rng
        league = self.league or {"fixed": self.factory.ids(), "fixed_weights": None, "slots": [], "slot_weights": [], "fixed_prob": 1.0}
        if self.eval_mode:
            me = rng.randrange(4)
            picks = rng.choices(league["fixed"], weights=league.get("fixed_weights"), k=3)
            out, j = [], 0
            for seat in range(4):
                if seat == me:
                    out.append(("model", SLOT_CURRENT))
                else:
                    out.append(("fixed", picks[j]))
                    j += 1
            return out
        k = rng.choices((1, 2, 3, 4), weights=cfg.self_seat_probs)[0]
        mine = set(rng.sample(range(4), k))
        out = []
        for seat in range(4):
            if seat in mine:
                out.append(("model", SLOT_CURRENT))
            elif league["slots"] and rng.random() >= league["fixed_prob"]:
                out.append(("model", rng.choices(league["slots"], weights=league["slot_weights"])[0]))
            else:
                out.append(("fixed", rng.choices(league["fixed"], weights=league.get("fixed_weights"))[0]))
        return out

    def _next_hand(self):
        self.hand_no += 1
        self.state = HandState(
            self.config,
            derive_seed(self.game_seed, "wall", self.hand_no),
            self.dealer,
            self.hand_no,
            self.scores,
            self.history,
            False,
            previous_views=self.views,
        )
        self.state.rounds_total = self.config.rounds

    # ---- stepping -----------------------------------------------------------
    def drain_records(self):
        out, self.records = self.records, []
        return out

    def advance(self):
        """Run until a model decision is needed and return the request dict.
        Finished games are recorded in ``records`` and a new game starts."""
        while True:
            state = self.state
            if state.settled:
                finished = self._finish_hand()
                if finished is not None:
                    self.records.append(finished["game_over"])
                continue
            legal = state.legal_actions()
            if len(legal) == 1:
                state.step(legal[0])
                continue
            seat = state.turn
            kind, ident = self.seat_kinds[seat]
            if kind == "fixed":
                obs = state.observation()
                try:
                    action = self.policies[seat].act(obs)
                    if action not in legal:
                        raise ValueError("illegal action from fixed opponent")
                except Exception:
                    self.fallbacks += 1
                    action = state.timeout_action()
                state.step(action)
                continue
            obs = state.observation()
            enc = encode(obs, lookahead=self.cfg.lookahead)
            req = {
                "table": self.table_id,
                "slot": ident,
                "seat": seat,
                "greedy": self.eval_mode,
                "planes": enc.planes,
                "scalars": enc.scalars,
                "events": enc.events,
                "mask": enc.mask,
                "legal": enc.legal,
                "shanten": enc.shanten,
                "obs": obs,
            }
            if ident == SLOT_CURRENT and not self.eval_mode:
                req["oracle_planes"], req["oracle_scalars"] = encode_oracle(state, seat)
            self.request = req
            return req

    def apply(self, action_index: int, logprob: float, value: float):
        req = self.request
        self.request = None
        state = self.state
        seat = req["seat"]
        legal = req["obs"].legal_actions
        pos = int(np.nonzero(req["legal"] == action_index)[0][0])
        action = legal[pos]
        if req["slot"] == SLOT_CURRENT and not self.eval_mode:
            chain = self.chains[seat]
            self._close_pending(chain, next_value=value, next_shanten=req["shanten"], done=False)
            chain.pending = Pending(
                planes=(req["planes"] * 255).astype(np.uint8),
                scalars=req["scalars"].astype(np.float16),
                events=req["events"].astype(np.int8),
                oracle_planes=req["oracle_planes"],
                oracle_scalars=req["oracle_scalars"],
                mask=req["mask"],
                action=int(action_index),
                logprob=float(logprob),
                value=float(value),
                shanten=int(req["shanten"]),
                hand_no=self.hand_no,
            )
        state.step(action)

    def _close_pending(self, chain, next_value, next_shanten, done):
        p = chain.pending
        if p is None:
            return
        w = self.shaping_weight
        if w > 0.0:
            # potential-based shaping: phi(s) = -w * shanten, phi(terminal-of-hand) = 0
            phi_next = 0.0 if (done or next_shanten is None) else -w * next_shanten
            p.reward += self.cfg.gamma * phi_next - (-w * p.shanten)
        idx = self.store.write(p, next_value=next_value, done=done)
        chain.closed.append(idx)
        chain.pending = None

    def _finish_hand(self):
        state = self.state
        delta = state.delta
        self.scores = [s + d for s, d in zip(self.scores, delta)]
        for seat, chain in self.chains.items():
            p = chain.pending
            if p is not None:
                p.reward += delta[seat] / self.config.base_score
                if p.hand_no == self.hand_no and not p.aux_hand_known:
                    p.aux_hand = delta[seat] / self.config.base_score
                    p.aux_hand_known = True
            for idx in chain.closed:
                self.store.fill_hand_aux(idx, self.hand_no, delta[seat] / self.config.base_score)
        self.history, self.views = state.events, state.public_views()
        if state.winner is not None:
            self.dealer = state.winner
        game_over = self.hand_no >= self.config.rounds
        if game_over:
            for seat, chain in self.chains.items():
                # terminal: bootstrap 0
                self._close_pending(chain, next_value=0.0, next_shanten=None, done=True)
                for idx in chain.closed:
                    self.store.fill_game_aux(idx, self.scores[seat] / self.config.base_score)
                if chain.closed:
                    self.finished_chains.append(chain.closed)
            result = self._game_record()
            self.new_game()
            return {"game_over": result}
        self._next_hand()
        return None

    def _game_record(self):
        return {
            "table": self.table_id,
            "seats": self.seat_kinds,
            "scores": list(self.scores),
            "rounds": self.config.rounds,
            "you_cai_bi_kao": self.config.you_cai_bi_kao,
            "stage": str(self.config.stage),
            "fallbacks": self.fallbacks,
        }

    # ---- iteration boundary -------------------------------------------------
    def chains_for_gae(self):
        """(closed indices, bootstrap value, terminal flag) per current-policy chain;
        pending transitions are carried over to the next iteration untouched."""
        out = [(closed, 0.0) for closed in self.finished_chains]
        self.finished_chains = []
        for chain in self.chains.values():
            if chain.closed:
                boot = chain.pending.value if chain.pending is not None else 0.0
                out.append((chain.closed, boot))
                chain.closed = []
        return out
