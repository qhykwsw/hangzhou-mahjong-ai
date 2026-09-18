from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

RANK_OBJECTIVE = "four-player-settlement-rank-v1"  # 旧口径，仅用于识别历史检查点
SCORE_OBJECTIVE = "four-player-settlement-score-v1"  # 主口径：完整结算的桌面分


class StageKind(StrEnum):
    QUALIFIER = "qualifier"
    ADVANCEMENT = "advancement"
    FINAL = "final"


@dataclass(frozen=True)
class RoundConfig:
    rounds: int = 8
    stage: StageKind = StageKind.QUALIFIER
    base_score: int = 1
    you_cai_bi_kao: bool = False
    allow_tiebreak: bool | None = None
    max_tiebreak_rounds: int | None = None  # execution budget, NOT a game rule
    max_actions_per_hand: int = 2000
    games_per_round: int = 1
    rotate_seats: bool = True
    player_ids: tuple[str, ...] = ("player-0", "player-1", "player-2", "player-3")

    def __post_init__(self):
        object.__setattr__(self, "stage", StageKind(self.stage))
        object.__setattr__(self, "player_ids", tuple(self.player_ids))
        if type(self.games_per_round) is not int or self.games_per_round < 1:
            raise ValueError("games_per_round must be a positive integer")
        if (
            len(self.player_ids) != 4
            or len(set(self.player_ids)) != 4
            or any(not isinstance(x, str) or not x for x in self.player_ids)
        ):
            raise ValueError("four distinct stable player IDs required")
        if self.allow_tiebreak is None:
            object.__setattr__(self, "allow_tiebreak", self.stage == StageKind.FINAL)
        if self.rounds < 1 or self.base_score < 1:
            raise ValueError("rounds and base_score must be positive")
        if self.max_actions_per_hand < 1:
            raise ValueError("action budget must be positive")
        if self.max_tiebreak_rounds is not None and self.max_tiebreak_rounds < 0:
            raise ValueError("tiebreak budget must be nonnegative")
        if self.allow_tiebreak and self.stage != StageKind.FINAL:
            raise ValueError("only final configuration has score tiebreak games")

    def as_dict(self):
        result = asdict(self)
        result["player_ids"] = list(self.player_ids)
        return result


@dataclass(frozen=True)
class StageContext:
    kind: StageKind
    seat: int
    score_by_seat: tuple[int, ...]
    round_no: int
    rounds_total: int
    dealer: int
    stage_no: int = 1
    stage_scores: tuple[int, ...] | None = None
    rank_by_seat: tuple[int, ...] | None = None
    in_tiebreak: bool = False

    stage_round_no: int | None = None
    stage_rounds_total: int | None = None

    @property
    def effective_scores(self):
        return (
            self.stage_scores if self.stage_scores is not None else self.score_by_seat
        )

    @property
    def effective_horizon(self):
        if self.stage_round_no is not None and self.stage_rounds_total is not None:
            return self.stage_round_no, self.stage_rounds_total
        return self.round_no, self.rounds_total

    @property
    def remaining_rounds(self) -> int:
        current, total = self.effective_horizon
        return max(0, total - current)

    @property
    def progress(self) -> float:
        current, total = self.effective_horizon
        return min(1.0, max(0.0, (current - 1) / max(1, total - 1)))

    @property
    def own_score(self) -> int:
        return self.effective_scores[self.seat]


@dataclass(frozen=True)
class Action:
    action: str
    tile: int | None = None
    tiles: tuple[int, ...] = ()
    kind: str = ""

    def as_dict(self):
        from mahjong_ai.domain.rules import tile_code

        d: dict[str, Any] = {"action": self.action}
        if self.tile is not None:
            d["tile"] = tile_code(self.tile)
        if self.tiles:
            d["tiles"] = [tile_code(t) for t in self.tiles]
        if self.kind:
            d["kind"] = self.kind
        return d


@dataclass(frozen=True)
class Meld:
    kind: str
    tiles: tuple[int, ...]
    from_seat: int | None = None
    concealed: bool = False


@dataclass(frozen=True)
class PublicEvent:
    kind: str
    seat: int | None = None
    tile: int | None = None
    tiles: tuple[int, ...] = ()
    hand_no: int = 0
    scores: tuple[int, ...] = ()
    concealed: bool = False


@dataclass(frozen=True)
class Observation:
    stage: StageContext
    hand: tuple[int, ...]
    public_events: tuple[PublicEvent, ...]
    legal_actions: tuple[Action, ...]
    melds: tuple[tuple[Meld, ...], ...]
    discards: tuple[tuple[int, ...], ...]
    hand_sizes: tuple[int, ...]
    wall_remaining: int
    phase: str
    drawn_tile: int | None
    catch_play: bool
    chain_count: int
    piao_count: int
    hu_fan: int = 0
    you_cai_bi_kao: bool = False
    god_discarder_seat: int = -1  # public physical seat; -1 when no catch-play circle
    base_score: int = 1

    @property
    def catch_restricted(self) -> bool:
        return self.catch_play and self.god_discarder_seat != self.stage.seat


@dataclass(frozen=True)
class RoundResult:
    engine: str
    seed: int
    stage: StageKind
    rounds_played: int
    total_score_by_seat: tuple[int, ...]
    winner_seat: int | None
    top_two_seats: tuple[int, ...]
    completed: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    rank_by_seat: tuple[
        float, ...
    ] = ()  # stable participant slots, NOT the last game's seats

    def score(self, seat):
        return self.total_score_by_seat[seat]

    def rank_utility(self, seat):
        if not self.completed or len(self.rank_by_seat) != 4:
            raise ValueError(
                "rank reward requires a completed settlement with four ranks"
            )
        return 4.0 - self.rank_by_seat[seat]

    def as_dict(self):
        return asdict(self)
