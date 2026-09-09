"""Official player snapshot adapter, including v26 catch-play exemptions."""

from __future__ import annotations

from mahjong_ai.platform.online import MAPPING_TEMPLATE, SnapshotAdapter, SnapshotError

EVENTS = {
    "tile_drawn": "draw",
    "tile_discarded": "discard",
    "round_ended": "hand_ended",
    "chi": "chi",
    "peng": "peng",
    "gang": "gang",
    "pass": "pass",
    "timeout": "timeout",
    "game_ended": "game_ended",
}


class OfficialAdapter:
    def __init__(self):
        self.normalizer = SnapshotAdapter(
            {key: "normalized." + key for key in MAPPING_TEMPLATE}
        )

    def __call__(self, document):
        snap, tournament = document["snapshot"], document["tournament"]
        seat = snap["seat"]
        phase = snap["phase"]
        responding = snap.get("responding_seats") or []
        if not (
            phase == "draw"
            and snap["turn"] == seat
            or phase in ("response_peng", "response_chi")
            and seat in responding
        ):
            return None
        cfg = tournament["config"]
        god = snap["god"]
        # Older captured snapshots without an active circle remain readable.
        # An active circle requires an explicit owner; guessing can change legality.
        owner = god.get("god_discarder_seat", -1)
        wire_events = document.get("wire_events", [])
        own_events = [
            e
            for e in wire_events
            if e.get("seat") == seat
            and e.get("_round_no", snap["round_no"]) == snap["round_no"]
        ]
        actions = [e for e in own_events if e["type"] in ("tile_discarded", "gang")]
        chain_count = god["chain_count"]
        if len(actions) < chain_count:
            raise SnapshotError(
                "insufficient own chain history; cannot infer piao count"
            )
        chain_actions = actions[-chain_count:] if chain_count else []
        if any(
            e["type"] == "tile_discarded" and e.get("tile") != "白"
            for e in chain_actions
        ):
            raise SnapshotError(
                "official chain disagrees with received own event history"
            )
        piao = sum(e["type"] == "tile_discarded" for e in chain_actions)
        # Gang replenishment is the draw following own gang with no intervening
        # own discard/call; event data alone is not assumed to carry a flag.
        own_relevant = [
            e
            for e in own_events
            if e["type"] in ("gang", "chi", "peng", "tile_discarded")
        ]
        kong_draw = (
            bool(snap.get("drawn_tile"))
            and bool(own_relevant)
            and own_relevant[-1]["type"] == "gang"
        )
        public = []
        for e in wire_events:
            if e["type"] not in EVENTS:
                continue
            tile = e.get("tile") or None
            if tile == "0w":
                tile = None
            data = e.get("data") or {}
            tiles = data.get("tiles") or []
            if e["type"] == "tile_drawn" and e.get("seat") != seat:
                tile = None
            if (
                e["type"] == "gang"
                and data.get("kind") == "an"
                and e.get("seat") != seat
            ):
                tile, tiles = None, []
            public.append(
                {
                    "kind": EVENTS[e["type"]],
                    "seat": e.get("seat"),
                    "tile": tile,
                    "tiles": tiles,
                    "hand_no": e.get("_round_no", snap["round_no"]),
                    "scores": data.get("scores") or [],
                }
            )
        melds = []
        for s, row in enumerate(snap["melds"]):
            normalized = []
            for m in row:
                kind = m["kind"]
                concealed = kind == "gang_an" or m.get("concealed", False)
                if kind in ("gang_an", "gang_ming", "gang_bu"):
                    kind = "gang"
                tiles = m.get("tiles") or []
                if concealed and s != seat:
                    tiles = []
                normalized.append(
                    {"kind": kind, "tiles": tiles, "concealed": concealed}
                )
            melds.append(normalized)
        stage = tournament.get("stage") or {}
        if cfg.get("Kind") == "test":
            stage_kind = "qualifier"
        else:
            stage_kind = (
                "final"
                if stage.get("role") == "final"
                else "qualifier"
                if stage.get("no", 1) == 1
                else "advancement"
            )
        # Ranking is keyed by identities, and the player snapshot has no seat->id
        # map. Do not silently treat leaderboard order as seat order.
        scores = document.get("stage_scores")
        normalized = dict(
            seat=seat,
            phase=phase,
            turn=snap["turn"],
            responding_seats=responding,
            hand=snap["my_hand"],
            drawn_tile=snap.get("drawn_tile") or None,
            melds=melds,
            discards=snap["discards"],
            hand_sizes=snap["hand_counts"],
            scores=snap["scores"],
            round_no=snap["round_no"],
            dealer=snap["dealer"],
            rounds_total=cfg["Rounds"],
            base_score=cfg["BaseScore"],
            you_cai_bi_kao=cfg["YouCaiBiKao"],
            wall_remaining=snap["wall_remaining"],
            discarded_tile=None
            if snap["last_discard"] == "0w"
            else snap["last_discard"],
            catch_play=god["catch_play"],
            chain_count=chain_count,
            piao_count=piao,
            kong_draw=kong_draw,
            god_discarder_seat=owner,
            public_events=public,
            stage_kind=stage_kind,
            stage_no=stage.get("no", 1),
            stage_scores=scores,
            stage_round_no=None,
            stage_rounds_total=None,
            in_tiebreak=bool(document.get("in_tiebreak", False)),
        )
        return self.normalizer({"normalized": normalized})
