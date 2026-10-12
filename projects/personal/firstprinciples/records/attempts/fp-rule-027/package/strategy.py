"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Between two of my draws the three
opponents win with chance Q. A waiting hand that wins on a fraction p of draws,
collecting e fan per draw on average, is worth e / (1 - (1 - p)(1 - Q)) just
before my draw, where Q for a waiting hand rises as the wall runs down (the
others close in; measured waits are worth more early and less late than a
fixed Q says); a wait holding a white also counts the draws that let a discard
turn it into baotou, when that beats winning on them at once. One step from waiting, each improving draw leads to the best
waiting hand it allows; further away a generic discount per step stands in.
A win also cancels the payment I would owe to someone else's win, SAVED fan.
All of this counts only the draws I have left before the wall runs out.
Passing on a discard lets the opponents between its discarder and me draw
before my turn, so a call or kong also takes their draws away. A wait my
discard leaves also counts its non-winning draws that a later discard turns
into a better wait. Unseen copies of a tile already discarded count for more,
as they are likelier to be in the wall than in an opponent's hand. Discards of
equal value keep the hand whose tiles connect to more live tiles. When the hand is already baotou (every draw wins), a win now is traded for a
doubling that keeps the certain win: a kong whose replacement draw still wins,
else a white discard that scores piao. A call from a baotou wait is priced with
the piao its white discard scores when the wait stays baotou.
"""
COMBINATION = "firstprinciples-rule"

from functools import lru_cache

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai.domain.rules import WHITE, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles, shanten

Q = 0.15  # an opponent self-draws during the three draws between two of mine
DRAW_RISK = 1 - (1 - Q) ** (1 / 3)  # the same, for one opponent's draw
SAVED = 0.467  # my win cancels the payment to another winner, in fan units
ONE_STEP = 0.22  # typical value of a hand one step from waiting
FAR_STEP = 0.45  # discount for each further step
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}
LAST_WALL = 20  # the hand is drawn once the wall is down to this
Q_EARLY = 0.10
Q_LATE = 0.30
TIE = 1e-9  # weight of shape potential (in hundredths), which only separates equal values


LONG = 16  # more draws of mine than any hand holds

# Only the wall can be drawn, and unseen copies sit either there or in the
# opponents' hands. A tile someone has discarded is one they did not keep, so
# its other copies are likelier to be in the wall. Measured share in the wall
# relative to an undiscarded tile of the same kind, by stage (wall above 60,
# above 44, below), kind (honour, terminal, 2/8, 3-7) and copies discarded
# (one, two or more).
LIVE = (
    ((1.33, 1.43), (1.20, 1.28), (1.16, 1.20), (1.13, 1.23)),
    ((1.46, 1.59), (1.29, 1.36), (1.24, 1.30), (1.19, 1.28)),
    ((1.52, 1.88), (1.38, 1.52), (1.36, 1.47), (1.33, 1.39)),
)
KIND = tuple(
    0 if t >= 27 else 1 if t % 9 in (0, 8) else 2 if t % 9 in (1, 7) else 3
    for t in range(34)
)


def live_tiles(obs, available):
    """Unseen copies in hundredths, weighted up for tiles already discarded."""
    wall = obs.wall_remaining
    stage = LIVE[0 if wall > 60 else 1 if wall > 44 else 2]
    gone = [0] * 34
    for row in obs.discards:
        for tile in row:
            gone[tile] += 1
    return tuple(
        100 * n
        if t == WHITE or not gone[t]
        else round(100 * n * stage[KIND[t]][min(gone[t], 2) - 1])
        for t, n in enumerate(available)
    )


def repeat(stay, draws):
    """Expected number of my draws while a state lasts, within the draws left."""
    return (1 - stay**draws) / (1 - stay)


def tenpai_hazard(draws):
    """Opponents close in as the wall runs down: the race gets harder late."""
    return min(Q_LATE, max(Q_EARLY, 0.35 - 0.018 * draws))


def race(p, e, draws=LONG):
    return e * repeat((1 - p) * (1 - tenpai_hazard(draws)), draws)


@lru_cache(maxsize=200000)
def waiting_value(hand, opened, available, chain=0, piao=0, draws=LONG):
    """Expected fan of a 13-tile hand just before my next draw."""
    unseen = max(1, sum(available))
    row = analyze(hand, opened, available, chain, piao)
    if row["shanten"] <= 0:
        copies = row["winning_copies"]
        gain = row["fan_value"] + SAVED * copies
        if WHITE not in hand or is_baotou(hand, opened):
            return race(copies / unseen, gain / unseen, draws)
        # A draw that frees a white lets a discard turn the wait into baotou,
        # which may beat winning on it at once.
        fans = row["fans"]
        gain, reach = 0.0, 0
        for tile in range(34):
            if available[tile] <= 0 or hand.count(tile) >= 4:
                continue
            drawn = list(hand)
            drawn.append(tile)
            best = 0.0
            for drop in set(drawn):
                rest = tuple(sorted(without(drawn, drop)))
                if is_baotou(rest, opened):
                    best = max(
                        best, waiting_value(rest, opened, available, draws=draws - 1)
                    )
            if tile in fans:
                gain += available[tile] * max(fans[tile] + SAVED, (1 - Q) * best)
            elif best:
                gain += available[tile] * (1 - Q) * best
                reach += available[tile]
        stay = (1 - tenpai_hazard(draws)) * (1 - (copies + reach) / unseen)
        return gain / unseen * repeat(stay, draws)
    u = row["ukeire"] / unseen
    if row["shanten"] == 1:
        weighted = 0.0
        for tile in row["improving"]:
            drawn = list(hand)
            drawn.append(tile)
            best = 0.0
            for drop in set(drawn):
                rest = list(drawn)
                rest.remove(drop)
                rest = tuple(sorted(rest))
                best = max(best, tenpai_value(rest, opened, available, draws - 1))
            weighted += available[tile] * best
        # Improving draws move to a waiting hand; the others are thrown back.
        stay = (1 - u) * (1 - Q)
        return (1 - Q) * (weighted / unseen) * repeat(stay, draws - 1)
    step = (1 - Q) * u * repeat((1 - u) * (1 - Q), draws - row["shanten"])
    return step * ONE_STEP * FAR_STEP ** (row["shanten"] - 2)


@lru_cache(maxsize=200000)
def tenpai_value(hand, opened, available, draws=LONG):
    row = analyze(hand, opened, available)
    if row["shanten"] > 0 or draws < 1:
        return 0.0
    return waiting_value(hand, opened, available, draws=draws)


@lru_cache(maxsize=20000)
def reshaped_value(hand, opened, available, draws):
    """A waiting hand with the option to trade a non-winning draw for a better wait."""
    value = waiting_value(hand, opened, available, draws=draws)
    if draws < 2 or shanten(hand, opened) > 0:
        return value
    unseen = max(1, sum(available))
    row = analyze(hand, opened, available)
    wins = set(row["winning_tiles"])
    stay = waiting_value(hand, opened, available, draws=draws - 1)
    gain, reach = 0.0, 0
    for tile in range(34):
        if tile in wins or available[tile] <= 0 or hand.count(tile) >= 4:
            continue
        drawn = sorted((*hand, tile))
        best = stay
        for drop in set(drawn):
            rest = list(drawn)
            rest.remove(drop)
            rest = tuple(rest)
            if rest != hand and shanten(rest, opened) <= 0:
                best = max(best, waiting_value(rest, opened, available, draws=draws - 1))
        if best > stay:
            gain += available[tile] * (best - stay)
            reach += available[tile]
    survive = (1 - Q) * (1 - (row["winning_copies"] + reach) / unseen)
    return value + (1 - Q) * gain / unseen * repeat(survive, draws - 1)


def potential(hand, available):
    """Live tiles that pair with or sit within two of each kept tile."""
    total = 0
    for tile in hand:
        if tile >= 27:
            total += available[tile]
            continue
        low, high = tile - tile % 9, tile - tile % 9 + 8
        total += sum(available[t] for t in range(max(low, tile - 2), min(high, tile + 2) + 1))
    return total


def best_discard_value(hand, opened, available, draws=LONG):
    return max(
        waiting_value(tuple(sorted(rest)), opened, available, draws=draws)
        for rest in (without(hand, tile) for tile in set(hand))
    )


def without(hand, tile):
    rest = list(hand)
    rest.remove(tile)
    return rest


class Race:
    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass

    def piao(self, obs, opened):
        return can_discard_piao(
            obs.hand,
            obs.drawn_tile,
            opened,
            obs.public_events,
            obs.stage.seat,
            obs.stage.round_no,
            catch_exempt=obs.catch_play and not obs.catch_restricted,
        )

    def value(self, obs, action, opened, available):
        hand = list(obs.hand)
        # My draws left before the wall is down to its last 20 tiles.
        draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
        if action.action == "hu":
            return obs.hu_fan + SAVED
        if action.action == "pass":
            # Passing lets everyone between the discarder and me draw first.
            discarder = next(
                e.seat for e in reversed(obs.public_events) if e.kind == "discard"
            )
            ahead = (obs.stage.seat - discarder - 1) % 4
            return (1 - DRAW_RISK) ** ahead * waiting_value(
                tuple(hand), opened, available, obs.chain_count, obs.piao_count, draws
            )
        if action.action == "discard":
            rest = tuple(sorted(without(hand, action.tile)))
            if action.tile == WHITE and self.piao(obs, opened):
                chain, piao = obs.chain_count + 1, obs.piao_count + 1
            else:
                chain, piao = 0, 0
            if chain:
                value = waiting_value(rest, opened, available, chain, piao, draws)
            else:
                value = reshaped_value(rest, opened, available, draws)
            return (1 - Q) * value + TIE * potential(rest, available)
        if action.action in ("peng", "chi"):
            used = action.tiles if action.action == "chi" else (action.tile,) * 2
            for tile in used:
                hand.remove(tile)
            value = best_discard_value(hand, opened + 1, available, draws)
            # Called from a baotou wait, a white discard that keeps baotou
            # scores piao without waiting for my own draw.
            if WHITE in hand and is_baotou(obs.hand, opened):
                rest = tuple(sorted(without(hand, WHITE)))
                if is_baotou(rest, opened + 1):
                    owner = obs.god_discarder_seat == obs.stage.seat
                    chain = (obs.chain_count if owner else 0) + 1
                    piao = (obs.piao_count if owner else 0) + 1
                    value = max(
                        value,
                        waiting_value(rest, opened + 1, available, chain, piao, draws),
                    )
            return (1 - Q) * value
        if action.action == "gang":
            for _ in range(KONG_TILES[action.kind]):
                hand.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            rest = tuple(sorted(hand))
            row = analyze(rest, after, available, obs.chain_count + 1, obs.piao_count)
            if row["shanten"] > 0:
                return waiting_value(rest, after, available, draws=draws)
            # The replacement draw comes at once and carries the kong's doubling.
            unseen = max(1, sum(available))
            copies = row["winning_copies"]
            now = (row["fan_value"] + SAVED * copies) / unseen
            miss = 1 - copies / unseen
            return now + miss * (1 - Q) * waiting_value(
                rest, after, available, draws=draws
            )
        return float("-inf")

    def certain_double(self, obs, opened):
        """A chain-extending action that keeps every next draw a win."""
        hand = list(obs.hand)
        for action in obs.legal_actions:
            if action.action != "gang":
                continue
            rest = list(hand)
            for _ in range(KONG_TILES[action.kind]):
                rest.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            if is_baotou(tuple(sorted(rest)), after):
                return action
        for action in obs.legal_actions:
            if action.action == "discard" and action.tile == WHITE:
                if self.piao(obs, opened):
                    return action
        return None

    def act(self, obs):
        available = live_tiles(obs, remaining_tiles(obs))
        opened = len(obs.melds[obs.stage.seat])
        if any(a.action == "hu" for a in obs.legal_actions):
            double = self.certain_double(obs, opened)
            if double is not None:
                return double
        return max(obs.legal_actions, key=lambda a: self.value(obs, a, opened, available))


def build(model_dir):
    return Race()
