"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Between two of my draws the three
opponents win with chance Q. A waiting hand that wins on a fraction p of draws,
collecting e fan per draw on average, is worth e / (1 - (1 - p)(1 - Q)) just
before my draw, where Q for a waiting hand rises as the wall runs down (the
others close in; measured waits are worth more early and less late than a
fixed Q says); a wait holding a white also counts the draws that let a discard
turn it into baotou, when that beats winning on them at once; a wait without
one counts drawing a white that way, and the kong a concealed triplet's last
copy gives, whose replacement draw wins at double fan. One step from waiting, each improving draw leads to the best
waiting hand it allows; further away a generic discount per step stands in.
A win also cancels the payment I would owe to someone else's win, SAVED fan.
All of this counts only the draws I have left before the wall runs out.
Passing on a discard lets the opponents between its discarder and me draw
before my turn, so a call or kong also takes their draws away. A wait my
discard leaves also counts its non-winning draws that a later discard turns
into a better wait; a discard leaving a hand one step away counts the same
option for the waits it leads to, pricing each wait there as a plain race
(only when discarding, to keep call and pass replies fast). Discards of
equal value keep the hand whose tiles connect to more live tiles. When the hand is already baotou (every draw wins), a win now is traded for a
doubling that keeps the certain win: a kong whose replacement draw still wins,
else a white discard that scores piao. A call from a baotou wait is priced with
the piao its white discard scores when the wait stays baotou. At a win, giving
it up for a bigger one is priced with the opponents' own chance to win before
my next draw, read from their melds and recent discards, in place of the
generic one.
"""
COMBINATION = "firstprinciples-rule"

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai import _fast
from mahjong_ai.domain.rules import WHITE, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles

Q = 0.15  # an opponent self-draws during the three draws between two of mine
DRAW_RISK = 1 - (1 - Q) ** (1 / 3)  # the same, for one opponent's draw
SAVED = 0.467  # my win cancels the payment to another winner, in fan units
ONE_STEP = 0.22  # typical value of a hand one step from waiting
FAR_STEP = 0.45  # discount for each further step
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}
LAST_WALL = 20  # the hand is drawn once the wall is down to this
Q_EARLY = 0.10
Q_LATE = 0.30
OPTION_REACH = 1.35  # the largest measured ratio was 1.30
TIE = 1e-7  # weight of shape potential, which only separates equal values
# Measured against the fixed pool: the chance one opponent self-draws before my
# next draw, by stage (my draws left 13+, 9-12, fewer), its melds (0-3+) and how
# many of its last three discards were middle suit tiles (first entry: fewer
# than three discards yet; then 0-3).
OPPONENT_HAZARD = (
    ((0.003, 0.004, 0.017, 0.030, 0.020), (0.014, 0.009, 0.033, 0.069, 0.055), (0.042, 0.036, 0.059, 0.059, 0.050), (0.075, 0.065, 0.119, 0.047, 0.070)),
    ((0.024, 0.013, 0.033, 0.044, 0.074), (0.046, 0.029, 0.052, 0.071, 0.082), (0.074, 0.049, 0.074, 0.091, 0.129), (0.131, 0.101, 0.133, 0.148, 0.159)),
    ((0.060, 0.048, 0.060, 0.074, 0.058), (0.083, 0.069, 0.084, 0.093, 0.089), (0.105, 0.093, 0.106, 0.109, 0.121), (0.164, 0.133, 0.161, 0.173, 0.225)),
)


LONG = 16  # more draws of mine than any hand holds


def tenpai_hazard(draws, speed=1.0):
    """Opponents close in as the wall runs down: the race gets harder late."""
    return min(Q_LATE, max(Q_EARLY, 0.35 - 0.018 * draws)) * speed


def opponent_speed(obs, draws):
    """The opponents' chance to win before my next draw, read from their melds
    and recent discards, as a multiple of the generic tenpai_hazard, in steps
    of 0.05 within 0.1-2."""
    stage = 0 if draws >= 13 else (1 if draws >= 9 else 2)
    stay = 1.0
    for seat in range(4):
        if seat == obs.stage.seat:
            continue
        last = obs.discards[seat][-3:]
        middle = sum(1 for t in last if t < 27 and 2 <= t % 9 <= 6) if len(last) == 3 else -1
        stay *= 1 - OPPONENT_HAZARD[stage][min(3, len(obs.melds[seat]))][middle + 1]
    ratio = (1 - stay) / tenpai_hazard(draws)
    return min(2.0, max(0.1, round(ratio * 20) / 20))


# The valuation above, run by the SDK's C race valuation with this line's
# constants: completed waits and a guard for no draws left switched on.
PROFILE = _fast.race_profile(
    Q, SAVED, ONE_STEP, FAR_STEP, Q_EARLY, Q_LATE, 1.0, True, DRAW_RISK, TIE, LAST_WALL, 0, True, True
)


def waiting_value(hand, opened, available, chain=0, piao=0, draws=LONG, options=False, speed=1.0):
    """Expected fan of a 13-tile hand just before my next draw; speed scales
    the opponents' chance to win first."""
    return _fast.race_waiting(hand, opened, available, chain, piao, draws, PROFILE, analyze, options, speed)


def reshaped_value(hand, opened, available, draws, options=True, speed=1.0):
    """A waiting hand with the option to trade a non-winning draw for a better wait."""
    return _fast.race_reshaped(hand, opened, available, draws, PROFILE, analyze, options, speed)



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

    def value(self, obs, action, opened, available, options=True, speed=1.0):
        hand = list(obs.hand)
        # My draws left before the wall is down to its last 20 tiles.
        draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
        if action.action == "hu":
            return obs.hu_fan + SAVED
        if action.action == "pass":
            # Passing lets everyone between the discarder and me draw first.
            discarder = next(
                (e.seat for e in reversed(obs.public_events) if e.kind == "discard"),
                None,
            )
            ahead = 0 if discarder is None else (obs.stage.seat - discarder - 1) % 4
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
                value = waiting_value(rest, opened, available, chain, piao, draws, speed=speed)
            else:
                value = reshaped_value(rest, opened, available, draws, options, speed)
            return (1 - Q * speed) * value + TIE * potential(rest, available)
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
                return waiting_value(rest, after, available, draws=draws, speed=speed)
            # The replacement draw comes at once and carries the kong's doubling.
            unseen = max(1, sum(available))
            copies = row["winning_copies"]
            now = (row["fan_value"] + SAVED * copies) / unseen
            miss = 1 - copies / unseen
            return now + miss * (1 - Q * speed) * waiting_value(
                rest, after, available, draws=draws, speed=speed
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
        available = remaining_tiles(obs)
        opened = len(obs.melds[obs.stage.seat])
        speed = 1.0
        if any(a.action == "hu" for a in obs.legal_actions):
            double = self.certain_double(obs, opened)
            if double is not None:
                return double
            # Whether to give up this win for a bigger one depends on how
            # soon the opponents are likely to win: read it from their table.
            draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
            speed = opponent_speed(obs, draws)
        # The trade option one step away adds at most about a quarter to a
        # hand's value, so only discards that close to the best need it.
        values = [self.value(obs, a, opened, available, False, speed) for a in obs.legal_actions]
        top = max(values)
        for i, a in enumerate(obs.legal_actions):
            if a.action == "discard" and values[i] * OPTION_REACH >= top:
                values[i] = self.value(obs, a, opened, available, True, speed)
        return obs.legal_actions[max(range(len(values)), key=values.__getitem__)]


def build(model_dir):
    return Race()
