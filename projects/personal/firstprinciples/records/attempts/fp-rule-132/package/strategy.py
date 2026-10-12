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
equal value keep the hand whose tiles connect to more live tiles. A discard
leaving a hand one step away also counts the draws that do not improve it
but, kept in place of another tile, leave a better hand one step away (the
best such trade by a quick proxy, priced by the race; native.c). When the hand is already baotou (every draw wins), a win now is traded for a
doubling that keeps the certain win: a kong whose replacement draw still wins,
else a white discard that scores piao. A call from a baotou wait is priced with
the piao its white discard scores when the wait stays baotou; fishing for
piao at a win counts these calls at the measured feed rates, and a piao
that leaves a spare white fishes on. At a win holding a white,
the gap between going on and winning now takes a correction fitted to
true-state rollouts of such windows. At a win, giving
it up for a bigger one is priced with the opponents' own chance to win before
my next draw, read from their melds and recent discards, in place of the
generic one. A hand not yet waiting also counts the calls it can make (the
race counts only draws): each tile a pair of mine can peng, or two neighbours
of mine can chi from my left, comes at the rate the opponents were measured
to discard it, and is taken when the best hand the call keeps beats staying,
both with the shape correction; this enumeration is in the line's own C,
native.c. A kong's hand is valued as a discard's is.
"""
COMBINATION = "firstprinciples-rule"

import math
from functools import lru_cache

from mahjong_ai.domain.observations import can_discard_piao
from mahjong_ai import _fast
from mahjong_ai.domain.rules import WHITE, is_baotou
from mahjong_ai.sdk.shape import analyze, remaining_tiles

Q = 0.15  # an opponent self-draws during the three draws between two of mine
DRAW_RISK = 1 - (1 - Q) ** (1 / 3)  # the same, for one opponent's draw
SAVED = 0.467  # my win cancels the payment to another winner, in fan units (set per hand below)
# What my win is worth beyond its fan, against another player's win, in fan
# units of my own win: the payment it saves (the pool's winners average 1.52
# fan; as dealer I pay 8 to any winner, otherwise 8 to the dealer and 1 to the
# others, and the dealer wins a little more often), plus the next hand as
# dealer, which the winner takes (measured: a hand as dealer is worth 1.0
# point more than one as another player), except in the game's last hand.
# By (dealer, last hand).
# My win also hands me the deal next hand (or keeps it), worth DEALER_EDGE
# points a hand over sitting out of it, again each time the dealer wins (KEEP),
# while hands remain: on top of the payment my win cancels (PAYMENT), in fan
# units of my own win (24 a fan as dealer, 10 otherwise); by (dealer, hands
# left after this one, up to 3).
DEALER_EDGE = 1.03
KEEP = 0.27
PAYMENT = {True: 0.499, False: 0.530}
SAVED_BY_ROLE = {
    (dealer, left): round(PAYMENT[dealer] + DEALER_EDGE * (1 - KEEP ** left) / (1 - KEEP) / (24 if dealer else 10), 3)
    for dealer in (True, False) for left in range(4)
}
ONE_STEP = 0.22  # typical value of a hand one step from waiting
FAR_STEP = 0.45  # discount for each further step
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}
LAST_WALL = 20  # the hand is drawn once the wall is down to this
Q_EARLY = 0.10
Q_LATE = 0.30
OPTION_REACH = 1.35  # the largest measured ratio was 1.30
TIE = 1e-7  # weight of shape potential, which only separates equal values
# Measured against the fixed pool (logistic fit over 200k opponent-rounds): the
# log-odds that one opponent self-draws before my next draw, weighting in order
# 1, my draws left d, d*d/16, its melds being 1 / 2 / 3+, melds*d, middle suit
# tiles among its last three discards, whether it has three yet, middle tiles
# among its last six, honors and terminals among its last three, its discards
# /10, its discards since its last call /9 (9 without a call), whether it
# deals, and whether it has discarded a white (a piao: it waits on baotou).
OPPONENT_LOGIT = (-0.0571, -0.1131, -0.2408, -0.5687, -0.2777, 0.1483, 0.0804, 0.0021, 1.0988, 0.1779, -0.4002, -0.1582, -1.9073, -0.6204, 0.1573, 5.1224)
SHAPE_PAIR = 0.009
SHAPE_ADJACENT = 0.0114
SHAPE_UKEIRE = 0.0078
SHAPE_DISTANCE = -0.045
SHAPE_OPEN = 0.0105
SHAPE_EDGE = -0.0055
CALL_WEIGHT = 0.7  # the share of the call option measured best against the pool
# Opening a closed hand far from waiting with a chi gives up more flexibility
# (seven pairs, the shapes a full hand still allows) than the race sees: by the
# kept hand's steps from waiting: one, and two or more.
OPEN_CHI = (0.10, 0.03)
# A closed hand of many pairs holds on to seven pairs more than it should: a
# peng there is worth more than the race sees, for each pair beyond two, by the
# hand's steps from waiting (one, two or more).
PAIR_PENG = (0.06, 0.02)
# Holding a white, a peng is worth more than the race sees (the set it completes
# brings baotou nearer): by the hand's steps from waiting before the peng
# (waiting, one, two or more).
WHITE_PENG = (0.06, 0.03, 0.02)
# When seven pairs is the hand's shortest route, a peng that sets it one step
# further back on the usual route is still worth more than the race sees.
SEVEN_PAIR_PENG = 0.05
FISH_DEPTH = 8  # draws a baotou wait may let go while fishing for piao
# Margins (going on less winning now, with CONTINUE_SHIFT) within this reach
# are decided by the fitted correction (native.c: fish_gbm).
FISH_REACH = 0.8
MIMIC = 0.5
CALL_MIMIC = 0.75
LONG = 16  # more draws of mine than any hand holds
# At a win holding a white, what simulation says the valuation misses in the
# gap between going on and winning now (ridge fit over 7916 such windows,
# 12 true-state rollouts each): constant, one white not yet baotou, my draws
# left /16, 12 draws or more, the opponents' speed, a win worth two fan or
# more, two melds or more.
CONTINUE_SHIFT = (-0.417, -0.169, 0.833, -0.182, 0.112, -0.107, -0.083)


# Distilled from RL fp-rl-158's discards (conditional logits over 300,328 of
# this line's own discard decisions, one per best reachable steps: 0, 1, 2, 3+):
# next to this line's value (first weight), the weights of the kept hand's shape
# (native.c: mimic_features) and of the discarded tile's kind, alone and with a
# white held, late in the hand and with melds out. A discard's value gets MIMIC
# times their sum, in value units.
MIMIC_WEIGHTS = (
    (11.60959, -2.37706, 0.54165, 3.02301, 6.45068, 0.83092, 2.37499, -1.82188, -0.31721, 1.60978, -0.26522, -0.30616, 1.30004, 1.32844, 1.47932, 0.93873, -2.41092, -1.49276, -0.63173, -1.32423, -1.08089, -5.32974, 0.50247, 1.04156, -0.17379, -0.22393, -0.37183, -0.28825, 0.34643, 0.24656, -0.24844, -0.58589, 0.08021, 1.33038, 0.26947, 0.02289, 1.06146, 0.24708, 1.82644, -0.60965, 1.12643, 0.25244, -0.55970, 0.61236, 0.58522, -0.05715, -0.15485, -1.03858, -0.63118, -0.32315, 0.20999, 0.24894, -0.49440, 0.46285, -0.65955, 0.80287, 1.58215, 1.39995, -0.34406, 0.23627, 0.85692, -0.51704, 0.48636, -0.21646, 0.68793, 0.10534, -0.07496, 0.23263, -0.44902),
    (18.95606, -1.57789, 1.95694, 1.67863, 0.00000, 3.28078, 5.58173, -4.23506, -0.73363, 2.86823, 0.61027, 0.99719, 2.08679, 2.21700, 1.05003, 1.04987, 8.15837, 2.89358, 0.01287, -1.74796, -1.34391, 0.00000, -0.05450, 0.01635, -0.10288, -0.41337, -0.98791, 0.28551, -0.51142, 0.27504, 0.29509, 0.26725, 0.10763, -1.97630, 2.81426, -0.31283, 0.23521, 0.30233, 0.00000, -0.73156, 0.24107, 0.67980, -0.45600, 0.19249, -0.72133, -0.52477, -0.51643, -2.41729, -1.63052, -0.91711, 1.72845, -0.29687, -0.69550, -0.18147, -0.97010, 0.00000, 2.85041, 0.38767, -1.43950, -0.39122, 0.61840, -0.05560, 0.38674, 0.94515, 0.75475, 0.30609, 0.02742, 1.39023, 0.89094),
    (64.68793, -11.20497, 4.57859, -2.81460, 0.00000, 2.80838, 0.64731, -6.60751, -2.00536, 0.21165, 3.75427, -0.16494, 4.18181, 4.01300, 2.05964, 1.52188, 0.00000, -1.59283, -0.87987, 1.74519, -0.84687, 0.00000, 1.99376, 0.04247, 0.05128, -0.57817, -0.94705, 0.10619, -0.79650, -0.05133, 0.13560, 0.06968, 0.20031, 0.00000, -0.76698, -0.00031, 0.06764, 0.13528, 0.00000, -0.14020, 0.01459, -0.01115, -0.18717, 0.61578, 0.55046, -0.44213, 0.03180, -1.31079, -0.67389, -1.55012, 0.00000, -0.02213, -2.46187, 2.21667, -0.70232, 0.00000, 1.60599, 0.01522, 0.72826, 0.02565, -1.05767, -0.28500, -0.45579, -0.15438, 0.17145, 0.17403, -0.15807, 0.00000, 0.24011),
    (229.64373, -0.11553, 2.06602, -0.76374, 0.00000, -0.25169, 0.38625, -5.37998, -1.91949, 1.37174, 0.70990, 0.10636, 5.46723, 4.30291, 1.91762, 3.19659, 0.00000, 0.00000, 0.38337, 1.96433, 0.56018, 0.00000, -0.32337, 0.64073, -0.23089, 0.02408, -0.12162, -0.12035, -0.26079, 0.19235, 0.34286, -0.57963, 0.37190, 0.00000, 0.00000, 0.00027, 0.11127, 0.05307, 0.00000, 0.00125, -0.00000, 0.01718, -0.11343, 0.12329, 0.12364, -0.08064, -0.01718, -0.56724, 0.33364, 0.06116, 0.00000, 0.00000, 0.01984, 0.17652, -0.45175, 0.00000, 0.01440, -0.05562, 0.45091, -0.19574, -0.92461, -0.92354, -0.06972, -0.45082, 0.06626, -0.25039, -1.47264, 0.00000, 0.00000),
)


def mimic_shifts(obs, tiles, opened, available):
    """The distilled correction of each discard, by the weights of the decision's level."""
    held = _fast.counts(tuple(obs.hand))
    w = 1.0 if held[WHITE] else 0.0
    late = max(0.0, (60 - obs.wall_remaining) / 40)
    op = min(opened, 2) / 2
    rows = [list(native.mimic_features(held, opened, available, tile)) for tile in tiles]
    weights = MIMIC_WEIGHTS[min(3, int(min(r[0] for r in rows)))]
    out = []
    for base in rows:
        x = base + [f * w for f in base] + [f * late for f in base] + [f * op for f in base]
        out.append(MIMIC * sum(b * f for b, f in zip(weights[1:], x)) / (10 * weights[0]))
    return out


# Distilled from RL fp-rl-158's responses: a small network (native.c: call_net)
# fit to its chi/peng/pass choices over this line's own response windows reads
# each option's hand next to this line's value; its weight on the value (x10) is
# CALL_VALUE. An option's value gets CALL_MIMIC times its score, in value units.
CALL_VALUE = 2.646332


def call_shifts(obs, opened, available):
    """The distilled correction of each response option (pass, chi, peng)."""
    held = _fast.counts(tuple(obs.hand))
    discarder = next((e.seat for e in reversed(obs.public_events) if e.kind == "discard"), obs.stage.seat)
    options = [(0, -1, -1, -1) if a.action == "pass" else (1, *a.tiles, a.tile) if a.action == "chi" else (2, a.tile, a.tile, a.tile)
               for a in obs.legal_actions]
    scores = native.call_net(held, opened, available, options, obs.wall_remaining,
                             obs.stage.dealer == obs.stage.seat, (obs.stage.seat - discarder - 1) % 4)
    return [CALL_MIMIC * s / (10 * CALL_VALUE) for s in scores]


def kong_features(obs, counts, opened, available, kong, keep, v_kong, v_keep):
    """The kong-or-keep model's features: whites, the kong's kind, the wall,
    melds, seat, both values and their gap, and for the hand kept by each choice
    its steps, winning and improving shares of the unseen tiles, baotou and the
    value-weighted winning share; chain, an honor kong and chis."""
    seat = obs.stage.seat
    unseen = max(1, sum(available))
    kept = list(counts)
    kept[keep.tile] -= 1
    row_keep = _fast.analyze_counts(list(kept), opened, available, 0, 0, 0)
    after = opened if kong.kind == "added" else opened + 1
    melded = list(counts)
    for _ in range(KONG_TILES[kong.kind]):
        melded[kong.tile] -= 1
    row_kong = _fast.analyze_counts(list(melded), after, available, 0, 0, 0)
    return [counts[WHITE], {"added": 0, "concealed": 1, "exposed": 2}[kong.kind], obs.wall_remaining, opened,
            float(obs.stage.dealer == seat), v_kong - v_keep, v_kong, v_keep,
            row_keep[0], sum(available[t] for t in row_keep[2]) / unseen, row_keep[4] / unseen, float(_fast.baotou(list(kept), opened)),
            row_kong[0], sum(available[t] for t in row_kong[2]) / unseen, row_kong[4] / unseen, float(_fast.baotou(list(melded), after)),
            obs.chain_count, row_keep[6] / unseen, row_kong[6] / unseen, float(kong.tile >= 27),
            sum(1 for m in obs.melds[seat] if m.kind == "chi")]


def fish_features(obs, opened, available, speed, cont, margin, v_hu, v_cont):
    """The fitted correction's features at a win holding a white: the margin, the
    window's kind (fishing with two whites or more, baotou with one, or not yet
    baotou), whites, my draws left, the opponents' speed, the fan now, melds,
    chain, seat, and for going on by a discard the kept hand's winning and
    improving shares of the unseen tiles, the shares of draws that score piao,
    a kong or an upgrade to baotou, the chance to piao by a call, both values
    and the wall."""
    seat = obs.stage.seat
    whites = obs.hand.count(WHITE)
    baotou = is_baotou(tuple(sorted(without(obs.hand, obs.drawn_tile))), opened)
    kind = 0 if baotou and whites >= 2 else 1 if baotou else 2
    draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
    win = uke = piao = kong = upgrade = callp = -1.0
    if cont.action == "discard":
        rest = without(list(obs.hand), cont.tile)
        c = list(_fast.counts(tuple(rest)))
        if sum(c) == 13 - 3 * opened:
            row = _fast.analyze_counts(list(c), opened, available, 0, 0, 0)
            unseen = max(1, sum(available))
            win = sum(available[t] for t in row[2]) / unseen
            uke = row[4] / unseen
            bt = _fast.baotou(list(c), opened)
            pi = kg = up = 0
            pungs = [m.tiles[0] for m in obs.melds[seat] if m.kind == "peng"]
            for t in range(34):
                if not available[t] or c[t] >= 4:
                    continue
                c[t] += 1
                if bt and c[WHITE] >= 2:
                    c[WHITE] -= 1
                    pi += available[t] * _fast.baotou(list(c), opened)
                    c[WHITE] += 1
                if bt and (c[t] == 4 or t in pungs):
                    kg += available[t]
                if not bt and c[WHITE]:
                    ok = False
                    for x in range(34):
                        if c[x] and x != t:
                            c[x] -= 1
                            ok = ok or _fast.baotou(list(c), opened)
                            c[x] += 1
                        if ok:
                            break
                    up += available[t] * ok
                c[t] -= 1
            piao, kong, upgrade = pi / unseen, kg / unseen, up / unseen
            stage = 0 if obs.wall_remaining >= 70 else 1 if obs.wall_remaining >= 45 else 2
            chis = sum(1 for m in obs.melds[seat] if m.kind == "chi")
            calls = piao_calls(tuple(sorted(rest)), opened, available, 0, 0, draws, speed, tuple(pungs), stage, chis) if bt and c[WHITE] >= 2 else ()
            callp = sum(p for p, _ in calls)
    return [margin, kind, whites, min(16, draws), speed, obs.hu_fan, opened, obs.chain_count, float(obs.stage.dealer == seat),
            win, uke, piao, kong, upgrade, callp, v_hu, v_cont, obs.wall_remaining]


def tenpai_hazard(draws, speed=1.0):
    """Opponents close in as the wall runs down: the race gets harder late."""
    return min(Q_LATE, max(Q_EARLY, 0.35 - 0.018 * draws)) * speed


def opponent_speed(obs, draws):
    """The opponents' chance to win before my next draw, read from their melds
    and discards, as a multiple of the generic tenpai_hazard, in steps of 0.05
    within 0.1-3."""
    hand_no = obs.public_events[-1].hand_no if obs.public_events else 0
    since = {}
    for event in obs.public_events:
        if event.hand_no != hand_no:
            continue
        if event.kind in ("chi", "peng", "gang"):
            since[event.seat] = 0
        elif event.kind == "discard" and event.seat in since:
            since[event.seat] += 1
    d = min(draws, 16)
    stay = 1.0
    for seat in range(4):
        if seat == obs.stage.seat:
            continue
        discards = obs.discards[seat]
        last3, last6 = discards[-3:], discards[-6:]
        melds = min(3, len(obs.melds[seat]))
        features = (
            1, d, d * d / 16, melds == 1, melds == 2, melds == 3, (melds >= 1) * d,
            sum(1 for t in last3 if t < 27 and 2 <= t % 9 <= 6), len(last3) == 3,
            sum(1 for t in last6 if t < 27 and 2 <= t % 9 <= 6),
            sum(1 for t in last3 if t >= 27), sum(1 for t in last3 if t < 27 and t % 9 in (0, 8)),
            len(discards) / 10, min(since.get(seat, 9), 9) / 9, obs.stage.dealer == seat, WHITE in discards,
        )
        logit = sum(w * x for w, x in zip(OPPONENT_LOGIT, features))
        stay *= 1 - 1 / (1 + math.exp(-logit))
    ratio = (1 - stay) / tenpai_hazard(draws)
    return min(3.0, max(0.1, round(ratio * 20) / 20))


# The valuation above, run by the SDK's C race valuation with this line's
# constants: completed waits and a guard for no draws left switched on.
PROFILES = {
    saved: _fast.race_profile(Q, saved, ONE_STEP, FAR_STEP, Q_EARLY, Q_LATE, 1.0, True, DRAW_RISK, TIE, LAST_WALL, 0, True, True)
    for saved in (SAVED, *SAVED_BY_ROLE.values())
}
PROFILE = PROFILES[SAVED]
# This line's own C (native.c) keeps its own table of race profiles.
native = _fast.load_native(__file__)
NATIVE_PROFILES = {
    saved: native.race_profile(Q, saved, ONE_STEP, FAR_STEP, Q_EARLY, Q_LATE, 1.0, True, DRAW_RISK, TIE, LAST_WALL, 0, True, True)
    for saved in (SAVED, *SAVED_BY_ROLE.values())
}
NATIVE_PROFILE = NATIVE_PROFILES[SAVED]
# The call option weighs hands with the same shape correction the decisions use.
native.shape_constants(SHAPE_PAIR, SHAPE_ADJACENT, SHAPE_UKEIRE, SHAPE_DISTANCE, SHAPE_OPEN, SHAPE_EDGE)


def waiting_value(hand, opened, available, chain=0, piao=0, draws=LONG, options=False, speed=1.0):
    """Expected fan of a 13-tile hand just before my next draw; speed scales
    the opponents' chance to win first."""
    return _fast.race_waiting(hand, opened, available, chain, piao, draws, PROFILE, analyze, options, speed)


def reshaped_value(hand, opened, available, draws, options=True, speed=1.0):
    """A waiting hand with the option to trade a non-winning draw for a better
    wait; with options, a hand one step away also counts trading a draw that
    does not improve it for a better hand one step away (native.c)."""
    value = _fast.race_reshaped(hand, opened, available, draws, PROFILE, analyze, options, speed)
    if options:
        value += native.reshape_gain(hand, opened, available, draws, NATIVE_PROFILE, analyze, Q, speed)
    return value



@lru_cache(maxsize=2000)
def piao_value(hand, opened, available, chain, piao, draws, speed=1.0, fish=False, pungs=(), stage=1, chis=0, saved=SAVED):
    """A baotou wait holding a spare white: each draw either wins now or, when
    a white discard keeps baotou, scores piao and waits on at double fan. At a
    win (fish) a kong that keeps baotou doubles too, an opponent's discard I
    can chi or peng keeping baotou scores piao without a draw of mine, and a
    draw doing neither may be let go to keep fishing: an optimal stop over the
    draws left, each round risking the tenpai hazard. A piao that leaves a
    spare white fishes on."""
    unseen = max(1, sum(available))
    held = list(_fast.counts(hand))
    q = tenpai_hazard(draws, speed) if fish else Q * speed
    total = 0.0
    options = []
    for tile in range(34):
        if available[tile] <= 0 or held[tile] >= 4:
            continue
        kong = fish and (held[tile] == 3 or tile in pungs)
        held[tile] += 1
        now = _fast.fan_value(held, tile, opened, chain, piao, False, False) + saved
        best = now
        if kong:
            # A kong keeping baotou: its replacement draw wins at double fan.
            after = list(held)
            after[tile] -= 4 if held[tile] == 4 and tile not in pungs else 1
            if _fast.baotou(after, opened + (0 if tile in pungs else 1)):
                best = max(best, 2 * (now - saved) + saved)
        if held[WHITE] >= 2:
            held[WHITE] -= 1
            if _fast.baotou(held, opened):
                rest = sorted((*hand, tile))
                rest.remove(WHITE)
                rest = tuple(rest)
                later = _fast.race_waiting(rest, opened, available, chain + 1, piao + 1, draws - 1, PROFILES[saved], analyze, False, speed)
                if held[WHITE] >= 2 and draws > 2:
                    later = max(later, piao_value(rest, opened, available, chain + 1, piao + 1, draws - 1, speed, fish, pungs, stage, chis, saved))
                best = max(now, (1 - q) * later)
            held[WHITE] += 1
        held[tile] -= 1
        options.append((available[tile], now, best))
        total += available[tile] * best
    calls = piao_calls(hand, opened, available, chain, piao, draws, speed, pungs, stage, chis, saved) if fish else ()

    def called(value):
        """The value after my discard: a call that scores piao is taken."""
        return value + sum(chance * max(0.0, worth - value) for chance, worth in calls)

    # A draw that neither wins big nor scores piao can be let go too, keeping
    # the baotou wait for the next draw: an optimal stop over the draws left.
    value = total / unseen
    for _ in range(min(draws - 1, FISH_DEPTH) if fish else 0):
        again = (1 - q) * called(value)
        value = sum(n * max(best, again) for n, now, best in options) / unseen
    return called(value)


def piao_calls(hand, opened, available, chain, piao, draws, speed, pungs, stage, chis, saved=SAVED):
    """The opponents' discards a baotou wait with a spare white can chi or peng
    and stay baotou once the white goes (a piao): each tile's chance per round
    at its measured feed rate, and the wait it leaves."""
    if opened >= 4:
        return ()
    held = _fast.counts(hand)
    unseen = max(1, sum(available))
    calls = []
    for tile in range(WHITE):
        if available[tile] <= 0:
            continue
        uses = []
        if held[tile] >= 2:
            uses.append(((tile, tile), native.feed_rate(stage, tile, False), (*pungs, tile), chis))
        if tile < 27 and chis < 2:
            for a, b in ((tile - 2, tile - 1), (tile - 1, tile + 1), (tile + 1, tile + 2)):
                if a >= 0 and b < 27 and a // 9 == tile // 9 == b // 9 and held[a] and held[b]:
                    uses.append(((a, b), native.feed_rate(stage, tile, True), pungs, chis + 1))
        best = None
        for used, rate, kept, chi_count in uses:
            rest = list(hand)
            for t in (*used, WHITE):
                rest.remove(t)
            rest = tuple(rest)
            if not _fast.baotou(list(_fast.counts(rest)), opened + 1):
                continue
            worth = _fast.race_waiting(rest, opened + 1, available, chain + 1, piao + 1, draws, PROFILES[saved], analyze, False, speed)
            if rest.count(WHITE) >= 2 and draws > 1:
                worth = max(worth, piao_value(rest, opened + 1, available, chain + 1, piao + 1, draws, speed, True, kept, stage, chi_count, saved))
            if best is None or worth > best[1]:
                best = (rate * available[tile] / unseen, worth)
        if best is not None:
            calls.append(best)
    return tuple(calls)


def piao_gain(hand, opened, available, draws, speed, pungs=(), stage=1, chis=0):
    """What the piao option adds to a wait holding a white, over the C race:
    the white upgrade's baotou targets priced with piao_value when they keep
    a spare white (only at a win, where this trade is made)."""
    if WHITE not in hand or draws < 2:
        return 0.0
    if is_baotou(hand, opened):
        if hand.count(WHITE) < 2:
            return 0.0
        return piao_value(hand, opened, available, 0, 0, draws, speed, True, pungs, stage, chis, SAVED) - waiting_value(hand, opened, available, draws=draws, speed=speed)
    held = list(_fast.counts(hand))
    row = _fast.analyze_counts(held, opened, available, 0, 0, 0)
    if row[0] > 0:
        return 0.0
    unseen = max(1, sum(available))
    q = Q * speed
    fans = dict(zip(row[2], row[3]))
    kinds = [t for t in range(34) if held[t]]
    plain = richer = 0.0
    reach = 0
    for tile in range(34):
        if available[tile] <= 0 or held[tile] >= 4:
            continue
        held[tile] += 1
        best = best_p = 0.0
        for drop in kinds:
            if drop == tile or not held[drop]:
                continue
            held[drop] -= 1
            if _fast.baotou(held, opened):
                rest = sorted((*hand, tile))
                rest.remove(drop)
                rest = tuple(rest)
                v = waiting_value(rest, opened, available, draws=draws - 1, speed=speed)
                vp = piao_value(rest, opened, available, 0, 0, draws - 1, speed, saved=SAVED) if rest.count(WHITE) >= 2 else v
                best, best_p = max(best, v), max(best_p, vp)
            held[drop] += 1
        held[tile] -= 1
        if tile in fans:
            plain += available[tile] * max(fans[tile] + SAVED, (1 - q) * best)
            richer += available[tile] * max(fans[tile] + SAVED, (1 - q) * best_p)
        elif best:
            plain += available[tile] * (1 - q) * best
            richer += available[tile] * (1 - q) * best_p
            reach += available[tile]
    stay = (1 - tenpai_hazard(draws, speed)) * (1 - (row[5] + reach) / unseen)
    return (richer - plain) / unseen * (1 - stay ** draws) / (1 - stay)


@lru_cache(maxsize=4096)
def call_option(hand, opened, available, draws, stage, chis, saved=SAVED):
    """What a hand not yet waiting gains from the calls it can make: each tile a
    pair of mine can peng, or two neighbours of mine can chi from my left, comes
    at its measured rate per round (native.c), and is taken when the best hand
    the call keeps beats staying as I am, over the rounds the hand stays as it is."""
    return native.call_option(hand, opened, available, draws, stage, chis, NATIVE_PROFILES[saved], analyze, Q, CALL_WEIGHT)


def shape_bonus(rest, opened, available, full):
    """What simulation says the race misses in a kept hand: pairs and
    adjacent pairs of suit tiles are worth more than their draws alone."""
    held = _fast.counts(rest)
    pairs = sum(1 for t in range(33) if held[t] >= 2)
    adjacent = sum(1 for t in range(27) if t % 9 <= 7 and held[t] and held[t + 1])
    edges = sum(1 for t in range(27) if t % 9 in (0, 7) and held[t] and held[t + 1])
    bonus = SHAPE_PAIR * pairs + SHAPE_ADJACENT * adjacent
    # Two-sided pairs more, edge pairs (12, 89) less than adjacency alone.
    bonus += SHAPE_OPEN * (adjacent - edges) + SHAPE_EDGE * edges
    if full:
        row = _fast.analyze_counts(list(held), opened, available, 0, 0, 0)
        bonus += SHAPE_UKEIRE * row[4] / 10 + SHAPE_DISTANCE * row[0]
    return bonus


def potential(hand, available):
    """Live tiles that pair with or sit within two of each kept tile (in C)."""
    return _fast.race_potential(hand, available)


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

    def value(self, obs, action, opened, available, options=True, speed=1.0, at_win=False):
        hand = list(obs.hand)
        # My draws left before the wall is down to its last 20 tiles.
        draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
        stage = 0 if obs.wall_remaining >= 70 else 1 if obs.wall_remaining >= 45 else 2
        chis = sum(1 for m in obs.melds[obs.stage.seat] if m.kind == "chi")
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
            ) + shape_bonus(tuple(sorted(hand)), opened, available, True) + call_option(
                tuple(sorted(hand)), opened, available, draws, stage, chis, SAVED
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
                if at_win:
                    pungs = tuple(m.tiles[0] for m in obs.melds[obs.stage.seat] if m.kind == "peng")
                    value += piao_gain(rest, opened, available, draws, speed, pungs, stage, chis)
            value = (1 - Q * speed) * value + TIE * potential(rest, available)
            if not at_win:
                called = call_option(rest, opened, available, draws, stage, chis, SAVED)
                value += shape_bonus(rest, opened, available, True) + (1 - Q * speed) * called
            return value
        if action.action in ("peng", "chi"):
            used = action.tiles if action.action == "chi" else (action.tile,) * 2
            for tile in used:
                hand.remove(tile)
            rests = [tuple(sorted(without(hand, tile))) for tile in set(hand)]
            kept = max(rests, key=lambda rest: waiting_value(rest, opened + 1, available, draws=draws))
            value = waiting_value(kept, opened + 1, available, draws=draws)
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
            called = call_option(kept, opened + 1, available, draws, stage, chis + (action.action == "chi"), SAVED)
            cost = 0.0
            if action.action == "chi" and opened == 0:
                distance = _fast.shanten_raw(list(_fast.counts(kept)), opened + 1)
                cost = OPEN_CHI[1] if distance >= 2 else OPEN_CHI[0] if distance == 1 else 0.0
            elif action.action == "peng" and opened == 0:
                held = _fast.counts(tuple(obs.hand))
                steps = _fast.shanten_raw(list(held), 0)
                extra = max(0, sum(1 for t in range(33) if held[t] >= 2) - 2)
                cost = -extra * (PAIR_PENG[0] if steps == 1 else PAIR_PENG[1] if steps >= 2 else 0.0)
                if _fast.shanten_raw(list(_fast.counts(kept)), opened + 1) > steps:
                    cost -= SEVEN_PAIR_PENG
            if action.action == "peng" and WHITE in obs.hand:
                steps = _fast.shanten_raw(list(_fast.counts(tuple(obs.hand))), opened)
                cost -= WHITE_PENG[min(max(steps, 0), 2)]
            return (1 - Q) * value + shape_bonus(kept, opened + 1, available, True) + (1 - Q) * called - cost
        if action.action == "gang":
            for _ in range(KONG_TILES[action.kind]):
                hand.remove(action.tile)
            after = opened if action.kind == "added" else opened + 1
            rest = tuple(sorted(hand))
            row = analyze(rest, after, available, obs.chain_count + 1, obs.piao_count)
            shape = shape_bonus(rest, after, available, True)
            if row["shanten"] > 0:
                return reshaped_value(rest, after, available, draws, options, speed) + shape + call_option(
                    rest, after, available, draws, stage, chis, SAVED
                )
            # The replacement draw comes at once and carries the kong's doubling.
            unseen = max(1, sum(available))
            copies = row["winning_copies"]
            now = (row["fan_value"] + SAVED * copies) / unseen
            miss = 1 - copies / unseen
            return now + miss * (1 - Q * speed) * waiting_value(
                rest, after, available, draws=draws, speed=speed
            ) + shape
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
        global SAVED, PROFILE, NATIVE_PROFILE
        left = min(3, max(0, obs.stage.rounds_total - obs.stage.round_no))
        SAVED = SAVED_BY_ROLE[(obs.stage.dealer == obs.stage.seat, left)]
        PROFILE, NATIVE_PROFILE = PROFILES[SAVED], NATIVE_PROFILES[SAVED]
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
        at_win = any(a.action == "hu" for a in obs.legal_actions)
        values = [self.value(obs, a, opened, available, False, speed, at_win) for a in obs.legal_actions]
        top = max(values)
        for i, a in enumerate(obs.legal_actions):
            if a.action in ("discard", "gang") and values[i] * OPTION_REACH >= top:
                values[i] = self.value(obs, a, opened, available, True, speed, at_win)
        if not at_win and obs.phase == "draw" and not any(a.action == "gang" for a in obs.legal_actions):
            discards = [i for i, a in enumerate(obs.legal_actions) if a.action == "discard"]
            if discards:
                shifts = mimic_shifts(obs, [obs.legal_actions[i].tile for i in discards], opened, available)
                for i, s in zip(discards, shifts):
                    values[i] += s
        if obs.phase.startswith("response") and all(a.action in ("pass", "chi", "peng") for a in obs.legal_actions):
            values = [v + s for v, s in zip(values, call_shifts(obs, opened, available))]
        if at_win and WHITE in obs.hand:
            fitted = self.fish_decision(obs, opened, available, speed)
            if fitted is not None:
                return fitted
            shift = self.continue_shift(obs, opened, speed)
            values = [v if a.action == "hu" else v + shift for a, v in zip(obs.legal_actions, values)]
        choice = obs.legal_actions[max(range(len(values)), key=values.__getitem__)]
        if choice.action == "gang" and not at_win and obs.phase == "draw" and WHITE in obs.hand:
            kept = self.keep_kong(obs, choice, opened, available)
            if kept is not None:
                return kept
        if choice.action == "gang" and not at_win and obs.phase != "draw" and WHITE in obs.hand:
            # Holding a white at a hand waiting or one step away, peng the
            # discard rather than meld all four: the copy kept in hand can be
            # added later, once the wait is baotou, and keep_kong may still add
            # it at once (simulated: better by 0.2 fan with one white).
            peng = next((a for a in obs.legal_actions if a.action == "peng"), None)
            if peng is not None and _fast.shanten_raw(list(_fast.counts(tuple(obs.hand))), opened) <= 1:
                return peng
        return choice

    def keep_kong(self, obs, kong, opened, available):
        """At a hand holding a white that waits or is one step away, keep the
        kong tile (and throw the best other tile) when the fitted gain of keeping
        it is positive: the kong is worth more later, once the wait is baotou.
        None to meld it."""
        counts = _fast.counts(tuple(obs.hand))
        if _fast.shanten_raw(list(counts), opened) > 1:
            return None
        kong_tiles = {a.tile for a in obs.legal_actions if a.action == "gang"}
        keeps = [a for a in obs.legal_actions if a.action == "discard" and a.tile not in kong_tiles]
        if not keeps:
            return None
        values = {a: self.value(obs, a, opened, available, True, 1.0, False) for a in obs.legal_actions}
        keep = max(keeps, key=values.get)
        x = kong_features(obs, counts, opened, available, kong, keep, values[kong], values[keep])
        return keep if native.kong_gbm(x) > 0 else None

    def fish_decision(self, obs, opened, available, speed):
        """At a win holding a white whose margin is within FISH_REACH, win now or
        go on by the margin plus the fitted correction; None otherwise."""
        legal = obs.legal_actions
        values = {a: self.value(obs, a, opened, available, True, speed, True) for a in legal}
        hu = next(a for a in legal if a.action == "hu")
        rest_actions = [a for a in legal if a.action != "hu"]
        if not rest_actions:
            return None
        cont = max(rest_actions, key=values.get)
        margin = values[cont] - values[hu] + self.continue_shift(obs, opened, speed)
        if abs(margin) > FISH_REACH:
            return None
        x = fish_features(obs, opened, available, speed, cont, margin, values[hu], values[cont])
        return cont if margin + native.fish_gbm(x) > 0 else hu

    def continue_shift(self, obs, opened, speed):
        """The simulation-fitted correction to going on at a win holding a white."""
        whites = obs.hand.count(WHITE)
        fishing = is_baotou(tuple(sorted(without(obs.hand, obs.drawn_tile))), opened)
        if fishing and whites < 2:
            return 0.0
        draws = min(16, max(1, -(-(obs.wall_remaining - LAST_WALL) // 4)))
        c, one, per_draw, early, fast, big, melds = CONTINUE_SHIFT
        return (
            c + one * (not fishing and whites == 1) + per_draw * draws / 16
            + early * (draws >= 12) + fast * speed + big * (obs.hu_fan >= 2) + melds * (opened >= 2)
        )


def build(model_dir):
    return Race()
