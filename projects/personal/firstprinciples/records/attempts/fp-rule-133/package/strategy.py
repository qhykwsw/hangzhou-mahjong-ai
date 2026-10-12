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


# Distilled from RL fp-rl-112's discards (conditional logits over 200,328 of
# fp-rule-113's own discard decisions, one per best reachable steps: 0, 1, 2, 3+):
# next to this line's value (first weight), the weights of the kept hand's shape
# (native.c: mimic_features) and of the discarded tile's kind, alone and with a
# white held, late in the hand and with melds out. A discard's value gets MIMIC
# times their sum, in value units.
MIMIC_WEIGHTS = (
    (8.49804, -2.82415, 0.13226, 2.66164, 3.46129, 1.50783, 2.49394, -1.01515, -0.16094, 1.38564, -0.67288, -0.37826, 0.35961, 1.36882, 0.98300, 0.47412, -1.42470, -1.01132, -1.45497, -0.43071, -1.05968, -2.98879, 0.21718, 0.79259, -0.44034, -0.04730, -0.14254, -0.28562, 0.22294, 0.18813, 0.00558, -0.27585, -0.00393, 1.06450, 0.25085, 0.15255, -0.72849, 0.27882, -0.90991, 1.00490, 0.73153, 0.04974, -0.30781, 0.98706, 0.59105, 0.63799, 0.79359, -0.43611, -0.57394, 0.04654, -0.20590, 0.04795, -0.64761, 0.40493, -0.89147, -0.38804, 0.46558, 1.02809, -0.59225, 0.33661, 0.21520, -0.09084, 0.68030, -0.10176, -0.09091, -0.05132, 0.05946, 0.72457, 0.21139),
    (10.97233, -1.06045, 1.59918, 0.72562, 0.00000, 2.84223, 3.61476, -2.99149, -0.35018, 2.40768, -0.31287, 0.63956, 1.06874, 1.59796, 0.58406, 0.54468, 5.95084, 2.07496, 0.43510, 0.68317, -1.47343, 0.00000, -0.06022, 0.23083, -0.21064, -0.18405, -0.26723, -0.09159, -0.23991, -0.17750, 0.26362, 0.23558, 0.17382, -2.58205, 3.10883, -0.16088, -0.48300, 0.48468, 0.00000, -0.29341, 0.18381, 0.21072, -0.78329, 0.84392, -0.75444, -0.00580, -0.63672, -1.08319, -0.91898, -0.64059, 0.25130, -0.63987, 0.48470, 0.61727, -0.57247, 0.00000, 1.78258, -0.76248, 0.42601, -0.42122, -0.07502, 0.19266, 0.24357, 0.62695, 0.25245, 0.25822, -0.02892, -0.45671, 1.24652),
    (38.28590, -11.65169, 4.57407, -3.20465, 0.00000, 3.01443, 1.82548, -3.93259, -1.11880, -0.22298, 1.58098, -0.08617, 2.80829, 2.68201, 1.38190, 0.83202, 0.00000, -1.28703, -2.76498, 2.37974, -1.97332, 0.00000, 1.33193, 0.07615, 0.18513, 0.17085, 0.45586, -0.43426, 0.03923, -0.19616, 0.43945, 0.32297, -0.03911, 0.00000, 0.07045, 0.04374, 0.84779, 0.72043, 0.00000, 0.22509, 0.04196, 0.39141, -0.94427, -0.54932, -0.29774, 0.52868, -0.42482, -1.36019, -0.36418, -1.71356, 0.00000, -0.49572, -2.56563, 1.21863, -0.26418, 0.00000, 1.70359, 0.02971, 0.29175, -0.23315, -0.49127, -0.11170, -0.55583, -0.12680, 0.23240, 0.08856, -0.16778, 0.00000, -0.19175),
    (147.94558, -0.06125, 1.91347, -2.23011, 0.00000, 0.52381, -2.13582, -3.53971, -0.91679, -1.22770, 0.10889, 0.45657, 4.15390, 3.36319, 1.53247, 2.02909, 0.00000, 0.00000, -0.00149, 1.34472, 0.67094, 0.00000, 0.00000, -0.43225, -0.21757, -0.58585, 1.32288, 1.32032, 0.17601, 0.61791, 0.06106, -0.75331, 0.69094, 0.00000, 0.00000, -0.00051, -0.07879, -0.00949, 0.00000, 0.00004, 0.00000, -0.02611, 0.11148, -0.19975, -0.20046, 0.09546, 0.02611, 0.26575, -0.05760, -0.92275, 0.00000, 0.00000, 0.00206, 0.17307, 0.19860, 0.00000, -0.00456, 0.11196, 0.45359, -0.11117, 0.29294, 0.29079, -0.70431, -0.45360, -0.14910, 0.07923, -1.70450, 0.00000, 0.00000),
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


# Fitted (ridge, alpha 1000 on standardized features) to 129,108 near-tied
# discard pairs of fp-rule-131's own hands, 16 paired true-state rollouts each:
# the second-best discard's value over the first's, from the context (whites,
# melds, wall, dealer, chain, steps, the opponents' speed, their melds, the
# number of discards), each of the two discards' score parts, baotou kept,
# honor or white thrown and distilled features, their differences and the gap.
# Within DISCARD_GAP the second is played when the fit exceeds DISCARD_TAU
# (held out across two batches: +0.05 to +0.07 fan a game).
DISCARD_GAP = 0.03
DISCARD_TAU = 0.01
DISCARD_SWITCH = (0.0147213, -0.00321937, -0.00584804, -5.80028e-05, -0.00121406, -0.00153449, 0.00490122, 0.0015191, 1.22696e-06, -0.00183245, 0.00686089, 0.01001, -0.017508, -0.0507067, 0.000594546, 0, -0.00123867, 8.43382e-05, -0.000348586, -0.00162679, -0.00113745, -0.00165842, -0.000456871, 0.000418373, 0.000181436, -9.14578e-05, 0.000745865, 0.000594546, 0.00344548, 0.00398349, 0.000628844, 0.000202311, -0.000457435, 0.00706953, 0.0102136, -0.0179721, -0.0507067, 0.00208271, 0, -0.00145685, -0.000115448, 0.000346157, -0.00242285, -0.00137716, -0.000766064, 0.00144759, 0.000225017, -0.000566964, 0.00102935, 0.000406393, 0.00208271, 0.00175696, 0.00171904, -1.00563e-05, 2.89112e-05, -0.000345265, 0.16258, 0.19016, 0.0718561, 0, 0.000976911, 0, -0.0920847, -0.00895155, 0.0271956, -0.0220205, -0.00389674, 0.0281538, 0.0046752, -0.000886953, -0.0160712, 0.00633159, -0.00194505, 0.000976911, -0.000810003, -0.000849514, -0.000532485, -0.00361053, 0.0120842, -0.16258)


def discard_switch(obs, discards, raw, shifts, opened, available):
    """The second-best discard when it is within DISCARD_GAP of the best and
    the fitted correction says it is better by more than DISCARD_TAU; else None."""
    legal = obs.legal_actions
    ranked = sorted(((raw[i] + s, raw[i], s, legal[i]) for i, s in zip(discards, shifts)), key=lambda r: -r[0])
    if len(ranked) < 2 or ranked[0][0] - ranked[1][0] > DISCARD_GAP:
        return None
    held = _fast.counts(tuple(obs.hand))
    seat = obs.stage.seat
    draws = max(1, -(-(obs.wall_remaining - LAST_WALL) // 4))
    x = [held[WHITE], opened, obs.wall_remaining, float(obs.stage.dealer == seat), obs.chain_count,
         _fast.shanten_raw(list(held), opened), opponent_speed(obs, draws),
         sum(len(obs.melds[k]) for k in range(4) if k != seat), len(discards)]
    parts = []
    for score, value, shift, action in ranked[:2]:
        kept = list(held)
        kept[action.tile] -= 1
        parts.append([score, value, shift, float(_fast.baotou(kept, opened)), float(action.tile >= 27), float(action.tile == WHITE)]
                     + list(native.mimic_features(held, opened, available, action.tile)))
    first, second = parts
    x += first + second + [b - a for a, b in zip(first, second)] + [ranked[0][0] - ranked[1][0]]
    fit = DISCARD_SWITCH[0] + sum(w * f for w, f in zip(DISCARD_SWITCH[1:], x))
    return ranked[1][3] if fit > DISCARD_TAU else None


# Distilled from RL fp-rl-112's responses: a small network (native.c: call_net)
# fit to its chi/peng/pass choices over this line's own response windows reads
# each option's hand next to this line's value; its weight on the value (x10) is
# CALL_VALUE. An option's value gets CALL_MIMIC times its score, in value units.
CALL_VALUE = 2.035599


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
                raw = list(values)
                for i, s in zip(discards, shifts):
                    values[i] += s
                switched = discard_switch(obs, discards, raw, shifts, opened, available)
                if switched is not None:
                    return switched
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
