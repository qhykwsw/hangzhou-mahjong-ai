"""Rule line: expected fan of a self-draw race, doubling when the win is certain.

Only self-draws win, so a hand is worth the fan it can collect times its chance
of drawing it before an opponent does. Between two of my draws the three
opponents win with chance h(d), measured by my draws left d and rising from
almost nothing to about 0.38 as the wall runs down. A waiting hand that wins on
a fraction p of draws, collecting e fan per draw, is worth e times the draws it
reaches: the first, then each later one surviving my misses (1 - p) and the
opponents' h at that draw's level, round by round (native.c, the line's own
race); a move made at a later draw is priced at the level it is made at. A
wait holding a white also counts the draws that let a discard
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
DEALER_EDGE = 1.75  # R37 audit: 1.37 at the pool's tables, 2.13 at the proxy tables
KEEP = 0.27
PAYMENT = {True: 0.499, False: 0.530}
SAVED_BY_ROLE = {
    (dealer, left): round(PAYMENT[dealer] + DEALER_EDGE * (1 - KEEP ** left) / (1 - KEEP) / (24 if dealer else 10), 3)
    for dealer in (True, False) for left in range(4)
}
ONE_STEP = 0.22  # typical value of a hand one step from waiting
FAR_STEP = 0.38  # discount for each further step (0.45 under the held hazard; R39 ordering check)
KONG_TILES = {"concealed": 4, "exposed": 3, "added": 1}
LAST_WALL = 20  # the hand is drawn once the wall is down to this
OPTION_REACH = 1.35  # the largest measured ratio was 1.30
TIE = 1e-7  # weight of shape potential, which only separates equal values
SHAPE_PAIR = 0.009
SHAPE_ADJACENT = 0.0114
SHAPE_UKEIRE = 0.0078
SHAPE_DISTANCE = -0.045
SHAPE_OPEN = 0.0105
SHAPE_EDGE = -0.0055
CALL_WEIGHT = 0.7  # the share of the call option measured best against the pool
# The feed rates' scale by stage (wall >= 70, >= 45, else): the R37 audit saw
# about a quarter more chances to chi or peng early than the rates say.
K_FEED = (1.25, 1.0, 1.0)
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
FISH_REACH = 1.5
MIMIC = 0.5
CALL_MIMIC = 0.75
LONG = 16  # more draws of mine than any hand holds
# R39 structure v2 (from the R37 audit, pool and proxy tables): the chance an
# opponent wins before my next draw, by my draws left d at the decision,
# rising from about 0.004 to 0.38 as the wall runs down: HAZARD_K / (1 +
# exp(-(HAZARD_A + HAZARD_B d))), fitted to the audit's buckets. The race
# (this line's own, native.c) takes it round by round: before each later draw
# the value at that draw's level, times the opponents' speed; HAZARD_SCALE and
# STEP_SCALE scale it for the race of a wait and for the steps of a hand not
# yet waiting.
HAZARD_K, HAZARD_A, HAZARD_B = 0.3886, 4.826, -0.5074
HAZARD_SCALE = 1.0
STEP_SCALE = 1.0
# A hand holding whites wins bigger than its waits say (baotou and piao come
# later): the fan part of its race value is scaled by its whites (0, 1, 2+),
# by the R39 audit (pool and proxy tables: realized fan per win of waits not
# yet baotou). A baotou wait is not scaled (fanm).
FAN_BOOST = (1.0, 1.0, 1.0)
# One opponent's log-odds of winning before my next draw (logistic fit over
# 365k opponent-draws at the pool's and proxy tables, R39): intercept; my draws
# left d/16 and its hinges past 3, 5, 7, 9, 11 and 13; its chis, pengs, exposed and concealed kongs, melds*d/16;
# middle, terminal and honor tiles among its last three discards, whether it
# has three, the same among its last six; its discards since its last call /9
# (9 without a call), whether it has called; whether it has discarded a white,
# whether its last discard was one; whites visible to me; whether it deals; the
# hand number /16; its score over the table's mean /100 (within +-2); its
# discards /10; the table (K_POP: 0 the pool's code players, 1 the proxies).
OPPONENT_V2 = (-1.8671, -1.4926, -1.1527, -0.9303, -0.7360, -0.8787, -0.9678, -0.5584, -0.1204, -0.1430, -0.3032, -0.3195, 0.7252, 0.4276, 0.2691, 0.1861, 0.6351, 0.1054, 0.0458, -0.0989, -0.4212, -0.0968, 0.1021, 0.0293, -0.3226, 0.1156, 0.0659, 0.0453, -1.3161, -0.0799)
K_POP = 0.5
# The model's mean chance that any opponent wins before my next draw, by d
# (both tables; d <= 4 pooled): the opponents' speed is their chance over it.
SPEED_BASE = (0.2254, 0.2254, 0.2254, 0.2254, 0.2254, 0.2602, 0.2550, 0.2480, 0.2242, 0.1948, 0.1555, 0.1199, 0.0832, 0.0502, 0.0201, 0.0073, 0.0040)
# The race value of a hand, scaled by its steps from waiting (0..4+) and whites
# (0..2+): realized over raced value at the pool's and proxy tables (R39
# audit on this structure, 160k discards; a hand not yet waiting also gains
# from the calls it makes and the policy's later choices).
FAR_SCALE = {}


def qd(draws):
    d = min(max(draws, 1), 16)
    return HAZARD_K / (1 + math.exp(-(HAZARD_A + HAZARD_B * d)))


def rounds(keep, draws, speed=1.0):
    """Draws a stay of keep reaches over the draws left: the first, then each
    later one surviving the opponents round by round."""
    total, prod = 0.0, 1.0
    for k in range(max(0, draws)):
        total += prod
        prod *= keep * max(0.0, 1 - HAZARD_SCALE * qd(draws - k - 1) * speed)
    return total


def fanm(hand, opened):
    """The fan multiplier of a hand's race: by its whites, except a baotou wait,
    whose next win's fan is exact (fishing on from it is priced by piao_value)."""
    whites = hand.count(WHITE)
    if whites and _fast.baotou(list(_fast.counts(hand)), opened):
        return 1.0
    return FAN_BOOST[min(whites, 2)]


# Per-tile weights on the live copies (1: every unseen copy as likely drawn).
WEIGHTS = None


def far_scale(hand, opened):
    if not FAR_SCALE:
        return 1.0
    counts = list(_fast.counts(hand))
    steps = _fast.shanten_raw(counts, opened)
    if steps <= 0 and counts[WHITE] and _fast.baotou(counts, opened):
        return 1.0  # every draw wins: the race is exact
    return FAR_SCALE.get((min(max(steps, 0), 4), min(hand.count(WHITE), 2)), 1.0)
# At a win holding a white, what simulation says the valuation misses in the
# gap between going on and winning now (ridge fit over 13,143 such windows of
# this structure, R39 v2b, 8 true-state rollouts each): constant, one white not yet baotou, my draws
# left /16, 12 draws or more, the opponents' speed, a win worth two fan or
# more, two melds or more.
CONTINUE_SHIFT = (-0.051, 0.033, 0.166, 0.028, 0.162, -0.311, -0.044)


# Distilled from RL fp-rl-172's discards (conditional logits over 298,661 of
# this line's own discard decisions, one per best reachable steps: 0, 1, 2, 3+):
# next to this line's value (first weight), the weights of the kept hand's shape
# (native.c: mimic_features) and of the discarded tile's kind, alone and with a
# white held, late in the hand and with melds out. A discard's value gets MIMIC
# times their sum, in value units.
MIMIC_WEIGHTS = (
    (4.29581, -1.50051, 0.23797, 2.06922, 4.34080, 1.33455, 2.13377, -1.67834, 0.89054, 2.61243, -1.05717, 0.21555, 0.22169, 1.73138, 1.33449, 1.05544, -1.93376, -0.87494, -0.15096, -0.88899, -1.07936, -4.44313, 0.02009, -0.65193, -1.46538, -1.02836, -0.30202, -0.41079, 0.10185, -0.35161, -0.38671, -0.50342, -0.10731, 1.24231, 0.23548, 0.28547, 2.44424, -1.83902, 0.01076, -0.34966, 0.94519, 0.56532, 0.19644, -0.58718, 0.41061, 0.42617, -0.70876, -1.47145, -0.87988, -0.35059, 2.52949, 2.80835, 0.64477, 1.08632, -0.61589, 2.45548, 0.81369, 5.09165, -0.09852, -0.57859, 1.55425, -1.31307, 0.67115, 0.37267, 0.31367, 0.17685, -0.05854, -0.04891, 2.08976),
    (7.29391, -1.14450, 1.87780, -2.10647, 0.00000, 3.87025, 3.77157, -3.88999, -0.25307, 4.30746, -1.43309, 1.31910, 1.58298, 2.01731, 0.96435, 0.88479, 6.32081, -1.44862, 0.13760, -0.03224, -2.57719, 0.00000, 0.04053, -0.83071, -0.18637, 0.15856, -0.47719, 0.24389, -0.09069, 0.09792, 0.30100, 0.20942, 0.06871, -1.40985, 0.42988, -0.00024, -1.56957, 0.07628, 0.00000, -0.23867, 0.37981, 0.35005, 0.76324, -0.99946, -0.34101, 0.54279, -0.25288, -2.99044, -1.66492, -0.71501, -1.64838, -0.44574, -0.44528, -0.32479, -0.48014, 0.00000, 1.41079, 0.35779, -0.15404, -0.30900, 0.64353, -0.21839, 0.65401, 0.66653, 0.60166, 0.44670, -0.00782, 0.34812, 2.39104),
    (21.39025, -13.24006, 5.38800, -4.28787, 0.00000, 3.35486, 1.45605, -5.18258, -1.41014, 3.18949, -0.06607, 0.29996, 3.10605, 2.78771, 1.46518, 1.02358, 0.00000, 3.43841, -3.66669, 1.91757, -1.68280, 0.00000, 1.05400, 0.02654, -0.31790, -0.50889, 1.03937, -1.02850, -0.46750, 0.39490, 0.54293, 0.25162, 0.29959, 0.00000, -3.71761, -0.01138, -1.50531, -0.56590, 0.00000, 0.22201, 0.31281, 0.02553, 0.23211, -0.85900, -0.37276, 1.40088, -0.07837, -2.82171, -1.01671, -1.74417, 0.00000, 0.70876, -3.90388, 1.51067, 0.39347, 0.00000, 2.88664, 0.00085, 0.07074, -0.26004, -0.77134, 0.10655, -0.24524, -0.43998, 0.16251, 0.24329, -0.12727, 0.00000, 0.94471),
    (78.10407, -6.74567, 5.68318, -4.43996, 0.00000, 3.96843, -3.22799, -4.11326, -1.39825, 1.26210, 1.45857, 0.30633, 4.09791, 3.12291, 1.38902, 2.72654, 0.00000, 0.00000, -0.42662, 1.49807, 0.14867, 0.00000, 0.01343, -0.83624, 0.39165, 0.32693, 0.69740, 0.71141, 0.10435, -0.10823, -0.17763, -0.31055, 0.39621, 0.00000, 0.00000, 0.00001, 0.00095, 0.03214, 0.00000, -0.00648, 0.00000, 0.00800, -0.67409, 0.11924, 0.12069, 0.45790, -0.00800, -0.49798, 0.04091, -1.29271, 0.00000, 0.00000, -0.22976, 1.12594, 0.74140, 0.00000, 0.00270, -0.00644, 0.94807, 0.16528, -0.41813, -0.41477, -0.43171, -0.94723, -0.38052, -0.38673, -1.97150, 0.00000, 0.00000),
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


# Distilled from RL fp-rl-172's responses: a small network (native.c: call_net)
# fit to its chi/peng/pass choices over this line's own response windows reads
# each option's hand next to this line's value; its weight on the value (x10) is
# CALL_VALUE. An option's value gets CALL_MIMIC times its score, in value units.
CALL_VALUE = 1.463089


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


def opponent_speed(obs, draws):
    """The opponents' chance to win before my next draw, read from their melds,
    discards and the table (OPPONENT_V2, per opponent), as a multiple of the
    model's mean at the same draws left, in steps of 0.05 within 0.1-3."""
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
    scores = obs.stage.score_by_seat
    mean = sum(scores) / 4 if scores else 0.0
    visible = obs.hand.count(WHITE) + sum(x.count(WHITE) for x in obs.discards)
    stay = 1.0
    for seat in range(4):
        if seat == obs.stage.seat:
            continue
        discards = obs.discards[seat]
        last3, last6 = discards[-3:], discards[-6:]
        kinds = [(m.kind, m.concealed) for m in obs.melds[seat]]
        chi = sum(1 for k, _ in kinds if k == "chi")
        peng = sum(1 for k, _ in kinds if k == "peng")
        mkong = sum(1 for k, c in kinds if k == "gang" and not c)
        akong = sum(1 for k, c in kinds if k == "gang" and c)
        mid = lambda ts: sum(1 for t in ts if t < 27 and 1 <= t % 9 <= 7)
        term = lambda ts: sum(1 for t in ts if t < 27 and t % 9 in (0, 8))
        hon = lambda ts: sum(1 for t in ts if t >= 27)
        score = scores[seat] - mean if scores else 0.0
        features = (
            1, d / 16, *(max(0, d - k) / 16 for k in (3, 5, 7, 9, 11, 13)),
            min(chi, 3), min(peng, 3), mkong, akong, min(len(kinds), 3) * d / 16,
            mid(last3), term(last3), hon(last3), len(last3) == 3, mid(last6), term(last6), hon(last6),
            min(since.get(seat, 9), 9) / 9, seat in since, WHITE in discards, bool(discards) and discards[-1] == WHITE,
            visible, obs.stage.dealer == seat, obs.stage.round_no / 16, max(-200, min(200, score)) / 100,
            len(discards) / 10, K_POP,
        )
        logit = sum(w * x for w, x in zip(OPPONENT_V2, features))
        stay *= 1 - 1 / (1 + math.exp(-logit))
    ratio = (1 - stay) / SPEED_BASE[min(max(draws, 1), 16)]
    return min(3.0, max(0.1, round(ratio * 20) / 20))


# The valuation above, run by this line's own race (native.c, v2): the curve
# round by round (V2_ROUND) with each move priced at the level it is made at
# (V2_TIMING); completed waits and a guard for no draws left switched on.
native = _fast.load_native(__file__)
V2_FLAGS = 3
HAZARD_VECTOR = [HAZARD_SCALE * qd(d) for d in range(-64, 64)]
STEP_VECTOR = [STEP_SCALE * qd(d) for d in range(-64, 64)]
PROFILES = {
    saved: native.v2_profile(HAZARD_VECTOR, STEP_VECTOR, V2_FLAGS, saved, ONE_STEP, FAR_STEP, 1.0, True, True, True)
    for saved in (SAVED, *SAVED_BY_ROLE.values())
}
PROFILE = PROFILES[SAVED]
native.v2_feed_scale(*K_FEED)
# The call option weighs hands with the same shape correction the decisions use.
native.shape_constants(SHAPE_PAIR, SHAPE_ADJACENT, SHAPE_UKEIRE, SHAPE_DISTANCE, SHAPE_OPEN, SHAPE_EDGE)


def waiting_value(hand, opened, available, chain=0, piao=0, draws=LONG, options=False, speed=1.0):
    """Expected fan of a 13-tile hand just before my next draw; speed scales
    the opponents' chance to win first."""
    value = native.v2_waiting(hand, opened, available, chain, piao, draws, PROFILE, analyze, options, speed, WEIGHTS, fanm(hand, opened))
    return value * far_scale(hand, opened)


def reshaped_value(hand, opened, available, draws, options=True, speed=1.0):
    """A waiting hand with the option to trade a non-winning draw for a better
    wait; with options, a hand one step away also counts trading a draw that
    does not improve it for a better hand one step away (native.c)."""
    m = fanm(hand, opened)
    value = native.v2_reshaped(hand, opened, available, draws, PROFILE, analyze, options, speed, WEIGHTS, m)
    if options:
        value += native.v2_reshape_gain(hand, opened, available, draws, PROFILE, analyze, speed, WEIGHTS, m)
    return value * far_scale(hand, opened)


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
    q = HAZARD_SCALE * qd(draws - 1) * speed
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
                later = native.v2_waiting(rest, opened, available, chain + 1, piao + 1, draws - 1, PROFILES[saved], analyze, False, speed, WEIGHTS, fanm(rest, opened))
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
            uses.append(((tile, tile), native.feed_rate(stage, tile, False) * K_FEED[stage], (*pungs, tile), chis))
        if tile < 27 and chis < 2:
            for a, b in ((tile - 2, tile - 1), (tile - 1, tile + 1), (tile + 1, tile + 2)):
                if a >= 0 and b < 27 and a // 9 == tile // 9 == b // 9 and held[a] and held[b]:
                    uses.append(((a, b), native.feed_rate(stage, tile, True) * K_FEED[stage], pungs, chis + 1))
        best = None
        for used, rate, kept, chi_count in uses:
            rest = list(hand)
            for t in (*used, WHITE):
                rest.remove(t)
            rest = tuple(rest)
            if not _fast.baotou(list(_fast.counts(rest)), opened + 1):
                continue
            worth = native.v2_waiting(rest, opened + 1, available, chain + 1, piao + 1, draws, PROFILES[saved], analyze, False, speed, WEIGHTS, fanm(rest, opened + 1))
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
    q = HAZARD_SCALE * qd(draws - 1) * speed
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
    return (richer - plain) / unseen * rounds(1 - (row[5] + reach) / unseen, draws, speed)


@lru_cache(maxsize=4096)
def call_option(hand, opened, available, draws, stage, chis, saved=SAVED):
    """What a hand not yet waiting gains from the calls it can make: each tile a
    pair of mine can peng, or two neighbours of mine can chi from my left, comes
    at its measured rate per round (native.c), and is taken when the best hand
    the call keeps beats staying as I am, over the rounds the hand stays as it is."""
    value = native.v2_call_option(hand, opened, available, draws, stage, chis, PROFILES[saved], analyze, CALL_WEIGHT, 1.0, WEIGHTS, fanm(hand, opened))
    return value * far_scale(hand, opened)


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
            return (1 - qd(draws)) ** (ahead / 3) * waiting_value(
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
            value = (1 - qd(draws) * speed) * value + TIE * potential(rest, available)
            if not at_win:
                called = call_option(rest, opened, available, draws, stage, chis, SAVED)
                value += shape_bonus(rest, opened, available, True) + (1 - qd(draws) * speed) * called
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
            return (1 - qd(draws)) * value + shape_bonus(kept, opened + 1, available, True) + (1 - qd(draws)) * called - cost
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
            return now + miss * (1 - qd(draws) * speed) * waiting_value(
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
        global SAVED, PROFILE
        left = min(3, max(0, obs.stage.rounds_total - obs.stage.round_no))
        SAVED = SAVED_BY_ROLE[(obs.stage.dealer == obs.stage.seat, left)]
        PROFILE = PROFILES[SAVED]
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
