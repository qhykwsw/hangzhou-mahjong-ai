/* fp-rule-131 whole in C, for the training game (guided/fastgame.py).
 *
 * The rule line's fp-rule-131, frozen in
 * projects/personal/firstprinciples/.local/attempts/fp-rule-131/package
 * (strategy.py sha256 b18d020d42a979405a2b0a81faa6360185d8626b06eab7d07e781670d943a43c,
 *  native.c    sha256 74114e5e54d446a9aee12371b391258475b537c061f350363eca1f1607ba442f),
 * decides here instead of on a Python mirror of the hand. It is fp-rule-068,
 * whose port is the runtime's r68_* (the third fixed pool), with:
 *   - SAVED by role: each decision binds the race profile of
 *     SAVED_BY_ROLE[(dealer, hands left up to 3)] (registered from the
 *     package's own values, fastgame.rule131_member);
 *   - the reshape option one step away (native.c reshape_gain) on top of the
 *     reshaped value with options, and that option for kongs as for discards;
 *   - a kong away from a wait valued by the reshaped value;
 *   - the call costs (OPEN_CHI, PAIR_PENG, WHITE_PENG, SEVEN_PAIR_PENG);
 *   - the distilled corrections: a discard's mimic shift (native.c
 *     mimic_features), a response's call_net shift, fish_gbm at a win holding
 *     a white, kong_gbm keeping a kong tile, and a peng rather than an exposed
 *     kong holding a white.
 * Every value in the Python's arithmetic order, as the r68 port; the native.c
 * functions below are the package's, line for line, taking counts and a race
 * context instead of Python arguments (call_net and mimic_features compute
 * only the parts of an analysis they read), and its tables are
 * rule131_tables.h. */

#include "rule131_tables.h"

#define R131_OPTION_REACH 1.35
#define R131_FISH_REACH 0.8
#define R131_MIMIC 0.5
#define R131_CALL_MIMIC 0.75
#define R131_CALL_VALUE 2.035599
#define R131_SEVEN_PAIR_PENG 0.05
static const double R131_OPEN_CHI[2] = {0.1, 0.03};
static const double R131_PAIR_PENG[2] = {0.06, 0.02};
static const double R131_WHITE_PENG[3] = {0.06, 0.03, 0.02};

/* The race profiles a seat of fp-rule-131 binds: role[dealer][left] for
 * SAVED_BY_ROLE, and plain for the SAVED that piao_calls defaults to (the
 * module's SAVED as defined, which fish_features' call leaves in place). */
static struct {
    int ready, plain, role[2][4];
} r131;

/* sum() of floats as CPython 3.12 adds them (py_fsum), as the r68 port does:
 * the same on every host, the package's decisions as evaluated (on 3.12). */
static double r131_sum(const double *x, int n) {
    py_fsum s = {0};
    for (int i = 0; i < n; i++) py_fsum_add(&s, x[i]);
    return py_fsum_value(&s);
}

/* ---------- native.c: the reshape option one step away ---------- */

typedef struct {
    uint64_t k0, k1, a0, a1;
    double value;
    long ukeire;
} route_slot;

#define ROUTE_BITS 12
static route_slot route_cache[1 << ROUTE_BITS];

static double route_value(race_ctx *x, int *c);

static double route_of(race_ctx *x, int *c, long *ukeire) {
    uint64_t k0, k1;
    shanten_key(c, x->opened, &k0, &k1);
    k1 ^= (uint64_t)x->opened << 60;
    uint64_t h = (k0 * 0x9E3779B97F4A7C15ULL) ^ (k1 * 0xC2B2AE3D27D4EB4FULL) ^ (x->a0 * 0x165667B19E3779F9ULL) ^ x->a1;
    route_slot *slot = &route_cache[(h >> 20) & ((1 << ROUTE_BITS) - 1)];
    if (x->keyed && slot->k0 == k0 && slot->k1 == k1 && slot->a0 == x->a0 && slot->a1 == x->a1) {
        *ukeire = slot->ukeire;
        return slot->value;
    }
    const draw_slot *table = draw_distances(c, x->opened);
    long u = 0;
    for (int t = 0; t < NTILES; t++)
        if (x->avail[t] > 0 && c[t] < 4 && table->draw[t] < table->distance) u += x->avail[t];
    double v = route_value(x, c);
    if (x->keyed) {
        slot->k0 = k0; slot->k1 = k1; slot->a0 = x->a0; slot->a1 = x->a1;
        slot->value = v; slot->ukeire = u;
    }
    *ukeire = u;
    return v;
}

static double route_value(race_ctx *x, int *c) {
    int opened = x->opened;
    const int *avail = x->avail;
    const draw_slot *table = draw_distances(c, opened);
    if (table->distance != 1) return 0.0;
    uint64_t improving = 0;
    for (int t = 0; t < NTILES; t++)
        if (avail[t] > 0 && c[t] < 4 && table->draw[t] < 1) improving |= (uint64_t)1 << t;
    /* Only the discards race_swaps admits can leave a wait; the winning
     * draws and their fans are the race's own wait row. */
    uint64_t k0, k1, after[NTILES];
    shanten_key(c, opened, &k0, &k1);
    race_swaps(c, opened, k0, k1, after);
    long total = 0;
    double weighted = 0.0;
    for (int tile = 0; tile < NTILES; tile++) {
        if (!(improving >> tile & 1)) continue;
        uint64_t d0, d1;
        shanten_key_draw(k0, k1, tile, &d0, &d1);
        c[tile]++;
        long best = 0;
        uint64_t drops = after[tile];
        for (int drop = 0; drops >> drop; drop++) {
            if (!(drops >> drop & 1) || !c[drop]) continue;
            uint64_t r0, r1;
            race_key_drop(d0, d1, drop, &r0, &r1);
            c[drop]--;
            if (shanten_keyed(r0, r1, c, opened) <= 0) {
                analysis_row row;
                race_wait_row(x, c, opened, r0, r1, 0, 0, &row);
                if (row.value > best) best = row.value;
            }
            c[drop]++;
        }
        c[tile]--;
        total += avail[tile];
        weighted += (double)avail[tile] * best;
    }
    return total ? weighted / total : 0.0;
}

/* reshape_gain(hand, opened, available, draws, profile, fallback, q, speed),
 * the hand's counts c (restored) and the context base at opened and speed. */
static int r131_reshape_gain(const race_ctx *base, int *c, int opened, int draws, double q, double speed, double *out) {
    int held[NTILES];
    race_ctx x;
    r68_at(&x, base, opened, speed);
    *out = 0.0;
    analysis_row row;
    if (race_analysis(&x, c, 0, 0, &row) < 0) return -1;
    if (row.distance != 1 || draws < 3) return 0;
    int improving[NTILES] = {0};
    for (int i = 0; i < row.n_improving; i++) improving[row.improving[i]] = 1;
    double ukeire = (double)row.ukeire;
    double stay;
    if (race_waiting(&x, c, 0, 0, draws - 1, 0, &stay) < 0) return -1;
    memcpy(held, c, sizeof(held));
    long own;
    double floor_ = ukeire * route_of(&x, c, &own);
    double gain = 0.0;
    long reach = 0;
    for (int tile = 0; tile < NTILES; tile++) {
        if (improving[tile] || x.avail[tile] <= 0 || c[tile] >= 4) continue;
        /* A draw joining no tile of mine can only swap one lone tile for another. */
        int near = c[tile] > 0;
        if (tile < 27)
            for (int u = tile - 2; u <= tile + 2 && !near; u++)
                near = u >= 0 && u < 27 && u / 9 == tile / 9 && c[u] > 0;
        if (!near) continue;
        c[tile]++;
        int pick = -1;
        double lead = floor_;
        for (int drop = 0; drop < NTILES; drop++) {
            if (!held[drop] || drop == tile) continue;
            c[drop]--;
            if (shanten_raw(c, opened) == 1) {
                long alt;
                double estimate = route_of(&x, c, &alt);
                if (estimate > 0) {
                    estimate *= (double)alt;
                    if (estimate > lead) {
                        pick = drop;
                        lead = estimate;
                    }
                }
            }
            c[drop]++;
        }
        double best = stay;
        if (pick >= 0) {
            c[pick]--;
            double v;
            int rc = race_waiting(&x, c, 0, 0, draws - 1, 0, &v);
            c[pick]++;
            if (rc < 0) {
                c[tile]--;
                return -1;
            }
            if (v > best) best = v;
        }
        c[tile]--;
        if (best > stay) {
            gain += (double)x.avail[tile] * (best - stay);
            reach += x.avail[tile];
        }
    }
    double unseen = (double)x.unseen, qs = q * speed;
    double survive = (1 - qs) * (1 - (ukeire + (double)reach) / unseen);
    double rounds = draws - 1 <= 0 ? 0.0 : (1 - pow(survive, (double)(draws - 1))) / (1 - survive);
    *out = (1 - qs) * gain / unseen * rounds;
    return 0;
}

/* ---------- native.c: the distilled corrections ---------- */

/* analyze_values' distance and ukeire alone, without the fans of winning draws. */
static int r131_steps(const int *c, const int *av, int opened, long *ukeire) {
    const draw_slot *table = draw_distances(c, opened);
    long u = 0;
    for (int t = 0; t < NTILES; t++)
        if (av[t] > 0 && c[t] < 4 && table->draw[t] < table->distance) u += av[t];
    *ukeire = u;
    return table->distance;
}

static double call_net_score(const double *x) {
    double h0[64], h1[32], out = CALL_B2;
    for (int i = 0; i < 64; i++) {
        double s = CALL_B0[i];
        for (int k = 0; k < 22; k++) s += CALL_W0[i * 22 + k] * x[k];
        h0[i] = s > 0 ? s : 0;
    }
    for (int i = 0; i < 32; i++) {
        double s = CALL_B1[i];
        for (int k = 0; k < 64; k++) s += CALL_W1[i * 64 + k] * h0[k];
        h1[i] = s > 0 ? s : 0;
    }
    for (int k = 0; k < 32; k++) out += CALL_W2[k] * h1[k];
    return out;
}

/* call_net(held, opened, available, options, wall, dealer, ahead) over the
 * legal actions, each read as its option: (0, -1, -1, -1) a pass, (1, a, b,
 * tile) a chi using a and b, (2, tile, tile, tile) a peng. Only the rows'
 * distances and ukeire count (r131_steps). */
static void r131_call_net(const int *c, int opened, const int *av, const race_action *legal, int n, int wall,
                          int dealer, int ahead, double *scores) {
    int before = shanten_raw(c, opened), pairs = 0;
    for (int t = 0; t < 33; t++)
        if (c[t] >= 2) pairs++;
    double late = (60 - wall) / 40.0;
    for (int i = 0; i < n; i++) {
        const race_action *option = &legal[i];
        int kind = option->kind == ACT_CHI ? 1 : option->kind == ACT_PENG ? 2 : 0;
        int a = kind == 1 ? option->tiles[0] : kind == 2 ? option->tile : -1;
        int b = kind == 1 ? option->tiles[1] : kind == 2 ? option->tile : -1;
        int tile = kind ? option->tile : -1;
        int after = before;
        long uke = 0;
        if (kind == 0) {
            r131_steps(c, av, opened, &uke);
        } else {
            int k[NTILES], found = 0, best_d = 0;
            long best_u = 0;
            memcpy(k, c, sizeof k);
            k[a]--;
            k[b]--;
            for (int d = 0; d < 33; d++) {
                if (!k[d]) continue;
                k[d]--;
                long ukeire;
                int distance = r131_steps(k, av, opened + 1, &ukeire);
                if (!found || distance < best_d || (distance == best_d && ukeire > best_u)) {
                    found = 1;
                    best_d = distance;
                    best_u = ukeire;
                }
                k[d]++;
            }
            if (found) {
                after = best_d;
                uke = best_u;
            }
        }
        int call = kind != 0, chi = kind == 1, peng = kind == 2;
        int tk = !call ? 0 : tile >= 27 ? 1 : (tile % 9 == 0 || tile % 9 == 8) ? 2 : 3;
        double steps = (before < 4 ? before : 4) / 4.0;
        double x[22] = {chi, peng, call * (double)(after - before), call && after < before, call && after == before,
                        call && after > before, uke / 10.0 * call, call && tk == 1, call && tk == 2, call && tk == 3,
                        call * steps, call * (pairs / 7.0), call && ahead == 0, c[WHITE] ? 1.0 : 0.0, c[WHITE] / 2.0,
                        late > 0 ? late : 0.0, wall / 100.0, (opened < 2 ? opened : 2) / 2.0, dealer ? 1.0 : 0.0,
                        steps, pairs / 7.0, uke / 10.0};
        scores[i] = call_net_score(x);
    }
}

/* mimic_features(held, opened, available, tile), into f[17]. */
static int r131_mimic_features(const int *held, int opened, const int *av, int tile, double *f) {
    int c[NTILES];
    if (tile < 0 || tile >= NTILES || !held[tile]) {
        PyErr_SetString(PyExc_ValueError, "tile must be held");
        return -1;
    }
    memcpy(c, held, sizeof c);
    c[tile]--;
    analysis_row row;
    analyze_values(c, av, opened, 0, 0, 0, &row);
    int dist = row.distance;
    long wincp = dist == 0 ? row.copies : 0;
    int pairs = 0, trips = 0, isoh = 0, isos = 0, adj = 0, edge = 0, gap = 0;
    for (int t = 0; t < 33; t++) {
        if (c[t] >= 2) pairs++;
        if (c[t] >= 3) trips++;
    }
    for (int t = 27; t < 33; t++)
        if (c[t] == 1) isoh++;
    for (int t = 0; t < 27; t++) {
        if (c[t] == 1) {
            int near = 0;
            for (int u = t - 2; u <= t + 2 && !near; u++)
                if (u != t && u >= 0 && u < 27 && u / 9 == t / 9 && c[u]) near = 1;
            if (!near) isos++;
        }
        if (t % 9 <= 7 && c[t] && c[t + 1]) {
            adj++;
            if (t % 9 == 0 || t % 9 == 7) edge++;
        }
        if (t % 9 <= 6 && c[t] && c[t + 2] && !c[t + 1]) gap++;
    }
    int kind = tile >= 27 ? 0 : (tile % 9 == 0 || tile % 9 == 8) ? 1 : (tile % 9 == 1 || tile % 9 == 7) ? 2 : 3;
    int seen = 4 - av[tile] - held[tile];
    double good = 0.0, avgw = 0.0, tot = 0.0;
    if (dist == 1 || dist == 2) {
        for (int i = 0; i < row.n_improving; i++) {
            int t = row.improving[i];
            if (av[t] <= 0) continue;
            c[t]++;
            long best = -1;
            for (int x = 0; x < 33; x++) {
                if (c[x] && (x != t || c[x] > 1)) {
                    c[x]--;
                    /* the row's distance first (analyze_values' is shanten_raw's), the rest only at dist - 1 */
                    if (shanten_raw(c, opened) == dist - 1) {
                        long q;
                        if (dist == 1) {
                            analysis_row rr;
                            analyze_values(c, av, opened, 0, 0, 0, &rr);
                            q = rr.copies;
                        } else {
                            r131_steps(c, av, opened, &q);
                        }
                        if (q > best) best = q;
                    }
                    c[x]++;
                }
            }
            c[t]--;
            if (best >= 0) {
                tot += av[t];
                avgw += av[t] * (double)best;
                if (dist == 1 && best >= 5) good += av[t];
            }
        }
        avgw = tot > 0 ? avgw / tot : 0.0;
    }
    const double out[17] = {(double)dist, row.ukeire / 10.0, row.n_improving / 5.0, wincp / 5.0,
                            (double)pairs, (double)trips, (double)isoh, (double)isos, (double)adj, (double)edge,
                            (double)gap, (double)(kind == 0), (double)(kind == 1), (double)(kind == 2), (double)seen,
                            good / 10.0, avgw / 10.0};
    memcpy(f, out, sizeof(out));
    return 0;
}

static double r131_fish_gbm(const double *x) {
    double s = 0.0018105362747059411;
    for (int t = 0; t < 400; t++) {
        int n = FISH_ROOT[t];
        while (!FISH_LEAF[n]) n = x[FISH_FEAT[n]] <= FISH_THR[n] ? FISH_LEFT[n] : FISH_RIGHT[n];
        s += FISH_VAL[n];
    }
    return s;
}

static double r131_kong_gbm(const double *x) {
    double s = 0.12013278721682846;
    for (int t = 0; t < 300; t++) {
        int n = KONG_ROOT[t];
        while (!KONG_LEAF[n]) n = x[KONG_FEAT[n]] <= KONG_THR[n] ? KONG_LEFT[n] : KONG_RIGHT[n];
        s += KONG_VAL[n];
    }
    return s;
}

/* ---------- strategy.py ---------- */

/* What fp-rule-131 reads beyond fp-rule-068's view: whether obs.phase is a response. */
typedef struct {
    r68_view o;
    int response;
} r131_view;

/* Race.value: r68_value with the reshape option, a kong away from a wait
 * valued as a discard is, and the call costs. */
static int r131_value(race_act_ctx *A, const r131_view *w, const race_ctx *base, const race_action *a, int options,
                      double speed, int at_win, r68_parts *parts, double *out) {
    const r68_view *o = &w->o;
    const race_profile *p = base->p;
    int *c = A->c, opened = A->opened, draws = A->draws;
    double v;
    switch (a->kind) {
    case ACT_HU:
        *out = (double)A->hu_fan + p->saved;
        return 0;
    case ACT_PASS: {
        int ahead = o->ahead < 0 ? 0 : o->ahead;  /* no discard to pass on: nobody ahead */
        if (r68_waiting(base, c, opened, A->chain, A->piao, draws, 0, 1.0, &v) < 0) return -1;
        double shape = r68_shape(base, c, opened), called;
        if (r68_call_option(base, c, opened, draws, o->stage, o->chis, &called) < 0) return -1;
        *out = pow(1 - p->draw_risk, (double)ahead) * v + shape + called;
        return 0;
    }
    case ACT_DISCARD: {
        int known = parts && parts->known, piao = known ? parts->piao : 0;
        if (!known && a->tile == WHITE) {
            piao = act_piao(A);
            if (piao < 0) return -1;
        }
        if (known && piao) {  /* the piao wait takes no trade option: the first pass's value */
            *out = parts->extra;
            return 0;
        }
        c[a->tile]--;
        int rc = 0;
        double gain = 0.0, potential, extra = 0.0;
        if (piao) {
            rc = r68_waiting(base, c, opened, A->chain + 1, A->piao + 1, draws, 0, speed, &v);
        } else {
            rc = r68_reshaped(base, c, opened, draws, options, speed, &v);
            if (rc == 0 && options) {
                double trade;
                rc = r131_reshape_gain(base, c, opened, draws, p->q, speed, &trade);
                if (rc == 0) v += trade;
            }
            if (rc == 0 && at_win) {
                if (known) gain = parts->gain;
                else rc = r68_piao_gain(base, c, opened, draws, speed, o->pungs, o->n_pungs, o->stage, o->chis, &gain);
                if (rc == 0) v += gain;
            }
        }
        if (rc == 0) {
            potential = known ? parts->potential : p->tie * (double)potential_counts(c, base->avail);
            v = (1 - p->q * speed) * v + potential;
            if (!at_win) {
                if (known) {
                    extra = parts->extra;
                } else {
                    double called;
                    rc = r68_call_option(base, c, opened, draws, o->stage, o->chis, &called);
                    if (rc == 0) extra = r68_shape(base, c, opened) + (1 - p->q * speed) * called;
                }
                if (rc == 0) v += extra;
            }
            if (rc == 0 && parts && !known) {
                parts->known = 1;
                parts->piao = piao;
                parts->gain = gain;
                parts->potential = potential;
                parts->extra = piao ? v : extra;
            }
        }
        c[a->tile]++;
        if (rc < 0) return -1;
        *out = v;
        return 0;
    }
    case ACT_PENG:
    case ACT_CHI: {
        int used[2], n = 2;
        if (a->kind == ACT_CHI) {
            used[0] = a->tiles[0];
            used[1] = a->tiles[1];
        } else {
            used[0] = used[1] = a->tile;
        }
        /* hand.remove(tile) for each used tile: the list order without them. */
        int list[18], m = 0, taken[2] = {0, 0};
        for (int i = 0; i < o->n_hand; i++) {
            int t = o->hand[i], skip = 0;
            for (int k = 0; k < n && !skip; k++)
                if (!taken[k] && used[k] == t) taken[k] = skip = 1;
            if (!skip) list[m++] = t;
        }
        int h[NTILES];
        memset(h, 0, sizeof(h));
        for (int i = 0; i < m; i++) h[list[i]]++;
        int order[18];
        int kinds = py_set_order(list, m, order);
        /* kept = max(rests, key=waiting_value at opened + 1): the first best in set order. */
        int kept = -1;
        double best = 0.0;
        for (int k = 0; k < kinds; k++) {
            int t = order[k];
            h[t]--;
            int rc = r68_waiting(base, h, opened + 1, 0, 0, draws, 0, 1.0, &v);
            h[t]++;
            if (rc < 0) return -1;
            if (kept < 0 || v > best) {
                kept = order[k];
                best = v;
            }
        }
        double value = best;
        if (h[WHITE] && is_baotou_counts(c, A->size, opened)) {
            h[WHITE]--;
            if (is_baotou_counts(h, A->size - n - 1, opened + 1)) {
                int owner = o->god == A->seat;
                int chain = (owner ? A->chain : 0) + 1, piao = (owner ? A->piao : 0) + 1;
                if (r68_waiting(base, h, opened + 1, chain, piao, draws, 0, 1.0, &v) < 0) return -1;
                if (v > value) value = v;
            }
            h[WHITE]++;
        }
        h[kept]--;
        double called;
        if (r68_call_option(base, h, opened + 1, draws, o->stage, o->chis + (a->kind == ACT_CHI), &called) < 0) return -1;
        double shape = r68_shape(base, h, opened + 1);
        double cost = 0.0;
        if (a->kind == ACT_CHI && opened == 0) {
            int distance = shanten_raw(h, opened + 1);
            cost = distance >= 2 ? R131_OPEN_CHI[1] : distance == 1 ? R131_OPEN_CHI[0] : 0.0;
        } else if (a->kind == ACT_PENG && opened == 0) {
            int steps = shanten_raw(c, 0), pairs = 0;
            for (int t = 0; t < 33; t++) pairs += c[t] >= 2;
            int extra = pairs - 2 > 0 ? pairs - 2 : 0;
            cost = (double)(-extra) * (steps == 1 ? R131_PAIR_PENG[0] : steps >= 2 ? R131_PAIR_PENG[1] : 0.0);
            if (shanten_raw(h, opened + 1) > steps) cost -= R131_SEVEN_PAIR_PENG;
        }
        if (a->kind == ACT_PENG && c[WHITE]) {
            int steps = shanten_raw(c, opened);
            cost -= R131_WHITE_PENG[steps < 0 ? 0 : steps > 2 ? 2 : steps];
        }
        *out = (1 - p->q) * value + shape + (1 - p->q) * called - cost;
        return 0;
    }
    case ACT_GANG: {
        int h[NTILES];
        memcpy(h, c, sizeof(h));
        h[a->tile] -= a->kong;
        int after = a->added ? opened : opened + 1;
        race_ctx y;
        r68_at(&y, base, after, 1.0);
        analysis_row row;
        if (race_analysis(&y, h, A->chain + 1, A->piao, &row) < 0) return -1;
        double shape = r68_shape(base, h, after);
        if (row.distance > 0) {
            double called;
            if (r68_reshaped(base, h, after, draws, options, speed, &v) < 0) return -1;
            if (options) {
                double trade;
                if (r131_reshape_gain(base, h, after, draws, p->q, speed, &trade) < 0) return -1;
                v += trade;
            }
            if (r68_call_option(base, h, after, draws, o->stage, o->chis, &called) < 0) return -1;
            *out = v + shape + called;
            return 0;
        }
        if (r68_waiting(base, h, after, 0, 0, draws, 0, speed, &v) < 0) return -1;
        double unseen = base->unseen;
        long copies = row.copies;
        double now = ((double)row.value + p->saved * (double)copies) / unseen;
        double miss = 1 - (double)copies / unseen;
        *out = now + miss * (1 - p->q * speed) * v + shape;
        return 0;
    }
    default:
        *out = -INFINITY;
        return 0;
    }
}

/* mimic_shifts for the discards at legal indices idx. */
static int r131_mimic(const race_act_ctx *A, const r131_view *w, const int *idx, int m, double *shift) {
    const int *held = A->c, *av = A->x.avail;
    double white = held[WHITE] ? 1.0 : 0.0;
    double late = (double)(60 - w->o.wall) / 40;
    if (!(late > 0.0)) late = 0.0;  /* max(0.0, late): the first unless the second is larger */
    double op = (double)(A->opened < 2 ? A->opened : 2) / 2;
    double rows[64][17];
    for (int k = 0; k < m; k++)
        if (r131_mimic_features(held, A->opened, av, A->legal[idx[k]].tile, rows[k]) < 0) return -1;
    double low = rows[0][0];
    for (int k = 1; k < m; k++)
        if (rows[k][0] < low) low = rows[k][0];
    long level = (long)low;  /* int() truncates; min(3, level); a negative index counts from the end */
    if (level > 3) level = 3;
    if (level < 0) level += 4;
    if (level < 0) {
        PyErr_SetString(PyExc_IndexError, "tuple index out of range");
        return -1;
    }
    const double *weights = MIMIC_WEIGHTS[level];
    for (int k = 0; k < m; k++) {
        double terms[68];
        for (int f = 0; f < 17; f++) {
            double base = rows[k][f];
            terms[f] = weights[1 + f] * base;
            terms[17 + f] = weights[18 + f] * (base * white);
            terms[34 + f] = weights[35 + f] * (base * late);
            terms[51 + f] = weights[52 + f] * (base * op);
        }
        shift[k] = R131_MIMIC * r131_sum(terms, 68) / (10 * weights[0]);
    }
    return 0;
}

/* fish_features(obs, opened, available, speed, cont, margin, v_hu, v_cont) into x[18]. */
static int r131_fish_features(const race_act_ctx *A, const r131_view *w, const race_ctx *base, double speed, int cont,
                              double margin, double v_hu, double v_cont, double *x) {
    const r68_view *o = &w->o;
    const int *av = base->avail;
    int opened = A->opened, whites = A->c[WHITE], draws = A->draws, c[NTILES];
    if (o->drawn < 0 || !A->c[o->drawn]) {
        PyErr_SetString(PyExc_ValueError, "list.remove(x): x not in list");
        return -1;
    }
    memcpy(c, A->c, sizeof(c));
    c[o->drawn]--;
    int baotou = is_baotou_counts(c, A->size - 1, opened);
    int kind = baotou && whites >= 2 ? 0 : baotou ? 1 : 2;
    double win = -1.0, uke = -1.0, piao = -1.0, kong = -1.0, upgrade = -1.0, callp = -1.0;
    const race_action *a = &A->legal[cont];
    if (a->kind == ACT_DISCARD) {
        memcpy(c, A->c, sizeof(c));
        c[a->tile]--;
        if (A->size - 1 == 13 - 3 * opened) {
            analysis_row row;
            analyze_values(c, av, opened, 0, 0, 0, &row);
            double unseen = (double)base->unseen;
            win = (double)row.copies / unseen;
            uke = (double)row.ukeire / unseen;
            int bt = baotou_shape(c, opened);
            long pi = 0, kg = 0, up = 0;
            for (int t = 0; t < NTILES; t++) {
                if (!av[t] || c[t] >= 4) continue;
                c[t]++;
                if (bt && c[WHITE] >= 2) {
                    c[WHITE]--;
                    pi += av[t] * baotou_shape(c, opened);
                    c[WHITE]++;
                }
                int in_pungs = 0;
                for (int i = 0; i < o->n_pungs; i++) in_pungs |= o->pungs[i] == t;
                if (bt && (c[t] == 4 || in_pungs)) kg += av[t];
                if (!bt && c[WHITE]) {
                    int ok = 0;
                    for (int y = 0; y < NTILES; y++) {
                        if (c[y] && y != t) {
                            c[y]--;
                            ok = ok || baotou_shape(c, opened);
                            c[y]++;
                        }
                        if (ok) break;
                    }
                    up += av[t] * ok;
                }
                c[t]--;
            }
            piao = (double)pi / unseen;
            kong = (double)kg / unseen;
            upgrade = (double)up / unseen;
            double chance[NTILES], worth[NTILES];
            int n_calls = 0;
            if (bt && c[WHITE] >= 2) {
                /* piao_calls at its default SAVED */
                race_ctx plain = *base;
                plain.p = &race_profiles[r131.plain];
                plain.profile = r131.plain;
                if (r68_piao_calls(&plain, c, opened, 0, 0, draws, speed, o->pungs, o->n_pungs, o->stage, o->chis,
                                   chance, worth, &n_calls) < 0)
                    return -1;
            }
            callp = r131_sum(chance, n_calls);
        }
    }
    const double out[18] = {margin, (double)kind, (double)whites, (double)(draws < 16 ? draws : 16), speed,
                            (double)A->hu_fan, (double)opened, (double)A->chain, (double)(o->dealer == A->seat),
                            win, uke, piao, kong, upgrade, callp, v_hu, v_cont, (double)o->wall};
    memcpy(x, out, sizeof(out));
    return 0;
}

/* fish_decision: at a win holding a white, the action fish_gbm settles a close margin on; -1 for none. */
static int r131_fish(race_act_ctx *A, const r131_view *w, const race_ctx *base, double speed, int *out) {
    int n = A->n_legal, hu = -1, cont = -1;
    double values[64];
    *out = -1;
    for (int i = 0; i < n; i++)
        if (r131_value(A, w, base, &A->legal[i], 1, speed, 1, NULL, &values[i]) < 0) return -1;
    for (int i = 0; i < n && hu < 0; i++)
        if (A->legal[i].kind == ACT_HU) hu = i;
    for (int i = 0; i < n; i++)
        if (A->legal[i].kind != ACT_HU && (cont < 0 || values[i] > values[cont])) cont = i;
    if (cont < 0) return 0;
    int error;
    double margin = values[cont] - values[hu] + r68_continue_shift(A, &w->o, speed, &error);
    if (error) return -1;
    if (fabs(margin) > R131_FISH_REACH) return 0;
    double x[18];
    if (r131_fish_features(A, w, base, speed, cont, margin, values[hu], values[cont], x) < 0) return -1;
    *out = margin + r131_fish_gbm(x) > 0 ? cont : hu;
    return 0;
}

/* keep_kong: holding a white one step away or waiting, the discard kong_gbm keeps the kong tile for; -1 for none. */
static int r131_keep_kong(race_act_ctx *A, const r131_view *w, const race_ctx *base, int kong, int *out) {
    const r68_view *o = &w->o;
    int n = A->n_legal, opened = A->opened, keep = -1;
    *out = -1;
    if (shanten_raw(A->c, opened) > 1) return 0;
    uint64_t kong_tiles = 0;
    for (int i = 0; i < n; i++)
        if (A->legal[i].kind == ACT_GANG) kong_tiles |= 1ULL << A->legal[i].tile;
    int keeps = 0;
    for (int i = 0; i < n; i++)
        keeps |= A->legal[i].kind == ACT_DISCARD && !(kong_tiles >> A->legal[i].tile & 1);
    if (!keeps) return 0;
    double values[64];
    for (int i = 0; i < n; i++)
        if (r131_value(A, w, base, &A->legal[i], 1, 1.0, 0, NULL, &values[i]) < 0) return -1;
    for (int i = 0; i < n; i++)
        if (A->legal[i].kind == ACT_DISCARD && !(kong_tiles >> A->legal[i].tile & 1) && (keep < 0 || values[i] > values[keep]))
            keep = i;
    /* kong_features */
    const race_action *k = &A->legal[kong];
    const int *av = base->avail;
    double unseen = (double)base->unseen;
    int kept[NTILES], melded[NTILES];
    memcpy(kept, A->c, sizeof(kept));
    kept[A->legal[keep].tile]--;
    analysis_row row_keep, row_kong;
    analyze_values(kept, av, opened, 0, 0, 0, &row_keep);
    int after = k->added ? opened : opened + 1;
    memcpy(melded, A->c, sizeof(melded));
    melded[k->tile] -= k->kong;
    analyze_values(melded, av, after, 0, 0, 0, &row_kong);
    double v_kong = values[kong], v_keep = values[keep];
    const double x[21] = {(double)A->c[WHITE], k->added ? 0.0 : k->kong == 4 ? 1.0 : 2.0, (double)o->wall, (double)opened,
                          (double)(o->dealer == A->seat), v_kong - v_keep, v_kong, v_keep,
                          (double)row_keep.distance, (double)row_keep.copies / unseen, (double)row_keep.ukeire / unseen,
                          (double)baotou_shape(kept, opened),
                          (double)row_kong.distance, (double)row_kong.copies / unseen, (double)row_kong.ukeire / unseen,
                          (double)baotou_shape(melded, after),
                          (double)A->chain, (double)row_keep.value / unseen, (double)row_kong.value / unseen,
                          (double)(k->tile >= 27), (double)o->chis};
    if (r131_kong_gbm(x) > 0) *out = keep;
    return 0;
}

/* Race.act of fp-rule-131, A bound to its role's profile. */
static int r131_act(race_act_ctx *A, const r131_view *w, int *choice) {
    const r68_view *o = &w->o;
    const race_profile *p = A->x.p;
    race_ctx base = A->x;
    int n = A->n_legal, has_hu = 0, has_gang = 0, calls_only = 1;
    for (int i = 0; i < n; i++) {
        int k = A->legal[i].kind;
        has_hu |= k == ACT_HU;
        has_gang |= k == ACT_GANG;
        calls_only &= k == ACT_PASS || k == ACT_CHI || k == ACT_PENG;
    }
    if (has_hu) {
        int d = act_double(A);
        if (d == -2) return -1;
        if (d >= 0) {
            *choice = d;
            return 0;
        }
    }
    double speed = has_hu ? r68_opponent_speed(p, o, A->seat, A->draws) : 1.0;
    double values[64], top = 0.0;
    r68_parts parts[64];
    memset(parts, 0, sizeof(parts));
    for (int i = 0; i < n; i++) {
        if (r131_value(A, w, &base, &A->legal[i], 0, speed, has_hu, &parts[i], &values[i]) < 0) return -1;
        if (i == 0 || values[i] > top) top = values[i];
    }
    /* The trade options for the discards and kongs close to the best. */
    for (int i = 0; i < n; i++) {
        int k = A->legal[i].kind;
        if ((k == ACT_DISCARD || k == ACT_GANG) && values[i] * R131_OPTION_REACH >= top)
            if (r131_value(A, w, &base, &A->legal[i], 1, speed, has_hu, &parts[i], &values[i]) < 0) return -1;
    }
    if (!has_hu && !w->response && !has_gang) {
        int idx[64], m = 0;
        for (int i = 0; i < n; i++)
            if (A->legal[i].kind == ACT_DISCARD) idx[m++] = i;
        if (m) {
            double shift[64];
            if (r131_mimic(A, w, idx, m, shift) < 0) return -1;
            for (int k = 0; k < m; k++) values[idx[k]] += shift[k];
        }
    }
    if (w->response && calls_only) {
        double scores[64];
        int ahead = o->ahead < 0 ? 3 : o->ahead;  /* without a discard the discarder is me */
        r131_call_net(A->c, A->opened, base.avail, A->legal, n, o->wall, o->dealer == A->seat, ahead, scores);
        for (int i = 0; i < n; i++) values[i] = values[i] + R131_CALL_MIMIC * scores[i] / (10 * R131_CALL_VALUE);
    }
    if (has_hu && A->c[WHITE]) {
        int fitted;
        if (r131_fish(A, w, &base, speed, &fitted) < 0) return -1;
        if (fitted >= 0) {
            *choice = fitted;
            return 0;
        }
        int error;
        double shift = r68_continue_shift(A, o, speed, &error);
        if (error) return -1;
        for (int i = 0; i < n; i++)
            if (A->legal[i].kind != ACT_HU) values[i] = values[i] + shift;
    }
    int best = 0;
    for (int i = 1; i < n; i++)
        if (values[i] > values[best]) best = i;
    if (A->legal[best].kind == ACT_GANG && !has_hu && A->c[WHITE]) {
        if (!w->response) {
            int kept;
            if (r131_keep_kong(A, w, &base, best, &kept) < 0) return -1;
            if (kept >= 0) {
                *choice = kept;
                return 0;
            }
        } else {
            /* Holding a white near a wait, peng the discard rather than meld all four. */
            int peng = -1;
            for (int i = 0; i < n && peng < 0; i++)
                if (A->legal[i].kind == ACT_PENG) peng = i;
            if (peng >= 0 && shanten_raw(A->c, A->opened) <= 1) {
                *choice = peng;
                return 0;
            }
        }
    }
    *choice = best;
    return 0;
}
