/* The training game's hot loop in C: the hand engine (domain/engine.py
 * HandState), the fixed pool's race members deciding on that state
 * (_fast.race_act without an Observation), the learner's encoding
 * (guided/encoding.py over selfplay_core/encoding.py) and the default
 * privileged Critic features (guided/encoding.py oracle).
 *
 * Everything is a port of the Python it replaces, with the same results bit
 * for bit: the same wall (CPython's Mersenne Twister and shuffle), the same
 * legal actions in the same order, the same floats rounded the same way (the
 * loader builds without fused multiply-adds). guided/fastgame.py drives it and
 * falls back to the Python engine wherever this port does not reach (hooks
 * that read the Python state beyond its plain fields, teachers, other rule
 * versions). It includes the bound runtime's _fast.c, so its primitives are
 * the ones the Python engine calls through that runtime, and fp-rule-131
 * whole (rule131.h), which decides here as a race member does.
 */
#include "_fast.c"
#include "rule131.h"

#define G_WALL 136
#define G_EVENTS 512
#define G_RIVER 80
#define G_HAND 18
#define G_LOG 2048

enum { PH_DRAW, PH_PENG, PH_CHI, PH_SETTLED };
/* Event kinds carry selfplay_core.encoding.EVENT_KINDS' codes. */
enum { EV_DRAW = 1, EV_DISCARD = 2, EV_CHI = 3, EV_PENG = 4, EV_GANG = 5, EV_END = 6 };
enum { A_HU, A_DISCARD, A_PENG, A_CHI, A_GANG, A_PASS };
enum { K_EXPOSED, K_CONCEALED, K_ADDED };

/* ---------- CPython's random.Random(seed).shuffle ---------- */

typedef struct {
    uint32_t mt[624];
    int index;
} g_mt;

static void mt_init(g_mt *r, uint32_t s) {
    r->mt[0] = s;
    for (int i = 1; i < 624; i++)
        r->mt[i] = 1812433253U * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i;
    r->index = 624;
}

static void mt_init_array(g_mt *r, const uint32_t *key, int n) {
    mt_init(r, 19650218U);
    uint32_t *mt = r->mt;
    int i = 1, j = 0;
    for (int k = 624 > n ? 624 : n; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + (uint32_t)j;
        i++;
        j++;
        if (i >= 624) { mt[0] = mt[623]; i = 1; }
        if (j >= n) j = 0;
    }
    for (int k = 623; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
        i++;
        if (i >= 624) { mt[0] = mt[623]; i = 1; }
    }
    mt[0] = 0x80000000U;
    r->index = 624;
}

static uint32_t mt_next(g_mt *r) {
    static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
    uint32_t y, *mt = r->mt;
    if (r->index >= 624) {
        int kk;
        for (kk = 0; kk < 624 - 397; kk++) {
            y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
            mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (; kk < 623; kk++) {
            y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
            mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        y = (mt[623] & 0x80000000U) | (mt[0] & 0x7fffffffU);
        mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 0x1U];
        r->index = 0;
    }
    y = mt[r->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

/* random.Random(seed) for a nonnegative seed below 2**64. */
static void mt_seed(g_mt *r, uint64_t seed) {
    uint32_t key[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    mt_init_array(r, key, key[1] ? 2 : 1);
}

/* Random._randbelow_with_getrandbits. */
static int mt_below(g_mt *r, int n) {
    int k = 0;
    while ((1 << k) <= n) k++;  /* n.bit_length() */
    uint32_t v = mt_next(r) >> (32 - k);
    while ((int)v >= n) v = mt_next(r) >> (32 - k);
    return (int)v;
}

/* ---------- the hand ---------- */

typedef struct {
    int8_t kind, tile, n, from, concealed; /* kind: EV_CHI, EV_PENG or EV_GANG; from -1 for none */
    int8_t tiles[4];
} g_meld;

typedef struct {
    int8_t kind, seat, tile, priv, concealed, n; /* seat, tile, priv: -1 for None */
    int8_t tiles[4];
} g_event;

typedef struct {
    int kind, tile, u0, u1, gk; /* u0 < u1: the chi's two tiles from the hand; gk the kong's kind */
} g_action;

typedef struct {
    /* the hand's settings and its start */
    int dealer, hand_no, rounds_total, base_score, ycbk, max_steps;
    int scores[4];
    int8_t initial[4][13];
    /* the wall: wall[lo..hi), dealt from the back, kong replacements from the front */
    int8_t wall[G_WALL];
    int lo, hi;
    /* hands in insertion order, with their counts */
    int8_t hand[4][G_HAND];
    int n_hand[4];
    int c[4][NTILES];
    int white_acquired[4];
    g_meld melds[4][4];
    int n_melds[4];
    int8_t river[4][G_RIVER];
    int n_river[4];
    /* this hand's events, deals excluded */
    g_event ev[G_EVENTS];
    int n_ev;
    int phase, turn, drawn, kong_draw, catch_owner;
    int chain[4], piao[4];
    int queue[3], n_queue;
    int discarder, discarded, winner, fan, settled, steps;
    int delta[4];
    /* every applied action's index, for replaying the hand on the Python engine */
    int16_t log[G_LOG];
    int n_log;
    /* guided.walls: a stable wall (each seat's normal draws from its own
     * interleaved positions of the wall, counted from the back as of the
     * conversion; kong replacements from the front) and the learner's draw
     * luck (luck_seat -1: not tracked). */
    int stable, sub_back, sub_next[4], sub_off[4], n_left;
    uint8_t used[G_WALL];
    int luck_seat;
    double luck_pay, luck_deal;
    /* and its race luck (guided.walls, cfg["luck_kind"] "race") when race_on */
    int race_on;
    double race_pay, race_deal;
} g_hand;

static int wall_len(const g_hand *h) { return h->stable ? h->n_left : h->hi - h->lo; }

/* The remaining wall's tiles in wall order (a stable wall skips the drawn positions). */
static int g_wall_tiles(const g_hand *h, int8_t *out) {
    int n = 0;
    for (int k = h->lo; k < h->hi; k++)
        if (!h->stable || !h->used[k]) out[n++] = h->wall[k];
    return n;
}

static void g_event_add(g_hand *h, int kind, int seat, int tile, const int8_t *tiles, int n, int priv) {
    if (h->n_ev >= G_EVENTS) return;  /* checked by the caller's step budget long before */
    g_event *e = &h->ev[h->n_ev++];
    e->kind = (int8_t)kind;
    e->seat = (int8_t)seat;
    e->tile = (int8_t)tile;
    e->priv = (int8_t)priv;
    e->concealed = kind == EV_GANG && priv >= 0;
    e->n = (int8_t)n;
    for (int i = 0; i < n && i < 4; i++) e->tiles[i] = tiles[i];
}

static void g_take(g_hand *h, int seat, int tile) {
    h->hand[seat][h->n_hand[seat]++] = (int8_t)tile;
    h->c[seat][tile]++;
}

/* list.remove: the first copy, keeping the others in order. */
static int g_remove(g_hand *h, int seat, int tile) {
    int8_t *x = h->hand[seat];
    int n = h->n_hand[seat];
    for (int i = 0; i < n; i++) {
        if (x[i] == tile) {
            memmove(x + i, x + i + 1, (size_t)(n - i - 1));
            h->n_hand[seat]--;
            h->c[seat][tile]--;
            return 0;
        }
    }
    PyErr_SetString(PyExc_ValueError, "list.remove(x): x not in list");
    return -1;
}

static int g_fan(const g_hand *h) {
    if (h->phase != PH_DRAW) return 0;
    int s = h->turn;
    return (int)fan_value(h->c[s], h->drawn, h->n_melds[s], h->chain[s], h->piao[s], h->ycbk, h->kong_draw);
}

/* rules.payments. */
static void g_payments(g_hand *h, int winner, int fan) {
    for (int s = 0; s < 4; s++) h->delta[s] = 0;
    for (int s = 0; s < 4; s++) {
        if (s == winner) continue;
        int loss = h->base_score * fan * (h->dealer == s || h->dealer == winner ? 8 : 1);
        h->delta[s] = -loss;
        h->delta[winner] += loss;
    }
}

static void g_finish(g_hand *h, int winner) {
    h->winner = winner;
    h->settled = 1;
    h->phase = PH_SETTLED;
    if (winner >= 0) g_payments(h, winner, h->fan);
    g_event_add(h, EV_END, winner, -1, NULL, 0, -1);
}

/* The next tile a seat draws (guided.walls.TrainingHand._take). */
static int g_take_wall(g_hand *h, int seat, int kong) {
    if (!h->stable) return kong ? h->wall[h->lo++] : h->wall[--h->hi];
    int k;
    if (kong) {
        for (k = h->lo; h->used[k]; k++) {}
    } else {
        int target = h->sub_back - (4 * h->sub_next[seat] + h->sub_off[seat]);
        h->sub_next[seat]++;
        if (target >= h->lo && target < h->hi && !h->used[target]) {
            k = target;
        } else {
            for (k = h->hi - 1; h->used[k]; k--) {}
        }
    }
    h->used[k] = 1;
    h->n_left--;
    while (h->hi > h->lo && h->used[h->hi - 1]) h->hi--;
    while (h->lo < h->hi && h->used[h->lo]) h->lo++;
    return h->wall[k];
}

/* The learner's payoff (base units) if `seat` won now with `fan`. */
static double g_luck_payoff(const g_hand *h, int seat, long fan) {
    int learner = h->luck_seat;
    if (seat == learner) {
        double total = 0.0;
        for (int t = 0; t < 4; t++)
            if (t != seat) total += (double)(fan * (h->dealer == t || h->dealer == seat ? 8 : 1));
        return total;
    }
    return -(double)(fan * (h->dealer == learner || h->dealer == seat ? 8 : 1));
}

/* Before `seat` draws: whether its hand is ready (else its draw carries no
 * luck), and the means over the remaining wall of the payoff and of the
 * learner dealing next if it won at once with that tile. */
static int g_luck_expect(const g_hand *h, int seat, int kong, double *pay, double *deal) {
    int c[NTILES], w[NTILES] = {0}, opened = h->n_melds[seat], n = 0;
    *pay = *deal = 0.0;
    memcpy(c, h->c[seat], sizeof(c));
    if (shanten_raw(c, opened) > 0) return 0;
    for (int k = h->lo; k < h->hi; k++)
        if (!h->stable || !h->used[k]) {
            w[h->wall[k]]++;
            n++;
        }
    for (int y = 0; y < NTILES; y++) {
        if (!w[y]) continue;
        c[y]++;
        long fan = fan_value(c, y, opened, h->chain[seat], h->piao[seat], h->ycbk, kong);
        c[y]--;
        if (fan > 0) {
            *pay += (double)w[y] * g_luck_payoff(h, seat, fan);
            if (seat == h->luck_seat) *deal += (double)w[y];
        }
    }
    *pay /= (double)n;
    *deal /= (double)n;
    return 1;
}

/* guided.walls.race_value: R (pay, deal) of a wall of n tiles, seats drawing in
 * turn from `first`; racers in seat order, each with hits > 0 copies that win
 * for it and sum, the learner's payoffs summed over them. */
#define G_RACE_DRAWS 32 /* a seat's own draws: at most (136 - 20 + 3) / 4 */

static void g_race(int n, int first, int nr, const int *seat, const int *hits, const double *sum, int learner,
                   int learner_deals, double *pay, double *deal) {
    int draws = n - 20, off[4], m[4];
    double miss[4][G_RACE_DRAWS + 1];
    for (int i = 0; i < nr; i++) {
        off[i] = ((seat[i] - first) % 4 + 4) % 4;
        m[i] = draws > off[i] ? (draws - off[i] + 3) / 4 : 0;
        if (m[i] > G_RACE_DRAWS) m[i] = G_RACE_DRAWS; /* never: a wall holds 136 tiles */
        miss[i][0] = 1.0;
        for (int j = 0; j < m[i]; j++) {
            int keep = n - hits[i] - j;
            miss[i][j + 1] = miss[i][j] * ((double)(keep > 0 ? keep : 0) / (double)(n - j));
        }
    }
    double total = 0.0, none = 1.0, mine = 0.0;
    for (int i = 0; i < nr; i++) {
        double first_p = 0.0;
        for (int j = 0; j < m[i]; j++) {
            int k = off[i] + 4 * j;
            double others = 1.0;
            for (int q = 0; q < nr; q++)
                if (q != i) others *= miss[q][k > off[q] ? (k - off[q] + 3) / 4 : 0];
            first_p += miss[i][j] * (double)hits[i] / (double)(n - j) * others;
        }
        total += first_p * (sum[i] / (double)hits[i]);
        if (seat[i] == learner) mine = first_p;
        none *= miss[i][m[i]];
    }
    *pay = total;
    *deal = learner_deals ? mine + none : mine;
}

/* TrainingHand._race_values: before `seat` draws, v(y) per tile y of the wall
 * (vp, vd) and their means over it; 0 when nobody can win (no race luck). */
static int g_race_values(const g_hand *h, int seat, int kong, double *vp, double *vd, double *ep, double *ed) {
    int w[NTILES] = {0}, n = 0;
    for (int k = h->lo; k < h->hi; k++)
        if (!h->stable || !h->used[k]) {
            w[h->wall[k]]++;
            n++;
        }
    int nr = 0, seats[4], hits[4], wins[4][NTILES], now[NTILES] = {0}, anything_now = 0;
    double sums[4], pays[4][NTILES], now_pay[NTILES];
    for (int p = 0; p < 4; p++) {
        int c[NTILES], opened = h->n_melds[p], got = 0;
        memcpy(c, h->c[p], sizeof(c));
        if (shanten_raw(c, opened) > 0) continue;
        double total = 0.0;
        for (int y = 0; y < NTILES; y++) {
            wins[nr][y] = 0;
            if (!w[y]) continue;
            c[y]++;
            long fan = fan_value(c, y, opened, h->chain[p], h->piao[p], h->ycbk, 0), at_once = fan;
            if (p == seat && kong) at_once = fan_value(c, y, opened, h->chain[p], h->piao[p], h->ycbk, 1);
            c[y]--;
            if (p == seat && at_once > 0) {
                now[y] = 1;
                now_pay[y] = g_luck_payoff(h, p, at_once);
                anything_now = 1;
            }
            if (fan > 0) {
                wins[nr][y] = 1;
                pays[nr][y] = g_luck_payoff(h, p, fan);
                got += w[y];
                total += (double)w[y] * pays[nr][y];
            }
        }
        if (got > 0) {
            seats[nr] = p;
            hits[nr] = got;
            sums[nr] = total;
            nr++;
        }
    }
    if (!nr && !anything_now) return 0;
    int learner = h->luck_seat, next = (seat + 1) % 4, deals = h->dealer == learner;
    double missed_pay, missed_deal, pay = 0.0, deal = 0.0;
    g_race(n - 1, next, nr, seats, hits, sums, learner, deals, &missed_pay, &missed_deal);
    for (int y = 0; y < NTILES; y++) {
        if (!w[y]) continue;
        double vy_pay, vy_deal;
        int waited = 0;
        for (int i = 0; i < nr; i++) waited |= wins[i][y];
        if (now[y]) {
            vy_pay = now_pay[y];
            vy_deal = seat == learner ? 1.0 : 0.0;
        } else if (!waited) {
            vy_pay = missed_pay;
            vy_deal = missed_deal;
        } else {
            int ls[4], lh[4], nl = 0;
            double lt[4];
            for (int i = 0; i < nr; i++) {
                if (!wins[i][y]) {
                    ls[nl] = seats[i];
                    lh[nl] = hits[i];
                    lt[nl++] = sums[i];
                } else if (hits[i] > 1) {
                    ls[nl] = seats[i];
                    lh[nl] = hits[i] - 1;
                    lt[nl++] = sums[i] - pays[i][y];
                }
            }
            g_race(n - 1, next, nl, ls, lh, lt, learner, deals, &vy_pay, &vy_deal);
        }
        vp[y] = vy_pay;
        vd[y] = vy_deal;
        pay += (double)w[y] * vy_pay;
        deal += (double)w[y] * vy_deal;
    }
    *ep = pay / (double)n;
    *ed = deal / (double)n;
    return 1;
}

static void g_draw(g_hand *h, int seat, int kong, int initial) {
    if (wall_len(h) <= 20) {
        g_finish(h, -1);
        return;
    }
    double pay = 0.0, deal = 0.0, race_pay[NTILES], race_deal[NTILES], race_mean_pay = 0.0, race_mean_deal = 0.0;
    int luck = !initial && h->luck_seat >= 0 && g_luck_expect(h, seat, kong, &pay, &deal);
    int race = !initial && h->luck_seat >= 0 && h->race_on &&
               g_race_values(h, seat, kong, race_pay, race_deal, &race_mean_pay, &race_mean_deal);
    h->turn = seat;
    h->drawn = g_take_wall(h, seat, kong);
    g_take(h, seat, h->drawn);
    if (h->drawn == WHITE) h->white_acquired[seat]++;
    h->kong_draw = kong;
    h->phase = PH_DRAW;
    if (!initial) g_event_add(h, EV_DRAW, seat, h->drawn, NULL, 0, seat);
    if (luck) {
        long fan = fan_value(h->c[seat], h->drawn, h->n_melds[seat], h->chain[seat], h->piao[seat], h->ycbk, kong);
        double got = fan > 0 ? g_luck_payoff(h, seat, fan) : 0.0;
        double dealt = fan > 0 && seat == h->luck_seat ? 1.0 : 0.0;
        h->luck_pay += got - pay;
        h->luck_deal += dealt - deal;
    }
    if (race) {
        h->race_pay += race_pay[h->drawn] - race_mean_pay;
        h->race_deal += race_deal[h->drawn] - race_mean_deal;
    }
}

/* guided.walls.stabilize: from here on each seat draws from its own positions. */
static void g_stabilize(g_hand *h) {
    if (h->stable) return;
    int next = h->phase == PH_DRAW ? (h->turn + 1) % 4 : (h->discarder + 1) % 4;
    h->stable = 1;
    memset(h->used, 0, sizeof(h->used));
    h->n_left = h->hi - h->lo;
    h->sub_back = h->hi - 1;
    for (int s = 0; s < 4; s++) {
        h->sub_next[s] = 0;
        h->sub_off[s] = ((s - next) % 4 + 4) % 4;
    }
}

static void g_start(g_hand *h, uint64_t seed, int dealer, int hand_no, const int *scores,
                    int rounds_total, int base_score, int ycbk, int max_steps) {
    memset(h, 0, sizeof(*h));
    h->dealer = dealer;
    h->hand_no = hand_no;
    h->rounds_total = rounds_total;
    h->base_score = base_score;
    h->ycbk = ycbk;
    h->max_steps = max_steps;
    for (int s = 0; s < 4; s++) h->scores[s] = scores[s];
    for (int t = 0; t < NTILES; t++)
        for (int k = 0; k < 4; k++) h->wall[4 * t + k] = (int8_t)t;
    g_mt r;
    mt_seed(&r, seed);
    for (int i = G_WALL - 1; i > 0; i--) {
        int j = mt_below(&r, i + 1);
        int8_t x = h->wall[i];
        h->wall[i] = h->wall[j];
        h->wall[j] = x;
    }
    h->lo = 0;
    h->hi = G_WALL;
    for (int k = 0; k < 13; k++)
        for (int s = 0; s < 4; s++) g_take(h, s, h->wall[--h->hi]);
    for (int s = 0; s < 4; s++) {
        h->white_acquired[s] = h->c[s][WHITE];
        memcpy(h->initial[s], h->hand[s], 13);
    }
    h->turn = dealer;
    h->drawn = -1;
    h->luck_seat = -1;
    h->catch_owner = -1;
    h->discarder = h->discarded = h->winner = -1;
    g_draw(h, dealer, 0, 1);
}

/* HandState.legal_actions, in its order. */
static int g_legal(const g_hand *h, g_action *out) {
    int n = 0;
    if (h->settled) return 0;
    int s = h->turn;
    const int *c = h->c[s];
    int restricted = h->catch_owner >= 0 && h->catch_owner != s;
    if (h->phase == PH_DRAW) {
        if (g_fan(h)) out[n++] = (g_action){A_HU, -1, -1, -1, -1};
        if (restricted) {
            if (h->drawn >= 0) out[n++] = (g_action){A_DISCARD, h->drawn, -1, -1, -1};
        } else {
            for (int t = 0; t < NTILES; t++)
                if (c[t]) out[n++] = (g_action){A_DISCARD, t, -1, -1, -1};
        }
        if (wall_len(h) > 20) {
            for (int t = 0; t < WHITE; t++)
                if (c[t] == 4) out[n++] = (g_action){A_GANG, t, -1, -1, K_CONCEALED};
            if (!restricted)
                for (int i = 0; i < h->n_melds[s]; i++) {
                    const g_meld *m = &h->melds[s][i];
                    if (m->kind == EV_PENG && c[m->tiles[0]])
                        out[n++] = (g_action){A_GANG, m->tiles[0], -1, -1, K_ADDED};
                }
        }
        return n;
    }
    out[n++] = (g_action){A_PASS, -1, -1, -1, -1};
    int t = h->discarded;
    if (t < 0 || t == WHITE || restricted) return n;
    if (h->phase == PH_PENG) {
        if (c[t] >= 2) out[n++] = (g_action){A_PENG, t, -1, -1, -1};
        if (c[t] == 3 && wall_len(h) > 20) out[n++] = (g_action){A_GANG, t, -1, -1, K_EXPOSED};
    } else if (h->phase == PH_CHI && t < 27) {
        int chis = 0;
        for (int i = 0; i < h->n_melds[s]; i++) chis += h->melds[s][i].kind == EV_CHI;
        if (chis < 2) {
            int base = t / 9 * 9;
            int lo = t - 2 > base ? t - 2 : base, hi = t < base + 6 ? t : base + 6;
            for (int start = lo; start <= hi; start++) {
                int u[2], k = 0;
                for (int x = start; x < start + 3; x++)
                    if (x != t) u[k++] = x;
                if (c[u[0]] && c[u[1]]) out[n++] = (g_action){A_CHI, t, u[0], u[1], -1};
            }
        }
    }
    return n;
}

/* selfplay_core.actions.action_index. */
static int g_index(const g_action *a) {
    switch (a->kind) {
    case A_DISCARD: return a->tile;
    case A_CHI: {
        int low = a->u0 < a->tile ? a->u0 : a->tile;
        return 34 + low / 9 * 21 + low % 9 * 3 + (a->tile - low);
    }
    case A_PENG: return 97 + a->tile;
    case A_GANG: return (a->gk == K_EXPOSED ? 130 : a->gk == K_CONCEALED ? 163 : 196) + a->tile;
    case A_HU: return 229;
    default: return 230;
    }
}

/* rules.is_baotou of a hand's counts, through the runtime's is_baotou. */
static int g_is_baotou(const int *c, int opened) {
    int size = 0;
    for (int t = 0; t < NTILES; t++) size += c[t];
    if (size != 13 - 3 * opened || !c[WHITE] || opened < 0 || opened > 4) return 0;
    return baotou_cached(c, opened);
}

/* observations.can_discard_piao for the seat to move, from this hand's events. */
static int g_piao(const g_hand *h, int s) {
    int c[NTILES];
    memcpy(c, h->c[s], sizeof(c));
    int opened = h->n_melds[s];
    if (!c[WHITE]) return 0;
    c[WHITE]--;
    int ok = g_is_baotou(c, opened);
    c[WHITE]++;
    if (!ok) return 0;
    if (h->drawn >= 0) {
        c[h->drawn]--;
        return g_is_baotou(c, opened);
    }
    if (h->catch_owner == s) return 1;
    for (int i = h->n_ev - 1; i >= 0; i--) {
        const g_event *e = &h->ev[i];
        if (e->seat != s || e->kind == EV_END) continue;
        if ((e->kind != EV_CHI && e->kind != EV_PENG) || opened < 1) return 0;
        int held = -1;
        for (int k = 0; k < e->n; k++)
            if (e->tiles[k] == e->tile) { held = k; break; }
        if (e->n != 3 || held < 0) return 0;
        for (int k = 0; k < 3; k++)
            if (k != held) c[e->tiles[k]]++;
        return g_is_baotou(c, opened - 1);
    }
    return 0;
}

static void g_sorted(int8_t *tiles, int n) {
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && tiles[j - 1] > tiles[j]; j--) {
            int8_t x = tiles[j];
            tiles[j] = tiles[j - 1];
            tiles[j - 1] = x;
        }
}

/* HandState.step for a legal action. */
static int g_step(g_hand *h, const g_action *a) {
    if (h->n_ev > G_EVENTS - 2) {  /* a hand never comes near: each event takes a step */
        PyErr_SetString(PyExc_RuntimeError, "hand event log overflow");
        return -1;
    }
    h->steps++;
    if (h->n_log < G_LOG) h->log[h->n_log++] = (int16_t)g_index(a);
    int s = h->turn;
    switch (a->kind) {
    case A_HU:
        h->fan = g_fan(h);
        g_finish(h, s);
        return 0;
    case A_DISCARD: {
        int t = a->tile;
        int piao = t == WHITE && g_piao(h, s);
        if (g_remove(h, s, t) < 0) return -1;
        if (piao) {
            h->chain[s]++;
            h->piao[s]++;
        } else {
            h->chain[s] = h->piao[s] = 0;
        }
        if (h->n_river[s] >= G_RIVER) {
            PyErr_SetString(PyExc_RuntimeError, "river overflow");
            return -1;
        }
        h->river[s][h->n_river[s]++] = (int8_t)t;
        g_event_add(h, EV_DISCARD, s, t, NULL, 0, -1);
        h->discarder = s;
        h->discarded = t;
        h->drawn = -1;
        h->kong_draw = 0;
        if (t == WHITE) h->catch_owner = s;
        else if (h->catch_owner == s) h->catch_owner = -1;
        if (t == WHITE) {
            g_draw(h, (s + 1) % 4, 0, 0);
        } else {
            h->phase = PH_PENG;
            if (h->catch_owner >= 0) {
                h->queue[0] = h->catch_owner;
                h->n_queue = 1;
            } else {
                for (int i = 0; i < 3; i++) h->queue[i] = (s + i + 1) % 4;
                h->n_queue = 3;
            }
            h->turn = h->queue[0];
            h->n_queue--;
            memmove(h->queue, h->queue + 1, sizeof(int) * (size_t)h->n_queue);
        }
        return 0;
    }
    case A_PASS:
        if (h->phase == PH_PENG && h->n_queue) {
            h->turn = h->queue[0];
            h->n_queue--;
            memmove(h->queue, h->queue + 1, sizeof(int) * (size_t)h->n_queue);
        } else if (h->phase == PH_PENG && (h->catch_owner < 0 || h->catch_owner == (h->discarder + 1) % 4)) {
            h->phase = PH_CHI;
            h->turn = (h->discarder + 1) % 4;
        } else {
            g_draw(h, (h->discarder + 1) % 4, 0, 0);
        }
        return 0;
    case A_CHI:
    case A_PENG: {
        int8_t tiles[3];
        int used[2];
        if (a->kind == A_CHI) { used[0] = a->u0; used[1] = a->u1; }
        else { used[0] = used[1] = a->tile; }
        for (int i = 0; i < 2; i++)
            if (g_remove(h, s, used[i]) < 0) return -1;
        h->n_river[h->discarder]--;
        tiles[0] = (int8_t)used[0];
        tiles[1] = (int8_t)used[1];
        tiles[2] = (int8_t)a->tile;
        g_sorted(tiles, 3);
        g_meld *m = &h->melds[s][h->n_melds[s]++];
        m->kind = a->kind == A_CHI ? EV_CHI : EV_PENG;
        m->n = 3;
        memcpy(m->tiles, tiles, 3);
        m->tile = (int8_t)a->tile;
        m->from = (int8_t)h->discarder;
        m->concealed = 0;
        g_event_add(h, m->kind, s, a->tile, tiles, 3, -1);
        h->phase = PH_DRAW;
        h->drawn = -1;
        h->kong_draw = 0;
        if (h->catch_owner != s) h->chain[s] = h->piao[s] = 0;
        return 0;
    }
    case A_GANG: {
        int t = a->tile;
        int8_t four[4] = {(int8_t)t, (int8_t)t, (int8_t)t, (int8_t)t};
        if (a->gk == K_EXPOSED) {
            for (int i = 0; i < 3; i++)
                if (g_remove(h, s, t) < 0) return -1;
            h->n_river[h->discarder]--;
            g_meld *m = &h->melds[s][h->n_melds[s]++];
            *m = (g_meld){EV_GANG, (int8_t)t, 4, (int8_t)h->discarder, 0, {0}};
            memcpy(m->tiles, four, 4);
        } else if (a->gk == K_CONCEALED) {
            for (int i = 0; i < 4; i++)
                if (g_remove(h, s, t) < 0) return -1;
            g_meld *m = &h->melds[s][h->n_melds[s]++];
            *m = (g_meld){EV_GANG, (int8_t)t, 4, -1, 1, {0}};
            memcpy(m->tiles, four, 4);
        } else {
            if (g_remove(h, s, t) < 0) return -1;
            for (int i = 0; i < h->n_melds[s]; i++) {
                g_meld *m = &h->melds[s][i];
                if (m->kind == EV_PENG && m->tiles[0] == t) {
                    m->kind = EV_GANG;
                    m->n = 4;
                    memcpy(m->tiles, four, 4);
                    break;
                }
            }
        }
        h->chain[s]++;
        g_event_add(h, EV_GANG, s, t, four, 4, a->gk == K_CONCEALED ? s : -1);
        g_draw(h, s, 1, 0);
        return 0;
    }
    }
    PyErr_SetString(PyExc_ValueError, "unknown action");
    return -1;
}

/* ---------- the public view of a seat ---------- */

/* Unseen copies as sdk.shape.remaining_tiles counts them from an Observation:
 * the seat's hand, every river, every meld the seat may see. */
static void g_available(const g_hand *h, int seat, int *avail) {
    for (int t = 0; t < NTILES; t++) avail[t] = 4;
    for (int t = 0; t < NTILES; t++) avail[t] -= h->c[seat][t];
    for (int s = 0; s < 4; s++)
        for (int i = 0; i < h->n_river[s]; i++) avail[h->river[s][i]]--;
    for (int s = 0; s < 4; s++)
        for (int i = 0; i < h->n_melds[s]; i++) {
            const g_meld *m = &h->melds[s][i];
            if (m->concealed && s != seat) continue;
            for (int k = 0; k < m->n; k++) avail[m->tiles[k]]--;
        }
    for (int t = 0; t < NTILES; t++)
        if (avail[t] < 0) avail[t] = 0;
}

static int g_hidden(const g_event *e, int viewer) {
    return (e->priv >= 0 && e->priv != viewer) || (e->kind == EV_DRAW && e->seat != viewer) ||
           (e->concealed && e->seat != viewer);
}

/* ---------- a race member's decision (_fast.race_act) ---------- */

/* Observation stand-ins for what race_act still reads through attributes:
 * the last discard's seat (Race.value of a pass) and the catch-play circle's
 * owner (a call's piao). [discarder][owner + 1]. */
static PyObject *g_stub[4][5];

static int g_stubs(void) {
    PyObject *types = PyImport_ImportModule("types");
    if (!types) return -1;
    PyObject *ns = PyObject_GetAttrString(types, "SimpleNamespace");
    Py_DECREF(types);
    if (!ns) return -1;
    int rc = 0;
    for (int d = 0; d < 4 && !rc; d++) {
        PyObject *kw = Py_BuildValue("{s:s,s:i}", "kind", "discard", "seat", d);
        PyObject *empty = PyTuple_New(0);
        PyObject *event = kw && empty ? PyObject_Call(ns, empty, kw) : NULL;
        Py_XDECREF(kw);
        for (int g = -1; g < 4 && event; g++) {
            PyObject *kw2 = Py_BuildValue("{s:(O),s:i}", "public_events", event, "god_discarder_seat", g);
            g_stub[d][g + 1] = kw2 ? PyObject_Call(ns, empty, kw2) : NULL;
            Py_XDECREF(kw2);
            if (!g_stub[d][g + 1]) rc = -1;
        }
        if (!event) rc = -1;
        Py_XDECREF(event);
        Py_XDECREF(empty);
    }
    Py_DECREF(ns);
    return rc;
}

/* Race.piao of act_piao, from the state instead of the Observation. */
static int g_act_piao(const g_hand *h, int seat, const int *hc, int size, int opened) {
    int c[NTILES];
    memcpy(c, hc, sizeof(c));
    if (!c[WHITE]) return 0;
    c[WHITE]--;
    int ok = is_baotou_counts(c, size - 1, opened);
    c[WHITE]++;
    if (!ok) return 0;
    int drawn = h->phase == PH_DRAW ? h->drawn : -1;
    if (drawn >= 0) {
        c[drawn]--;
        return is_baotou_counts(c, size - 1, opened);
    }
    if (h->catch_owner >= 0 && h->catch_owner == seat) return 1;
    for (int i = h->n_ev - 1; i >= 0; i--) {
        const g_event *e = &h->ev[i];
        if (e->seat != seat || e->kind == EV_END) continue;
        if ((e->kind == EV_CHI || e->kind == EV_PENG) && opened >= 1) {
            int held = -1;
            for (int k = 0; k < e->n && e->n == 3; k++)
                if (e->tiles[k] == e->tile) { held = k; break; }
            if (e->n == 3 && held >= 0) {
                for (int k = 0; k < 3; k++)
                    if (k != held) c[e->tiles[k]]++;
                return is_baotou_counts(c, size + 2, opened - 1);
            }
        }
        return 0;
    }
    return 0;
}

static void g_race_action(const g_action *a, race_action *out) {
    memset(out, 0, sizeof(*out));
    static const int kinds[6] = {ACT_HU, ACT_DISCARD, ACT_PENG, ACT_CHI, ACT_GANG, ACT_PASS};
    out->kind = kinds[a->kind];
    out->tile = a->tile;
    if (a->kind == A_CHI) {
        out->n_tiles = 2;
        out->tiles[0] = a->u0;
        out->tiles[1] = a->u1;
    }
    if (a->kind == A_GANG) {
        out->kong = a->gk == K_CONCEALED ? 4 : a->gk == K_EXPOSED ? 3 : 1;
        out->added = a->gk == K_ADDED;
    }
}

static PyObject *g_int_tuple(const int *v, int n) {
    PyObject *t = PyTuple_New(n);
    if (!t) return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *x = PyLong_FromLong(v[i]);
        if (!x) { Py_DECREF(t); return NULL; }
        PyTuple_SET_ITEM(t, i, x);
    }
    return t;
}

/* r68_view of the seat to move (the third pool's members, fp-rule-068 whole in C),
 * as the Observation shows it: the hand sorted, each river, the calls and
 * discards of this hand, the catch-play owner and the drawn tile. (Before the
 * hand's first event, at the dealer's first turn, opponent_speed reads the
 * previous hand's calls instead; it is at its floor 0.1 there whatever they
 * are: no discards or melds yet, sixteen draws left.) */
static void g_r68_view(const g_hand *h, int s, r68_view *o) {
    memset(o, 0, sizeof(*o));
    o->wall = wall_len(h);
    o->stage = o->wall >= 70 ? 0 : o->wall >= 45 ? 1 : 2;
    o->god = h->catch_owner;
    o->drawn = h->phase == PH_DRAW ? h->drawn : -1;
    o->dealer = h->dealer;
    for (int t = 0; t < NTILES; t++)
        for (int k = 0; k < h->c[s][t] && o->n_hand < 18; k++) o->hand[o->n_hand++] = t;
    for (int i = 0; i < h->n_melds[s]; i++) {
        if (h->melds[s][i].kind == EV_CHI) o->chis++;
        if (h->melds[s][i].kind == EV_PENG && o->n_pungs < 4) o->pungs[o->n_pungs++] = h->melds[s][i].tiles[0];
    }
    for (int seat = 0; seat < 4; seat++) {
        int n = h->n_river[seat], from = n > 6 ? n - 6 : 0;
        o->melds[seat] = h->n_melds[seat];
        o->n_discards[seat] = n;
        o->n_last[seat] = n - from;
        for (int i = 0; i < n; i++) {
            if (h->river[seat][i] == WHITE) o->white_out[seat] = 1;
            if (i >= from) o->last[seat][i - from] = h->river[seat][i];
        }
        o->since[seat] = -1;
    }
    o->ahead = h->discarder >= 0 ? ((s - h->discarder - 1) % 4 + 4) % 4 : -1;
    for (int i = 0; i < h->n_ev; i++) {
        const g_event *e = &h->ev[i];
        if (e->seat < 0 || e->seat > 3) continue;
        if (e->kind == EV_CHI || e->kind == EV_PENG || e->kind == EV_GANG) o->since[e->seat] = 0;
        else if (e->kind == EV_DISCARD && o->since[e->seat] >= 0) o->since[e->seat]++;
    }
}

/* A seat of fp-rule-131 (rule131.h) in place of a race profile: rule131() registers its profiles. */
#define G_R131 (1 << 20)

/* py_race_act for the seat to move: the index of its choice among the legal actions.
 * G_R131 plays fp-rule-131 on the race profile of its role in this hand. */
static int g_member_act(const g_hand *h, const g_action *legal, int n_legal, int member,
                        PyObject *fallback, int *choice) {
    int profile = member;
    if (member == G_R131) {
        if (!r131.ready) {
            PyErr_SetString(PyExc_RuntimeError, "fp-rule-131's profiles are not registered");
            return -1;
        }
        /* SAVED_BY_ROLE[(dealer, min(3, max(0, rounds_total - round_no)))] */
        int left = h->rounds_total - h->hand_no;
        profile = r131.role[h->dealer == h->turn][left < 0 ? 0 : left > 3 ? 3 : left];
    }
    if (profile < 0 || profile >= race_profile_count) {
        PyErr_SetString(PyExc_ValueError, "unknown race profile");
        return -1;
    }
    if (n_legal > 64 || n_legal < 1) {
        PyErr_SetString(PyExc_ValueError, "legal actions out of range");
        return -1;
    }
    race_act_ctx *A = PyMem_Calloc(1, sizeof(race_act_ctx));
    if (!A) { PyErr_NoMemory(); return -1; }
    int rc = -1, avail[NTILES];
    PyObject *available = NULL;
    const race_profile *p = &race_profiles[profile];
    int s = h->turn;
    A->seat = s;
    memcpy(A->c, h->c[s], sizeof(A->c));
    A->size = h->n_hand[s];
    A->opened = h->n_melds[s];
    A->obs = g_stub[h->discarder >= 0 ? h->discarder : 0][h->catch_owner + 1];
    g_available(h, s, avail);
    if (!(available = g_int_tuple(avail, NTILES))) goto done;
    A->chain = h->chain[s];
    A->piao = h->piao[s];
    A->hu_fan = g_fan(h);
    {
        int left = -floor_div(-(wall_len(h) - p->last_wall), 4);
        A->draws = left > 1 ? left : 1;
    }
    if (race_bind(&A->x, available, profile, fallback) < 0) goto done;
    A->n_legal = n_legal;
    for (int i = 0; i < n_legal; i++) g_race_action(&legal[i], &A->legal[i]);
    A->piao_known = 1;
    A->piao_ok = g_act_piao(h, s, A->c, A->size, A->opened);
    if (member == G_R131) {
        r131_view w;
        g_r68_view(h, s, &w.o);
        w.response = h->phase != PH_DRAW;
        rc = r131_act(A, &w, choice);
        goto done;
    }
    if (p->style >= 10) {
        r68_view o;
        g_r68_view(h, s, &o);
        rc = r68_act(A, &o, choice);
        goto done;
    }
    int has_hu = 0;
    for (int i = 0; i < A->n_legal; i++) has_hu |= A->legal[i].kind == ACT_HU;
    if (has_hu) {
        int d = act_double(A);
        if (d == -2) goto done;
        if (d >= 0) { *choice = d; rc = 0; goto done; }
    }
    int allowed[64];
    for (int i = 0; i < A->n_legal; i++) allowed[i] = 1;
    int style = p->style;
    if (style == 2) {
        int now;
        if (A->size % 3 == 2) {
            now = 1000;
            for (int t = 0; t < NTILES; t++) {
                if (!A->c[t]) continue;
                A->c[t]--;
                int v = sdk_shanten(A->c, A->opened);
                A->c[t]++;
                if (v < now) now = v;
            }
        } else {
            now = sdk_shanten(A->c, A->opened);
        }
        for (int i = 0; i < A->n_legal; i++) {
            race_action *a = &A->legal[i];
            if (a->kind != ACT_GANG) continue;
            int take = now > 0;
            if (!take) {
                int x[NTILES];
                memcpy(x, A->c, sizeof(x));
                x[a->tile] -= a->kong;
                take = sdk_shanten(x, a->added ? A->opened : A->opened + 1) <= 0;
            }
            if (take) { *choice = i; rc = 0; goto done; }
        }
    } else if (style == 3 || style == 4) {
        int kind = style == 3 ? ACT_PENG : ACT_CHI;
        if (A->size % 3 == 1) {
            int now = sdk_shanten(A->c, A->opened);
            for (int i = 0; i < A->n_legal; i++) {
                race_action *a = &A->legal[i];
                if (a->kind != kind) continue;
                int after;
                if (act_after_call(A, a, &after, NULL) < 0) goto done;
                int take = after < now;
                if (!take && after == now) {
                    long wide, own;
                    int ignore;
                    if (act_after_call(A, a, &ignore, &wide) < 0) goto done;
                    ctx_at(A, A->opened);
                    analysis_row row;
                    if (race_analysis(&A->x, A->c, 0, 0, &row) < 0) goto done;
                    own = row.ukeire;
                    take = wide > own;
                }
                if (take) { *choice = i; rc = 0; goto done; }
            }
        }
    } else if (style == 5) {
        if (A->size % 3 == 1 && sdk_shanten(A->c, A->opened) >= 3)
            for (int i = 0; i < A->n_legal; i++)
                if (A->legal[i].kind == ACT_CHI || A->legal[i].kind == ACT_PENG) allowed[i] = 0;
    }
    {
        int best = -1;
        double top = 0.0;
        for (int i = 0; i < A->n_legal; i++) {
            if (!allowed[i]) continue;
            double value;
            if (act_value(A, &A->legal[i], &value) < 0) goto done;
            if (best < 0 || value > top) {
                best = i;
                top = value;
            }
        }
        if (best < 0) {
            PyErr_SetString(PyExc_ValueError, "max() arg is an empty sequence");
            goto done;
        }
        *choice = best;
        rc = 0;
    }
done:
    Py_XDECREF(available);
    PyMem_Free(A);
    return rc;
}

/* ---------- the learner's encoding ---------- */

#define N_PLANES 57
#define N_SCALARS 79
#define N_ACTIONS 231

/* NumPy's float32 -> float16 cast: round to nearest, ties to even. */
static uint16_t g_half(float value) {
    uint32_t f;
    memcpy(&f, &value, 4);
    uint32_t f_exp, f_sig;
    uint16_t h_sgn = (uint16_t)((f & 0x80000000u) >> 16), h_exp, h_sig;
    f_exp = f & 0x7f800000u;
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            f_sig = f & 0x007fffffu;
            if (f_sig != 0) {
                uint16_t ret = (uint16_t)(0x7c00u + (f_sig >> 13));
                if (ret == 0x7c00u) ret++;
                return h_sgn + ret;
            }
            return (uint16_t)(h_sgn + 0x7c00u);
        }
        return (uint16_t)(h_sgn + 0x7c00u);
    }
    if (f_exp <= 0x38000000u) {
        if (f_exp < 0x33000000u) return h_sgn;
        f_exp >>= 23;
        f_sig = 0x00800000u + (f & 0x007fffffu);
        f_sig >>= (113 - f_exp);
        if (((f_sig & 0x00003fffu) != 0x00001000u) || (f & 0x000007ffu)) f_sig += 0x00001000u;
        h_sig = (uint16_t)(f_sig >> 13);
        return (uint16_t)(h_sgn + h_sig);
    }
    h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
    f_sig = f & 0x007fffffu;
    if ((f_sig & 0x00003fffu) != 0x00001000u) f_sig += 0x00001000u;
    h_sig = (uint16_t)(f_sig >> 13);
    return (uint16_t)(h_sgn + h_exp + h_sig);
}

static double g_symlog(double x) {
    return copysign(log1p(fabs(x)), x);
}

/* math.log1p as CPython calls it: zero keeps its sign. */
static double g_log1p(double x) {
    return x == 0.0 ? x : log1p(x);
}

typedef struct {
    float planes[N_PLANES * NTILES];
    float scalars[N_SCALARS];
    uint8_t mask[N_ACTIONS];
    int16_t events[G_EVENTS > 0 ? G_EVENTS * 5 : 5];
    int n_events;
} g_encoded;

/* The seat's visible meld tiles (others' concealed kongs redacted). */
static int g_meld_visible(const g_hand *h, int s, int viewer, int i) {
    return !(h->melds[s][i].concealed && s != viewer);
}

/* guided.encoding.encode for the seat to move: 1 done, 0 when the Python
 * encoder must decide (a hand past the winning boundary), -1 error. */
static int g_encode(const g_hand *h, const g_action *legal, int n_legal, int lookahead, g_encoded *out) {
    int seat = h->turn;
    float *planes = out->planes;
    memset(planes, 0, sizeof(out->planes));
    int opened = h->n_melds[seat];
    const int *hc = h->c[seat];
    int size = h->n_hand[seat];
    /* public planes (py_encode_public_planes) */
    int visible[NTILES];
    memcpy(visible, hc, sizeof(visible));
    count_planes(planes, hc);
    if (h->phase == PH_DRAW && h->drawn >= 0) planes[4 * NTILES + h->drawn] = 1.0f;
    for (int i = 0; i < 4; i++) {
        int physical = (seat + i) % 4;
        int counted[NTILES] = {0};
        for (int k = 0; k < h->n_melds[physical]; k++) {
            if (!g_meld_visible(h, physical, seat, k)) continue;
            const g_meld *m = &h->melds[physical][k];
            for (int j = 0; j < m->n; j++) counted[m->tiles[j]]++;
        }
        count_planes(planes + (5 + 4 * i) * NTILES, counted);
        for (int t = 0; t < NTILES; t++) visible[t] += counted[t];
        memset(counted, 0, sizeof(counted));
        for (int k = 0; k < h->n_river[physical]; k++) counted[h->river[physical][k]]++;
        count_planes(planes + (21 + 4 * i) * NTILES, counted);
        for (int t = 0; t < NTILES; t++) visible[t] += counted[t];
        if (h->n_river[physical]) planes[(37 + i) * NTILES + h->river[physical][h->n_river[physical] - 1]] = 1.0f;
    }
    int remaining[NTILES];
    for (int t = 0; t < NTILES; t++) remaining[t] = visible[t] < 4 ? 4 - visible[t] : 0;
    count_planes(planes + 41 * NTILES, remaining);
    /* constant suit and number planes */
    for (int t = 0; t < NTILES; t++) {
        planes[(52 + (t / 9 < 3 ? t / 9 : 3)) * NTILES + t] = 1.0f;
        planes[56 * NTILES + t] = (float)(t < 27 ? (t % 9) / 8.0 : 0.5);
    }
    memset(out->mask, 0, sizeof(out->mask));
    int kinds[6] = {0};
    for (int i = 0; i < n_legal; i++) {
        out->mask[g_index(&legal[i])] = 1;
        kinds[legal[i].kind] = 1;
    }
    for (int t = 0; t < NTILES; t++) planes[51 * NTILES + t] = out->mask[t] ? 1.0f : 0.0f;
    /* shape lookahead (py_encode_lookahead) */
    int ref_shanten = 8;
    long ref_ukeire = 0, ref_copies = 0, ref_value = 0;
    if (lookahead) {
        /* The race tables' meld table is built on first use (the members' race_bind). */
        if (!h->ycbk && meld_tables() < 0) return -1;
        if (size == 13 - 3 * opened || size == 14 - 3 * opened) {
            analysis_row rows[NTILES], best;
            int counts[NTILES];
            memcpy(counts, hc, sizeof(counts));
            int discard = size == 14 - 3 * opened, found = 0;
            for (int tile = 0; tile < (discard ? NTILES : 1); tile++) {
                if (discard && !counts[tile]) continue;
                if (discard) counts[tile]--;
                /* analyze_values, through the race members' shape tables when nothing hinges on
                 * you_cai_bi_kao (the two agree on every draw distance and fan). */
                if (h->ycbk)
                    analyze_values(counts, remaining, opened, h->chain[seat], h->piao[seat], h->ycbk, &rows[tile]);
                else
                    race_analyze_values(counts, remaining, opened, h->chain[seat], h->piao[seat], &rows[tile]);
                if (discard) counts[tile]++;
                analysis_row *r = &rows[tile];
                if (r->distance < 0) return 0;
                if (!found || r->distance < best.distance ||
                    (r->distance == best.distance && (r->ukeire > best.ukeire ||
                     (r->ukeire == best.ukeire && r->value > best.value)))) {
                    best = *r;
                    found = 1;
                }
            }
            float *look = planes + 45 * NTILES;
            if (discard) {
                for (int tile = 0; tile < NTILES; tile++) {
                    if (!counts[tile]) continue;
                    analysis_row *r = &rows[tile];
                    look[tile] = (8 - r->distance) / 9.0;
                    look[NTILES + tile] = (r->ukeire < 40 ? r->ukeire : 40) / 40.0;
                    look[2 * NTILES + tile] = (r->copies < 12 ? r->copies : 12) / 12.0;
                    look[3 * NTILES + tile] = log1p((double)r->value) / 8.0;
                }
            }
            for (int i = 0; i < best.n_improving; i++)
                look[4 * NTILES + best.improving[i]] = remaining[best.improving[i]] / 4.0;
            for (int i = 0; i < best.n_wins; i++)
                look[5 * NTILES + best.wins[i]] = log2((double)best.fans[i] + 1.0) / 9.0;
            ref_shanten = best.distance;
            ref_ukeire = best.ukeire;
            ref_copies = best.copies;
            ref_value = best.value;
        }
    } else if (size == 13 - 3 * opened || size == 14 - 3 * opened) {
        int raw = shanten_raw(hc, opened);
        ref_shanten = raw >= 0 ? raw : (hu_shape(hc, opened) ? -1 : 0);
    }
    /* scalars */
    int rel[4];
    for (int i = 0; i < 4; i++) rel[i] = (seat + i) % 4;
    const int *scores = h->scores;
    int own = scores[seat];
    double ranks[4];
    for (int i = 0; i < 4; i++) {
        int above = 0, same = 0;
        for (int j = 0; j < 4; j++) {
            above += scores[j] > scores[i];
            same += scores[j] == scores[i];
        }
        ranks[i] = 1 + above + (same - 1) / 2.0;
    }
    int whites = 0;
    for (int s = 0; s < 4; s++) {
        for (int k = 0; k < h->n_river[s]; k++) whites += h->river[s][k] == WHITE;
        for (int k = 0; k < h->n_melds[s]; k++) {
            if (!g_meld_visible(h, s, seat, k)) continue;
            const g_meld *m = &h->melds[s][k];
            for (int j = 0; j < m->n; j++) whites += m->tiles[j] == WHITE;
        }
    }
    int current = h->hand_no, total = h->rounds_total, fan = g_fan(h);
    int god = h->catch_owner, wall = wall_len(h);
    int drawn = h->phase == PH_DRAW ? h->drawn : -1;
    double f[N_SCALARS];
    int n = 0;
    for (int i = 0; i < 4; i++) f[n++] = g_symlog(scores[rel[i]] / 8.0) / 4.0;
    for (int i = 1; i < 4; i++) f[n++] = g_symlog((own - scores[rel[i]]) / 8.0) / 4.0;
    f[n++] = (ranks[seat] - 1) / 3.0;
    for (int i = 1; i < 4; i++) f[n++] = (ranks[rel[i]] - 1) / 3.0;
    {
        double x = (double)(current - 1) / (total - 1 > 1 ? total - 1 : 1);
        x = x > 0.0 ? x : 0.0;
        f[n++] = x < 1.0 ? x : 1.0;
        double y = (total - current > 0 ? total - current : 0) / 16.0;
        f[n++] = y < 1.0 ? y : 1.0;
        double z = total / 20.0;
        f[n++] = z < 1.0 ? z : 1.0;
    }
    f[n++] = 0.0; /* in_tiebreak */
    f[n++] = 1.0; /* StageKind.QUALIFIER */
    f[n++] = 0.0;
    f[n++] = 0.0;
    for (int i = 0; i < 4; i++) f[n++] = h->dealer == rel[i];
    for (int i = 0; i < 4; i++) f[n++] = god == rel[i];
    f[n++] = god < 0;
    f[n++] = god >= 0;
    f[n++] = god >= 0 && god != seat;
    f[n++] = wall / 136.0;
    f[n++] = (wall - 20 > 0 ? wall - 20 : 0) / 116.0;
    f[n++] = h->phase == PH_DRAW;
    f[n++] = h->phase == PH_PENG;
    f[n++] = h->phase == PH_CHI;
    f[n++] = drawn >= 0;
    f[n++] = (h->chain[seat] < 8 ? h->chain[seat] : 8) / 8.0;
    f[n++] = (h->piao[seat] < 4 ? h->piao[seat] : 4) / 4.0;
    f[n++] = log2((double)(fan > 1 ? fan : 1)) / 9.0;
    f[n++] = fan > 0;
    f[n++] = h->ycbk ? 1.0 : 0.0;
    f[n++] = hc[WHITE] / 4.0;
    f[n++] = whites / 4.0;
    f[n++] = remaining[WHITE] / 4.0;
    for (int i = 0; i < 4; i++) f[n++] = h->n_melds[rel[i]] / 4.0;
    for (int i = 0; i < 4; i++) {
        int k = 0;
        for (int j = 0; j < h->n_melds[rel[i]]; j++) k += h->melds[rel[i]][j].kind == EV_CHI;
        f[n++] = k / 2.0;
    }
    for (int i = 0; i < 4; i++) {
        int k = 0;
        for (int j = 0; j < h->n_melds[rel[i]]; j++) k += h->melds[rel[i]][j].kind == EV_GANG;
        f[n++] = k / 4.0;
    }
    for (int i = 0; i < 4; i++) {
        int k = 0;
        for (int j = 0; j < h->n_melds[rel[i]]; j++) k += h->melds[rel[i]][j].concealed;
        f[n++] = k / 4.0;
    }
    for (int i = 0; i < 4; i++) f[n++] = h->n_hand[rel[i]] / 14.0;
    for (int i = 0; i < 4; i++) f[n++] = h->n_river[rel[i]] / 24.0;
    f[n++] = (8 - ref_shanten) / 9.0;
    f[n++] = ref_shanten <= 0;
    f[n++] = (ref_ukeire < 40 ? ref_ukeire : 40) / 40.0;
    f[n++] = (ref_copies < 12 ? ref_copies : 12) / 12.0;
    f[n++] = g_log1p((double)ref_value) / 8.0;
    f[n++] = n_legal / 16.0;
    f[n++] = kinds[A_HU];
    f[n++] = kinds[A_DISCARD];
    f[n++] = kinds[A_CHI];
    f[n++] = kinds[A_PENG];
    f[n++] = kinds[A_GANG];
    f[n++] = kinds[A_PASS];
    for (int i = 0; i < N_SCALARS; i++) out->scalars[i] = (float)f[i];
    /* this hand's events as the seat sees them */
    int16_t *ev = out->events;
    out->n_events = h->n_ev > 0 ? h->n_ev : 1;
    memset(ev, 0, sizeof(int16_t) * 5 * (size_t)out->n_events);
    for (int i = 0; i < h->n_ev; i++) {
        const g_event *e = &h->ev[i];
        int hidden = g_hidden(e, seat);
        int16_t *row = ev + 5 * i;
        row[0] = e->kind;
        row[1] = (int16_t)(e->seat < 0 ? 4 : ((e->seat - seat) % 4 + 4) % 4);
        row[2] = (int16_t)(hidden || e->tile < 0 ? 34 : e->tile);
        row[3] = (int16_t)(e->kind == EV_CHI && !hidden && e->n ? e->tiles[0] : 34);
        row[4] = 0;
    }
    return 1;
}

/* ---------- the Python face ---------- */

static PyObject *np_empty, *dt_f16, *dt_f32, *dt_i16, *dt_bool;

static PyObject *g_array(PyObject *shape, PyObject *dtype, const void *data, size_t bytes) {
    PyObject *a = PyObject_CallFunctionObjArgs(np_empty, shape, dtype, NULL);
    if (!a) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(a, &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) { Py_DECREF(a); return NULL; }
    if ((size_t)view.len != bytes) {
        PyBuffer_Release(&view);
        Py_DECREF(a);
        PyErr_SetString(PyExc_RuntimeError, "array size mismatch");
        return NULL;
    }
    memcpy(view.buf, data, bytes);
    PyBuffer_Release(&view);
    return a;
}

typedef struct {
    PyObject_HEAD
    g_hand h;
} HandObject;

static PyTypeObject HandType;

static PyObject *hand_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    unsigned long long seed;
    int dealer, hand_no, rounds_total, base_score = 1, ycbk = 0, max_steps = 2000;
    PyObject *scores_obj;
    static char *names[] = {"seed", "dealer", "hand_no", "scores", "rounds_total", "base_score",
                            "you_cai_bi_kao", "max_steps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "KiiOi|iii", names, &seed, &dealer, &hand_no, &scores_obj,
                                     &rounds_total, &base_score, &ycbk, &max_steps))
        return NULL;
    int scores[4];
    PyObject *fast = PySequence_Fast(scores_obj, "scores");
    if (!fast) return NULL;
    if (PySequence_Fast_GET_SIZE(fast) != 4) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "four scores");
        return NULL;
    }
    for (int i = 0; i < 4; i++) scores[i] = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
    Py_DECREF(fast);
    if (PyErr_Occurred()) return NULL;
    if (dealer < 0 || dealer > 3) {
        PyErr_SetString(PyExc_ValueError, "dealer must be 0..3");
        return NULL;
    }
    HandObject *self = (HandObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    g_start(&self->h, (uint64_t)seed, dealer, hand_no, scores, rounds_total, base_score, ycbk, max_steps);
    return (PyObject *)self;
}

static PyObject *hand_fork(HandObject *self, PyObject *unused) {
    HandObject *copy = (HandObject *)HandType.tp_alloc(&HandType, 0);
    if (!copy) return NULL;
    memcpy(&copy->h, &self->h, sizeof(g_hand));
    return (PyObject *)copy;
}

static PyObject *hand_legal(HandObject *self, PyObject *unused) {
    g_action legal[64];
    int n = g_legal(&self->h, legal), index[64];
    for (int i = 0; i < n; i++) index[i] = g_index(&legal[i]);
    return g_int_tuple(index, n);
}

static int g_step_index(g_hand *h, int index) {
    g_action legal[64];
    int n = g_legal(h, legal);
    for (int i = 0; i < n; i++)
        if (g_index(&legal[i]) == index) return g_step(h, &legal[i]);
    PyErr_Format(PyExc_ValueError, "action %d not legal", index);
    return -1;
}

static PyObject *hand_step(HandObject *self, PyObject *arg) {
    long index = PyLong_AsLong(arg);
    if (index == -1 && PyErr_Occurred()) return NULL;
    if (g_step_index(&self->h, (int)index) < 0) return NULL;
    Py_RETURN_NONE;
}

/* run(profiles, fallbacks): play every race member's turn; the seat of the
 * next other decision, or None once the hand is settled. */
static PyObject *hand_run(HandObject *self, PyObject *args) {
    PyObject *profiles_obj, *fallbacks;
    if (!PyArg_ParseTuple(args, "OO", &profiles_obj, &fallbacks)) return NULL;
    int profiles[4];
    PyObject *fast = PySequence_Fast(profiles_obj, "profiles");
    if (!fast) return NULL;
    if (PySequence_Fast_GET_SIZE(fast) != 4 || PySequence_Size(fallbacks) != 4) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "four seats");
        return NULL;
    }
    for (int i = 0; i < 4; i++) profiles[i] = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
    Py_DECREF(fast);
    if (PyErr_Occurred()) return NULL;
    g_hand *h = &self->h;
    while (!h->settled) {
        if (h->steps >= h->max_steps) {
            PyErr_SetString(PyExc_RuntimeError, "hand action budget exceeded");
            return NULL;
        }
        g_action legal[64];
        int n = g_legal(h, legal);
        if (n == 1) {
            if (g_step(h, &legal[0]) < 0) return NULL;
            continue;
        }
        int seat = h->turn;
        if (profiles[seat] < 0) return PyLong_FromLong(seat);
        PyObject *fallback = PySequence_GetItem(fallbacks, seat);
        if (!fallback) return NULL;
        int choice;
        int rc = g_member_act(h, legal, n, profiles[seat], fallback, &choice);
        Py_DECREF(fallback);
        if (rc < 0) return NULL;
        if (g_step(h, &legal[choice]) < 0) return NULL;
    }
    Py_RETURN_NONE;
}

/* act(profile, fallback): the race member's choice for the seat to move, as an action index. */
static PyObject *hand_act(HandObject *self, PyObject *args) {
    int profile;
    PyObject *fallback;
    if (!PyArg_ParseTuple(args, "iO", &profile, &fallback)) return NULL;
    g_action legal[64];
    int n = g_legal(&self->h, legal), choice;
    if (g_member_act(&self->h, legal, n, profile, fallback, &choice) < 0) return NULL;
    return PyLong_FromLong(g_index(&legal[choice]));
}

/* encode(lookahead): the seat to move's planes, scalars, events and mask, or None for the Python encoder. */
static PyObject *hand_encode(HandObject *self, PyObject *arg) {
    int lookahead = PyObject_IsTrue(arg);
    if (lookahead < 0) return NULL;
    g_action legal[64];
    int n = g_legal(&self->h, legal);
    g_encoded *e = PyMem_Malloc(sizeof(g_encoded));
    if (!e) return PyErr_NoMemory();
    PyObject *result = NULL, *planes = NULL, *scalars = NULL, *events = NULL, *mask = NULL;
    int rc = g_encode(&self->h, legal, n, lookahead, e);
    if (rc <= 0) {
        PyMem_Free(e);
        if (rc < 0) return NULL;
        Py_RETURN_NONE;
    }
    uint16_t hp[N_PLANES * NTILES], hs[N_SCALARS];
    for (int i = 0; i < N_PLANES * NTILES; i++) hp[i] = g_half(e->planes[i]);
    for (int i = 0; i < N_SCALARS; i++) hs[i] = g_half(e->scalars[i]);
    PyObject *s1 = Py_BuildValue("(ii)", N_PLANES, NTILES), *s2 = Py_BuildValue("(i)", N_SCALARS),
             *s3 = Py_BuildValue("(ii)", e->n_events, 5), *s4 = Py_BuildValue("(i)", N_ACTIONS);
    if (s1 && s2 && s3 && s4 && (planes = g_array(s1, dt_f16, hp, sizeof(hp))) &&
        (scalars = g_array(s2, dt_f16, hs, sizeof(hs))) &&
        (events = g_array(s3, dt_i16, e->events, sizeof(int16_t) * 5 * (size_t)e->n_events)) &&
        (mask = g_array(s4, dt_bool, e->mask, N_ACTIONS)))
        result = Py_BuildValue("{sOsOsOsO}", "planes", planes, "scalars", scalars, "events", events, "mask", mask);
    Py_XDECREF(s1); Py_XDECREF(s2); Py_XDECREF(s3); Py_XDECREF(s4);
    Py_XDECREF(planes); Py_XDECREF(scalars); Py_XDECREF(events); Py_XDECREF(mask);
    PyMem_Free(e);
    return result;
}

/* A writable C-contiguous column of fixed-size rows: the byte address of row
 * `row` holding `bytes` (a shared array's column; the host owns its memory). */
static char *g_row(Py_buffer *view, PyObject *column, const char *name, Py_ssize_t row, Py_ssize_t stride, Py_ssize_t bytes) {
    if (PyObject_GetBuffer(column, view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (row < 0 || stride <= 0 || view->len % stride || (row + 1) * stride > view->len || bytes > stride) {
        PyBuffer_Release(view);
        view->obj = NULL;
        PyErr_Format(PyExc_ValueError, "%s: row %zd outside a column of %zd-byte rows", name, row, stride);
        return NULL;
    }
    return (char *)view->buf + row * stride;
}

static PyObject *hand_encoded_event_count(HandObject *self, PyObject *unused) {
    return PyLong_FromLong(self->h.n_ev > 0 ? self->h.n_ev : 1);
}

/* encode_into(planes, scalars, events, mask, row_index, event_offset,
 * event_capacity, lookahead): encode's arrays written into whole columns of
 * the shared row store (abi v1), the events at event_offset of the arena; the
 * number of events written, or None where encode returns None. */
static PyObject *hand_encode_into(HandObject *self, PyObject *args) {
    PyObject *planes_obj, *scalars_obj, *events_obj, *mask_obj, *look_obj;
    Py_ssize_t row, event_offset, event_capacity;
    if (!PyArg_ParseTuple(args, "OOOOnnnO", &planes_obj, &scalars_obj, &events_obj, &mask_obj, &row,
                          &event_offset, &event_capacity, &look_obj))
        return NULL;
    int lookahead = PyObject_IsTrue(look_obj);
    if (lookahead < 0) return NULL;
    g_action legal[64];
    int n = g_legal(&self->h, legal);
    g_encoded *e = PyMem_Malloc(sizeof(g_encoded));
    if (!e) return PyErr_NoMemory();
    int rc = g_encode(&self->h, legal, n, lookahead, e);
    if (rc <= 0) {
        PyMem_Free(e);
        if (rc < 0) return NULL;
        Py_RETURN_NONE;
    }
    if (event_capacity < e->n_events) {
        PyMem_Free(e);
        PyErr_Format(PyExc_ValueError, "%d events need more than the %zd reserved", e->n_events, event_capacity);
        return NULL;
    }
    Py_buffer views[4] = {{0}};
    PyObject *result = NULL;
    char *planes = g_row(&views[0], planes_obj, "planes", row, N_PLANES * NTILES * 2, N_PLANES * NTILES * 2);
    char *scalars = planes ? g_row(&views[1], scalars_obj, "scalars", row, N_SCALARS * 2, N_SCALARS * 2) : NULL;
    char *mask = scalars ? g_row(&views[2], mask_obj, "mask", row, N_ACTIONS, N_ACTIONS) : NULL;
    char *events = NULL;
    if (mask && PyObject_GetBuffer(events_obj, &views[3], PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) == 0) {
        if (event_offset < 0 || views[3].len % 10 || (event_offset + event_capacity) * 10 > views[3].len)
            PyErr_Format(PyExc_ValueError, "events %zd..%zd outside the arena", event_offset, event_offset + event_capacity);
        else
            events = (char *)views[3].buf + event_offset * 10;
    }
    if (events) {
        uint16_t *hp = (uint16_t *)planes, *hs = (uint16_t *)scalars;
        for (int i = 0; i < N_PLANES * NTILES; i++) hp[i] = g_half(e->planes[i]);
        for (int i = 0; i < N_SCALARS; i++) hs[i] = g_half(e->scalars[i]);
        memcpy(mask, e->mask, N_ACTIONS);
        memcpy(events, e->events, sizeof(int16_t) * 5 * (size_t)e->n_events);
        result = PyLong_FromLong(e->n_events);
    }
    for (int i = 0; i < 4; i++)
        if (views[i].obj) PyBuffer_Release(&views[i]);
    PyMem_Free(e);
    return result;
}

static void g_oracle(const g_hand *h, int seat, float *planes, float *scalars);

/* oracle_into(oracle_planes, oracle_scalars, row_index, seat): oracle's arrays into whole columns. */
static PyObject *hand_oracle_into(HandObject *self, PyObject *args) {
    PyObject *planes_obj, *scalars_obj;
    Py_ssize_t row;
    int seat;
    if (!PyArg_ParseTuple(args, "OOni", &planes_obj, &scalars_obj, &row, &seat)) return NULL;
    if (seat < 0 || seat > 3) {
        PyErr_SetString(PyExc_ValueError, "seat must be 0..3");
        return NULL;
    }
    Py_buffer views[2] = {{0}};
    char *planes = g_row(&views[0], planes_obj, "oracle_planes", row, 16 * NTILES * 4, 16 * NTILES * 4);
    char *scalars = planes ? g_row(&views[1], scalars_obj, "oracle_scalars", row, 16, 16) : NULL;
    if (scalars) g_oracle(&self->h, seat, (float *)planes, (float *)scalars);
    for (int i = 0; i < 2; i++)
        if (views[i].obj) PyBuffer_Release(&views[i]);
    if (!scalars) return NULL;
    Py_RETURN_NONE;
}

static void g_count_planes4(float *out, const int *c) {
    for (int n = 0; n < 4; n++)
        for (int t = 0; t < NTILES; t++) out[n * NTILES + t] = c[t] >= n + 1 ? 1.0f : 0.0f;
}

/* guided.encoding.oracle: the other hands and the wall. */
static void g_oracle(const g_hand *h, int seat, float *planes, float *scalars) {
    memset(scalars, 0, 4 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        int c[NTILES] = {0};
        if (i < 3) {
            int s = (seat + i + 1) % 4;
            memcpy(c, h->c[s], sizeof(c));
            int opened = h->n_melds[s], size = h->n_hand[s];
            if (size == 13 - 3 * opened || size == 14 - 3 * opened) {
                int raw = shanten_raw(c, opened);
                int sh = raw >= 0 ? raw : (hu_shape(c, opened) ? -1 : 0);
                scalars[i] = (float)((8 - sh) / 9.0);
            }
        } else {
            for (int k = h->lo; k < h->hi; k++)
                if (!h->stable || !h->used[k]) c[h->wall[k]]++;
        }
        g_count_planes4(planes + 4 * i * NTILES, c);
    }
    scalars[3] = (float)(wall_len(h) / 136.0);
}

/* oracle(seat): guided.encoding.oracle as arrays. */
static PyObject *hand_oracle(HandObject *self, PyObject *arg) {
    long seat = PyLong_AsLong(arg);
    if (seat == -1 && PyErr_Occurred()) return NULL;
    if (seat < 0 || seat > 3) {
        PyErr_SetString(PyExc_ValueError, "seat must be 0..3");
        return NULL;
    }
    float planes[16 * NTILES], scalars[4];
    g_oracle(&self->h, (int)seat, planes, scalars);
    PyObject *s1 = Py_BuildValue("(ii)", 16, NTILES), *s2 = Py_BuildValue("(i)", 4);
    PyObject *p = NULL, *s = NULL, *result = NULL;
    if (s1 && s2 && (p = g_array(s1, dt_f32, planes, sizeof(planes))) && (s = g_array(s2, dt_f32, scalars, sizeof(scalars))))
        result = Py_BuildValue("{sOsO}", "oracle_planes", p, "oracle_scalars", s);
    Py_XDECREF(s1); Py_XDECREF(s2); Py_XDECREF(p); Py_XDECREF(s);
    return result;
}

static PyObject *g_tiles(const int8_t *x, int n) {
    PyObject *t = PyTuple_New(n);
    if (!t) return NULL;
    for (int i = 0; i < n; i++) PyTuple_SET_ITEM(t, i, PyLong_FromLong(x[i]));
    return t;
}

static PyObject *g_none_or(int v) {
    if (v < 0) Py_RETURN_NONE;
    return PyLong_FromLong(v);
}

static const char *g_phase_names[4] = {"draw", "response_peng", "response_chi", "settled"};
static const char *g_kind_names[7] = {"", "draw", "discard", "chi", "peng", "gang", "hand_ended"};

/* snapshot(): every field of the hand as plain Python values, for the
 * Python engine's state (fastgame.materialize) and the equivalence tests. */
static PyObject *hand_snapshot(HandObject *self, PyObject *unused) {
    const g_hand *h = &self->h;
    PyObject *d = PyDict_New();
    if (!d) return NULL;
#define SET(key, value) do { PyObject *v_ = (value); if (!v_ || PyDict_SetItemString(d, key, v_) < 0) { Py_XDECREF(v_); Py_DECREF(d); return NULL; } Py_DECREF(v_); } while (0)
    SET("dealer", PyLong_FromLong(h->dealer));
    SET("hand_no", PyLong_FromLong(h->hand_no));
    SET("rounds_total", PyLong_FromLong(h->rounds_total));
    SET("scores", g_int_tuple(h->scores, 4));
    {
        int8_t wall[G_WALL];
        SET("wall", g_tiles(wall, g_wall_tiles(h, wall)));
    }
    PyObject *hands = PyTuple_New(4), *initial = PyTuple_New(4), *melds = PyTuple_New(4), *rivers = PyTuple_New(4);
    if (!hands || !initial || !melds || !rivers) { Py_XDECREF(hands); Py_XDECREF(initial); Py_XDECREF(melds); Py_XDECREF(rivers); Py_DECREF(d); return NULL; }
    for (int s = 0; s < 4; s++) {
        PyTuple_SET_ITEM(hands, s, g_tiles(h->hand[s], h->n_hand[s]));
        PyTuple_SET_ITEM(initial, s, g_tiles(h->initial[s], 13));
        PyTuple_SET_ITEM(rivers, s, g_tiles(h->river[s], h->n_river[s]));
        PyObject *row = PyTuple_New(h->n_melds[s]);
        for (int i = 0; row && i < h->n_melds[s]; i++) {
            const g_meld *m = &h->melds[s][i];
            PyTuple_SET_ITEM(row, i, Py_BuildValue("(sNNN)", g_kind_names[m->kind], g_tiles(m->tiles, m->n),
                                                   g_none_or(m->from), PyBool_FromLong(m->concealed)));
        }
        PyTuple_SET_ITEM(melds, s, row);
    }
    SET("hands", hands);
    SET("initial_hands", initial);
    SET("melds", melds);
    SET("discards", rivers);
    SET("white_acquired", g_int_tuple(h->white_acquired, 4));
    PyObject *events = PyTuple_New(h->n_ev);
    for (int i = 0; events && i < h->n_ev; i++) {
        const g_event *e = &h->ev[i];
        PyTuple_SET_ITEM(events, i, Py_BuildValue("(sNNNNNN)", g_kind_names[e->kind], g_none_or(e->seat), g_none_or(e->tile),
                                                  g_tiles(e->tiles, e->n), e->kind == EV_END ? g_int_tuple(h->delta, 4) : PyTuple_New(0),
                                                  PyBool_FromLong(e->concealed), g_none_or(e->priv)));
    }
    SET("events", events);
    SET("phase", PyUnicode_FromString(g_phase_names[h->phase]));
    SET("turn", PyLong_FromLong(h->turn));
    SET("drawn", g_none_or(h->drawn));
    SET("kong_draw", PyBool_FromLong(h->kong_draw));
    SET("chain", g_int_tuple(h->chain, 4));
    SET("piao", g_int_tuple(h->piao, 4));
    SET("catch_owner", g_none_or(h->catch_owner));
    SET("response_queue", g_int_tuple(h->queue, h->n_queue));
    SET("discarder", g_none_or(h->discarder));
    SET("discarded", g_none_or(h->discarded));
    SET("winner", g_none_or(h->winner));
    SET("fan", h->winner >= 0 ? PyLong_FromLong(h->fan) : (Py_INCREF(Py_None), Py_None));
    SET("delta", g_int_tuple(h->delta, 4));
    SET("settled", PyBool_FromLong(h->settled));
    SET("steps", PyLong_FromLong(h->steps));
    {
        int log[G_LOG];
        for (int i = 0; i < h->n_log; i++) log[i] = h->log[i];
        SET("log", h->n_log == h->steps ? g_int_tuple(log, h->n_log) : (Py_INCREF(Py_None), Py_None));
    }
#undef SET
    return d;
}

static PyObject *hand_get_settled(HandObject *self, void *c) { return PyBool_FromLong(self->h.settled); }
static PyObject *hand_get_winner(HandObject *self, void *c) { return g_none_or(self->h.winner); }
static PyObject *hand_get_delta(HandObject *self, void *c) { return g_int_tuple(self->h.delta, 4); }
static PyObject *hand_get_turn(HandObject *self, void *c) { return PyLong_FromLong(self->h.turn); }
static PyObject *hand_get_dealer(HandObject *self, void *c) { return PyLong_FromLong(self->h.dealer); }
static PyObject *hand_get_steps(HandObject *self, void *c) { return PyLong_FromLong(self->h.steps); }
static PyObject *hand_get_luck(HandObject *self, void *c) { return Py_BuildValue("(dd)", self->h.luck_pay, self->h.luck_deal); }
static PyObject *hand_get_race_luck(HandObject *self, void *c) { return Py_BuildValue("(dd)", self->h.race_pay, self->h.race_deal); }

static PyObject *hand_stabilize(HandObject *self, PyObject *unused) {
    g_stabilize(&self->h);
    Py_RETURN_NONE;
}

/* track_luck(seat, race=False): guided.walls.train_hand's luck_seat (and race). */
static PyObject *hand_track_luck(HandObject *self, PyObject *args) {
    int seat, race = 0;
    if (!PyArg_ParseTuple(args, "i|p", &seat, &race)) return NULL;
    if (seat < 0 || seat > 3) {
        PyErr_SetString(PyExc_ValueError, "seat must be 0..3");
        return NULL;
    }
    self->h.luck_seat = seat;
    self->h.race_on |= race;
    Py_RETURN_NONE;
}

static PyGetSetDef hand_getset[] = {
    {"settled", (getter)hand_get_settled, NULL, NULL, NULL},
    {"winner", (getter)hand_get_winner, NULL, NULL, NULL},
    {"delta", (getter)hand_get_delta, NULL, NULL, NULL},
    {"turn", (getter)hand_get_turn, NULL, NULL, NULL},
    {"dealer", (getter)hand_get_dealer, NULL, NULL, NULL},
    {"steps", (getter)hand_get_steps, NULL, NULL, NULL},
    {"luck", (getter)hand_get_luck, NULL, NULL, NULL},
    {"race_luck", (getter)hand_get_race_luck, NULL, NULL, NULL},
    {NULL},
};

static PyMethodDef hand_methods[] = {
    {"fork", (PyCFunction)hand_fork, METH_NOARGS, "A copy to play on."},
    {"legal", (PyCFunction)hand_legal, METH_NOARGS, "Action indices of the legal actions, in the engine's order."},
    {"step", (PyCFunction)hand_step, METH_O, "Apply the legal action with this index."},
    {"run", (PyCFunction)hand_run, METH_VARARGS, "Play the race members' turns; the next other seat, or None when settled."},
    {"act", (PyCFunction)hand_act, METH_VARARGS, "A race member's choice for the seat to move."},
    {"encode", (PyCFunction)hand_encode, METH_O, "guided.encoding.encode of the seat to move, or None."},
    {"oracle", (PyCFunction)hand_oracle, METH_O, "guided.encoding.oracle for a seat."},
    {"encoded_event_count", (PyCFunction)hand_encoded_event_count, METH_NOARGS, "Event rows encode writes for the seat to move."},
    {"encode_into", (PyCFunction)hand_encode_into, METH_VARARGS, "encode into whole columns of a row store; events written, or None."},
    {"oracle_into", (PyCFunction)hand_oracle_into, METH_VARARGS, "oracle into whole columns of a row store."},
    {"snapshot", (PyCFunction)hand_snapshot, METH_NOARGS, "Every field as plain values."},
    {"stabilize", (PyCFunction)hand_stabilize, METH_NOARGS, "guided.walls.stabilize: each seat draws from its own positions from here on."},
    {"track_luck", (PyCFunction)hand_track_luck, METH_VARARGS,
     "track_luck(seat, race=False): accumulate this seat's draw luck (guided.walls) from here on, and its race luck."},
    {NULL},
};

static PyTypeObject HandType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "native.Hand",
    .tp_basicsize = sizeof(HandObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = hand_new,
    .tp_methods = hand_methods,
    .tp_getset = hand_getset,
    .tp_doc = "One hand of the training game (domain.engine.HandState).",
};

static PyObject *py_game_profile(PyObject *self, PyObject *args) {
    return py_race_profile(self, args);
}

/* rule131(plain, roles): fp-rule-131's profiles here (race_profile's indices):
 * plain for its module's SAVED as defined, roles[dealer][left] for
 * SAVED_BY_ROLE[(dealer, left)]. The member code its seats play with. */
static PyObject *py_rule131(PyObject *self, PyObject *args) {
    int plain, role[2][4];
    if (!PyArg_ParseTuple(args, "i((iiii)(iiii))", &plain, &role[0][0], &role[0][1], &role[0][2], &role[0][3],
                          &role[1][0], &role[1][1], &role[1][2], &role[1][3]))
        return NULL;
    int ok = plain >= 0 && plain < race_profile_count;
    for (int d = 0; d < 2; d++)
        for (int k = 0; k < 4; k++) ok &= role[d][k] >= 0 && role[d][k] < race_profile_count;
    if (!ok) {
        PyErr_SetString(PyExc_ValueError, "unknown race profile");
        return NULL;
    }
    r131.plain = plain;
    memcpy(r131.role, role, sizeof(role));
    r131.ready = 1;
    return PyLong_FromLong(G_R131);
}

/* ---------- encoding v3 (guided/encoding.py): shape paths and a solitary lookahead ---------- */

/* Per discard of a hand that discards (a waiting-length hand for each tile
 * it can let go): the plain and seven-pairs paths apart (_fast's raw shanten
 * is their minimum), whether the hand is a baotou wait and how many draws
 * lead to one, and a lookahead of the learner's own next draws. A waiting
 * hand alone gets only the lookahead (the reference row). Draws come
 * uniformly from the tiles the learner has not seen (opponents' hands
 * included, so the lookahead is optimistic), and other seats are ignored. */
enum { V3_PLANES = 29, V3_LOOK = 20, V3_KS = 5, V3_MEMO = 1024 };
static const int v3_ks[V3_KS - 1] = {1, 2, 3, 5};

typedef struct {
    uint64_t k0, k1;
    long copies, value;
    int baotou, used;
} v3_slot;

typedef struct {
    v3_slot slot[V3_MEMO];
} v3_memo;

/* Seven pairs' raw distance, as shanten_uncached computes it (8 with melds open). */
static int v3_seven(const int *c, int opened) {
    if (opened) return 8;
    int pairs = 0, singles = 0;
    for (int i = 0; i < WHITE; i++) {
        pairs += c[i] / 2;
        singles += c[i] % 2;
    }
    int room = 7 - pairs, capped = singles < room ? singles : room;
    return 14 - (2 * pairs + capped) - c[WHITE] - 1;
}

/* The plain path's raw distance: four melds and a pair, whites filling in. */
static int v3_plain(const int *c, int opened) {
    int natural[NTILES];
    memcpy(natural, c, sizeof(natural));
    natural[WHITE] = 0;
    return shape_value(natural, opened) - c[WHITE];
}

/* Copies, availability-weighted fan and baotou of a waiting hand, memoised per call. */
static v3_slot *v3_wait(v3_memo *m, int *c, int opened, const int *avail, int chain, int piao, int you_cai) {
    uint64_t k0, k1;
    shanten_key(c, opened, &k0, &k1);
    uint64_t h = (k0 * 0x9E3779B97F4A7C15ULL) ^ (k1 * 0xC2B2AE3D27D4EB4FULL);
    int i = (int)((h ^ (h >> 31)) & (V3_MEMO - 1));
    for (int probe = 0; probe < V3_MEMO; probe++, i = (i + 1) & (V3_MEMO - 1)) {
        v3_slot *s = &m->slot[i];
        if (s->used && s->k0 == k0 && s->k1 == k1) return s;
        if (!s->used) {
            analysis_row row;
            analyze_values(c, avail, opened, chain, piao, you_cai, &row);
            s->k0 = k0;
            s->k1 = k1;
            s->copies = row.copies;
            s->value = row.value;
            s->baotou = c[WHITE] > 0 && baotou_cached(c, opened);
            s->used = 1;
            return s;
        }
    }
    return NULL; /* never: one call fills far fewer slots */
}

/* P(no hit in k draws without replacement) with `hits` of `pool` copies hitting (hits may be fractional). */
static double v3_miss(double pool, double hits, int k) {
    double p = 1.0;
    for (int i = 0; i < k; i++) {
        if (pool - i <= 0) break;
        double stay = (pool - hits - i) / (pool - i);
        p *= stay > 0 ? stay : 0.0;
    }
    return p;
}

/* P(at least `need` successes in n trials of probability p). */
static double v3_at_least(int n, double p, int need) {
    if (need <= 0) return 1.0;
    if (need > n || p <= 0) return 0.0;
    double total = 0.0, coef = 1.0;
    for (int i = 0; i <= n; i++) {
        if (i > 0) coef = coef * (n - i + 1) / i;
        if (i >= need) total += coef * pow(p, i) * pow(1.0 - p, n - i);
    }
    return total < 1.0 ? total : 1.0;
}

/* Draws (availability-weighted) after which some discard leaves a baotou wait.
 * A baotou wait is ready and holds a white; a tile added never lengthens the
 * raw distance, so only draws after which the hand is at most ready qualify. */
static long v3_baotou_outs(int *c, int opened, const int *avail, int baotou) {
    long outs = 0;
    if (baotou) { /* letting the drawn tile go keeps the wait */
        for (int x = 0; x < NTILES; x++)
            if (avail[x] > 0 && c[x] < 4) outs += avail[x];
        return outs;
    }
    int8_t after[NTILES];
    memcpy(after, draw_distances(c, opened)->draw, sizeof(after));
    for (int x = 0; x < NTILES; x++) {
        if (avail[x] <= 0 || c[x] >= 4 || after[x] > 0) continue;
        if (!c[WHITE] && x != WHITE) continue;
        c[x]++;
        int found = 0;
        for (int y = 0; y < NTILES && !found; y++) {
            if (!c[y]) continue;
            c[y]--;
            found = c[WHITE] && shanten_raw(c, opened) <= 0 && baotou_cached(c, opened);
            c[y]++;
        }
        c[x]--;
        if (found) outs += avail[x];
    }
    return outs;
}

typedef struct {
    int refine;          /* one away: walk each readying draw and its best discard */
    double wins, fan;    /* otherwise these mean winning copies and fan once ready */
    double got_wins, got_fan, got_mass; /* what a refined walk found */
} v3_ready;

/* The lookahead of a waiting hand: P(ready), P(baotou wait), P(self-drawn win), E[fan | win]
 * within each k of v3_ks and `draws`, into out[20] (k-major). Returns the hand's raw distance. */
static int v3_look(v3_memo *m, int *c, int opened, const int *avail, int chain, int piao, int you_cai,
                   int draws, long *baotou_outs, analysis_row *row, v3_ready *r, float *out) {
    analyze_values(c, avail, opened, chain, piao, you_cai, row);
    int d = row->distance;
    double pool = 0.0;
    for (int x = 0; x < NTILES; x++)
        if (avail[x] > 0 && c[x] < 4) pool += avail[x];
    int baotou = c[WHITE] > 0 && baotou_cached(c, opened);
    long outs = d <= 1 ? v3_baotou_outs(c, opened, avail, baotou) : 0;
    *baotou_outs = outs;
    double ukeire = (double)row->ukeire, wins = (double)row->copies, fan = 0.0, mean_wins = 0.0;
    r->got_mass = 0.0;
    if (d <= 0) {
        fan = row->copies ? (double)row->value / row->copies : 0.0;
    } else if (d == 1 && r->refine) {
        /* Each draw that makes the hand ready, then its best discard by copies (fan on ties). */
        const draw_slot *table = draw_distances(c, opened);
        int8_t ready[NTILES];
        memcpy(ready, table->draw, sizeof(ready)); /* nested lookups may replace the slot */
        double mass = 0.0, copies = 0.0, value = 0.0;
        for (int x = 0; x < NTILES; x++) {
            if (avail[x] <= 0 || c[x] >= 4 || ready[x] > 0) continue;
            c[x]++;
            long best_copies = -1, best_value = 0;
            for (int y = 0; y < NTILES; y++) {
                if (!c[y]) continue;
                c[y]--;
                if (shanten_raw(c, opened) <= 0) {
                    v3_slot *s = v3_wait(m, c, opened, avail, chain, piao, you_cai);
                    if (s && (s->copies > best_copies || (s->copies == best_copies && s->value > best_value))) {
                        best_copies = s->copies;
                        best_value = s->value;
                    }
                }
                c[y]++;
            }
            c[x]--;
            if (best_copies < 0) continue;
            mass += avail[x];
            copies += (double)avail[x] * best_copies;
            value += (double)avail[x] * best_value;
        }
        ukeire = mass;
        mean_wins = mass > 0 ? copies / mass : 0.0;
        fan = copies > 0 ? value / copies : 0.0;
        r->got_wins = mean_wins;
        r->got_fan = fan;
        r->got_mass = mass;
    } else {
        mean_wins = r->wins;
        fan = r->fan;
    }
    for (int i = 0; i < V3_KS; i++) {
        int k = i < V3_KS - 1 ? v3_ks[i] : draws;
        if (k > draws) k = draws;
        double ready_p, baotou_p, win_p;
        if (d <= 0) {
            ready_p = 1.0;
            win_p = 1.0 - v3_miss(pool, wins, k);
        } else if (d == 1) {
            ready_p = 1.0 - v3_miss(pool, ukeire, k);
            win_p = 0.0;
            for (int j = 1; j < k; j++) {
                double first = v3_miss(pool, ukeire, j - 1) * (pool - j + 1 > 0 ? ukeire / (pool - j + 1) : 0.0);
                win_p += first * (1.0 - v3_miss(pool - j, mean_wins, k - j));
            }
        } else {
            /* Two or more away: each improving draw at the first one's rate;
             * once ready, the mean winning copies of r. */
            double rate = pool > 0 ? ukeire / pool : 0.0, before = 0.0;
            ready_p = v3_at_least(k, rate, d);
            win_p = 0.0;
            for (int j = d; j < k; j++) {
                double by = v3_at_least(j, rate, d);
                win_p += (by - before) * (1.0 - v3_miss(pool - j, mean_wins, k - j));
                before = by;
            }
        }
        baotou_p = baotou ? 1.0 : (d <= 1 ? 1.0 - v3_miss(pool, (double)outs, k) : 0.0);
        out[4 * i] = (float)ready_p;
        out[4 * i + 1] = (float)baotou_p;
        out[4 * i + 2] = (float)win_p;
        out[4 * i + 3] = (float)(win_p > 0 ? log2(1.0 + fan) / 9.0 : 0.0);
    }
    return d;
}

/* v3_shape(hand counts, open melds, available, chain, piao, you_cai, own draws left,
 *          planes float32[29,34], reference float32[20]) -> best discard or -1.
 * Planes per discard t (zero for a waiting hand): 0-2 the plain path's
 * (8 - shanten) / 9, accepted copies / 40, winning copies / 12; 3-5 the same
 * for seven pairs; 6 a baotou wait; 7 baotou outs / 40; 8-27 the lookahead.
 * The reference is the lookahead of the best discard (v1's order: distance,
 * accepted copies, availability-weighted fan) or of the waiting hand itself.
 * A discard one away from ready is walked draw by draw (V3_WALKS best ones in
 * v1's order); the others, and those two or more away, take the walked ones'
 * mean winning copies and fan once ready (or the ready discards' own, or 4
 * copies at fan 1). */
enum { V3_WALKS = 3 };

static int v3_better(const analysis_row *a, const analysis_row *b) {
    return a->distance < b->distance ||
           (a->distance == b->distance && (a->ukeire > b->ukeire ||
            (a->ukeire == b->ukeire && a->value > b->value)));
}

static PyObject *py_v3_shape(PyObject *self, PyObject *args) {
    PyObject *hand_obj, *avail_obj, *planes_obj, *ref_obj;
    int opened, chain, piao, you_cai, draws, c[NTILES], avail[NTILES];
    if (!PyArg_ParseTuple(args, "OiOiiiiOO", &hand_obj, &opened, &avail_obj, &chain, &piao, &you_cai, &draws,
                          &planes_obj, &ref_obj)) return NULL;
    if (opened < 0 || opened > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    if (parse_counts(hand_obj, c) < 0 || parse_vector(avail_obj, avail, 1000000, "available") < 0) return NULL;
    if (draws < 0) draws = 0;
    Py_buffer pv, rv;
    if (PyObject_GetBuffer(planes_obj, &pv, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(ref_obj, &rv, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&pv);
        return NULL;
    }
    if (pv.len != (Py_ssize_t)(V3_PLANES * NTILES * sizeof(float)) || strcmp(pv.format, "f") != 0 ||
        rv.len != (Py_ssize_t)(V3_LOOK * sizeof(float)) || strcmp(rv.format, "f") != 0) {
        PyBuffer_Release(&pv);
        PyBuffer_Release(&rv);
        PyErr_SetString(PyExc_ValueError, "buffers must be contiguous float32[29,34] and float32[20]");
        return NULL;
    }
    float *planes = (float *)pv.buf, *ref = (float *)rv.buf;
    memset(planes, 0, pv.len);
    memset(ref, 0, rv.len);
    int size = 0;
    for (int t = 0; t < NTILES; t++) size += c[t];
    v3_memo *m = PyMem_Calloc(1, sizeof(v3_memo));
    if (!m) {
        PyBuffer_Release(&pv);
        PyBuffer_Release(&rv);
        return PyErr_NoMemory();
    }
    int best = -1;
    if (size == 13 - 3 * opened) {
        analysis_row row;
        long outs;
        v3_ready r = {1, 4.0, 1.0, 0.0, 0.0, 0.0};
        v3_look(m, c, opened, avail, chain, piao, you_cai, draws, &outs, &row, &r, ref);
    } else if (size == 14 - 3 * opened) {
        analysis_row rows[NTILES];
        int order[NTILES], n = 0;
        double ready_wins = 0.0, ready_value = 0.0, ready_n = 0.0;
        for (int t = 0; t < NTILES; t++) {
            if (!c[t]) continue;
            c[t]--;
            /* The two paths, each with its own accepted and winning draws. */
            const draw_slot *table = draw_distances(c, opened);
            int least = table->distance;
            int8_t after[NTILES];
            memcpy(after, table->draw, sizeof(after));
            int seven = v3_seven(c, opened);
            int plain = opened || seven > least ? least : v3_plain(c, opened);
            long plain_ukeire = 0, plain_wins = 0, seven_ukeire = 0, seven_wins = 0;
            for (int x = 0; x < NTILES; x++) {
                if (avail[x] <= 0 || c[x] >= 4) continue;
                c[x]++;
                int seven_x = v3_seven(c, opened), least_x = after[x];
                int plain_x = opened || seven_x > least_x ? least_x : v3_plain(c, opened);
                if (plain_x < plain) plain_ukeire += avail[x];
                if (!opened && seven_x < seven) seven_ukeire += avail[x];
                if (least_x < 0 && fan_value(c, x, opened, chain, piao, you_cai, 0) > 0) {
                    if (plain_x < 0) plain_wins += avail[x];
                    if (!opened && seven_x < 0) seven_wins += avail[x];
                }
                c[x]--;
            }
            float *col = planes + t;
            col[0] = (8 - plain) / 9.0f;
            col[NTILES] = (plain_ukeire < 40 ? plain_ukeire : 40) / 40.0f;
            col[2 * NTILES] = (plain_wins < 12 ? plain_wins : 12) / 12.0f;
            col[3 * NTILES] = (8 - seven) / 9.0f;
            col[4 * NTILES] = (seven_ukeire < 40 ? seven_ukeire : 40) / 40.0f;
            col[5 * NTILES] = (seven_wins < 12 ? seven_wins : 12) / 12.0f;
            col[6 * NTILES] = c[WHITE] > 0 && baotou_cached(c, opened) ? 1.0f : 0.0f;
            analyze_values(c, avail, opened, chain, piao, you_cai, &rows[t]);
            if (rows[t].distance <= 0 && rows[t].copies > 0) {
                ready_wins += rows[t].copies;
                ready_value += rows[t].value;
                ready_n += 1.0;
            }
            /* Insertion into v1's order. */
            int at = n++;
            while (at > 0 && v3_better(&rows[t], &rows[order[at - 1]])) {
                order[at] = order[at - 1];
                at--;
            }
            order[at] = t;
            c[t]++;
        }
        best = n ? order[0] : -1;
        double walked_wins = 0.0, walked_value = 0.0, walked_mass = 0.0;
        double base_wins = ready_n > 0 ? ready_wins / ready_n : 4.0;
        double base_fan = ready_wins > 0 ? ready_value / ready_wins : 1.0;
        for (int pass = 0; pass < 2; pass++) {
            v3_ready r = {pass == 0, base_wins, base_fan, 0.0, 0.0, 0.0};
            if (pass == 1 && walked_mass > 0) {
                r.wins = walked_wins / walked_mass;
                r.fan = walked_wins > 0 ? walked_value / walked_wins : 1.0;
            }
            for (int i = pass == 0 ? 0 : V3_WALKS; i < (pass == 0 ? (n < V3_WALKS ? n : V3_WALKS) : n); i++) {
                int t = order[i];
                float look[V3_LOOK];
                analysis_row row;
                long outs;
                c[t]--;
                v3_look(m, c, opened, avail, chain, piao, you_cai, draws, &outs, &row, &r, look);
                c[t]++;
                if (pass == 0 && r.got_mass > 0) {
                    walked_wins += r.got_wins * r.got_mass;
                    walked_value += r.got_fan * r.got_wins * r.got_mass;
                    walked_mass += r.got_mass;
                }
                float *col = planes + t;
                col[7 * NTILES] = (outs < 40 ? outs : 40) / 40.0f;
                for (int j = 0; j < V3_LOOK; j++) col[(8 + j) * NTILES] = look[j];
                if (t == best) memcpy(ref, look, sizeof(look));
            }
        }
    }
    PyMem_Free(m);
    PyBuffer_Release(&pv);
    PyBuffer_Release(&rv);
    return PyLong_FromLong(best);
}

static int v3_age_bucket(int turns) {
    return turns < 5 ? turns : turns < 7 ? 5 : turns < 10 ? 6 : 7;
}

/* v3_events(events int16[L,5] (this hand's, relative seats), tails int16[L], rivers float32[4,34])
 *   -> (discards, last discarder or -1, turns since each seat's last call (-1 without one),
 *       whether the learner's last event is its chi or peng).
 * Tails: forced * 8 + the age bucket of the turns (discards after / 4) since the
 * event, forced 1 for a discard in a catch round by a seat other than its owner
 * (a white discard opens the round, its owner's other discard closes it; the
 * others can only let the drawn tile go), 2 for the discard right after the same
 * seat's chi or peng (from the hand: no draw came in between). Rivers: per seat,
 * exp(-0.2 * turns since) of each discard nobody claimed, at its tile. */
static PyObject *py_v3_events(PyObject *self, PyObject *args) {
    PyObject *events_obj, *tails_obj, *rivers_obj;
    if (!PyArg_ParseTuple(args, "OOO", &events_obj, &tails_obj, &rivers_obj)) return NULL;
    Py_buffer ev, tv, rv;
    if (PyObject_GetBuffer(events_obj, &ev, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(tails_obj, &tv, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&ev);
        return NULL;
    }
    if (PyObject_GetBuffer(rivers_obj, &rv, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&ev);
        PyBuffer_Release(&tv);
        return NULL;
    }
    Py_ssize_t n = ev.len / (Py_ssize_t)(5 * sizeof(int16_t));
    if (ev.itemsize != 2 || ev.len != n * 5 * 2 || tv.itemsize != 2 || tv.len != n * 2 ||
        rv.len != (Py_ssize_t)(4 * NTILES * sizeof(float)) || strcmp(rv.format, "f") != 0) {
        PyBuffer_Release(&ev);
        PyBuffer_Release(&tv);
        PyBuffer_Release(&rv);
        PyErr_SetString(PyExc_ValueError, "expected int16[L,5] events, int16[L] tails and float32[4,34] rivers");
        return NULL;
    }
    const int16_t *e = (const int16_t *)ev.buf;
    int16_t *tails = (int16_t *)tv.buf;
    float *rivers = (float *)rv.buf;
    memset(tails, 0, tv.len);
    memset(rivers, 0, rv.len);
    int total = 0;
    for (Py_ssize_t i = 0; i < n; i++)
        if (e[5 * i] == EV_DISCARD) total++;
    int owner = -1, seen = 0, last_discarder = -1, previous[5] = {0, 0, 0, 0, 0};
    long last_call[4] = {-1, -1, -1, -1}, last_kind = 0, last_seat = -1;
    for (Py_ssize_t i = 0; i < n; i++) {
        int kind = e[5 * i], seat = e[5 * i + 1], tile = e[5 * i + 2];
        if (!kind) continue;
        int after = total - seen - (kind == EV_DISCARD);
        int forced = 0;
        if (kind == EV_DISCARD) {
            if (owner >= 0 && seat != owner) forced = 1;
            else if (seat >= 0 && seat < 5 && (previous[seat] == EV_CHI || previous[seat] == EV_PENG)) forced = 2;
            if (tile == WHITE) owner = seat;
            else if (seat == owner) owner = -1;
            int next = i + 1 < n ? e[5 * (i + 1)] : 0;
            int claimed = next == EV_CHI || next == EV_PENG || next == EV_GANG;
            if (!claimed && tile >= 0 && tile < NTILES && seat >= 0 && seat < 4)
                rivers[seat * NTILES + tile] += (float)exp(-0.2 * (after / 4.0));
            last_discarder = seat;
            seen++;
        } else if ((kind == EV_CHI || kind == EV_PENG || kind == EV_GANG) && seat >= 0 && seat < 4) {
            last_call[seat] = total - seen;
        }
        if (seat >= 0 && seat < 5) previous[seat] = kind;
        last_kind = kind;
        last_seat = seat;
        tails[i] = (int16_t)(forced * 8 + v3_age_bucket(after / 4));
    }
    PyBuffer_Release(&ev);
    PyBuffer_Release(&tv);
    PyBuffer_Release(&rv);
    return Py_BuildValue("(ii(llll)i)", total, last_discarder, last_call[0], last_call[1], last_call[2],
                         last_call[3], last_seat == 0 && (last_kind == EV_CHI || last_kind == EV_PENG));
}

static PyMethodDef native_methods[] = {
    {"race_profile", py_game_profile, METH_VARARGS, "Register a race member's profile here (the runtime's race_profile arguments)."},
    {"rule131", py_rule131, METH_VARARGS, "Register fp-rule-131's profiles (plain, roles[dealer][left]); its member code."},
    {"v3_shape", py_v3_shape, METH_VARARGS, "Encoding v3's per-discard shape paths, baotou and lookahead planes and the reference lookahead."},
    {"v3_events", py_v3_events, METH_VARARGS, "Encoding v3's event tails, decayed rivers and call timing."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef native_module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, native_methods};

PyMODINIT_FUNC PyInit_native(void) {
    if (fast_keys() < 0 || g_stubs() < 0 || meld_tables() < 0) return NULL;
    PyObject *np = PyImport_ImportModule("numpy");
    if (!np) return NULL;
    np_empty = PyObject_GetAttrString(np, "empty");
    dt_f16 = PyObject_GetAttrString(np, "float16");
    dt_f32 = PyObject_GetAttrString(np, "float32");
    dt_i16 = PyObject_GetAttrString(np, "int16");
    dt_bool = PyObject_GetAttrString(np, "bool_");
    Py_DECREF(np);
    if (!np_empty || !dt_f16 || !dt_f32 || !dt_i16 || !dt_bool) return NULL;
    if (PyType_Ready(&HandType) < 0) return NULL;
    PyObject *m = PyModule_Create(&native_module);
    if (!m) return NULL;
    Py_INCREF(&HandType);
    if (PyModule_AddObject(m, "Hand", (PyObject *)&HandType) < 0) {
        Py_DECREF(&HandType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
