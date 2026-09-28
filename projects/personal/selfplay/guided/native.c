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
 * the ones the Python engine calls through that runtime.
 */
#include "_fast.c"

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
} g_hand;

static int wall_len(const g_hand *h) { return h->hi - h->lo; }

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

static void g_draw(g_hand *h, int seat, int kong, int initial) {
    if (wall_len(h) <= 20) {
        g_finish(h, -1);
        return;
    }
    h->turn = seat;
    h->drawn = kong ? h->wall[h->lo++] : h->wall[--h->hi];
    g_take(h, seat, h->drawn);
    if (h->drawn == WHITE) h->white_acquired[seat]++;
    h->kong_draw = kong;
    h->phase = PH_DRAW;
    if (!initial) g_event_add(h, EV_DRAW, seat, h->drawn, NULL, 0, seat);
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

/* py_race_act for the seat to move: the index of its choice among the legal actions. */
static int g_member_act(const g_hand *h, const g_action *legal, int n_legal, int profile,
                        PyObject *fallback, int *choice) {
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
            for (int k = h->lo; k < h->hi; k++) c[h->wall[k]]++;
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
    SET("wall", g_tiles(h->wall + h->lo, wall_len(h)));
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

static PyGetSetDef hand_getset[] = {
    {"settled", (getter)hand_get_settled, NULL, NULL, NULL},
    {"winner", (getter)hand_get_winner, NULL, NULL, NULL},
    {"delta", (getter)hand_get_delta, NULL, NULL, NULL},
    {"turn", (getter)hand_get_turn, NULL, NULL, NULL},
    {"dealer", (getter)hand_get_dealer, NULL, NULL, NULL},
    {"steps", (getter)hand_get_steps, NULL, NULL, NULL},
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

static PyMethodDef native_methods[] = {
    {"race_profile", py_game_profile, METH_VARARGS, "Register a race member's profile here (the runtime's race_profile arguments)."},
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
