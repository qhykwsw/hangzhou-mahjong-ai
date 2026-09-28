/* Fast structural primitives for the mahjong SDK.
 *
 * This is a faithful port of the pure-Python reference implementations:
 *   - sdk/shape.py::_normal  (concealed shape distance / shanten)
 *   - sdk/shape.py::shanten  (white-tile completion, seven pairs, open melds)
 * The Python side keeps the reference implementation as the fallback. Negative
 * distances are guarded by the native winning predicate, also checked against
 * the reference by tests/test_shape_fast.py.
 *
 * The per-suit decomposition keeps, for every (melds, pair) pair, the largest
 * number of partial sets (taatsu) reachable, saturated at four: the terminal
 * formula 8 - 2*melds - min(partials, 4 - melds) - pair is monotone in that
 * count, so saturated states never lose an optimum.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define NTILES 34
#define WHITE 33
#define UNREACHABLE 255

typedef struct {
    uint8_t t[5][2]; /* [melds][pair] -> max partials, UNREACHABLE when impossible */
} suit_res;

/* Memos below are plain statics: every entry point keeps the GIL for the whole
 * call, so threads sharing a process are already serialized. The thread-local
 * alternative costs about 45% of the hot path in TLS addressing. */
typedef struct {
    uint32_t gen[10][5][5][5];
    suit_res val[10][5][5][5];
    uint32_t stamp;
} suit_ctx;

static suit_ctx ctx;

/* Merge src shifted by (dm, dt, dp) into dst. Values saturate at four. */
static void merge_shift(suit_res *dst, const suit_res *src, int dm, int dt, int dp) {
    for (int m = 0; m < 5; m++) {
        int nm = m + dm;
        if (nm > 4) continue;
        for (int p = 0; p < 2; p++) {
            int np = p + dp;
            if (np > 1) continue;
            uint8_t v = src->t[m][p];
            if (v == UNREACHABLE) continue;
            int nv = (int)v + dt;
            if (nv > 4) nv = 4;
            if (dst->t[nm][np] == UNREACHABLE || dst->t[nm][np] < nv) {
                dst->t[nm][np] = (uint8_t)nv;
            }
        }
    }
}

static suit_res suit_rec(const int *c, int n, int i, int a, int b, int d, int allow_seq) {
    suit_res out;
    memset(&out, UNREACHABLE, sizeof(out));
    if (a == 0) {
        if (i >= n - 1) {
            out.t[0][0] = 0;
            return out;
        }
        int next_d = i + 3 < n ? c[i + 3] : 0;
        return suit_rec(c, n, i + 1, b, d, next_d, allow_seq);
    }
    uint32_t *gen = &ctx.gen[i][a][b][d];
    suit_res *slot = &ctx.val[i][a][b][d];
    if (*gen == ctx.stamp) return *slot;

    suit_res cur;
    suit_res child;
    memset(&cur, UNREACHABLE, sizeof(cur));

    /* Leave one copy unused: the tile is not part of any block. */
    child = suit_rec(c, n, i, a - 1, b, d, allow_seq);
    merge_shift(&cur, &child, 0, 0, 0);
    if (a >= 3) {
        child = suit_rec(c, n, i, a - 3, b, d, allow_seq);
        merge_shift(&cur, &child, 1, 0, 0);
    }
    if (allow_seq && b >= 1 && d >= 1) {
        child = suit_rec(c, n, i, a - 1, b - 1, d - 1, allow_seq);
        merge_shift(&cur, &child, 1, 0, 0);
    }
    if (a >= 2) {
        child = suit_rec(c, n, i, a - 2, b, d, allow_seq);
        merge_shift(&cur, &child, 0, 0, 1);
        merge_shift(&cur, &child, 0, 1, 0);
    }
    if (allow_seq && b >= 1) {
        child = suit_rec(c, n, i, a - 1, b - 1, d, allow_seq);
        merge_shift(&cur, &child, 0, 1, 0);
    }
    if (allow_seq && d >= 1) {
        child = suit_rec(c, n, i, a - 1, b, d - 1, allow_seq);
        merge_shift(&cur, &child, 0, 1, 0);
    }

    *slot = cur;
    *gen = ctx.stamp;
    return cur;
}

/* Memo of one suit's table by its counts: a hand's suits recur across the
 * hands a strategy analyses far more than whole hands do, so a distance that
 * misses the hand memo usually costs four probes here instead of four
 * searches. Key: 3 bits per count, honours flagged above them. */
#define SUIT_CACHE_BITS 16
#define SUIT_CACHE_SIZE (1 << SUIT_CACHE_BITS)

typedef struct {
    uint32_t key; /* 0 is never a used key: the honours flag or a count is set */
    suit_res value;
} suit_slot;

static suit_slot suit_cache[SUIT_CACHE_SIZE];

static suit_res suit_table(const int *row, int n, int allow_seq) {
    uint32_t key = allow_seq ? 0 : 1u << 27;
    for (int i = 0; i < n; i++) key |= (uint32_t)row[i] << (3 * i);
    if (key == 0) key = 1u << 28; /* the empty numbered suit */
    suit_slot *slot = &suit_cache[(key * 2654435761u) >> (32 - SUIT_CACHE_BITS)];
    if (slot->key == key) return slot->value;
    ctx.stamp++;
    if (ctx.stamp == 0) {
        memset(ctx.gen, 0, sizeof(ctx.gen));
        ctx.stamp = 1;
    }
    suit_res r = suit_rec(row, n, 0, row[0], n > 1 ? row[1] : 0, n > 2 ? row[2] : 0, allow_seq);
    slot->key = key;
    slot->value = r;
    return r;
}

/* Best 8 - 2*melds - min(partials, 4 - melds) - pair over the four groups. */
static int shape_value(const int *c, int open_melds) {
    int g[5][2];
    int next[5][2];
    for (int m = 0; m < 5; m++)
        for (int p = 0; p < 2; p++) g[m][p] = -1;
    g[open_melds][0] = 0;
    for (int s = 0; s < 4; s++) {
        int n = s < 3 ? 9 : 7;
        const int *row = c + (s < 3 ? s * 9 : 27);
        suit_res r = suit_table(row, n, s < 3);
        for (int m = 0; m < 5; m++)
            for (int p = 0; p < 2; p++) next[m][p] = -1;
        for (int m = 0; m < 5; m++) {
            for (int p = 0; p < 2; p++) {
                if (g[m][p] < 0) continue;
                for (int dm = 0; dm < 5; dm++) {
                    int nm = m + dm;
                    if (nm > 4) continue;
                    for (int dp = 0; dp < 2; dp++) {
                        int np = p + dp;
                        if (np > 1) continue;
                        uint8_t v = r.t[dm][dp];
                        if (v == UNREACHABLE) continue;
                        int nv = g[m][p] + (int)v;
                        if (nv > 4) nv = 4;
                        if (next[nm][np] < nv) next[nm][np] = nv;
                    }
                }
            }
        }
        for (int m = 0; m < 5; m++)
            for (int p = 0; p < 2; p++) g[m][p] = next[m][p];
    }
    int best = 0x7FFFFF;
    for (int m = 0; m < 5; m++) {
        for (int p = 0; p < 2; p++) {
            if (g[m][p] < 0) continue;
            int capped = g[m][p] < 4 - m ? g[m][p] : 4 - m;
            if (capped < 0) capped = 0;
            int value = 8 - 2 * m - capped - p;
            if (value < best) best = value;
        }
    }
    return best;
}

/* Raw shape distance exactly like sdk/shape.py::shanten before the hu guard. */
static int shanten_uncached(const int *counts, int open_melds);

#define SHANTEN_CACHE_BITS 18
#define SHANTEN_CACHE_SIZE (1 << SHANTEN_CACHE_BITS)

/* Direct-mapped memo of raw distances, 16 bytes per slot (4 MB: the table
 * stays in cache, where the 37-byte slots it replaced spilled to memory and
 * an analyze_counts call, which probes it for every draw, spent most of its
 * time waiting on those misses). Key: 3 bits per tile count, tiles 0..20 in
 * k0, 21..33 in k1 with open_melds above them; k1 also holds a used bit and
 * the value, so an empty slot never matches. */
typedef struct {
    uint64_t k0, k1;
} shanten_slot;

static shanten_slot shanten_cache[SHANTEN_CACHE_SIZE];

#define KEY1_BITS ((1ULL << 42) - 1)
#define KEY_USED (1ULL << 42)
#define VALUE_SHIFT 43

static void shanten_key(const int *counts, int open_melds, uint64_t *k0, uint64_t *k1) {
    uint64_t a = 0, b = 0;
    for (int i = 0; i < 21; i++) a |= (uint64_t)counts[i] << (3 * i);
    for (int i = 21; i < NTILES; i++) b |= (uint64_t)counts[i] << (3 * (i - 21));
    *k0 = a;
    *k1 = b | ((uint64_t)open_melds << 39);
}

/* The key of counts with one more copy of tile t (its count below four). */
static void shanten_key_draw(uint64_t k0, uint64_t k1, int t, uint64_t *d0, uint64_t *d1) {
    *d0 = t < 21 ? k0 + (1ULL << (3 * t)) : k0;
    *d1 = t < 21 ? k1 : k1 + (1ULL << (3 * (t - 21)));
}

static int shanten_keyed(uint64_t k0, uint64_t k1, const int *counts, int open_melds) {
    uint64_t h = (k0 * 0x9E3779B97F4A7C15ULL) ^ (k1 * 0xC2B2AE3D27D4EB4FULL);
    h ^= h >> 29;
    shanten_slot *slot = &shanten_cache[h >> (64 - SHANTEN_CACHE_BITS)];
    if (slot->k0 == k0 && (slot->k1 & (KEY1_BITS | KEY_USED)) == (k1 | KEY_USED)) {
        return (int)((slot->k1 >> VALUE_SHIFT) & 63) - 16;
    }
    int value = shanten_uncached(counts, open_melds);
    slot->k0 = k0;
    slot->k1 = k1 | KEY_USED | ((uint64_t)(value + 16) << VALUE_SHIFT);
    return value;
}

static int shanten_raw(const int *counts, int open_melds) {
    uint64_t k0, k1;
    shanten_key(counts, open_melds, &k0, &k1);
    return shanten_keyed(k0, k1, counts, open_melds);
}

/* The distances a draw analysis needs, memoised per hand: the hand's own and
 * that of every tile drawn (NO_DRAW where four copies are held already). A
 * strategy analyses the same hand under several availability vectors and
 * the same draws from neighbouring hands, so one probe here replaces up to
 * 35 of the memo above; the values are what that memo would give. */
#define DRAW_CACHE_BITS 16
#define DRAW_CACHE_SIZE (1 << DRAW_CACHE_BITS)
#define NO_DRAW 127

typedef struct {
    uint64_t k0, k1; /* k1 as in shanten_slot, without the value */
    int8_t distance;
    int8_t draw[NTILES];
} draw_slot;

static draw_slot draw_cache[DRAW_CACHE_SIZE];

static const draw_slot *draw_distances(const int *counts, int open_melds) {
    uint64_t k0, k1;
    shanten_key(counts, open_melds, &k0, &k1);
    uint64_t h = (k0 * 0xD6E8FEB86659FD93ULL) ^ (k1 * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 31;
    draw_slot *slot = &draw_cache[h >> (64 - DRAW_CACHE_BITS)];
    if (slot->k0 == k0 && slot->k1 == (k1 | KEY_USED)) return slot;
    int drawn[NTILES];
    slot->k0 = k0;
    slot->k1 = k1 | KEY_USED;
    slot->distance = (int8_t)shanten_keyed(k0, k1, counts, open_melds);
    for (int t = 0; t < NTILES; t++) {
        if (counts[t] >= 4) {
            slot->draw[t] = NO_DRAW;
            continue;
        }
        uint64_t d0, d1;
        memcpy(drawn, counts, sizeof(drawn));
        drawn[t]++;
        shanten_key_draw(k0, k1, t, &d0, &d1);
        slot->draw[t] = (int8_t)shanten_keyed(d0, d1, drawn, open_melds);
    }
    return slot;
}

static int shanten_uncached(const int *counts, int open_melds) {
    int natural[NTILES];
    memcpy(natural, counts, sizeof(natural));
    int jokers = natural[WHITE];
    natural[WHITE] = 0;
    int normal = shape_value(natural, open_melds) - jokers;
    if (open_melds == 0) {
        int pairs = 0, singles = 0;
        for (int i = 0; i < NTILES; i++) {
            pairs += natural[i] / 2;
            singles += natural[i] % 2;
        }
        int pair_room = 7 - pairs;
        int capped = singles < pair_room ? singles : pair_room;
        int seven = 14 - (2 * pairs + capped) - jokers - 1;
        if (seven < normal) normal = seven;
    }
    return normal;
}

/* ---------- winning shape with white-tile completion ---------- */

/* Direct-mapped memo for the meld search: key is the 34 counts plus jokers. */
#define MELD_CACHE_BITS 17
#define MELD_CACHE_SIZE (1 << MELD_CACHE_BITS)

typedef struct { uint64_t k0, k1; } meld_slot;

static meld_slot meld_cache[MELD_CACHE_SIZE];

/* Port of domain/rules.py::_meldable: can every tile join a meld, jokers filling. */
static int meldable(const int *c, int jokers, uint64_t k0, uint64_t k1) {
    uint64_t h = (k0 * 0x9E3779B97F4A7C15ULL) ^ (k1 * 0xC2B2AE3D27D4EB4FULL);
    h ^= h >> 29;
    meld_slot *slot = &meld_cache[h >> (64 - MELD_CACHE_BITS)];
    if (slot->k0 == k0 && (slot->k1 & (KEY1_BITS | KEY_USED)) == (k1 | KEY_USED))
        return (int)(slot->k1 >> VALUE_SHIFT);
    int result = 0;
    int first = -1;
    for (int i = 0; i < 34; i++) {
        if (c[i]) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        result = jokers % 3 == 0;
    } else {
        int most = c[first] < 3 ? c[first] : 3;
        for (int take = most; take >= 1; take--) {
            int need = 3 - take;
            if (need > jokers) continue;
            int x[34];
            memcpy(x, c, sizeof(x));
            x[first] -= take;
            uint64_t n0 = first < 21 ? k0 - ((uint64_t)take << (3 * first)) : k0;
            uint64_t n1 = first < 21 ? k1 : k1 - ((uint64_t)take << (3 * (first - 21)));
            n1 -= (uint64_t)need << 39;
            if (meldable(x, jokers - need, n0, n1)) {
                result = 1;
                break;
            }
        }
        if (!result && first < 27) {
            int suit = first / 9 * 9;
            int low = suit > first - 2 ? suit : first - 2;
            int high = (suit + 6) < first ? (suit + 6) : first;
            for (int start = low; start <= high && !result; start++) {
                int x[34];
                memcpy(x, c, sizeof(x));
                int missing = 0;
                uint64_t n0 = k0, n1 = k1;
                for (int t = start; t < start + 3; t++) {
                    if (x[t]) {
                        x[t] -= 1;
                        if (t < 21) n0 -= 1ULL << (3 * t);
                        else n1 -= 1ULL << (3 * (t - 21));
                    } else {
                        missing += 1;
                    }
                }
                if (missing > jokers) continue;
                n1 -= (uint64_t)missing << 39;
                if (meldable(x, jokers - missing, n0, n1)) {
                    result = 1;
                }
            }
        }
    }
    slot->k0 = k0;
    slot->k1 = k1 | KEY_USED | ((uint64_t)result << VALUE_SHIFT);
    return result;
}

/* Port of domain/rules.py::_shape: 4 - open_melds melds plus one pair. */
static int hu_shape(const int *counts, int open_melds) {
    if (open_melds < 0 || open_melds > 4) return 0;
    int total = 0;
    for (int i = 0; i < 34; i++) total += counts[i];
    if (total != 14 - 3 * open_melds) return 0;
    int jokers = counts[WHITE];
    int natural[34];
    memcpy(natural, counts, sizeof(natural));
    natural[WHITE] = 0;
    uint64_t k0, k1;
    shanten_key(natural, jokers, &k0, &k1);
    if (open_melds == 0) {
        int odd = 0;
        for (int i = 0; i < 33; i++) odd += natural[i] % 2;
        if (odd <= jokers) return 1;
    }
    if (jokers >= 2) {
        if (meldable(natural, jokers - 2, k0, k1 - (2ULL << 39))) return 1;
    }
    for (int i = 0; i < 33; i++) {
        int most = natural[i] < 2 ? natural[i] : 2;
        for (int take = most; take >= 1; take--) {
            int need = 2 - take;
            if (need > jokers) continue;
            int x[34];
            memcpy(x, natural, sizeof(x));
            x[i] -= take;
            uint64_t n0 = i < 21 ? k0 - ((uint64_t)take << (3 * i)) : k0;
            uint64_t n1 = i < 21 ? k1 : k1 - ((uint64_t)take << (3 * (i - 21)));
            n1 -= (uint64_t)need << 39;
            if (meldable(x, jokers - need, n0, n1)) return 1;
        }
    }
    return 0;
}

/* The SDK's calculate_fan only needs the numeric fan while analyze() is
 * enumerating waits. Keep the detailed Python Fan object on the reference
 * path, but calculate this hot numeric projection without crossing Python for
 * every candidate tile. */
static int baotou_shape(const int *waiting, int open_melds) {
    if (waiting[WHITE] == 0) return 0;
    int total = 0;
    for (int i = 0; i < NTILES; i++) total += waiting[i];
    if (total != 13 - 3 * open_melds) return 0;
    /* Live v35 four-white/two-meld natural-triplet exception. Keep in sync
     * with rules._baotou; do not suppress other four-white waits. */
    if (open_melds == 2 && waiting[WHITE] == 4) {
        for (int t = 0; t < WHITE; t++) if (waiting[t] == 3) return 0;
    }
    int draw[NTILES];
    memcpy(draw, waiting, sizeof(draw));
    /* An absent non-white honor cannot join a sequence or an existing natural
     * pair. If it wins, its group uses white jokers, so replacing that one draw
     * by any other legal tile preserves a winning partition (including seven
     * pairs). Conversely baotou must win on this draw. One representative
     * therefore proves the all-draw condition; retain enumeration when every
     * non-white honor is already present. */
    for (int t = 27; t < WHITE; t++) {
        if (draw[t] == 0) {
            /* Reserve one joker to pair with the arbitrary draw. If a winning
             * partition instead puts that draw in a joker triplet, move one
             * of those jokers to its existing pair: that pair becomes a meld
             * and the draw takes the remaining joker as its new pair. Thus
             * ordinary hu is equivalent to melding the remaining tiles. */
            int jokers = draw[WHITE] - 1;
            draw[WHITE] = 0;
            if (open_melds == 0) {
                int odd = 0;
                for (int i = 0; i < WHITE; i++) odd += draw[i] % 2;
                if (odd <= jokers) return 1;
            }
            uint64_t k0, k1;
            shanten_key(draw, jokers, &k0, &k1);
            return meldable(draw, jokers, k0, k1);
        }
    }
    /* Honor draws cannot join a sequence. Reject them first, like the
     * Python reference, before exploring more permissive suited draws. */
    for (int offset = 0; offset < NTILES; offset++) {
        int t = offset < 7 ? 27 + offset : offset - 7;
        if (draw[t] >= 4) continue;
        draw[t]++;
        int ok = hu_shape(draw, open_melds);
        draw[t]--;
        if (!ok) return 0;
    }
    return 1;
}

/* Direct-mapped memo for the baotou test: the 34 counts plus open melds. A
 * policy pricing waits asks it for the same waiting hands over and over. */
#define BAOTOU_CACHE_BITS 16
#define BAOTOU_CACHE_SIZE (1 << BAOTOU_CACHE_BITS)

typedef struct { uint64_t k0, k1; } baotou_slot;

static baotou_slot baotou_cache[BAOTOU_CACHE_SIZE];

static int baotou_keyed(uint64_t k0, uint64_t k1, const int *counts, int open_melds) {
    uint64_t h = (k0 * 0x9E3779B97F4A7C15ULL) ^ (k1 * 0xC2B2AE3D27D4EB4FULL);
    h ^= h >> 29;
    baotou_slot *slot = &baotou_cache[h >> (64 - BAOTOU_CACHE_BITS)];
    if (slot->k0 == k0 && (slot->k1 & (KEY1_BITS | KEY_USED)) == (k1 | KEY_USED))
        return (int)(slot->k1 >> VALUE_SHIFT);
    int value = baotou_shape(counts, open_melds);
    slot->k0 = k0;
    slot->k1 = k1 | KEY_USED | ((uint64_t)value << VALUE_SHIFT);
    return value;
}

static int baotou_cached(const int *counts, int open_melds) {
    uint64_t k0, k1;
    shanten_key(counts, open_melds, &k0, &k1);
    return baotou_keyed(k0, k1, counts, open_melds);
}

static long fan_value(const int *counts, int drawn_tile, int open_melds,
                      int chain_count, int piao_count, int you_cai_bi_kao,
                      int kong_draw) {
    if (drawn_tile < 0 || drawn_tile >= NTILES || counts[drawn_tile] <= 0 ||
        !hu_shape(counts, open_melds))
        return 0;
    int waiting[NTILES];
    memcpy(waiting, counts, sizeof(waiting));
    waiting[drawn_tile]--;
    int bt = baotou_cached(waiting, open_melds);
    /* Match the live-server gate in domain/rules.py, including kong draws. */
    if (you_cai_bi_kao && counts[WHITE] && !bt) return 0;

    long fan = 1;
    if (open_melds == 0) {
        int odd = 0, luxury = 0;
        for (int i = 0; i < WHITE; i++) {
            odd += counts[i] % 2;
            if (counts[i] == 4) luxury++;
        }
        if (odd <= counts[WHITE]) {
            if (counts[WHITE] == 4 && odd == 0) luxury++;
            fan = 1L << (1 + luxury);
        }
    }
    if (chain_count > 0) fan *= 1L << chain_count;
    if (counts[WHITE] + piao_count == 4) fan *= 2;
    if (bt) fan *= 2;
    return fan;
}

static int parse_vector(PyObject *obj, int *out, int max_value, const char *what) {
    PyObject *fast = PySequence_Fast(obj, "expected a sequence of 34 ints");
    if (!fast) return -1;
    if (PySequence_Fast_GET_SIZE(fast) != NTILES) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError, "%s must contain exactly 34 entries", what);
        return -1;
    }
    for (int i = 0; i < NTILES; i++) {
        long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
        if (v == -1 && PyErr_Occurred()) {
            Py_DECREF(fast);
            return -1;
        }
        if (v < 0 || v > max_value) {
            Py_DECREF(fast);
            PyErr_Format(PyExc_ValueError, "%s must be 0..%d", what, max_value);
            return -1;
        }
        out[i] = (int)v;
    }
    Py_DECREF(fast);
    return 0;
}

static int parse_counts(PyObject *obj, int *out) {
    return parse_vector(obj, out, 4, "counts");
}

static PyObject *py_shape_value(PyObject *self, PyObject *args) {
    PyObject *obj;
    int open_melds;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &open_melds)) return NULL;
    if (open_melds < 0 || open_melds > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    int counts[NTILES];
    if (parse_counts(obj, counts) < 0) return NULL;
    return PyLong_FromLong(shape_value(counts, open_melds));
}

static PyObject *py_shanten_raw(PyObject *self, PyObject *args) {
    PyObject *obj;
    int open_melds;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &open_melds)) return NULL;
    if (open_melds < 0 || open_melds > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    int counts[NTILES];
    if (parse_counts(obj, counts) < 0) return NULL;
    return PyLong_FromLong(shanten_raw(counts, open_melds));
}

static PyObject *py_can_hu(PyObject *self, PyObject *args) {
    PyObject *obj;
    int open_melds;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &open_melds)) return NULL;
    int counts[NTILES];
    if (parse_counts(obj, counts) < 0) return NULL;
    if (hu_shape(counts, open_melds)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *py_fan_value(PyObject *self, PyObject *args) {
    PyObject *obj;
    int drawn_tile, open_melds, chain_count, piao_count, you_cai_bi_kao,
        kong_draw;
    if (!PyArg_ParseTuple(args, "Oiiiiii", &obj, &drawn_tile, &open_melds,
                          &chain_count, &piao_count, &you_cai_bi_kao,
                          &kong_draw))
        return NULL;
    int counts[NTILES];
    if (parse_counts(obj, counts) < 0) return NULL;
    return PyLong_FromLong(fan_value(counts, drawn_tile, open_melds,
                                     chain_count, piao_count,
                                     you_cai_bi_kao, kong_draw));
}

/* Raw shanten of the hand plus every legal draw, in one C call. */
static PyObject *py_improving(PyObject *self, PyObject *args) {
    PyObject *counts_obj, *avail_obj;
    int open_melds;
    if (!PyArg_ParseTuple(args, "OiO", &counts_obj, &open_melds, &avail_obj)) {
        return NULL;
    }
    if (open_melds < 0 || open_melds > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    int counts[NTILES];
    int avail[NTILES];
    if (parse_counts(counts_obj, counts) < 0) return NULL;
    if (parse_vector(avail_obj, avail, 1000000, "available") < 0) return NULL;
    const draw_slot *table = draw_distances(counts, open_melds);
    int distance = table->distance;
    uint64_t improving = 0, negatives = 0;
    for (int t = 0; t < 34; t++) {
        if (avail[t] <= 0 || counts[t] >= 4) continue;
        int d = table->draw[t];
        if (d < distance) improving |= (uint64_t)1 << t;
        if (d < 0) negatives |= (uint64_t)1 << t;
    }
    PyObject *result = PyTuple_New(3);
    if (!result) return NULL;
    PyObject *distance_obj = PyLong_FromLong(distance);
    if (!distance_obj) {
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, distance_obj);
    for (int k = 0; k < 2; k++) {
        uint64_t mask = k == 0 ? improving : negatives;
        int n = 0;
        for (int t = 0; t < 34; t++)
            if (mask >> t & 1) n++;
        PyObject *row = PyTuple_New(n);
        if (!row) {
            Py_DECREF(result);
            return NULL;
        }
        int pos = 0;
        for (int t = 0; t < 34; t++) {
            if (!(mask >> t & 1)) continue;
            PyObject *item = PyLong_FromLong(t);
            if (!item) {
                Py_DECREF(row);
                Py_DECREF(result);
                return NULL;
            }
            PyTuple_SET_ITEM(row, pos++, item);
        }
        PyTuple_SET_ITEM(result, k + 1, row);
    }
    return result;
}

/* The whole fast path of sdk.shape.analyze for a waiting-length hand: raw
 * distance, improving draws, and the numeric fan of every winning draw
 * (fan_value of the hand plus that draw, as the Python loop computes it). */
static PyObject *analysis_keys[7];

typedef struct {
    int distance, n_improving, n_wins;
    int improving[NTILES], wins[NTILES];
    long fans[NTILES], ukeire, copies, value;
} analysis_row;

static void analyze_values(const int *counts, const int *avail, int open_melds,
                           int chain_count, int piao_count, int you_cai_bi_kao,
                           analysis_row *out) {
    int drawn[NTILES];
    const draw_slot *table = draw_distances(counts, open_melds);
    int distance = table->distance;
    int improving[NTILES], wins[NTILES];
    long fans[NTILES];
    int n_improving = 0, n_wins = 0;
    for (int t = 0; t < NTILES; t++) {
        if (avail[t] <= 0 || counts[t] >= 4) continue;
        int d = table->draw[t];
        if (d < distance) improving[n_improving++] = t;
        if (d < 0) {
            memcpy(drawn, counts, sizeof(drawn));
            drawn[t]++;
            long fan = fan_value(drawn, t, open_melds, chain_count, piao_count,
                                 you_cai_bi_kao, 0);
            if (fan) {
                wins[n_wins] = t;
                fans[n_wins++] = fan;
            }
        }
    }
    /* The availability-weighted sums sdk.shape.analyze reports, so the
     * Python side builds its result without iterating the tuples again. */
    long ukeire = 0, copies = 0, value = 0;
    for (int i = 0; i < n_improving; i++) ukeire += avail[improving[i]];
    for (int i = 0; i < n_wins; i++) {
        copies += avail[wins[i]];
        value += avail[wins[i]] * fans[i];
    }
    out->distance = distance;
    out->n_improving = n_improving;
    out->n_wins = n_wins;
    out->ukeire = ukeire;
    out->copies = copies;
    out->value = value;
    memcpy(out->improving, improving, n_improving * sizeof(int));
    memcpy(out->wins, wins, n_wins * sizeof(int));
    memcpy(out->fans, fans, n_wins * sizeof(long));
}

static PyObject *analyze_result(const int *counts, const int *avail,
                                int open_melds, int chain_count, int piao_count,
                                int you_cai_bi_kao, int as_dict) {
    analysis_row row;
    analyze_values(counts, avail, open_melds, chain_count, piao_count, you_cai_bi_kao, &row);
    int distance = row.distance;
    if (as_dict && distance < 0) Py_RETURN_NONE;
    int n_improving = row.n_improving, n_wins = row.n_wins;
    int *improving = row.improving, *wins = row.wins;
    long *fans = row.fans, ukeire = row.ukeire, copies = row.copies, value = row.value;
    if (as_dict) {
        PyObject *imp = PyList_New(n_improving), *win = PyList_New(n_wins),
                 *fan = PyDict_New();
        if (!imp || !win || !fan) {
            Py_XDECREF(imp); Py_XDECREF(win); Py_XDECREF(fan);
            return NULL;
        }
        for (int i = 0; i < n_improving; i++)
            PyList_SET_ITEM(imp, i, PyLong_FromLong(improving[i]));
        for (int i = 0; i < n_wins; i++) {
            PyObject *tile = PyLong_FromLong(wins[i]);
            PyObject *number = PyLong_FromLong(fans[i]);
            if (!tile || !number || PyDict_SetItem(fan, tile, number) < 0) {
                Py_XDECREF(tile); Py_XDECREF(number);
                Py_DECREF(imp); Py_DECREF(win); Py_DECREF(fan);
                return NULL;
            }
            PyList_SET_ITEM(win, i, tile);
            Py_DECREF(number);
        }
        PyObject *values[7] = {PyLong_FromLong(distance), imp,
            PyLong_FromLong(ukeire), win, PyLong_FromLong(copies), fan,
            PyLong_FromLong(value)};
        PyObject *result = PyDict_New();
        for (int i = 0; i < 7; i++) {
            if (!values[i] || (result && PyDict_SetItem(result, analysis_keys[i], values[i]) < 0)) {
                Py_CLEAR(result);
            }
            Py_XDECREF(values[i]);
        }
        return result;
    }
    PyObject *imp = PyTuple_New(n_improving), *win = PyTuple_New(n_wins),
             *fan = PyTuple_New(n_wins);
    if (!imp || !win || !fan) goto fail;
    for (int i = 0; i < n_improving; i++)
        PyTuple_SET_ITEM(imp, i, PyLong_FromLong(improving[i]));
    for (int i = 0; i < n_wins; i++) {
        PyTuple_SET_ITEM(win, i, PyLong_FromLong(wins[i]));
        PyTuple_SET_ITEM(fan, i, PyLong_FromLong(fans[i]));
    }
    return Py_BuildValue("(iNNNlll)", distance, imp, win, fan, ukeire, copies, value);
fail:
    Py_XDECREF(imp);
    Py_XDECREF(win);
    Py_XDECREF(fan);
    return NULL;
}

static PyObject *py_analyze_counts(PyObject *self, PyObject *args) {
    PyObject *counts_obj, *avail_obj;
    int open_melds, chain_count, piao_count, you_cai_bi_kao;
    if (!PyArg_ParseTuple(args, "OiOiii", &counts_obj, &open_melds, &avail_obj,
                          &chain_count, &piao_count, &you_cai_bi_kao)) return NULL;
    if (open_melds < 0 || open_melds > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    int counts[NTILES], avail[NTILES];
    if (parse_counts(counts_obj, counts) < 0 ||
        parse_vector(avail_obj, avail, 1000000, "available") < 0) return NULL;
    return analyze_result(counts, avail, open_melds, chain_count, piao_count,
                          you_cai_bi_kao, 0);
}

/* Count tile ids and build the public result without a 34-element Python
 * count tuple and intermediate result tuples on every hypothetical discard. */
static PyObject *py_analyze_hand(PyObject *self, PyObject *args) {
    PyObject *hand, *avail_obj;
    int opened, chain, piao, you_cai;
    if (!PyArg_ParseTuple(args, "OiOiii", &hand, &opened, &avail_obj,
                          &chain, &piao, &you_cai)) return NULL;
    if (opened < 0 || opened > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(hand, "hand must be a sequence");
    if (!seq) return NULL;
    int counts[NTILES] = {0}, avail[NTILES];
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long tile = PyLong_Check(item) ? PyLong_AsLong(item) : -1;
        if (tile == -1 && PyErr_Occurred()) PyErr_Clear();
        if (tile < 0 || tile >= NTILES) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "invalid tile");
            return NULL;
        }
        if (++counts[tile] > 4) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "more than four copies");
            return NULL;
        }
    }
    Py_DECREF(seq);
    if (size != 13 - 3 * opened) {
        PyErr_SetString(PyExc_ValueError, "analyze_hand expects a waiting hand");
        return NULL;
    }
    if (avail_obj == Py_None) {
        for (int i = 0; i < NTILES; i++) avail[i] = 4 - counts[i];
    } else if (parse_vector(avail_obj, avail, 1000000, "available") < 0) return NULL;
    return analyze_result(counts, avail, opened, chain, piao, you_cai, 1);
}

static PyObject *py_baotou(PyObject *self, PyObject *args) {
    PyObject *obj;
    int open_melds;
    int counts[NTILES];
    if (!PyArg_ParseTuple(args, "Oi", &obj, &open_melds)) return NULL;
    if (parse_counts(obj, counts) < 0) return NULL;
    if (open_melds < 0 || open_melds > 4) Py_RETURN_FALSE;
    return PyBool_FromLong(baotou_shape(counts, open_melds));
}

/* rules.is_baotou for a hand of tile ids: the counts, the cheap rejections of
 * rules._baotou (wrong length, no white), then the memoized shape test. None
 * hands back to the Python path where it would iterate itself (a count above
 * four, which never occurs in play). */
static PyObject *py_is_baotou(PyObject *self, PyObject *args) {
    PyObject *hand;
    int open_melds;
    if (!PyArg_ParseTuple(args, "Oi", &hand, &open_melds)) return NULL;
    PyObject *fast = PySequence_Fast(hand, "hand must be a sequence");
    if (!fast) return NULL;
    int counts[NTILES] = {0};
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        long t = PyLong_Check(item) ? PyLong_AsLong(item) : -1;
        if (t == -1 && PyErr_Occurred()) PyErr_Clear();
        if (!PyLong_Check(item) || t < 0 || t >= NTILES) {
            PyObject *text = PyObject_Str(item);
            if (text) {
                PyErr_Format(PyExc_ValueError, "invalid tile: %U", text);
                Py_DECREF(text);
            }
            Py_DECREF(fast);
            return NULL;
        }
        counts[t]++;
    }
    Py_DECREF(fast);
    if (n != 13 - 3 * (Py_ssize_t)open_melds || counts[WHITE] == 0) Py_RETURN_FALSE;
    for (int i = 0; i < NTILES; i++)
        if (counts[i] > 4) Py_RETURN_NONE;
    if (open_melds < 0 || open_melds > 4) Py_RETURN_FALSE;
    return PyBool_FromLong(baotou_cached(counts, open_melds));
}

/* rules.counts for a list or tuple of tile ids, with the same validation:
 * any int in 0..33 (bool included, as isinstance(t, int) accepts it). */
static PyObject *py_counts(PyObject *self, PyObject *hand) {
    PyObject *fast = PySequence_Fast(hand, "hand must be a sequence");
    if (!fast) return NULL;
    long counts[NTILES] = {0};
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        long t = PyLong_Check(item) ? PyLong_AsLong(item) : -1;
        if (t == -1 && PyErr_Occurred()) PyErr_Clear();
        if (!PyLong_Check(item) || t < 0 || t >= NTILES) {
            PyObject *text = PyObject_Str(item);
            if (text) {
                PyErr_Format(PyExc_ValueError, "invalid tile: %U", text);
                Py_DECREF(text);
            }
            Py_DECREF(fast);
            return NULL;
        }
        counts[t]++;
    }
    Py_DECREF(fast);
    PyObject *result = PyTuple_New(NTILES);
    if (!result) return NULL;
    for (int i = 0; i < NTILES; i++) {
        PyObject *v = PyLong_FromLong(counts[i]);
        if (!v) { Py_DECREF(result); return NULL; }
        PyTuple_SET_ITEM(result, i, v);
    }
    return result;
}

/* Two-ply public-hand enumeration. Keep tile counts in C instead of sorting
 * and allocating a Python hand and analysis dictionary for every discard. */
static PyObject *py_baotou_outs(PyObject *self, PyObject *args) {
    PyObject *obj, *available;
    int opened, c[NTILES], avail[NTILES];
    if (!PyArg_ParseTuple(args, "OiO", &obj, &opened, &available)) return NULL;
    if (opened < 0 || opened > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    if (parse_counts(obj, c) < 0 ||
        parse_vector(available, avail, 1000000, "available") < 0) return NULL;
    long total = 0;
    if (!c[WHITE]) return PyLong_FromLong(0);
    uint64_t k0, k1;
    shanten_key(c, opened, &k0, &k1);
    for (int tile = 0; tile < NTILES; tile++) {
        if (avail[tile] <= 0 || c[tile] >= 4) continue;
        uint64_t d0, d1;
        shanten_key_draw(k0, k1, tile, &d0, &d1);
        c[tile]++;
        for (int drop = 0; drop < NTILES; drop++) {
            if (!c[drop]) continue;
            c[drop]--;
            uint64_t r0 = drop < 21 ? d0 - (1ULL << (3 * drop)) : d0;
            uint64_t r1 = drop < 21 ? d1 : d1 - (1ULL << (3 * (drop - 21)));
            int ok = baotou_keyed(r0, r1, c, opened);
            c[drop]++;
            if (ok) {
                total += avail[tile];
                break;
            }
        }
        c[tile]--;
    }
    return PyLong_FromLong(total);
}

static PyObject *py_tenpai_route_value(PyObject *self, PyObject *args) {
    PyObject *obj, *available;
    int opened, chain, piao, you_cai, c[NTILES], avail[NTILES];
    if (!PyArg_ParseTuple(args, "OiOiii", &obj, &opened, &available,
                         &chain, &piao, &you_cai)) return NULL;
    if (opened < 0 || opened > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    if (parse_counts(obj, c) < 0 ||
        parse_vector(available, avail, 1000000, "available") < 0) return NULL;
    int size = 0;
    for (int t = 0; t < NTILES; t++) size += c[t];
    if (size != 13 - 3 * opened) return PyFloat_FromDouble(0.0);
    const draw_slot *table = draw_distances(c, opened);
    if (table->distance != 1) return PyFloat_FromDouble(0.0);
    /* Nested lookups can replace this direct-mapped cache entry. */
    uint64_t improving = 0;
    for (int t = 0; t < NTILES; t++)
        if (avail[t] > 0 && c[t] < 4 && table->draw[t] < 1)
            improving |= (uint64_t)1 << t;
    long total = 0;
    double weighted = 0.0;
    for (int tile = 0; tile < NTILES; tile++) {
        if (!(improving >> tile & 1)) continue;
        c[tile]++;
        long best = 0;
        for (int drop = 0; drop < NTILES; drop++) {
            if (!c[drop]) continue;
            c[drop]--;
            if (shanten_raw(c, opened) > 0) {
                c[drop]++;
                continue;
            }
            const draw_slot *dest = draw_distances(c, opened);
            /* A waiting-length hand cannot be complete. The Python guarded
             * shanten maps negative raw distances to zero here. */
            if (dest->distance <= 0) {
                long value = 0;
                for (int t = 0; t < NTILES; t++) {
                    if (avail[t] <= 0 || c[t] >= 4 || dest->draw[t] >= 0) continue;
                    c[t]++;
                    value += avail[t] * fan_value(c, t, opened, chain, piao, you_cai, 0);
                    c[t]--;
                }
                if (value > best) best = value;
            }
            c[drop]++;
        }
        c[tile]--;
        total += avail[tile];
        weighted += (double)avail[tile] * best;
    }
    return PyFloat_FromDouble(total ? weighted / total : 0.0);
}

/* Count only visible tile ids. Concealed opponent melds have already been
 * redacted by Observation, so this routine never reads an engine state. */
static int subtract_visible(PyObject *tiles, int *available) {
    PyObject *seq = PySequence_Fast(tiles, "expected visible tile sequence");
    if (!seq) return -1;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
        long tile = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) { Py_DECREF(seq); return -1; }
        /* The Python Counter reference ignores keys outside 0..33. */
        if (tile >= 0 && tile < NTILES && available[tile] > 0) available[tile]--;
    }
    Py_DECREF(seq);
    return 0;
}

static PyObject *py_remaining_tiles(PyObject *self, PyObject *args) {
    PyObject *hand, *discards, *melds;
    if (!PyArg_ParseTuple(args, "OOO", &hand, &discards, &melds)) return NULL;
    int available[NTILES];
    for (int t = 0; t < NTILES; t++) available[t] = 4;
    if (subtract_visible(hand, available) < 0) return NULL;
    PyObject *rows = PySequence_Fast(discards, "expected discard rows");
    if (!rows) return NULL;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(rows); i++) {
        if (subtract_visible(PySequence_Fast_GET_ITEM(rows, i), available) < 0) {
            Py_DECREF(rows); return NULL;
        }
    }
    Py_DECREF(rows);
    rows = PySequence_Fast(melds, "expected meld rows");
    if (!rows) return NULL;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(rows); i++) {
        PyObject *row = PySequence_Fast(PySequence_Fast_GET_ITEM(rows, i), "expected melds");
        if (!row) { Py_DECREF(rows); return NULL; }
        for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(row); j++) {
            PyObject *tiles = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(row, j), "tiles");
            if (!tiles) { Py_DECREF(row); Py_DECREF(rows); return NULL; }
            int ok = subtract_visible(tiles, available);
            Py_DECREF(tiles);
            if (ok < 0) { Py_DECREF(row); Py_DECREF(rows); return NULL; }
        }
        Py_DECREF(row);
    }
    Py_DECREF(rows);
    PyObject *result = PyTuple_New(NTILES);
    if (!result) return NULL;
    for (int t = 0; t < NTILES; t++) {
        PyObject *value = PyLong_FromLong(available[t]);
        if (!value) { Py_DECREF(result); return NULL; }
        PyTuple_SET_ITEM(result, t, value);
    }
    return result;
}

/* Fill the six public lookahead planes in one native call. Negative raw
 * boundaries defer to the existing reference-aware encoder before any writes. */
static PyObject *py_encode_lookahead(PyObject *self, PyObject *args) {
    PyObject *hand, *available, *buffer;
    int opened, chain, piao, you_cai, counts[NTILES] = {0}, avail[NTILES];
    if (!PyArg_ParseTuple(args, "OiOiiiO", &hand, &opened, &available,
                          &chain, &piao, &you_cai, &buffer)) return NULL;
    if (opened < 0 || opened > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(hand, "hand must be a sequence");
    if (!seq) return NULL;
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        long tile = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (tile == -1 && PyErr_Occurred()) { Py_DECREF(seq); return NULL; }
        if (tile < 0 || tile >= NTILES || ++counts[tile] > 4) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "invalid physical hand");
            return NULL;
        }
    }
    Py_DECREF(seq);
    if (size != 13 - 3 * opened && size != 14 - 3 * opened) Py_RETURN_NONE;
    if (parse_vector(available, avail, 1000000, "available") < 0) return NULL;
    analysis_row rows[NTILES], best;
    int discard = size == 14 - 3 * opened, found = 0;
    for (int tile = 0; tile < (discard ? NTILES : 1); tile++) {
        if (discard && !counts[tile]) continue;
        if (discard) counts[tile]--;
        analyze_values(counts, avail, opened, chain, piao, you_cai, &rows[tile]);
        if (discard) counts[tile]++;
        analysis_row *r = &rows[tile];
        if (r->distance < 0) Py_RETURN_NONE;
        if (!found || r->distance < best.distance ||
            (r->distance == best.distance && (r->ukeire > best.ukeire ||
             (r->ukeire == best.ukeire && r->value > best.value)))) {
            best = *r;
            found = 1;
        }
    }
    Py_buffer view;
    if (PyObject_GetBuffer(buffer, &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (view.len != 6 * NTILES * sizeof(float) || view.itemsize != sizeof(float) ||
        strcmp(view.format, "f") != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "lookahead buffer must be contiguous native float32[6,34]");
        return NULL;
    }
    float *out = (float *)view.buf;
    memset(out, 0, view.len);
    if (discard) {
        for (int tile = 0; tile < NTILES; tile++) {
            if (!counts[tile]) continue;
            analysis_row *r = &rows[tile];
            out[tile] = (8 - r->distance) / 9.0;
            out[NTILES + tile] = (r->ukeire < 40 ? r->ukeire : 40) / 40.0;
            out[2 * NTILES + tile] = (r->copies < 12 ? r->copies : 12) / 12.0;
            out[3 * NTILES + tile] = log1p((double)r->value) / 8.0;
        }
    }
    for (int i = 0; i < best.n_improving; i++)
        out[4 * NTILES + best.improving[i]] = avail[best.improving[i]] / 4.0;
    for (int i = 0; i < best.n_wins; i++)
        out[5 * NTILES + best.wins[i]] = log2((double)best.fans[i] + 1.0) / 9.0;
    PyBuffer_Release(&view);
    return Py_BuildValue("(illl)", best.distance, best.ukeire, best.copies, best.value);
}

/* Public count features share one traversal instead of nine NumPy bincounts. */
static int public_counts(PyObject *tiles, int *counts, int *last) {
    PyObject *seq = PySequence_Fast(tiles, "tiles must be a sequence");
    if (!seq) return -1;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
        long tile = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if ((tile == -1 && PyErr_Occurred()) || tile < 0 || tile >= NTILES) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "invalid tile");
            Py_DECREF(seq);
            return -1;
        }
        counts[tile]++;
        if (last) *last = (int)tile;
    }
    Py_DECREF(seq);
    return 0;
}

static void count_planes(float *out, const int *counts) {
    for (int n = 0; n < 4; n++)
        for (int tile = 0; tile < NTILES; tile++)
            out[n * NTILES + tile] = counts[tile] > n;
}

static PyObject *py_encode_public_planes(PyObject *self, PyObject *args) {
    PyObject *hand, *discards, *melds, *drawn, *buffer;
    int seat;
    if (!PyArg_ParseTuple(args, "OOOOiO", &hand, &discards, &melds, &drawn, &seat, &buffer))
        return NULL;
    if (seat < 0 || seat > 3) {
        PyErr_SetString(PyExc_ValueError, "seat must be 0..3");
        return NULL;
    }
    PyObject *rivers = PySequence_Fast(discards, "expected four discard rows");
    PyObject *groups = PySequence_Fast(melds, "expected four meld rows");
    if (!rivers || !groups) { Py_XDECREF(rivers); Py_XDECREF(groups); return NULL; }
    if (PySequence_Fast_GET_SIZE(rivers) != 4 || PySequence_Fast_GET_SIZE(groups) != 4) {
        Py_DECREF(rivers); Py_DECREF(groups);
        PyErr_SetString(PyExc_ValueError, "expected four public seats");
        return NULL;
    }
    Py_buffer view;
    if (PyObject_GetBuffer(buffer, &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        Py_DECREF(rivers); Py_DECREF(groups); return NULL;
    }
    if (view.len != 45 * NTILES * sizeof(float) || view.itemsize != sizeof(float) ||
        strcmp(view.format, "f") != 0) {
        PyErr_SetString(PyExc_ValueError, "public buffer must be contiguous native float32[45,34]");
        goto fail;
    }
    float *out = (float *)view.buf;
    memset(out, 0, view.len);
    int visible[NTILES] = {0};
    if (public_counts(hand, visible, NULL) < 0) goto fail;
    count_planes(out, visible);
    if (drawn != Py_None) {
        long tile = PyLong_AsLong(drawn);
        if ((tile == -1 && PyErr_Occurred()) || tile < 0 || tile >= NTILES) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "invalid drawn tile");
            goto fail;
        }
        out[4 * NTILES + tile] = 1.0f;
    }
    for (int i = 0; i < 4; i++) {
        int physical = (seat + i) % 4;
        int counted[NTILES] = {0};
        PyObject *row = PySequence_Fast(PySequence_Fast_GET_ITEM(groups, physical), "expected melds");
        if (!row) goto fail;
        for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(row); j++) {
            PyObject *tiles = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(row, j), "tiles");
            if (!tiles) { Py_DECREF(row); goto fail; }
            int ok = public_counts(tiles, counted, NULL);
            Py_DECREF(tiles);
            if (ok < 0) { Py_DECREF(row); goto fail; }
        }
        Py_DECREF(row);
        count_planes(out + (5 + 4 * i) * NTILES, counted);
        for (int tile = 0; tile < NTILES; tile++) visible[tile] += counted[tile];
        memset(counted, 0, sizeof(counted));
        int last = -1;
        if (public_counts(PySequence_Fast_GET_ITEM(rivers, physical), counted, &last) < 0) goto fail;
        count_planes(out + (21 + 4 * i) * NTILES, counted);
        for (int tile = 0; tile < NTILES; tile++) visible[tile] += counted[tile];
        if (last >= 0) out[(37 + i) * NTILES + last] = 1.0f;
    }
    PyObject *result = PyTuple_New(NTILES);
    if (!result) goto fail;
    for (int tile = 0; tile < NTILES; tile++) {
        visible[tile] = visible[tile] < 4 ? 4 - visible[tile] : 0;
        PyObject *v = PyLong_FromLong(visible[tile]);
        if (!v) { Py_DECREF(result); goto fail; }
        PyTuple_SET_ITEM(result, tile, v);
    }
    count_planes(out + 41 * NTILES, visible);
    PyBuffer_Release(&view); Py_DECREF(rivers); Py_DECREF(groups);
    return result;
fail:
    PyBuffer_Release(&view); Py_DECREF(rivers); Py_DECREF(groups);
    return NULL;
}


/* ---------- race valuation of the first-principles rule strategies ---------- */

/* waiting_value and reshaped_value of the rule strategies that value a hand by
 * racing the opponents to a win (firstprinciples fp-rule-033 and the fixed
 * pool's members built on it), ported with the same arithmetic in the same
 * order so the results equal the Python ones bit for bit. A profile holds a
 * strategy's constants. Whether tiles meld, win or wait on baotou is read
 * from per-suit meld tables (meld_tables) instead of meldable's search.
 * Values are memoised per hand, availability, profile and the other
 * arguments; the memo only saves recomputation. A hand the
 * extension counts as complete although it waits (a white-joker boundary)
 * is analysed by the caller's Python function, as sdk.shape.analyze does. */

/* Python rounds every product and sum on its own: no fused multiply-adds here,
 * whatever the compiler's defaults. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#define RACE_PROFILES 64
#define RACE_CACHE_BITS 18

typedef struct {
    double q, saved, one_step, far_step, q_early, q_late, hunt, draw_risk, tie;
    int upgrade, last_wall, style;
    /* complete: a wait without a white also counts drawing a white that a
     * discard turns into baotou, and the kong a concealed triplet's last copy
     * gives (fp-rule-043); guard: no draws left counts no draws (repeat of 0
     * or fewer is 0, as those strategies write it). */
    int complete, guard;
} race_profile;

typedef struct {
    uint64_t k[4];
    double value;
} race_slot;

static race_profile race_profiles[RACE_PROFILES];
static int race_profile_count;
static race_slot race_cache[1 << RACE_CACHE_BITS];

typedef struct {
    const race_profile *p;
    int profile, opened, unseen, size, keyed;
    double speed;  /* scales the opponents' chance to win first; speed_key * 20 of it, for the memo */
    int speed_key;
    int avail[NTILES];
    uint64_t a0, a1; /* the availability's part of the memo key */
    PyObject *available, *fallback;
} race_ctx;

static double race_repeat(double stay, int draws) {
    return (1 - pow(stay, (double)draws)) / (1 - stay);
}

static double race_hazard(const race_profile *p, int draws) {
    double x = 0.35 - 0.018 * draws;
    double y = x > p->q_early ? x : p->q_early;  /* max(Q_EARLY, x) */
    return y < p->q_late ? y : p->q_late;        /* min(Q_LATE, y) */
}

/* race_repeat under the profile's guard. */
static double race_repeat_x(const race_ctx *x, double stay, int draws) {
    if (x->p->guard && draws <= 0) return 0.0;
    return race_repeat(stay, draws);
}

/* tenpai_hazard(draws, speed): the curve, then the speed. */
static double race_hazard_x(const race_ctx *x, int draws) {
    return race_hazard(x->p, draws) * x->speed;
}

static double race_race(const race_ctx *x, double chance, double gain, int draws) {
    return gain * race_repeat_x(x, (1 - chance) * (1 - race_hazard_x(x, draws)), draws);
}

/* The order Python iterates a set of small ints built by adding them in the
 * given order (CPython's open addressing: 8 slots, grown when five in eight
 * are used, to the power of two above four times the count; a probe checks
 * the slot and the nine after it when they fit, then moves to i * 5 + 1 +
 * perturb). A sum over set(hand) adds in this order, and floats added in
 * another order may round differently. */
static void py_set_insert(int *table, int mask, int key) {
    size_t perturb = (size_t)key, i = (size_t)key & (size_t)mask;
    for (;;) {
        int probes = i + 9 <= (size_t)mask ? 9 : 0;
        size_t j = i;
        do {
            if (table[j] < 0) {
                table[j] = key;
                return;
            }
            j++;
        } while (probes--);
        perturb >>= 5;
        i = (i * 5 + 1 + perturb) & (size_t)mask;
    }
}

static int py_set_order(const int *items, int n, int *out) {
    int table[128], old[128], mask = 7, used = 0;
    for (int i = 0; i < 128; i++) table[i] = -1;
    for (int k = 0; k < n; k++) {
        py_set_insert(table, mask, items[k]);
        if (++used * 5 >= mask * 3) {
            int size = 8, old_mask = mask;
            while (size <= used * 4) size <<= 1;
            memcpy(old, table, sizeof(old));
            mask = size - 1;
            for (int i = 0; i < 128; i++) table[i] = -1;
            for (int i = 0; i <= old_mask; i++)
                if (old[i] >= 0) py_set_insert(table, mask, old[i]);
        }
    }
    int m = 0;
    for (int i = 0; i <= mask; i++)
        if (table[i] >= 0) out[m++] = table[i];
    return m;
}

/* fan_value of each draw on a baotou wait, summed as analyze_values does:
 * every drawable tile completes a baotou wait (baotou_shape), so hu_shape
 * holds for each and the wait itself is the baotou the fan doubles for. */
static void baotou_fans(const int *c, const int *avail, int opened, int chain, int piao,
                        long *copies, long *value) {
    int odd = 0, luxury = 0;
    for (int i = 0; i < WHITE; i++) {
        odd += c[i] % 2;
        if (c[i] == 4) luxury++;
    }
    long total = 0, weighted = 0;
    for (int t = 0; t < NTILES; t++) {
        if (avail[t] <= 0 || c[t] >= 4) continue;
        int o = odd, l = luxury, whites = c[WHITE];
        if (t == WHITE) whites++;
        else {
            o += c[t] % 2 ? -1 : 1;
            if (c[t] == 3) l++;
        }
        long fan = 1;
        if (opened == 0 && o <= whites) {
            if (whites == 4 && o == 0) l++;
            fan = 1L << (1 + l);
        }
        if (chain > 0) fan *= 1L << chain;
        if (whites + piao == 4) fan *= 2;
        fan *= 2;
        total += avail[t];
        weighted += avail[t] * fan;
    }
    *copies = total;
    *value = weighted;
}

static PyObject *py_race_baotou_fans(PyObject *self, PyObject *args) {
    PyObject *counts_obj, *avail_obj;
    int opened, chain, piao, c[NTILES], avail[NTILES];
    if (!PyArg_ParseTuple(args, "OiOii", &counts_obj, &opened, &avail_obj, &chain, &piao)) return NULL;
    if (parse_counts(counts_obj, c) < 0 || parse_vector(avail_obj, avail, 1000000, "available") < 0) return NULL;
    long copies, value;
    baotou_fans(c, avail, opened, chain, piao, &copies, &value);
    return Py_BuildValue("ll", copies, value);
}

/* Meld tables. Per group of tiles (a numbered suit, or one honor), the fewest
 * jokers that complete its tiles into melds (triplets, and sequences within
 * a suit, a joker in the place of each missing tile), and the fewest that
 * do so with one pair among them: meldable's search, answered per group from
 * a table. A move of that search (the group of the lowest tile) costs 3 less
 * the tiles it takes, so the tiles and jokers left keep their count modulo
 * three, and by induction c melds with J jokers exactly when its tiles and
 * the jokers make whole melds (a multiple of three) and its groups' needs sum
 * to at most J: the jokers to spare meld on their own. Needs are capped at
 * MELD_FAR, more than the four whites there are, so every J passed is at
 * most four. A numbered suit's nine counts index its table in base five.
 * Whites are never natural here: callers pass counts without them. */
#define MELD_FAR 5
#define MELD_SUITS 1953125

static uint8_t *meld_table; /* need | pair need << 4, built on first use */
static const int meld_pow5[9] = {1, 5, 25, 125, 625, 3125, 15625, 78125, 390625};
/* An honor's copies: 0..4 */
static const uint8_t meld_honor[5] = {0 | MELD_FAR << 4, 2 | 1 << 4, 1 | 0 << 4, 0 | 2 << 4, 2 | 1 << 4};

static int meld_tables(void) {
    if (meld_table) return 0;
    uint8_t *t = PyMem_RawMalloc(MELD_SUITS);
    if (!t) {
        PyErr_NoMemory();
        return -1;
    }
    int d[9] = {0};
    t[0] = 0 | MELD_FAR << 4;
    for (int x = 1; x < MELD_SUITS; x++) {
        for (int i = 0; ++d[i] == 5; i++) d[i] = 0;
        int f = 0;
        while (!d[f]) f++;
        int need = MELD_FAR, pair = MELD_FAR;
        for (int take = d[f] < 3 ? d[f] : 3; take >= 1; take--) {
            int v = 3 - take + (t[x - take * meld_pow5[f]] & 15);
            if (v < need) need = v;
        }
        for (int s = f > 2 ? f - 2 : 0; s <= f && s <= 6; s++) {
            int y = x, missing = 0;
            for (int i = s; i < s + 3; i++) {
                if (d[i]) y -= meld_pow5[i];
                else missing++;
            }
            int v = missing + (t[y] & 15);
            if (v < need) need = v;
        }
        for (int i = f; i < 9; i++) {
            if (!d[i]) continue;
            int v = 1 + (t[x - meld_pow5[i]] & 15);
            if (d[i] >= 2 && (t[x - 2 * meld_pow5[i]] & 15) < v) v = t[x - 2 * meld_pow5[i]] & 15;
            if (v < pair) pair = v;
        }
        t[x] = (uint8_t)(need | pair << 4);
    }
    meld_table = t;
    return 0;
}

typedef struct {
    int index[3];         /* the suits' table indices */
    int need[4], pair[4]; /* three suits, then the honors together */
    int total, tiles;     /* the needs summed; the natural tiles */
} meld_groups;

static void meld_honors(const int *c, meld_groups *g) {
    int need = 0, delta = MELD_FAR;
    for (int t = 27; t < WHITE; t++) {
        uint8_t e = meld_honor[c[t]];
        need += e & 15;
        if ((e >> 4) - (e & 15) < delta) delta = (e >> 4) - (e & 15);
    }
    g->need[3] = need;
    g->pair[3] = need + delta;
}

static void meld_groups_of(const int *c, meld_groups *g) {
    g->tiles = 0;
    for (int s = 0; s < 3; s++) {
        int x = 0;
        for (int i = 8; i >= 0; i--) {
            x = x * 5 + c[9 * s + i];
            g->tiles += c[9 * s + i];
        }
        g->index[s] = x;
        g->need[s] = meld_table[x] & 15;
        g->pair[s] = meld_table[x] >> 4;
    }
    for (int t = 27; t < WHITE; t++) g->tiles += c[t];
    meld_honors(c, g);
    g->total = g->need[0] + g->need[1] + g->need[2] + g->need[3];
}

/* g with one natural tile t more (c: the counts g was made of, t added). */
static void meld_groups_add(meld_groups *g, const int *c, int t) {
    g->tiles++;
    if (t < 27) {
        int s = t / 9;
        g->index[s] += meld_pow5[t - 9 * s];
        uint8_t e = meld_table[g->index[s]];
        g->total += (e & 15) - g->need[s];
        g->need[s] = e & 15;
        g->pair[s] = e >> 4;
    } else {
        g->total -= g->need[3];
        meld_honors(c, g);
        g->total += g->need[3];
    }
}

/* meldable(c, jokers): the naturals of c meld with the jokers. */
static int meld_groups_meld(const meld_groups *g, int jokers) {
    return (g->tiles + jokers) % 3 == 0 && g->total <= jokers;
}

/* hu_shape's standard shape: a pair (two jokers, or a group's) and melds. */
static int meld_groups_hu(const meld_groups *g, int jokers) {
    if (jokers >= 2 && g->total + 2 <= jokers) return 1;
    for (int s = 0; s < 4; s++)
        if (g->total - g->need[s] + g->pair[s] <= jokers) return 1;
    return 0;
}

/* The natural tiles d whose removal leaves c meldable with the jokers, as bit d. */
static uint64_t meld_groups_leftovers(const meld_groups *g, const int *c, int jokers) {
    if ((g->tiles - 1 + jokers) % 3) return 0;
    uint64_t mask = 0;
    for (int s = 0; s < 3; s++) {
        int rest = g->total - g->need[s];
        if (rest > jokers) continue;
        for (int i = 0; i < 9; i++)
            if (c[9 * s + i] && rest + (meld_table[g->index[s] - meld_pow5[i]] & 15) <= jokers)
                mask |= 1ULL << (9 * s + i);
    }
    for (int t = 27; t < WHITE; t++)
        if (c[t] && g->total - (meld_honor[c[t]] & 15) + (meld_honor[c[t] - 1] & 15) <= jokers) mask |= 1ULL << t;
    return mask;
}

/* hu_shape(counts, open_melds), with the tables. */
static int race_hu(const int *counts, int open_melds) {
    if (open_melds < 0 || open_melds > 4) return 0;
    int total = 0;
    for (int i = 0; i < NTILES; i++) total += counts[i];
    if (total != 14 - 3 * open_melds) return 0;
    int jokers = counts[WHITE];
    if (open_melds == 0) {
        int odd = 0;
        for (int i = 0; i < WHITE; i++) odd += counts[i] % 2;
        if (odd <= jokers) return 1;
    }
    meld_groups g;
    meld_groups_of(counts, &g);
    return meld_groups_hu(&g, jokers);
}

/* baotou_shape(waiting, open_melds), with the tables: the same tests. */
static int race_baotou_shape(const int *waiting, int open_melds) {
    if (waiting[WHITE] == 0) return 0;
    int total = 0;
    for (int i = 0; i < NTILES; i++) total += waiting[i];
    if (total != 13 - 3 * open_melds) return 0;
    if (open_melds == 2 && waiting[WHITE] == 4) {
        for (int t = 0; t < WHITE; t++) if (waiting[t] == 3) return 0;
    }
    int odd = 0;
    for (int i = 0; i < WHITE; i++) odd += waiting[i] % 2;
    meld_groups g;
    meld_groups_of(waiting, &g);
    for (int t = 27; t < WHITE; t++) {
        if (waiting[t] == 0) {
            int jokers = waiting[WHITE] - 1;
            if (open_melds == 0 && odd <= jokers) return 1;
            return meld_groups_meld(&g, jokers);
        }
    }
    /* Every draw must win. */
    int jokers = waiting[WHITE], w[NTILES];
    memcpy(w, waiting, sizeof(w));
    for (int t = 0; t < NTILES; t++) {
        if (waiting[t] >= 4) continue;
        int ok;
        if (t == WHITE) {
            ok = (open_melds == 0 && odd <= jokers + 1) || meld_groups_hu(&g, jokers + 1);
        } else {
            int o = odd + (waiting[t] % 2 ? -1 : 1);
            if (open_melds == 0 && o <= jokers) {
                ok = 1;
            } else {
                meld_groups d = g;
                w[t]++;
                meld_groups_add(&d, w, t);
                w[t]--;
                ok = meld_groups_hu(&d, jokers);
            }
        }
        if (!ok) return 0;
    }
    return 1;
}

static baotou_slot race_baotou_cache[BAOTOU_CACHE_SIZE];

static int race_baotou_keyed(uint64_t k0, uint64_t k1, const int *counts, int open_melds) {
    uint64_t h = (k0 * 0xC2B2AE3D27D4EB4FULL) ^ (k1 * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 29;
    baotou_slot *slot = &race_baotou_cache[h >> (64 - BAOTOU_CACHE_BITS)];
    if (slot->k0 == k0 && (slot->k1 & (KEY1_BITS | KEY_USED)) == (k1 | KEY_USED))
        return (int)(slot->k1 >> VALUE_SHIFT);
    int value = race_baotou_shape(counts, open_melds);
    slot->k0 = k0;
    slot->k1 = k1 | KEY_USED | ((uint64_t)value << VALUE_SHIFT);
    return value;
}

static int race_baotou_cached(const int *counts, int open_melds) {
    uint64_t k0, k1;
    shanten_key(counts, open_melds, &k0, &k1);
    return race_baotou_keyed(k0, k1, counts, open_melds);
}

/* fan_value with the tables. */
static long race_fan_value(int *counts, int drawn_tile, int open_melds, int chain_count, int piao_count) {
    if (drawn_tile < 0 || drawn_tile >= NTILES || counts[drawn_tile] <= 0 || !race_hu(counts, open_melds))
        return 0;
    counts[drawn_tile]--;
    int bt = race_baotou_cached(counts, open_melds);
    counts[drawn_tile]++;
    long fan = 1;
    if (open_melds == 0) {
        int odd = 0, luxury = 0;
        for (int i = 0; i < WHITE; i++) {
            odd += counts[i] % 2;
            if (counts[i] == 4) luxury++;
        }
        if (odd <= counts[WHITE]) {
            if (counts[WHITE] == 4 && odd == 0) luxury++;
            fan = 1L << (1 + luxury);
        }
    }
    if (chain_count > 0) fan *= 1L << chain_count;
    if (counts[WHITE] + piao_count == 4) fan *= 2;
    if (bt) fan *= 2;
    return fan;
}

/* Bit d set when the waiting hand c plus tile, less one d (a kind c holds,
 * other than tile), is baotou: baotou_shape's test, answered for all d at
 * once where a non-white honor is absent from the result (the common case:
 * the natural tiles must meld with the whites but one as jokers, or form
 * pairs), and by race_baotou_keyed itself otherwise. What depends on the hand
 * alone is prepared once for all its draws. */
typedef struct {
    int nat[NTILES], kinds[NTILES], n_kinds, whites, odd, absent, opened;
    uint64_t k0, k1;  /* shanten_key(c, opened) */
    meld_groups g;    /* of nat */
    uint64_t kind_mask, odd_mask; /* the kinds c holds; the naturals held an odd number of times */
} upgrade_ctx;

static void upgrade_prepare(const int *c, int opened, uint64_t k0, uint64_t k1, upgrade_ctx *u) {
    memcpy(u->nat, c, sizeof(u->nat));
    u->nat[WHITE] = 0;
    u->whites = c[WHITE];
    u->odd = u->absent = u->n_kinds = 0;
    for (int i = 0; i < WHITE; i++) u->odd += u->nat[i] % 2;
    for (int h = 27; h < WHITE; h++) u->absent += c[h] == 0;
    u->kind_mask = u->odd_mask = 0;
    for (int d = 0; d < NTILES; d++) {
        if (c[d]) {
            u->kinds[u->n_kinds++] = d;
            u->kind_mask |= 1ULL << d;
        }
        if (d < WHITE && c[d] % 2) u->odd_mask |= 1ULL << d;
    }
    u->opened = opened;
    u->k0 = k0;
    u->k1 = k1;
    meld_groups_of(u->nat, &u->g);
}

static uint64_t upgrade_drops_slow(upgrade_ctx *u, int *c, int tile) {
    int opened = u->opened, whites = u->whites;
    uint64_t drops = 0, left = 0, d0, d1;
    int left_ready = 0;
    shanten_key_draw(u->k0, u->k1, tile, &d0, &d1);
    /* The naturals' groups with the drawn tile, when it is natural. */
    meld_groups g = u->g;
    if (tile != WHITE) {
        u->nat[tile]++;
        meld_groups_add(&g, u->nat, tile);
        u->nat[tile]--;
    }
    int tile_honor_new = tile >= 27 && tile < WHITE && c[tile] == 0;
    for (int k = 0; k < u->n_kinds; k++) {
        int d = u->kinds[k];
        if (d == tile) continue;
        int rest_whites = whites + (tile == WHITE) - (d == WHITE);
        if (rest_whites == 0) continue;
        int missing = u->absent - tile_honor_new + (d >= 27 && d < WHITE && c[d] == 1);
        int ok;
        if (missing == 0 || (opened == 2 && rest_whites == 4)) {
            c[tile]++;
            c[d]--;
            uint64_t r0 = d < 21 ? d0 - (1ULL << (3 * d)) : d0;
            uint64_t r1 = d < 21 ? d1 : d1 - (1ULL << (3 * (d - 21)));
            ok = race_baotou_keyed(r0, r1, c, opened);
            c[d]++;
            c[tile]--;
        } else {
            int pairs = 0;
            if (opened == 0) {
                int o = u->odd;
                if (tile != WHITE) o += u->nat[tile] % 2 ? -1 : 1;
                if (d != WHITE) o += u->nat[d] % 2 ? -1 : 1;
                pairs = o <= rest_whites - 1;
            }
            if (pairs) {
                ok = 1;
            } else if (d == WHITE) {
                /* The naturals with the drawn tile, one white fewer. */
                ok = meld_groups_meld(&g, whites - 2);
            } else {
                if (!left_ready) {
                    /* Draw a natural tile: its naturals, the whites but one as
                     * jokers. Draw a white: the naturals, every white a joker. */
                    int jokers = tile == WHITE ? whites : whites - 1;
                    if (tile != WHITE) u->nat[tile]++;
                    left = meld_groups_leftovers(&g, u->nat, jokers);
                    if (tile != WHITE) u->nat[tile]--;
                    left_ready = 1;
                }
                ok = (int)(left >> d & 1);
            }
        }
        if (ok) drops |= 1ULL << d;
    }
    return drops;
}

/* upgrade_drops_slow's answer where no drop can leave every other honor
 * present (nor four whites beside two open melds): then each d is decided by
 * the pairs or the melds alone, for all d at once as bit masks. */
static uint64_t upgrade_drops_of(upgrade_ctx *u, int *c, int tile) {
    int opened = u->opened, whites = u->whites;
    int tile_honor_new = tile >= 27 && tile < WHITE && c[tile] == 0;
    int draw_whites = whites + (tile == WHITE);
    if (u->absent - tile_honor_new == 0 || (opened == 2 && draw_whites == 4))
        return upgrade_drops_slow(u, c, tile);
    uint64_t drops = 0;
    int o = u->odd;
    if (tile != WHITE) o += u->nat[tile] % 2 ? -1 : 1;
    meld_groups g = u->g;
    if (tile != WHITE) {
        u->nat[tile]++;
        meld_groups_add(&g, u->nat, tile);
        u->nat[tile]--;
    }
    uint64_t naturals = u->kind_mask & ~(1ULL << tile) & ~(1ULL << WHITE);
    if (draw_whites >= 1 && naturals) {
        uint64_t pairs = 0;
        if (opened == 0) {
            /* Dropping d changes the odd count by one: down for an odd d. */
            if (o - 1 <= draw_whites - 1) pairs |= naturals & u->odd_mask;
            if (o + 1 <= draw_whites - 1) pairs |= naturals & ~u->odd_mask;
        }
        drops |= pairs;
        uint64_t rest = naturals & ~pairs;
        if (rest) {
            /* The whites but one as jokers (every white, when a white is drawn). */
            if (tile != WHITE) u->nat[tile]++;
            drops |= rest & meld_groups_leftovers(&g, u->nat, draw_whites - 1);
            if (tile != WHITE) u->nat[tile]--;
        }
    }
    if ((u->kind_mask >> WHITE & 1) && tile != WHITE && whites >= 2) {
        if ((opened == 0 && o <= whites - 2) || meld_groups_meld(&g, whites - 2)) drops |= 1ULL << WHITE;
    }
    return drops;
}

static uint64_t upgrade_drops(int *c, int opened, int tile, uint64_t k0, uint64_t k1) {
    upgrade_ctx u;
    upgrade_prepare(c, opened, k0, k1, &u);
    return upgrade_drops_of(&u, c, tile);
}

static PyObject *py_race_upgrade_drops(PyObject *self, PyObject *args) {
    PyObject *counts_obj;
    int opened, tile, c[NTILES];
    if (!PyArg_ParseTuple(args, "Oii", &counts_obj, &opened, &tile)) return NULL;
    if (parse_counts(counts_obj, c) < 0 || meld_tables() < 0) return NULL;
    uint64_t k0, k1;
    shanten_key(c, opened, &k0, &k1);
    return PyLong_FromUnsignedLongLong(upgrade_drops(c, opened, tile, k0, k1));
}

/* draw_distances for the race valuation, with the same values: a draw only
 * changes its own group (a suit, or the honors), so the other three groups
 * are combined once per hand and each draw costs one group lookup and one
 * combining step instead of shape_value's four of each. */
typedef struct { int g[5][2]; } race_groups;

static void race_groups_add(const race_groups *in, const suit_res *r, race_groups *out) {
    for (int m = 0; m < 5; m++)
        for (int p = 0; p < 2; p++) out->g[m][p] = -1;
    for (int m = 0; m < 5; m++) {
        for (int p = 0; p < 2; p++) {
            if (in->g[m][p] < 0) continue;
            for (int dm = 0; m + dm < 5; dm++) {
                for (int dp = 0; p + dp < 2; dp++) {
                    uint8_t v = r->t[dm][dp];
                    if (v == UNREACHABLE) continue;
                    int nv = in->g[m][p] + (int)v;
                    if (nv > 4) nv = 4;
                    if (out->g[m + dm][p + dp] < nv) out->g[m + dm][p + dp] = nv;
                }
            }
        }
    }
}

static int race_groups_value(const race_groups *g) {
    int best = 0x7FFFFF;
    for (int m = 0; m < 5; m++) {
        for (int p = 0; p < 2; p++) {
            if (g->g[m][p] < 0) continue;
            int capped = g->g[m][p] < 4 - m ? g->g[m][p] : 4 - m;
            if (capped < 0) capped = 0;
            int value = 8 - 2 * m - capped - p;
            if (value < best) best = value;
        }
    }
    return best;
}

static const suit_res *race_group_table(const int *natural, int s, suit_res *buf) {
    *buf = suit_table(natural + (s < 3 ? s * 9 : 27), s < 3 ? 9 : 7, s < 3);
    return buf;
}

/* shanten_uncached from the naturals' shape value: less the jokers, or seven
 * pairs when nothing is open. */
static inline int race_distance(int shape, int jokers, int pairs, int singles, int open_melds) {
    int normal = shape - jokers;
    if (open_melds == 0) {
        int room = 7 - pairs;
        int capped = singles < room ? singles : room;
        int seven = 14 - (2 * pairs + capped) - jokers - 1;
        if (seven < normal) normal = seven;
    }
    return normal;
}

#define RACE_DRAW_BITS 17

static draw_slot race_draw_cache[1 << RACE_DRAW_BITS];

static const draw_slot *race_draws(const int *counts, int open_melds) {
    uint64_t k0, k1;
    shanten_key(counts, open_melds, &k0, &k1);
    uint64_t h = (k0 * 0x9E3779B97F4A7C15ULL) ^ (k1 * 0xD6E8FEB86659FD93ULL);
    h ^= h >> 31;
    draw_slot *slot = &race_draw_cache[h >> (64 - RACE_DRAW_BITS)];
    if (slot->k0 == k0 && slot->k1 == (k1 | KEY_USED)) return slot;
    int natural[NTILES];
    memcpy(natural, counts, sizeof(natural));
    int jokers = natural[WHITE];
    natural[WHITE] = 0;
    suit_res tables[4], drawn;
    for (int s = 0; s < 4; s++) race_group_table(natural, s, &tables[s]);
    /* others[s]: every group but s, from the open melds. */
    race_groups start, others[4], tmp, full;
    for (int m = 0; m < 5; m++)
        for (int p = 0; p < 2; p++) start.g[m][p] = -1;
    start.g[open_melds][0] = 0;
    for (int s = 0; s < 4; s++) {
        race_groups acc = start;
        for (int o = 0; o < 4; o++) {
            if (o == s) continue;
            race_groups_add(&acc, &tables[o], &tmp);
            acc = tmp;
        }
        others[s] = acc;
    }
    race_groups_add(&others[3], &tables[3], &full);
    int shape = race_groups_value(&full);
    int pairs = 0, singles = 0;
    if (open_melds == 0) {
        for (int i = 0; i < NTILES; i++) {
            pairs += natural[i] / 2;
            singles += natural[i] % 2;
        }
    }
    slot->k0 = k0;
    slot->k1 = k1 | KEY_USED;
    slot->distance = (int8_t)race_distance(shape, jokers, pairs, singles, open_melds);
    for (int t = 0; t < NTILES; t++) {
        if (counts[t] >= 4) {
            slot->draw[t] = NO_DRAW;
            continue;
        }
        if (t == WHITE) {
            slot->draw[t] = (int8_t)race_distance(shape, jokers + 1, pairs, singles, open_melds);
            continue;
        }
        int s = t < 27 ? t / 9 : 3;
        natural[t]++;
        race_group_table(natural, s, &drawn);
        natural[t]--;
        race_groups_add(&others[s], &drawn, &tmp);
        int odd = natural[t] % 2;
        slot->draw[t] = (int8_t)race_distance(race_groups_value(&tmp), jokers, pairs + odd, singles + (odd ? -1 : 1), open_melds);
    }
    return slot;
}

/* analyze_values over race_draws. */
static void race_analyze_values(const int *counts, const int *avail, int open_melds,
                           int chain_count, int piao_count, analysis_row *out) {
    int drawn[NTILES];
    const draw_slot *table = race_draws(counts, open_melds);
    int distance = table->distance;
    int improving[NTILES], wins[NTILES];
    long fans[NTILES];
    int n_improving = 0, n_wins = 0;
    for (int t = 0; t < NTILES; t++) {
        if (avail[t] <= 0 || counts[t] >= 4) continue;
        int d = table->draw[t];
        if (d < distance) improving[n_improving++] = t;
        if (d < 0) {
            memcpy(drawn, counts, sizeof(drawn));
            drawn[t]++;
            long fan = race_fan_value(drawn, t, open_melds, chain_count, piao_count);
            if (fan) {
                wins[n_wins] = t;
                fans[n_wins++] = fan;
            }
        }
    }
    /* The availability-weighted sums sdk.shape.analyze reports, so the
     * Python side builds its result without iterating the tuples again. */
    long ukeire = 0, copies = 0, value = 0;
    for (int i = 0; i < n_improving; i++) ukeire += avail[improving[i]];
    for (int i = 0; i < n_wins; i++) {
        copies += avail[wins[i]];
        value += avail[wins[i]] * fans[i];
    }
    out->distance = distance;
    out->n_improving = n_improving;
    out->n_wins = n_wins;
    out->ukeire = ukeire;
    out->copies = copies;
    out->value = value;
    memcpy(out->improving, improving, n_improving * sizeof(int));
    memcpy(out->wins, wins, n_wins * sizeof(int));
    memcpy(out->fans, fans, n_wins * sizeof(long));
}

/* The memo key: 3 bits per count and per available copy, and the rest.
 * kind: 0 waiting, 1 reshaped, 2 reshape gain. */
static int race_key(const race_ctx *x, const int *c, int kind, int chain, int piao, int draws, int options,
                    uint64_t *k) {
    if (!x->keyed || chain < 0 || chain > 255 || piao < 0 || piao > 255 || draws < -64 || draws > 63) return 0;
    uint64_t k0 = 0, k1 = 0;
    for (int t = 0; t < NTILES; t++) {
        if (c[t] > 7) return 0;
        if (t < 21) k0 |= (uint64_t)c[t] << (3 * t);
        else k1 |= (uint64_t)c[t] << (3 * (t - 21));
    }
    k1 |= (uint64_t)x->opened << 39 | (uint64_t)(kind & 1) << 42 | (uint64_t)(draws + 64) << 43
        | (uint64_t)x->profile << 50 | 1ULL << 63;
    k[0] = k0;
    k[1] = k1;
    k[2] = x->a0;
    k[3] = x->a1 | (uint64_t)chain << 39 | (uint64_t)piao << 47 | (uint64_t)x->speed_key << 55
         | (uint64_t)(options != 0) << 61 | (uint64_t)(kind >> 1) << 62;
    return 1;
}

static race_slot *race_slot_of(const uint64_t *k) {
    uint64_t h = (k[0] * 0x9E3779B97F4A7C15ULL) ^ (k[1] * 0xC2B2AE3D27D4EB4FULL)
               ^ (k[2] * 0xD6E8FEB86659FD93ULL) ^ (k[3] * 0x94D049BB133111EBULL);
    h ^= h >> 31;
    return &race_cache[h >> (64 - RACE_CACHE_BITS)];
}

/* sdk.shape.analyze for a hand the extension counts as complete: the caller's function. */
static int race_fallback(race_ctx *x, const int *c, int opened, int chain, int piao, analysis_row *row) {
    int size = 0;
    for (int t = 0; t < NTILES; t++) size += c[t];
    PyObject *hand = PyTuple_New(size);
    if (!hand) return -1;
    int n = 0;
    for (int t = 0; t < NTILES; t++)
        for (int i = 0; i < c[t]; i++) PyTuple_SET_ITEM(hand, n++, PyLong_FromLong(t));
    PyObject *result = PyObject_CallFunction(x->fallback, "OiOii", hand, opened, x->available, chain, piao);
    Py_DECREF(hand);
    if (!result) return -1;
    int ok = 0;
    PyObject *shanten = PyMapping_GetItemString(result, "shanten");
    PyObject *improving = PyMapping_GetItemString(result, "improving");
    PyObject *ukeire = PyMapping_GetItemString(result, "ukeire");
    PyObject *wins = PyMapping_GetItemString(result, "winning_tiles");
    PyObject *copies = PyMapping_GetItemString(result, "winning_copies");
    PyObject *fans = PyMapping_GetItemString(result, "fans");
    PyObject *value = PyMapping_GetItemString(result, "fan_value");
    if (shanten && improving && ukeire && wins && copies && fans && value) {
        row->distance = (int)PyLong_AsLong(shanten);
        row->ukeire = PyLong_AsLong(ukeire);
        row->copies = PyLong_AsLong(copies);
        row->value = PyLong_AsLong(value);
        Py_ssize_t ni = PySequence_Size(improving), nw = PySequence_Size(wins);
        if (ni >= 0 && ni <= NTILES && nw >= 0 && nw <= NTILES) {
            row->n_improving = (int)ni;
            row->n_wins = (int)nw;
            for (Py_ssize_t i = 0; i < ni; i++) {
                PyObject *item = PySequence_GetItem(improving, i);
                row->improving[i] = item ? (int)PyLong_AsLong(item) : -1;
                Py_XDECREF(item);
            }
            for (Py_ssize_t i = 0; i < nw; i++) {
                PyObject *tile = PySequence_GetItem(wins, i);
                PyObject *fan = tile ? PyObject_GetItem(fans, tile) : NULL;
                row->wins[i] = tile ? (int)PyLong_AsLong(tile) : -1;
                row->fans[i] = fan ? PyLong_AsLong(fan) : 0;
                Py_XDECREF(tile);
                Py_XDECREF(fan);
            }
            ok = !PyErr_Occurred();
        }
    }
    Py_XDECREF(shanten); Py_XDECREF(improving); Py_XDECREF(ukeire); Py_XDECREF(wins);
    Py_XDECREF(copies); Py_XDECREF(fans); Py_XDECREF(value);
    Py_DECREF(result);
    if (!ok && !PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "fallback analysis is incomplete");
    return ok ? 0 : -1;
}

static int race_analysis_at(race_ctx *x, const int *c, int opened, int chain, int piao, analysis_row *row) {
    race_analyze_values(c, x->avail, opened, chain, piao, row);
    return row->distance < 0 ? race_fallback(x, c, opened, chain, piao, row) : 0;
}

static int race_analysis(race_ctx *x, const int *c, int chain, int piao, analysis_row *row) {
    return race_analysis_at(x, c, x->opened, chain, piao, row);
}

/* kong_draw_value: after a concealed kong of this tile (three held, the
 * fourth drawn), the replacement draw wins at double fan (chain 1). */
static int race_kong_draw(race_ctx *x, int *c, int tile, double *out) {
    analysis_row row;
    c[tile] -= 3;
    int rc = race_analysis_at(x, c, x->opened + 1, 1, 0, &row);
    c[tile] += 3;
    if (rc < 0) return -1;
    *out = row.distance > 0 ? 0.0 : ((double)row.value + x->p->saved * (double)row.copies) / x->unseen;
    return 0;
}

static int race_waiting(race_ctx *x, int *c, int chain, int piao, int draws, int options, double *out);

/* reshape_gain(hand, draws): the trade option of reshaped_value for a wait
 * one step ahead, pricing every wait there as a plain race. */
static int race_reshape_gain(race_ctx *x, int *c, int draws, double *out) {
    if (draws < 2) {
        *out = 0.0;
        return 0;
    }
    uint64_t key[4];
    int keyed = race_key(x, c, 2, 0, 0, draws, 0, key);
    race_slot *slot = keyed ? race_slot_of(key) : NULL;
    if (slot && !memcmp(slot->k, key, sizeof(key))) {
        *out = slot->value;
        return 0;
    }
    const race_profile *p = x->p;
    const int *avail = x->avail;
    int opened = x->opened, kinds[NTILES], n_kinds = 0;
    double unseen = x->unseen, q = p->q * x->speed;
    analysis_row row, r;
    if (race_analysis(x, c, 0, 0, &row) < 0) return -1;
    int wins[NTILES] = {0};
    for (int i = 0; i < row.n_wins; i++) wins[row.wins[i]] = 1;
    for (int t = 0; t < NTILES; t++)
        if (c[t]) kinds[n_kinds++] = t;
    double stay = race_race(x, (double)row.copies / unseen, ((double)row.value + p->saved * (double)row.copies) / unseen,
                            draws - 1);
    uint64_t k0, k1;
    shanten_key(c, opened, &k0, &k1);
    double gain = 0.0;
    long reach = 0;
    for (int tile = 0; tile < NTILES; tile++) {
        if (wins[tile] || avail[tile] <= 0 || c[tile] >= 4) continue;
        uint64_t d0, d1;
        shanten_key_draw(k0, k1, tile, &d0, &d1);
        c[tile]++;
        double best = stay;
        for (int k = 0; k < n_kinds; k++) {
            int drop = kinds[k];
            if (drop == tile) continue;
            c[drop]--;
            uint64_t r0 = drop < 21 ? d0 - (1ULL << (3 * drop)) : d0;
            uint64_t r1 = drop < 21 ? d1 : d1 - (1ULL << (3 * (drop - 21)));
            if (shanten_keyed(r0, r1, c, opened) <= 0) {
                /* The C analysis as it is, even past the winning boundary. */
                race_analyze_values(c, avail, opened, 0, 0, &r);
                double v = race_race(x, (double)r.copies / unseen,
                                     ((double)r.value + p->saved * (double)r.copies) / unseen, draws - 1);
                if (v > best) best = v;
            }
            c[drop]++;
        }
        c[tile]--;
        if (best > stay) {
            gain += (double)avail[tile] * (best - stay);
            reach += avail[tile];
        }
    }
    double survive = (1 - q) * (1 - (double)(row.copies + reach) / unseen);
    double result = (1 - q) * gain / unseen * race_repeat_x(x, survive, draws - 1);
    if (slot) {
        memcpy(slot->k, key, sizeof(key));
        slot->value = result;
    }
    *out = result;
    return 0;
}

/* waiting_value(hand, opened, available, chain, piao, draws, options) at the context's speed */
static int race_waiting(race_ctx *x, int *c, int chain, int piao, int draws, int options, double *out) {
    uint64_t key[4];
    int keyed = race_key(x, c, 0, chain, piao, draws, options, key);
    race_slot *slot = keyed ? race_slot_of(key) : NULL;
    if (slot && !memcmp(slot->k, key, sizeof(key))) {
        *out = slot->value;
        return 0;
    }
    const race_profile *p = x->p;
    const int *avail = x->avail;
    int opened = x->opened;
    double unseen = x->unseen, result, q = p->q * x->speed;
    uint64_t k0, k1;
    shanten_key(c, opened, &k0, &k1);
    analysis_row row;
    if (c[WHITE] && race_baotou_keyed(k0, k1, c, opened)) {
        /* A baotou wait wins on every drawable tile: the Python loop's copies
         * and fan-weighted copies over fan_value. */
        long copies, fan;
        baotou_fans(c, avail, opened, chain, piao, &copies, &fan);
        result = race_race(x, (double)copies / unseen, ((double)fan + p->saved * (double)copies) / unseen, draws);
    } else {
        if (race_analysis(x, c, chain, piao, &row) < 0) return -1;
        if (row.distance <= 0) {
            long copies = row.copies;
            double gain = (double)row.value + p->saved * (double)copies;
            if (!c[WHITE] && p->complete) {
                /* Drawing a white wins, or a discard may turn the wait into
                 * baotou; drawing the last copy of a concealed triplet gives
                 * a kong and at once an extra draw, which wins at double fan. */
                int has_fan[NTILES] = {0};
                long fan_white = 0;
                for (int i = 0; i < row.n_wins; i++) {
                    has_fan[row.wins[i]] = 1;
                    if (row.wins[i] == WHITE) fan_white = row.fans[i];
                }
                double extra = 0.0;
                if (avail[WHITE] > 0 && has_fan[WHITE]) {
                    upgrade_ctx up;
                    upgrade_prepare(c, opened, k0, k1, &up);
                    uint64_t drops = upgrade_drops_of(&up, c, WHITE);
                    double best = 0.0;
                    int any = 0;
                    c[WHITE]++;
                    for (int drop = 0; drops; drop++, drops >>= 1) {
                        if (!(drops & 1)) continue;
                        c[drop]--;
                        double value;
                        int rc = race_waiting(x, c, 0, 0, draws - 1, 0, &value);
                        c[drop]++;
                        if (rc < 0) {
                            c[WHITE]--;
                            return -1;
                        }
                        if (!any || value > best) best = value;
                        any = 1;
                    }
                    c[WHITE]--;
                    double v = (1 - q) * best - (double)fan_white - p->saved;
                    extra += (double)avail[WHITE] * (v > 0.0 ? v : 0.0);
                }
                int held[NTILES], order[NTILES], n = 0;
                for (int t = 0; t < NTILES; t++)
                    if (c[t]) held[n++] = t;
                n = py_set_order(held, n, order);
                for (int i = 0; i < n; i++) {
                    int tile = order[i];
                    if (c[tile] == 3 && avail[tile] > 0 && !has_fan[tile]) {
                        double kong;
                        if (race_kong_draw(x, c, tile, &kong) < 0) return -1;
                        extra += (double)avail[tile] * kong;
                    }
                }
                result = race_race(x, (double)copies / unseen, (gain + extra) / unseen, draws);
            } else if (!c[WHITE] || !p->upgrade) {
                result = race_race(x, (double)copies / unseen, gain / unseen, draws);
            } else {
                /* A draw that frees a white lets a discard turn the wait into baotou. */
                long fan_of[NTILES];
                int has_fan[NTILES] = {0};
                for (int i = 0; i < row.n_wins; i++) {
                    has_fan[row.wins[i]] = 1;
                    fan_of[row.wins[i]] = row.fans[i];
                }
                double total = 0.0;
                long reach = 0;
                upgrade_ctx up;
                upgrade_prepare(c, opened, k0, k1, &up);
                for (int tile = 0; tile < NTILES; tile++) {
                    if (avail[tile] <= 0 || c[tile] >= 4) continue;
                    double best = 0.0;
                    uint64_t drops = upgrade_drops_of(&up, c, tile);
                    c[tile]++;
                    for (int drop = 0; drops; drop++, drops >>= 1) {
                        if (!(drops & 1)) continue;
                        c[drop]--;
                        double value;
                        if (race_waiting(x, c, 0, 0, draws - 1, 0, &value) < 0) {
                            c[drop]++;
                            c[tile]--;
                            return -1;
                        }
                        if (value > best) best = value;
                        c[drop]++;
                    }
                    c[tile]--;
                    if (has_fan[tile]) {
                        double win = (double)fan_of[tile] + p->saved, hunt = (1 - q) * best * p->hunt;
                        total += (double)avail[tile] * (hunt > win ? hunt : win);
                    } else if (best != 0.0) {
                        total += (double)avail[tile] * (1 - q) * best * p->hunt;
                        reach += avail[tile];
                    }
                }
                double stay = (1 - race_hazard_x(x, draws)) * (1 - (double)(copies + reach) / unseen);
                result = total / unseen * race_repeat_x(x, stay, draws);
            }
        } else {
            double u = (double)row.ukeire / unseen;
            if (row.distance == 1) {
                double weighted = 0.0;
                for (int i = 0; i < row.n_improving; i++) {
                    int tile = row.improving[i];
                    c[tile]++;
                    double best = 0.0;
                    int target = -1; /* the discard leaving the best wait */
                    if (draws > 1) {
                        uint64_t d0, d1;
                        shanten_key_draw(k0, k1, tile, &d0, &d1);
                        for (int drop = 0; drop < NTILES; drop++) {
                            if (!c[drop]) continue;
                            c[drop]--;
                            uint64_t r0 = drop < 21 ? d0 - (1ULL << (3 * drop)) : d0;
                            uint64_t r1 = drop < 21 ? d1 : d1 - (1ULL << (3 * (drop - 21)));
                            if (shanten_keyed(r0, r1, c, opened) <= 0) {
                                double value;
                                if (race_waiting(x, c, 0, 0, draws - 1, 0, &value) < 0) {
                                    c[drop]++;
                                    c[tile]--;
                                    return -1;
                                }
                                if (value > best) {
                                    best = value;
                                    target = drop;
                                }
                            }
                            c[drop]++;
                        }
                    }
                    if (options && target >= 0 && chain == 0) {
                        double g;
                        c[target]--;
                        int rc = race_reshape_gain(x, c, draws - 1, &g);
                        c[target]++;
                        if (rc < 0) {
                            c[tile]--;
                            return -1;
                        }
                        best += g;
                    }
                    c[tile]--;
                    weighted += (double)avail[tile] * best;
                }
                double stay = (1 - u) * (1 - q);
                result = (1 - q) * (weighted / unseen) * race_repeat_x(x, stay, draws - 1);
            } else {
                double step = (1 - q) * u * race_repeat_x(x, (1 - u) * (1 - q), draws - row.distance);
                result = step * p->one_step * pow(p->far_step, (double)(row.distance - 2));
            }
        }
    }
    if (slot) {
        memcpy(slot->k, key, sizeof(key));
        slot->value = result;
    }
    *out = result;
    return 0;
}

/* reshaped_value(hand, opened, available, draws): a waiting hand with the
 * option to trade a non-winning draw for a better wait. */
static int race_reshaped(race_ctx *x, int *c, int draws, int options, double *out) {
    uint64_t key[4];
    int keyed = race_key(x, c, 1, 0, 0, draws, options, key);
    race_slot *slot = keyed ? race_slot_of(key) : NULL;
    if (slot && !memcmp(slot->k, key, sizeof(key))) {
        *out = slot->value;
        return 0;
    }
    const race_profile *p = x->p;
    const int *avail = x->avail;
    int opened = x->opened;
    double unseen = x->unseen, value, result, q = p->q * x->speed;
    uint64_t k0, k1;
    shanten_key(c, opened, &k0, &k1);
    /* sdk.shape.shanten is positive exactly when the raw distance is, and
     * then equal to it: one step away, the trade option there is wanted. */
    int distance = shanten_keyed(k0, k1, c, opened);
    if (distance > 0) {
        if (race_waiting(x, c, 0, 0, draws, options && distance == 1, &result) < 0) return -1;
    } else if (race_waiting(x, c, 0, 0, draws, 0, &value) < 0) {
        return -1;
    } else if (draws < 2) {
        result = value;
    } else {
        analysis_row row;
        if (race_analysis(x, c, 0, 0, &row) < 0) return -1;
        int wins[NTILES] = {0};
        for (int i = 0; i < row.n_wins; i++) wins[row.wins[i]] = 1;
        double stay;
        if (race_waiting(x, c, 0, 0, draws - 1, 0, &stay) < 0) return -1;
        double gain = 0.0;
        long reach = 0;
        for (int tile = 0; tile < NTILES; tile++) {
            if (wins[tile] || avail[tile] <= 0 || c[tile] >= 4) continue;
            uint64_t d0, d1;
            shanten_key_draw(k0, k1, tile, &d0, &d1);
            c[tile]++;
            double best = stay;
            for (int drop = 0; drop < NTILES; drop++) {
                if (drop == tile || !c[drop]) continue;
                c[drop]--;
                uint64_t r0 = drop < 21 ? d0 - (1ULL << (3 * drop)) : d0;
                uint64_t r1 = drop < 21 ? d1 : d1 - (1ULL << (3 * (drop - 21)));
                if (shanten_keyed(r0, r1, c, opened) <= 0) {
                    double v;
                    if (race_waiting(x, c, 0, 0, draws - 1, 0, &v) < 0) {
                        c[drop]++;
                        c[tile]--;
                        return -1;
                    }
                    if (v > best) best = v;
                }
                c[drop]++;
            }
            c[tile]--;
            if (best > stay) {
                gain += (double)avail[tile] * (best - stay);
                reach += avail[tile];
            }
        }
        double survive = (1 - q) * (1 - (double)(row.copies + reach) / unseen);
        result = value + (1 - q) * gain / unseen * race_repeat_x(x, survive, draws - 1);
    }
    if (slot) {
        memcpy(slot->k, key, sizeof(key));
        slot->value = result;
    }
    *out = result;
    return 0;
}

/* The availability (and its memo key part), the profile and the fallback of a context. */
static int race_bind(race_ctx *x, PyObject *available, int profile, PyObject *fallback) {
    if (meld_tables() < 0) return -1;
    if (parse_vector(available, x->avail, 1000000, "available") < 0) return -1;
    long unseen = 0;
    x->keyed = 1;
    x->a0 = x->a1 = 0;
    for (int t = 0; t < NTILES; t++) {
        unseen += x->avail[t];
        if (x->avail[t] < 0 || x->avail[t] > 7) x->keyed = 0;
        else if (t < 21) x->a0 |= (uint64_t)x->avail[t] << (3 * t);
        else x->a1 |= (uint64_t)x->avail[t] << (3 * (t - 21));
    }
    x->unseen = unseen < 1 ? 1 : (int)unseen;
    x->p = &race_profiles[profile];
    x->speed = 1.0;
    x->speed_key = 20;
    x->profile = profile;
    x->available = available;
    x->fallback = fallback;
    return 0;
}

/* The opponents' speed of a context: its memo key when speed is n / 20 for
 * some n in 1..63 (as a strategy's rounding to steps of 0.05 gives), else
 * the context keeps no memo. */
static void race_speed(race_ctx *x, double speed) {
    double n = floor(speed * 20 + 0.5);
    x->speed = speed;
    if (n >= 1 && n <= 63 && n / 20 == speed) x->speed_key = (int)n;
    else x->keyed = 0;
}

/* Parse (hand, opened, available, profile, fallback) into a context and the hand's counts. */
static int race_setup(race_ctx *x, int *c, PyObject *hand, int opened, PyObject *available,
                      int profile, PyObject *fallback) {
    if (profile < 0 || profile >= race_profile_count) {
        PyErr_SetString(PyExc_ValueError, "unknown race profile");
        return -1;
    }
    if (opened < 0 || opened > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return -1;
    }
    PyObject *seq = PySequence_Fast(hand, "hand must be a sequence");
    if (!seq) return -1;
    memset(c, 0, NTILES * sizeof(int));
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        long tile = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (tile < 0 || tile >= NTILES || ++c[tile] > 4) {
            Py_DECREF(seq);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "invalid hand");
            return -1;
        }
    }
    Py_DECREF(seq);
    if (size != 13 - 3 * opened) {
        PyErr_SetString(PyExc_ValueError, "race valuation expects a waiting hand");
        return -1;
    }
    if (race_bind(x, available, profile, fallback) < 0) return -1;
    x->opened = opened;
    x->size = (int)size;
    return 0;
}

/* potential(hand, available): live tiles that pair with or sit within two of
 * each kept tile (an honor counts its own copies). */
static PyObject *py_race_potential(PyObject *self, PyObject *args) {
    PyObject *hand, *available;
    int avail[NTILES];
    if (!PyArg_ParseTuple(args, "OO", &hand, &available)) return NULL;
    if (parse_vector(available, avail, 1000000, "available") < 0) return NULL;
    PyObject *seq = PySequence_Fast(hand, "hand must be a sequence");
    if (!seq) return NULL;
    long total = 0;
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        long tile = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (tile < 0 || tile >= NTILES) {
            Py_DECREF(seq);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "invalid tile");
            return NULL;
        }
        if (tile >= 27) {
            total += avail[tile];
            continue;
        }
        long low = tile - tile % 9, high = low + 8;
        for (long t = tile - 2 > low ? tile - 2 : low; t <= (tile + 2 < high ? tile + 2 : high); t++) total += avail[t];
    }
    Py_DECREF(seq);
    return PyLong_FromLong(total);
}

static PyObject *py_race_profile(PyObject *self, PyObject *args) {
    race_profile p;
    memset(&p, 0, sizeof(p));  /* padding too: profiles are compared with memcmp */
    if (!PyArg_ParseTuple(args, "dddddddpddii|pp", &p.q, &p.saved, &p.one_step, &p.far_step,
                          &p.q_early, &p.q_late, &p.hunt, &p.upgrade, &p.draw_risk, &p.tie,
                          &p.last_wall, &p.style, &p.complete, &p.guard)) return NULL;
    if (p.style < 0 || p.style > 5) {
        PyErr_SetString(PyExc_ValueError, "unknown race style");
        return NULL;
    }
    for (int i = 0; i < race_profile_count; i++)
        if (!memcmp(&race_profiles[i], &p, sizeof(p))) return PyLong_FromLong(i);
    if (race_profile_count >= RACE_PROFILES) {
        PyErr_SetString(PyExc_RuntimeError, "too many race profiles");
        return NULL;
    }
    race_profiles[race_profile_count] = p;
    return PyLong_FromLong(race_profile_count++);
}

static PyObject *py_race_waiting(PyObject *self, PyObject *args) {
    PyObject *hand, *available, *fallback;
    int opened, chain, piao, draws, profile, c[NTILES];
    race_ctx x;
    memset(&x, 0, sizeof(x));
    int options = 0;
    double speed = 1.0;
    if (!PyArg_ParseTuple(args, "OiOiiiiO|pd", &hand, &opened, &available, &chain, &piao, &draws,
                          &profile, &fallback, &options, &speed)) return NULL;
    if (race_setup(&x, c, hand, opened, available, profile, fallback) < 0) return NULL;
    race_speed(&x, speed);
    double value;
    if (race_waiting(&x, c, chain, piao, draws, options, &value) < 0) return NULL;
    return PyFloat_FromDouble(value);
}

static PyObject *py_race_reshaped(PyObject *self, PyObject *args) {
    PyObject *hand, *available, *fallback;
    int opened, draws, profile, c[NTILES];
    race_ctx x;
    memset(&x, 0, sizeof(x));
    int options = 0;
    double speed = 1.0;
    if (!PyArg_ParseTuple(args, "OiOiiO|pd", &hand, &opened, &available, &draws, &profile, &fallback,
                          &options, &speed))
        return NULL;
    if (race_setup(&x, c, hand, opened, available, profile, fallback) < 0) return NULL;
    race_speed(&x, speed);
    double value;
    if (race_reshaped(&x, c, draws, options, &value) < 0) return NULL;
    return PyFloat_FromDouble(value);
}


/* ---------- the whole decision of a race strategy ---------- */

/* Race.act of the fixed pool's members and its style variants, over the
 * Observation's fields: every value the same arithmetic as the Python, the
 * same first-maximum choice. Styles: 0 Race, 1 discards valued without the
 * trade option (race-classic), 2 kong whenever not waiting or the kong keeps
 * the wait (kong-eager), 3 and 4 call a peng / chi that brings the hand
 * closer or widens it (peng-speed, chi-speed), 5 no calls three or more
 * steps away (closed-early). */

enum { ACT_HU, ACT_PASS, ACT_DISCARD, ACT_PENG, ACT_CHI, ACT_GANG, ACT_OTHER };

typedef struct {
    int kind, tile, n_tiles, tiles[4], kong; /* kong: tiles a kong takes from the hand */
    int added;
} race_action;

typedef struct {
    PyObject *obs;
    race_ctx x;           /* availability, profile, fallback; opened and size set per use */
    int c[NTILES], size, opened, seat, chain, piao, hu_fan, draws, piao_known, piao_ok;
    int n_legal;
    race_action legal[64];
} race_act_ctx;

static int attr_int(PyObject *obj, const char *name, long *out) {
    PyObject *v = PyObject_GetAttrString(obj, name);
    if (!v) return -1;
    if (v == Py_None) {
        *out = -1;
    } else {
        *out = PyLong_AsLong(v);
    }
    Py_DECREF(v);
    return PyErr_Occurred() ? -1 : 0;
}

static int counts_of(PyObject *seq_obj, int *c, int *size) {
    PyObject *seq = PySequence_Fast(seq_obj, "expected tiles");
    if (!seq) return -1;
    memset(c, 0, NTILES * sizeof(int));
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; i++) {
        long t = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (t < 0 || t >= NTILES || ++c[t] > 4) {
            Py_DECREF(seq);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "invalid tiles");
            return -1;
        }
    }
    Py_DECREF(seq);
    if (size) *size = (int)n;
    return 0;
}

static int parse_action_tiles(PyObject *a, race_action *out) {
    long tile;
    if (attr_int(a, "tile", &tile) < 0) return -1;
    out->tile = (int)tile;
    PyObject *tiles = PyObject_GetAttrString(a, "tiles");
    if (!tiles) return -1;
    PyObject *seq = PySequence_Fast(tiles, "expected tiles");
    Py_DECREF(tiles);
    if (!seq) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    out->n_tiles = n < 4 ? (int)n : 4;
    for (int i = 0; i < out->n_tiles; i++) out->tiles[i] = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
    if (n > 4) out->n_tiles = (int)n;  /* only the count matters beyond four */
    Py_DECREF(seq);
    return PyErr_Occurred() ? -1 : 0;
}

static int parse_action(PyObject *a, race_action *out) {
    memset(out, 0, sizeof(*out));
    PyObject *name = PyObject_GetAttrString(a, "action");
    if (!name) return -1;
    const char *names[6] = {"hu", "pass", "discard", "peng", "chi", "gang"};
    out->kind = ACT_OTHER;
    for (int i = 0; i < 6; i++)
        if (PyUnicode_CompareWithASCIIString(name, names[i]) == 0) out->kind = i;
    Py_DECREF(name);
    if (parse_action_tiles(a, out) < 0) return -1;
    if (out->kind == ACT_GANG) {
        PyObject *kind = PyObject_GetAttrString(a, "kind");
        if (!kind) return -1;
        if (PyUnicode_CompareWithASCIIString(kind, "concealed") == 0) out->kong = 4;
        else if (PyUnicode_CompareWithASCIIString(kind, "exposed") == 0) out->kong = 3;
        else if (PyUnicode_CompareWithASCIIString(kind, "added") == 0) { out->kong = 1; out->added = 1; }
        Py_DECREF(kind);
        if (!out->kong) {
            PyErr_SetString(PyExc_KeyError, "kong kind");
            return -1;
        }
    }
    return PyErr_Occurred() ? -1 : 0;
}

/* sdk.shape.shanten: the raw distance, or -1 / 0 past the winning boundary. */
static int sdk_shanten(const int *c, int opened) {
    int raw = shanten_raw(c, opened);
    if (raw >= 0) return raw;
    return race_hu(c, opened) ? -1 : 0;
}

/* rules.is_baotou for counts of a hand of `size` tiles. */
static int is_baotou_counts(const int *c, int size, int opened) {
    if (size != 13 - 3 * opened || !c[WHITE]) return 0;
    return race_baotou_cached(c, opened);
}

static void ctx_at(race_act_ctx *A, int opened) {
    A->x.opened = opened;
    A->x.size = 13 - 3 * opened;
}

static int act_waiting(race_act_ctx *A, int *c, int opened, int chain, int piao, int draws, double *out) {
    ctx_at(A, opened);
    return race_waiting(&A->x, c, chain, piao, draws, 0, out);
}

static long potential_counts(const int *c, const int *avail) {
    long total = 0;
    for (int tile = 0; tile < NTILES; tile++) {
        for (int k = 0; k < c[tile]; k++) {
            if (tile >= 27) {
                total += avail[tile];
                continue;
            }
            int low = tile - tile % 9, high = low + 8;
            for (int t = tile - 2 > low ? tile - 2 : low; t <= (tile + 2 < high ? tile + 2 : high); t++) total += avail[t];
        }
    }
    return total;
}

/* Race.piao(obs, opened): domain.observations.can_discard_piao of the
 * observation, whether a white discard now scores piao. */
static int act_piao(race_act_ctx *A) {
    if (A->piao_known) return A->piao_ok;
    int *c = A->c, ok = 0;
    long drawn, god, catch_play, hand_no;
    PyObject *events = NULL, *seq = NULL, *stage = NULL;
    if (!c[WHITE]) goto known;
    /* Without a white the rest must be baotou. */
    c[WHITE]--;
    ok = is_baotou_counts(c, A->size - 1, A->opened);
    c[WHITE]++;
    if (!ok) goto known;
    if (attr_int(A->obs, "drawn_tile", &drawn) < 0) return -1;
    if (drawn >= 0) {
        /* After a draw: the waiting hand before it must have been baotou. */
        if (drawn >= NTILES || !c[drawn]) {
            PyErr_SetString(PyExc_ValueError, "drawn tile not in hand");
            return -1;
        }
        c[drawn]--;
        ok = is_baotou_counts(c, A->size - 1, A->opened);
        c[drawn]++;
        goto known;
    }
    if (attr_int(A->obs, "catch_play", &catch_play) < 0 || attr_int(A->obs, "god_discarder_seat", &god) < 0) return -1;
    /* catch_exempt = catch_play and not catch_restricted. */
    if (catch_play && god == A->seat) {
        ok = 1;
        goto known;
    }
    /* After a call: the waiting hand before it, from my last event of this hand. */
    ok = 0;
    if (!(stage = PyObject_GetAttrString(A->obs, "stage"))) return -1;
    int bad = attr_int(stage, "round_no", &hand_no) < 0;
    Py_DECREF(stage);
    if (bad) return -1;
    if (!(events = PyObject_GetAttrString(A->obs, "public_events"))) return -1;
    seq = PySequence_Fast(events, "expected events");
    Py_DECREF(events);
    if (!seq) return -1;
    for (Py_ssize_t i = PySequence_Fast_GET_SIZE(seq) - 1; i >= 0; i--) {
        PyObject *e = PySequence_Fast_GET_ITEM(seq, i);
        long seat, no;
        if (attr_int(e, "seat", &seat) < 0 || attr_int(e, "hand_no", &no) < 0) { Py_DECREF(seq); return -1; }
        if (seat != A->seat || no != hand_no) continue;
        PyObject *kind = PyObject_GetAttrString(e, "kind");
        if (!kind) { Py_DECREF(seq); return -1; }
        const char *mine[5] = {"draw", "discard", "chi", "peng", "gang"};
        int which = -1;
        for (int k = 0; k < 5; k++)
            if (PyUnicode_CompareWithASCIIString(kind, mine[k]) == 0) which = k;
        Py_DECREF(kind);
        if (which < 0) continue;
        if ((which == 2 || which == 3) && A->opened >= 1) {
            race_action call;
            if (parse_action_tiles(e, &call) < 0) { Py_DECREF(seq); return -1; }
            int held = -1;
            for (int k = 0; k < call.n_tiles && call.n_tiles == 3; k++)
                if (call.tiles[k] == call.tile) { held = k; break; }
            if (call.n_tiles == 3 && held >= 0) {
                int h[NTILES];
                memcpy(h, c, sizeof(h));
                for (int k = 0; k < 3; k++)
                    if (k != held) h[call.tiles[k]]++;
                int over = 0;
                for (int t = 0; t < NTILES; t++) over |= h[t] > 4;
                if (over) {
                    PyErr_SetString(PyExc_ValueError, "more than four copies before the call");
                    Py_DECREF(seq);
                    return -1;
                }
                ok = is_baotou_counts(h, A->size + 2, A->opened - 1);
            }
        }
        break;
    }
    Py_DECREF(seq);
known:
    A->piao_known = 1;
    A->piao_ok = ok;
    return ok;
}

/* Python's floor division by a positive divisor. */
static int floor_div(int a, int b) {
    int q = a / b;
    if (a % b != 0 && a < 0) q -= 1;
    return q;
}

static int act_ahead(race_act_ctx *A, int *ahead) {
    PyObject *events = PyObject_GetAttrString(A->obs, "public_events");
    if (!events) return -1;
    PyObject *seq = PySequence_Fast(events, "expected events");
    Py_DECREF(events);
    if (!seq) return -1;
    int discarder = -1000;
    for (Py_ssize_t i = PySequence_Fast_GET_SIZE(seq) - 1; i >= 0; i--) {
        PyObject *e = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *kind = PyObject_GetAttrString(e, "kind");
        if (!kind) { Py_DECREF(seq); return -1; }
        int is_discard = PyUnicode_CompareWithASCIIString(kind, "discard") == 0;
        Py_DECREF(kind);
        if (is_discard) {
            long s;
            if (attr_int(e, "seat", &s) < 0) { Py_DECREF(seq); return -1; }
            discarder = (int)s;
            break;
        }
    }
    Py_DECREF(seq);
    if (discarder == -1000) {
        PyErr_SetString(PyExc_StopIteration, "no discard to pass on");
        return -1;
    }
    *ahead = ((A->seat - discarder - 1) % 4 + 4) % 4;
    return 0;
}

/* Race.value (with the race-classic discard for style 1). */
static int act_value(race_act_ctx *A, const race_action *a, double *out) {
    const race_profile *p = A->x.p;
    int *c = A->c, opened = A->opened, draws = A->draws;
    double v;
    switch (a->kind) {
    case ACT_HU:
        *out = (double)A->hu_fan + p->saved;
        return 0;
    case ACT_PASS: {
        int ahead;
        if (act_ahead(A, &ahead) < 0) return -1;
        if (act_waiting(A, c, opened, A->chain, A->piao, draws, &v) < 0) return -1;
        *out = pow(1 - p->draw_risk, (double)ahead) * v;
        return 0;
    }
    case ACT_DISCARD: {
        int piao = 0;
        if (a->tile == WHITE) {
            piao = act_piao(A);
            if (piao < 0) return -1;
        }
        c[a->tile]--;
        int rc;
        if (piao) {
            rc = act_waiting(A, c, opened, A->chain + 1, A->piao + 1, draws, &v);
        } else if (p->style == 1) {
            rc = act_waiting(A, c, opened, 0, 0, draws, &v);
        } else {
            ctx_at(A, opened);
            rc = race_reshaped(&A->x, c, draws, 0, &v);
        }
        long pot = potential_counts(c, A->x.avail);
        c[a->tile]++;
        if (rc < 0) return -1;
        *out = (1 - p->q) * v + p->tie * (double)pot;
        return 0;
    }
    case ACT_PENG:
    case ACT_CHI: {
        int used[3], n = 0;
        if (a->kind == ACT_CHI) {
            for (int i = 0; i < a->n_tiles; i++) used[n++] = a->tiles[i];
        } else {
            used[n++] = a->tile;
            used[n++] = a->tile;
        }
        int h[NTILES];
        memcpy(h, c, sizeof(h));
        for (int i = 0; i < n; i++) h[used[i]]--;
        double best = 0.0;
        int first = 1;
        for (int t = 0; t < NTILES; t++) {
            if (!h[t]) continue;
            h[t]--;
            int rc = act_waiting(A, h, opened + 1, 0, 0, draws, &v);
            h[t]++;
            if (rc < 0) return -1;
            if (first || v > best) best = v;
            first = 0;
        }
        if (h[WHITE] && is_baotou_counts(c, A->size, opened)) {
            h[WHITE]--;
            if (is_baotou_counts(h, A->size - n - 1, opened + 1)) {
                long god;
                if (attr_int(A->obs, "god_discarder_seat", &god) < 0) return -1;
                int owner = god == A->seat;
                int chain = (owner ? A->chain : 0) + 1, piao = (owner ? A->piao : 0) + 1;
                if (act_waiting(A, h, opened + 1, chain, piao, draws, &v) < 0) return -1;
                if (v > best) best = v;
            }
            h[WHITE]++;
        }
        *out = (1 - p->q) * best;
        return 0;
    }
    case ACT_GANG: {
        int h[NTILES];
        memcpy(h, c, sizeof(h));
        h[a->tile] -= a->kong;
        int after = a->added ? opened : opened + 1;
        ctx_at(A, after);
        analysis_row row;
        if (race_analysis(&A->x, h, A->chain + 1, A->piao, &row) < 0) return -1;
        if (act_waiting(A, h, after, 0, 0, draws, &v) < 0) return -1;
        if (row.distance > 0) {
            *out = v;
            return 0;
        }
        double unseen = A->x.unseen;
        long copies = row.copies;
        double now = ((double)row.value + p->saved * (double)copies) / unseen;
        double miss = 1 - (double)copies / unseen;
        *out = now + miss * (1 - p->q) * v;
        return 0;
    }
    default:
        *out = -INFINITY;
        return 0;
    }
}

/* Race.certain_double: a kong that leaves baotou, else a white discard that scores piao. */
static int act_double(race_act_ctx *A) {
    for (int i = 0; i < A->n_legal; i++) {
        race_action *a = &A->legal[i];
        if (a->kind != ACT_GANG) continue;
        int h[NTILES];
        memcpy(h, A->c, sizeof(h));
        h[a->tile] -= a->kong;
        int after = a->added ? A->opened : A->opened + 1;
        if (is_baotou_counts(h, A->size - a->kong, after)) return i;
    }
    for (int i = 0; i < A->n_legal; i++) {
        race_action *a = &A->legal[i];
        if (a->kind == ACT_DISCARD && a->tile == WHITE) {
            int piao = act_piao(A);
            if (piao < 0) return -2;
            if (piao) return i;
        }
    }
    return -1;
}

/* The shanten of the best hand a call leaves, and (wanted) the ukeire of the best ones. */
static int act_after_call(race_act_ctx *A, const race_action *a, int *shanten_out, long *ukeire_out) {
    int h[NTILES];
    memcpy(h, A->c, sizeof(h));
    if (a->kind == ACT_CHI) {
        for (int i = 0; i < a->n_tiles; i++) h[a->tiles[i]]--;
    } else {
        h[a->tile] -= 2;
    }
    int best = 1000;
    for (int t = 0; t < NTILES; t++) {
        if (!h[t]) continue;
        h[t]--;
        int s = sdk_shanten(h, A->opened + 1);
        h[t]++;
        if (s < best) best = s;
    }
    *shanten_out = best;
    if (!ukeire_out) return 0;
    /* after_call_ukeire: analyze's shanten and ukeire over the same discards. */
    ctx_at(A, A->opened + 1);
    int low = 1000;
    long most = 0;
    int have = 0;
    analysis_row rows[NTILES];
    int n = 0;
    for (int t = 0; t < NTILES; t++) {
        if (!h[t]) continue;
        h[t]--;
        int rc = race_analysis(&A->x, h, 0, 0, &rows[n]);
        h[t]++;
        if (rc < 0) return -1;
        if (rows[n].distance < low) low = rows[n].distance;
        n++;
    }
    for (int i = 0; i < n; i++) {
        if (rows[i].distance != low) continue;
        if (!have || rows[i].ukeire > most) most = rows[i].ukeire;
        have = 1;
    }
    *ukeire_out = most;
    return 0;
}

static PyObject *py_race_act(PyObject *self, PyObject *args) {
    PyObject *obs, *fallback;
    int profile;
    if (!PyArg_ParseTuple(args, "OiO", &obs, &profile, &fallback)) return NULL;
    if (profile < 0 || profile >= race_profile_count) {
        PyErr_SetString(PyExc_ValueError, "unknown race profile");
        return NULL;
    }
    race_act_ctx *A = PyMem_Calloc(1, sizeof(race_act_ctx));
    if (!A) return PyErr_NoMemory();
    PyObject *result = NULL, *hand = NULL, *discards = NULL, *melds = NULL, *stage = NULL, *legal = NULL;
    PyObject *available = NULL, *seat_melds = NULL;
    A->obs = obs;
    const race_profile *p = &race_profiles[profile];
    long v;
    if (!(stage = PyObject_GetAttrString(obs, "stage"))) goto done;
    if (attr_int(stage, "seat", &v) < 0) goto done;
    A->seat = (int)v;
    if (!(hand = PyObject_GetAttrString(obs, "hand"))) goto done;
    if (counts_of(hand, A->c, &A->size) < 0) goto done;
    if (!(melds = PyObject_GetAttrString(obs, "melds"))) goto done;
    if (!(seat_melds = PySequence_GetItem(melds, A->seat))) goto done;
    Py_ssize_t opened = PySequence_Size(seat_melds);
    if (opened < 0) goto done;
    A->opened = (int)opened;
    if (!(discards = PyObject_GetAttrString(obs, "discards"))) goto done;
    {
        PyObject *call = PyTuple_Pack(3, hand, discards, melds);
        if (!call) goto done;
        available = py_remaining_tiles(NULL, call);
        Py_DECREF(call);
        if (!available) goto done;
    }
    if (attr_int(obs, "chain_count", &v) < 0) goto done;
    A->chain = (int)v;
    if (attr_int(obs, "piao_count", &v) < 0) goto done;
    A->piao = (int)v;
    if (attr_int(obs, "hu_fan", &v) < 0) goto done;
    A->hu_fan = (int)v;
    if (attr_int(obs, "wall_remaining", &v) < 0) goto done;
    {
        /* My draws left before the wall is down to its last tiles: max(1, -(-(wall - last) // 4)). */
        int left = -floor_div(-((int)v - p->last_wall), 4);
        A->draws = left > 1 ? left : 1;
    }
    if (race_bind(&A->x, available, profile, fallback) < 0) goto done;
    if (!(legal = PyObject_GetAttrString(obs, "legal_actions"))) goto done;
    {
        PyObject *seq = PySequence_Fast(legal, "expected legal actions");
        if (!seq) goto done;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        if (n > 64 || n < 1) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "legal actions out of range");
            goto done;
        }
        A->n_legal = (int)n;
        for (Py_ssize_t i = 0; i < n; i++) {
            if (parse_action(PySequence_Fast_GET_ITEM(seq, i), &A->legal[i]) < 0) {
                Py_DECREF(seq);
                goto done;
            }
        }
        Py_DECREF(seq);
    }
    int has_hu = 0;
    for (int i = 0; i < A->n_legal; i++) has_hu |= A->legal[i].kind == ACT_HU;
    if (has_hu) {
        int d = act_double(A);
        if (d == -2) goto done;
        if (d >= 0) { result = PyLong_FromLong(d); goto done; }
    }
    int allowed[64];
    for (int i = 0; i < A->n_legal; i++) allowed[i] = 1;
    int style = p->style;
    if (style == 2) {
        /* kong-eager: kong unless already waiting and the kong would break the wait. */
        int now;
        if (A->size % 3 == 2) {
            now = 1000;
            for (int t = 0; t < NTILES; t++) {
                if (!A->c[t]) continue;
                A->c[t]--;
                int s = sdk_shanten(A->c, A->opened);
                A->c[t]++;
                if (s < now) now = s;
            }
        } else {
            now = sdk_shanten(A->c, A->opened);
        }
        for (int i = 0; i < A->n_legal; i++) {
            race_action *a = &A->legal[i];
            if (a->kind != ACT_GANG) continue;
            int take = now > 0;
            if (!take) {
                int h[NTILES];
                memcpy(h, A->c, sizeof(h));
                h[a->tile] -= a->kong;
                take = sdk_shanten(h, a->added ? A->opened : A->opened + 1) <= 0;
            }
            if (take) { result = PyLong_FromLong(i); goto done; }
        }
    } else if (style == 3 || style == 4) {
        /* peng-speed / chi-speed: a call that brings the hand closer, or as close and wider. */
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
                if (take) { result = PyLong_FromLong(i); goto done; }
            }
        }
    } else if (style == 5) {
        /* closed-early: no calls three or more steps from waiting. */
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
        result = PyLong_FromLong(best);
    }
done:
    Py_XDECREF(stage);
    Py_XDECREF(hand);
    Py_XDECREF(melds);
    Py_XDECREF(seat_melds);
    Py_XDECREF(discards);
    Py_XDECREF(legal);
    Py_XDECREF(available);
    PyMem_Free(A);
    return result;
}


/* race_set_order(items): py_set_order, for the tests to check it against Python's sets. */
static PyObject *py_race_set_order(PyObject *self, PyObject *args) {
    PyObject *seq;
    if (!PyArg_ParseTuple(args, "O", &seq)) return NULL;
    PyObject *fast = PySequence_Fast(seq, "sequence");
    if (!fast) return NULL;
    int n = (int)PySequence_Fast_GET_SIZE(fast), items[64], out[64];
    if (n > 64) n = 64;
    for (int i = 0; i < n; i++) items[i] = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
    Py_DECREF(fast);
    int m = py_set_order(items, n, out);
    PyObject *r = PyList_New(m);
    for (int i = 0; i < m; i++) PyList_SET_ITEM(r, i, PyLong_FromLong(out[i]));
    return r;
}

static PyMethodDef fast_methods[] = {
    {"encode_public_planes", py_encode_public_planes, METH_VARARGS, "Public count feature planes."},
    {"encode_lookahead", py_encode_lookahead, METH_VARARGS, "Public lookahead feature planes."},
    {"remaining_tiles", py_remaining_tiles, METH_VARARGS, "Unseen copies from public observation fields."},
    {"baotou_outs", py_baotou_outs, METH_VARARGS, "Weighted draws with a baotou discard."},
    {"tenpai_route_value", py_tenpai_route_value, METH_VARARGS, "Best tenpai fan after each improving draw."},
    {"analyze_hand", py_analyze_hand, METH_VARARGS,
     "Analyze tile ids directly; None defers the winning boundary to Python."},
    {"analyze_counts", py_analyze_counts, METH_VARARGS,
     "Distance, improving draws, winning draws and their numeric fans."},
    {"baotou", py_baotou, METH_VARARGS,
     "Whether every drawable tile completes the 13-3*open waiting hand."},
    {"is_baotou", py_is_baotou, METH_VARARGS,
     "rules.is_baotou for a hand of tile ids; None when Python must decide."},
    {"counts", py_counts, METH_O, "Per-tile counts of a hand, validating ids."},
    {"shape_value", py_shape_value, METH_VARARGS,
     "concealed shape value 8 - 2*melds - min(partials, 4 - melds) - pair"},
    {"shanten_raw", py_shanten_raw, METH_VARARGS,
     "shanten before the winning-boundary guard (white completes, seven pairs)"},
    {"can_hu", py_can_hu, METH_VARARGS,
     "winning shape: 4 - open_melds melds plus a pair, white completing gaps"},
    {"fan_value", py_fan_value, METH_VARARGS,
     "numeric fan for a winning draw, without constructing detail strings"},
    {"improving", py_improving, METH_VARARGS,
     "raw shanten and (improving, winning-candidate) tiles for every legal draw"},
    {"race_profile", py_race_profile, METH_VARARGS,
     "Register a race strategy's constants (q, saved, one_step, far_step, q_early, q_late, hunt, upgrade); returns its id."},
    {"race_waiting", py_race_waiting, METH_VARARGS,
     "waiting_value(hand, opened, available, chain, piao, draws) of a race profile; fallback analyses boundary hands."},
    {"race_reshaped", py_race_reshaped, METH_VARARGS,
     "reshaped_value(hand, opened, available, draws) of a race profile."},
    {"race_baotou_fans", py_race_baotou_fans, METH_VARARGS,
     "(winning copies, fan-weighted copies) of a baotou wait's draws."},
    {"race_upgrade_drops", py_race_upgrade_drops, METH_VARARGS,
     "Bit d: the waiting counts plus tile, less d, are baotou."},
    {"race_potential", py_race_potential, METH_VARARGS,
     "potential(hand, available): live tiles pairing with or within two of each tile."},
    {"race_act", py_race_act, METH_VARARGS,
     "race_act(obs, profile, analyze): index of the race strategy's action in obs.legal_actions."},
    {"race_set_order", py_race_set_order, METH_VARARGS,
     "The order Python iterates a set of small ints added in the given order."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fast_module = {
    PyModuleDef_HEAD_INIT, "_fast",
    "C fast path for hand-shape primitives; falls back to Python at boundaries.",
    -1, fast_methods,
};

PyMODINIT_FUNC PyInit__fast(void) {
    const char *names[7] = {"shanten", "improving", "ukeire", "winning_tiles",
                           "winning_copies", "fans", "fan_value"};
    for (int i = 0; i < 7; i++) {
        if (!analysis_keys[i]) analysis_keys[i] = PyUnicode_InternFromString(names[i]);
        if (!analysis_keys[i]) return NULL;
    }
    return PyModule_Create(&fast_module);
}
