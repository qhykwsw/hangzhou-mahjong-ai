/* Fast structural primitives for the mahjong SDK.
 *
 * This is a faithful port of the pure-Python reference implementations:
 *   - sdk/shape.py::_normal  (concealed shape distance / shanten)
 *   - sdk/shape.py::shanten  (white-tile completion, seven pairs, open melds)
 * The Python side keeps the reference implementation as the fallback and defers
 * to it for every negative (winning-boundary) result, so the two must agree on
 * all non-negative values. tests/test_shape_fast.py checks them exhaustively.
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

static uint32_t byte_hash(const uint8_t *key, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        h ^= key[i];
        h *= 16777619u;
    }
    return h;
}

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

typedef struct {
    uint8_t key[35];
    uint8_t value;
    uint8_t used;
} meld_slot;

static meld_slot meld_cache[MELD_CACHE_SIZE];

/* Port of domain/rules.py::_meldable: can every tile join a meld, jokers filling. */
static int meldable(const int *c, int jokers, uint8_t *key) {
    uint32_t index = byte_hash(key, 35) & (MELD_CACHE_SIZE - 1);
    meld_slot *slot = &meld_cache[index];
    if (slot->used && memcmp(slot->key, key, 35) == 0) {
        return slot->value;
    }
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
            uint8_t next[35];
            for (int i = 0; i < 34; i++) next[i] = (uint8_t)x[i];
            next[34] = (uint8_t)(jokers - need);
            if (meldable(x, jokers - need, next)) {
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
                for (int t = start; t < start + 3; t++) {
                    if (x[t]) {
                        x[t] -= 1;
                    } else {
                        missing += 1;
                    }
                }
                if (missing > jokers) continue;
                uint8_t next[35];
                for (int i = 0; i < 34; i++) next[i] = (uint8_t)x[i];
                next[34] = (uint8_t)(jokers - missing);
                if (meldable(x, jokers - missing, next)) {
                    result = 1;
                }
            }
        }
    }
    memcpy(slot->key, key, 35);
    slot->value = (uint8_t)result;
    slot->used = 1;
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
    uint8_t key[35];
    if (open_melds == 0) {
        int odd = 0;
        for (int i = 0; i < 33; i++) odd += natural[i] % 2;
        if (odd <= jokers) return 1;
    }
    if (jokers >= 2) {
        for (int i = 0; i < 34; i++) key[i] = (uint8_t)natural[i];
        key[34] = (uint8_t)(jokers - 2);
        if (meldable(natural, jokers - 2, key)) return 1;
    }
    for (int i = 0; i < 33; i++) {
        int most = natural[i] < 2 ? natural[i] : 2;
        for (int take = most; take >= 1; take--) {
            int need = 2 - take;
            if (need > jokers) continue;
            int x[34];
            memcpy(x, natural, sizeof(x));
            x[i] -= take;
            for (int k = 0; k < 34; k++) key[k] = (uint8_t)x[k];
            key[34] = (uint8_t)(jokers - need);
            if (meldable(x, jokers - need, key)) return 1;
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
    int draw[NTILES];
    memcpy(draw, waiting, sizeof(draw));
    for (int t = 0; t < NTILES; t++) {
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

typedef struct {
    uint8_t key[35];
    uint8_t value;
    uint8_t used;
} baotou_slot;

static baotou_slot baotou_cache[BAOTOU_CACHE_SIZE];

static int baotou_cached(const int *counts, int open_melds) {
    uint8_t key[35];
    for (int i = 0; i < NTILES; i++) key[i] = (uint8_t)counts[i];
    key[34] = (uint8_t)open_melds;
    uint32_t index = byte_hash(key, 35) & (BAOTOU_CACHE_SIZE - 1);
    baotou_slot *slot = &baotou_cache[index];
    if (slot->used && memcmp(slot->key, key, 35) == 0) {
        return slot->value;
    }
    int value = baotou_shape(counts, open_melds);
    memcpy(slot->key, key, 35);
    slot->value = (uint8_t)value;
    slot->used = 1;
    return value;
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
    int bt = baotou_shape(waiting, open_melds);
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
static PyObject *py_analyze_counts(PyObject *self, PyObject *args) {
    PyObject *counts_obj, *avail_obj;
    int open_melds, chain_count, piao_count, you_cai_bi_kao;
    if (!PyArg_ParseTuple(args, "OiOiii", &counts_obj, &open_melds, &avail_obj,
                          &chain_count, &piao_count, &you_cai_bi_kao))
        return NULL;
    if (open_melds < 0 || open_melds > 4) {
        PyErr_SetString(PyExc_ValueError, "open_melds must be 0..4");
        return NULL;
    }
    int counts[NTILES], avail[NTILES], drawn[NTILES];
    if (parse_counts(counts_obj, counts) < 0) return NULL;
    if (parse_vector(avail_obj, avail, 1000000, "available") < 0) return NULL;
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

static PyMethodDef fast_methods[] = {
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
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fast_module = {
    PyModuleDef_HEAD_INIT, "_fast",
    "C fast path for hand-shape primitives; falls back to Python at boundaries.",
    -1, fast_methods,
};

PyMODINIT_FUNC PyInit__fast(void) { return PyModule_Create(&fast_module); }
