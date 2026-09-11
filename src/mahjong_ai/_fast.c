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

/* Best 8 - 2*melds - min(partials, 4 - melds) - pair over the four groups. */
static int shape_value(const int *c, int open_melds) {
    int g[5][2];
    int next[5][2];
    for (int m = 0; m < 5; m++)
        for (int p = 0; p < 2; p++) g[m][p] = -1;
    g[open_melds][0] = 0;
    for (int s = 0; s < 4; s++) {
        ctx.stamp++;
        if (ctx.stamp == 0) {
            memset(ctx.gen, 0, sizeof(ctx.gen));
            ctx.stamp = 1;
        }
        int n = s < 3 ? 9 : 7;
        const int *row = c + (s < 3 ? s * 9 : 27);
        suit_res r = suit_rec(row, n, 0, row[0], n > 1 ? row[1] : 0, n > 2 ? row[2] : 0,
                              s < 3);
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

#define SHANTEN_CACHE_BITS 15
#define SHANTEN_CACHE_SIZE (1 << SHANTEN_CACHE_BITS)

typedef struct {
    uint8_t key[35];
    int8_t value;
    uint8_t used;
} shanten_slot;

static shanten_slot shanten_cache[SHANTEN_CACHE_SIZE];

static uint32_t byte_hash(const uint8_t *key, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        h ^= key[i];
        h *= 16777619u;
    }
    return h;
}

static int shanten_raw(const int *counts, int open_melds) {
    uint8_t key[35];
    for (int i = 0; i < 34; i++) key[i] = (uint8_t)counts[i];
    key[34] = (uint8_t)open_melds;
    uint32_t index = byte_hash(key, 35) & (SHANTEN_CACHE_SIZE - 1);
    shanten_slot *slot = &shanten_cache[index];
    if (slot->used && memcmp(slot->key, key, 35) == 0) {
        return slot->value;
    }
    int value = shanten_uncached(counts, open_melds);
    memcpy(slot->key, key, 35);
    slot->value = (int8_t)value;
    slot->used = 1;
    return value;
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
#define MELD_CACHE_BITS 15
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
    int distance = shanten_raw(counts, open_melds);
    int drawn[NTILES];
    uint64_t improving = 0, negatives = 0;
    for (int t = 0; t < 34; t++) {
        if (avail[t] <= 0 || counts[t] >= 4) continue;
        memcpy(drawn, counts, sizeof(drawn));
        drawn[t]++;
        int d = shanten_raw(drawn, open_melds);
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

static PyMethodDef fast_methods[] = {
    {"shape_value", py_shape_value, METH_VARARGS,
     "concealed shape value 8 - 2*melds - min(partials, 4 - melds) - pair"},
    {"shanten_raw", py_shanten_raw, METH_VARARGS,
     "shanten before the winning-boundary guard (white completes, seven pairs)"},
    {"can_hu", py_can_hu, METH_VARARGS,
     "winning shape: 4 - open_melds melds plus a pair, white completing gaps"},
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
