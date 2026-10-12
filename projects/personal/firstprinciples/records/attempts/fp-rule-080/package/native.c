/* The rule line's own C: the call option of a hand not yet waiting, and the
 * reshape option of a hand one step away.
 *
 * The race values a hand by the draws that improve it; an opponent's discard
 * can improve it too, by a peng of a pair of mine or a chi (from my left) of
 * two neighbours of mine. Per round, each such tile comes at a rate measured
 * against the fixed pool, relative to my own chance of drawing it, and is
 * taken when the best hand the call keeps beats staying as I am. Written in
 * the same arithmetic order as the Python it replaces, over the runtime's
 * race valuation. */
#include "_fast.c"

/* Opponents' discards of a tile in one round between two of my draws, over my
 * chance of drawing it, by stage (wall >= 70, >= 45, else) and kind (honor,
 * terminal, 2/8, middle): any opponent's for a peng, my left one's for a chi. */
static const double FEED_PENG[3][4] = {{9.05, 4.38, 2.24, 0.99}, {3.55, 3.97, 3.17, 2.08}, {3.41, 3.09, 2.68, 1.97}};
static const double FEED_CHI[3][4] = {{0.0, 1.50, 0.79, 0.35}, {0.0, 1.23, 1.01, 0.67}, {0.0, 0.84, 0.82, 0.57}};

/* The strategy's shape correction (strategy.shape_bonus), set once from Python. */
static double SHAPE[6]; /* pair, adjacent, ukeire, distance, open, edge */

static int shape_bonus(race_ctx *y, const int *c, double *out) {
    int pairs = 0, adjacent = 0, edges = 0;
    for (int t = 0; t < 33; t++)
        if (c[t] >= 2) pairs++;
    for (int t = 0; t < 27; t++) {
        if (t % 9 <= 7 && c[t] && c[t + 1]) {
            adjacent++;
            if (t % 9 == 0 || t % 9 == 7) edges++;
        }
    }
    double bonus = SHAPE[0] * pairs + SHAPE[1] * adjacent;
    bonus += SHAPE[4] * (adjacent - edges) + SHAPE[5] * edges;
    analysis_row row;
    if (race_analysis_at(y, c, y->opened, 0, 0, &row) < 0) return -1;
    bonus += SHAPE[2] * (double)row.ukeire / 10 + SHAPE[3] * row.distance;
    *out = bonus;
    return 0;
}

static PyObject *py_shape_constants(PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "dddddd", &SHAPE[0], &SHAPE[1], &SHAPE[2], &SHAPE[3], &SHAPE[4], &SHAPE[5]))
        return NULL;
    Py_RETURN_NONE;
}

static int tile_kind(int t) {
    if (t >= 27) return 0;
    int r = t % 9;
    return (r == 0 || r == 8) ? 1 : (r == 1 || r == 7) ? 2 : 3;
}

/* The best (1 - q) * value of a hand kept after calling with tiles a and b:
 * a waiting value one step away, else the reshaped value. */
static int called_value(race_ctx *y, int *c, int a, int b, int far, int draws, double q, double *out) {
    c[a]--;
    c[b]--;
    double best = 0.0;
    int rc = 0, near = 99, dist[NTILES];
    /* Only the discards that keep the hand closest to waiting. */
    for (int d = 0; d < NTILES; d++) {
        if (!c[d]) continue;
        c[d]--;
        dist[d] = shanten_raw(c, y->opened);
        c[d]++;
        if (dist[d] < near) near = dist[d];
    }
    for (int d = 0; d < NTILES && rc == 0; d++) {
        if (!c[d] || dist[d] != near) continue;
        c[d]--;
        double v;
        rc = far ? race_reshaped(y, c, draws, 0, &v) : race_waiting(y, c, 0, 0, draws, 0, &v);
        double shape = 0.0;
        if (rc == 0) rc = shape_bonus(y, c, &shape);
        c[d]++;
        if (rc == 0) {
            v = (1 - q) * v + shape;
            if (v > best) best = v;
        }
    }
    c[a]++;
    c[b]++;
    *out = best;
    return rc;
}

/* call_option(hand, opened, available, draws, stage, chis, profile, fallback, q, weight) */
static PyObject *py_call_option(PyObject *self, PyObject *args) {
    PyObject *hand, *available, *fallback;
    int opened, draws, stage, chis, profile, c[NTILES];
    double q, weight;
    if (!PyArg_ParseTuple(args, "OiOiiiiOdd", &hand, &opened, &available, &draws, &stage, &chis, &profile,
                          &fallback, &q, &weight))
        return NULL;
    if (stage < 0 || stage > 2) {
        PyErr_SetString(PyExc_ValueError, "stage must be 0..2");
        return NULL;
    }
    if (opened >= 4) return PyFloat_FromDouble(0.0);
    race_ctx x;
    memset(&x, 0, sizeof(x));
    if (race_setup(&x, c, hand, opened, available, profile, fallback) < 0) return NULL;
    analysis_row row;
    if (race_analysis(&x, c, 0, 0, &row) < 0) return NULL;
    if (row.distance < 1) return PyFloat_FromDouble(0.0);
    int far = row.distance >= 2;
    double stay_value;
    if (race_reshaped(&x, c, draws, 0, &stay_value) < 0) return NULL;
    double stay_shape;
    if (shape_bonus(&x, c, &stay_shape) < 0) return NULL;
    stay_value += stay_shape;
    race_ctx y = x;
    y.opened = opened + 1;
    y.size = x.size - 3;
    double unseen = (double)x.unseen, gain = 0.0;
    for (int t = 0; t < WHITE; t++) {
        if (!x.avail[t]) continue;
        double best_gain = 0.0, best_rate = 0.0, v;
        if (c[t] >= 2) {
            if (called_value(&y, c, t, t, far, draws, q, &v) < 0) return NULL;
            if (v - stay_value > best_gain) {
                best_gain = v - stay_value;
                best_rate = FEED_PENG[stage][tile_kind(t)];
            }
        }
        if (t < 27 && chis < 2) {
            const int pairs[3][2] = {{t - 2, t - 1}, {t - 1, t + 1}, {t + 1, t + 2}};
            for (int k = 0; k < 3; k++) {
                int a = pairs[k][0], b = pairs[k][1];
                if (a < 0 || b >= 27 || a / 9 != t / 9 || b / 9 != t / 9 || !c[a] || !c[b]) continue;
                if (called_value(&y, c, a, b, far, draws, q, &v) < 0) return NULL;
                if (v - stay_value > best_gain) {
                    best_gain = v - stay_value;
                    best_rate = FEED_CHI[stage][tile_kind(t)];
                }
            }
        }
        gain += best_rate * (double)x.avail[t] / unseen * best_gain;
    }
    double u = (double)row.ukeire / unseen;
    double stay = (1 - u) * (1 - q);
    double rounds = stay < 1 ? (1 - pow(stay, (double)draws)) / (1 - stay) : (double)draws;
    return PyFloat_FromDouble(weight * gain * rounds);
}

/* reshape_gain(hand, opened, available, draws, profile, fallback, q, speed):
 * the option of a hand one step from waiting to keep a draw that does not
 * improve it and discard another tile, when that leaves a better hand one
 * step away. For each such draw every trade that stays one step away is
 * priced by the race, and the best is taken against staying as I
 * am; the gains accrue each round the hand has not yet moved on. */
static PyObject *py_reshape_gain(PyObject *self, PyObject *args) {
    PyObject *hand, *available, *fallback;
    int opened, draws, profile, c[NTILES], held[NTILES];
    double q, speed;
    if (!PyArg_ParseTuple(args, "OiOiiOdd", &hand, &opened, &available, &draws, &profile, &fallback, &q, &speed))
        return NULL;
    race_ctx x;
    memset(&x, 0, sizeof(x));
    if (race_setup(&x, c, hand, opened, available, profile, fallback) < 0) return NULL;
    race_speed(&x, speed);
    analysis_row row;
    if (race_analysis(&x, c, 0, 0, &row) < 0) return NULL;
    if (row.distance != 1 || draws < 3) return PyFloat_FromDouble(0.0);
    int improving[NTILES] = {0};
    for (int i = 0; i < row.n_improving; i++) improving[row.improving[i]] = 1;
    double ukeire = (double)row.ukeire;
    double stay;
    if (race_waiting(&x, c, 0, 0, draws - 1, 0, &stay) < 0) return NULL;
    memcpy(held, c, sizeof(held));
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
        double best = stay;
        for (int drop = 0; drop < NTILES; drop++) {
            if (!held[drop] || drop == tile) continue;
            c[drop]--;
            if (shanten_raw(c, opened) == 1) {
                double v;
                if (race_waiting(&x, c, 0, 0, draws - 1, 0, &v) < 0) {
                    c[drop]++;
                    c[tile]--;
                    return NULL;
                }
                if (v > best) best = v;
            }
            c[drop]++;
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
    return PyFloat_FromDouble((1 - qs) * gain / unseen * rounds);
}

/* feed_rate(stage, tile, chi): the rate above for one tile. */
static PyObject *py_feed_rate(PyObject *self, PyObject *args) {
    int stage, tile, chi;
    if (!PyArg_ParseTuple(args, "iip", &stage, &tile, &chi)) return NULL;
    if (stage < 0 || stage > 2 || tile < 0 || tile >= WHITE) {
        PyErr_SetString(PyExc_ValueError, "stage must be 0..2 and tile 0..32");
        return NULL;
    }
    return PyFloat_FromDouble(chi ? FEED_CHI[stage][tile_kind(tile)] : FEED_PENG[stage][tile_kind(tile)]);
}

static PyMethodDef native_methods[] = {
    {"race_profile", py_race_profile, METH_VARARGS,
     "Register a race profile in this module's own table (the runtime's ids do not carry over)."},
    {"shape_constants", py_shape_constants, METH_VARARGS, "Set the strategy's shape correction constants."},
    {"feed_rate", py_feed_rate, METH_VARARGS,
     "feed_rate(stage, tile, chi): opponents' discards of a tile per round over my chance of drawing it."},
    {"call_option", py_call_option, METH_VARARGS,
     "call_option(hand, opened, available, draws, stage, chis, profile, fallback, q, weight): the value "
     "calls add to a hand not yet waiting."},
    {"reshape_gain", py_reshape_gain, METH_VARARGS,
     "reshape_gain(hand, opened, available, draws, profile, fallback, q, speed): the option of a hand one "
     "step away to trade a non-improving draw for a better hand one step away."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef native_module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, native_methods};

PyMODINIT_FUNC PyInit_native(void) {
    if (fast_keys() < 0) return NULL;
    return PyModule_Create(&native_module);
}
