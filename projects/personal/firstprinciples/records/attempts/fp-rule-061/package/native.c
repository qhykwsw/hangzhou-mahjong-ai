/* The rule line's own C: the call option of a hand not yet waiting.
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
        c[d]++;
        if (rc == 0) {
            v = (1 - q) * v;
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

static PyMethodDef native_methods[] = {
    {"race_profile", py_race_profile, METH_VARARGS,
     "Register a race profile in this module's own table (the runtime's ids do not carry over)."},
    {"call_option", py_call_option, METH_VARARGS,
     "call_option(hand, opened, available, draws, stage, chis, profile, fallback, q, weight): the value "
     "calls add to a hand not yet waiting."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef native_module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, native_methods};

PyMODINIT_FUNC PyInit_native(void) {
    if (fast_keys() < 0) return NULL;
    return PyModule_Create(&native_module);
}
