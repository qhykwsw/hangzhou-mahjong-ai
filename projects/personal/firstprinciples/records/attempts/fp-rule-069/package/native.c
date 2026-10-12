/* The self-draw simulation: a kept hand is worth what its own future draws
 * make of it. Each simulation shuffles the tiles I cannot see and lets me draw
 * them in turn, discarding by a quick greedy rule (least distance from a win,
 * then the least connected tile, never a white while another will do), until
 * the hand wins or my draws run out. A win at my t-th draw is worth its fan
 * (the runtime's fan_value: baotou, seven pairs, four whites) plus the payment
 * it saves, times the chance no opponent has won before it, from the measured
 * per-round hazard. Every candidate hand meets the same shuffles. */
#include "_fast.c"

static uint64_t rng_state = 88172645463325252ULL;

static inline uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

/* The opponents' chance to win in the round before my draw, with d draws of
 * mine left (measured against the fixed pool). */
static double round_hazard(int d) {
    double q = 0.35 - 0.018 * d;
    if (q < 0.10) q = 0.10;
    if (q > 0.30) q = 0.30;
    return q;
}

static int connection(const int *c, int t) {
    int s = 3 * c[t];
    if (t >= 27) return s;
    int r = t % 9;
    if (r >= 1 && c[t - 1]) s += 2;
    if (r <= 7 && c[t + 1]) s += 2;
    if (r >= 2 && c[t - 2]) s += 1;
    if (r <= 6 && c[t + 2]) s += 1;
    return s;
}

static int greedy_discard(int *c, int opened) {
    int best = -1, best_sh = 99, best_score = 1 << 30;
    for (int t = 0; t < NTILES; t++) {
        if (!c[t]) continue;
        c[t]--;
        int sh = shanten_raw(c, opened);
        c[t]++;
        int score = t == WHITE ? 1000 : connection(c, t);
        if (sh < best_sh || (sh == best_sh && score < best_score)) {
            best_sh = sh;
            best_score = score;
            best = t;
        }
    }
    return best;
}

#define MAX_CANDS 34

/* sim_values(candidates, opened, unseen, draws, sims, seed, saved) */
static PyObject *py_sim_values(PyObject *self, PyObject *args) {
    PyObject *cands, *unseen_obj;
    int opened, draws, sims;
    unsigned long long seed;
    double saved;
    if (!PyArg_ParseTuple(args, "OiOiiKd", &cands, &opened, &unseen_obj, &draws, &sims, &seed, &saved)) return NULL;
    PyObject *fast = PySequence_Fast(cands, "candidates must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t nc = PySequence_Fast_GET_SIZE(fast);
    if (nc < 1 || nc > MAX_CANDS || opened < 0 || opened > 4 || draws < 0 || sims < 1) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "bad simulation arguments");
        return NULL;
    }
    int hands[MAX_CANDS][NTILES];
    for (Py_ssize_t i = 0; i < nc; i++) {
        if (parse_counts(PySequence_Fast_GET_ITEM(fast, i), hands[i]) < 0) {
            Py_DECREF(fast);
            return NULL;
        }
    }
    Py_DECREF(fast);
    int unseen[NTILES];
    if (parse_counts(unseen_obj, unseen) < 0) return NULL;
    int pool[136], n = 0;
    for (int t = 0; t < NTILES; t++)
        for (int k = 0; k < unseen[t]; k++) pool[n++] = t;
    if (draws > n) draws = n;
    double surv[40];
    surv[0] = 1.0;
    for (int t = 1; t <= draws && t < 40; t++) surv[t] = surv[t - 1] * (1 - round_hazard(draws - t + 1));
    double value[MAX_CANDS];
    for (Py_ssize_t i = 0; i < nc; i++) value[i] = 0.0;
    rng_state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 1;
    for (int s = 0; s < sims; s++) {
        for (int k = 0; k < draws; k++) {
            int j = k + (int)(rng_next() % (uint64_t)(n - k));
            int tmp = pool[k];
            pool[k] = pool[j];
            pool[j] = tmp;
        }
        for (Py_ssize_t i = 0; i < nc; i++) {
            int c[NTILES];
            memcpy(c, hands[i], sizeof(c));
            for (int t = 1; t <= draws; t++) {
                int d = pool[t - 1];
                if (c[d] >= 4) continue;
                c[d]++;
                if (hu_shape(c, opened)) {
                    long fan = fan_value(c, d, opened, 0, 0, 0, 0);
                    value[i] += surv[t] * ((double)fan + saved);
                    break;
                }
                c[greedy_discard(c, opened)]--;
            }
        }
    }
    PyObject *out = PyList_New(nc);
    if (!out) return NULL;
    for (Py_ssize_t i = 0; i < nc; i++) PyList_SET_ITEM(out, i, PyFloat_FromDouble(value[i] / sims));
    return out;
}

static PyMethodDef native_methods[] = {
    {"sim_values", py_sim_values, METH_VARARGS,
     "sim_values(candidates, opened, unseen, draws, sims, seed, saved): the simulated value of each kept hand."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef native_module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, native_methods};

PyMODINIT_FUNC PyInit_native(void) {
    if (fast_keys() < 0) return NULL;
    return PyModule_Create(&native_module);
}
