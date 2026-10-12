/* The self-draw simulation: a kept hand is worth what its own future makes of
 * it. Each simulation shuffles the tiles I cannot see and deals them out four
 * per round, as the wall is: the first three are what my right, opposite and
 * left opponents draw and throw, the fourth is my draw. A thrown tile I can
 * peng (from anyone) or chi (from my left) is called when the best discard
 * after the call brings the hand strictly closer to a win; such a round has no
 * draw of mine. Otherwise I draw and discard by a quick greedy rule (least
 * distance from a win, then the least connected tile, never a white while
 * another will do), until the hand wins or my draws run out. A win at my t-th
 * draw is worth its fan (the runtime's fan_value: baotou, seven pairs, four
 * whites) plus the payment it saves, times the chance no opponent has won
 * before it, from the measured per-round hazard. Every candidate hand meets
 * the same shuffles. */
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

static int live[NTILES];

/* The live tiles whose draw brings a hand (13 - 3*opened tiles) closer to a win. */
static int ukeire(const int *c, int opened) {
    const draw_slot *ds = draw_distances(c, opened);
    int u = 0;
    for (int t = 0; t < NTILES; t++)
        if (ds->draw[t] != NO_DRAW && ds->draw[t] < ds->distance) u += live[t];
    return u;
}

/* The discard leaving the hand closest to a win; among those, the one whose
 * hand has the most live improving tiles, then the least connected tile, and a
 * white only when nothing else is as close. */
static int greedy_discard(int *c, int opened) {
    int sh[NTILES], best_sh = 99;
    for (int t = 0; t < NTILES; t++) {
        sh[t] = 99;
        if (!c[t]) continue;
        c[t]--;
        sh[t] = shanten_raw(c, opened);
        c[t]++;
        if (t != WHITE && sh[t] < best_sh) best_sh = sh[t];
    }
    if (best_sh == 99 || (c[WHITE] && sh[WHITE] < best_sh)) return WHITE;
    int best = -1, best_uk = -1, best_score = 1 << 30, ties = 0;
    for (int t = 0; t < NTILES; t++)
        if (t != WHITE && sh[t] == best_sh) ties++;
    for (int t = 0; t < NTILES; t++) {
        if (t == WHITE || sh[t] != best_sh) continue;
        int uk = 0;
        if (ties > 1) {
            c[t]--;
            uk = ukeire(c, opened);
            c[t]++;
        }
        int score = connection(c, t);
        if (uk > best_uk || (uk == best_uk && score < best_score)) {
            best_uk = uk;
            best_score = score;
            best = t;
        }
    }
    return best;
}

/* The distance after the best discard of a hand holding one tile too many. */
static int best_after(int *c, int opened) {
    int best = 99;
    for (int t = 0; t < NTILES; t++) {
        if (!c[t]) continue;
        c[t]--;
        int sh = shanten_raw(c, opened);
        c[t]++;
        if (sh < best) best = sh;
    }
    return best;
}

/* Call the thrown tile x if that brings the hand strictly closer to a win:
 * a peng from anyone, a chi only from my left (while under two chis). */
static int try_call(int *c, int *opened, int *chis, int x, int from_left) {
    if (x == WHITE || *opened >= 4) return 0;
    int now = shanten_raw(c, *opened);
    int best = now, how = -1;
    if (c[x] >= 2) {
        c[x] -= 2;
        int sh = best_after(c, *opened + 1);
        c[x] += 2;
        if (sh < best) { best = sh; how = 0; }
    }
    if (from_left && x < 27 && *chis < 2) {
        const int pairs[3][2] = {{x - 2, x - 1}, {x - 1, x + 1}, {x + 1, x + 2}};
        for (int k = 0; k < 3; k++) {
            int a = pairs[k][0], b = pairs[k][1];
            if (a < 0 || b >= 27 || a / 9 != x / 9 || b / 9 != x / 9 || !c[a] || !c[b]) continue;
            c[a]--; c[b]--;
            int sh = best_after(c, *opened + 1);
            c[a]++; c[b]++;
            if (sh < best) { best = sh; how = 1 + k; }
        }
    }
    if (how < 0) return 0;
    if (how == 0) {
        c[x] -= 2;
    } else {
        const int pairs[3][2] = {{x - 2, x - 1}, {x - 1, x + 1}, {x + 1, x + 2}};
        c[pairs[how - 1][0]]--;
        c[pairs[how - 1][1]]--;
        (*chis)++;
    }
    (*opened)++;
    c[greedy_discard(c, *opened)]--;
    return 1;
}

#define MAX_CANDS 34

/* sim_values(candidates, opened, chis, unseen, draws, sims, seed, saved, chain, piao) */
static PyObject *py_sim_values(PyObject *self, PyObject *args) {
    PyObject *cands, *unseen_obj;
    int opened, chis0, draws, sims, chain = 0, piao = 0;
    unsigned long long seed;
    double saved;
    if (!PyArg_ParseTuple(args, "OiiOiiKd|ii", &cands, &opened, &chis0, &unseen_obj, &draws, &sims, &seed, &saved, &chain, &piao)) return NULL;
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
    memcpy(live, unseen, sizeof(live));
    int pool[136], n = 0;
    for (int t = 0; t < NTILES; t++)
        for (int k = 0; k < unseen[t]; k++) pool[n++] = t;
    if (4 * draws > n) draws = n / 4;
    double surv[40];
    surv[0] = 1.0;
    for (int t = 1; t <= draws && t < 40; t++) surv[t] = surv[t - 1] * (1 - round_hazard(draws - t + 1));
    double value[MAX_CANDS];
    for (Py_ssize_t i = 0; i < nc; i++) value[i] = 0.0;
    rng_state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 1;
    int need = 4 * draws;
    if (need > n) need = n;
    for (int s = 0; s < sims; s++) {
        for (int k = 0; k < need; k++) {
            int j = k + (int)(rng_next() % (uint64_t)(n - k));
            int tmp = pool[k];
            pool[k] = pool[j];
            pool[j] = tmp;
        }
        for (Py_ssize_t i = 0; i < nc; i++) {
            int c[NTILES], op = opened, chis = chis0;
            memcpy(c, hands[i], sizeof(c));
            for (int t = 1; t <= draws; t++) {
                int base = 4 * (t - 1);
                if (base + 3 >= need) break;
                int called = 0;
                for (int k = 0; k < 3 && !called; k++) called = try_call(c, &op, &chis, pool[base + k], k == 2);
                if (called) continue;
                int d = pool[base + 3];
                if (c[d] >= 4) continue;
                c[d]++;
                if (hu_shape(c, op)) {
                    long fan = fan_value(c, d, op, chain, piao, 0, 0);
                    value[i] += surv[t] * ((double)fan + saved);
                    break;
                }
                c[greedy_discard(c, op)]--;
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
     "sim_values(candidates, opened, chis, unseen, draws, sims, seed, saved, chain=0, piao=0): the simulated value of each kept hand."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef native_module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, native_methods};

PyMODINIT_FUNC PyInit_native(void) {
    if (fast_keys() < 0) return NULL;
    return PyModule_Create(&native_module);
}
