/* grid_bench.c
   Side-by-side equivalence + benchmark of libgrd (original) vs
   libgrdopt (parallel/SIMD variant) for GridSort and GridLocateCell. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "griddata.h"
#include "griddata_parallel.h"
#include "adapter.h"

/* Symbols from libgrdopt that the harness calls. */
struct GridDataOpt *GridMakeOpt(void);
void GridFreeOpt(struct GridDataOpt *);
void GridSortOpt(struct GridDataOpt *);
int  GridLocateCellOpt(int npnt, struct GridGVecOpt *ptr, int index);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static struct GridData *make_synth(int n_records, int n_stations, unsigned seed) {
    struct GridData *g = GridMake();
    g->st_time = 0.0; g->ed_time = 3600.0; g->xtd = 0;
    g->stnum = n_stations;
    g->sdata = calloc(n_stations, sizeof(struct GridSVec));
    for (int i = 0; i < n_stations; i++) {
        g->sdata[i].st_id = i;
        g->sdata[i].chn   = 0;
        g->sdata[i].npnt  = n_records / n_stations;
    }
    g->vcnum = n_records;
    g->data  = calloc(n_records, sizeof(struct GridGVec));
    srand(seed);
    for (int i = 0; i < n_records; i++) {
        /* deliberately unsorted: st_id chosen randomly, index sequentially
           shuffled so qsort/parallel-sort actually do work. */
        g->data[i].st_id = rand() % n_stations;
        g->data[i].chn   = 0;
        g->data[i].index = rand() % (n_records * 4);
        g->data[i].mlat  = -90.0 + 180.0 * rand() / RAND_MAX;
        g->data[i].mlon  = -180.0 + 360.0 * rand() / RAND_MAX;
        g->data[i].azm   = 360.0 * rand() / RAND_MAX;
        g->data[i].vel.median = -1000 + 2000.0 * rand() / RAND_MAX;
        g->data[i].vel.sd     = 50.0  * rand() / RAND_MAX;
        g->data[i].pwr.median = 30.0  * rand() / RAND_MAX;
        g->data[i].pwr.sd     = 3.0   * rand() / RAND_MAX;
        g->data[i].wdt.median = 500.0 * rand() / RAND_MAX;
        g->data[i].wdt.sd     = 25.0  * rand() / RAND_MAX;
    }
    return g;
}

static struct GridDataOpt *clone_to_opt(const struct GridData *g) {
    struct GridDataOpt *o = GridMakeOpt();
    if (!o) o = calloc(1, sizeof(struct GridDataOpt));
    grid_copy_to_opt(g, o);
    return o;
}

static void bench_sort(int n_records, int n_stations, int iters) {
    double t_orig = 0.0, t_opt = 0.0;
    int equiv_failures = 0;
    for (int it = 0; it < iters; it++) {
        struct GridData    *g = make_synth(n_records, n_stations, 42 + it);
        struct GridDataOpt *o = clone_to_opt(g);

        double t0 = now_s();
        GridSort(g);
        double t1 = now_s();

        double t2 = now_s();
        GridSortOpt(o);
        double t3 = now_s();

        t_orig += (t1 - t0);
        t_opt  += (t3 - t2);

        if (grid_compare(g, o, 1e-9) != 0) equiv_failures++;

        GridFree(g);
        GridFreeOpt(o);
    }
    double speedup = (t_opt > 0) ? t_orig / t_opt : 0.0;
    printf("Sort  N=%-8d  orig=%8.4f ms  opt=%8.4f ms  speedup=%5.2fx  equiv_fail=%d/%d\n",
           n_records, 1000.0 * t_orig / iters, 1000.0 * t_opt / iters,
           speedup, equiv_failures, iters);
}

static void bench_locate(int n_records, int n_stations, int iters) {
    double t_orig = 0.0, t_opt = 0.0;
    int mismatches = 0;
    struct GridData *g = make_synth(n_records, n_stations, 17);
    GridSort(g);
    struct GridDataOpt *o = clone_to_opt(g);
    GridSortOpt(o);
    /* probe with 100 random indices that exist in the data */
    int *probe = malloc(sizeof(int) * 100);
    for (int i = 0; i < 100; i++) probe[i] = g->data[rand() % n_records].index;
    for (int it = 0; it < iters; it++) {
        double t0 = now_s();
        int r1 = 0;
        for (int i = 0; i < 100; i++) r1 += GridLocateCell(g->vcnum, g->data, probe[i]);
        double t1 = now_s();
        double t2 = now_s();
        int r2 = 0;
        for (int i = 0; i < 100; i++) r2 += GridLocateCellOpt(o->vcnum, o->data, probe[i]);
        double t3 = now_s();
        t_orig += (t1 - t0);
        t_opt  += (t3 - t2);
        if (r1 != r2) mismatches++;
    }
    double speedup = (t_opt > 0) ? t_orig / t_opt : 0.0;
    printf("Locate N=%-8d orig=%8.4f us  opt=%8.4f us  speedup=%5.2fx  mismatches=%d/%d\n",
           n_records, 1e6 * t_orig / iters, 1e6 * t_opt / iters,
           speedup, mismatches, iters);
    free(probe);
    GridFree(g);
    GridFreeOpt(o);
}

int main(void) {
    printf("=== Grid library equivalence + benchmark ===\n");
    printf("OMP_NUM_THREADS=%s\n", getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "default");
    printf("\n--- Sort (iters=5) ---\n");
    int sizes_s[] = {1000, 10000, 100000, 500000}; for (int s = 0; s < 4; s++) bench_sort(sizes_s[s], 20, 5);
    printf("\n--- LocateCell (iters=20, 100 probes/iter) ---\n");
    int sizes_l[] = {1000, 10000, 100000}; for (int s = 0; s < 3; s++) bench_locate(sizes_l[s], 20, 20);
    return 0;
}
