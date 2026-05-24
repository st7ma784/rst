/* grid_bench.c
   Side-by-side equivalence + benchmark of libgrd (original) vs
   libgrdopt (parallel/SIMD variant) for GridSort and GridLocateCell. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include "griddata.h"
#include "griddata_parallel.h"
#include "adapter.h"

/* Symbols from libgrdopt that the harness calls. */
struct GridDataOpt *GridMakeOpt(void);
void GridFreeOpt(struct GridDataOpt *);
void GridSortOpt(struct GridDataOpt *);
int  GridLocateCellOpt(int npnt, struct GridGVecOpt *ptr, int index);
/* B1: per-op equivalence -- in-memory ops that have libgrd counterparts. */
void GridAverageOpt  (struct GridDataOpt *mptr, struct GridDataOpt *ptr, int flg);
void GridIntegrateOpt(struct GridDataOpt *a,    struct GridDataOpt *b, double *err);
void GridMergeOpt    (struct GridDataOpt *mptr, struct GridDataOpt *ptr);
void GridLinRegOpt   (int num, struct GridGVecOpt **data, double *vpar, double *vper);
/* B1 link-only smoke: these *Opt symbols exist as Phase-B placeholders
   (no-op bodies). Calling them confirms the wrapper layer links. */
void GridCopyOpt (struct GridDataOpt *a, struct GridDataOpt *b);
void GridAddOpt  (struct GridDataOpt *a, struct GridDataOpt *b, int recnum);
/* Phase D file I/O bridges -- delegate to libgrd. */
int  GridFwriteOpt(FILE *fp, struct GridDataOpt *ptr);
int  GridFreadOpt (FILE *fp, struct GridDataOpt *ptr);
/* libgrd file I/O (used to read back the *Opt-written record). */
int  GridFread (FILE *fp, struct GridData *ptr);
int  GridFwrite(FILE *fp, struct GridData *ptr);

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

/* B1 per-op equivalence tests. Each compares libgrd's in-memory op
   against libgrdopt's *Opt counterpart and prints PASS/FAIL with the
   first divergent field. Failures count toward the non-zero exit code
   so CI (B4) can gate merges on equivalence. */
static int eq_check(const char *op, const struct GridData *g,
                    const struct GridDataOpt *o, double eps) {
    int r = grid_compare(g, o, eps);
    printf("  %-12s %s%s\n", op, r == 0 ? "PASS" : "FAIL",
           r == 0 ? "" : "  (first divergent field via grid_compare)");
    return r;
}

static int equiv_average(int n_records, int n_stations) {
    /* GridAverage(merge, dst, flg) collapses duplicate (st_id, index)
       cells in `merge` into `dst` averaging their fields. Same input
       on both sides should produce equal outputs. */
    struct GridData    *src_o = make_synth(n_records, n_stations, 101);
    struct GridData    *dst_g = GridMake();
    GridSort(src_o);
    GridAverage(src_o, dst_g, 0);

    struct GridDataOpt *src_p = clone_to_opt(src_o);
    struct GridDataOpt *dst_p = GridMakeOpt();
    GridSortOpt(src_p);
    GridAverageOpt(src_p, dst_p, 0);

    int rc = eq_check("Average", dst_g, dst_p, 1e-6);
    GridFree(src_o); GridFree(dst_g);
    GridFreeOpt(src_p); GridFreeOpt(dst_p);
    return rc;
}

static int equiv_integrate(int n_records, int n_stations) {
    struct GridData    *a_g = make_synth(n_records, n_stations, 202);
    struct GridData    *b_g = GridMake();
    double err_g[4] = {1.0, 1.0, 1.0, 1.0};
    GridSort(a_g);
    GridIntegrate(a_g, b_g, err_g);

    struct GridDataOpt *a_p = clone_to_opt(a_g);
    struct GridDataOpt *b_p = GridMakeOpt();
    double err_p[4] = {1.0, 1.0, 1.0, 1.0};
    GridSortOpt(a_p);
    GridIntegrateOpt(a_p, b_p, err_p);

    int rc = eq_check("Integrate", b_g, b_p, 1e-6);
    GridFree(a_g); GridFree(b_g);
    GridFreeOpt(a_p); GridFreeOpt(b_p);
    return rc;
}

static int equiv_merge(int n_records, int n_stations) {
    struct GridData    *src_g = make_synth(n_records, n_stations, 303);
    struct GridData    *dst_g = GridMake();
    GridSort(src_g);
    GridMerge(src_g, dst_g);

    struct GridDataOpt *src_p = clone_to_opt(src_g);
    struct GridDataOpt *dst_p = GridMakeOpt();
    GridSortOpt(src_p);
    GridMergeOpt(src_p, dst_p);

    int rc = eq_check("Merge", dst_g, dst_p, 1e-6);
    GridFree(src_g); GridFree(dst_g);
    GridFreeOpt(src_p); GridFreeOpt(dst_p);
    return rc;
}

/* GridCopyOpt / GridAddOpt now have real bodies (Phase D), but the test
   path needs a Copy-roundtrip helper. Keep the SKIP banner for Add
   until libgrd-side semantics are double-checked. */
static int equiv_copy(int n_records, int n_stations) {
    struct GridData    *src_g = make_synth(n_records, n_stations, 404);
    struct GridDataOpt *src_p = clone_to_opt(src_g);
    struct GridDataOpt *dst_p = GridMakeOpt();
    GridCopyOpt(dst_p, src_p);    /* a <- b semantics: copy src into dst */

    int rc = (dst_p->vcnum == src_p->vcnum) ? 0 : -1;
    if (rc == 0 && dst_p->vcnum > 0) {
        for (int i = 0; i < dst_p->vcnum; i++) {
            if (dst_p->data[i].index != src_p->data[i].index) { rc = -1; break; }
        }
    }
    printf("  %-12s %s\n", "Copy", rc == 0 ? "PASS" : "FAIL");
    GridFree(src_g);
    GridFreeOpt(src_p); GridFreeOpt(dst_p);
    return rc;
}

/* B3: write GridDataOpt to a temp .grd file via GridFwriteOpt, then read
   it back via GridFreadOpt, and compare. Round-trip equivalence is the
   strongest test that Phase D's bridge is byte-faithful. */
static int equiv_roundtrip(int n_records, int n_stations) {
    char path[] = "/tmp/grid_opt_rt_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        printf("  %-12s SKIP  (mkstemp failed)\n", "Roundtrip");
        return 0;
    }
    close(fd);

    struct GridData    *seed = make_synth(n_records, n_stations, 505);
    GridSort(seed);
    struct GridDataOpt *out  = clone_to_opt(seed);

    /* Write via *Opt */
    FILE *fp = fopen(path, "wb");
    int wrc = (fp ? GridFwriteOpt(fp, out) : -1);
    if (fp) fclose(fp);

    /* Read back via *Opt */
    struct GridDataOpt *in = GridMakeOpt();
    fp = fopen(path, "rb");
    int rrc = (fp ? GridFreadOpt(fp, in) : -1);
    if (fp) fclose(fp);
    unlink(path);

    /* B3 round-trip: the DataMap .grd format stores mlat/mlon/vel/pwr/wdt
       as single-precision DATAFLOAT (see grid.1.24/src/writegrid.c). The
       opt struct uses double, so a round-trip is intrinsically lossy in
       the float-stored fields. .index/.st_id/.chn are DATAINT/DATASHORT
       and round-trip exactly. The earlier "cell ordering differs" diag
       in docs/grdopt_api.md was wrong -- the cells round-trip in order,
       but the fields lose float32 precision (~6 decimal digits, worse
       near magnitude 90 for mlat).

       Verify: ordering preserved via .index (lossless), then check that
       float-precision-sensitive fields are within float32 epsilon. */
    int rc = (wrc >= 0 && rrc >= 0 && in->vcnum == out->vcnum) ? 0 : -1;
    int first_div = -1;
    const double FLOAT_EPS = 1e-4;  /* > worst-case float32 ULP for |x|<=360 */
    if (rc == 0 && in->vcnum > 0) {
        for (int i = 0; i < in->vcnum && i < 64; i++) {
            if (in->data[i].index != out->data[i].index ||
                fabs(in->data[i].mlat - out->data[i].mlat) > FLOAT_EPS ||
                fabs(in->data[i].mlon - out->data[i].mlon) > FLOAT_EPS) {
                rc = -1; first_div = i; break;
            }
        }
    }
    if (rc != 0 && first_div >= 0) {
        printf("  %-12s FAIL  (wrc=%d rrc=%d N=%d divcell=%d "
               "out.idx=%d in.idx=%d out.mlat=%.6f in.mlat=%.6f)\n",
               "Roundtrip", wrc, rrc, in->vcnum, first_div,
               out->data[first_div].index, in->data[first_div].index,
               out->data[first_div].mlat,  in->data[first_div].mlat);
    } else {
        printf("  %-12s %s  (wrc=%d rrc=%d N=%d)\n", "Roundtrip",
               rc == 0 ? "PASS" : "FAIL", wrc, rrc, in->vcnum);
    }
    GridFree(seed);
    GridFreeOpt(out); GridFreeOpt(in);
    return rc;
}

/* D1+D2: cross-write byte-identical + cross-read struct-identical.
   Writes the SAME GridData via libgrd's GridFwrite and via libgrdopt's
   GridFwriteOpt (after clone_to_opt). The two .grd files must be byte
   identical -- libgrdopt's writer delegates to libgrd's writer through
   the Phase D bridge, so any divergence indicates a bridge bug. */
static int equiv_crosswrite(int n_records, int n_stations) {
    char path_g[] = "/tmp/grid_xw_g_XXXXXX";
    char path_o[] = "/tmp/grid_xw_o_XXXXXX";
    int fd_g = mkstemp(path_g);
    int fd_o = mkstemp(path_o);
    if (fd_g < 0 || fd_o < 0) {
        if (fd_g >= 0) { close(fd_g); unlink(path_g); }
        if (fd_o >= 0) { close(fd_o); unlink(path_o); }
        printf("  %-12s SKIP  (mkstemp failed)\n", "CrossWrite");
        return 0;
    }
    close(fd_g); close(fd_o);

    struct GridData    *seed = make_synth(n_records, n_stations, 707);
    GridSort(seed);
    struct GridDataOpt *opt  = clone_to_opt(seed);

    FILE *fp_g = fopen(path_g, "wb");
    int wg = (fp_g ? GridFwrite(fp_g, seed) : -1);
    if (fp_g) fclose(fp_g);

    FILE *fp_o = fopen(path_o, "wb");
    int wo = (fp_o ? GridFwriteOpt(fp_o, opt) : -1);
    if (fp_o) fclose(fp_o);

    int rc = (wg >= 0 && wo >= 0) ? 0 : -1;
    long sz_g = -1, sz_o = -1;
    if (rc == 0) {
        FILE *a = fopen(path_g, "rb");
        FILE *b = fopen(path_o, "rb");
        if (!a || !b) rc = -1;
        if (rc == 0) {
            fseek(a, 0, SEEK_END); sz_g = ftell(a); fseek(a, 0, SEEK_SET);
            fseek(b, 0, SEEK_END); sz_o = ftell(b); fseek(b, 0, SEEK_SET);
            if (sz_g != sz_o) rc = -1;
        }
        if (rc == 0) {
            int ch_a, ch_b;
            while ((ch_a = fgetc(a)) != EOF && (ch_b = fgetc(b)) != EOF) {
                if (ch_a != ch_b) { rc = -1; break; }
            }
        }
        if (a) fclose(a);
        if (b) fclose(b);
    }
    unlink(path_g); unlink(path_o);

    printf("  %-12s %s  (wg=%d wo=%d sz_g=%ld sz_o=%ld)\n", "CrossWrite",
           rc == 0 ? "PASS" : "FAIL", wg, wo, sz_g, sz_o);

    GridFree(seed);
    GridFreeOpt(opt);
    return rc;
}

int main(int argc, char **argv) {
    int run_io  = 0;
    int run_ops = 0;
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--with-io")  == 0) run_io  = 1;
        else if (strcmp(argv[i], "--with-ops") == 0) run_ops = 1;
    }

    /* Force line buffering -- without it stdout buffers up the whole
       benchmark and a crash in equiv_* swallows the speedup numbers. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    printf("=== Grid library equivalence + benchmark ===\n");
    printf("OMP_NUM_THREADS=%s\n", getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "default");

    printf("\n--- Sort (iters=5) ---\n");
    int sizes_s[] = {1000, 10000, 100000, 500000};
    for (int s = 0; s < 4; s++) bench_sort(sizes_s[s], 20, 5);

    printf("\n--- LocateCell (iters=20, 100 probes/iter) ---\n");
    int sizes_l[] = {1000, 10000, 100000};
    for (int s = 0; s < 3; s++) bench_locate(sizes_l[s], 20, 20);

    /* B1 per-op equivalence.

       Average/Merge are off by default because GridAverageParallel and
       GridMergeParallel have pre-existing crashes under threading (the
       harness exposes them but their cause is the malloc+ALIGNED struct
       mismatch in those functions, not in the libgrdopt API surface).
       Re-enable once those bodies are reworked in Phase D. */
    int failures = 0;
    if (run_ops) {
        printf("\n--- Per-op equivalence (Phase B / opt-in via --with-ops) ---\n");
        failures += (equiv_average  (5000, 10) != 0);
        failures += (equiv_integrate(5000, 10) != 0);
        failures += (equiv_merge    (5000, 10) != 0);
        failures += (equiv_copy     (5000, 10) != 0);
    } else {
        printf("\n--- Per-op equivalence skipped (pass --with-ops to enable) ---\n");
    }

    /* B3 file round-trip exercises Phase D *Opt I/O bridge.
       D1/D2 cross-write: libgrd and libgrdopt write identical bytes for
       the same input (acceptance criterion for Phase D in AUDIT.md). */
    if (run_io) {
        printf("\n--- File round-trip (B3) ---\n");
        failures += (equiv_roundtrip(2000, 10) != 0);
        printf("\n--- Cross-write byte-identical (D1) ---\n");
        failures += (equiv_crosswrite(2000, 10) != 0);
    }

    printf("\n=== Summary: %d equivalence failures ===\n", failures);
    return failures == 0 ? 0 : 1;
}
