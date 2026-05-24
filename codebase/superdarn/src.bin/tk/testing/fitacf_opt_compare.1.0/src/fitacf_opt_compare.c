/* fitacf_opt_compare -- equivalence + benchmark harness for FITACF v3.
 *
 * F0: stands up the harness modeled on grid_opt_compare. The reference
 * path calls Fitacf() (the canonical libfitacf.3.0 entry point); a
 * future optimized path will call into a Fitacf_AVX2 / Fitacf_Array
 * variant once F3/F4 land. For now this is the timing + correctness
 * baseline.
 *
 * Synthetic ACF model per range gate r in [0, nrang):
 *   pwr0[r] = base_power * range_decay(r)
 *   acf[r][k] = pwr0[r] * exp(-w_true * t_k)
 *                       * (cos(omega_v * t_k) + i*sin(omega_v * t_k))
 *               + noise
 * where t_k = mpinc * lag[k] (seconds) and omega_v is set so the
 * recovered velocity should match v_true.
 *
 * Acceptance (F0): harness builds, runs, and reports a non-zero count
 * of valid range gates with the mean recovered velocity within
 * ~10% of v_true.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <zlib.h>   /* gzFile, needed before dmap.h */

#include "rtypes.h"
#include "dmap.h"
#include "rprm.h"
#include "rawdata.h"
#include "fitdata.h"
#include "fitblk.h"
#include "fit_structures.h"
#include "fitacftoplevel.h"
#include "fit_structures_array.h"   /* Fitacf_Array_From_Prms (F2 SoA path) */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Speed of light, m/s. Used to convert between phase-rate and velocity. */
static const double C = 2.9979e8;

/* Standard FITACF 8-pulse pattern (katscan): index of each pulse in
   units of mpinc. */
static const int PULSE_TABLE_8[8]  = {0, 14, 22, 24, 27, 31, 42, 43};
/* 17 unique lags reachable with the 8-pulse pattern (in mpinc units). */
static const int LAG_TABLE_17[18][2] = {
    { 0,  0}, {42, 43}, {22, 24}, {24, 27},
    {27, 31}, {22, 27}, {24, 31}, {14, 22},
    {22, 31}, {14, 24}, {31, 42}, {31, 43},
    {14, 27}, { 0, 14}, {27, 42}, {27, 43},
    {14, 31}, {24, 42}
};

/* gauss_noise: cheap Box-Muller. amp_db is relative to signal. */
static double gauss_noise(double sigma) {
    double u1 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return sigma * sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
}

static FITPRMS *make_synth_prms(int nrang, int mplgs, int mppul) {
    FITPRMS *p = calloc(1, sizeof(FITPRMS));
    if (!p) return NULL;

    p->channel = 0;
    p->offset  = 0;
    p->cp      = 153;
    p->xcf     = 0;
    p->tfreq   = 12000;   /* kHz */
    p->noise   = 100.0;
    p->nrang   = nrang;
    p->smsep   = 300;     /* us */
    p->nave    = 50;
    p->mplgs   = mplgs;
    p->mpinc   = 1500;    /* us */
    p->txpl    = 300;
    p->lagfr   = 1200;
    p->mppul   = mppul;
    p->bmnum   = 7;
    p->old     = 0;
    p->maxbeam = 16;
    p->bmoff   = 0.0;
    p->bmsep   = 3.24;
    p->phidiff = 1.0;
    p->tdiff   = 0.0;
    p->vdir    = 1.0;
    p->interfer[0] = 0.0;
    p->interfer[1] = -100.0;
    p->interfer[2] = 0.0;
    p->time.yr = 2026; p->time.mo = 5; p->time.dy = 25;
    p->time.hr = 0; p->time.mt = 0; p->time.sc = 0; p->time.us = 0;

    p->pulse = calloc(mppul, sizeof(int));
    for (int i = 0; i < mppul && i < 8; i++)
        p->pulse[i] = PULSE_TABLE_8[i];

    for (int n = 0; n < 2; n++) {
        p->lag[n] = calloc(mplgs + 1, sizeof(int));
        for (int k = 0; k <= mplgs; k++) {
            int idx = (k < 18) ? k : 17;
            p->lag[n][k] = LAG_TABLE_17[idx][n];
        }
    }

    p->pwr0 = calloc(nrang, sizeof(double));

    /* acfd/xcfd are indexed by Fitacf as flat [range*mplgs + lag][0/1].
       Match Allocate_Fit_Prm's contiguous arena layout: first `rows`
       slots are row pointers, the rest is the actual double[2] data. */
    size_t rows = (size_t)nrang * mplgs;
    size_t columns = 2;
    size_t bytes = (sizeof(double *) + columns * sizeof(double)) * rows;
    p->acfd = malloc(bytes); memset(p->acfd, 0, bytes);
    p->xcfd = malloc(bytes); memset(p->xcfd, 0, bytes);
    p->acfd[0] = (double *)(p->acfd + rows);
    p->xcfd[0] = (double *)(p->xcfd + rows);
    for (size_t is = 1; is < rows; is++) {
        p->acfd[is] = (double *)(p->acfd[0] + is * columns);
        p->xcfd[is] = (double *)(p->xcfd[0] + is * columns);
    }
    return p;
}

static void free_synth_prms(FITPRMS *p) {
    if (!p) return;
    free(p->acfd); free(p->xcfd);
    free(p->pwr0);
    free(p->lag[0]); free(p->lag[1]);
    free(p->pulse);
    free(p);
}

/* Populate ACF data with a decay+rotate model so Fitacf has something
   physically plausible to fit. v_true in m/s, w_true in m/s (spectral
   width as velocity), pwr0_base in linear units (not dB). */
static void populate_synth_acf(FITPRMS *p, double v_true, double w_true,
                               double pwr0_base, double noise_sigma) {
    double freq_hz = p->tfreq * 1e3;
    double k_wave = 2.0 * 2.0 * M_PI * freq_hz / C;  /* 2k for backscatter */
    double mpinc_s = p->mpinc * 1e-6;

    for (int r = 0; r < p->nrang; r++) {
        /* Range-dependent power: roughly r^-4 plus a small range
           cluster of strong scatter around mid-range. */
        double rng_factor = 1.0 / pow((r + 5.0) / 25.0, 4);
        p->pwr0[r] = pwr0_base * rng_factor + gauss_noise(noise_sigma);
        if (p->pwr0[r] < p->noise * 2.0) p->pwr0[r] = p->noise * 2.0;

        for (int k = 0; k < p->mplgs; k++) {
            double t = (double)p->lag[0][k] * mpinc_s;
            double decay = exp(-k_wave * w_true * t);
            double phase = k_wave * v_true * t;
            size_t idx = (size_t)r * p->mplgs + k;
            p->acfd[idx][0] = p->pwr0[r] * decay * cos(phase) + gauss_noise(noise_sigma);
            p->acfd[idx][1] = p->pwr0[r] * decay * sin(phase) + gauss_noise(noise_sigma);
            p->xcfd[idx][0] = p->acfd[idx][0];
            p->xcfd[idx][1] = p->acfd[idx][1];
        }
    }
}

static double ms_since(struct timespec *t0) {
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0->tv_sec) * 1000.0
         + (t1.tv_nsec - t0->tv_nsec) * 1e-6;
}

enum fitacf_path { PATH_REF = 0, PATH_ARRAY = 1 };

/* Run one of the two paths once, return wallclock ms + qflg=1 count
   and the per-range mean v/w for that path. */
static double run_fitacf_path(enum fitacf_path path,
                              FITPRMS *prms, struct FitData *fit,
                              int num_threads,
                              int *out_ngood, double *out_v, double *out_w) {
    /* Both code paths chatter to stderr (preprocessing.c phase unwrap
       debug, and Fitacf_Array printf logs). Mute around the timed call. */
    int saved_stderr = dup(2), saved_stdout = dup(1);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
        dup2(devnull, 2); dup2(devnull, 1); close(devnull);
    }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (path == PATH_REF) {
        Fitacf(prms, fit, 0);
    } else {
        Fitacf_Array_From_Prms(prms, fit, PROCESS_MODE_ARRAYS, num_threads);
    }

    double ms = ms_since(&t0);

    if (saved_stderr >= 0) { dup2(saved_stderr, 2); close(saved_stderr); }
    if (saved_stdout >= 0) { dup2(saved_stdout, 1); close(saved_stdout); }

    int ngood = 0;
    double v_sum = 0.0, w_sum = 0.0;
    if (fit->rng) {
        for (int r = 0; r < prms->nrang; r++) {
            if (fit->rng[r].qflg == 1) {
                ngood++;
                v_sum += fit->rng[r].v;
                w_sum += fit->rng[r].w_l;
            }
        }
    }
    if (out_ngood) *out_ngood = ngood;
    if (out_v)     *out_v = (ngood > 0) ? v_sum / ngood : 0.0;
    if (out_w)     *out_w = (ngood > 0) ? w_sum / ngood : 0.0;
    return ms;
}

static int bench_fitacf(int nrang, int mplgs, int mppul,
                        double v_true, double w_true, int iters,
                        int num_threads) {
    int    good_ref = 0,  good_arr = 0;
    double ms_ref = 0,    ms_arr   = 0;
    double v_ref = 0, w_ref = 0, v_arr = 0, w_arr = 0;

    for (int it = 0; it < iters; it++) {
        /* Identical seed → identical synthetic input for both paths. */
        srand(0xACF0 + it);
        FITPRMS *prms_ref = make_synth_prms(nrang, mplgs, mppul);
        if (!prms_ref) { fprintf(stderr, "make_synth_prms failed\n"); return -1; }
        populate_synth_acf(prms_ref, v_true, w_true, 2000.0, 5.0);
        srand(0xACF0 + it);
        FITPRMS *prms_arr = make_synth_prms(nrang, mplgs, mppul);
        populate_synth_acf(prms_arr, v_true, w_true, 2000.0, 5.0);

        struct FitData *fit_ref = FitMake();
        FitSetRng(fit_ref, nrang); FitSetXrng(fit_ref, nrang); FitSetElv(fit_ref, nrang);
        struct FitData *fit_arr = FitMake();
        FitSetRng(fit_arr, nrang); FitSetXrng(fit_arr, nrang); FitSetElv(fit_arr, nrang);

        int g; double v, w;
        ms_ref += run_fitacf_path(PATH_REF,   prms_ref, fit_ref, num_threads, &g, &v, &w);
        good_ref += g; v_ref += v; w_ref += w;
        ms_arr += run_fitacf_path(PATH_ARRAY, prms_arr, fit_arr, num_threads, &g, &v, &w);
        good_arr += g; v_arr += v; w_arr += w;

        FitFree(fit_ref); FitFree(fit_arr);
        free_synth_prms(prms_ref); free_synth_prms(prms_arr);
    }
    ms_ref /= iters; ms_arr /= iters;
    v_ref  /= iters; w_ref  /= iters;
    v_arr  /= iters; w_arr  /= iters;
    int avg_good_ref = good_ref / iters;
    int avg_good_arr = good_arr / iters;

    double speedup = (ms_arr > 0) ? ms_ref / ms_arr : 0.0;

    printf("nrang=%-3d v_true=%6.1f w_true=%5.1f  "
           "ref: ms=%6.2f good=%2d v=%7.1f w=%5.1f  "
           "arr: ms=%6.2f good=%2d v=%7.1f w=%5.1f  "
           "speedup=%5.2fx\n",
           nrang, v_true, w_true,
           ms_ref, avg_good_ref, v_ref, w_ref,
           ms_arr, avg_good_arr, v_arr, w_arr,
           speedup);

    if (avg_good_ref < 1 && avg_good_arr < 1) {
        printf("  FAIL: neither path returned any valid ranges\n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int iters = 5, threads = 4;
    for (int i = 1; i < argc; i++) {
        if      (strncmp(argv[i], "--iters=",   8) == 0) iters   = atoi(argv[i] + 8);
        else if (strncmp(argv[i], "--threads=", 10) == 0) threads = atoi(argv[i] + 10);
    }

    setvbuf(stdout, NULL, _IOLBF, 0);

    printf("=== FITACF v3 reference vs array-SoA benchmark ===\n");
    printf("iters=%d  threads=%d\n", iters, threads);
    printf("                                            "
           "reference path                  "
           "array-SoA path (Fitacf_Array_From_Prms)\n\n");

    int failures = 0;
    failures += (bench_fitacf( 75, 18, 8,  100.0,  50.0, iters, threads) != 0);
    failures += (bench_fitacf( 75, 18, 8,  500.0, 100.0, iters, threads) != 0);
    failures += (bench_fitacf( 75, 18, 8, -300.0,  80.0, iters, threads) != 0);
    failures += (bench_fitacf(150, 18, 8,  200.0,  60.0, iters, threads) != 0);

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures == 0 ? 0 : 1;
}
