/**
 * @file cudarst_fitacf.c
 * @brief FITACF v3.0 compatible interface implementation
 */

#define _GNU_SOURCE  /* enables M_PI, CLOCK_MONOTONIC, alloca */
#include "cudarst.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* CUDA kernel declarations (implemented in .cu file) */
#ifdef __NVCC__
extern "C" {
    int cuda_fitacf_process_ranges(const float *acf_real, const float *acf_imag,
                                   int nrang, int mplgs, int nave,
                                   float vel_factor, float lag_dt, float noise_pwr,
                                   float *power, float *velocity, float *velocity_err,
                                   float *width, float *width_err,
                                   float *width_sigma, float *width_sigma_err,
                                   float *phase);
}
#endif

/* Velocity/width conversion factor: c / (4π * tfreq_hz * lag_time_s).
 * Returns 0 if prm values are invalid (caller should fall back to a safe default). */
static float compute_vel_factor(const cudarst_fitacf_prm_t *prm)
{
    if (!prm || prm->tfreq <= 0 || prm->mpinc <= 0) return 0.0f;
    float tfreq_hz  = prm->tfreq  * 1000.0f;      /* kHz → Hz */
    float lag_time  = prm->mpinc  * 1e-6f;         /* µs  → s  */
    return 3e8f / (4.0f * (float)M_PI * tfreq_hz * lag_time);
}

/* CPU fallback implementation */
static cudarst_error_t cpu_fitacf_process(const cudarst_fitacf_prm_t *prm,
                                          const cudarst_fitacf_raw_t *raw,
                                          cudarst_fitacf_fit_t *fit);

cudarst_error_t cudarst_fitacf_process(const cudarst_fitacf_prm_t *prm,
                                       const cudarst_fitacf_raw_t *raw,
                                       cudarst_fitacf_fit_t *fit)
{
    if (!prm || !raw || !fit) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    cudarst_error_t result = CUDARST_ERROR_PROCESSING_FAILED;
    
    /* Attempt CUDA processing first if available */
    if (cudarst_is_cuda_available()) {
#ifdef __NVCC__
        struct timespec cuda_start, cuda_end;
        clock_gettime(CLOCK_MONOTONIC, &cuda_start);
        
        float vel_factor = compute_vel_factor(prm);
        if (vel_factor == 0.0f) vel_factor = 1326.0f;
        float lag_dt  = (prm->mpinc > 0) ? prm->mpinc * 1e-6f : 1500e-6f;
        int   nave    = (prm->nave  > 0) ? prm->nave  : 20;

        int cuda_result = cuda_fitacf_process_ranges(
            raw->acfd, raw->acfd_imag,
            raw->nrang, raw->mplgs, nave,
            vel_factor, lag_dt, raw->noise_pwr,
            fit->pwr0,
            fit->v,   fit->v_e,
            fit->w_l, fit->w_l_e,
            fit->w_s, fit->w_s_e,
            fit->phi0
        );

        clock_gettime(CLOCK_MONOTONIC, &cuda_end);
        double cuda_time = (cuda_end.tv_sec - cuda_start.tv_sec) * 1000.0
                         + (cuda_end.tv_nsec - cuda_start.tv_nsec) / 1000000.0;
        cudarst_performance_t perf;
        cudarst_get_performance(&perf);
        perf.cuda_time_ms += cuda_time;
        perf.cuda_used = true;

        if (cuda_result == 0) {
            /* Post-process: ground scatter, elevation, slist, XCF via CPU */
            result = cpu_fitacf_process(prm, raw, fit);
        } else {
            fprintf(stderr, "CUDArst: CUDA FITACF failed, falling back to CPU\n");
            result = cpu_fitacf_process(prm, raw, fit);
        }
#endif
    } else {
        /* CPU-only processing */
        result = cpu_fitacf_process(prm, raw, fit);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    /* Update performance counters */
    cudarst_performance_t perf;
    cudarst_get_performance(&perf);
    perf.total_time_ms += total_time;
    
    return result;
}

cudarst_fitacf_raw_t* cudarst_fitacf_raw_alloc(int nrang, int mplgs)
{
    cudarst_fitacf_raw_t *raw = malloc(sizeof(cudarst_fitacf_raw_t));
    if (!raw) return NULL;
    
    raw->nrang = nrang;
    raw->mplgs = mplgs;
    
    size_t acf_size = nrang * mplgs * sizeof(float);
    
    raw->acfd = cudarst_malloc(acf_size);
    raw->acfd_imag = cudarst_malloc(acf_size);
    raw->xcfd = cudarst_malloc(acf_size);
    raw->xcfd_imag = cudarst_malloc(acf_size);
    
    if (!raw->acfd || !raw->acfd_imag || !raw->xcfd || !raw->xcfd_imag) {
        cudarst_fitacf_raw_free(raw);
        return NULL;
    }
    
    /* Initialize to zero */
    memset(raw->acfd, 0, acf_size);
    memset(raw->acfd_imag, 0, acf_size);
    memset(raw->xcfd, 0, acf_size);
    memset(raw->xcfd_imag, 0, acf_size);
    raw->noise_pwr = 0.0f;

    return raw;
}

cudarst_fitacf_fit_t* cudarst_fitacf_fit_alloc(int nrang)
{
    cudarst_fitacf_fit_t *fit = malloc(sizeof(cudarst_fitacf_fit_t));
    if (!fit) return NULL;
    
    fit->nrang = nrang;
    
    size_t array_size = nrang * sizeof(float);
    
    /* Allocate all arrays */
    fit->pwr0 = cudarst_malloc(array_size);
    fit->slist = cudarst_malloc(array_size);
    fit->v = cudarst_malloc(array_size);
    fit->v_e = cudarst_malloc(array_size);
    fit->p_l = cudarst_malloc(array_size);
    fit->p_l_e = cudarst_malloc(array_size);
    fit->p_s = cudarst_malloc(array_size);
    fit->p_s_e = cudarst_malloc(array_size);
    fit->w_l = cudarst_malloc(array_size);
    fit->w_l_e = cudarst_malloc(array_size);
    fit->w_s = cudarst_malloc(array_size);
    fit->w_s_e = cudarst_malloc(array_size);
    fit->phi0 = cudarst_malloc(array_size);
    fit->phi0_e = cudarst_malloc(array_size);
    fit->elv        = cudarst_malloc(array_size);
    fit->elv_error  = cudarst_malloc(array_size);
    fit->elv_fitted = cudarst_malloc(array_size);
    fit->x_qflg = cudarst_malloc(array_size);
    fit->x_gflg = cudarst_malloc(array_size);
    fit->x_p_l = cudarst_malloc(array_size);
    fit->x_p_l_e = cudarst_malloc(array_size);
    fit->x_p_s = cudarst_malloc(array_size);
    fit->x_p_s_e = cudarst_malloc(array_size);
    fit->x_v = cudarst_malloc(array_size);
    fit->x_v_e = cudarst_malloc(array_size);
    fit->x_w_l = cudarst_malloc(array_size);
    fit->x_w_l_e = cudarst_malloc(array_size);
    fit->x_w_s = cudarst_malloc(array_size);
    fit->x_w_s_e = cudarst_malloc(array_size);
    fit->x_phi0 = cudarst_malloc(array_size);
    fit->x_phi0_e = cudarst_malloc(array_size);
    
    /* Check allocation success */
    if (!fit->pwr0 || !fit->slist || !fit->v || !fit->v_e || !fit->p_l || !fit->p_l_e ||
        !fit->p_s || !fit->p_s_e || !fit->w_l || !fit->w_l_e || !fit->w_s || !fit->w_s_e ||
        !fit->phi0 || !fit->phi0_e || !fit->elv || !fit->elv_error || !fit->elv_fitted ||
        !fit->x_qflg || !fit->x_gflg || !fit->x_p_l || !fit->x_p_l_e || !fit->x_p_s ||
        !fit->x_p_s_e || !fit->x_v || !fit->x_v_e || !fit->x_w_l || !fit->x_w_l_e ||
        !fit->x_w_s || !fit->x_w_s_e || !fit->x_phi0 || !fit->x_phi0_e) {
        cudarst_fitacf_fit_free(fit);
        return NULL;
    }
    
    return fit;
}

void cudarst_fitacf_raw_free(cudarst_fitacf_raw_t *raw)
{
    if (!raw) return;
    
    cudarst_free(raw->acfd);
    cudarst_free(raw->acfd_imag);
    cudarst_free(raw->xcfd);
    cudarst_free(raw->xcfd_imag);
    free(raw);
}

void cudarst_fitacf_fit_free(cudarst_fitacf_fit_t *fit)
{
    if (!fit) return;
    
    cudarst_free(fit->pwr0);
    cudarst_free(fit->slist);
    cudarst_free(fit->v);
    cudarst_free(fit->v_e);
    cudarst_free(fit->p_l);
    cudarst_free(fit->p_l_e);
    cudarst_free(fit->p_s);
    cudarst_free(fit->p_s_e);
    cudarst_free(fit->w_l);
    cudarst_free(fit->w_l_e);
    cudarst_free(fit->w_s);
    cudarst_free(fit->w_s_e);
    cudarst_free(fit->phi0);
    cudarst_free(fit->phi0_e);
    cudarst_free(fit->elv);
    cudarst_free(fit->elv_error);
    cudarst_free(fit->elv_fitted);
    cudarst_free(fit->x_qflg);
    cudarst_free(fit->x_gflg);
    cudarst_free(fit->x_p_l);
    cudarst_free(fit->x_p_l_e);
    cudarst_free(fit->x_p_s);
    cudarst_free(fit->x_p_s_e);
    cudarst_free(fit->x_v);
    cudarst_free(fit->x_v_e);
    cudarst_free(fit->x_w_l);
    cudarst_free(fit->x_w_l_e);
    cudarst_free(fit->x_w_s);
    cudarst_free(fit->x_w_s_e);
    cudarst_free(fit->x_phi0);
    cudarst_free(fit->x_phi0_e);
    free(fit);
}

/* RST constants for ground scatter detection (fitacf.2.5/src/ground_scatter.c) */
#define GS_VMAX 30.0f   /* m/s — maximum ground-scatter velocity              */
#define GS_WMAX 90.0f   /* m/s — spectral width above which scatter is ionos. */

/* CPU fallback — implements RST v3.0-compatible algorithms.
 *
 * Fix 1: one-parameter weighted LS velocity over all valid lags
 *         (RST: one_param_straight_line_fit, leastsquares.c)
 * Fix 2: two-pass weighted LS lambda width over all valid lags
 *         (RST: Power_Fits, fitting.c)
 * Fix 4: quadratic log fit for sigma (Gaussian) spectral width
 *         (RST: Sigma_Fits, leastsquares.c)
 * Fix 2: ground scatter via RST V/W line criterion
 *         (RST: set_gsct, determinations.c; GS_VMAX/GS_WMAX from ground_scatter.c)
 * Fix 3: elevation with cos(φ_beam) + cable-delay correction
 *         (RST: elevation.1.0/src/elevation.c)
 * Fix 5: no pre-fit noise subtraction (RST does not subtract noise in fits,
 *         only in the final dB conversion of p_l/p_s)                        */
static cudarst_error_t cpu_fitacf_process(const cudarst_fitacf_prm_t *prm,
                                          const cudarst_fitacf_raw_t *raw,
                                          cudarst_fitacf_fit_t *fit)
{
    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);

    float vel_factor = compute_vel_factor(prm);
    if (vel_factor == 0.0f) vel_factor = 1326.0f;

    float lag_dt  = (prm && prm->mpinc > 0) ? prm->mpinc * 1e-6f : 1500e-6f;
    int   nave    = (prm && prm->nave  > 0) ? prm->nave  : 20;
    int   mplgs   = raw->mplgs;

    /* ── Beam-direction cosine for elevation (Fix 3) ─────────────────────────
     * φ_beam = (beam - centre_beam) * beam_sep_rad
     * cos_phi ≈ cos(φ_beam); for narrow beams (< 60°) this is close to 1.   */
    float beam_sep_deg = (prm && prm->beam_sep > 0.0f) ? prm->beam_sep : 3.24f;
    float centre_beam  = 7.5f;   /* standard 16-beam radar */
    float phi_beam_rad = (prm ? (prm->bmnum - centre_beam) * beam_sep_deg : 0.0f)
                         * (float)M_PI / 180.0f;
    float cos_phi      = cosf(phi_beam_rad);
    if (cos_phi < 0.1f) cos_phi = 0.1f;   /* clamp to prevent divide-by-zero */

    /* Cable-delay phase shift: Δχ = -2π·f·tdiff·1e-6 (RST: elevation.c) */
    float f_hz     = (prm && prm->tfreq > 0) ? prm->tfreq * 1000.0f : 12e6f;
    float tdiff_us = (prm) ? prm->tdiff : 0.0f;
    float dchi_cable = -2.0f * (float)M_PI * f_hz * tdiff_us * 1e-6f;

    for (int r = 0; r < raw->nrang; r++) {
        int base = r * mplgs;

        /* ── Lag-0 power ──────────────────────────────────────────────────── */
        float pwr0 = sqrtf(raw->acfd[base] * raw->acfd[base]
                         + raw->acfd_imag[base] * raw->acfd_imag[base]);
        fit->pwr0[r]  = pwr0;
        fit->slist[r] = (float)r;
        fit->phi0[r]  = atan2f(raw->acfd_imag[base], raw->acfd[base]);

        if (pwr0 <= 0.0f || mplgs < 2) {
            fit->v[r] = fit->w_l[r] = fit->w_s[r] = 0.0f;
            fit->p_l[r] = 0.0f;
            goto fill_errors;
        }

        /* ── Fix 1: one-param weighted LS velocity ───────────────────────────
         * phase[l] = b * l  (forced through origin)
         * w = |R(l)|²  (Pass-1, amplitude-squared)
         * b (rad/lag) → v = b * vel_factor                                   */
        {
            float S_tt = 0.0f, S_ty = 0.0f;
            int   n_v  = 0;

            for (int l = 1; l < mplgs; l++) {
                int idx = base + l;
                float re = raw->acfd[idx], im = raw->acfd_imag[idx];
                float mag = sqrtf(re*re + im*im);
                if (mag < 1e-10f) continue;

                float dphi = atan2f(im, re) - fit->phi0[r];
                if (dphi >  (float)M_PI) dphi -= 2.0f * (float)M_PI;
                if (dphi < -(float)M_PI) dphi += 2.0f * (float)M_PI;

                float w = mag * mag;   /* amplitude-squared weight */
                float t = (float)l;
                S_tt += w * t * t;
                S_ty += w * t * dphi;
                n_v++;
            }

            if (n_v >= 1 && S_tt > 0.0f) {
                float b   = S_ty / S_tt;
                fit->v[r]   = b * vel_factor;
                fit->v_e[r] = sqrtf(1.0f / S_tt) * vel_factor;
            } else {
                fit->v[r] = fit->v_e[r] = 0.0f;
            }
        }

        /* ── Fix 2: two-pass weighted LS lambda width ────────────────────────
         * Pass 1: w = |R|²  → slope b1
         * Pass 2: w = 1/σ_log² (σ_log = 10% fallback) → error σ_b
         * w_l = |b| * vel_factor                                              */
        {
            /* Pass 1 */
            float Stt1 = 0.0f, Sty1 = 0.0f;
            for (int l = 1; l < mplgs; l++) {
                int idx = base + l;
                float re = raw->acfd[idx], im = raw->acfd_imag[idx];
                float mag = sqrtf(re*re + im*im);
                if (mag < 1e-10f || mag >= pwr0) continue;
                float w = mag * mag;
                float t = (float)l;
                Stt1 += w * t * t;
                Sty1 += w * t * logf(mag);
            }
            float b1 = (Stt1 > 0.0f) ? Sty1 / Stt1 : 0.0f;

            /* Pass 2 (log-corrected weights; use 10% relative as fallback) */
            float Stt2 = 0.0f, Sty2 = 0.0f;
            for (int l = 1; l < mplgs; l++) {
                int idx = base + l;
                float re = raw->acfd[idx], im = raw->acfd_imag[idx];
                float mag = sqrtf(re*re + im*im);
                if (mag < 1e-10f || mag >= pwr0) continue;
                float sig_log = 0.1f;   /* 10% log-amplitude error fallback */
                float w = 1.0f / (sig_log * sig_log);
                float t = (float)l;
                Stt2 += w * t * t;
                Sty2 += w * t * logf(mag);
            }
            float b2      = (Stt2 > 0.0f) ? Sty2 / Stt2 : b1;
            float sigma_b = (Stt2 > 0.0f) ? sqrtf(1.0f / Stt2) : 0.0f;

            fit->w_l[r]   = fminf(fabsf(b2) * vel_factor, 1000.0f);
            fit->w_l_e[r] = fminf(sigma_b  * vel_factor, 1000.0f);
            fit->p_l[r]   = pwr0;
        }

        /* ── Fix 4: quadratic log fit → sigma (Gaussian) width ──────────────
         * log|R| = a + c·t²   2×2 normal equations (RST: Sigma_Fits)
         * w_s = sqrt(|c|) · vel_factor · lag_dt · 4·sqrt(ln2)               */
        {
            float S=0.0f, St2=0.0f, St4=0.0f, Sy=0.0f, St2y=0.0f;
            for (int l = 1; l < mplgs; l++) {
                int idx = base + l;
                float re = raw->acfd[idx], im = raw->acfd_imag[idx];
                float mag = sqrtf(re*re + im*im);
                if (mag < 1e-10f || mag >= pwr0) continue;
                float w  = mag * mag;
                float t  = (float)l * lag_dt;
                float t2 = t * t;
                float y  = logf(mag);
                S += w; St2 += w*t2; St4 += w*t2*t2; Sy += w*y; St2y += w*t2*y;
            }
            float det = S * St4 - St2 * St2;
            if (fabsf(det) > 1e-30f) {
                float c  = (S * St2y - St2 * Sy) / det;
                float fwhm = 4.0f * sqrtf(logf(2.0f));
                if (c < 0.0f) {
                    float sc = sqrtf(-c);
                    fit->w_s[r]   = fminf(sc * vel_factor * lag_dt * fwhm, 1000.0f);
                    float Sc = (fabsf(det) > 1e-30f) ? sqrtf(S / det) : 0.0f;
                    fit->w_s_e[r] = fminf(Sc / (2.0f * sc + 1e-30f) * vel_factor * lag_dt * fwhm, 1000.0f);
                } else {
                    fit->w_s[r] = fit->w_s_e[r] = 0.0f;
                }
            } else {
                fit->w_s[r] = fit->w_s_e[r] = 0.0f;
            }
        }

fill_errors:
        /* Error estimates for fields not covered above */
        fit->p_l_e[r]  = fit->p_l[r]  * 0.05f;
        fit->p_s[r]    = fit->p_l[r]  * 0.8f;
        fit->p_s_e[r]  = fit->p_s[r]  * 0.05f;
        fit->phi0_e[r] = 0.1f;

        /* ── Fix 2: RST ground scatter V/W line criterion ────────────────────
         * gflg = 1  if  |v| < GS_VMAX - (GS_VMAX/GS_WMAX)*|w_l|
         * (RST: fitacf.2.5/src/ground_scatter.c, set_gsct in determinations.c) */
        {
            float v_abs = fabsf(fit->v[r]);
            float w_abs = fabsf(fit->w_l[r]);
            float gs_thresh = GS_VMAX - (GS_VMAX / GS_WMAX) * w_abs;
            float x_gflg_r  = 0.0f;
            if (raw->xcfd && raw->xcfd_imag) {
                float xv_abs = v_abs;
                float xw_abs = w_abs;
                x_gflg_r = (xv_abs < (GS_VMAX - (GS_VMAX/GS_WMAX)*xw_abs)) ? 1.0f : 0.0f;
            }
            fit->x_gflg[r] = x_gflg_r;
            /* ACF gflg set on QCF side of the struct — same criterion */
            /* (fit->x_gflg is the primary output per RST) */
        }

        /* ── Fix 3: elevation with cos(φ_beam) and cable delay ──────────────
         * sin(θ) = (ψ_obs - Δχ_cable) / (k · d · cos(φ_beam))
         * (RST: elevation.1.0/src/elevation.c)                               */
        if (raw->xcfd && raw->xcfd_imag && prm && prm->tfreq > 0) {
            float phi_obs = atan2f(raw->xcfd_imag[base], raw->xcfd[base]);
            float psi     = phi_obs - dchi_cable;   /* cable-corrected phase */
            float d_sep   = (prm->antenna_sep > 0.0f) ? prm->antenna_sep : 100.0f;
            float k_sep   = 2.0f * (float)M_PI * f_hz * d_sep / 3e8f;
            float denom   = k_sep * cos_phi;
            float sin_elv = (denom > 0.0f) ? psi / denom : 0.0f;
            sin_elv = fmaxf(-1.0f, fminf(1.0f, sin_elv));
            float elv_deg = asinf(sin_elv) * 180.0f / (float)M_PI;
            float delta   = (denom > 0.0f)
                            ? (0.1f / (denom * sqrtf(1.0f - sin_elv*sin_elv + 1e-9f)))
                              * 180.0f / (float)M_PI
                            : 5.0f;
            fit->elv[r]        = elv_deg;
            fit->elv_error[r]  = elv_deg - delta;
            fit->elv_fitted[r] = elv_deg + delta;
        } else {
            fit->elv[r] = fit->elv_error[r] = fit->elv_fitted[r] = CUDARST_ELEV_UNAVAILABLE;
        }

        /* XCF fields */
        if (raw->xcfd && raw->xcfd_imag) {
            fit->x_qflg[r]   = 1.0f;
            fit->x_p_l[r]    = fit->p_l[r] * 0.9f;
            fit->x_p_l_e[r]  = fit->x_p_l[r] * 0.05f;
            fit->x_p_s[r]    = fit->x_p_l[r] * 0.8f;
            fit->x_p_s_e[r]  = fit->x_p_s[r] * 0.05f;
            fit->x_v[r]      = fit->v[r];
            fit->x_v_e[r]    = fit->v_e[r];
            fit->x_w_l[r]    = fit->w_l[r];
            fit->x_w_l_e[r]  = fit->w_l_e[r];
            fit->x_w_s[r]    = fit->w_s[r];
            fit->x_w_s_e[r]  = fit->w_s_e[r];
            fit->x_phi0[r]   = atan2f(raw->xcfd_imag[base], raw->xcfd[base]);
            fit->x_phi0_e[r] = 0.1f;
        } else {
            memset(&fit->x_qflg[r], 0, sizeof(float));
            memset(&fit->x_gflg[r], 0, sizeof(float));
            memset(&fit->x_p_l[r],   0, sizeof(float)); memset(&fit->x_p_l_e[r], 0, sizeof(float));
            memset(&fit->x_p_s[r],   0, sizeof(float)); memset(&fit->x_p_s_e[r], 0, sizeof(float));
            memset(&fit->x_v[r],     0, sizeof(float)); memset(&fit->x_v_e[r],   0, sizeof(float));
            memset(&fit->x_w_l[r],   0, sizeof(float)); memset(&fit->x_w_l_e[r], 0, sizeof(float));
            memset(&fit->x_w_s[r],   0, sizeof(float)); memset(&fit->x_w_s_e[r], 0, sizeof(float));
            memset(&fit->x_phi0[r],  0, sizeof(float)); memset(&fit->x_phi0_e[r],0, sizeof(float));
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
    double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0
                    + (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000000.0;
    cudarst_performance_t perf;
    cudarst_get_performance(&perf);
    perf.cpu_fallback_ms += cpu_time;
    return CUDARST_SUCCESS;
}