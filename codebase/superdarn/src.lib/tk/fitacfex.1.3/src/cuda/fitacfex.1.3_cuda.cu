/*
 * fitacfex.1.3_cuda.cu
 * ====================
 * CUDA implementation of the FitACFex algorithm (Greenwald/Oskavik).
 *
 * Parallelises FitACFex() from fitacfex.c across all range gates.
 *
 * CPU algorithm summary (per range gate R):
 *   1. lagpwr[L] = sqrt(re²+im²);  avail[L] = (lagpwr > threshold)
 *   2. Least-squares fit of log(lagpwr) vs L → fitted_width, fitted_power
 *   3. For i in 0..nslopes: compute chi-squared error vs phase model
 *   4. Pick slope with minimum error → velocity
 *
 * GPU mapping:
 *   Kernel 1 (model_gen):   one thread per (slope, lag)
 *   Kernel 2 (lag_power):   one thread per (range, lag)
 *   Kernel 3 (power_fit):   one thread per range gate
 *   Kernel 4 (phase_match): one thread per (range, slope)
 *   Kernel 5 (velocity):    one thread per range gate
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA fitacfex error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
        } \
    } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NSLOPES      120
#define N_MODEL_VELS (2*NSLOPES+1)
/* Maximum lags / range gates per radar mode */
#define MAX_MPLGS    70

/* -----------------------------------------------------------------------
 * Kernel 1: phase model table generation
 * One thread per (i=slope, j=lag).
 * model_phi[i * mplgs + j] = (j * 180*i/nslopes) mod 360
 * model_vels[nslopes-i] = -vel_pos,  model_vels[nslopes+i] = +vel_pos
 */
__global__ void cuda_fitacfex_model_gen_kernel(
    float *d_model_phi,         /* [(nslopes+1) * mplgs] */
    float *d_model_vels,        /* [2*nslopes+1] */
    int nslopes, int mplgs, int tfreq, int mpinc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  /* slope index */
    int j = blockIdx.y * blockDim.y + threadIdx.y;  /* lag   index */

    if (i > nslopes || j >= mplgs) return;

    float model_slope = 180.0f * i / nslopes;
    float phi = j * model_slope;
    int p = (int)(phi / 360.0f);
    d_model_phi[i * mplgs + j] = phi - p * 360.0f;

    /* Write velocity once per slope (lag 0 thread) */
    if (j == 0) {
        float vel_pos = 2.9979e8f / 2.0f *
            (1.0f - 1000.0f * tfreq /
             (1000.0f * tfreq + model_slope / 360.0f / (mpinc * 1.0e-6f)));
        if (i < nslopes) {
            d_model_vels[nslopes - i] = -vel_pos;
            d_model_vels[nslopes + i] =  vel_pos;
        }
        if (i == nslopes) d_model_vels[0] = 0.0f;
    }
}

/* -----------------------------------------------------------------------
 * Kernel 2: lag power and availability
 * One thread per (range R, lag L).
 * lagpwr[R*mplgs+L] = sqrt(re²+im²) if > threshold, else 0.
 * avail[R*mplgs+L]  = 1 if above threshold.
 */
__global__ void cuda_fitacfex_lag_power_kernel(
    const float *d_acfd_re,        /* [nrang * mplgs] */
    const float *d_acfd_im,        /* [nrang * mplgs] */
    const float *d_pwr0,           /* [nrang] lag-0 power */
    float noise_search,
    int nrang, int mplgs, int nave,
    float *d_lagpwr,               /* [nrang * mplgs] */
    unsigned char *d_avail)        /* [nrang * mplgs] */
{
    int R = blockIdx.x * blockDim.x + threadIdx.x;
    int L = blockIdx.y * blockDim.y + threadIdx.y;
    if (R >= nrang || L >= mplgs) return;

    int idx = R * mplgs + L;
    float re = d_acfd_re[idx], im = d_acfd_im[idx];
    float pwr = sqrtf(re*re + im*im);

    float threshold = (nave > 0) ? d_pwr0[R] / sqrtf((float)nave) : 0.0f;
    if (pwr > threshold) {
        d_lagpwr[idx] = pwr;
        d_avail[idx]  = 1;
    } else {
        d_lagpwr[idx] = 0.0f;
        d_avail[idx]  = 0;
    }
}

/* -----------------------------------------------------------------------
 * Kernel 3: exponential power fit per range gate
 * One thread per range gate R.
 * Fits log(lagpwr) vs lag index using unweighted least squares:
 *   y = a + b*x  →  width = -c*b / (tfreq*2π),  power = exp(a)
 */
__global__ void cuda_fitacfex_power_fit_kernel(
    const float *d_lagpwr,
    const unsigned char *d_avail,
    int nrang, int mplgs, float noise_search,
    int tfreq, int mpinc,
    float *d_width,   /* [nrang] */
    float *d_power)   /* [nrang] */
{
    int R = blockIdx.x * blockDim.x + threadIdx.x;
    if (R >= nrang) return;

    float sx = 0, sy = 0, sx2 = 0, sxy = 0;
    int n = 0;

    for (int L = 0; L < mplgs; L++) {
        int idx = R * mplgs + L;
        if (!d_avail[idx] || d_lagpwr[idx] <= 0.0f) continue;
        float x = (float)L;
        float y = logf(d_lagpwr[idx]);
        sx  += x;   sy  += y;
        sx2 += x*x; sxy += x*y;
        n++;
    }

    d_width[R]  = 0.0f;
    d_power[R]  = 0.0f;

    if (n < 2) return;

    float denom = n * sx2 - sx * sx;
    if (fabsf(denom) < 1e-10f) return;

    float a = (sy * sx2 - sx * sxy) / denom;  /* intercept */
    float b = (n * sxy  - sx * sy)  / denom;  /* slope     */

    float c = 2.9979e8f * b / (mpinc * 1.0e-6f) / (2.0f * (float)M_PI * 1000.0f * tfreq);
    float w = -c;
    d_width[R] = (w > 0.0f) ? w : 0.0f;
    d_power[R] = logf(expf(a) + noise_search);
}

/* -----------------------------------------------------------------------
 * Kernel 4: phase model matching — chi-squared error
 * Grid: (nrang, 2*nslopes+1),  one thread per (R, slope).
 * Each thread computes the weighted RMS phase error between the data phase
 * and the model phase for its assigned slope.
 */
__global__ void cuda_fitacfex_phase_match_kernel(
    const float *d_acfd_re,
    const float *d_acfd_im,
    const float *d_lagpwr,
    const unsigned char *d_avail,
    const float *d_model_phi,       /* [(nslopes+1) * mplgs] */
    int nrang, int mplgs, int nslopes,
    float *d_errors)                 /* [nrang * (2*nslopes+1)] */
{
    int R  = blockIdx.x * blockDim.x + threadIdx.x;
    int si = blockIdx.y * blockDim.y + threadIdx.y;  /* 0..2*nslopes */
    if (R >= nrang || si > 2 * nslopes) return;

    /* Map slope index to model row */
    int model_row = (si <= nslopes) ? (nslopes - si) : (si - nslopes);

    float pwr_sum  = 0.0f;
    float err_sum  = 0.0f;

    for (int L = 0; L < mplgs - 1; L++) {
        int idx = R * mplgs + L;
        if (!d_avail[idx]) continue;

        float re = d_acfd_re[idx], im = d_acfd_im[idx];
        float data_phi = atan2f(re, im) * (180.0f / (float)M_PI);
        if (data_phi < 0) data_phi += 360.0f;

        /* For si <= nslopes use positive branch, otherwise negative */
        float dphi;
        if (si >= nslopes) {
            dphi = fabsf(data_phi - d_model_phi[model_row * mplgs + L]);
        } else {
            float neg_phi = 360.0f - data_phi;
            dphi = fabsf(neg_phi  - d_model_phi[model_row * mplgs + L]);
        }
        if (dphi > 180.0f) dphi = 360.0f - dphi;

        float w = d_lagpwr[idx];
        pwr_sum += w;
        err_sum += dphi * dphi * w;
    }

    float err = (pwr_sum > 0.0f) ? sqrtf(err_sum / pwr_sum) : 1.0e30f;
    d_errors[R * (2 * nslopes + 1) + si] = err;
}

/* -----------------------------------------------------------------------
 * Kernel 5: velocity extraction
 * One thread per range gate.
 * Scans d_errors[R, :] to find minimum error, assigns velocity and flags.
 */
__global__ void cuda_fitacfex_velocity_kernel(
    const float *d_errors,        /* [nrang * (2*nslopes+1)] */
    const float *d_model_vels,    /* [2*nslopes+1] */
    const float *d_width,
    const float *d_power,
    const float *d_pwr0,
    float noise_search,
    int nrang, int nslopes, float minpwr, float sderr,
    float *d_v,    float *d_p_l,  float *d_w_l,  float *d_p_0,
    int   *d_qflg, int   *d_gsct)
{
    int R = blockIdx.x * blockDim.x + threadIdx.x;
    if (R >= nrang) return;

    int n_errors = 2 * nslopes + 1;
    const float *err = d_errors + R * n_errors;

    /* Compute mean and min */
    float mean_err = 0.0f, min_err = 1.0e30f;
    int   mininx   = 0;
    for (int i = 0; i < n_errors; i++) {
        mean_err += err[i];
        if (err[i] < min_err) { min_err = err[i]; mininx = i; }
    }
    mean_err /= n_errors;

    /* Compute std-dev */
    float sd = 0.0f;
    for (int i = 0; i < n_errors; i++) {
        float d = err[i] - mean_err;
        sd += d * d;
    }
    sd = sqrtf(sd / n_errors);

    /* Default: no fit */
    d_v[R] = d_p_l[R] = d_w_l[R] = d_p_0[R] = 0.0f;
    d_qflg[R] = d_gsct[R] = 0;

    /* lag-0 SNR check */
    float lag0_snr = (noise_search > 0.0f)
        ? 10.0f * log10f((d_pwr0[R] + noise_search) / noise_search)
        : 0.0f;
    if (lag0_snr < minpwr) return;

    if (min_err < (mean_err - sderr * sd)) {
        d_v[R]    = d_model_vels[mininx];
        d_p_l[R]  = d_power[R];
        d_w_l[R]  = d_width[R];
        d_p_0[R]  = d_pwr0[R];
        d_qflg[R] = 1;
        d_gsct[R] = (fabsf(d_v[R]) < 30.0f && d_w_l[R] < 30.0f) ? 1 : 0;
    }
}

/* -----------------------------------------------------------------------
 * C-linkage host wrapper
 *
 * cuda_fitacfex_process:
 *   d_acfd_re/im [nrang*mplgs] — ACF real and imaginary parts (row-major)
 *   d_pwr0       [nrang]       — lag-0 power
 *   noise_search               — noise floor
 *   tfreq, mpinc, nave         — from RadarParm
 *   nrang, mplgs               — dimensions
 *   Output arrays (caller allocates, host memory):
 *     h_v, h_p_l, h_w_l, h_p_0 [nrang] — fitted parameters
 *     h_qflg, h_gsct            [nrang] — quality and ground-scatter flags
 */
extern "C" {

cudaError_t cuda_fitacfex_process(
    const float *h_acfd_re, const float *h_acfd_im,
    const float *h_pwr0,
    float noise_search, int tfreq, int mpinc, int nave,
    int nrang, int mplgs,
    float *h_v,    float *h_p_l, float *h_w_l, float *h_p_0,
    int   *h_qflg, int   *h_gsct)
{
    const int nslopes  = NSLOPES;
    const float minpwr = 3.0f;
    const float sderr  = 3.0f;
    int n_acf    = nrang * mplgs;
    int n_model  = (nslopes + 1) * mplgs;
    int n_errors = nrang * (2 * nslopes + 1);

    float *d_acfd_re, *d_acfd_im, *d_pwr0;
    float *d_lagpwr, *d_model_phi, *d_model_vels;
    float *d_errors, *d_width, *d_power;
    float *d_v, *d_p_l, *d_w_l, *d_p_0;
    int   *d_qflg, *d_gsct;
    unsigned char *d_avail;

    CUDA_CHECK(cudaMalloc(&d_acfd_re,    n_acf    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acfd_im,    n_acf    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pwr0,       nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lagpwr,     n_acf    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_avail,      n_acf    * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_model_phi,  n_model  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_model_vels, N_MODEL_VELS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_errors,     n_errors * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_width,      nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_power,      nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,          nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_l,        nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_l,        nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_0,        nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qflg,       nrang    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gsct,       nrang    * sizeof(int)));

    cudaMemcpy(d_acfd_re, h_acfd_re, n_acf  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acfd_im, h_acfd_im, n_acf  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pwr0,    h_pwr0,    nrang  * sizeof(float), cudaMemcpyHostToDevice);

    /* K1: model table */
    {
        dim3 block(16, 16);
        dim3 grid((nslopes/16)+2, (mplgs/16)+1);
        cuda_fitacfex_model_gen_kernel<<<grid, block>>>(
            d_model_phi, d_model_vels, nslopes, mplgs, tfreq, mpinc);
    }

    /* K2: lag power */
    {
        dim3 block(16, 16);
        dim3 grid((nrang+15)/16, (mplgs+15)/16);
        cuda_fitacfex_lag_power_kernel<<<grid, block>>>(
            d_acfd_re, d_acfd_im, d_pwr0, noise_search,
            nrang, mplgs, nave, d_lagpwr, d_avail);
    }

    /* K3: power fit */
    {
        int threads = 256, blocks = (nrang + 255) / 256;
        cuda_fitacfex_power_fit_kernel<<<blocks, threads>>>(
            d_lagpwr, d_avail, nrang, mplgs, noise_search,
            tfreq, mpinc, d_width, d_power);
    }

    /* K4: phase matching */
    {
        dim3 block(8, 32);
        dim3 grid((nrang + 7) / 8, (2*nslopes+1+31) / 32);
        cuda_fitacfex_phase_match_kernel<<<grid, block>>>(
            d_acfd_re, d_acfd_im, d_lagpwr, d_avail,
            d_model_phi, nrang, mplgs, nslopes, d_errors);
    }

    /* K5: velocity extraction */
    {
        int threads = 256, blocks = (nrang + 255) / 256;
        cuda_fitacfex_velocity_kernel<<<blocks, threads>>>(
            d_errors, d_model_vels, d_width, d_power, d_pwr0,
            noise_search, nrang, nslopes, minpwr, sderr,
            d_v, d_p_l, d_w_l, d_p_0, d_qflg, d_gsct);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(h_v,    d_v,    nrang * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_l,  d_p_l,  nrang * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w_l,  d_w_l,  nrang * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_0,  d_p_0,  nrang * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_qflg, d_qflg, nrang * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gsct, d_gsct, nrang * sizeof(int),   cudaMemcpyDeviceToHost);

    cudaFree(d_acfd_re);   cudaFree(d_acfd_im);   cudaFree(d_pwr0);
    cudaFree(d_lagpwr);    cudaFree(d_avail);
    cudaFree(d_model_phi); cudaFree(d_model_vels); cudaFree(d_errors);
    cudaFree(d_width);     cudaFree(d_power);
    cudaFree(d_v);         cudaFree(d_p_l);        cudaFree(d_w_l);
    cudaFree(d_p_0);       cudaFree(d_qflg);       cudaFree(d_gsct);

    return cudaGetLastError();
}

bool cuda_fitacfex_is_available(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n > 0;
}

} /* extern "C" */
