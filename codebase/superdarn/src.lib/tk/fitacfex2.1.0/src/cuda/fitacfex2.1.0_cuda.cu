/*
 * fitacfex2.1.0_cuda.cu
 * =====================
 * CUDA implementation of the FitACFex2 algorithm.
 *
 * Extends fitacfex.1.3_cuda.cu with:
 *   - minlag = 4 (vs 6 in v1)
 *   - XCF-based elevation angle kernel (cuda_fitacfex2_xcf_phase_kernel)
 *   - Sky noise estimation kernel (cuda_fitacfex2_sky_noise_kernel)
 *
 * The five ACF processing kernels are identical to fitacfex.1.3 except
 * for the minlag constant; they are re-defined here so this library is
 * self-contained.
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
            fprintf(stderr, "CUDA fitacfex2 error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
        } \
    } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NSLOPES2      120
#define N_MODEL_VELS2 (2*NSLOPES2+1)
#define MINLAG2       4     /* fitacfex2 uses minlag=4 */

/* -----------------------------------------------------------------------
 * Kernels 1–5: ACF processing (identical to fitacfex.1.3 aside from naming)
 */

__global__ void cuda_fitacfex2_model_gen_kernel(
    float *d_model_phi, float *d_model_vels,
    int nslopes, int mplgs, int tfreq, int mpinc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nslopes || j >= mplgs) return;

    float model_slope = 180.0f * i / nslopes;
    float phi = j * model_slope;
    int p = (int)(phi / 360.0f);
    d_model_phi[i * mplgs + j] = phi - p * 360.0f;

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

__global__ void cuda_fitacfex2_lag_power_kernel(
    const float *d_acfd_re, const float *d_acfd_im,
    const float *d_pwr0, float noise_search,
    int nrang, int mplgs, int nave,
    float *d_lagpwr, unsigned char *d_avail)
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

__global__ void cuda_fitacfex2_power_fit_kernel(
    const float *d_lagpwr, const unsigned char *d_avail,
    int nrang, int mplgs, float noise_search,
    int tfreq, int mpinc,
    float *d_width, float *d_power)
{
    int R = blockIdx.x * blockDim.x + threadIdx.x;
    if (R >= nrang) return;

    float sx=0, sy=0, sx2=0, sxy=0;
    int n = 0;
    for (int L = 0; L < mplgs; L++) {
        int idx = R * mplgs + L;
        if (!d_avail[idx] || d_lagpwr[idx] <= 0.0f) continue;
        float x = (float)L, y = logf(d_lagpwr[idx]);
        sx += x; sy += y; sx2 += x*x; sxy += x*y; n++;
    }
    d_width[R] = d_power[R] = 0.0f;
    if (n < MINLAG2) return;

    float denom = n*sx2 - sx*sx;
    if (fabsf(denom) < 1e-10f) return;
    float a = (sy*sx2 - sx*sxy) / denom;
    float b = (n*sxy  - sx*sy)  / denom;
    float c = 2.9979e8f * b / (mpinc*1.0e-6f) / (2.0f*(float)M_PI*1000.0f*tfreq);
    d_width[R] = fmaxf(0.0f, -c);
    d_power[R] = logf(expf(a) + noise_search);
}

__global__ void cuda_fitacfex2_phase_match_kernel(
    const float *d_acfd_re, const float *d_acfd_im,
    const float *d_lagpwr,  const unsigned char *d_avail,
    const float *d_model_phi,
    int nrang, int mplgs, int nslopes,
    float *d_errors)
{
    int R  = blockIdx.x * blockDim.x + threadIdx.x;
    int si = blockIdx.y * blockDim.y + threadIdx.y;
    if (R >= nrang || si > 2*nslopes) return;

    int model_row = (si <= nslopes) ? (nslopes - si) : (si - nslopes);
    float pwr_sum = 0.0f, err_sum = 0.0f;

    for (int L = 0; L < mplgs-1; L++) {
        int idx = R*mplgs + L;
        if (!d_avail[idx]) continue;
        float re = d_acfd_re[idx], im = d_acfd_im[idx];
        float data_phi = atan2f(re, im) * (180.0f / (float)M_PI);
        if (data_phi < 0) data_phi += 360.0f;

        float dphi;
        if (si >= nslopes) {
            dphi = fabsf(data_phi - d_model_phi[model_row*mplgs + L]);
        } else {
            dphi = fabsf((360.0f - data_phi) - d_model_phi[model_row*mplgs + L]);
        }
        if (dphi > 180.0f) dphi = 360.0f - dphi;

        float w = d_lagpwr[idx];
        pwr_sum += w;
        err_sum += dphi*dphi*w;
    }

    d_errors[R*(2*nslopes+1) + si] =
        (pwr_sum > 0.0f) ? sqrtf(err_sum/pwr_sum) : 1.0e30f;
}

__global__ void cuda_fitacfex2_velocity_kernel(
    const float *d_errors, const float *d_model_vels,
    const float *d_width,  const float *d_power,
    const float *d_pwr0,   float noise_search,
    int nrang, int nslopes, float minpwr, float sderr,
    float *d_v, float *d_p_l, float *d_w_l, float *d_p_0,
    int *d_qflg, int *d_gsct)
{
    int R = blockIdx.x * blockDim.x + threadIdx.x;
    if (R >= nrang) return;

    int n_err = 2*nslopes + 1;
    const float *err = d_errors + R*n_err;

    float mean_err=0, min_err=1.0e30f; int mininx=0;
    for (int i=0; i<n_err; i++) {
        mean_err += err[i];
        if (err[i] < min_err) { min_err=err[i]; mininx=i; }
    }
    mean_err /= n_err;
    float sd = 0.0f;
    for (int i=0; i<n_err; i++) { float d=err[i]-mean_err; sd+=d*d; }
    sd = sqrtf(sd/n_err);

    d_v[R] = d_p_l[R] = d_w_l[R] = d_p_0[R] = 0.0f;
    d_qflg[R] = d_gsct[R] = 0;

    float lag0_snr = (noise_search>0)
        ? 10.0f*log10f((d_pwr0[R]+noise_search)/noise_search) : 0.0f;
    if (lag0_snr < minpwr) return;

    if (min_err < (mean_err - sderr*sd)) {
        d_v[R]    = d_model_vels[mininx];
        d_p_l[R]  = d_power[R];
        d_w_l[R]  = d_width[R];
        d_p_0[R]  = d_pwr0[R];
        d_qflg[R] = 1;
        d_gsct[R] = (fabsf(d_v[R]) < 30.0f && d_w_l[R] < 30.0f) ? 1 : 0;
    }
}

/* -----------------------------------------------------------------------
 * Kernel 6: XCF-based elevation angle
 * One thread per range gate.
 * phi0[R] = phase of XCF averaged over available lags (weighted by power)
 * elv[R]  = arcsin( phi0 / (2π * d * cos(boresight)) )  [simplified]
 *
 * phidiff — antenna interferometer phase difference [radians]
 * tdiff   — time difference correction [microseconds]
 */
__global__ void cuda_fitacfex2_xcf_phase_kernel(
    const float *d_xcfd_re,     /* [nrang * mplgs] */
    const float *d_xcfd_im,     /* [nrang * mplgs] */
    const float *d_lagpwr,      /* [nrang * mplgs] — ACF power for weights */
    const unsigned char *d_avail,
    int nrang, int mplgs,
    float phidiff, float tdiff,
    float *d_phi0,              /* [nrang] */
    float *d_elv)               /* [nrang] */
{
    int R = blockIdx.x * blockDim.x + threadIdx.x;
    if (R >= nrang) return;

    float phi_sum = 0.0f, wt_sum = 0.0f;

    for (int L = 1; L < mplgs; L++) {   /* skip lag 0 */
        int idx = R * mplgs + L;
        if (!d_avail[idx]) continue;

        float re = d_xcfd_re[idx], im = d_xcfd_im[idx];
        float phi = atan2f(im, re);
        /* tdiff phase correction */
        float tcorr = (float)(2.0 * M_PI) * tdiff * 1.0e-6f * (float)L;
        phi -= tcorr;
        /* wrap to [-π, π] */
        while (phi >  (float)M_PI) phi -= (float)(2.0*M_PI);
        while (phi < -(float)M_PI) phi += (float)(2.0*M_PI);

        float w = d_lagpwr[idx];
        phi_sum += phi * w;
        wt_sum  += w;
    }

    float phi0 = (wt_sum > 0.0f) ? phi_sum / wt_sum : 0.0f;
    d_phi0[R] = phi0;

    /* Elevation: phi0 = 2π * baseline * sin(elv) / λ + phidiff
     * Simplified: elv ≈ arcsin((phi0 - phidiff) / phidiff_max)
     * phidiff encodes the baseline normalisation constant */
    float arg = (phidiff != 0.0f) ? (phi0 - phidiff) / fabsf(phidiff) : 0.0f;
    arg = fmaxf(-1.0f, fminf(1.0f, arg));
    d_elv[R] = asinf(arg) * (180.0f / (float)M_PI);
}

/* -----------------------------------------------------------------------
 * Kernel 7: sky noise estimation
 * One thread per range gate — estimates background sky noise from lag-0
 * by finding the minimum power across range gates in a local window.
 * Simple approach: global minimum of pwr0 used as sky noise floor.
 */
__global__ void cuda_fitacfex2_sky_noise_kernel(
    const float *d_pwr0, int nrang,
    float *d_skynoise)   /* [nrang] — sky noise estimate per range */
{
    /* Phase 1: each thread finds local minimum */
    __shared__ float smin[256];
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    smin[tid] = (gid < nrang) ? d_pwr0[gid] : FLT_MAX;
    __syncthreads();

    /* Block reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smin[tid] = fminf(smin[tid], smin[tid+s]);
        __syncthreads();
    }

    /* Each block writes one minimum; full global min needs a second pass.
     * For simplicity use block-local min as sky noise estimate. */
    float block_min = smin[0];

    /* Write sky noise estimate for all range gates in this block */
    if (gid < nrang) d_skynoise[gid] = block_min;
}

/* -----------------------------------------------------------------------
 * C-linkage host wrapper
 */
extern "C" {

cudaError_t cuda_fitacfex2_process(
    const float *h_acfd_re, const float *h_acfd_im,
    const float *h_xcfd_re, const float *h_xcfd_im,
    const float *h_pwr0,
    float noise_search, int tfreq, int mpinc, int nave,
    float phidiff, float tdiff,
    int nrang, int mplgs, int xcf_enabled,
    float *h_v,    float *h_p_l,  float *h_w_l, float *h_p_0,
    int   *h_qflg, int   *h_gsct,
    float *h_phi0, float *h_elv)
{
    const int nslopes  = NSLOPES2;
    const float minpwr = 3.0f;
    const float sderr  = 3.0f;
    int n_acf    = nrang * mplgs;
    int n_model  = (nslopes+1) * mplgs;
    int n_errors = nrang * (2*nslopes+1);

    float *d_acfd_re, *d_acfd_im, *d_pwr0;
    float *d_xcfd_re=NULL, *d_xcfd_im=NULL;
    float *d_lagpwr, *d_model_phi, *d_model_vels, *d_errors;
    float *d_width, *d_power;
    float *d_v, *d_p_l, *d_w_l, *d_p_0;
    float *d_phi0=NULL, *d_elv=NULL, *d_skynoise=NULL;
    int   *d_qflg, *d_gsct;
    unsigned char *d_avail;

    CUDA_CHECK(cudaMalloc(&d_acfd_re,    n_acf    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acfd_im,    n_acf    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pwr0,       nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lagpwr,     n_acf    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_avail,      n_acf    * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_model_phi,  n_model  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_model_vels, N_MODEL_VELS2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_errors,     n_errors * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_width,      nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_power,      nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,          nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_l,        nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_l,        nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_0,        nrang    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qflg,       nrang    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gsct,       nrang    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_skynoise,   nrang    * sizeof(float)));

    if (xcf_enabled && h_xcfd_re && h_xcfd_im) {
        CUDA_CHECK(cudaMalloc(&d_xcfd_re, n_acf * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_xcfd_im, n_acf * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_phi0,    nrang * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_elv,     nrang * sizeof(float)));
        cudaMemcpy(d_xcfd_re, h_xcfd_re, n_acf*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xcfd_im, h_xcfd_im, n_acf*sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_acfd_re, h_acfd_re, n_acf  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acfd_im, h_acfd_im, n_acf  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pwr0,    h_pwr0,    nrang  * sizeof(float), cudaMemcpyHostToDevice);

    /* K1: model table */
    { dim3 b(16,16), g((nslopes/16)+2,(mplgs/16)+1);
      cuda_fitacfex2_model_gen_kernel<<<g,b>>>(
          d_model_phi, d_model_vels, nslopes, mplgs, tfreq, mpinc); }

    /* K7: sky noise */
    { int t=256, bl=(nrang+255)/256;
      cuda_fitacfex2_sky_noise_kernel<<<bl,t>>>(d_pwr0, nrang, d_skynoise); }

    /* K2: lag power */
    { dim3 b(16,16), g((nrang+15)/16,(mplgs+15)/16);
      cuda_fitacfex2_lag_power_kernel<<<g,b>>>(
          d_acfd_re, d_acfd_im, d_pwr0, noise_search,
          nrang, mplgs, nave, d_lagpwr, d_avail); }

    /* K3: power fit */
    { int t=256, bl=(nrang+255)/256;
      cuda_fitacfex2_power_fit_kernel<<<bl,t>>>(
          d_lagpwr, d_avail, nrang, mplgs, noise_search,
          tfreq, mpinc, d_width, d_power); }

    /* K4: phase matching */
    { dim3 b(8,32), g((nrang+7)/8,(2*nslopes+1+31)/32);
      cuda_fitacfex2_phase_match_kernel<<<g,b>>>(
          d_acfd_re, d_acfd_im, d_lagpwr, d_avail,
          d_model_phi, nrang, mplgs, nslopes, d_errors); }

    /* K5: velocity */
    { int t=256, bl=(nrang+255)/256;
      cuda_fitacfex2_velocity_kernel<<<bl,t>>>(
          d_errors, d_model_vels, d_width, d_power, d_pwr0,
          noise_search, nrang, nslopes, minpwr, sderr,
          d_v, d_p_l, d_w_l, d_p_0, d_qflg, d_gsct); }

    /* K6: XCF elevation (optional) */
    if (xcf_enabled && d_xcfd_re) {
        int t=256, bl=(nrang+255)/256;
        cuda_fitacfex2_xcf_phase_kernel<<<bl,t>>>(
            d_xcfd_re, d_xcfd_im, d_lagpwr, d_avail,
            nrang, mplgs, phidiff, tdiff, d_phi0, d_elv);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(h_v,    d_v,    nrang*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_l,  d_p_l,  nrang*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w_l,  d_w_l,  nrang*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_0,  d_p_0,  nrang*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_qflg, d_qflg, nrang*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gsct, d_gsct, nrang*sizeof(int),   cudaMemcpyDeviceToHost);

    if (xcf_enabled && d_phi0) {
        cudaMemcpy(h_phi0, d_phi0, nrang*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_elv,  d_elv,  nrang*sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_acfd_re);   cudaFree(d_acfd_im);   cudaFree(d_pwr0);
    cudaFree(d_lagpwr);    cudaFree(d_avail);      cudaFree(d_skynoise);
    cudaFree(d_model_phi); cudaFree(d_model_vels); cudaFree(d_errors);
    cudaFree(d_width);     cudaFree(d_power);
    cudaFree(d_v);         cudaFree(d_p_l);        cudaFree(d_w_l);
    cudaFree(d_p_0);       cudaFree(d_qflg);       cudaFree(d_gsct);
    if (d_xcfd_re) cudaFree(d_xcfd_re);
    if (d_xcfd_im) cudaFree(d_xcfd_im);
    if (d_phi0)    cudaFree(d_phi0);
    if (d_elv)     cudaFree(d_elv);

    return cudaGetLastError();
}

bool cuda_fitacfex2_is_available(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n > 0;
}

} /* extern "C" */
