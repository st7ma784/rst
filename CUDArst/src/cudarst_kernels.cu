/**
 * @file cudarst_kernels.cu
 * @brief CUDA kernel implementations for CUDArst library
 * 
 * Contains optimized CUDA kernels for all SuperDARN processing modules:
 * - FITACF v3.0: Auto-correlation function processing
 * - LMFIT v2.0: Levenberg-Marquardt fitting
 * - ACF v1.16: Auto-correlation functions
 * - IQ v1.7: I/Q data processing
 * - CNVMAP v1.17: Convection mapping
 * - GRID v1.24: Spatial grid processing
 * - FIT v1.35: Fitting algorithms
 * 
 * @version 2.0.0 - Complete module integration
 * @date 2025-09-20
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return -1; \
        } \
    } while (0)

/* ====================================================================
 * FITACF CUDA Kernels
 * ====================================================================*/

__device__ float cuda_magnitude(float real, float imag)
{
    return sqrtf(real * real + imag * imag);
}

__device__ float cuda_phase(float real, float imag)
{
    return atan2f(imag, real);
}

__global__ void cuda_fitacf_power_kernel(const float *acf_real, const float *acf_imag,
                                         int nrang, int mplgs, float *power)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r >= nrang) return;
    
    /* Calculate lag-0 power */
    int lag0_idx = r * mplgs + 0;
    power[r] = cuda_magnitude(acf_real[lag0_idx], acf_imag[lag0_idx]);
}

/* ── Fix 1: RST one-parameter weighted LS velocity  ─────────────────────────
 * RST fitacf_v3.0/src/leastsquares.c: one_param_straight_line_fit()
 * Fits phase[l] = b * l  (forced through origin — lag-0 is the reference).
 * Weights: 1/sigma_phase² ≈ mag²  (amplitude-squared, Pass-1 equivalent).
 * velocity = b * vel_factor   where vel_factor = c/(4π·f·Δτ).              */
__global__ void cuda_fitacf_velocity_ls_kernel(
    const float *acf_real, const float *acf_imag,
    const float *bp_sigma,     /* (nrang,mplgs) — NULL for amplitude weighting */
    const uint8_t *bad_mask,   /* (nrang,mplgs) 1=bad */
    int nrang, int mplgs,
    float vel_factor,
    float *velocity,
    float *velocity_err,
    float *phase_out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrang) return;

    int   base = r * mplgs;
    float phi0 = cuda_phase(acf_real[base], acf_imag[base]);
    phase_out[r] = phi0;

    float S_tt = 0.0f, S_ty = 0.0f;
    int n_valid = 0;

    for (int l = 1; l < mplgs; l++) {
        if (bad_mask[base + l]) continue;
        int   idx = base + l;
        float re  = acf_real[idx], im = acf_imag[idx];
        float mag = sqrtf(re*re + im*im);
        if (mag < 1e-10f) continue;

        float dphi = cuda_phase(re, im) - phi0;
        if (dphi >  M_PI) dphi -= 2.0f * M_PI;
        if (dphi < -M_PI) dphi += 2.0f * M_PI;

        /* Weight: B-P phase variance ≈ sigma_amplitude / mag; else mag² */
        float w;
        if (bp_sigma != NULL) {
            float sp = bp_sigma[idx] / (mag + 1e-30f);
            w = (sp > 1e-10f) ? 1.0f / (sp * sp) : mag * mag;
        } else {
            w = mag * mag;
        }

        float t = (float)l;
        S_tt += w * t * t;
        S_ty += w * t * dphi;
        n_valid++;
    }

    if (n_valid >= 1 && S_tt > 0.0f) {
        float b = S_ty / S_tt;                     /* rad/lag-index */
        velocity[r]     = b * vel_factor;
        velocity_err[r] = sqrtf(1.0f / S_tt) * vel_factor;
    } else {
        velocity[r] = velocity_err[r] = 0.0f;
    }
}

/* ── Fix 1 + 4: RST two-pass weighted LS lambda width  ──────────────────────
 * RST fitacf_v3.0/src/fitting.c: Power_Fits() / leastsquares.c
 * Pass 1: w = |R|²  → unbiased slope b  (RST: linear-power weights)
 * Pass 2: w = 1/σ_log²  → error σ_b    (RST: log-corrected B-P weights)
 * w_l = |b| * vel_factor,   w_l_err = σ_b * vel_factor                     */
__global__ void cuda_fitacf_width_ls_kernel(
    const float *acf_real, const float *acf_imag,
    const float *pwr0,
    const float *bp_sigma,
    const uint8_t *bad_mask,
    int nrang, int mplgs,
    float vel_factor,
    float *width,
    float *width_err)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrang) return;

    int   base = r * mplgs;
    float p0   = pwr0[r] + 1e-30f;

    /* Pass 1: linear-power weights */
    float Stt1 = 0.0f, Sty1 = 0.0f;
    for (int l = 1; l < mplgs; l++) {
        if (bad_mask[base + l]) continue;
        int idx = base + l;
        float mag = sqrtf(acf_real[idx]*acf_real[idx] + acf_imag[idx]*acf_imag[idx]);
        if (mag < 1e-10f || mag >= p0) continue;
        float w = mag * mag;
        float t = (float)l;
        Stt1 += w * t * t;
        Sty1 += w * t * logf(mag);
    }

    float b1 = (Stt1 > 0.0f) ? Sty1 / Stt1 : 0.0f;

    /* Pass 2: log-corrected B-P weights */
    float Stt2 = 0.0f, Sty2 = 0.0f;
    for (int l = 1; l < mplgs; l++) {
        if (bad_mask[base + l]) continue;
        int idx = base + l;
        float mag = sqrtf(acf_real[idx]*acf_real[idx] + acf_imag[idx]*acf_imag[idx]);
        if (mag < 1e-10f || mag >= p0) continue;
        float sig_log = (bp_sigma != NULL)
                         ? bp_sigma[idx] / (mag + 1e-30f)
                         : 0.1f;
        float w = (sig_log > 1e-10f) ? 1.0f / (sig_log * sig_log) : 1.0f;
        float t = (float)l;
        Stt2 += w * t * t;
        Sty2 += w * t * logf(mag);
    }

    float b2      = (Stt2 > 0.0f) ? Sty2 / Stt2 : b1;
    float sigma_b = (Stt2 > 0.0f) ? sqrtf(1.0f / Stt2) : 0.0f;

    width[r]     = fminf(fabsf(b2) * vel_factor, 1000.0f);
    width_err[r] = fminf(sigma_b  * vel_factor, 1000.0f);
}

/* Simple lag-0 phase kernel */
__global__ void cuda_fitacf_lag0_phase_kernel(
    const float *acf_real, const float *acf_imag,
    int nrang, int mplgs, float *phase)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrang) return;
    phase[r] = cuda_phase(acf_real[r * mplgs], acf_imag[r * mplgs]);
}

/* ── Invert bad_mask → valid_mask for kernels that expect 1=valid ─────────── */
__global__ void cuda_invert_mask_kernel(const uint8_t *bad, uint8_t *valid, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) valid[i] = bad[i] ? 0 : 1;
}

/* ── Host function: full RST-accurate FITACF pipeline ────────────────────────
 * Fix 1: one-param weighted LS velocity (all valid lags)
 * Fix 2: two-pass weighted LS lambda width
 * Fix 4: sigma (Gaussian) width via quadratic fit
 * All use lag-validity mask + Bendat-Piersol sigma.                         */
extern "C" int cuda_fitacf_process_ranges(
    const float *acf_real, const float *acf_imag,
    int nrang, int mplgs, int nave,
    float vel_factor, float lag_dt, float noise_pwr,
    float *power,           /* host output */
    float *velocity,        /* host output */
    float *velocity_err,    /* host output */
    float *width,           /* host output */
    float *width_err,       /* host output */
    float *width_sigma,     /* host output — may be NULL to skip */
    float *width_sigma_err, /* host output — may be NULL to skip */
    float *phase)           /* host output */
{
    size_t acf_sz  = (size_t)nrang * mplgs * sizeof(float);
    size_t rng_sz  = (size_t)nrang * sizeof(float);
    size_t mask_sz = (size_t)nrang * mplgs * sizeof(uint8_t);
    size_t sig_sz  = (size_t)nrang * mplgs * sizeof(float);

    float   *d_acf_r, *d_acf_i, *d_pwr, *d_sig;
    float   *d_vel, *d_vel_err, *d_wid, *d_wid_err, *d_ph;
    uint8_t *d_bad, *d_valid;

    CUDA_CHECK(cudaMalloc(&d_acf_r,  acf_sz));
    CUDA_CHECK(cudaMalloc(&d_acf_i,  acf_sz));
    CUDA_CHECK(cudaMalloc(&d_pwr,    rng_sz));
    CUDA_CHECK(cudaMalloc(&d_sig,    sig_sz));
    CUDA_CHECK(cudaMalloc(&d_bad,    mask_sz));
    CUDA_CHECK(cudaMalloc(&d_valid,  mask_sz));
    CUDA_CHECK(cudaMalloc(&d_vel,    rng_sz));
    CUDA_CHECK(cudaMalloc(&d_vel_err,rng_sz));
    CUDA_CHECK(cudaMalloc(&d_wid,    rng_sz));
    CUDA_CHECK(cudaMalloc(&d_wid_err,rng_sz));
    CUDA_CHECK(cudaMalloc(&d_ph,     rng_sz));

    CUDA_CHECK(cudaMemcpy(d_acf_r, acf_real, acf_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_acf_i, acf_imag, acf_sz, cudaMemcpyHostToDevice));

    int blk = 256;
    int grd = (nrang + blk - 1) / blk;
    dim3 blk2(16, 16);
    dim3 grd2((nrang + 15) / 16, (mplgs + 15) / 16);
    int  n_elem = nrang * mplgs;
    int  grd_e  = (n_elem + blk - 1) / blk;

    /* Step 1: lag-0 power */
    cuda_fitacf_power_kernel<<<grd, blk>>>(d_acf_r, d_acf_i, nrang, mplgs, d_pwr);
    CUDA_CHECK(cudaGetLastError());

    /* Step 2: bad lag mask (parallel flag + per-range cumsum propagation) */
    cuda_compute_lag_bad_kernel<<<grd2, blk2>>>(
        d_acf_r, d_acf_i, d_pwr, noise_pwr, nave, nrang, mplgs, d_bad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_propagate_lag_cutoff_kernel<<<grd, blk>>>(d_bad, nrang, mplgs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Step 3: B-P sigma (all elements, fully parallel) */
    cuda_bp_sigma_kernel<<<grd2, blk2>>>(
        d_acf_r, d_acf_i, d_pwr, nave, nrang, mplgs, d_sig);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Step 4: RST one-param weighted LS velocity */
    cuda_fitacf_velocity_ls_kernel<<<grd, blk>>>(
        d_acf_r, d_acf_i, d_sig, d_bad,
        nrang, mplgs, vel_factor, d_vel, d_vel_err, d_ph);
    CUDA_CHECK(cudaGetLastError());

    /* Step 5: RST two-pass weighted LS lambda width */
    cuda_fitacf_width_ls_kernel<<<grd, blk>>>(
        d_acf_r, d_acf_i, d_pwr, d_sig, d_bad,
        nrang, mplgs, vel_factor, d_wid, d_wid_err);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy primary results to host */
    CUDA_CHECK(cudaMemcpy(power,       d_pwr,    rng_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity,    d_vel,    rng_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity_err,d_vel_err,rng_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(width,       d_wid,    rng_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(width_err,   d_wid_err,rng_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phase,       d_ph,     rng_sz, cudaMemcpyDeviceToHost));

    /* Step 6: sigma (Gaussian) width — optional */
    if (width_sigma != NULL) {
        /* Need valid_mask (1=good) for sigma-width kernel */
        cuda_invert_mask_kernel<<<grd_e, blk>>>(d_bad, d_valid, n_elem);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float *d_ws, *d_wse;
        CUDA_CHECK(cudaMalloc(&d_ws,  rng_sz));
        CUDA_CHECK(cudaMalloc(&d_wse, rng_sz));

        cuda_fitacf_sigma_width_kernel<<<grd, blk>>>(
            d_acf_r, d_acf_i, d_sig, d_valid,
            nrang, mplgs, vel_factor, lag_dt, d_ws, d_wse);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(width_sigma,    d_ws,  rng_sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(width_sigma_err,d_wse, rng_sz, cudaMemcpyDeviceToHost));
        cudaFree(d_ws); cudaFree(d_wse);
    }

    cudaFree(d_acf_r); cudaFree(d_acf_i); cudaFree(d_pwr);   cudaFree(d_sig);
    cudaFree(d_bad);   cudaFree(d_valid);  cudaFree(d_vel);   cudaFree(d_vel_err);
    cudaFree(d_wid);   cudaFree(d_wid_err);cudaFree(d_ph);
    return 0;
}

/* ====================================================================
 * LMFIT CUDA Kernels
 * ====================================================================*/

__global__ void cuda_lmfit_calculate_residuals(const float *y, const float *x, const float *a,
                                                const float *sig, int ndata, int ma,
                                                float *residuals, float *chisq_partial)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= ndata) return;
    
    /* Calculate model value (polynomial) */
    float y_model = 0.0f;
    float x_power = 1.0f;
    for (int j = 0; j < ma; j++) {
        y_model += a[j] * x_power;
        x_power *= x[i];
    }
    
    /* Calculate weighted residual */
    float dy = y[i] - y_model;
    float weight = 1.0f / (sig[i] * sig[i]);
    residuals[i] = dy * sqrtf(weight);
    chisq_partial[i] = dy * dy * weight;
}

__global__ void cuda_lmfit_calculate_jacobian(const float *x, const float *sig, int ndata, int ma,
                                               float *jacobian)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= ndata || j >= ma) return;
    
    /* Calculate jacobian element: d(y_model)/da[j] for polynomial model */
    float x_power = 1.0f;
    for (int k = 0; k < j; k++) {
        x_power *= x[i];
    }
    
    float weight = 1.0f / sig[i];
    jacobian[i * ma + j] = x_power * weight;
}

__global__ void cuda_lmfit_build_normal_equations(const float *jacobian, const float *residuals,
                                                   int ndata, int ma, float *alpha, float *beta)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j >= ma || k >= ma) return;
    
    /* Calculate alpha[j][k] = sum(jacobian[i][j] * jacobian[i][k]) */
    float sum = 0.0f;
    for (int i = 0; i < ndata; i++) {
        sum += jacobian[i * ma + j] * jacobian[i * ma + k];
    }
    alpha[j * ma + k] = sum;
    
    /* Calculate beta[j] = sum(jacobian[i][j] * residuals[i]) */
    if (k == 0) {
        float beta_sum = 0.0f;
        for (int i = 0; i < ndata; i++) {
            beta_sum += jacobian[i * ma + j] * residuals[i];
        }
        beta[j] = beta_sum;
    }
}

void cuda_matrix_solve_3x3(float *matrix, float *rhs, float *solution)
{
    /* Simple 3x3 system solver using Cramer's rule */
    /* (Only works for small matrices - would need better solver for larger systems) */
    
    float det = matrix[0] * (matrix[4] * matrix[8] - matrix[7] * matrix[5])
              - matrix[1] * (matrix[3] * matrix[8] - matrix[6] * matrix[5])
              + matrix[2] * (matrix[3] * matrix[7] - matrix[6] * matrix[4]);
    
    if (fabsf(det) < 1e-10f) {
        /* Singular matrix */
        solution[0] = solution[1] = solution[2] = 0.0f;
        return;
    }
    
    float inv_det = 1.0f / det;
    
    solution[0] = inv_det * (
        rhs[0] * (matrix[4] * matrix[8] - matrix[7] * matrix[5])
      - rhs[1] * (matrix[1] * matrix[8] - matrix[7] * matrix[2])
      + rhs[2] * (matrix[1] * matrix[5] - matrix[4] * matrix[2])
    );
    
    solution[1] = inv_det * (
        matrix[0] * (rhs[1] * matrix[8] - rhs[2] * matrix[5])
      - matrix[3] * (rhs[0] * matrix[8] - rhs[2] * matrix[2])
      + matrix[6] * (rhs[0] * matrix[5] - rhs[1] * matrix[2])
    );
    
    solution[2] = inv_det * (
        matrix[0] * (matrix[4] * rhs[2] - matrix[7] * rhs[1])
      - matrix[3] * (matrix[1] * rhs[2] - matrix[7] * rhs[0])
      + matrix[6] * (matrix[1] * rhs[1] - matrix[4] * rhs[0])
    );
}

/* Host function for LMFIT processing */
extern "C" int cuda_lmfit_solve(float *y, float *x, float *sig, int ndata,
                                 float *a, int ma, float **covar, float *chisq,
                                 int max_iter, float tolerance)
{
    if (ma > 3) {
        /* For now, only support up to 3 parameters (would need better matrix solver) */
        return -1;
    }
    
    /* Allocate device memory */
    float *d_y, *d_x, *d_sig, *d_a;
    float *d_residuals, *d_chisq_partial, *d_jacobian;
    float *d_alpha, *d_beta;
    
    size_t data_size = ndata * sizeof(float);
    size_t param_size = ma * sizeof(float);
    size_t jacobian_size = ndata * ma * sizeof(float);
    size_t alpha_size = ma * ma * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_y, data_size));
    CUDA_CHECK(cudaMalloc(&d_x, data_size));
    CUDA_CHECK(cudaMalloc(&d_sig, data_size));
    CUDA_CHECK(cudaMalloc(&d_a, param_size));
    CUDA_CHECK(cudaMalloc(&d_residuals, data_size));
    CUDA_CHECK(cudaMalloc(&d_chisq_partial, data_size));
    CUDA_CHECK(cudaMalloc(&d_jacobian, jacobian_size));
    CUDA_CHECK(cudaMalloc(&d_alpha, alpha_size));
    CUDA_CHECK(cudaMalloc(&d_beta, param_size));
    
    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_y, y, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig, sig, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, a, param_size, cudaMemcpyHostToDevice));
    
    float lambda = 0.001f;
    float old_chisq = FLT_MAX;
    
    /* Levenberg-Marquardt iterations */
    for (int iter = 0; iter < max_iter; iter++) {
        /* Calculate residuals and chi-squared */
        int block_size = 256;
        int grid_size = (ndata + block_size - 1) / block_size;
        
        cuda_lmfit_calculate_residuals<<<grid_size, block_size>>>(
            d_y, d_x, d_a, d_sig, ndata, ma, d_residuals, d_chisq_partial);
        CUDA_CHECK(cudaGetLastError());
        
        /* Calculate jacobian */
        dim3 jac_block(16, 16);
        dim3 jac_grid((ndata + jac_block.x - 1) / jac_block.x, (ma + jac_block.y - 1) / jac_block.y);
        
        cuda_lmfit_calculate_jacobian<<<jac_grid, jac_block>>>(
            d_x, d_sig, ndata, ma, d_jacobian);
        CUDA_CHECK(cudaGetLastError());
        
        /* Build normal equations */
        dim3 alpha_block(16, 16);
        dim3 alpha_grid((ma + alpha_block.x - 1) / alpha_block.x, (ma + alpha_block.y - 1) / alpha_block.y);
        
        cuda_lmfit_build_normal_equations<<<alpha_grid, alpha_block>>>(
            d_jacobian, d_residuals, ndata, ma, d_alpha, d_beta);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        /* Copy results to host for solving (small matrices only) */
        float *h_alpha = (float*)malloc(alpha_size);
        float *h_beta = (float*)malloc(param_size);
        float *h_chisq_partial = (float*)malloc(data_size);
        
        CUDA_CHECK(cudaMemcpy(h_alpha, d_alpha, alpha_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_beta, d_beta, param_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_chisq_partial, d_chisq_partial, data_size, cudaMemcpyDeviceToHost));
        
        /* Calculate chi-squared */
        float current_chisq = 0.0f;
        for (int i = 0; i < ndata; i++) {
            current_chisq += h_chisq_partial[i];
        }
        
        /* Check convergence */
        if (iter > 0 && fabsf(old_chisq - current_chisq) < tolerance * current_chisq) {
            *chisq = current_chisq;
            free(h_alpha);
            free(h_beta);
            free(h_chisq_partial);
            break;
        }
        
        /* Augment diagonal elements */
        for (int j = 0; j < ma; j++) {
            h_alpha[j * ma + j] *= (1.0f + lambda);
        }
        
        /* Solve for parameter increments */
        float da[3] = {0}; /* Maximum 3 parameters */
        
        if (ma == 3) {
            cuda_matrix_solve_3x3(h_alpha, h_beta, da);
        } else if (ma == 2) {
            float det = h_alpha[0] * h_alpha[3] - h_alpha[1] * h_alpha[2];
            if (fabsf(det) > 1e-10f) {
                float inv_det = 1.0f / det;
                da[0] = inv_det * (h_alpha[3] * h_beta[0] - h_alpha[1] * h_beta[1]);
                da[1] = inv_det * (h_alpha[0] * h_beta[1] - h_alpha[2] * h_beta[0]);
            }
        } else if (ma == 1) {
            if (fabsf(h_alpha[0]) > 1e-10f) {
                da[0] = h_beta[0] / h_alpha[0];
            }
        }
        
        /* Update parameters */
        for (int j = 0; j < ma; j++) {
            a[j] += da[j];
        }
        
        CUDA_CHECK(cudaMemcpy(d_a, a, param_size, cudaMemcpyHostToDevice));
        
        /* Adjust lambda */
        if (current_chisq < old_chisq) {
            lambda *= 0.1f;
        } else {
            lambda *= 10.0f;
        }
        
        old_chisq = current_chisq;
        *chisq = current_chisq;
        
        free(h_alpha);
        free(h_beta);
        free(h_chisq_partial);
    }
    
    /* Cleanup device memory */
    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_sig);
    cudaFree(d_a);
    cudaFree(d_residuals);
    cudaFree(d_chisq_partial);
    cudaFree(d_jacobian);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    
    return 0; /* Success */
}

/* ====================================================================
 * ACF v1.16 CUDA Kernels
 * ====================================================================*/

__global__ void cuda_acf_calculate_kernel(const int16_t *inbuf,
                                          float *acfbuf, float *xcfbuf,
                                          const int *lagfr, const int *smsep, const int *pat,
                                          int nrang, int mplgs, int mpinc, int nave,
                                          int offset, bool xcf_enabled) {
    int range = blockIdx.x;
    int lag = threadIdx.x;
    
    if (range >= nrang || lag >= mplgs) return;
    
    extern __shared__ int s_pat[];
    if (threadIdx.x < mpinc) {
        s_pat[threadIdx.x] = pat[threadIdx.x];
    }
    __syncthreads();
    
    int pulse1_idx = lagfr[lag * 2] + offset;
    int pulse2_idx = lagfr[lag * 2 + 1] + offset;
    
    float real_sum = 0.0f, imag_sum = 0.0f;
    float xcf_real_sum = 0.0f, xcf_imag_sum = 0.0f;
    
    for (int n = 0; n < nave; n++) {
        for (int p = 0; p < mpinc; p++) {
            if (s_pat[p] == 0) continue;
            
            int sample1_i = (range * nave * mpinc + n * mpinc + pulse1_idx + p) * 2;
            int sample1_q = sample1_i + 1;
            int sample2_i = (range * nave * mpinc + n * mpinc + pulse2_idx + p) * 2;
            int sample2_q = sample2_i + 1;
            
            int16_t i1 = inbuf[sample1_i];
            int16_t q1 = inbuf[sample1_q];
            int16_t i2 = inbuf[sample2_i];
            int16_t q2 = inbuf[sample2_q];
            
            real_sum += i1 * i2 + q1 * q2;
            imag_sum += q1 * i2 - i1 * q2;
            
            if (xcf_enabled) {
                xcf_real_sum += i1 * i2 - q1 * q2;
                xcf_imag_sum += i1 * q2 + q1 * i2;
            }
        }
    }
    
    int idx = range * mplgs + lag;
    acfbuf[idx * 2] = real_sum / (nave * mpinc);
    acfbuf[idx * 2 + 1] = imag_sum / (nave * mpinc);
    
    if (xcf_enabled) {
        xcfbuf[idx * 2] = xcf_real_sum / (nave * mpinc);
        xcfbuf[idx * 2 + 1] = xcf_imag_sum / (nave * mpinc);
    }
}

__global__ void cuda_acf_power_kernel(const float *acfbuf, float *power, int nrang, int mplgs) {
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nrang) return;
    
    int lag0_idx = range * mplgs;
    float real = acfbuf[lag0_idx * 2];
    float imag = acfbuf[lag0_idx * 2 + 1];
    power[range] = sqrtf(real * real + imag * imag);
}

extern "C" int cuda_acf_process(const int16_t *inbuf, float *acfbuf, float *xcfbuf,
                                const int *lagfr, const int *smsep, const int *pat,
                                int nrang, int mplgs, int mpinc, int nave,
                                int offset, bool xcf_enabled) {
    
    size_t inbuf_size = nrang * nave * mpinc * 2 * sizeof(int16_t);
    size_t acf_size = nrang * mplgs * 2 * sizeof(float);
    size_t lag_size = mplgs * 2 * sizeof(int);
    size_t pat_size = mpinc * sizeof(int);
    
    int16_t *d_inbuf;
    float *d_acfbuf, *d_xcfbuf;
    int *d_lagfr, *d_smsep, *d_pat;
    
    CUDA_CHECK(cudaMalloc(&d_inbuf, inbuf_size));
    CUDA_CHECK(cudaMalloc(&d_acfbuf, acf_size));
    CUDA_CHECK(cudaMalloc(&d_xcfbuf, xcf_enabled ? acf_size : sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lagfr, lag_size));
    CUDA_CHECK(cudaMalloc(&d_smsep, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pat, pat_size));
    
    CUDA_CHECK(cudaMemcpy(d_inbuf, inbuf, inbuf_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lagfr, lagfr, lag_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_smsep, smsep, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pat, pat, pat_size, cudaMemcpyHostToDevice));
    
    dim3 grid(nrang);
    dim3 block(mplgs);
    size_t shared_size = mpinc * sizeof(int);
    
    cuda_acf_calculate_kernel<<<grid, block, shared_size>>>(
        d_inbuf, d_acfbuf, d_xcfbuf, d_lagfr, d_smsep, d_pat,
        nrang, mplgs, mpinc, nave, offset, xcf_enabled);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(acfbuf, d_acfbuf, acf_size, cudaMemcpyDeviceToHost));
    if (xcf_enabled) {
        CUDA_CHECK(cudaMemcpy(xcfbuf, d_xcfbuf, acf_size, cudaMemcpyDeviceToHost));
    }
    
    cudaFree(d_inbuf);
    cudaFree(d_acfbuf);
    cudaFree(d_xcfbuf);
    cudaFree(d_lagfr);
    cudaFree(d_smsep);
    cudaFree(d_pat);
    
    return 0;
}

/* ====================================================================
 * IQ v1.7 CUDA Kernels
 * ====================================================================*/

__global__ void cuda_iq_time_convert_kernel(const double *input_time,
                                            long *tv_sec, long *tv_nsec,
                                            int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    
    double time_val = input_time[idx];
    tv_sec[idx] = (long)time_val;
    tv_nsec[idx] = (long)((time_val - (double)tv_sec[idx]) * 1e9);
}

__global__ void cuda_iq_encode_kernel(const float *iq_samples, int16_t *encoded_data,
                                      float scale_factor, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    
    float scaled_val = iq_samples[idx] * scale_factor;
    scaled_val = fmaxf(-32768.0f, fminf(32767.0f, scaled_val));
    encoded_data[idx] = (int16_t)roundf(scaled_val);
}

extern "C" int cuda_iq_process_time_series(const double *input_time, const float *iq_data,
                                           long *tv_sec, long *tv_nsec, int16_t *encoded_iq,
                                           int num_samples, float scale_factor) {
    
    size_t time_size = num_samples * sizeof(double);
    size_t iq_size = num_samples * 2 * sizeof(float);
    size_t tv_size = num_samples * sizeof(long);
    size_t encoded_size = num_samples * 2 * sizeof(int16_t);
    
    double *d_input_time;
    float *d_iq_data;
    long *d_tv_sec, *d_tv_nsec;
    int16_t *d_encoded_iq;
    
    CUDA_CHECK(cudaMalloc(&d_input_time, time_size));
    CUDA_CHECK(cudaMalloc(&d_iq_data, iq_size));
    CUDA_CHECK(cudaMalloc(&d_tv_sec, tv_size));
    CUDA_CHECK(cudaMalloc(&d_tv_nsec, tv_size));
    CUDA_CHECK(cudaMalloc(&d_encoded_iq, encoded_size));
    
    CUDA_CHECK(cudaMemcpy(d_input_time, input_time, time_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iq_data, iq_data, iq_size, cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;
    
    cuda_iq_time_convert_kernel<<<grid_size, block_size>>>(
        d_input_time, d_tv_sec, d_tv_nsec, num_samples);
    
    cuda_iq_encode_kernel<<<grid_size * 2, block_size>>>(
        d_iq_data, d_encoded_iq, scale_factor, num_samples * 2);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(tv_sec, d_tv_sec, tv_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tv_nsec, d_tv_nsec, tv_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(encoded_iq, d_encoded_iq, encoded_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input_time);
    cudaFree(d_iq_data);
    cudaFree(d_tv_sec);
    cudaFree(d_tv_nsec);
    cudaFree(d_encoded_iq);
    
    return 0;
}

/* ====================================================================
 * CNVMAP v1.17 CUDA Kernels
 * ====================================================================*/

__global__ void cuda_legendre_eval_kernel(int lmax, const double *x,
                                          double *plm, int n_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    
    double xi = x[idx];
    double p_mm = 1.0;
    
    for (int m = 0; m <= lmax; m++) {
        if (m > 0) {
            double somx2 = sqrt((1.0 - xi) * (1.0 + xi));
            double fact = 1.0;
            for (int i = 1; i <= m; i++) {
                fact *= (2 * i - 1);
            }
            p_mm = pow(-1.0, m) * fact * pow(somx2, m);
        }
        
        plm[idx * (lmax + 1) * (lmax + 2) / 2 + m * (lmax + 1) + m] = p_mm;
        
        if (m < lmax) {
            double p_mp1m = xi * (2 * m + 1) * p_mm;
            plm[idx * (lmax + 1) * (lmax + 2) / 2 + m * (lmax + 1) + m + 1] = p_mp1m;
            
            for (int l = m + 2; l <= lmax; l++) {
                double p_lm = (xi * (2 * l - 1) * p_mp1m - (l + m - 1) * p_mm) / (l - m);
                plm[idx * (lmax + 1) * (lmax + 2) / 2 + m * (lmax + 1) + l] = p_lm;
                p_mm = p_mp1m;
                p_mp1m = p_lm;
            }
        }
    }
}

/* Kernel: transform raw colatitude (radians) to cos(colatitude) in-place */
__global__ void cuda_cos_transform_kernel(const double *theta, double *cos_theta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) cos_theta[idx] = cos(theta[idx]);
}

extern "C" int cuda_cnvmap_spherical_harmonic_fit(const double *theta, const double *phi,
                                                   const double *v_los, int n_points,
                                                   double *coefficients, int lmax) {

    /* n_coeff: number of real spherical harmonic coefficients for degree lmax */
    int n_coeff    = (lmax + 1) * (lmax + 1);
    int plm_cols   = (lmax + 1) * (lmax + 2) / 2;

    size_t points_size = n_points * sizeof(double);
    size_t coeff_size  = n_coeff  * sizeof(double);
    size_t plm_size    = (size_t)n_points * plm_cols * sizeof(double);

    double *d_theta, *d_cos_theta, *d_phi, *d_v_los, *d_plm;

    CUDA_CHECK(cudaMalloc(&d_theta,     points_size));
    CUDA_CHECK(cudaMalloc(&d_cos_theta, points_size));
    CUDA_CHECK(cudaMalloc(&d_phi,       points_size));
    CUDA_CHECK(cudaMalloc(&d_v_los,     points_size));
    CUDA_CHECK(cudaMalloc(&d_plm,       plm_size));

    CUDA_CHECK(cudaMemcpy(d_theta, theta, points_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi,   phi,   points_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_los, v_los, points_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size  = (n_points + block_size - 1) / block_size;

    /* Step 1: convert colatitude → cos(colatitude) — Legendre polys need cos(θ) */
    cuda_cos_transform_kernel<<<grid_size, block_size>>>(d_theta, d_cos_theta, n_points);
    CUDA_CHECK(cudaGetLastError());

    /* Step 2: evaluate P_l^m(cos θ) for all points */
    cuda_legendre_eval_kernel<<<grid_size, block_size>>>(lmax, d_cos_theta, d_plm, n_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Step 3: copy P_lm and observations to host, then solve A·c = v_los
     * via normal equations (A^T A) c = A^T v_los on the CPU.
     * The design matrix A[i, col] = P_lm(cos θ_i) * {cos(m·φ_i), m≥0; sin(|m|·φ_i), m<0}.
     * This CPU solve is exact for small n_coeff (typically ≤ 100). */
    double *h_plm  = (double*)malloc(plm_size);
    double *h_phi  = (double*)malloc(points_size);
    double *h_vlos = (double*)malloc(points_size);
    if (!h_plm || !h_phi || !h_vlos) {
        free(h_plm); free(h_phi); free(h_vlos);
        cudaFree(d_theta); cudaFree(d_cos_theta); cudaFree(d_phi);
        cudaFree(d_v_los); cudaFree(d_plm);
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(h_plm,  d_plm,  plm_size,    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_phi,  d_phi,  points_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vlos, d_v_los,points_size, cudaMemcpyDeviceToHost));

    /* Build design matrix on heap (n_points × n_coeff) */
    double *A  = (double*)calloc((size_t)n_points * n_coeff, sizeof(double));
    double *AtA = (double*)calloc((size_t)n_coeff  * n_coeff, sizeof(double));
    double *Atb = (double*)calloc(n_coeff, sizeof(double));
    if (!A || !AtA || !Atb) {
        free(h_plm); free(h_phi); free(h_vlos); free(A); free(AtA); free(Atb);
        cudaFree(d_theta); cudaFree(d_cos_theta); cudaFree(d_phi);
        cudaFree(d_v_los); cudaFree(d_plm);
        return -1;
    }

    int col = 0;
    for (int l = 0; l <= lmax; l++) {
        /* m = 0: Y_l0 = P_l^0(cos θ) */
        for (int i = 0; i < n_points; i++) {
            double plm_val = h_plm[i * plm_cols + 0 * (lmax + 1) + l];
            A[i * n_coeff + col] = plm_val;
        }
        col++;
        for (int m = 1; m <= l; m++) {
            /* cos branch */
            for (int i = 0; i < n_points; i++) {
                double plm_val = h_plm[i * plm_cols + m * (lmax + 1) + l];
                A[i * n_coeff + col] = plm_val * cos(m * h_phi[i]);
            }
            col++;
            /* sin branch */
            for (int i = 0; i < n_points; i++) {
                double plm_val = h_plm[i * plm_cols + m * (lmax + 1) + l];
                A[i * n_coeff + col] = plm_val * sin(m * h_phi[i]);
            }
            col++;
        }
    }

    /* Normal equations: AtA = A^T A,  Atb = A^T v_los */
    for (int j = 0; j < n_coeff; j++) {
        for (int k = 0; k < n_coeff; k++) {
            double s = 0.0;
            for (int i = 0; i < n_points; i++)
                s += A[i * n_coeff + j] * A[i * n_coeff + k];
            AtA[j * n_coeff + k] = s;
        }
        double s = 0.0;
        for (int i = 0; i < n_points; i++)
            s += A[i * n_coeff + j] * h_vlos[i];
        Atb[j] = s;
    }

    /* Gaussian elimination with partial pivoting */
    for (int i = 0; i < n_coeff; i++) {
        int pivot = i;
        double max_val = fabs(AtA[i * n_coeff + i]);
        for (int j = i + 1; j < n_coeff; j++) {
            if (fabs(AtA[j * n_coeff + i]) > max_val) {
                max_val = fabs(AtA[j * n_coeff + i]);
                pivot = j;
            }
        }
        if (pivot != i) {
            for (int k = 0; k < n_coeff; k++) {
                double tmp = AtA[i * n_coeff + k];
                AtA[i * n_coeff + k] = AtA[pivot * n_coeff + k];
                AtA[pivot * n_coeff + k] = tmp;
            }
            double tmp = Atb[i]; Atb[i] = Atb[pivot]; Atb[pivot] = tmp;
        }
        double diag = AtA[i * n_coeff + i];
        if (fabs(diag) < 1e-14) { coefficients[i] = 0.0; continue; }
        for (int j = i + 1; j < n_coeff; j++) {
            double f = AtA[j * n_coeff + i] / diag;
            for (int k = i; k < n_coeff; k++)
                AtA[j * n_coeff + k] -= f * AtA[i * n_coeff + k];
            Atb[j] -= f * Atb[i];
        }
    }
    for (int i = n_coeff - 1; i >= 0; i--) {
        double val = Atb[i];
        for (int j = i + 1; j < n_coeff; j++)
            val -= AtA[i * n_coeff + j] * coefficients[j];
        double diag = AtA[i * n_coeff + i];
        coefficients[i] = (fabs(diag) > 1e-14) ? val / diag : 0.0;
    }

    free(h_plm); free(h_phi); free(h_vlos);
    free(A); free(AtA); free(Atb);
    cudaFree(d_theta); cudaFree(d_cos_theta); cudaFree(d_phi);
    cudaFree(d_v_los); cudaFree(d_plm);

    return 0;
}

/* ====================================================================
 * GRID v1.24 CUDA Kernels
 * ====================================================================*/

__global__ void cuda_grid_locate_cell_kernel(const float *x_data, const float *y_data,
                                             const float *grid_x, const float *grid_y,
                                             int *cell_indices, int n_points,
                                             int grid_nx, int grid_ny, float cell_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    
    float x = x_data[idx];
    float y = y_data[idx];
    
    int grid_i = (int)floorf((x - grid_x[0]) / cell_size);
    int grid_j = (int)floorf((y - grid_y[0]) / cell_size);
    
    if (grid_i >= 0 && grid_i < grid_nx && grid_j >= 0 && grid_j < grid_ny) {
        cell_indices[idx] = grid_j * grid_nx + grid_i;
    } else {
        cell_indices[idx] = -1;
    }
}

extern "C" int cuda_grid_interpolate_data(const float *x_data, const float *y_data,
                                          const float *values, int n_points,
                                          const float *grid_x, const float *grid_y,
                                          float *grid_values, int grid_nx, int grid_ny,
                                          float cell_size) {

    size_t data_size  = n_points * sizeof(float);
    size_t gsize      = (size_t)grid_nx * grid_ny * sizeof(float);
    size_t index_size = n_points * sizeof(int);

    float *d_x_data, *d_y_data, *d_values, *d_grid_x, *d_grid_y, *d_grid_values;
    int   *d_cell_indices;

    CUDA_CHECK(cudaMalloc(&d_x_data,       data_size));
    CUDA_CHECK(cudaMalloc(&d_y_data,       data_size));
    CUDA_CHECK(cudaMalloc(&d_values,       data_size));
    CUDA_CHECK(cudaMalloc(&d_grid_x,       grid_nx * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_y,       grid_ny * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_values,  gsize));
    CUDA_CHECK(cudaMalloc(&d_cell_indices, index_size));

    /* Initialise grid to NaN so unfilled cells are distinguishable */
    {
        float *h_nan = (float*)malloc(gsize);
        for (int i = 0; i < grid_nx * grid_ny; i++) h_nan[i] = __builtin_nanf("");
        CUDA_CHECK(cudaMemcpy(d_grid_values, h_nan, gsize, cudaMemcpyHostToDevice));
        free(h_nan);
    }

    CUDA_CHECK(cudaMemcpy(d_x_data, x_data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_data, y_data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_x, grid_x, grid_nx * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_y, grid_y, grid_ny * sizeof(float), cudaMemcpyHostToDevice));

    int block_size  = 256;
    int grid_size_1d = (n_points + block_size - 1) / block_size;

    /* Step 1: compute cell index for every data point */
    cuda_grid_locate_cell_kernel<<<grid_size_1d, block_size>>>(
        d_x_data, d_y_data, d_grid_x, d_grid_y, d_cell_indices,
        n_points, grid_nx, grid_ny, cell_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Step 2: scatter-average on host (grid is small; avoids atomics in CUDA) */
    int *h_indices   = (int*)  malloc(index_size);
    float *h_values  = (float*)malloc(data_size);
    float *h_sum     = (float*)calloc(grid_nx * grid_ny, sizeof(float));
    int   *h_cnt     = (int*)  calloc(grid_nx * grid_ny, sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_indices, d_cell_indices, index_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values,  d_values,       data_size,  cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_points; i++) {
        int ci = h_indices[i];
        if (ci >= 0 && ci < grid_nx * grid_ny) {
            h_sum[ci] += h_values[i];
            h_cnt[ci]++;
        }
    }

    float nan_val = __builtin_nanf("");
    for (int i = 0; i < grid_nx * grid_ny; i++) {
        grid_values[i] = (h_cnt[i] > 0) ? h_sum[i] / h_cnt[i] : nan_val;
    }

    free(h_indices); free(h_values); free(h_sum); free(h_cnt);
    cudaFree(d_x_data);
    cudaFree(d_y_data);
    cudaFree(d_values);
    cudaFree(d_grid_x);
    cudaFree(d_grid_y);
    cudaFree(d_grid_values);
    cudaFree(d_cell_indices);

    return 0;
}

/* ====================================================================
 * Lag Validity Kernels  (Fix 1 — vectorised bad-lag detection)
 * Replaces RST's per-range linked-list walk with two GPU passes:
 *   Pass 1: fully parallel per-element threshold check → uint8 bad[] array
 *   Pass 2: per-range sequential cumulative-OR → propagate cutoff forward
 * ====================================================================*/

__global__ void cuda_compute_lag_bad_kernel(
    const float *acf_real, const float *acf_imag,
    const float *pwr0, float noise_pwr,
    int nave, int nrang, int mplgs,
    uint8_t *bad)   /* (nrang, mplgs): 1 = bad, 0 = good */
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= nrang || l >= mplgs) return;

    int idx = r * mplgs + l;
    if (l == 0) { bad[idx] = 0; return; }  /* lag-0 always valid */

    float re  = acf_real[idx];
    float im  = acf_imag[idx];
    float mag = sqrtf(re*re + im*im);

    if (mag <= noise_pwr) { bad[idx] = 1; return; }

    float p0      = pwr0[r] + 1e-30f;
    float R_norm  = mag / p0;
    float alpha_2 = (float)nave / (1.0f + (float)nave * R_norm * R_norm + 1e-30f);

    float fluct_sigma = 2.0f * p0 / (sqrtf(2.0f * (float)max(nave, 1)) + 1e-30f);
    float log_R       = logf(mag + 1e-30f);
    float log_fluct   = logf(fluct_sigma + 1e-30f);

    bad[idx] = (alpha_2 < 0.25f && log_R <= log_fluct) ? 1 : 0;
}

/* One thread per range: sequential cumulative-OR along lags. */
__global__ void cuda_propagate_lag_cutoff_kernel(
    uint8_t *bad, int nrang, int mplgs)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrang) return;

    int base = r * mplgs;
    uint8_t seen = 0;
    for (int l = 1; l < mplgs; l++) {
        seen |= bad[base + l];
        bad[base + l] = seen;
    }
}

extern "C" int cuda_compute_lag_validity(
    const float *acf_real, const float *acf_imag,
    const float *pwr0, float noise_pwr,
    int nave, int nrang, int mplgs,
    uint8_t *valid_out)   /* host output: 1=valid, 0=invalid */
{
    size_t sz = (size_t)nrang * mplgs;
    uint8_t *d_bad;
    CUDA_CHECK(cudaMalloc(&d_bad, sz));

    dim3 block(16, 16);
    dim3 grid((nrang + 15) / 16, (mplgs + 15) / 16);
    cuda_compute_lag_bad_kernel<<<grid, block>>>(
        acf_real, acf_imag, pwr0, noise_pwr, nave, nrang, mplgs, d_bad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int pblock = 256;
    cuda_propagate_lag_cutoff_kernel<<<(nrang + pblock - 1) / pblock, pblock>>>(
        d_bad, nrang, mplgs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t *h_bad = (uint8_t *)malloc(sz);
    CUDA_CHECK(cudaMemcpy(h_bad, d_bad, sz, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < sz; i++) valid_out[i] = h_bad[i] ? 0 : 1;
    free(h_bad);
    cudaFree(d_bad);
    return 0;
}

/* ====================================================================
 * Bendat-Piersol Sigma Kernel  (Fix 2)
 * Fully parallel over all (nrang, mplgs) elements.
 * ====================================================================*/

__global__ void cuda_bp_sigma_kernel(
    const float *acf_real, const float *acf_imag,
    const float *pwr0, int nave,
    int nrang, int mplgs,
    float *sigma_out)   /* (nrang, mplgs) */
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= nrang || l >= mplgs) return;

    int   idx  = r * mplgs + l;
    float re   = acf_real[idx];
    float im   = acf_imag[idx];
    float mag  = sqrtf(re*re + im*im);
    float p0   = pwr0[r] + 1e-30f;
    float Rn   = mag / p0;
    float a2   = (float)nave / (1.0f + (float)nave * Rn * Rn + 1e-30f);
    float sigma = p0 * sqrtf((Rn*Rn + 1.0f / (a2 + 1e-30f)) /
                              (2.0f * (float)max(nave, 1)));
    sigma_out[idx] = fmaxf(sigma, 1e-10f);
}

extern "C" int cuda_compute_bp_sigma(
    const float *acf_real, const float *acf_imag,
    const float *pwr0, int nave,
    int nrang, int mplgs,
    float *sigma_out)
{
    dim3 block(16, 16);
    dim3 grid((nrang + 15) / 16, (mplgs + 15) / 16);
    cuda_bp_sigma_kernel<<<grid, block>>>(
        acf_real, acf_imag, pwr0, nave, nrang, mplgs, sigma_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ====================================================================
 * Two-pass error estimation kernel  (Fix 3)
 * One thread per range; iterates over valid lags to solve weighted
 * log-linear normal equations → velocity_error, power_error.
 * ====================================================================*/

__global__ void cuda_fitacf_error_kernel(
    const float *acf_real, const float *acf_imag,
    const float *pwr0, const float *bp_sigma,
    const uint8_t *valid_mask,
    int nrang, int mplgs,
    float vel_factor, float lag_dt,
    float *vel_err_out, float *pwr_err_out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrang) return;

    int   base = r * mplgs;
    float p0   = pwr0[r] + 1e-30f;

    float S = 0, St = 0, Stt = 0, Sy = 0, Sty = 0;
    for (int l = 0; l < mplgs; l++) {
        if (!valid_mask[base + l]) continue;
        float re  = acf_real[base + l];
        float im  = acf_imag[base + l];
        float mag = sqrtf(re*re + im*im) + 1e-30f;
        float sig = bp_sigma[base + l];
        float lsig = sig / (mag + 1e-30f);
        float w   = 1.0f / (lsig * lsig + 1e-30f);
        float t   = (float)l * lag_dt;
        float y   = logf(mag);
        S   += w; St  += w*t; Stt += w*t*t; Sy  += w*y; Sty += w*t*y;
    }
    float delta  = S * Stt - St * St + 1e-30f;
    float sigma_b = sqrtf(S / delta);
    float sigma_a = sqrtf(Stt / delta);
    vel_err_out[r] = sigma_b * vel_factor;
    pwr_err_out[r] = p0 * sigma_a;
}

extern "C" int cuda_fitacf_error(
    const float *acf_real, const float *acf_imag,
    const float *pwr0, const float *bp_sigma,
    const uint8_t *valid_mask,
    int nrang, int mplgs,
    float vel_factor, float lag_dt,
    float *vel_err_out, float *pwr_err_out)
{
    int block = 256;
    cuda_fitacf_error_kernel<<<(nrang + block - 1) / block, block>>>(
        acf_real, acf_imag, pwr0, bp_sigma, valid_mask,
        nrang, mplgs, vel_factor, lag_dt, vel_err_out, pwr_err_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ====================================================================
 * Sigma spectral width kernel  (Fix 4)
 * Quadratic log fit: log|R| = a + c*t^2  → w_s = sqrt(|c|) * vel_factor * ...
 * One thread per range.
 * ====================================================================*/

__global__ void cuda_fitacf_sigma_width_kernel(
    const float *acf_real, const float *acf_imag,
    const float *bp_sigma, const uint8_t *valid_mask,
    int nrang, int mplgs,
    float vel_factor, float lag_dt,
    float *width_sigma_out, float *width_sigma_err_out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrang) return;

    int base = r * mplgs;
    float S = 0, St2 = 0, St4 = 0, Sy = 0, St2y = 0;
    for (int l = 0; l < mplgs; l++) {
        if (!valid_mask[base + l]) continue;
        float re  = acf_real[base + l];
        float im  = acf_imag[base + l];
        float mag = sqrtf(re*re + im*im) + 1e-30f;
        float sig = bp_sigma[base + l];
        float lsig = sig / (mag + 1e-30f);
        float w   = 1.0f / (lsig * lsig + 1e-30f);
        float t   = (float)l * lag_dt;
        float t2  = t * t;
        float y   = logf(mag);
        S += w; St2 += w*t2; St4 += w*t2*t2; Sy += w*y; St2y += w*t2*y;
    }
    float det = S * St4 - St2 * St2 + 1e-30f;
    float c   = (S * St2y - St2 * Sy) / det;
    float fwhm_factor = 4.0f * sqrtf(logf(2.0f));  /* ≈ 2.355 */
    float W_S = sqrtf(fabsf(c)) * vel_factor * lag_dt * fwhm_factor;
    width_sigma_out[r] = fminf(W_S, 1000.0f);

    float sigma_c = sqrtf(S / det);
    width_sigma_err_out[r] = (fabsf(c) > 1e-10f)
        ? 0.5f * sigma_c / sqrtf(fabsf(c)) * vel_factor * lag_dt * fwhm_factor
        : 0.0f;
}

extern "C" int cuda_fitacf_sigma_width(
    const float *acf_real, const float *acf_imag,
    const float *bp_sigma, const uint8_t *valid_mask,
    int nrang, int mplgs,
    float vel_factor, float lag_dt,
    float *width_sigma_out, float *width_sigma_err_out)
{
    int block = 256;
    cuda_fitacf_sigma_width_kernel<<<(nrang + block - 1) / block, block>>>(
        acf_real, acf_imag, bp_sigma, valid_mask,
        nrang, mplgs, vel_factor, lag_dt,
        width_sigma_out, width_sigma_err_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}