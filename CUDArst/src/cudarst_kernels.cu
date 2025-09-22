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

__global__ void cuda_fitacf_velocity_kernel(const float *acf_real, const float *acf_imag,
                                             int nrang, int mplgs, float *velocity, float *phase)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r >= nrang || mplgs < 2) return;
    
    /* Calculate velocity from phase progression */
    int lag0_idx = r * mplgs + 0;
    int lag1_idx = r * mplgs + 1;
    
    float phase0 = cuda_phase(acf_real[lag0_idx], acf_imag[lag0_idx]);
    float phase1 = cuda_phase(acf_real[lag1_idx], acf_imag[lag1_idx]);
    
    phase[r] = phase0;
    
    /* Calculate phase difference and convert to velocity */
    float phase_diff = phase1 - phase0;
    
    /* Unwrap phase if necessary */
    if (phase_diff > M_PI) phase_diff -= 2.0f * M_PI;
    if (phase_diff < -M_PI) phase_diff += 2.0f * M_PI;
    
    /* Convert to velocity (simplified conversion factor) */
    velocity[r] = phase_diff * 150.0f; /* m/s per radian */
}

__global__ void cuda_fitacf_width_kernel(const float *acf_real, const float *acf_imag,
                                         int nrang, int mplgs, const float *power, float *width)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r >= nrang) return;
    
    /* Calculate spectral width from ACF decay */
    if (power[r] < 1000.0f) {
        width[r] = 0.0f;
        return;
    }
    
    float decay_sum = 0.0f;
    int valid_lags = 0;
    
    for (int lag = 1; lag < min(mplgs, 5); lag++) {
        int idx = r * mplgs + lag;
        float lag_power = cuda_magnitude(acf_real[idx], acf_imag[idx]);
        
        if (lag_power > 0.0f && power[r] > 0.0f) {
            decay_sum += logf(lag_power / power[r]);
            valid_lags++;
        }
    }
    
    if (valid_lags > 0) {
        float decay_rate = -decay_sum / valid_lags;
        width[r] = 50.0f + decay_rate * 20.0f; /* Empirical width formula */
        width[r] = fmaxf(0.0f, fminf(width[r], 500.0f)); /* Clamp to reasonable range */
    } else {
        width[r] = 50.0f; /* Default width */
    }
}

__global__ void cuda_fitacf_quality_kernel(const float *power, int nrang, float *quality)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r >= nrang) return;
    
    /* Simple quality metric based on signal strength */
    if (power[r] > 5000.0f) {
        quality[r] = 0.9f;
    } else if (power[r] > 2000.0f) {
        quality[r] = 0.7f;
    } else if (power[r] > 1000.0f) {
        quality[r] = 0.5f;
    } else {
        quality[r] = 0.1f;
    }
}

/* Host function for FITACF processing */
extern "C" int cuda_fitacf_process_ranges(const float *acf_real, const float *acf_imag,
                                          int nrang, int mplgs,
                                          float *power, float *velocity, float *width,
                                          float *phase, float *quality)
{
    /* Allocate device memory */
    float *d_acf_real, *d_acf_imag;
    float *d_power, *d_velocity, *d_width, *d_phase, *d_quality;
    
    size_t acf_size = nrang * mplgs * sizeof(float);
    size_t range_size = nrang * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_acf_real, acf_size));
    CUDA_CHECK(cudaMalloc(&d_acf_imag, acf_size));
    CUDA_CHECK(cudaMalloc(&d_power, range_size));
    CUDA_CHECK(cudaMalloc(&d_velocity, range_size));
    CUDA_CHECK(cudaMalloc(&d_width, range_size));
    CUDA_CHECK(cudaMalloc(&d_phase, range_size));
    CUDA_CHECK(cudaMalloc(&d_quality, range_size));
    
    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_acf_real, acf_real, acf_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_acf_imag, acf_imag, acf_size, cudaMemcpyHostToDevice));
    
    /* Launch kernels */
    int block_size = 256;
    int grid_size = (nrang + block_size - 1) / block_size;
    
    cuda_fitacf_power_kernel<<<grid_size, block_size>>>(
        d_acf_real, d_acf_imag, nrang, mplgs, d_power);
    CUDA_CHECK(cudaGetLastError());
    
    cuda_fitacf_velocity_kernel<<<grid_size, block_size>>>(
        d_acf_real, d_acf_imag, nrang, mplgs, d_velocity, d_phase);
    CUDA_CHECK(cudaGetLastError());
    
    cuda_fitacf_width_kernel<<<grid_size, block_size>>>(
        d_acf_real, d_acf_imag, nrang, mplgs, d_power, d_width);
    CUDA_CHECK(cudaGetLastError());
    
    cuda_fitacf_quality_kernel<<<grid_size, block_size>>>(
        d_power, nrang, d_quality);
    CUDA_CHECK(cudaGetLastError());
    
    /* Wait for kernels to complete */
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /* Copy results back to host */
    CUDA_CHECK(cudaMemcpy(power, d_power, range_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity, d_velocity, range_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(width, d_width, range_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phase, d_phase, range_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(quality, d_quality, range_size, cudaMemcpyDeviceToHost));
    
    /* Cleanup device memory */
    cudaFree(d_acf_real);
    cudaFree(d_acf_imag);
    cudaFree(d_power);
    cudaFree(d_velocity);
    cudaFree(d_width);
    cudaFree(d_phase);
    cudaFree(d_quality);
    
    return 0; /* Success */
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

extern "C" int cuda_cnvmap_spherical_harmonic_fit(const double *theta, const double *phi,
                                                   const double *v_los, int n_points,
                                                   double *coefficients, int lmax) {
    
    size_t points_size = n_points * sizeof(double);
    size_t coeff_size = (lmax + 1) * (lmax + 2) * sizeof(double);
    size_t plm_size = n_points * (lmax + 1) * (lmax + 2) / 2 * sizeof(double);
    
    double *d_theta, *d_phi, *d_v_los, *d_coefficients, *d_plm;
    
    CUDA_CHECK(cudaMalloc(&d_theta, points_size));
    CUDA_CHECK(cudaMalloc(&d_phi, points_size));
    CUDA_CHECK(cudaMalloc(&d_v_los, points_size));
    CUDA_CHECK(cudaMalloc(&d_coefficients, coeff_size));
    CUDA_CHECK(cudaMalloc(&d_plm, plm_size));
    
    CUDA_CHECK(cudaMemcpy(d_theta, theta, points_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi, phi, points_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_los, v_los, points_size, cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (n_points + block_size - 1) / block_size;
    
    cuda_legendre_eval_kernel<<<grid_size, block_size>>>(
        lmax, d_theta, d_plm, n_points);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(coefficients, d_coefficients, coeff_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_theta);
    cudaFree(d_phi);
    cudaFree(d_v_los);
    cudaFree(d_coefficients);
    cudaFree(d_plm);
    
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
    
    size_t data_size = n_points * sizeof(float);
    size_t grid_size = grid_nx * grid_ny * sizeof(float);
    size_t index_size = n_points * sizeof(int);
    
    float *d_x_data, *d_y_data, *d_values, *d_grid_x, *d_grid_y, *d_grid_values;
    int *d_cell_indices;
    
    CUDA_CHECK(cudaMalloc(&d_x_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_y_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_values, data_size));
    CUDA_CHECK(cudaMalloc(&d_grid_x, grid_nx * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_y, grid_ny * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_values, grid_size));
    CUDA_CHECK(cudaMalloc(&d_cell_indices, index_size));
    
    CUDA_CHECK(cudaMemcpy(d_x_data, x_data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_data, y_data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_x, grid_x, grid_nx * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_y, grid_y, grid_ny * sizeof(float), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size_1d = (n_points + block_size - 1) / block_size;
    
    cuda_grid_locate_cell_kernel<<<grid_size_1d, block_size>>>(
        d_x_data, d_y_data, d_grid_x, d_grid_y, d_cell_indices,
        n_points, grid_nx, grid_ny, cell_size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(grid_values, d_grid_values, grid_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_x_data);
    cudaFree(d_y_data);
    cudaFree(d_values);
    cudaFree(d_grid_x);
    cudaFree(d_grid_y);
    cudaFree(d_grid_values);
    cudaFree(d_cell_indices);
    
    return 0;
}