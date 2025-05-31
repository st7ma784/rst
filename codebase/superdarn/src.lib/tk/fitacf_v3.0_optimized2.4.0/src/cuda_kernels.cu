/*
 * CUDA kernels for SuperDARN FitACF v3.0_optimized2
 * 
 * This module implements GPU kernels for massive parallelization
 * of FitACF processing algorithms.
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <math_constants.h>

#include "fitting_optimized.h"
#include "fit_structures_optimized.h"

/* CUDA block and grid sizes */
#define CUDA_THREADS_PER_BLOCK 256
#define CUDA_MAX_BLOCKS 65535

/*
 * CUDA kernel for parallel ACF preprocessing
 * Processes multiple ranges simultaneously on GPU
 */
__global__ void PreprocessACFKernel(cuDoubleComplex *acfd_data, cuDoubleComplex *xcfd_data,
                                    double *power_data, double *noise_data,
                                    int *range_status, double *phase_data,
                                    int nranges, int mplgs, double lagfr, double smsep)
{
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nranges) return;
    
    /* Initialize range status */
    range_status[range] = 0;
    
    /* Check power threshold */
    if (power_data[range] <= 0.0) return;
    
    /* Count valid lags and calculate statistics */
    int valid_lags = 0;
    double total_power = 0.0;
    double max_power = 0.0;
    
    for (int lag = 0; lag < mplgs; lag++) {
        int idx = range * mplgs + lag;
        double power = cuCabs(acfd_data[idx]);
        
        if (power > 0.0) {
            valid_lags++;
            total_power += power;
            if (power > max_power) max_power = power;
            
            /* Calculate and store phase */
            phase_data[idx] = cuCarg(acfd_data[idx]);
        } else {
            phase_data[idx] = 0.0;
        }
    }
    
    /* Set range status based on validity criteria */
    if (valid_lags >= 3 && max_power > noise_data[range] * 2.0) {
        range_status[range] = 1;
    }
}

/*
 * CUDA kernel for phase unwrapping
 * Performs parallel phase unwrapping across all ranges
 */
__global__ void PhaseUnwrapKernel(double *phase_data, cuDoubleComplex *acfd_data,
                                  int *range_status, int nranges, int mplgs)
{
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nranges || !range_status[range]) return;
    
    /* Phase unwrapping for this range */
    double prev_phase = 0.0;
    double cumulative_offset = 0.0;
    
    for (int lag = 0; lag < mplgs; lag++) {
        int idx = range * mplgs + lag;
        
        if (cuCabs(acfd_data[idx]) > 0.0) {
            double raw_phase = cuCarg(acfd_data[idx]);
            
            if (lag > 0) {
                double phase_diff = raw_phase - prev_phase;
                
                /* Handle 2π jumps */
                if (phase_diff > CUDART_PI) {
                    cumulative_offset -= 2.0 * CUDART_PI;
                } else if (phase_diff < -CUDART_PI) {
                    cumulative_offset += 2.0 * CUDART_PI;
                }
            }
            
            /* Store unwrapped phase */
            phase_data[idx] = raw_phase + cumulative_offset;
            prev_phase = raw_phase;
        }
    }
}

/*
 * CUDA kernel for velocity calculation using linear regression
 * Parallel fitting of phase slopes for velocity estimation
 */
__global__ void VelocityFitKernel(cuDoubleComplex *acfd_data, double *phase_data,
                                  int *range_status, double *velocity_out,
                                  double *quality_out, int nranges, int mplgs,
                                  double lagfr, double tfreq)
{
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nranges || !range_status[range]) {
        if (range < nranges) {
            velocity_out[range] = 0.0;
            quality_out[range] = 0.0;
        }
        return;
    }
    
    /* Collect valid phase and lag data */
    double sum_phase = 0.0, sum_lag = 0.0;
    double sum_phase_lag = 0.0, sum_lag_sq = 0.0;
    int valid_points = 0;
    
    for (int lag = 0; lag < mplgs; lag++) {
        int idx = range * mplgs + lag;
        
        if (cuCabs(acfd_data[idx]) > 0.0) {
            double phase = phase_data[idx];
            double lag_time = lag * lagfr * 1e-6; /* Convert to seconds */
            
            sum_phase += phase;
            sum_lag += lag_time;
            sum_phase_lag += phase * lag_time;
            sum_lag_sq += lag_time * lag_time;
            valid_points++;
        }
    }
    
    /* Calculate velocity using least squares */
    double velocity = 0.0;
    double quality = 0.0;
    
    if (valid_points >= 3) {
        double denominator = valid_points * sum_lag_sq - sum_lag * sum_lag;
        
        if (fabs(denominator) > 1e-10) {
            double phase_slope = (valid_points * sum_phase_lag - sum_lag * sum_phase) / denominator;
            velocity = phase_slope * 2.998e8 / (4.0 * CUDART_PI * tfreq * 1e6);
            
            /* Calculate fit quality (correlation coefficient) */
            double mean_phase = sum_phase / valid_points;
            double mean_lag = sum_lag / valid_points;
            
            double ss_phase = 0.0, ss_lag = 0.0, ss_cross = 0.0;
            
            for (int lag = 0; lag < mplgs; lag++) {
                int idx = range * mplgs + lag;
                
                if (cuCabs(acfd_data[idx]) > 0.0) {
                    double phase = phase_data[idx];
                    double lag_time = lag * lagfr * 1e-6;
                    
                    double phase_dev = phase - mean_phase;
                    double lag_dev = lag_time - mean_lag;
                    
                    ss_phase += phase_dev * phase_dev;
                    ss_lag += lag_dev * lag_dev;
                    ss_cross += phase_dev * lag_dev;
                }
            }
            
            if (ss_phase > 0.0 && ss_lag > 0.0) {
                quality = fabs(ss_cross / sqrt(ss_phase * ss_lag));
            }
        }
    }
    
    velocity_out[range] = velocity;
    quality_out[range] = quality;
}

/*
 * CUDA kernel for power and spectral width calculation
 * Parallel processing of ACF magnitude for power/width estimation
 */
__global__ void PowerWidthKernel(cuDoubleComplex *acfd_data, int *range_status,
                                 double *power_out, double *width_out,
                                 int nranges, int mplgs, double lagfr)
{
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nranges || !range_status[range]) {
        if (range < nranges) {
            power_out[range] = 0.0;
            width_out[range] = 0.0;
        }
        return;
    }
    
    /* Calculate power from lag 0 */
    double power = cuCabs(acfd_data[range * mplgs]);
    power_out[range] = (power > 0.0) ? 10.0 * log10(power) : 0.0;
    
    /* Calculate spectral width from decorrelation */
    double lag0_power = cuCabs(acfd_data[range * mplgs]);
    double threshold = lag0_power * exp(-1.0); /* 1/e threshold */
    
    int decorr_lag = mplgs;
    for (int lag = 1; lag < mplgs; lag++) {
        int idx = range * mplgs + lag;
        if (cuCabs(acfd_data[idx]) < threshold) {
            decorr_lag = lag;
            break;
        }
    }
    
    /* Calculate spectral width */
    double tau = decorr_lag * lagfr * 1e-6; /* Decorrelation time */
    double spectral_width = 1.0 / (sqrt(2.0) * CUDART_PI * tau);
    
    /* Convert to m/s and clamp */
    spectral_width = spectral_width * 2.998e8 / (2.0 * 10e6);
    if (spectral_width < 10.0) spectral_width = 10.0;
    if (spectral_width > 1000.0) spectral_width = 1000.0;
    
    width_out[range] = spectral_width;
}

/*
 * CUDA kernel for noise filtering
 * Parallel noise threshold application across all data
 */
__global__ void NoiseFilterKernel(cuDoubleComplex *acfd_data, cuDoubleComplex *xcfd_data,
                                  double *noise_data, int *range_status,
                                  int nranges, int mplgs, double noise_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = nranges * mplgs;
    
    if (idx >= total_samples) return;
    
    int range = idx / mplgs;
    int lag = idx % mplgs;
    
    if (!range_status[range]) return;
    
    /* Calculate adaptive noise threshold */
    double noise_threshold = noise_data[range] * noise_factor;
    
    /* Apply threshold to ACF data */
    double acf_power = cuCabs(acfd_data[idx]);
    if (acf_power < noise_threshold) {
        acfd_data[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
    
    /* Apply threshold to XCF data if available */
    if (xcfd_data) {
        double xcf_power = cuCabs(xcfd_data[idx]);
        if (xcf_power < noise_threshold) {
            xcfd_data[idx] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
}

/*
 * CUDA kernel for interference detection
 * Parallel analysis of coherence patterns for CRI detection
 */
__global__ void InterferenceDetectionKernel(cuDoubleComplex *acfd_data, double *phase_data,
                                            int *range_status, int *cri_flags,
                                            int nranges, int mplgs)
{
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nranges || !range_status[range]) {
        if (range < nranges) {
            cri_flags[range] = 0;
        }
        return;
    }
    
    /* Calculate phase coherence metrics */
    double mean_phase = 0.0;
    double phase_variance = 0.0;
    int valid_phases = 0;
    
    /* First pass: calculate mean phase */
    for (int lag = 0; lag < mplgs; lag++) {
        int idx = range * mplgs + lag;
        if (cuCabs(acfd_data[idx]) > 0.0) {
            mean_phase += phase_data[idx];
            valid_phases++;
        }
    }
    
    if (valid_phases < 3) {
        cri_flags[range] = 0;
        return;
    }
    
    mean_phase /= valid_phases;
    
    /* Second pass: calculate variance */
    for (int lag = 0; lag < mplgs; lag++) {
        int idx = range * mplgs + lag;
        if (cuCabs(acfd_data[idx]) > 0.0) {
            double phase_diff = phase_data[idx] - mean_phase;
            phase_variance += phase_diff * phase_diff;
        }
    }
    phase_variance /= valid_phases;
    
    /* CRI detection based on low phase variance */
    cri_flags[range] = (phase_variance < 0.1) ? 1 : 0;
}

/*
 * CUDA kernel for alpha calculation
 * Parallel fitting of power law decay for alpha estimation
 */
__global__ void AlphaCalculationKernel(cuDoubleComplex *acfd_data, int *range_status,
                                       double *alpha_out, int nranges, int mplgs)
{
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nranges || !range_status[range]) {
        if (range < nranges) {
            alpha_out[range] = 0.0;
        }
        return;
    }
    
    /* Collect power values */
    double sum_log_power = 0.0;
    double sum_lag = 0.0;
    double sum_lag_log_power = 0.0;
    double sum_lag_squared = 0.0;
    int valid_lags = 0;
    
    for (int lag = 0; lag < mplgs; lag++) {
        int idx = range * mplgs + lag;
        double power = cuCabs(acfd_data[idx]);
        
        if (power > 0.0) {
            double log_power = log(power);
            double lag_val = (double)lag;
            
            sum_log_power += log_power;
            sum_lag += lag_val;
            sum_lag_log_power += lag_val * log_power;
            sum_lag_squared += lag_val * lag_val;
            valid_lags++;
        }
    }
    
    /* Calculate alpha using least squares */
    double alpha = 0.0;
    
    if (valid_lags >= 3) {
        double denominator = valid_lags * sum_lag_squared - sum_lag * sum_lag;
        
        if (fabs(denominator) > 1e-10) {
            alpha = -(valid_lags * sum_lag_log_power - sum_lag * sum_log_power) / denominator;
            
            /* Clamp alpha to reasonable bounds */
            if (alpha < 0.0) alpha = 0.0;
            if (alpha > 6.0) alpha = 6.0;
        }
    }
    
    alpha_out[range] = alpha;
}

/*
 * Host function to launch preprocessing kernels
 */
extern "C" int LaunchPreprocessingKernels(cuDoubleComplex *d_acfd_data, cuDoubleComplex *d_xcfd_data,
                                          double *d_power_data, double *d_noise_data,
                                          int *d_range_status, double *d_phase_data,
                                          int nranges, int mplgs, double lagfr, double smsep)
{
    /* Calculate grid dimensions */
    int threads_per_block = CUDA_THREADS_PER_BLOCK;
    int blocks_per_grid = (nranges + threads_per_block - 1) / threads_per_block;
    
    /* Launch preprocessing kernel */
    PreprocessACFKernel<<<blocks_per_grid, threads_per_block>>>(
        d_acfd_data, d_xcfd_data, d_power_data, d_noise_data,
        d_range_status, d_phase_data, nranges, mplgs, lagfr, smsep
    );
    
    /* Check for kernel launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/*
 * Host function to launch fitting kernels
 */
extern "C" int LaunchFittingKernels(cuDoubleComplex *d_acfd_data, double *d_phase_data,
                                    int *d_range_status, double *d_velocity, double *d_power,
                                    double *d_width, double *d_quality, int nranges, int mplgs,
                                    double lagfr, double tfreq)
{
    int threads_per_block = CUDA_THREADS_PER_BLOCK;
    int blocks_per_grid = (nranges + threads_per_block - 1) / threads_per_block;
    
    /* Launch phase unwrapping kernel */
    PhaseUnwrapKernel<<<blocks_per_grid, threads_per_block>>>(
        d_phase_data, d_acfd_data, d_range_status, nranges, mplgs
    );
    
    /* Launch velocity fitting kernel */
    VelocityFitKernel<<<blocks_per_grid, threads_per_block>>>(
        d_acfd_data, d_phase_data, d_range_status, d_velocity, d_quality,
        nranges, mplgs, lagfr, tfreq
    );
    
    /* Launch power and width calculation kernel */
    PowerWidthKernel<<<blocks_per_grid, threads_per_block>>>(
        d_acfd_data, d_range_status, d_power, d_width, nranges, mplgs, lagfr
    );
    
    /* Synchronize to ensure all kernels complete */
    cudaDeviceSynchronize();
    
    /* Check for errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA fitting kernels error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

#endif /* __CUDACC__ */
