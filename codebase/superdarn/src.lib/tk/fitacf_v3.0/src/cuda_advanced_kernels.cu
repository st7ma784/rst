#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include "cuda_llist.h"

/**
 * SUPERDARN Advanced CUDA Optimization Kernels
 * 
 * This file implements highly optimized CUDA kernels for the most
 * computationally intensive patterns identified in the SUPERDARN codebase:
 * 
 * 1. Parallel ACF/XCF Data Copying (2D range×lag grids)
 * 2. Complex Number Processing (real/imaginary operations)
 * 3. Power Computation and Normalization
 * 4. Phase Calculation and Correction
 * 5. Statistical Reduction and Aggregation
 * 6. Memory Initialization Patterns
 */

// ============================================================================
// KERNEL 1: Parallel ACF/XCF Data Copying with Complex Operations
// Optimizes the nested loops in Copy_Fitting_Prms function
// ============================================================================

__global__ void cuda_parallel_acf_copy_kernel(
    float* raw_acfd_real,          // Input: raw ACF data (real component)
    float* raw_acfd_imag,          // Input: raw ACF data (imaginary component)
    cuFloatComplex* fit_acfd,      // Output: fitted ACF data (complex)
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    bool zero_fill                 // Whether to zero-fill missing data
) {
    // 2D grid: blockIdx.x = range, blockIdx.y = lag
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lag_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (range_idx >= nrang || lag_idx >= mplgs) return;
    
    int linear_idx = range_idx * mplgs + lag_idx;
    
    if (zero_fill || (raw_acfd_real == NULL || raw_acfd_imag == NULL)) {
        // Zero-fill pattern (highly parallel)
        fit_acfd[linear_idx] = make_cuFloatComplex(0.0f, 0.0f);
    } else {
        // Copy and convert to complex format
        float real_val = raw_acfd_real[linear_idx];
        float imag_val = raw_acfd_imag[linear_idx];
        fit_acfd[linear_idx] = make_cuFloatComplex(real_val, imag_val);
    }
}

__global__ void cuda_parallel_xcf_copy_kernel(
    float* raw_xcfd_real,          // Input: raw XCF data (real component)
    float* raw_xcfd_imag,          // Input: raw XCF data (imaginary component)
    cuFloatComplex* fit_xcfd,      // Output: fitted XCF data (complex)
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    bool zero_fill                 // Whether to zero-fill missing data
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lag_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (range_idx >= nrang || lag_idx >= mplgs) return;
    
    int linear_idx = range_idx * mplgs + lag_idx;
    
    if (zero_fill || (raw_xcfd_real == NULL || raw_xcfd_imag == NULL)) {
        fit_xcfd[linear_idx] = make_cuFloatComplex(0.0f, 0.0f);
    } else {
        float real_val = raw_xcfd_real[linear_idx];
        float imag_val = raw_xcfd_imag[linear_idx];
        fit_xcfd[linear_idx] = make_cuFloatComplex(real_val, imag_val);
    }
}

// ============================================================================
// KERNEL 2: Advanced Power and Phase Computation
// Parallel computation of power, phase, and normalization
// ============================================================================

__global__ void cuda_power_phase_computation_kernel(
    cuFloatComplex* acf_data,      // Input: ACF complex data
    float* pwr0,                   // Input: lag-0 power for normalization
    float* output_power,           // Output: computed power values
    float* output_phase,           // Output: computed phase values
    float* output_normalized,      // Output: normalized power values
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    float noise_threshold          // Noise filtering threshold
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lag_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (range_idx >= nrang || lag_idx >= mplgs) return;
    
    int linear_idx = range_idx * mplgs + lag_idx;
    cuFloatComplex acf_val = acf_data[linear_idx];
    
    // Compute power (magnitude squared)
    float power = cuCabsf(acf_val);
    power = power * power;
    
    // Compute phase
    float phase = atan2f(cuCimagf(acf_val), cuCrealf(acf_val));
    
    // Normalize by lag-0 power
    float normalized_power = 0.0f;
    if (pwr0[range_idx] > noise_threshold && pwr0[range_idx] > 0.0f) {
        normalized_power = power / pwr0[range_idx];
    }
    
    // Store results
    output_power[linear_idx] = power;
    output_phase[linear_idx] = phase;
    output_normalized[linear_idx] = normalized_power;
}

// ============================================================================
// KERNEL 3: Advanced Statistical Reduction with Shared Memory
// Optimized reduction operations for SUPERDARN statistics
// ============================================================================

__global__ void cuda_advanced_statistical_reduction_kernel(
    cuFloatComplex* acf_data,      // Input: ACF data
    float* pwr0,                   // Input: lag-0 power
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    float* global_stats,           // Output: [mean_power, max_power, total_power, valid_count]
    float noise_threshold          // Noise threshold
) {
    __shared__ float shared_power_sum[256];
    __shared__ float shared_power_max[256];
    __shared__ int shared_valid_count[256];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_power_sum[tid] = 0.0f;
    shared_power_max[tid] = 0.0f;
    shared_valid_count[tid] = 0;
    
    // Process multiple elements per thread for better occupancy
    int elements_per_thread = (nrang * mplgs + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = global_idx + i * blockDim.x;
        if (idx >= nrang * mplgs) break;
        
        int range_idx = idx / mplgs;
        int lag_idx = idx % mplgs;
        
        // Skip if below noise threshold
        if (pwr0[range_idx] <= noise_threshold) continue;
        
        cuFloatComplex acf_val = acf_data[idx];
        float power = cuCabsf(acf_val);
        power = power * power;
        
        if (power > 0.0f && !isnan(power)) {
            shared_power_sum[tid] += power;
            shared_power_max[tid] = fmaxf(shared_power_max[tid], power);
            shared_valid_count[tid]++;
        }
    }
    
    __syncthreads();
    
    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_power_sum[tid] += shared_power_sum[tid + stride];
            shared_power_max[tid] = fmaxf(shared_power_max[tid], shared_power_max[tid + stride]);
            shared_valid_count[tid] += shared_valid_count[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block results to global memory
    if (tid == 0) {
        atomicAdd(&global_stats[0], shared_power_sum[0]);    // Total power
        atomicAdd((int*)&global_stats[3], shared_valid_count[0]); // Valid count
        
        // Atomic max for maximum power
        float old_max = global_stats[1];
        while (shared_power_max[0] > old_max) {
            float assumed = old_max;
            old_max = atomicCAS((int*)&global_stats[1], __float_as_int(assumed), __float_as_int(shared_power_max[0]));
            if (old_max == assumed) break;
            old_max = __int_as_float(old_max);
        }
    }
}

// ============================================================================
// KERNEL 4: Parallel Lag Processing with Complex Operations
// Optimized processing of lag-based computations
// ============================================================================

__global__ void cuda_parallel_lag_processing_kernel(
    cuFloatComplex* acf_data,      // Input: ACF data
    float* lag_times,              // Input: lag time values
    float mpinc,                   // Pulse increment time
    float* output_alpha2,          // Output: alpha-squared values
    float* output_sigma,           // Output: sigma values
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    int nave                       // Number of averages
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lag_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (range_idx >= nrang || lag_idx >= mplgs) return;
    
    int linear_idx = range_idx * mplgs + lag_idx;
    cuFloatComplex acf_val = acf_data[linear_idx];
    
    // Compute magnitude and phase
    float magnitude = cuCabsf(acf_val);
    float phase = atan2f(cuCimagf(acf_val), cuCrealf(acf_val));
    
    // Compute alpha-squared (Bendat & Piersol)
    float alpha2 = 1.0f;
    if (magnitude > 0.0f && nave > 0) {
        alpha2 = 1.0f / (2.0f * nave * magnitude * magnitude);
    }
    
    // Compute sigma (uncertainty)
    float sigma = 0.0f;
    if (magnitude > 0.0f && alpha2 > 0.0f) {
        sigma = sqrtf(alpha2);
    }
    
    // Store results
    output_alpha2[linear_idx] = alpha2;
    output_sigma[linear_idx] = sigma;
}

// ============================================================================
// KERNEL 5: Advanced Memory Coalescing Optimization
// Optimized memory access patterns for better bandwidth utilization
// ============================================================================

__global__ void cuda_coalesced_data_transform_kernel(
    float* input_data,             // Input: raw data (AoS format)
    cuFloatComplex* output_acf,    // Output: ACF complex data (SoA format)
    cuFloatComplex* output_xcf,    // Output: XCF complex data (SoA format)
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    int data_stride                // Stride between real/imaginary components
) {
    // Use 1D indexing for better coalescing
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = nrang * mplgs;
    
    if (global_idx >= total_elements) return;
    
    int range_idx = global_idx / mplgs;
    int lag_idx = global_idx % mplgs;
    
    // Coalesced memory access pattern
    int acf_real_idx = range_idx * mplgs * 2 + lag_idx;
    int acf_imag_idx = acf_real_idx + data_stride;
    int xcf_real_idx = acf_imag_idx + data_stride;
    int xcf_imag_idx = xcf_real_idx + data_stride;
    
    // Load data with coalesced access
    float acf_real = input_data[acf_real_idx];
    float acf_imag = input_data[acf_imag_idx];
    float xcf_real = input_data[xcf_real_idx];
    float xcf_imag = input_data[xcf_imag_idx];
    
    // Store as complex numbers
    output_acf[global_idx] = make_cuFloatComplex(acf_real, acf_imag);
    output_xcf[global_idx] = make_cuFloatComplex(xcf_real, xcf_imag);
}

// ============================================================================
// HOST INTERFACE FUNCTIONS
// C-compatible interface for integration with existing SUPERDARN code
// ============================================================================

extern "C" {

/**
 * Launch parallel ACF data copying with advanced optimization
 */
cudaError_t launch_parallel_acf_copy(
    float* raw_acfd_real,
    float* raw_acfd_imag,
    cuFloatComplex* fit_acfd,
    int nrang,
    int mplgs,
    bool zero_fill
) {
    // Optimize block dimensions for 2D grid
    dim3 block_size(16, 16);  // 256 threads per block
    dim3 grid_size(
        (nrang + block_size.x - 1) / block_size.x,
        (mplgs + block_size.y - 1) / block_size.y
    );
    
    cuda_parallel_acf_copy_kernel<<<grid_size, block_size>>>(
        raw_acfd_real, raw_acfd_imag, fit_acfd, nrang, mplgs, zero_fill
    );
    
    return cudaGetLastError();
}

/**
 * Launch parallel XCF data copying
 */
cudaError_t launch_parallel_xcf_copy(
    float* raw_xcfd_real,
    float* raw_xcfd_imag,
    cuFloatComplex* fit_xcfd,
    int nrang,
    int mplgs,
    bool zero_fill
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (nrang + block_size.x - 1) / block_size.x,
        (mplgs + block_size.y - 1) / block_size.y
    );
    
    cuda_parallel_xcf_copy_kernel<<<grid_size, block_size>>>(
        raw_xcfd_real, raw_xcfd_imag, fit_xcfd, nrang, mplgs, zero_fill
    );
    
    return cudaGetLastError();
}

/**
 * Launch advanced power and phase computation
 */
cudaError_t launch_power_phase_computation(
    cuFloatComplex* acf_data,
    float* pwr0,
    float* output_power,
    float* output_phase,
    float* output_normalized,
    int nrang,
    int mplgs,
    float noise_threshold
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (nrang + block_size.x - 1) / block_size.x,
        (mplgs + block_size.y - 1) / block_size.y
    );
    
    cuda_power_phase_computation_kernel<<<grid_size, block_size>>>(
        acf_data, pwr0, output_power, output_phase, output_normalized,
        nrang, mplgs, noise_threshold
    );
    
    return cudaGetLastError();
}

/**
 * Launch advanced statistical reduction
 */
cudaError_t launch_advanced_statistical_reduction(
    cuFloatComplex* acf_data,
    float* pwr0,
    int nrang,
    int mplgs,
    float* global_stats,
    float noise_threshold
) {
    int total_elements = nrang * mplgs;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // Initialize global statistics
    cudaMemset(global_stats, 0, 4 * sizeof(float));
    
    cuda_advanced_statistical_reduction_kernel<<<grid_size, block_size>>>(
        acf_data, pwr0, nrang, mplgs, global_stats, noise_threshold
    );
    
    return cudaGetLastError();
}

/**
 * Launch parallel lag processing
 */
cudaError_t launch_parallel_lag_processing(
    cuFloatComplex* acf_data,
    float* lag_times,
    float mpinc,
    float* output_alpha2,
    float* output_sigma,
    int nrang,
    int mplgs,
    int nave
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (nrang + block_size.x - 1) / block_size.x,
        (mplgs + block_size.y - 1) / block_size.y
    );
    
    cuda_parallel_lag_processing_kernel<<<grid_size, block_size>>>(
        acf_data, lag_times, mpinc, output_alpha2, output_sigma,
        nrang, mplgs, nave
    );
    
    return cudaGetLastError();
}

/**
 * Launch coalesced data transformation
 */
cudaError_t launch_coalesced_data_transform(
    float* input_data,
    cuFloatComplex* output_acf,
    cuFloatComplex* output_xcf,
    int nrang,
    int mplgs,
    int data_stride
) {
    int total_elements = nrang * mplgs;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    cuda_coalesced_data_transform_kernel<<<grid_size, block_size>>>(
        input_data, output_acf, output_xcf, nrang, mplgs, data_stride
    );
    
    return cudaGetLastError();
}

} // extern "C"
