#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "cuda_llist.h"

/**
 * CUDA Kernel for parallel phase unwrapping
 * Optimized for SUPERDARN ACF data processing
 */
__global__ void cuda_phase_unwrap_kernel(
    float* phase_in,      // Input phase data (wrapped, in radians)
    float* phase_out,     // Output phase data (unwrapped)
    float* quality_metric,// Quality metric for each phase point
    int num_ranges,       // Number of range gates
    int num_lags,         // Number of lags per range gate
    float phase_threshold // Threshold for phase jumps (typically ~π)
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (range_idx >= num_ranges) return;
    
    int base_idx = range_idx * num_lags;
    
    // First point remains unchanged
    if (num_lags > 0) {
        phase_out[base_idx] = phase_in[base_idx];
    }
    
    // Unwrap subsequent points
    for (int i = 1; i < num_lags; i++) {
        int curr_idx = base_idx + i;
        float delta = phase_in[curr_idx] - phase_in[curr_idx - 1];
        
        // Handle phase wrapping
        if (delta > phase_threshold) {
            delta -= 2.0f * CUDART_PI_F;
        } else if (delta < -phase_threshold) {
            delta += 2.0f * CUDART_PI_F;
        }
        
        // Apply correction
        phase_out[curr_idx] = phase_out[curr_idx - 1] + delta;
        
        // Optional: Update quality metric
        if (quality_metric) {
            quality_metric[range_idx] += fabsf(delta);
        }
    }
}

/**
 * Host wrapper for the phase unwrapping kernel
 */
void cuda_phase_unwrap(
    float* h_phase_in,    // Input phase data (host memory)
    float* h_phase_out,   // Output phase data (host memory)
    float* h_quality,     // Output quality metric (optional, can be NULL)
    int num_ranges,       // Number of range gates
    int num_lags,         // Number of lags per range gate
    float phase_threshold // Threshold for phase jumps (typically ~π)
) {
    // Allocate device memory
    float *d_phase_in, *d_phase_out, *d_quality = nullptr;
    size_t data_size = num_ranges * num_lags * sizeof(float);
    
    cudaMalloc(&d_phase_in, data_size);
    cudaMalloc(&d_phase_out, data_size);
    
    if (h_quality) {
        cudaMalloc(&d_quality, num_ranges * sizeof(float));
        cudaMemset(d_quality, 0, num_ranges * sizeof(float));
    }
    
    // Copy input data to device
    cudaMemcpy(d_phase_in, h_phase_in, data_size, cudaMemcpyHostToDevice);
    
    // Configure and launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_ranges + threads_per_block - 1) / threads_per_block;
    
    cuda_phase_unwrap_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_phase_in, d_phase_out, d_quality, num_ranges, num_lags, phase_threshold
    );
    
    // Copy results back to host
    cudaMemcpy(h_phase_out, d_phase_out, data_size, cudaMemcpyDeviceToHost);
    
    if (h_quality && d_quality) {
        cudaMemcpy(h_quality, d_quality, num_ranges * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Free device memory
    cudaFree(d_phase_in);
    cudaFree(d_phase_out);
    if (d_quality) cudaFree(d_quality);
}

/**
 * Optimized CPU implementation using OpenMP
 */
void omp_phase_unwrap(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
) {
    #pragma omp parallel for
    for (int r = 0; r < num_ranges; r++) {
        int base_idx = r * num_lags;
        
        if (num_lags > 0) {
            phase_out[base_idx] = phase_in[base_idx];
            
            for (int i = 1; i < num_lags; i++) {
                int curr_idx = base_idx + i;
                float delta = phase_in[curr_idx] - phase_in[curr_idx - 1];
                
                // Handle phase wrapping
                if (delta > phase_threshold) {
                    delta -= 2.0f * M_PI;
                } else if (delta < -phase_threshold) {
                    delta += 2.0f * M_PI;
                }
                
                phase_out[curr_idx] = phase_out[curr_idx - 1] + delta;
                
                if (quality_metric) {
                    quality_metric[r] += fabsf(delta);
                }
            }
        }
    }
}
