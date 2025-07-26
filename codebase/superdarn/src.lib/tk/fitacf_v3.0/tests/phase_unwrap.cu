#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>  // For fprintf and stderr

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__global__ void phase_unwrap_kernel(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < num_ranges) {
        int base_idx = r * num_lags;
        
        if (num_lags > 0) {
            phase_out[base_idx] = phase_in[base_idx];
            float quality = 0.0f;
            
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
                quality += fabsf(delta);
            }
            
            if (quality_metric) {
                quality_metric[r] = quality;
            }
        }
    }
}

extern "C" void cuda_phase_unwrap(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
) {
    // Allocate device memory
    float *d_phase_in, *d_phase_out, *d_quality_metric = nullptr;
    size_t data_size = num_ranges * num_lags * sizeof(float);
    
    cudaMalloc((void**)&d_phase_in, data_size);
    cudaMalloc((void**)&d_phase_out, data_size);
    
    if (quality_metric) {
        cudaMalloc((void**)&d_quality_metric, num_ranges * sizeof(float));
    }
    
    // Copy input data to device
    cudaMemcpy(d_phase_in, phase_in, data_size, cudaMemcpyHostToDevice);
    
    // Initialize output
    cudaMemset(d_phase_out, 0, data_size);
    if (d_quality_metric) {
        cudaMemset(d_quality_metric, 0, num_ranges * sizeof(float));
    }
    
    // Configure and launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_ranges + threadsPerBlock - 1) / threadsPerBlock;
    
    phase_unwrap_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_phase_in, d_phase_out, d_quality_metric,
        num_ranges, num_lags, phase_threshold
    );
    
    // Copy results back to host
    cudaMemcpy(phase_out, d_phase_out, data_size, cudaMemcpyDeviceToHost);
    if (d_quality_metric && quality_metric) {
        cudaMemcpy(quality_metric, d_quality_metric, 
                  num_ranges * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Use printf instead of fprintf(stderr) which might not be available in CUDA device code
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // Free device memory
    cudaFree(d_phase_in);
    cudaFree(d_phase_out);
    if (d_quality_metric) {
        cudaFree(d_quality_metric);
    }
}
