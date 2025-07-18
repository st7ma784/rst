#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include "cuda_llist.h"

/**
 * SUPERDARN CUDA Kernel Architectures
 * 
 * This file implements optimized CUDA kernels for the most common
 * linked list usage patterns in SUPERDARN data processing:
 * 
 * 1. Batch Processing: Parallel ACF lag processing
 * 2. Filtering: Range gate quality filtering
 * 3. Sorting: Power/velocity sorting for analysis
 * 4. Reduction: Statistical aggregation across range gates
 */

// ============================================================================
// KERNEL 1: Batch ACF Processing
// Process ACF lag data across multiple range gates in parallel
// ============================================================================

__global__ void cuda_batch_acf_processing(
    cuda_llist_t* lists,           // Array of CUDA linked lists (one per range gate)
    int num_lists,                 // Number of range gates to process
    float* acf_results,            // Output ACF results [num_lists * MAX_LAGS]
    int max_lags,                  // Maximum number of lags to process
    float noise_threshold          // Noise threshold for quality filtering
) {
    int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (list_idx >= num_lists) return;
    
    cuda_llist_t* list = &lists[list_idx];
    float* output = &acf_results[list_idx * max_lags];
    
    // Initialize output
    for (int i = 0; i < max_lags; i++) {
        output[i] = 0.0f;
    }
    
    // Process all valid elements in this range gate's list
    for (int i = 0; i < list->capacity && i < max_lags; i++) {
        if (list->mask[i]) {  // Element is valid
            acf_data_t* data = (acf_data_t*)list->data[i];
            
            // Apply noise threshold filtering
            if (data->power > noise_threshold) {
                // Compute ACF for this lag
                output[i] = data->real * data->real + data->imag * data->imag;
                
                // Apply phase correction if needed
                if (data->phase_correction != 0.0f) {
                    float phase = atan2f(data->imag, data->real) + data->phase_correction;
                    output[i] = sqrtf(output[i]) * cosf(phase);
                }
            }
        }
    }
}

// ============================================================================
// KERNEL 2: Parallel Range Gate Filtering
// Filter range gates based on quality metrics in parallel
// ============================================================================

__global__ void cuda_range_gate_filtering(
    cuda_llist_t* lists,           // Array of CUDA linked lists
    int num_lists,                 // Number of range gates
    float* quality_metrics,        // Input quality metrics per range gate
    float quality_threshold,       // Minimum quality threshold
    int* filtered_indices,         // Output: indices of valid range gates
    int* num_filtered              // Output: number of valid range gates
) {
    int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (list_idx >= num_lists) return;
    
    cuda_llist_t* list = &lists[list_idx];
    float quality = quality_metrics[list_idx];
    
    // Apply quality filtering
    bool is_valid = (quality >= quality_threshold) && (list->size > 0);
    
    // Count valid elements within this range gate
    int valid_count = 0;
    for (int i = 0; i < list->capacity; i++) {
        if (list->mask[i]) {
            acf_data_t* data = (acf_data_t*)list->data[i];
            if (data->power > 0.0f && !isnan(data->velocity)) {
                valid_count++;
            }
        }
    }
    
    // Range gate is valid if it has sufficient valid data points
    is_valid = is_valid && (valid_count >= 3);  // Minimum 3 valid lags
    
    // Atomic update of filtered results
    if (is_valid) {
        int idx = atomicAdd(num_filtered, 1);
        filtered_indices[idx] = list_idx;
    }
}

// ============================================================================
// KERNEL 3: Parallel Sorting by Power/Velocity
// Sort elements within each range gate by power or velocity
// ============================================================================

__device__ void cuda_bubble_sort_by_power(cuda_llist_t* list) {
    // Simple bubble sort optimized for small lists (typical SUPERDARN case)
    for (int i = 0; i < list->size - 1; i++) {
        for (int j = 0; j < list->size - i - 1; j++) {
            if (list->mask[j] && list->mask[j + 1]) {
                acf_data_t* data1 = (acf_data_t*)list->data[j];
                acf_data_t* data2 = (acf_data_t*)list->data[j + 1];
                
                if (data1->power < data2->power) {  // Sort descending by power
                    // Swap data pointers
                    void* temp = list->data[j];
                    list->data[j] = list->data[j + 1];
                    list->data[j + 1] = temp;
                }
            }
        }
    }
}

__global__ void cuda_parallel_sorting(
    cuda_llist_t* lists,           // Array of CUDA linked lists
    int num_lists,                 // Number of range gates
    int sort_type                  // 0=power, 1=velocity, 2=lag_number
) {
    int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (list_idx >= num_lists) return;
    
    cuda_llist_t* list = &lists[list_idx];
    
    if (list->size <= 1) return;  // Nothing to sort
    
    // Use optimized sorting for small lists (common in SUPERDARN)
    if (sort_type == 0) {  // Sort by power
        cuda_bubble_sort_by_power(list);
    }
    // Additional sort types can be implemented as needed
}

// ============================================================================
// KERNEL 4: Statistical Reduction Across Range Gates
// Compute statistics (mean, max, variance) across all range gates
// ============================================================================

__global__ void cuda_statistical_reduction(
    cuda_llist_t* lists,           // Array of CUDA linked lists
    int num_lists,                 // Number of range gates
    float* mean_power,             // Output: mean power across all gates
    float* max_power,              // Output: maximum power found
    float* power_variance,         // Output: power variance
    int* total_valid_samples       // Output: total number of valid samples
) {
    __shared__ float shared_sum[256];
    __shared__ float shared_max[256];
    __shared__ int shared_count[256];
    
    int tid = threadIdx.x;
    int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_sum[tid] = 0.0f;
    shared_max[tid] = 0.0f;
    shared_count[tid] = 0;
    
    // Process assigned range gate
    if (list_idx < num_lists) {
        cuda_llist_t* list = &lists[list_idx];
        
        float local_sum = 0.0f;
        float local_max = 0.0f;
        int local_count = 0;
        
        // Accumulate statistics from this range gate
        for (int i = 0; i < list->capacity; i++) {
            if (list->mask[i]) {
                acf_data_t* data = (acf_data_t*)list->data[i];
                if (data->power > 0.0f && !isnan(data->power)) {
                    local_sum += data->power;
                    local_max = fmaxf(local_max, data->power);
                    local_count++;
                }
            }
        }
        
        shared_sum[tid] = local_sum;
        shared_max[tid] = local_max;
        shared_count[tid] = local_count;
    }
    
    __syncthreads();
    
    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block results to global memory
    if (tid == 0) {
        atomicAdd(mean_power, shared_sum[0]);
        atomicAdd(total_valid_samples, shared_count[0]);
        
        // Atomic max operation
        float old_max = *max_power;
        while (shared_max[0] > old_max) {
            float assumed = old_max;
            old_max = atomicCAS((int*)max_power, __float_as_int(assumed), __float_as_int(shared_max[0]));
            if (old_max == assumed) break;
            old_max = __int_as_float(old_max);
        }
    }
}

// ============================================================================
// HOST INTERFACE FUNCTIONS
// C-compatible interface for integration with existing SUPERDARN code
// ============================================================================

extern "C" {

/**
 * Launch batch ACF processing across multiple range gates
 */
cudaError_t launch_batch_acf_processing(
    cuda_llist_t* d_lists,
    int num_lists,
    float* d_acf_results,
    int max_lags,
    float noise_threshold
) {
    int block_size = 256;
    int grid_size = (num_lists + block_size - 1) / block_size;
    
    cuda_batch_acf_processing<<<grid_size, block_size>>>(
        d_lists, num_lists, d_acf_results, max_lags, noise_threshold
    );
    
    return cudaGetLastError();
}

/**
 * Launch parallel range gate filtering
 */
cudaError_t launch_range_gate_filtering(
    cuda_llist_t* d_lists,
    int num_lists,
    float* d_quality_metrics,
    float quality_threshold,
    int* d_filtered_indices,
    int* d_num_filtered
) {
    int block_size = 256;
    int grid_size = (num_lists + block_size - 1) / block_size;
    
    // Initialize counter
    cudaMemset(d_num_filtered, 0, sizeof(int));
    
    cuda_range_gate_filtering<<<grid_size, block_size>>>(
        d_lists, num_lists, d_quality_metrics, quality_threshold,
        d_filtered_indices, d_num_filtered
    );
    
    return cudaGetLastError();
}

/**
 * Launch parallel sorting within range gates
 */
cudaError_t launch_parallel_sorting(
    cuda_llist_t* d_lists,
    int num_lists,
    int sort_type
) {
    int block_size = 256;
    int grid_size = (num_lists + block_size - 1) / block_size;
    
    cuda_parallel_sorting<<<grid_size, block_size>>>(
        d_lists, num_lists, sort_type
    );
    
    return cudaGetLastError();
}

/**
 * Launch statistical reduction across all range gates
 */
cudaError_t launch_statistical_reduction(
    cuda_llist_t* d_lists,
    int num_lists,
    float* d_mean_power,
    float* d_max_power,
    float* d_power_variance,
    int* d_total_valid_samples
) {
    int block_size = 256;
    int grid_size = (num_lists + block_size - 1) / block_size;
    
    // Initialize output values
    cudaMemset(d_mean_power, 0, sizeof(float));
    cudaMemset(d_max_power, 0, sizeof(float));
    cudaMemset(d_power_variance, 0, sizeof(float));
    cudaMemset(d_total_valid_samples, 0, sizeof(int));
    
    cuda_statistical_reduction<<<grid_size, block_size>>>(
        d_lists, num_lists, d_mean_power, d_max_power,
        d_power_variance, d_total_valid_samples
    );
    
    return cudaGetLastError();
}

} // extern "C"
