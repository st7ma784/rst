/*
 * CUDA Kernels for Linked List Replacement
 * 
 * This file is part of the Radar Software Toolkit (RST).
 * 
 * Implements GPU kernels for all major linked list usage patterns
 * found in SUPERDARN processing (fitacf_v3.0 and lmfit_v2.0).
 * 
 * Author: CUDA Conversion Project
 * Date: 2025
 */

#include "llist_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/* === Kernel Architecture Patterns === */

/**
 * @brief Generic batch processing kernel - replaces llist_for_each_arg
 * 
 * This is the most critical kernel as llist_for_each_arg is used extensively
 * in both fitacf_v3.0 and lmfit_v2.0 for processing multiple ranges.
 * 
 * Pattern: Each thread block processes one list, threads within block process elements
 */
template<typename ProcessFunc>
__global__ void llist_cuda_batch_process_kernel(
    void** flat_data,           // Flattened data from all lists
    uint32_t* flat_valid_mask,  // Flattened validity masks
    uint32_t* list_offsets,     // Starting offset for each list
    uint32_t* list_counts,      // Element count for each list
    uint32_t num_lists,         // Number of lists to process
    void* arg1,                 // First argument to processing function
    void* arg2,                 // Second argument to processing function
    ProcessFunc process_func    // Device function to apply to each element
) {
    uint32_t list_id = blockIdx.x;
    uint32_t thread_id = threadIdx.x;
    uint32_t block_size = blockDim.x;
    
    if (list_id >= num_lists) return;
    
    uint32_t list_offset = list_offsets[list_id];
    uint32_t list_count = list_counts[list_id];
    
    // Process elements in this list with stride
    for (uint32_t i = thread_id; i < list_count; i += block_size) {
        uint32_t global_idx = list_offset + i;
        
        // Check if element is valid
        if (llist_cuda_is_valid(flat_valid_mask, global_idx)) {
            void* element = flat_data[global_idx];
            if (element) {
                process_func(element, arg1, arg2);
            }
        }
    }
}

/**
 * @brief Range-based processing kernel for FITACF/LMFIT patterns
 * 
 * Specialized for the common pattern where each range has multiple data lists
 * (acf, pwrs, scpwr, phases, elev) that need coordinated processing.
 */
__global__ void llist_cuda_range_process_kernel(
    void** acf_data,            // ACF data arrays
    void** pwr_data,            // Power data arrays  
    void** scpwr_data,          // Self-clutter power arrays
    void** phase_data,          // Phase data arrays
    void** elev_data,           // Elevation data arrays
    uint32_t* acf_valid_mask,   // Validity masks for each data type
    uint32_t* pwr_valid_mask,
    uint32_t* scpwr_valid_mask,
    uint32_t* phase_valid_mask,
    uint32_t* elev_valid_mask,
    uint32_t* range_offsets,    // Starting offset for each range
    uint32_t* range_counts,     // Element count for each range
    uint32_t num_ranges,        // Number of ranges to process
    void* fit_params,           // Fitting parameters
    void* output_data           // Output results
) {
    uint32_t range_id = blockIdx.x;
    uint32_t thread_id = threadIdx.x;
    uint32_t block_size = blockDim.x;
    
    if (range_id >= num_ranges) return;
    
    uint32_t range_offset = range_offsets[range_id];
    uint32_t range_count = range_counts[range_id];
    
    // Shared memory for reduction operations
    extern __shared__ float shared_data[];
    
    // Process elements in this range
    for (uint32_t i = thread_id; i < range_count; i += block_size) {
        uint32_t global_idx = range_offset + i;
        
        // Check if all required data is valid for this element
        bool all_valid = llist_cuda_is_valid(acf_valid_mask, global_idx) &&
                        llist_cuda_is_valid(pwr_valid_mask, global_idx) &&
                        llist_cuda_is_valid(scpwr_valid_mask, global_idx) &&
                        llist_cuda_is_valid(phase_valid_mask, global_idx) &&
                        llist_cuda_is_valid(elev_valid_mask, global_idx);
        
        if (all_valid) {
            // Process coordinated data for this lag/element
            // This would be specialized for specific FITACF/LMFIT operations
            // Example: compute fitting parameters, apply filters, etc.
        }
    }
    
    // Synchronize threads in block for any reduction operations
    __syncthreads();
}

/**
 * @brief Filtering kernel - replaces llist_delete_node patterns
 * 
 * Instead of deleting nodes, marks them as invalid based on filter criteria.
 * Common in TX overlap filtering and bad sample removal.
 */
template<typename FilterFunc>
__global__ void llist_cuda_filter_kernel(
    void** flat_data,
    uint32_t* flat_valid_mask,
    uint32_t* list_offsets,
    uint32_t* list_counts,
    uint32_t num_lists,
    void* filter_criteria,
    FilterFunc filter_func
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = 0;
    
    // Calculate total elements across all lists
    for (uint32_t i = 0; i < num_lists; i++) {
        total_elements += list_counts[i];
    }
    
    if (global_idx >= total_elements) return;
    
    // Check if element is currently valid
    if (llist_cuda_is_valid(flat_valid_mask, global_idx)) {
        void* element = flat_data[global_idx];
        if (element && filter_func(element, filter_criteria)) {
            // Mark as invalid (mask-based deletion)
            llist_cuda_clear_valid(flat_valid_mask, global_idx);
        }
    }
}

/**
 * @brief Sorting kernel using bitonic sort for GPU efficiency
 * 
 * Replaces llist_sort functionality with GPU-optimized sorting.
 */
template<typename CompareFunc>
__global__ void llist_cuda_bitonic_sort_kernel(
    void** data,
    uint32_t* valid_mask,
    uint32_t* indices,
    uint32_t count,
    uint32_t stage,
    uint32_t step,
    bool ascending,
    CompareFunc compare_func
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pair_distance = 1 << (step - 1);
    uint32_t block_width = 2 * pair_distance;
    
    if (tid >= count / 2) return;
    
    uint32_t left_id = (tid % pair_distance) + (tid / pair_distance) * block_width;
    uint32_t right_id = left_id + pair_distance;
    
    if (right_id >= count) return;
    
    uint32_t left_idx = indices[left_id];
    uint32_t right_idx = indices[right_id];
    
    // Only compare valid elements
    if (llist_cuda_is_valid(valid_mask, left_idx) && 
        llist_cuda_is_valid(valid_mask, right_idx)) {
        
        void* left_data = data[left_idx];
        void* right_data = data[right_idx];
        
        bool swap_needed = false;
        if (ascending) {
            swap_needed = compare_func(left_data, right_data) > 0;
        } else {
            swap_needed = compare_func(left_data, right_data) < 0;
        }
        
        // Determine if we're in ascending or descending block
        uint32_t block_id = tid / (1 << (stage - 1));
        if (block_id % 2 == 1) {
            swap_needed = !swap_needed;
        }
        
        if (swap_needed) {
            // Swap indices
            indices[left_id] = right_idx;
            indices[right_id] = left_idx;
        }
    }
}

/**
 * @brief Reduction kernel for finding min/max elements
 * 
 * Replaces llist_get_min/llist_get_max with parallel reduction.
 */
template<typename CompareFunc>
__global__ void llist_cuda_reduction_kernel(
    void** data,
    uint32_t* valid_mask,
    uint32_t count,
    void** result,
    bool find_max,
    CompareFunc compare_func
) {
    extern __shared__ void* shared_ptrs[];
    
    uint32_t tid = threadIdx.x;
    uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (global_id < count && llist_cuda_is_valid(valid_mask, global_id)) {
        shared_ptrs[tid] = data[global_id];
    } else {
        shared_ptrs[tid] = nullptr;
    }
    
    __syncthreads();
    
    // Parallel reduction
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            void* a = shared_ptrs[tid];
            void* b = shared_ptrs[tid + stride];
            
            if (a && b) {
                int cmp = compare_func(a, b);
                if ((find_max && cmp < 0) || (!find_max && cmp > 0)) {
                    shared_ptrs[tid] = b;
                }
            } else if (b && !a) {
                shared_ptrs[tid] = b;
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0 && shared_ptrs[0]) {
        atomicExch((unsigned long long*)result, (unsigned long long)shared_ptrs[0]);
    }
}

/**
 * @brief Compaction kernel - removes invalid elements and compacts arrays
 * 
 * Used for periodic cleanup of heavily filtered lists.
 */
__global__ void llist_cuda_compact_kernel(
    void** input_data,
    uint32_t* input_valid_mask,
    void** output_data,
    uint32_t* output_indices,
    uint32_t* scan_results,
    uint32_t input_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= input_count) return;
    
    if (llist_cuda_is_valid(input_valid_mask, tid)) {
        uint32_t output_pos = scan_results[tid];
        output_data[output_pos] = input_data[tid];
        output_indices[output_pos] = tid;
    }
}

/* === Host Interface Functions === */

/**
 * @brief Launch batch processing kernel with proper grid configuration
 */
template<typename ProcessFunc>
cudaError_t llist_cuda_launch_batch_process(
    llist_cuda_batch_t* batch,
    void* arg1,
    void* arg2,
    ProcessFunc process_func,
    cudaStream_t stream = 0
) {
    if (!batch) return cudaErrorInvalidValue;
    
    // Configure grid: one block per list
    dim3 grid(batch->num_lists);
    dim3 block(LLIST_CUDA_BLOCK_SIZE);
    
    // Launch kernel
    llist_cuda_batch_process_kernel<<<grid, block, 0, stream>>>(
        batch->flat_data,
        batch->flat_valid_mask,
        batch->list_offsets,
        batch->list_counts,
        batch->num_lists,
        arg1,
        arg2,
        process_func
    );
    
    return cudaGetLastError();
}

/**
 * @brief Launch range processing kernel for FITACF/LMFIT patterns
 */
cudaError_t llist_cuda_launch_range_process(
    void** acf_data,
    void** pwr_data,
    void** scpwr_data,
    void** phase_data,
    void** elev_data,
    uint32_t* acf_valid_mask,
    uint32_t* pwr_valid_mask,
    uint32_t* scpwr_valid_mask,
    uint32_t* phase_valid_mask,
    uint32_t* elev_valid_mask,
    uint32_t* range_offsets,
    uint32_t* range_counts,
    uint32_t num_ranges,
    void* fit_params,
    void* output_data,
    size_t shared_mem_size = 0,
    cudaStream_t stream = 0
) {
    // Configure grid: one block per range
    dim3 grid(num_ranges);
    dim3 block(LLIST_CUDA_BLOCK_SIZE);
    
    llist_cuda_range_process_kernel<<<grid, block, shared_mem_size, stream>>>(
        acf_data, pwr_data, scpwr_data, phase_data, elev_data,
        acf_valid_mask, pwr_valid_mask, scpwr_valid_mask, phase_valid_mask, elev_valid_mask,
        range_offsets, range_counts, num_ranges,
        fit_params, output_data
    );
    
    return cudaGetLastError();
}

/**
 * @brief Launch filtering kernel
 */
template<typename FilterFunc>
cudaError_t llist_cuda_launch_filter(
    llist_cuda_batch_t* batch,
    void* filter_criteria,
    FilterFunc filter_func,
    cudaStream_t stream = 0
) {
    if (!batch) return cudaErrorInvalidValue;
    
    uint32_t total_elements = 0;
    for (uint32_t i = 0; i < batch->num_lists; i++) {
        total_elements += batch->list_counts[i];
    }
    
    dim3 grid((total_elements + LLIST_CUDA_BLOCK_SIZE - 1) / LLIST_CUDA_BLOCK_SIZE);
    dim3 block(LLIST_CUDA_BLOCK_SIZE);
    
    llist_cuda_filter_kernel<<<grid, block, 0, stream>>>(
        batch->flat_data,
        batch->flat_valid_mask,
        batch->list_offsets,
        batch->list_counts,
        batch->num_lists,
        filter_criteria,
        filter_func
    );
    
    return cudaGetLastError();
}

/* === Device Function Examples for Common SUPERDARN Operations === */

/**
 * @brief Device function for LMFIT processing - replaces do_LMFIT callback
 */
__device__ void lmfit_process_range(void* range_node, void* acf_data, void* fit_params) {
    // This would contain the actual LMFIT algorithm
    // Converted from the original callback function
    // Example placeholder - actual implementation would be much more complex
}

/**
 * @brief Device function for elevation finding - replaces find_elevation callback
 */
__device__ void find_elevation_process(void* range_node, void* elv_data, void* fit_params) {
    // Elevation calculation algorithm
    // Converted from original callback
}

/**
 * @brief Device function for TX overlap filtering
 */
__device__ bool tx_overlap_filter(void* element, void* bad_samples) {
    // Return true if element should be filtered out
    // Based on TX overlap criteria
    return false; // Placeholder
}
