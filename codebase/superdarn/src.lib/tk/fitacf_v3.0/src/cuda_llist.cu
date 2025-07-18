#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cuda_llist.h"
#include "llist.h"

/**
 * SUPERDARN CUDA Linked List Implementation
 * 
 * This file implements the CUDA-compatible linked list functions
 * and provides the bridge between CPU and GPU processing.
 */

// ============================================================================
// CUDA Memory Management Implementation
// ============================================================================

cudaError_t cuda_llist_create(cuda_llist_t** d_list, int capacity) {
    cudaError_t err;
    cuda_llist_t* h_list;
    cuda_llist_t* d_list_ptr;
    
    // Allocate host structure
    h_list = (cuda_llist_t*)malloc(sizeof(cuda_llist_t));
    if (!h_list) return cudaErrorMemoryAllocation;
    
    // Initialize host structure
    h_list->size = 0;
    h_list->capacity = capacity;
    h_list->iterator_pos = 0;
    h_list->sorted = false;
    h_list->sort_type = 0;
    
    // Allocate device memory for the structure
    err = cudaMalloc((void**)&d_list_ptr, sizeof(cuda_llist_t));
    if (err != cudaSuccess) {
        free(h_list);
        return err;
    }
    
    // Allocate device memory for data arrays
    void** d_data;
    bool* d_mask;
    
    err = cudaMalloc((void**)&d_data, capacity * sizeof(void*));
    if (err != cudaSuccess) {
        cudaFree(d_list_ptr);
        free(h_list);
        return err;
    }
    
    err = cudaMalloc((void**)&d_mask, capacity * sizeof(bool));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_list_ptr);
        free(h_list);
        return err;
    }
    
    // Initialize device arrays
    cudaMemset(d_data, 0, capacity * sizeof(void*));
    cudaMemset(d_mask, 0, capacity * sizeof(bool));
    
    // Set pointers in host structure
    h_list->data = d_data;
    h_list->mask = d_mask;
    
    // Copy structure to device
    err = cudaMemcpy(d_list_ptr, h_list, sizeof(cuda_llist_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_mask);
        cudaFree(d_data);
        cudaFree(d_list_ptr);
        free(h_list);
        return err;
    }
    
    *d_list = d_list_ptr;
    free(h_list);
    return cudaSuccess;
}

cudaError_t cuda_llist_destroy(cuda_llist_t* d_list) {
    if (!d_list) return cudaSuccess;
    
    cuda_llist_t h_list;
    cudaError_t err;
    
    // Copy structure from device to get pointers
    err = cudaMemcpy(&h_list, d_list, sizeof(cuda_llist_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    // Free device arrays
    if (h_list.data) cudaFree(h_list.data);
    if (h_list.mask) cudaFree(h_list.mask);
    
    // Free device structure
    cudaFree(d_list);
    
    return cudaSuccess;
}

cudaError_t cuda_llist_batch_create(cuda_llist_t** d_lists, int num_lists, int capacity_per_list) {
    cudaError_t err;
    cuda_llist_t* d_lists_array;
    
    // Allocate array of CUDA lists on device
    err = cudaMalloc((void**)&d_lists_array, num_lists * sizeof(cuda_llist_t));
    if (err != cudaSuccess) return err;
    
    // Create each list individually
    for (int i = 0; i < num_lists; i++) {
        cuda_llist_t* d_single_list;
        err = cuda_llist_create(&d_single_list, capacity_per_list);
        if (err != cudaSuccess) {
            // Cleanup previously created lists
            for (int j = 0; j < i; j++) {
                cuda_llist_t temp_list;
                cudaMemcpy(&temp_list, &d_lists_array[j], sizeof(cuda_llist_t), cudaMemcpyDeviceToHost);
                cuda_llist_destroy(&d_lists_array[j]);
            }
            cudaFree(d_lists_array);
            return err;
        }
        
        // Copy the created list to the array
        err = cudaMemcpy(&d_lists_array[i], d_single_list, sizeof(cuda_llist_t), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cuda_llist_destroy(d_single_list);
            cudaFree(d_lists_array);
            return err;
        }
        
        cudaFree(d_single_list);  // Free the temporary pointer
    }
    
    *d_lists = d_lists_array;
    return cudaSuccess;
}

// ============================================================================
// CPU-GPU Data Transfer Functions
// ============================================================================

cudaError_t cuda_llist_copy_from_cpu(cuda_llist_t* d_list, void* cpu_llist) {
    if (!d_list || !cpu_llist) return cudaErrorInvalidValue;
    
    llist cpu_list = (llist)cpu_llist;
    cuda_llist_t h_list;
    cudaError_t err;
    
    // Copy device structure to host to get pointers
    err = cudaMemcpy(&h_list, d_list, sizeof(cuda_llist_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    // Reset iterator and get first element
    llist_reset_iter(cpu_list);
    
    int count = 0;
    void* item;
    
    // Copy all elements from CPU list to GPU arrays
    while (llist_get_iter(cpu_list, &item) == LLIST_SUCCESS && count < h_list.capacity) {
        // Allocate device memory for this data item
        acf_data_t* d_item;
        err = cudaMalloc((void**)&d_item, sizeof(acf_data_t));
        if (err != cudaSuccess) return err;
        
        // Copy data to device
        err = cudaMemcpy(d_item, item, sizeof(acf_data_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_item);
            return err;
        }
        
        // Update device arrays
        err = cudaMemcpy(&h_list.data[count], &d_item, sizeof(void*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_item);
            return err;
        }
        
        bool valid = true;
        err = cudaMemcpy(&h_list.mask[count], &valid, sizeof(bool), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
        
        count++;
        llist_go_next(cpu_list);
    }
    
    // Update list size
    h_list.size = count;
    h_list.iterator_pos = 0;
    h_list.sorted = false;
    
    // Copy updated structure back to device
    err = cudaMemcpy(d_list, &h_list, sizeof(cuda_llist_t), cudaMemcpyHostToDevice);
    return err;
}

// ============================================================================
// High-Level CUDA Processing Interface
// ============================================================================

cudaError_t cuda_process_superdarn_data(
    void** cpu_range_gate_lists,
    int num_range_gates,
    cuda_batch_config_t* config,
    float* output_acf_results,
    int* output_filtered_gates,
    cuda_performance_metrics_t* metrics
) {
    cudaError_t err;
    struct timeval start_time, end_time;
    float processing_time = 0.0f;
    gettimeofday(&start_time, NULL);
    
    // Allocate CUDA lists for all range gates
    cuda_llist_t* d_lists;
    err = cuda_llist_batch_create(&d_lists, num_range_gates, config->max_lags_per_gate);
    if (err != cudaSuccess) return err;
    
    // Copy CPU data to CUDA structures
    for (int i = 0; i < num_range_gates; i++) {
        if (cpu_range_gate_lists[i]) {
            err = cuda_llist_copy_from_cpu(&d_lists[i], cpu_range_gate_lists[i]);
            if (err != cudaSuccess) {
                cuda_llist_batch_destroy(d_lists, num_range_gates);
                return err;
            }
        }
    }
    
    // Allocate device memory for results
    float* d_acf_results = NULL;
    int* d_filtered_indices = NULL;
    int* d_num_filtered = NULL;
    float* d_quality_metrics = NULL;
    float* h_quality_metrics = NULL;
    
    int total_acf_size = num_range_gates * config->max_lags_per_gate;
    
    // Initialize quality metrics (simplified for demo)
    h_quality_metrics = (float*)malloc(num_range_gates * sizeof(float));
    if (!h_quality_metrics) {
        err = cudaErrorMemoryAllocation;
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&d_acf_results, total_acf_size * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc((void**)&d_filtered_indices, num_range_gates * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc((void**)&d_num_filtered, sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc((void**)&d_quality_metrics, num_range_gates * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    for (int i = 0; i < num_range_gates; i++) {
        h_quality_metrics[i] = 0.8f;  // Default quality
    }
    cudaMemcpy(d_quality_metrics, h_quality_metrics, num_range_gates * sizeof(float), cudaMemcpyHostToDevice);
    free(h_quality_metrics);
    
    // Launch CUDA kernels
    
    // 1. Batch ACF processing
    err = launch_batch_acf_processing(
        d_lists, num_range_gates, d_acf_results,
        config->max_lags_per_gate, config->noise_threshold
    );
    if (err != cudaSuccess) goto cleanup;
    
    // 2. Range gate filtering
    err = launch_range_gate_filtering(
        d_lists, num_range_gates, d_quality_metrics,
        config->quality_threshold, d_filtered_indices, d_num_filtered
    );
    if (err != cudaSuccess) goto cleanup;
    
    // 3. Optional sorting
    if (config->enable_sorting) {
        err = launch_parallel_sorting(d_lists, num_range_gates, config->sort_criteria);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Copy results back to host
    cudaMemcpy(output_acf_results, d_acf_results, total_acf_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    int num_filtered;
    cudaMemcpy(&num_filtered, d_num_filtered, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_filtered_gates, d_filtered_indices, num_filtered * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Calculate performance metrics
    gettimeofday(&end_time, NULL);
    processing_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0f +
                     (end_time.tv_usec - start_time.tv_usec) / 1000.0f;
    
    if (metrics) {
        metrics->processing_time_ms = processing_time;
        metrics->total_elements_processed = num_range_gates * config->max_lags_per_gate;
        metrics->valid_range_gates = num_filtered;
        metrics->throughput_mps = (metrics->total_elements_processed / processing_time) / 1000.0f;
        metrics->speedup_factor = 0.0f;  // Will be calculated by comparison
    }
    
cleanup:
    // Cleanup device memory
    if (d_acf_results) cudaFree(d_acf_results);
    if (d_filtered_indices) cudaFree(d_filtered_indices);
    if (d_num_filtered) cudaFree(d_num_filtered);
    if (d_quality_metrics) cudaFree(d_quality_metrics);
    cuda_llist_batch_destroy(d_lists, num_range_gates);
    
    return err;
}

// ============================================================================
// Utility and Compatibility Functions
// ============================================================================

bool cuda_is_available(void) {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

cudaError_t cuda_get_device_info(int* device_count, size_t* total_memory, int* compute_capability) {
    cudaError_t err;
    
    err = cudaGetDeviceCount(device_count);
    if (err != cudaSuccess) return err;
    
    if (*device_count > 0) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) return err;
        
        *total_memory = prop.totalGlobalMem;
        *compute_capability = prop.major * 10 + prop.minor;
    }
    
    return cudaSuccess;
}

cudaError_t cuda_initialize(void) {
    // Set device and initialize context
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) return err;
    
    // Warm up GPU
    void* dummy;
    err = cudaMalloc(&dummy, 1024);
    if (err == cudaSuccess) {
        cudaFree(dummy);
    }
    
    return err;
}

cudaError_t cuda_cleanup(void) {
    return cudaDeviceReset();
}

const char* cuda_get_error_string(cudaError_t error) {
    return cudaGetErrorString(error);
}

void cuda_print_performance_metrics(const cuda_performance_metrics_t* metrics, const char* label) {
    printf("\n=== %s Performance Metrics ===\n", label);
    printf("Processing Time: %.2f ms\n", metrics->processing_time_ms);
    printf("Elements Processed: %d\n", metrics->total_elements_processed);
    printf("Valid Range Gates: %d\n", metrics->valid_range_gates);
    printf("Throughput: %.2f M elements/sec\n", metrics->throughput_mps);
    if (metrics->speedup_factor > 0.0f) {
        printf("Speedup Factor: %.2fx\n", metrics->speedup_factor);
    }
    printf("=====================================\n");
}

bool cuda_validate_results(
    float* cuda_results,
    float* cpu_results,
    int num_elements,
    float tolerance
) {
    for (int i = 0; i < num_elements; i++) {
        float diff = fabsf(cuda_results[i] - cpu_results[i]);
        float relative_error = diff / fmaxf(fabsf(cpu_results[i]), 1e-6f);
        
        if (relative_error > tolerance) {
            printf("Validation failed at element %d: CUDA=%.6f, CPU=%.6f, error=%.6f\n",
                   i, cuda_results[i], cpu_results[i], relative_error);
            return false;
        }
    }
    return true;
}
