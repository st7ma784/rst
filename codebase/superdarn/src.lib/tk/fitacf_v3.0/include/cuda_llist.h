#ifndef CUDA_LLIST_H
#define CUDA_LLIST_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SUPERDARN CUDA-Compatible Linked List Data Structures
 * 
 * This header defines CUDA-compatible data structures that replace
 * traditional linked lists with array-based structures using validity masks.
 * This approach eliminates dynamic memory allocation on GPU while maintaining
 * the same logical interface as the original CPU implementation.
 */

// ============================================================================
// CUDA-Compatible Data Structures
// ============================================================================

/**
 * SUPERDARN ACF (Auto-Correlation Function) data structure
 * Represents lag data for a single range gate
 */
typedef struct {
    float real;              // Real component of ACF
    float imag;              // Imaginary component of ACF
    float power;             // Power level
    float velocity;          // Doppler velocity
    float phase_correction;  // Phase correction factor
    int lag_number;          // Lag index
    int range_gate;          // Range gate index
    float quality_flag;      // Quality metric (0.0-1.0)
} acf_data_t;

/**
 * CUDA-compatible linked list structure
 * Uses arrays with validity masks instead of dynamic pointers
 */
typedef struct {
    void** data;             // Array of data pointers
    bool* mask;              // Validity mask (true = valid element)
    int size;                // Current number of valid elements
    int capacity;            // Maximum capacity
    int iterator_pos;        // Current iterator position
    bool sorted;             // Whether the list is currently sorted
    int sort_type;           // Type of sorting applied (0=none, 1=power, 2=velocity)
} cuda_llist_t;

/**
 * Batch processing configuration
 */
typedef struct {
    int num_range_gates;     // Number of range gates to process
    int max_lags_per_gate;   // Maximum lags per range gate
    float noise_threshold;   // Noise filtering threshold
    float quality_threshold; // Quality filtering threshold
    bool enable_sorting;     // Whether to sort results
    int sort_criteria;       // Sorting criteria (0=power, 1=velocity)
} cuda_batch_config_t;

/**
 * Performance metrics structure
 */
typedef struct {
    float processing_time_ms;    // Total processing time
    int total_elements_processed; // Total data elements processed
    int valid_range_gates;       // Number of valid range gates
    float throughput_mps;        // Million elements per second
    float speedup_factor;        // Speedup vs CPU implementation
} cuda_performance_metrics_t;

// ============================================================================
// CUDA Memory Management Functions
// ============================================================================

/**
 * Allocate CUDA-compatible linked list on device
 */
cudaError_t cuda_llist_create(cuda_llist_t** d_list, int capacity);

/**
 * Free CUDA-compatible linked list on device
 */
cudaError_t cuda_llist_destroy(cuda_llist_t* d_list);

/**
 * Copy data from CPU linked list to CUDA structure
 */
cudaError_t cuda_llist_copy_from_cpu(cuda_llist_t* d_list, void* cpu_llist);

/**
 * Copy results from CUDA structure back to CPU
 */
cudaError_t cuda_llist_copy_to_cpu(void* cpu_llist, cuda_llist_t* d_list);

/**
 * Allocate batch of CUDA linked lists for multiple range gates
 */
cudaError_t cuda_llist_batch_create(cuda_llist_t** d_lists, int num_lists, int capacity_per_list);

/**
 * Free batch of CUDA linked lists
 */
cudaError_t cuda_llist_batch_destroy(cuda_llist_t* d_lists, int num_lists);

// ============================================================================
// CUDA Kernel Launch Functions
// ============================================================================

/**
 * Launch batch ACF processing across multiple range gates
 */
cudaError_t launch_batch_acf_processing(
    cuda_llist_t* d_lists,
    int num_lists,
    float* d_acf_results,
    int max_lags,
    float noise_threshold
);

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
);

/**
 * Launch parallel sorting within range gates
 */
cudaError_t launch_parallel_sorting(
    cuda_llist_t* d_lists,
    int num_lists,
    int sort_type
);

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
);

// ============================================================================
// High-Level CUDA Processing Interface
// ============================================================================

/**
 * Process SUPERDARN data using CUDA acceleration
 * This is the main entry point for CUDA-accelerated processing
 */
cudaError_t cuda_process_superdarn_data(
    void** cpu_range_gate_lists,    // Array of CPU linked lists (one per range gate)
    int num_range_gates,            // Number of range gates
    cuda_batch_config_t* config,    // Processing configuration
    float* output_acf_results,      // Output ACF results
    int* output_filtered_gates,     // Output filtered range gate indices
    cuda_performance_metrics_t* metrics  // Performance metrics output
);

/**
 * Benchmark CUDA vs CPU performance
 */
cudaError_t cuda_benchmark_performance(
    int num_range_gates,
    int elements_per_gate,
    cuda_performance_metrics_t* cuda_metrics,
    cuda_performance_metrics_t* cpu_metrics
);

// ============================================================================
// Compatibility Layer Functions
// ============================================================================

/**
 * Check if CUDA is available and properly configured
 */
bool cuda_is_available(void);

/**
 * Get CUDA device information
 */
cudaError_t cuda_get_device_info(int* device_count, size_t* total_memory, int* compute_capability);

/**
 * Initialize CUDA context and allocate resources
 */
cudaError_t cuda_initialize(void);

/**
 * Cleanup CUDA resources
 */
cudaError_t cuda_cleanup(void);

// ============================================================================
// Error Handling and Debugging
// ============================================================================

/**
 * Convert CUDA error to human-readable string
 */
const char* cuda_get_error_string(cudaError_t error);

/**
 * Print CUDA performance metrics
 */
void cuda_print_performance_metrics(const cuda_performance_metrics_t* metrics, const char* label);

/**
 * Validate CUDA results against CPU baseline
 */
bool cuda_validate_results(
    float* cuda_results,
    float* cpu_results,
    int num_elements,
    float tolerance
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_LLIST_H
