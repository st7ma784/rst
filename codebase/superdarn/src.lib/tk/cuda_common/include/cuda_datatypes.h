#ifndef CUDA_DATATYPES_H
#define CUDA_DATATYPES_H

/**
 * @file cuda_datatypes.h
 * @brief Standardized CUDA datatypes and utilities for SuperDARN modules
 * 
 * This header provides a unified interface for CUDA operations across all
 * SuperDARN modules, ensuring consistent memory management, error handling,
 * and data structures.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

// =============================================================================
// CUDA AVAILABILITY AND COMPATIBILITY
// =============================================================================

/**
 * @brief Check if CUDA is available at runtime
 * @return true if CUDA is available, false otherwise
 */
bool cuda_is_available(void);

/**
 * @brief Get CUDA device information
 * @param device_id Device ID to query
 * @param info Pointer to device info structure to fill
 * @return 0 on success, negative on error
 */
typedef struct {
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_blocks_per_grid;
} cuda_device_info_t;

int cuda_get_device_info(int device_id, cuda_device_info_t *info);

// =============================================================================
// STANDARDIZED ERROR HANDLING
// =============================================================================

typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_ARGUMENT = -1,
    CUDA_ERROR_OUT_OF_MEMORY = -2,
    CUDA_ERROR_DEVICE_NOT_AVAILABLE = -3,
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = -4,
    CUDA_ERROR_SYNCHRONIZATION_FAILED = -5,
    CUDA_ERROR_COPY_FAILED = -6,
    CUDA_ERROR_UNKNOWN = -99
} cuda_error_t;

/**
 * @brief Convert CUDA runtime error to standardized error code
 */
cuda_error_t cuda_convert_error(cudaError_t cuda_err);

/**
 * @brief Get error string for standardized error code
 */
const char* cuda_get_error_string(cuda_error_t err);

/**
 * @brief Macro for checking CUDA errors with automatic error reporting
 */
#define CUDA_CHECK(call) do { \
    cudaError_t cuda_err = (call); \
    if (cuda_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(cuda_err)); \
        return cuda_convert_error(cuda_err); \
    } \
} while(0)

// =============================================================================
// STANDARDIZED MEMORY MANAGEMENT
// =============================================================================

/**
 * @brief Unified memory management structure
 */
typedef struct {
    void *host_ptr;      ///< Host memory pointer
    void *device_ptr;    ///< Device memory pointer
    size_t size;         ///< Size in bytes
    bool is_managed;     ///< Whether using unified memory
    bool host_valid;     ///< Whether host data is current
    bool device_valid;   ///< Whether device data is current
} cuda_memory_t;

/**
 * @brief Allocate unified CUDA memory
 * @param mem Pointer to memory structure to initialize
 * @param size Size in bytes to allocate
 * @param use_managed Whether to use CUDA managed memory
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_memory_alloc(cuda_memory_t *mem, size_t size, bool use_managed);

/**
 * @brief Free CUDA memory
 * @param mem Memory structure to free
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_memory_free(cuda_memory_t *mem);

/**
 * @brief Copy data from host to device
 * @param mem Memory structure
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_memory_host_to_device(cuda_memory_t *mem);

/**
 * @brief Copy data from device to host
 * @param mem Memory structure
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_memory_device_to_host(cuda_memory_t *mem);

/**
 * @brief Synchronize memory between host and device
 * @param mem Memory structure
 * @param to_device If true, sync to device; if false, sync to host
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_memory_sync(cuda_memory_t *mem, bool to_device);

// =============================================================================
// STANDARDIZED DATA STRUCTURES
// =============================================================================

/**
 * @brief CUDA-compatible array structure
 */
typedef struct {
    cuda_memory_t memory;
    size_t element_size;
    size_t count;
    size_t capacity;
} cuda_array_t;

/**
 * @brief CUDA-compatible matrix structure
 */
typedef struct {
    cuda_memory_t memory;
    size_t element_size;
    size_t rows;
    size_t cols;
    size_t stride;  ///< Row stride for alignment
} cuda_matrix_t;

/**
 * @brief CUDA-compatible complex number
 */
typedef struct {
    float real;
    float imag;
} cuda_complex_t;

/**
 * @brief CUDA-compatible double precision complex number
 */
typedef struct {
    double real;
    double imag;
} cuda_complex_double_t;

// Array operations
cuda_error_t cuda_array_create(cuda_array_t *arr, size_t element_size, size_t capacity);
cuda_error_t cuda_array_destroy(cuda_array_t *arr);
cuda_error_t cuda_array_resize(cuda_array_t *arr, size_t new_capacity);
cuda_error_t cuda_array_push(cuda_array_t *arr, const void *element);
cuda_error_t cuda_array_get(cuda_array_t *arr, size_t index, void *element);
cuda_error_t cuda_array_set(cuda_array_t *arr, size_t index, const void *element);

// Matrix operations
cuda_error_t cuda_matrix_create(cuda_matrix_t *mat, size_t element_size, size_t rows, size_t cols);
cuda_error_t cuda_matrix_destroy(cuda_matrix_t *mat);
cuda_error_t cuda_matrix_get(cuda_matrix_t *mat, size_t row, size_t col, void *element);
cuda_error_t cuda_matrix_set(cuda_matrix_t *mat, size_t row, size_t col, const void *element);

// =============================================================================
// CUDA LINKED LIST (GPU-COMPATIBLE)
// =============================================================================

/**
 * @brief GPU-compatible linked list node
 */
typedef struct cuda_list_node {
    void *data;
    struct cuda_list_node *next;
    struct cuda_list_node *prev;
} cuda_list_node_t;

/**
 * @brief GPU-compatible linked list
 */
typedef struct {
    cuda_memory_t nodes_memory;  ///< Pre-allocated node pool
    cuda_list_node_t *head;
    cuda_list_node_t *tail;
    cuda_list_node_t *free_list; ///< Free node pool
    size_t node_size;
    size_t max_nodes;
    size_t current_count;
    bool is_on_device;
} cuda_list_t;

// CUDA list operations
cuda_error_t cuda_list_create(cuda_list_t *list, size_t node_size, size_t max_nodes);
cuda_error_t cuda_list_destroy(cuda_list_t *list);
cuda_error_t cuda_list_push_front(cuda_list_t *list, const void *data);
cuda_error_t cuda_list_push_back(cuda_list_t *list, const void *data);
cuda_error_t cuda_list_pop_front(cuda_list_t *list, void *data);
cuda_error_t cuda_list_pop_back(cuda_list_t *list, void *data);
cuda_error_t cuda_list_find(cuda_list_t *list, const void *data, 
                            int (*compare)(const void *a, const void *b),
                            cuda_list_node_t **result);
cuda_error_t cuda_list_remove(cuda_list_t *list, cuda_list_node_t *node);
size_t cuda_list_size(const cuda_list_t *list);
bool cuda_list_empty(const cuda_list_t *list);

// =============================================================================
// PERFORMANCE PROFILING
// =============================================================================

/**
 * @brief CUDA performance timer
 */
typedef struct {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool is_recording;
} cuda_timer_t;

cuda_error_t cuda_timer_create(cuda_timer_t *timer);
cuda_error_t cuda_timer_destroy(cuda_timer_t *timer);
cuda_error_t cuda_timer_start(cuda_timer_t *timer);
cuda_error_t cuda_timer_stop(cuda_timer_t *timer);
cuda_error_t cuda_timer_get_elapsed(cuda_timer_t *timer, float *elapsed_ms);

/**
 * @brief Performance profiling structure
 */
typedef struct {
    cuda_timer_t total_timer;
    cuda_timer_t kernel_timer;
    cuda_timer_t memory_timer;
    size_t memory_transfers;
    size_t kernel_launches;
    float total_time_ms;
    float kernel_time_ms;
    float memory_time_ms;
} cuda_profile_t;

cuda_error_t cuda_profile_init(cuda_profile_t *profile);
cuda_error_t cuda_profile_cleanup(cuda_profile_t *profile);
cuda_error_t cuda_profile_start_total(cuda_profile_t *profile);
cuda_error_t cuda_profile_stop_total(cuda_profile_t *profile);
cuda_error_t cuda_profile_start_kernel(cuda_profile_t *profile);
cuda_error_t cuda_profile_stop_kernel(cuda_profile_t *profile);
cuda_error_t cuda_profile_start_memory(cuda_profile_t *profile);
cuda_error_t cuda_profile_stop_memory(cuda_profile_t *profile);
void cuda_profile_print_summary(const cuda_profile_t *profile, const char *operation_name);

// =============================================================================
// CUDA KERNEL UTILITIES
// =============================================================================

/**
 * @brief Calculate optimal CUDA grid and block dimensions
 * @param total_threads Total number of threads needed
 * @param block_size Pointer to store calculated block size
 * @param grid_size Pointer to store calculated grid size
 */
void cuda_calculate_launch_params(size_t total_threads, dim3 *block_size, dim3 *grid_size);

/**
 * @brief Get optimal block size for a given kernel
 * @param kernel_func Pointer to kernel function
 * @param dynamic_smem_size Dynamic shared memory size per block
 * @return Optimal block size
 */
int cuda_get_optimal_block_size(const void *kernel_func, size_t dynamic_smem_size);

// =============================================================================
// COMPATIBILITY LAYER
// =============================================================================

/**
 * @brief Runtime selection between CPU and CUDA implementations
 */
typedef enum {
    COMPUTE_MODE_AUTO,   ///< Automatically select best available
    COMPUTE_MODE_CPU,    ///< Force CPU implementation
    COMPUTE_MODE_CUDA    ///< Force CUDA implementation
} compute_mode_t;

/**
 * @brief Set global compute mode for all operations
 * @param mode Compute mode to use
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_set_compute_mode(compute_mode_t mode);

/**
 * @brief Get current compute mode
 * @return Current compute mode
 */
compute_mode_t cuda_get_compute_mode(void);

/**
 * @brief Check if CUDA should be used for current operation
 * @param min_elements Minimum number of elements to justify CUDA overhead
 * @return true if CUDA should be used, false for CPU
 */
bool cuda_should_use_gpu(size_t min_elements);

#ifdef __cplusplus
}
#endif

#endif // CUDA_DATATYPES_H
