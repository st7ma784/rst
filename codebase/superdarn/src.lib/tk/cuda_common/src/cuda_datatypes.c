/**
 * @file cuda_datatypes.c
 * @brief Implementation of standardized CUDA datatypes and utilities
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#include "cuda_datatypes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

// Global state
static compute_mode_t g_compute_mode = COMPUTE_MODE_AUTO;
static bool g_cuda_initialized = false;
static int g_cuda_device_count = 0;

// =============================================================================
// CUDA AVAILABILITY AND COMPATIBILITY
// =============================================================================

bool cuda_is_available(void) {
    if (!g_cuda_initialized) {
#ifdef __CUDACC__
        cudaError_t err = cudaGetDeviceCount(&g_cuda_device_count);
        g_cuda_initialized = true;
        return (err == cudaSuccess && g_cuda_device_count > 0);
#else
        g_cuda_initialized = true;
        g_cuda_device_count = 0;
        return false;
#endif
    }
    return g_cuda_device_count > 0;
}

int cuda_get_device_info(int device_id, cuda_device_info_t *info) {
    if (!info) return CUDA_ERROR_INVALID_ARGUMENT;
    
#ifdef __CUDACC__
    if (!cuda_is_available() || device_id >= g_cuda_device_count) {
        return CUDA_ERROR_DEVICE_NOT_AVAILABLE;
    }
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return cuda_convert_error(err);
    }
    
    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    info->total_memory = prop.totalGlobalMem;
    info->compute_capability_major = prop.major;
    info->compute_capability_minor = prop.minor;
    info->multiprocessor_count = prop.multiProcessorCount;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->max_blocks_per_grid = prop.maxGridSize[0];
    
    // Get current free memory
    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err == cudaSuccess) {
        info->free_memory = free_mem;
    } else {
        info->free_memory = 0;
    }
    
    return CUDA_SUCCESS;
#else
    return CUDA_ERROR_DEVICE_NOT_AVAILABLE;
#endif
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

cuda_error_t cuda_convert_error(cudaError_t cuda_err) {
#ifdef __CUDACC__
    switch (cuda_err) {
        case cudaSuccess:
            return CUDA_SUCCESS;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidMemcpyDirection:
            return CUDA_ERROR_INVALID_ARGUMENT;
        case cudaErrorMemoryAllocation:
        case cudaErrorOutOfMemory:
            return CUDA_ERROR_OUT_OF_MEMORY;
        case cudaErrorNoDevice:
        case cudaErrorInvalidDevice:
            return CUDA_ERROR_DEVICE_NOT_AVAILABLE;
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorLaunchOutOfResources:
            return CUDA_ERROR_KERNEL_LAUNCH_FAILED;
        case cudaErrorDeviceNotReady:
        case cudaErrorSyncDepthExceeded:
            return CUDA_ERROR_SYNCHRONIZATION_FAILED;
        default:
            return CUDA_ERROR_UNKNOWN;
    }
#else
    (void)cuda_err;
    return CUDA_ERROR_DEVICE_NOT_AVAILABLE;
#endif
}

const char* cuda_get_error_string(cuda_error_t err) {
    switch (err) {
        case CUDA_SUCCESS:
            return "Success";
        case CUDA_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case CUDA_ERROR_DEVICE_NOT_AVAILABLE:
            return "CUDA device not available";
        case CUDA_ERROR_KERNEL_LAUNCH_FAILED:
            return "Kernel launch failed";
        case CUDA_ERROR_SYNCHRONIZATION_FAILED:
            return "Synchronization failed";
        case CUDA_ERROR_COPY_FAILED:
            return "Memory copy failed";
        case CUDA_ERROR_UNKNOWN:
        default:
            return "Unknown error";
    }
}

// =============================================================================
// MEMORY MANAGEMENT
// =============================================================================

cuda_error_t cuda_memory_alloc(cuda_memory_t *mem, size_t size, bool use_managed) {
    if (!mem || size == 0) return CUDA_ERROR_INVALID_ARGUMENT;
    
    memset(mem, 0, sizeof(cuda_memory_t));
    mem->size = size;
    mem->is_managed = use_managed;
    
#ifdef __CUDACC__
    if (!cuda_is_available()) {
        // Fallback to host-only allocation
        mem->host_ptr = malloc(size);
        if (!mem->host_ptr) return CUDA_ERROR_OUT_OF_MEMORY;
        mem->host_valid = true;
        return CUDA_SUCCESS;
    }
    
    if (use_managed) {
        // Use CUDA unified memory
        cudaError_t err = cudaMallocManaged(&mem->device_ptr, size);
        if (err != cudaSuccess) return cuda_convert_error(err);
        mem->host_ptr = mem->device_ptr;  // Same pointer for managed memory
        mem->host_valid = mem->device_valid = true;
    } else {
        // Separate host and device allocations
        mem->host_ptr = malloc(size);
        if (!mem->host_ptr) return CUDA_ERROR_OUT_OF_MEMORY;
        
        cudaError_t err = cudaMalloc(&mem->device_ptr, size);
        if (err != cudaSuccess) {
            free(mem->host_ptr);
            mem->host_ptr = NULL;
            return cuda_convert_error(err);
        }
        mem->host_valid = true;
        mem->device_valid = false;
    }
#else
    // CPU-only fallback
    mem->host_ptr = malloc(size);
    if (!mem->host_ptr) return CUDA_ERROR_OUT_OF_MEMORY;
    mem->host_valid = true;
#endif
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_memory_free(cuda_memory_t *mem) {
    if (!mem) return CUDA_ERROR_INVALID_ARGUMENT;
    
#ifdef __CUDACC__
    if (mem->is_managed && mem->device_ptr) {
        cudaFree(mem->device_ptr);
    } else {
        if (mem->host_ptr) free(mem->host_ptr);
        if (mem->device_ptr) cudaFree(mem->device_ptr);
    }
#else
    if (mem->host_ptr) free(mem->host_ptr);
#endif
    
    memset(mem, 0, sizeof(cuda_memory_t));
    return CUDA_SUCCESS;
}

cuda_error_t cuda_memory_host_to_device(cuda_memory_t *mem) {
    if (!mem || !mem->host_ptr) return CUDA_ERROR_INVALID_ARGUMENT;
    if (mem->is_managed) return CUDA_SUCCESS;  // No copy needed for managed memory
    
#ifdef __CUDACC__
    if (!mem->device_ptr) return CUDA_ERROR_INVALID_ARGUMENT;
    
    cudaError_t err = cudaMemcpy(mem->device_ptr, mem->host_ptr, mem->size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cuda_convert_error(err);
    
    mem->device_valid = true;
#endif
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_memory_device_to_host(cuda_memory_t *mem) {
    if (!mem || !mem->host_ptr) return CUDA_ERROR_INVALID_ARGUMENT;
    if (mem->is_managed) return CUDA_SUCCESS;  // No copy needed for managed memory
    
#ifdef __CUDACC__
    if (!mem->device_ptr) return CUDA_ERROR_INVALID_ARGUMENT;
    
    cudaError_t err = cudaMemcpy(mem->host_ptr, mem->device_ptr, mem->size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return cuda_convert_error(err);
    
    mem->host_valid = true;
#endif
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_memory_sync(cuda_memory_t *mem, bool to_device) {
    if (!mem) return CUDA_ERROR_INVALID_ARGUMENT;
    
    if (to_device) {
        return cuda_memory_host_to_device(mem);
    } else {
        return cuda_memory_device_to_host(mem);
    }
}

// =============================================================================
// ARRAY OPERATIONS
// =============================================================================

cuda_error_t cuda_array_create(cuda_array_t *arr, size_t element_size, size_t capacity) {
    if (!arr || element_size == 0 || capacity == 0) return CUDA_ERROR_INVALID_ARGUMENT;
    
    memset(arr, 0, sizeof(cuda_array_t));
    arr->element_size = element_size;
    arr->capacity = capacity;
    arr->count = 0;
    
    size_t total_size = element_size * capacity;
    return cuda_memory_alloc(&arr->memory, total_size, cuda_should_use_gpu(capacity));
}

cuda_error_t cuda_array_destroy(cuda_array_t *arr) {
    if (!arr) return CUDA_ERROR_INVALID_ARGUMENT;
    
    cuda_error_t err = cuda_memory_free(&arr->memory);
    memset(arr, 0, sizeof(cuda_array_t));
    return err;
}

cuda_error_t cuda_array_resize(cuda_array_t *arr, size_t new_capacity) {
    if (!arr || new_capacity == 0) return CUDA_ERROR_INVALID_ARGUMENT;
    if (new_capacity == arr->capacity) return CUDA_SUCCESS;
    
    // Create new memory
    cuda_memory_t new_memory;
    size_t new_size = arr->element_size * new_capacity;
    cuda_error_t err = cuda_memory_alloc(&new_memory, new_size, arr->memory.is_managed);
    if (err != CUDA_SUCCESS) return err;
    
    // Copy existing data
    if (arr->count > 0) {
        size_t copy_count = (arr->count < new_capacity) ? arr->count : new_capacity;
        size_t copy_size = copy_count * arr->element_size;
        
        if (arr->memory.host_ptr && new_memory.host_ptr) {
            memcpy(new_memory.host_ptr, arr->memory.host_ptr, copy_size);
            new_memory.host_valid = true;
        }
        
#ifdef __CUDACC__
        if (arr->memory.device_ptr && new_memory.device_ptr && !new_memory.is_managed) {
            cudaMemcpy(new_memory.device_ptr, arr->memory.device_ptr, copy_size, cudaMemcpyDeviceToDevice);
            new_memory.device_valid = true;
        }
#endif
    }
    
    // Replace old memory
    cuda_memory_free(&arr->memory);
    arr->memory = new_memory;
    arr->capacity = new_capacity;
    if (arr->count > new_capacity) arr->count = new_capacity;
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_array_push(cuda_array_t *arr, const void *element) {
    if (!arr || !element) return CUDA_ERROR_INVALID_ARGUMENT;
    
    if (arr->count >= arr->capacity) {
        // Auto-resize with 50% growth
        size_t new_capacity = arr->capacity + (arr->capacity / 2);
        if (new_capacity <= arr->capacity) new_capacity = arr->capacity + 1;
        
        cuda_error_t err = cuda_array_resize(arr, new_capacity);
        if (err != CUDA_SUCCESS) return err;
    }
    
    return cuda_array_set(arr, arr->count++, element);
}

cuda_error_t cuda_array_get(cuda_array_t *arr, size_t index, void *element) {
    if (!arr || !element || index >= arr->count) return CUDA_ERROR_INVALID_ARGUMENT;
    
    if (!arr->memory.host_valid) {
        cuda_error_t err = cuda_memory_device_to_host(&arr->memory);
        if (err != CUDA_SUCCESS) return err;
    }
    
    char *base = (char*)arr->memory.host_ptr;
    memcpy(element, base + (index * arr->element_size), arr->element_size);
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_array_set(cuda_array_t *arr, size_t index, const void *element) {
    if (!arr || !element || index >= arr->capacity) return CUDA_ERROR_INVALID_ARGUMENT;
    
    if (!arr->memory.host_ptr) return CUDA_ERROR_INVALID_ARGUMENT;
    
    char *base = (char*)arr->memory.host_ptr;
    memcpy(base + (index * arr->element_size), element, arr->element_size);
    
    arr->memory.host_valid = true;
    if (!arr->memory.is_managed) arr->memory.device_valid = false;
    
    return CUDA_SUCCESS;
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

cuda_error_t cuda_matrix_create(cuda_matrix_t *mat, size_t element_size, size_t rows, size_t cols) {
    if (!mat || element_size == 0 || rows == 0 || cols == 0) return CUDA_ERROR_INVALID_ARGUMENT;
    
    memset(mat, 0, sizeof(cuda_matrix_t));
    mat->element_size = element_size;
    mat->rows = rows;
    mat->cols = cols;
    
    // Align stride to 32-byte boundary for better performance
    mat->stride = ((cols * element_size + 31) / 32) * 32 / element_size;
    
    size_t total_size = mat->stride * element_size * rows;
    return cuda_memory_alloc(&mat->memory, total_size, cuda_should_use_gpu(rows * cols));
}

cuda_error_t cuda_matrix_destroy(cuda_matrix_t *mat) {
    if (!mat) return CUDA_ERROR_INVALID_ARGUMENT;
    
    cuda_error_t err = cuda_memory_free(&mat->memory);
    memset(mat, 0, sizeof(cuda_matrix_t));
    return err;
}

cuda_error_t cuda_matrix_get(cuda_matrix_t *mat, size_t row, size_t col, void *element) {
    if (!mat || !element || row >= mat->rows || col >= mat->cols) return CUDA_ERROR_INVALID_ARGUMENT;
    
    if (!mat->memory.host_valid) {
        cuda_error_t err = cuda_memory_device_to_host(&mat->memory);
        if (err != CUDA_SUCCESS) return err;
    }
    
    char *base = (char*)mat->memory.host_ptr;
    size_t offset = (row * mat->stride + col) * mat->element_size;
    memcpy(element, base + offset, mat->element_size);
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_matrix_set(cuda_matrix_t *mat, size_t row, size_t col, const void *element) {
    if (!mat || !element || row >= mat->rows || col >= mat->cols) return CUDA_ERROR_INVALID_ARGUMENT;
    
    if (!mat->memory.host_ptr) return CUDA_ERROR_INVALID_ARGUMENT;
    
    char *base = (char*)mat->memory.host_ptr;
    size_t offset = (row * mat->stride + col) * mat->element_size;
    memcpy(base + offset, element, mat->element_size);
    
    mat->memory.host_valid = true;
    if (!mat->memory.is_managed) mat->memory.device_valid = false;
    
    return CUDA_SUCCESS;
}

// =============================================================================
// PERFORMANCE PROFILING
// =============================================================================

cuda_error_t cuda_timer_create(cuda_timer_t *timer) {
    if (!timer) return CUDA_ERROR_INVALID_ARGUMENT;
    
    memset(timer, 0, sizeof(cuda_timer_t));
    
#ifdef __CUDACC__
    if (cuda_is_available()) {
        CUDA_CHECK(cudaEventCreate(&timer->start_event));
        CUDA_CHECK(cudaEventCreate(&timer->stop_event));
    }
#endif
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_timer_destroy(cuda_timer_t *timer) {
    if (!timer) return CUDA_ERROR_INVALID_ARGUMENT;
    
#ifdef __CUDACC__
    if (timer->start_event) cudaEventDestroy(timer->start_event);
    if (timer->stop_event) cudaEventDestroy(timer->stop_event);
#endif
    
    memset(timer, 0, sizeof(cuda_timer_t));
    return CUDA_SUCCESS;
}

cuda_error_t cuda_timer_start(cuda_timer_t *timer) {
    if (!timer) return CUDA_ERROR_INVALID_ARGUMENT;
    
#ifdef __CUDACC__
    if (timer->start_event) {
        CUDA_CHECK(cudaEventRecord(timer->start_event));
        timer->is_recording = true;
    }
#endif
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_timer_stop(cuda_timer_t *timer) {
    if (!timer || !timer->is_recording) return CUDA_ERROR_INVALID_ARGUMENT;
    
#ifdef __CUDACC__
    if (timer->stop_event) {
        CUDA_CHECK(cudaEventRecord(timer->stop_event));
        CUDA_CHECK(cudaEventSynchronize(timer->stop_event));
        timer->is_recording = false;
    }
#endif
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_timer_get_elapsed(cuda_timer_t *timer, float *elapsed_ms) {
    if (!timer || !elapsed_ms) return CUDA_ERROR_INVALID_ARGUMENT;
    
#ifdef __CUDACC__
    if (timer->start_event && timer->stop_event) {
        CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, timer->start_event, timer->stop_event));
    } else {
        *elapsed_ms = 0.0f;
    }
#else
    *elapsed_ms = 0.0f;
#endif
    
    return CUDA_SUCCESS;
}

// =============================================================================
// KERNEL UTILITIES
// =============================================================================

void cuda_calculate_launch_params(size_t total_threads, dim3 *block_size, dim3 *grid_size) {
    if (!block_size || !grid_size) return;
    
    // Default block size - good for most kernels
    const int default_block_size = 256;
    
    block_size->x = default_block_size;
    block_size->y = 1;
    block_size->z = 1;
    
    // Calculate grid size
    grid_size->x = (total_threads + default_block_size - 1) / default_block_size;
    grid_size->y = 1;
    grid_size->z = 1;
    
    // Limit grid size to maximum
    const int max_grid_size = 65535;
    if (grid_size->x > max_grid_size) {
        grid_size->x = max_grid_size;
    }
}

int cuda_get_optimal_block_size(const void *kernel_func, size_t dynamic_smem_size) {
#ifdef __CUDACC__
    if (cuda_is_available() && kernel_func) {
        int min_grid_size, block_size;
        cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, kernel_func, dynamic_smem_size, 0);
        if (err == cudaSuccess) {
            return block_size;
        }
    }
#else
    (void)kernel_func;
    (void)dynamic_smem_size;
#endif
    
    return 256;  // Default fallback
}

// =============================================================================
// COMPATIBILITY LAYER
// =============================================================================

cuda_error_t cuda_set_compute_mode(compute_mode_t mode) {
    g_compute_mode = mode;
    return CUDA_SUCCESS;
}

compute_mode_t cuda_get_compute_mode(void) {
    return g_compute_mode;
}

bool cuda_should_use_gpu(size_t min_elements) {
    switch (g_compute_mode) {
        case COMPUTE_MODE_CPU:
            return false;
        case COMPUTE_MODE_CUDA:
            return cuda_is_available();
        case COMPUTE_MODE_AUTO:
        default:
            // Use GPU if available and problem size is large enough
            return cuda_is_available() && (min_elements >= 1000);
    }
}
