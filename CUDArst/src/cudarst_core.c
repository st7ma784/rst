/**
 * @file cudarst_core.c
 * @brief Core CUDArst library implementation
 * 
 * Provides initialization, cleanup, and runtime configuration
 * for the unified CUDA-accelerated RST library.
 */

#include "cudarst.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* CUDA headers (conditionally included) */
#ifdef __NVCC__
#include <cuda_runtime.h>
#include <cuda.h>
#endif

/* Global library state */
static struct {
    bool initialized;
    cudarst_mode_t mode;
    bool cuda_available;
    int cuda_device_count;
    int cuda_active_device;
    cudarst_performance_t performance;
    struct timespec start_time;
} cudarst_state = {0};

/* Forward declarations */
static bool check_cuda_availability(void);
static void reset_performance_counters(void);

cudarst_error_t cudarst_init(cudarst_mode_t mode)
{
    if (cudarst_state.initialized) {
        return CUDARST_SUCCESS; /* Already initialized */
    }
    
    /* Check CUDA availability */
    cudarst_state.cuda_available = check_cuda_availability();
    
    /* Validate mode selection */
    if (mode == CUDARST_MODE_CUDA_ONLY && !cudarst_state.cuda_available) {
        fprintf(stderr, "CUDArst: CUDA-only mode requested but CUDA not available\n");
        return CUDARST_ERROR_CUDA_UNAVAILABLE;
    }
    
    /* Set runtime mode */
    cudarst_state.mode = mode;
    if (mode == CUDARST_MODE_AUTO) {
        cudarst_state.mode = cudarst_state.cuda_available ? 
                            CUDARST_MODE_CUDA_ONLY : CUDARST_MODE_CPU_ONLY;
    }
    
    /* Initialize CUDA if available and requested */
    if (cudarst_state.cuda_available && 
        (cudarst_state.mode == CUDARST_MODE_CUDA_ONLY || 
         cudarst_state.mode == CUDARST_MODE_AUTO)) {
#ifdef __NVCC__
        cudaError_t cuda_err = cudaSetDevice(0);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDArst: Failed to set CUDA device: %s\n", 
                    cudaGetErrorString(cuda_err));
            cudarst_state.mode = CUDARST_MODE_CPU_ONLY;
        } else {
            cudarst_state.cuda_active_device = 0;
        }
#endif
    }
    
    /* Initialize performance tracking */
    reset_performance_counters();
    clock_gettime(CLOCK_MONOTONIC, &cudarst_state.start_time);
    
    cudarst_state.initialized = true;
    
    printf("CUDArst v%s initialized - Mode: %s\n", 
           CUDARST_VERSION_STRING,
           cudarst_state.mode == CUDARST_MODE_CPU_ONLY ? "CPU" :
           cudarst_state.mode == CUDARST_MODE_CUDA_ONLY ? "CUDA" : "AUTO");
    
    return CUDARST_SUCCESS;
}

void cudarst_cleanup(void)
{
    if (!cudarst_state.initialized) {
        return;
    }
    
    /* Cleanup CUDA resources */
    if (cudarst_state.cuda_available && cudarst_state.cuda_active_device >= 0) {
#ifdef __NVCC__
        cudaDeviceReset();
#endif
    }
    
    /* Reset state */
    memset(&cudarst_state, 0, sizeof(cudarst_state));
    
    printf("CUDArst cleanup complete\n");
}

const char* cudarst_get_version(void)
{
    return CUDARST_VERSION_STRING;
}

bool cudarst_is_cuda_available(void)
{
    return cudarst_state.cuda_available;
}

cudarst_error_t cudarst_get_performance(cudarst_performance_t *perf)
{
    if (!perf) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    *perf = cudarst_state.performance;
    return CUDARST_SUCCESS;
}

void cudarst_reset_performance(void)
{
    reset_performance_counters();
}

void cudarst_print_performance(void)
{
    printf("\nCUDArst Performance Summary\n");
    printf("===========================\n");
    printf("Total Time:        %8.3f ms\n", cudarst_state.performance.total_time_ms);
    printf("CUDA Kernel Time:  %8.3f ms\n", cudarst_state.performance.cuda_time_ms);
    printf("Memory Transfer:   %8.3f ms\n", cudarst_state.performance.memory_transfer_ms);
    printf("CPU Fallback:      %8.3f ms\n", cudarst_state.performance.cpu_fallback_ms);
    printf("Memory Used:       %8zu bytes\n", cudarst_state.performance.memory_used_bytes);
    printf("CUDA Device:       %8d\n", cudarst_state.performance.cuda_device_id);
    printf("CUDA Used:         %8s\n", cudarst_state.performance.cuda_used ? "Yes" : "No");
    
    if (cudarst_state.performance.total_time_ms > 0) {
        double cuda_pct = 100.0 * cudarst_state.performance.cuda_time_ms / 
                         cudarst_state.performance.total_time_ms;
        printf("CUDA Efficiency:   %8.1f%%\n", cuda_pct);
    }
}

/* Memory management implementation */
void* cudarst_malloc(size_t size)
{
    void *ptr = NULL;
    
    if (cudarst_state.cuda_available && 
        (cudarst_state.mode == CUDARST_MODE_CUDA_ONLY || 
         cudarst_state.mode == CUDARST_MODE_AUTO)) {
#ifdef __NVCC__
        /* Use CUDA unified memory */
        cudaError_t err = cudaMallocManaged(&ptr, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDArst: CUDA malloc failed: %s\n", cudaGetErrorString(err));
            ptr = NULL;
        }
#endif
    }
    
    /* Fallback to regular malloc */
    if (!ptr) {
        ptr = malloc(size);
    }
    
    if (ptr) {
        cudarst_state.performance.memory_used_bytes += size;
    }
    
    return ptr;
}

void cudarst_free(void *ptr)
{
    if (!ptr) return;
    
    if (cudarst_state.cuda_available && 
        (cudarst_state.mode == CUDARST_MODE_CUDA_ONLY || 
         cudarst_state.mode == CUDARST_MODE_AUTO)) {
#ifdef __NVCC__
        /* Check if this is CUDA memory */
        cudaPointerAttributes attrs;
        cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
        if (err == cudaSuccess && attrs.type == cudaMemoryTypeManaged) {
            cudaFree(ptr);
            return;
        }
#endif
    }
    
    /* Regular free */
    free(ptr);
}

cudarst_error_t cudarst_memcpy(void *dst, const void *src, size_t size, bool to_device)
{
    if (!dst || !src || size == 0) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    if (cudarst_state.cuda_available && 
        (cudarst_state.mode == CUDARST_MODE_CUDA_ONLY || 
         cudarst_state.mode == CUDARST_MODE_AUTO)) {
#ifdef __NVCC__
        cudaMemcpyKind kind = to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        cudaError_t err = cudaMemcpy(dst, src, size, kind);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDArst: CUDA memcpy failed: %s\n", cudaGetErrorString(err));
            return CUDARST_ERROR_PROCESSING_FAILED;
        }
#endif
    } else {
        /* CPU-only fallback */
        memcpy(dst, src, size);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double transfer_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                          (end.tv_nsec - start.tv_nsec) / 1000000.0;
    cudarst_state.performance.memory_transfer_ms += transfer_time;
    
    return CUDARST_SUCCESS;
}

/* Internal helper functions */
static bool check_cuda_availability(void)
{
#ifdef __NVCC__
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    
    /* Test basic CUDA functionality */
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return false;
    }
    
    /* Get device properties */
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return false;
    }
    
    /* Check minimum compute capability (2.0) */
    if (prop.major < 2) {
        return false;
    }
    
    cudarst_state.cuda_device_count = device_count;
    
    printf("CUDArst: CUDA available - %d device(s), using %s (CC %d.%d)\n",
           device_count, prop.name, prop.major, prop.minor);
    
    return true;
#else
    return false;
#endif
}

static void reset_performance_counters(void)
{
    memset(&cudarst_state.performance, 0, sizeof(cudarst_state.performance));
    cudarst_state.performance.cuda_device_id = cudarst_state.cuda_active_device;
}