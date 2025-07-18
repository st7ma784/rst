#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>
#include "llist.h"

// Conditional CUDA support
#ifdef __NVCC__
#include "cuda_llist.h"
#define CUDA_SUPPORT_ENABLED 1
#else
#define CUDA_SUPPORT_ENABLED 0
// Stub definitions for non-CUDA builds
typedef int cudaError_t;
#define cudaSuccess 0
typedef struct { float processing_time_ms; int total_elements_processed; int valid_range_gates; float throughput_mps; float speedup_factor; } cuda_performance_metrics_t;
typedef struct { int num_range_gates; int max_lags_per_gate; float noise_threshold; float quality_threshold; bool enable_sorting; int sort_criteria; } cuda_batch_config_t;
typedef struct { float real; float imag; float power; float velocity; float phase_correction; int lag_number; int range_gate; float quality_flag; } acf_data_t;

// Stub function declarations
static inline bool cuda_is_available(void) { return false; }
static inline cudaError_t cuda_get_device_info(int* a, size_t* b, int* c) { return 1; }
static inline cudaError_t cuda_initialize(void) { return 1; }
static inline void cuda_cleanup(void) {}
static inline const char* cuda_get_error_string(cudaError_t err) { return "CUDA not available"; }
static inline cudaError_t cuda_process_superdarn_data(void** a, int b, cuda_batch_config_t* c, float* d, int* e, cuda_performance_metrics_t* f) { return 1; }
static inline void cuda_print_performance_metrics(const cuda_performance_metrics_t* a, const char* b) {}
static inline cudaError_t cuda_benchmark_performance(int a, int b, cuda_performance_metrics_t* c, cuda_performance_metrics_t* d) { return 1; }
#endif

/**
 * SUPERDARN CPU-CUDA Bridge
 * 
 * This file provides the bridge between the original CPU-based linked list
 * implementation and the new CUDA-accelerated processing. It allows for
 * seamless side-by-side operation and performance comparison.
 */

// Global configuration for CUDA/CPU selection
static bool g_cuda_enabled = false;
static bool g_cuda_available = false;
static bool g_cuda_initialized = false;

// Performance tracking
static cuda_performance_metrics_t g_last_cuda_metrics = {0};
static cuda_performance_metrics_t g_last_cpu_metrics = {0};

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Initialize the CPU-CUDA bridge system
 */
int bridge_initialize(void) {
    printf("=== Initializing SUPERDARN CPU-CUDA Bridge ===\n");
    
    // Check CUDA availability
#if CUDA_SUPPORT_ENABLED
    g_cuda_available = cuda_is_available();
#else
    g_cuda_available = false;
#endif
    
    if (g_cuda_available) {
        int device_count;
        size_t total_memory;
        int compute_capability;
        
        cudaError_t err = cuda_get_device_info(&device_count, &total_memory, &compute_capability);
        if (err == cudaSuccess) {
            printf("✅ CUDA Available: %d device(s)\n", device_count);
            printf("   Total GPU Memory: %.1f GB\n", total_memory / (1024.0 * 1024.0 * 1024.0));
            printf("   Compute Capability: %d.%d\n", compute_capability / 10, compute_capability % 10);
            
            // Initialize CUDA context
            err = cuda_initialize();
            if (err == cudaSuccess) {
                g_cuda_initialized = true;
                g_cuda_enabled = true;  // Enable by default if available
                printf("✅ CUDA Initialized Successfully\n");
            } else {
                printf("❌ CUDA Initialization Failed: %s\n", cuda_get_error_string(err));
            }
        } else {
            printf("❌ Failed to get CUDA device info: %s\n", cuda_get_error_string(err));
        }
    } else {
        printf("ℹ️  CUDA Not Available - Using CPU-only processing\n");
    }
    
    printf("=== Bridge Initialization Complete ===\n\n");
    return g_cuda_available ? 1 : 0;
}

/**
 * Cleanup the CPU-CUDA bridge system
 */
void bridge_cleanup(void) {
    if (g_cuda_initialized) {
        cuda_cleanup();
        g_cuda_initialized = false;
    }
    g_cuda_enabled = false;
}

/**
 * Enable or disable CUDA processing
 */
void bridge_set_cuda_enabled(bool enabled) {
    if (enabled && !g_cuda_available) {
        printf("⚠️  Cannot enable CUDA: Not available on this system\n");
        return;
    }
    
    g_cuda_enabled = enabled;
    printf("ℹ️  CUDA processing %s\n", enabled ? "ENABLED" : "DISABLED");
}

/**
 * Check if CUDA processing is currently enabled
 */
bool bridge_is_cuda_enabled(void) {
    return g_cuda_enabled && g_cuda_available && g_cuda_initialized;
}

// ============================================================================
// High-Level Processing Interface
// ============================================================================

/**
 * Process SUPERDARN range gate data using optimal method (CPU or CUDA)
 */
int bridge_process_range_gates(
    llist* range_gate_lists,       // Array of CPU linked lists
    int num_range_gates,           // Number of range gates
    int max_lags_per_gate,         // Maximum lags per range gate
    float noise_threshold,         // Noise filtering threshold
    float quality_threshold,       // Quality filtering threshold
    float* output_acf_results,     // Output ACF results
    int* output_filtered_gates,    // Output filtered range gate indices
    int* num_filtered_gates        // Number of filtered gates
) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    int result = 0;
    
    if (bridge_is_cuda_enabled()) {
        printf("🚀 Processing %d range gates using CUDA acceleration...\n", num_range_gates);
        
        // Configure CUDA processing
        cuda_batch_config_t config = {
            .num_range_gates = num_range_gates,
            .max_lags_per_gate = max_lags_per_gate,
            .noise_threshold = noise_threshold,
            .quality_threshold = quality_threshold,
            .enable_sorting = true,
            .sort_criteria = 0  // Sort by power
        };
        
        // Process using CUDA
        cudaError_t err = cuda_process_superdarn_data(
            (void**)range_gate_lists,
            num_range_gates,
            &config,
            output_acf_results,
            output_filtered_gates,
            &g_last_cuda_metrics
        );
        
        if (err == cudaSuccess) {
            *num_filtered_gates = g_last_cuda_metrics.valid_range_gates;
            result = 1;  // Success
            
            gettimeofday(&end_time, NULL);
            float total_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0f +
                              (end_time.tv_usec - start_time.tv_usec) / 1000.0f;
            
            printf("✅ CUDA processing completed in %.2f ms\n", total_time);
            cuda_print_performance_metrics(&g_last_cuda_metrics, "CUDA");
        } else {
            printf("❌ CUDA processing failed: %s\n", cuda_get_error_string(err));
            printf("🔄 Falling back to CPU processing...\n");
            
            // Fall back to CPU processing
            result = bridge_process_range_gates_cpu(
                range_gate_lists, num_range_gates, max_lags_per_gate,
                noise_threshold, quality_threshold,
                output_acf_results, output_filtered_gates, num_filtered_gates
            );
        }
    } else {
        printf("🖥️  Processing %d range gates using CPU...\n", num_range_gates);
        
        result = bridge_process_range_gates_cpu(
            range_gate_lists, num_range_gates, max_lags_per_gate,
            noise_threshold, quality_threshold,
            output_acf_results, output_filtered_gates, num_filtered_gates
        );
    }
    
    return result;
}

/**
 * CPU-only processing implementation
 */
int bridge_process_range_gates_cpu(
    llist* range_gate_lists,
    int num_range_gates,
    int max_lags_per_gate,
    float noise_threshold,
    float quality_threshold,
    float* output_acf_results,
    int* output_filtered_gates,
    int* num_filtered_gates
) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    int total_elements = 0;
    int filtered_count = 0;
    
    // Process each range gate sequentially
    for (int gate = 0; gate < num_range_gates; gate++) {
        llist list = range_gate_lists[gate];
        if (!list) continue;
        
        // Reset iterator
        llist_reset_iter(list);
        
        float gate_quality = 0.8f;  // Simplified quality metric
        bool gate_valid = (gate_quality >= quality_threshold);
        
        int lag_count = 0;
        void* item;
        
        // Process all lags in this range gate
        while (llist_get_iter(list, &item) == LLIST_SUCCESS && lag_count < max_lags_per_gate) {
            acf_data_t* data = (acf_data_t*)item;
            
            if (data && data->power > noise_threshold) {
                // Compute ACF
                float acf_value = data->real * data->real + data->imag * data->imag;
                output_acf_results[gate * max_lags_per_gate + lag_count] = acf_value;
                total_elements++;
            } else {
                output_acf_results[gate * max_lags_per_gate + lag_count] = 0.0f;
            }
            
            lag_count++;
            llist_go_next(list);
        }
        
        // Add to filtered list if valid
        if (gate_valid && lag_count >= 3) {
            output_filtered_gates[filtered_count] = gate;
            filtered_count++;
        }
    }
    
    *num_filtered_gates = filtered_count;
    
    // Calculate CPU performance metrics
    gettimeofday(&end_time, NULL);
    float processing_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0f +
                           (end_time.tv_usec - start_time.tv_usec) / 1000.0f;
    
    g_last_cpu_metrics.processing_time_ms = processing_time;
    g_last_cpu_metrics.total_elements_processed = total_elements;
    g_last_cpu_metrics.valid_range_gates = filtered_count;
    g_last_cpu_metrics.throughput_mps = (total_elements / processing_time) / 1000.0f;
    g_last_cpu_metrics.speedup_factor = 1.0f;  // Baseline
    
    printf("✅ CPU processing completed in %.2f ms\n", processing_time);
    cuda_print_performance_metrics(&g_last_cpu_metrics, "CPU");
    
    return 1;  // Success
}

// ============================================================================
// Performance Comparison and Benchmarking
// ============================================================================

/**
 * Run side-by-side CPU vs CUDA benchmark
 */
int bridge_benchmark_cpu_vs_cuda(
    int num_range_gates,
    int elements_per_gate,
    bool print_detailed_results
) {
    printf("\n🏁 Starting CPU vs CUDA Benchmark\n");
    printf("   Range Gates: %d\n", num_range_gates);
    printf("   Elements per Gate: %d\n", elements_per_gate);
    printf("   Total Elements: %d\n", num_range_gates * elements_per_gate);
    printf("=====================================\n");
    
    cuda_performance_metrics_t cuda_metrics = {0};
    cuda_performance_metrics_t cpu_metrics = {0};
    
    // Run CUDA benchmark if available
    if (bridge_is_cuda_enabled()) {
        cudaError_t err = cuda_benchmark_performance(
            num_range_gates, elements_per_gate, &cuda_metrics, &cpu_metrics
        );
        
        if (err == cudaSuccess) {
            if (print_detailed_results) {
                cuda_print_performance_metrics(&cuda_metrics, "CUDA Benchmark");
                cuda_print_performance_metrics(&cpu_metrics, "CPU Benchmark");
            }
            
            // Calculate speedup
            float speedup = cpu_metrics.processing_time_ms / cuda_metrics.processing_time_ms;
            
            printf("\n🎯 BENCHMARK RESULTS:\n");
            printf("   CPU Time:    %.2f ms\n", cpu_metrics.processing_time_ms);
            printf("   CUDA Time:   %.2f ms\n", cuda_metrics.processing_time_ms);
            printf("   Speedup:     %.2fx\n", speedup);
            printf("   Throughput:  %.2f M elements/sec (CUDA) vs %.2f M elements/sec (CPU)\n",
                   cuda_metrics.throughput_mps, cpu_metrics.throughput_mps);
            
            if (speedup >= 2.0f) {
                printf("✅ EXCELLENT: CUDA provides significant acceleration (%.2fx)\n", speedup);
            } else if (speedup >= 1.2f) {
                printf("✅ GOOD: CUDA provides moderate acceleration (%.2fx)\n", speedup);
            } else if (speedup >= 0.8f) {
                printf("⚠️  MARGINAL: CUDA performance similar to CPU (%.2fx)\n", speedup);
            } else {
                printf("❌ POOR: CPU outperforms CUDA (%.2fx)\n", speedup);
            }
            
            return 1;  // Success
        } else {
            printf("❌ CUDA benchmark failed: %s\n", cuda_get_error_string(err));
        }
    } else {
        printf("ℹ️  CUDA not available - CPU-only benchmark\n");
    }
    
    return 0;  // Failed or not available
}

/**
 * Get the last performance metrics
 */
void bridge_get_last_metrics(
    cuda_performance_metrics_t* cuda_metrics,
    cuda_performance_metrics_t* cpu_metrics
) {
    if (cuda_metrics) *cuda_metrics = g_last_cuda_metrics;
    if (cpu_metrics) *cpu_metrics = g_last_cpu_metrics;
}

/**
 * Print current system status
 */
void bridge_print_status(void) {
    printf("\n=== SUPERDARN CPU-CUDA Bridge Status ===\n");
    printf("CUDA Available:    %s\n", g_cuda_available ? "YES" : "NO");
    printf("CUDA Initialized:  %s\n", g_cuda_initialized ? "YES" : "NO");
    printf("CUDA Enabled:      %s\n", g_cuda_enabled ? "YES" : "NO");
    printf("Active Mode:       %s\n", bridge_is_cuda_enabled() ? "CUDA" : "CPU");
    
    if (g_cuda_available) {
        int device_count;
        size_t total_memory;
        int compute_capability;
        
        if (cuda_get_device_info(&device_count, &total_memory, &compute_capability) == cudaSuccess) {
            printf("GPU Devices:       %d\n", device_count);
            printf("GPU Memory:        %.1f GB\n", total_memory / (1024.0 * 1024.0 * 1024.0));
            printf("Compute Cap:       %d.%d\n", compute_capability / 10, compute_capability % 10);
        }
    }
    
    printf("=======================================\n\n");
}
