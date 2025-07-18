#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cuda_llist.h"

/**
 * SUPERDARN CUDA FitACF Optimizer
 * 
 * This file provides high-level optimization functions that replace
 * the most computationally intensive loops in the SUPERDARN FitACF
 * processing pipeline with advanced CUDA kernels.
 */

// Forward declarations for CUDA kernel launches
extern cudaError_t launch_parallel_acf_copy(float*, float*, void*, int, int, bool);
extern cudaError_t launch_parallel_xcf_copy(float*, float*, void*, int, int, bool);
extern cudaError_t launch_power_phase_computation(void*, float*, float*, float*, float*, int, int, float);
extern cudaError_t launch_advanced_statistical_reduction(void*, float*, int, int, float*, float);
extern cudaError_t launch_parallel_lag_processing(void*, float*, float, float*, float*, int, int, int);
extern cudaError_t launch_coalesced_data_transform(float*, void*, void*, int, int, int);

// Performance tracking structure
typedef struct {
    double cpu_time_ms;
    double cuda_time_ms;
    double speedup_factor;
    int elements_processed;
    double throughput_mps;
    char operation_name[64];
} optimization_metrics_t;

// Global optimization statistics
static optimization_metrics_t g_optimization_stats[10];
static int g_stats_count = 0;

// ============================================================================
// High-Level Optimization Functions
// ============================================================================

/**
 * CUDA-optimized replacement for Copy_Fitting_Prms nested loops
 * Targets the most computationally intensive patterns in fitacftoplevel.c
 */
int cuda_optimized_copy_fitting_data(
    float* raw_acfd_real,          // Raw ACF data (real component)
    float* raw_acfd_imag,          // Raw ACF data (imaginary component)
    float* raw_xcfd_real,          // Raw XCF data (real component)
    float* raw_xcfd_imag,          // Raw XCF data (imaginary component)
    void* fit_acfd,                // Output: fitted ACF data
    void* fit_xcfd,                // Output: fitted XCF data
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    bool enable_cuda               // Whether to use CUDA acceleration
) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    int total_elements = nrang * mplgs;
    cudaError_t cuda_err = cudaSuccess;
    
    if (enable_cuda && cuda_is_available()) {
        printf("🚀 CUDA-optimizing data copying for %d×%d elements...\n", nrang, mplgs);
        
        // Allocate device memory
        float *d_raw_acfd_real, *d_raw_acfd_imag;
        float *d_raw_xcfd_real, *d_raw_xcfd_imag;
        void *d_fit_acfd, *d_fit_xcfd;
        
        // ACF data allocation
        cuda_err = cudaMalloc(&d_raw_acfd_real, total_elements * sizeof(float));
        if (cuda_err != cudaSuccess) goto cpu_fallback;
        
        cuda_err = cudaMalloc(&d_raw_acfd_imag, total_elements * sizeof(float));
        if (cuda_err != cudaSuccess) goto cpu_fallback;
        
        cuda_err = cudaMalloc(&d_fit_acfd, total_elements * sizeof(float) * 2); // Complex
        if (cuda_err != cudaSuccess) goto cpu_fallback;
        
        // XCF data allocation
        cuda_err = cudaMalloc(&d_raw_xcfd_real, total_elements * sizeof(float));
        if (cuda_err != cudaSuccess) goto cpu_fallback;
        
        cuda_err = cudaMalloc(&d_raw_xcfd_imag, total_elements * sizeof(float));
        if (cuda_err != cudaSuccess) goto cpu_fallback;
        
        cuda_err = cudaMalloc(&d_fit_xcfd, total_elements * sizeof(float) * 2); // Complex
        if (cuda_err != cudaSuccess) goto cpu_fallback;
        
        // Copy data to device
        if (raw_acfd_real && raw_acfd_imag) {
            cudaMemcpy(d_raw_acfd_real, raw_acfd_real, total_elements * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_raw_acfd_imag, raw_acfd_imag, total_elements * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        if (raw_xcfd_real && raw_xcfd_imag) {
            cudaMemcpy(d_raw_xcfd_real, raw_xcfd_real, total_elements * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_raw_xcfd_imag, raw_xcfd_imag, total_elements * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        // Launch optimized CUDA kernels
        bool acf_zero_fill = (raw_acfd_real == NULL || raw_acfd_imag == NULL);
        bool xcf_zero_fill = (raw_xcfd_real == NULL || raw_xcfd_imag == NULL);
        
        cuda_err = launch_parallel_acf_copy(
            acf_zero_fill ? NULL : d_raw_acfd_real,
            acf_zero_fill ? NULL : d_raw_acfd_imag,
            d_fit_acfd, nrang, mplgs, acf_zero_fill
        );
        if (cuda_err != cudaSuccess) goto cleanup_and_fallback;
        
        cuda_err = launch_parallel_xcf_copy(
            xcf_zero_fill ? NULL : d_raw_xcfd_real,
            xcf_zero_fill ? NULL : d_raw_xcfd_imag,
            d_fit_xcfd, nrang, mplgs, xcf_zero_fill
        );
        if (cuda_err != cudaSuccess) goto cleanup_and_fallback;
        
        // Copy results back to host
        cudaMemcpy(fit_acfd, d_fit_acfd, total_elements * sizeof(float) * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(fit_xcfd, d_fit_xcfd, total_elements * sizeof(float) * 2, cudaMemcpyDeviceToHost);
        
        // Cleanup device memory
        cudaFree(d_raw_acfd_real);
        cudaFree(d_raw_acfd_imag);
        cudaFree(d_raw_xcfd_real);
        cudaFree(d_raw_xcfd_imag);
        cudaFree(d_fit_acfd);
        cudaFree(d_fit_xcfd);
        
        gettimeofday(&end_time, NULL);
        double cuda_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                          (end_time.tv_usec - start_time.tv_usec) / 1000.0;
        
        printf("✅ CUDA optimization completed in %.2f ms\n", cuda_time);
        
        // Record performance metrics
        if (g_stats_count < 10) {
            optimization_metrics_t* stats = &g_optimization_stats[g_stats_count++];
            stats->cuda_time_ms = cuda_time;
            stats->elements_processed = total_elements * 2; // ACF + XCF
            stats->throughput_mps = (stats->elements_processed / cuda_time) / 1000.0;
            strcpy(stats->operation_name, "Data Copying");
        }
        
        return 1; // Success
        
cleanup_and_fallback:
        cudaFree(d_raw_acfd_real);
        cudaFree(d_raw_acfd_imag);
        cudaFree(d_raw_xcfd_real);
        cudaFree(d_raw_xcfd_imag);
        cudaFree(d_fit_acfd);
        cudaFree(d_fit_xcfd);
        
cpu_fallback:
        printf("⚠️  CUDA optimization failed (%s), falling back to CPU\n", 
               cuda_get_error_string(cuda_err));
    }
    
    // CPU fallback implementation
    printf("🖥️  Using CPU implementation for data copying...\n");
    
    // Simulate the original nested loop behavior
    for (int i = 0; i < nrang; i++) {
        for (int j = 0; j < mplgs; j++) {
            int idx = i * mplgs + j;
            
            // ACF data copying
            if (raw_acfd_real && raw_acfd_imag) {
                ((float*)fit_acfd)[idx * 2] = raw_acfd_real[idx];
                ((float*)fit_acfd)[idx * 2 + 1] = raw_acfd_imag[idx];
            } else {
                ((float*)fit_acfd)[idx * 2] = 0.0f;
                ((float*)fit_acfd)[idx * 2 + 1] = 0.0f;
            }
            
            // XCF data copying
            if (raw_xcfd_real && raw_xcfd_imag) {
                ((float*)fit_xcfd)[idx * 2] = raw_xcfd_real[idx];
                ((float*)fit_xcfd)[idx * 2 + 1] = raw_xcfd_imag[idx];
            } else {
                ((float*)fit_xcfd)[idx * 2] = 0.0f;
                ((float*)fit_xcfd)[idx * 2 + 1] = 0.0f;
            }
        }
    }
    
    gettimeofday(&end_time, NULL);
    double cpu_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                     (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    
    printf("✅ CPU processing completed in %.2f ms\n", cpu_time);
    
    return 1; // Success
}

/**
 * CUDA-optimized power and phase computation
 * Replaces sequential power/phase calculations with parallel GPU kernels
 */
int cuda_optimized_power_phase_computation(
    void* acf_data,                // Input: ACF complex data
    float* pwr0,                   // Input: lag-0 power values
    float* output_power,           // Output: computed power values
    float* output_phase,           // Output: computed phase values
    float* output_normalized,      // Output: normalized power values
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    float noise_threshold,         // Noise filtering threshold
    bool enable_cuda               // Whether to use CUDA acceleration
) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    int total_elements = nrang * mplgs;
    
    if (enable_cuda && cuda_is_available()) {
        printf("🚀 CUDA-optimizing power/phase computation for %d elements...\n", total_elements);
        
        // Allocate device memory
        void *d_acf_data;
        float *d_pwr0, *d_output_power, *d_output_phase, *d_output_normalized;
        
        cudaError_t err = cudaMalloc(&d_acf_data, total_elements * sizeof(float) * 2);
        if (err != cudaSuccess) goto cpu_fallback_power;
        
        err = cudaMalloc(&d_pwr0, nrang * sizeof(float));
        if (err != cudaSuccess) goto cpu_fallback_power;
        
        err = cudaMalloc(&d_output_power, total_elements * sizeof(float));
        if (err != cudaSuccess) goto cpu_fallback_power;
        
        err = cudaMalloc(&d_output_phase, total_elements * sizeof(float));
        if (err != cudaSuccess) goto cpu_fallback_power;
        
        err = cudaMalloc(&d_output_normalized, total_elements * sizeof(float));
        if (err != cudaSuccess) goto cpu_fallback_power;
        
        // Copy data to device
        cudaMemcpy(d_acf_data, acf_data, total_elements * sizeof(float) * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pwr0, pwr0, nrang * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch CUDA kernel
        err = launch_power_phase_computation(
            d_acf_data, d_pwr0, d_output_power, d_output_phase, d_output_normalized,
            nrang, mplgs, noise_threshold
        );
        
        if (err == cudaSuccess) {
            // Copy results back
            cudaMemcpy(output_power, d_output_power, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(output_phase, d_output_phase, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(output_normalized, d_output_normalized, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Cleanup
            cudaFree(d_acf_data);
            cudaFree(d_pwr0);
            cudaFree(d_output_power);
            cudaFree(d_output_phase);
            cudaFree(d_output_normalized);
            
            gettimeofday(&end_time, NULL);
            double cuda_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                              (end_time.tv_usec - start_time.tv_usec) / 1000.0;
            
            printf("✅ CUDA power/phase computation completed in %.2f ms\n", cuda_time);
            return 1;
        }
        
cpu_fallback_power:
        printf("⚠️  CUDA power/phase computation failed, using CPU fallback\n");
    }
    
    // CPU fallback
    printf("🖥️  Using CPU for power/phase computation...\n");
    
    for (int i = 0; i < nrang; i++) {
        for (int j = 0; j < mplgs; j++) {
            int idx = i * mplgs + j;
            
            float real = ((float*)acf_data)[idx * 2];
            float imag = ((float*)acf_data)[idx * 2 + 1];
            
            // Compute power
            float power = real * real + imag * imag;
            output_power[idx] = power;
            
            // Compute phase
            output_phase[idx] = atan2f(imag, real);
            
            // Normalize by lag-0 power
            if (pwr0[i] > noise_threshold && pwr0[i] > 0.0f) {
                output_normalized[idx] = power / pwr0[i];
            } else {
                output_normalized[idx] = 0.0f;
            }
        }
    }
    
    gettimeofday(&end_time, NULL);
    double cpu_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                     (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    
    printf("✅ CPU power/phase computation completed in %.2f ms\n", cpu_time);
    return 1;
}

/**
 * CUDA-optimized statistical reduction
 * Replaces sequential statistical computations with parallel reduction
 */
int cuda_optimized_statistical_reduction(
    void* acf_data,                // Input: ACF complex data
    float* pwr0,                   // Input: lag-0 power values
    int nrang,                     // Number of range gates
    int mplgs,                     // Number of lags
    float* statistics,             // Output: [mean, max, total, count]
    float noise_threshold,         // Noise filtering threshold
    bool enable_cuda               // Whether to use CUDA acceleration
) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    int total_elements = nrang * mplgs;
    
    if (enable_cuda && cuda_is_available()) {
        printf("🚀 CUDA-optimizing statistical reduction for %d elements...\n", total_elements);
        
        void *d_acf_data;
        float *d_pwr0, *d_statistics;
        
        cudaError_t err = cudaMalloc(&d_acf_data, total_elements * sizeof(float) * 2);
        if (err != cudaSuccess) goto cpu_fallback_stats;
        
        err = cudaMalloc(&d_pwr0, nrang * sizeof(float));
        if (err != cudaSuccess) goto cpu_fallback_stats;
        
        err = cudaMalloc(&d_statistics, 4 * sizeof(float));
        if (err != cudaSuccess) goto cpu_fallback_stats;
        
        // Copy data to device
        cudaMemcpy(d_acf_data, acf_data, total_elements * sizeof(float) * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pwr0, pwr0, nrang * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch advanced reduction kernel
        err = launch_advanced_statistical_reduction(
            d_acf_data, d_pwr0, nrang, mplgs, d_statistics, noise_threshold
        );
        
        if (err == cudaSuccess) {
            // Copy results back
            cudaMemcpy(statistics, d_statistics, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Compute mean from total and count
            if (statistics[3] > 0) {
                statistics[0] = statistics[0] / statistics[3]; // mean = total / count
            }
            
            // Cleanup
            cudaFree(d_acf_data);
            cudaFree(d_pwr0);
            cudaFree(d_statistics);
            
            gettimeofday(&end_time, NULL);
            double cuda_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                              (end_time.tv_usec - start_time.tv_usec) / 1000.0;
            
            printf("✅ CUDA statistical reduction completed in %.2f ms\n", cuda_time);
            printf("   Mean Power: %.3f, Max Power: %.3f, Valid Count: %.0f\n",
                   statistics[0], statistics[1], statistics[3]);
            return 1;
        }
        
cpu_fallback_stats:
        printf("⚠️  CUDA statistical reduction failed, using CPU fallback\n");
    }
    
    // CPU fallback
    printf("🖥️  Using CPU for statistical reduction...\n");
    
    float total_power = 0.0f;
    float max_power = 0.0f;
    int valid_count = 0;
    
    for (int i = 0; i < nrang; i++) {
        if (pwr0[i] <= noise_threshold) continue;
        
        for (int j = 0; j < mplgs; j++) {
            int idx = i * mplgs + j;
            float real = ((float*)acf_data)[idx * 2];
            float imag = ((float*)acf_data)[idx * 2 + 1];
            float power = real * real + imag * imag;
            
            if (power > 0.0f && !isnan(power)) {
                total_power += power;
                max_power = fmaxf(max_power, power);
                valid_count++;
            }
        }
    }
    
    statistics[0] = valid_count > 0 ? total_power / valid_count : 0.0f; // mean
    statistics[1] = max_power;     // max
    statistics[2] = total_power;   // total
    statistics[3] = valid_count;   // count
    
    gettimeofday(&end_time, NULL);
    double cpu_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                     (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    
    printf("✅ CPU statistical reduction completed in %.2f ms\n", cpu_time);
    return 1;
}

// ============================================================================
// Performance Analysis and Reporting
// ============================================================================

/**
 * Print comprehensive optimization performance report
 */
void print_optimization_performance_report(void) {
    if (g_stats_count == 0) {
        printf("ℹ️  No optimization statistics available\n");
        return;
    }
    
    printf("\n📊 CUDA Optimization Performance Report\n");
    printf("========================================\n");
    
    double total_cuda_time = 0.0;
    double total_cpu_time = 0.0;
    int total_elements = 0;
    
    for (int i = 0; i < g_stats_count; i++) {
        optimization_metrics_t* stats = &g_optimization_stats[i];
        
        printf("Operation: %s\n", stats->operation_name);
        printf("  CUDA Time: %.2f ms\n", stats->cuda_time_ms);
        printf("  CPU Time:  %.2f ms\n", stats->cpu_time_ms);
        printf("  Speedup:   %.2fx\n", stats->speedup_factor);
        printf("  Elements:  %d\n", stats->elements_processed);
        printf("  Throughput: %.2f M elements/sec\n", stats->throughput_mps);
        printf("\n");
        
        total_cuda_time += stats->cuda_time_ms;
        total_cpu_time += stats->cpu_time_ms;
        total_elements += stats->elements_processed;
    }
    
    printf("📈 Overall Performance Summary:\n");
    printf("  Total CUDA Time: %.2f ms\n", total_cuda_time);
    printf("  Total CPU Time:  %.2f ms\n", total_cpu_time);
    printf("  Overall Speedup: %.2fx\n", total_cpu_time / total_cuda_time);
    printf("  Total Elements:  %d\n", total_elements);
    printf("  Average Throughput: %.2f M elements/sec\n", 
           (total_elements / total_cuda_time) / 1000.0);
    printf("========================================\n\n");
}
