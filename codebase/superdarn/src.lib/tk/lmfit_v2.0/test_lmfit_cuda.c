/**
 * @file test_lmfit_cuda.c
 * @brief Test and demonstration program for LMFIT v2.0 CUDA acceleration
 * 
 * This program demonstrates the CUDA-accelerated LMFIT functionality and
 * compares performance between CPU and GPU implementations.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "lmfit2toplevel.h"
#include "lmfit_structures.h"

#ifdef USE_CUDA
#include "cuda_lmfit.h"
#endif

// =============================================================================
// TEST DATA GENERATION
// =============================================================================

/**
 * @brief Generate synthetic SuperDARN test data
 */
FITPRMS* generate_test_data(int num_ranges, int num_lags) {
    FITPRMS *fit_prms = (FITPRMS*)malloc(sizeof(FITPRMS));
    if (!fit_prms) return NULL;
    
    // Initialize basic parameters
    memset(fit_prms, 0, sizeof(FITPRMS));
    fit_prms->nrang = num_ranges;
    fit_prms->mplgs = num_lags;
    fit_prms->nave = 20;
    fit_prms->noise = 1000;
    fit_prms->tfreq = 12000;
    fit_prms->smsep = 300;
    
    // Allocate arrays
    fit_prms->pwr0 = (double*)malloc(num_ranges * sizeof(double));
    fit_prms->acfd = (double**)malloc(num_ranges * sizeof(double*));
    fit_prms->lag[0] = (int*)malloc(num_lags * sizeof(int));
    fit_prms->lag[1] = (int*)malloc(num_lags * sizeof(int));
    
    if (!fit_prms->pwr0 || !fit_prms->acfd || !fit_prms->lag[0] || !fit_prms->lag[1]) {
        free(fit_prms);
        return NULL;
    }
    
    // Generate synthetic ACF data
    for (int range = 0; range < num_ranges; range++) {
        fit_prms->acfd[range] = (double*)malloc(num_lags * 2 * sizeof(double));
        if (!fit_prms->acfd[range]) {
            free(fit_prms);
            return NULL;
        }
        
        // Lag 0 power (proportional to range)
        fit_prms->pwr0[range] = 10000.0 * exp(-range * 0.1);
        
        // Generate ACF values with realistic decay
        for (int lag = 0; lag < num_lags; lag++) {
            fit_prms->lag[0][lag] = lag;     // Pulse 1
            fit_prms->lag[1][lag] = lag + 1; // Pulse 2
            
            // Synthetic ACF with exponential decay and phase rotation
            double time = lag * fit_prms->smsep * 1e-6;  // Time in seconds
            double decay = exp(-time * time * 1000.0);   // Spectral width decay
            double phase = 2.0 * M_PI * 300.0 * time;    // Doppler frequency
            double power = fit_prms->pwr0[range] * decay;
            
            // Add some noise
            double noise_re = ((double)rand() / RAND_MAX - 0.5) * 0.1 * power;
            double noise_im = ((double)rand() / RAND_MAX - 0.5) * 0.1 * power;
            
            fit_prms->acfd[range][lag * 2 + 0] = power * cos(phase) + noise_re;  // Real
            fit_prms->acfd[range][lag * 2 + 1] = power * sin(phase) + noise_im;  // Imaginary
        }
    }
    
    return fit_prms;
}

/**
 * @brief Free test data
 */
void free_test_data(FITPRMS *fit_prms) {
    if (!fit_prms) return;
    
    if (fit_prms->acfd) {
        for (int i = 0; i < fit_prms->nrang; i++) {
            free(fit_prms->acfd[i]);
        }
        free(fit_prms->acfd);
    }
    free(fit_prms->pwr0);
    free(fit_prms->lag[0]);
    free(fit_prms->lag[1]);
    free(fit_prms);
}

// =============================================================================
// PERFORMANCE TESTING
// =============================================================================

typedef struct {
    double time_ms;
    bool success;
    char error_message[256];
} test_result_t;

/**
 * @brief Time a function execution
 */
double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/**
 * @brief Test CPU implementation
 */
test_result_t test_cpu_implementation(FITPRMS *fit_prms) {
    test_result_t result;
    memset(&result, 0, sizeof(result));
    
    printf("Testing CPU implementation...\n");
    
    double start_time = get_time_ms();
    
    // Create FitData structure for results
    struct FitData fit_data;
    memset(&fit_data, 0, sizeof(fit_data));
    
    // Call original LMFIT2 function
    int status = LMFIT2(fit_prms, &fit_data);
    
    double end_time = get_time_ms();
    
    result.time_ms = end_time - start_time;
    result.success = (status == 0);
    
    if (!result.success) {
        snprintf(result.error_message, sizeof(result.error_message), 
                "CPU implementation failed with status: %d", status);
    }
    
    printf("CPU implementation: %.2f ms, %s\n", 
           result.time_ms, result.success ? "SUCCESS" : "FAILED");
    
    return result;
}

/**
 * @brief Test CUDA implementation
 */
test_result_t test_cuda_implementation(FITPRMS *fit_prms) {
    test_result_t result;
    memset(&result, 0, sizeof(result));
    
#ifdef USE_CUDA
    printf("Testing CUDA implementation...\n");
    
    // Initialize CUDA system
    cuda_error_t cuda_err = cuda_lmfit_init();
    if (cuda_err != CUDA_SUCCESS) {
        result.success = false;
        snprintf(result.error_message, sizeof(result.error_message),
                "CUDA initialization failed: %s", cuda_get_error_string(cuda_err));
        return result;
    }
    
    double start_time = get_time_ms();
    
    // Create FitData structure for results
    struct FitData fit_data;
    memset(&fit_data, 0, sizeof(fit_data));
    
    // Call CUDA LMFIT function
    int status = LMFIT2_CUDA(fit_prms, &fit_data);
    
    double end_time = get_time_ms();
    
    result.time_ms = end_time - start_time;
    result.success = (status == 0);
    
    if (!result.success) {
        snprintf(result.error_message, sizeof(result.error_message),
                "CUDA implementation failed with status: %d", status);
    }
    
    printf("CUDA implementation: %.2f ms, %s\n", 
           result.time_ms, result.success ? "SUCCESS" : "FAILED");
    
    cuda_lmfit_cleanup();
#else
    printf("CUDA implementation not available (not compiled with CUDA support)\n");
    result.success = false;
    strcpy(result.error_message, "CUDA support not compiled");
#endif
    
    return result;
}

// =============================================================================
// MAIN TEST PROGRAM
// =============================================================================

int main(int argc, char *argv[]) {
    printf("=== LMFIT v2.0 CUDA Performance Test ===\n");
    
    // Parse command line arguments
    int num_ranges = 100;
    int num_lags = 17;
    
    if (argc >= 2) {
        num_ranges = atoi(argv[1]);
        if (num_ranges <= 0) num_ranges = 100;
    }
    if (argc >= 3) {
        num_lags = atoi(argv[2]);
        if (num_lags <= 0) num_lags = 17;
    }
    
    printf("Test configuration:\n");
    printf("  Number of ranges: %d\n", num_ranges);
    printf("  Lags per range: %d\n", num_lags);
    printf("  Total data points: %d\n", num_ranges * num_lags);
    
    // Generate test data
    printf("\nGenerating synthetic test data...\n");
    FITPRMS *fit_prms = generate_test_data(num_ranges, num_lags);
    if (!fit_prms) {
        printf("ERROR: Failed to generate test data\n");
        return 1;
    }
    
    // Test CPU implementation
    printf("\n--- CPU Implementation Test ---\n");
    test_result_t cpu_result = test_cpu_implementation(fit_prms);
    
    // Test CUDA implementation
    printf("\n--- CUDA Implementation Test ---\n");
    test_result_t cuda_result = test_cuda_implementation(fit_prms);
    
    // Performance comparison
    printf("\n=== Performance Summary ===\n");
    printf("CPU Time:   %.2f ms (%s)\n", 
           cpu_result.time_ms, cpu_result.success ? "OK" : "FAILED");
    if (!cpu_result.success) {
        printf("  Error: %s\n", cpu_result.error_message);
    }
    
    printf("CUDA Time:  %.2f ms (%s)\n", 
           cuda_result.time_ms, cuda_result.success ? "OK" : "FAILED");
    if (!cuda_result.success) {
        printf("  Error: %s\n", cuda_result.error_message);
    }
    
    if (cpu_result.success && cuda_result.success) {
        double speedup = cpu_result.time_ms / cuda_result.time_ms;
        printf("Speedup:    %.2fx %s\n", speedup, 
               speedup > 1.0 ? "(CUDA faster)" : "(CPU faster)");
        
        // Performance rating
        if (speedup > 5.0) {
            printf("Performance: EXCELLENT (>5x speedup)\n");
        } else if (speedup > 2.0) {
            printf("Performance: GOOD (2-5x speedup)\n");
        } else if (speedup > 1.0) {
            printf("Performance: MODEST (1-2x speedup)\n");
        } else {
            printf("Performance: NEEDS OPTIMIZATION (CPU faster)\n");
        }
    }
    
    // Cleanup
    free_test_data(fit_prms);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}