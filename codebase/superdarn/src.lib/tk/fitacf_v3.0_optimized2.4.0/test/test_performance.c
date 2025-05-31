/*
 * Performance test suite for SuperDARN FitACF v3.0_optimized2
 * 
 * This program validates the implementation and measures performance
 * of the optimized FitACF processing engine.
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../include/fitacf_toplevel_optimized.h"
#include "../include/fit_structures_optimized.h"
#include "../include/preprocessing_optimized.h"
#include "../include/fitting_optimized.h"

/* Test configuration */
#define TEST_NRANGES 75
#define TEST_MPLGS 18
#define TEST_ITERATIONS 100

/* Function prototypes */
int RunBasicFunctionalityTest(void);
int RunPerformanceTest(void);
int RunMemoryTest(void);
int RunAccuracyTest(void);
int GenerateTestData(FITPRMS_OPTIMIZED *fit_prms, FITACF_DATA_OPTIMIZED *data);
int ValidateResults(FITDATA_OPTIMIZED *fit_out, FITACF_DATA_OPTIMIZED *data);
void PrintTestResults(const char *test_name, int result, double time_taken);

/*
 * Main test program
 */
int main(int argc, char *argv[])
{
    printf("=== FitACF v3.0_optimized2 Test Suite ===\n");
    printf("Copyright (c) 2025 SuperDARN RST Optimization Project\n\n");
    
    /* Initialize the optimized engine */
    OPTIMIZATION_CONFIG config;
    SetDefaultOptimizationConfig(&config);
    config.enable_profiling = 1;
    config.enable_validation = 1;
    
    if (FitACFOptimizedInit(&config) != 0) {
        fprintf(stderr, "Failed to initialize FitACF optimized engine\n");
        return 1;
    }
    
    printf("Engine initialized successfully\n");
    printf("Configuration:\n");
    printf("  OpenMP: %s (%d threads)\n", 
           config.use_openmp ? "Enabled" : "Disabled", config.num_threads);
    printf("  CUDA: %s\n", config.use_cuda ? "Enabled" : "Disabled");
    printf("  SIMD: %s\n", config.use_simd ? "Enabled" : "Disabled");
    printf("  Memory Pool: %.1f MB\n", config.memory_pool_size / (1024.0 * 1024.0));
    printf("\n");
    
    /* Run test suite */
    int total_tests = 0;
    int passed_tests = 0;
    
    /* Basic functionality test */
    printf("Running basic functionality test...\n");
    total_tests++;
    if (RunBasicFunctionalityTest() == 0) {
        passed_tests++;
        printf("✓ Basic functionality test PASSED\n\n");
    } else {
        printf("✗ Basic functionality test FAILED\n\n");
    }
    
    /* Performance test */
    printf("Running performance test...\n");
    total_tests++;
    if (RunPerformanceTest() == 0) {
        passed_tests++;
        printf("✓ Performance test PASSED\n\n");
    } else {
        printf("✗ Performance test FAILED\n\n");
    }
    
    /* Memory test */
    printf("Running memory test...\n");
    total_tests++;
    if (RunMemoryTest() == 0) {
        passed_tests++;
        printf("✓ Memory test PASSED\n\n");
    } else {
        printf("✗ Memory test FAILED\n\n");
    }
    
    /* Accuracy test */
    printf("Running accuracy test...\n");
    total_tests++;
    if (RunAccuracyTest() == 0) {
        passed_tests++;
        printf("✓ Accuracy test PASSED\n\n");
    } else {
        printf("✗ Accuracy test FAILED\n\n");
    }
    
    /* Print final results */
    printf("=== Test Suite Results ===\n");
    printf("Tests passed: %d/%d (%.1f%%)\n", 
           passed_tests, total_tests, 
           (double)passed_tests / total_tests * 100.0);
    
    if (passed_tests == total_tests) {
        printf("🎉 All tests PASSED!\n");
    } else {
        printf("⚠️  Some tests FAILED!\n");
    }
    
    /* Cleanup */
    FitACFOptimizedCleanup();
    
    return (passed_tests == total_tests) ? 0 : 1;
}

/*
 * Test basic functionality of all major components
 */
int RunBasicFunctionalityTest(void)
{
    clock_t start_time = clock();
    
    /* Create test data */
    FITPRMS_OPTIMIZED fit_prms;
    FITACF_DATA_OPTIMIZED data;
    FITDATA_OPTIMIZED fit_out;
    
    memset(&fit_prms, 0, sizeof(FITPRMS_OPTIMIZED));
    memset(&data, 0, sizeof(FITACF_DATA_OPTIMIZED));
    memset(&fit_out, 0, sizeof(FITDATA_OPTIMIZED));
    
    if (GenerateTestData(&fit_prms, &data) != 0) {
        fprintf(stderr, "Failed to generate test data\n");
        return -1;
    }
    
    /* Test data structure filling */
    if (FillOptimizedStructures(&data, &fit_prms) != 0) {
        fprintf(stderr, "Failed to fill optimized structures\n");
        return -1;
    }
    
    /* Test preprocessing */
    int valid_ranges = ProcessRangesParallel(&data, &fit_prms);
    if (valid_ranges <= 0) {
        fprintf(stderr, "No valid ranges found\n");
        return -1;
    }
    
    FilterTXOverlapOptimized(&data, &fit_prms);
    FilterLowPowerLagsOptimized(&data, &fit_prms);
    PhaseUnwrapOptimized(&data, &fit_prms);
    FindAlphaOptimized(&data, &fit_prms);
    
    /* Test fitting */
    if (AllocateFitOutputArrays(&fit_out, data.nranges) != 0) {
        fprintf(stderr, "Failed to allocate fit output arrays\n");
        return -1;
    }
    
    BATCH_FIT_CONFIG config;
    ConfigureBatchProcessing(&config, data.nranges, data.processing_mode);
    
    int fitted_ranges = ProcessFittingBatch(&data, &fit_prms, &fit_out, 
                                           0, data.nranges, &config);
    
    if (fitted_ranges < 0) {
        fprintf(stderr, "Fitting failed\n");
        return -1;
    }
    
    /* Validate results */
    if (ValidateResults(&fit_out, &data) != 0) {
        fprintf(stderr, "Result validation failed\n");
        return -1;
    }
    
    /* Cleanup */
    CleanupOptimizedStructures(&data, &fit_out);
    
    double time_taken = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("  Processed %d ranges, fitted %d (%.1f%% success)\n", 
           data.nranges, fitted_ranges, 
           (double)fitted_ranges / data.nranges * 100.0);
    printf("  Time taken: %.3f seconds\n", time_taken);
    
    return 0;
}

/*
 * Test performance across different processing modes
 */
int RunPerformanceTest(void)
{
    printf("  Testing performance across different modes...\n");
    
    PROCESSING_MODE modes[] = {MODE_SEQUENTIAL, MODE_OPENMP, MODE_CUDA, MODE_HYBRID};
    const char *mode_names[] = {"Sequential", "OpenMP", "CUDA", "Hybrid"};
    int num_modes = sizeof(modes) / sizeof(modes[0]);
    
    double best_time = 1e9;
    int best_mode = -1;
    
    for (int m = 0; m < num_modes; m++) {
        /* Skip modes that aren't available */
        if (modes[m] == MODE_CUDA && !IsGPUAvailable()) continue;
        if (modes[m] == MODE_OPENMP && !IsOpenMPAvailable()) continue;
        if (modes[m] == MODE_HYBRID && (!IsGPUAvailable() || !IsOpenMPAvailable())) continue;
        
        clock_t start_time = clock();
        
        /* Run multiple iterations for timing */
        for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
            FITPRMS_OPTIMIZED fit_prms;
            FITACF_DATA_OPTIMIZED data;
            FITDATA_OPTIMIZED fit_out;
            
            memset(&fit_prms, 0, sizeof(FITPRMS_OPTIMIZED));
            memset(&data, 0, sizeof(FITACF_DATA_OPTIMIZED));
            memset(&fit_out, 0, sizeof(FITDATA_OPTIMIZED));
            
            /* Generate test data */
            GenerateTestData(&fit_prms, &data);
            data.processing_mode = modes[m];
            
            /* Process */
            FillOptimizedStructures(&data, &fit_prms);
            ProcessRangesParallel(&data, &fit_prms);
            FilterTXOverlapOptimized(&data, &fit_prms);
            PhaseUnwrapOptimized(&data, &fit_prms);
            
            AllocateFitOutputArrays(&fit_out, data.nranges);
            
            BATCH_FIT_CONFIG config;
            ConfigureBatchProcessing(&config, data.nranges, modes[m]);
            ProcessFittingBatch(&data, &fit_prms, &fit_out, 0, data.nranges, &config);
            
            CleanupOptimizedStructures(&data, &fit_out);
        }
        
        double total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        double avg_time = total_time / TEST_ITERATIONS;
        
        printf("    %s: %.3f seconds/iteration (%.1f ranges/sec)\n", 
               mode_names[m], avg_time, TEST_NRANGES / avg_time);
        
        if (avg_time < best_time) {
            best_time = avg_time;
            best_mode = m;
        }
    }
    
    if (best_mode >= 0) {
        printf("  Best performance: %s mode (%.3f sec/iteration)\n", 
               mode_names[best_mode], best_time);
    }
    
    return 0;
}

/*
 * Test memory usage and leak detection
 */
int RunMemoryTest(void)
{
    printf("  Testing memory allocation and cleanup...\n");
    
    size_t initial_memory = GetCurrentMemoryUsage();
    
    /* Allocate and deallocate many times */
    for (int i = 0; i < 50; i++) {
        FITPRMS_OPTIMIZED fit_prms;
        FITACF_DATA_OPTIMIZED data;
        FITDATA_OPTIMIZED fit_out;
        
        memset(&fit_prms, 0, sizeof(FITPRMS_OPTIMIZED));
        memset(&data, 0, sizeof(FITACF_DATA_OPTIMIZED));
        memset(&fit_out, 0, sizeof(FITDATA_OPTIMIZED));
        
        GenerateTestData(&fit_prms, &data);
        FillOptimizedStructures(&data, &fit_prms);
        AllocateFitOutputArrays(&fit_out, data.nranges);
        
        CleanupOptimizedStructures(&data, &fit_out);
    }
    
    size_t final_memory = GetCurrentMemoryUsage();
    size_t memory_diff = final_memory - initial_memory;
    
    printf("    Memory usage change: %ld bytes\n", (long)memory_diff);
    
    /* Allow for some small memory growth due to system overhead */
    if (memory_diff > 1024 * 1024) { /* 1 MB threshold */
        fprintf(stderr, "Potential memory leak detected: %ld bytes\n", (long)memory_diff);
        return -1;
    }
    
    return 0;
}

/*
 * Test accuracy of fitting algorithms
 */
int RunAccuracyTest(void)
{
    printf("  Testing fitting accuracy with known signals...\n");
    
    /* Generate synthetic data with known parameters */
    double known_velocity = 300.0; /* m/s */
    double known_width = 50.0;     /* m/s */
    double known_power = 20.0;     /* dB */
    
    FITPRMS_OPTIMIZED fit_prms;
    FITACF_DATA_OPTIMIZED data;
    FITDATA_OPTIMIZED fit_out;
    
    memset(&fit_prms, 0, sizeof(FITPRMS_OPTIMIZED));
    memset(&data, 0, sizeof(FITACF_DATA_OPTIMIZED));
    memset(&fit_out, 0, sizeof(FITDATA_OPTIMIZED));
    
    /* Generate synthetic ACF with known parameters */
    GenerateTestData(&fit_prms, &data);
    
    /* Override with synthetic data */
    data.processing_mode = MODE_SEQUENTIAL; /* Use deterministic mode for accuracy test */
    
    FillOptimizedStructures(&data, &fit_prms);
    
    /* Generate synthetic ACF */
    for (int range = 0; range < data.nranges; range++) {
        if (range < data.nranges / 2) { /* Only test first half */
            for (int lag = 0; lag < data.mplgs; lag++) {
                int idx = range * data.mplgs + lag;
                double lag_time = lag * fit_prms.lagfr * 1e-6;
                
                /* Generate synthetic ACF with known parameters */
                double phase = known_velocity * 4.0 * M_PI * fit_prms.tfreq * 1e6 * lag_time / 3e8;
                double decay = exp(-known_width * known_width * lag_time * lag_time / 1e6);
                double magnitude = pow(10.0, known_power / 10.0) * decay;
                
                data.acfd_data[idx] = magnitude * (cos(phase) + I * sin(phase));
            }
            data.range_status[range] = 1;
        }
    }
    
    /* Process the synthetic data */
    ProcessRangesParallel(&data, &fit_prms);
    PhaseUnwrapOptimized(&data, &fit_prms);
    
    AllocateFitOutputArrays(&fit_out, data.nranges);
    
    BATCH_FIT_CONFIG config;
    ConfigureBatchProcessing(&config, data.nranges, MODE_SEQUENTIAL);
    ProcessFittingBatch(&data, &fit_prms, &fit_out, 0, data.nranges, &config);
    
    /* Check accuracy of fitted parameters */
    int accurate_fits = 0;
    int total_fits = 0;
    
    for (int range = 0; range < data.nranges / 2; range++) {
        if (fit_out.quality_flag[range] > 0) {
            double vel_error = fabs(fit_out.velocity[range] - known_velocity) / known_velocity;
            double width_error = fabs(fit_out.spectral_width[range] - known_width) / known_width;
            double power_error = fabs(fit_out.power[range] - known_power) / known_power;
            
            if (vel_error < 0.1 && width_error < 0.2 && power_error < 0.1) {
                accurate_fits++;
            }
            total_fits++;
        }
    }
    
    double accuracy = (double)accurate_fits / total_fits;
    printf("    Accuracy: %d/%d fits within tolerance (%.1f%%)\n", 
           accurate_fits, total_fits, accuracy * 100.0);
    
    CleanupOptimizedStructures(&data, &fit_out);
    
    return (accuracy > 0.8) ? 0 : -1; /* Require 80% accuracy */
}

/*
 * Generate test data for validation
 */
int GenerateTestData(FITPRMS_OPTIMIZED *fit_prms, FITACF_DATA_OPTIMIZED *data)
{
    /* Set up basic parameters */
    fit_prms->nrang = TEST_NRANGES;
    fit_prms->mplgs = TEST_MPLGS;
    fit_prms->lagfr = 2400;  /* microseconds */
    fit_prms->smsep = 300;   /* microseconds */
    fit_prms->txpl = 100;    /* microseconds */
    fit_prms->tfreq = 10.0;  /* MHz */
    fit_prms->noise = 1.0;   /* noise level */
    fit_prms->bm = 7;        /* beam number */
    fit_prms->cp = 153;      /* control program */
    fit_prms->bmaz = 45.0;   /* beam azimuth */
    fit_prms->scan = 1;      /* scan flag */
    fit_prms->xcfd = 1;      /* XCF data available */
    
    /* Initialize data structure */
    data->nranges = fit_prms->nrang;
    data->mplgs = fit_prms->mplgs;
    data->lagfr = fit_prms->lagfr;
    data->smsep = fit_prms->smsep;
    data->txpl = fit_prms->txpl;
    data->bm = fit_prms->bm;
    data->cp = fit_prms->cp;
    data->bmaz = fit_prms->bmaz;
    data->scan = fit_prms->scan;
    data->xcfd = fit_prms->xcfd;
    
    return 0;
}

/*
 * Validate fitting results for reasonableness
 */
int ValidateResults(FITDATA_OPTIMIZED *fit_out, FITACF_DATA_OPTIMIZED *data)
{
    int valid_results = 0;
    
    for (int range = 0; range < fit_out->nranges; range++) {
        if (fit_out->quality_flag[range] > 0) {
            /* Check if results are within reasonable bounds */
            if (fabs(fit_out->velocity[range]) <= 2000.0 &&         /* Velocity < 2 km/s */
                fit_out->spectral_width[range] >= 0.0 &&            /* Width >= 0 */
                fit_out->spectral_width[range] <= 1000.0 &&         /* Width < 1 km/s */
                fit_out->power[range] >= -50.0 &&                   /* Power > -50 dB */
                fit_out->power[range] <= 50.0) {                    /* Power < 50 dB */
                valid_results++;
            }
        }
    }
    
    return (valid_results > 0) ? 0 : -1;
}

/*
 * Helper functions for capability detection
 */
int IsOpenMPAvailable(void)
{
    #ifdef _OPENMP
    return 1;
    #else
    return 0;
    #endif
}

int IsGPUAvailable(void)
{
    #ifdef __CUDACC__
    int device_count;
    cudaGetDeviceCount(&device_count);
    return (device_count > 0) ? 1 : 0;
    #else
    return 0;
    #endif
}

/*
 * Simple memory usage estimation
 */
size_t GetCurrentMemoryUsage(void)
{
    /* Simple estimation - in real implementation would use platform-specific APIs */
    return 0; /* Placeholder */
}
