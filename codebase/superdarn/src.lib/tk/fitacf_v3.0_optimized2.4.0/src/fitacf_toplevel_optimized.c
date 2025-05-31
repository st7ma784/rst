/*
 * Top-level orchestration for SuperDARN FitACF v3.0_optimized2
 * 
 * This module provides the main interface for the highly optimized
 * FitACF processing engine with full OpenMP and CUDA support.
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "fitacf_toplevel_optimized.h"
#include "fit_structures_optimized.h"
#include "preprocessing_optimized.h"
#include "fitting_optimized.h"
#include "rtypes.h"

/* Version information */
#define FITACF_OPTIMIZED_VERSION "3.0_optimized2.4.0"
#define FITACF_OPTIMIZED_DATE "2025-01-01"

/* Global configuration */
static OPTIMIZATION_CONFIG g_config = {0};
static int g_initialized = 0;

/*
 * Initialize the optimized FitACF engine
 * Sets up processing environment and resource allocation
 */
int FitACFOptimizedInit(OPTIMIZATION_CONFIG *config)
{
    if (g_initialized) {
        FitACFOptimizedCleanup();
    }
    
    /* Set default configuration if none provided */
    if (config) {
        memcpy(&g_config, config, sizeof(OPTIMIZATION_CONFIG));
    } else {
        SetDefaultOptimizationConfig(&g_config);
    }
    
    /* Validate configuration */
    if (ValidateOptimizationConfig(&g_config) != 0) {
        fprintf(stderr, "FitACF Optimized: Invalid configuration\n");
        return -1;
    }
    
    /* Initialize OpenMP if available */
    #ifdef _OPENMP
    if (g_config.use_openmp) {
        omp_set_num_threads(g_config.num_threads);
        omp_set_dynamic(0);
        printf("FitACF Optimized: OpenMP initialized with %d threads\n", 
               g_config.num_threads);
    }
    #endif
    
    /* Initialize CUDA if available */
    #ifdef __CUDACC__
    if (g_config.use_cuda) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        if (device_count > 0) {
            cudaSetDevice(g_config.cuda_device);
            cudaDeviceReset();
            
            /* Print GPU information */
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, g_config.cuda_device);
            printf("FitACF Optimized: CUDA initialized on %s\n", prop.name);
        } else {
            fprintf(stderr, "FitACF Optimized: No CUDA devices found, disabling GPU\n");
            g_config.use_cuda = 0;
        }
    }
    #endif
    
    /* Initialize performance monitoring */
    InitPerformanceMonitoring(&g_config);
    
    printf("FitACF Optimized %s initialized successfully\n", FITACF_OPTIMIZED_VERSION);
    g_initialized = 1;
    
    return 0;
}

/*
 * Main FitACF processing function
 * Orchestrates the entire optimized processing pipeline
 */
int FitACFOptimizedProcess(struct RadarParm *prm, struct RawData *raw,
                           struct FitData *fit)
{
    if (!g_initialized) {
        fprintf(stderr, "FitACF Optimized: Engine not initialized\n");
        return -1;
    }
    
    if (!prm || !raw || !fit) {
        fprintf(stderr, "FitACF Optimized: Invalid input parameters\n");
        return -1;
    }
    
    clock_t total_start = clock();
    
    /* Convert input to optimized structures */
    FITPRMS_OPTIMIZED fit_prms;
    FITACF_DATA_OPTIMIZED data;
    FITDATA_OPTIMIZED fit_out;
    
    /* Initialize structures */
    memset(&fit_prms, 0, sizeof(FITPRMS_OPTIMIZED));
    memset(&data, 0, sizeof(FITACF_DATA_OPTIMIZED));
    memset(&fit_out, 0, sizeof(FITDATA_OPTIMIZED));
    
    /* Convert legacy structures to optimized format */
    if (ConvertLegacyToOptimized(prm, raw, &fit_prms, &data) != 0) {
        fprintf(stderr, "FitACF Optimized: Failed to convert input data\n");
        return -1;
    }
    
    /* Set processing mode based on configuration */
    data.processing_mode = DetermineProcessingMode(&g_config, &data);
    
    /* Initialize performance monitoring for this processing run */
    memset(&data.perf_stats, 0, sizeof(PERFORMANCE_STATS));
    data.perf_stats.start_time = total_start;
    
    printf("FitACF Optimized: Processing %d ranges with mode %s\n", 
           data.nranges, GetProcessingModeString(data.processing_mode));
    
    /* PHASE 1: Preprocessing */
    clock_t preprocess_start = clock();
    
    if (FillOptimizedStructures(&data, &fit_prms) != 0) {
        fprintf(stderr, "FitACF Optimized: Failed to fill data structures\n");
        CleanupOptimizedStructures(&data, &fit_out);
        return -1;
    }
    
    int valid_ranges = ProcessRangesParallel(&data, &fit_prms);
    if (valid_ranges <= 0) {
        printf("FitACF Optimized: No valid ranges found\n");
        CleanupOptimizedStructures(&data, &fit_out);
        return 0;
    }
    
    /* Apply preprocessing filters */
    FilterTXOverlapOptimized(&data, &fit_prms);
    FilterLowPowerLagsOptimized(&data, &fit_prms);
    NoiseFilteringOptimized(&data, &fit_prms);
    
    /* Phase processing */
    PhaseUnwrapOptimized(&data, &fit_prms);
    if (data.xcfd_data) {
        XCFPhaseUnwrapOptimized(&data);
    }
    PhaseCorrectionsOptimized(&data);
    
    /* Advanced analysis */
    FindAlphaOptimized(&data, &fit_prms);
    FindCRIOptimized(&data, &fit_prms);
    ProcessInterferenceOptimized(&data, &fit_prms);
    FilterBadFitsOptimized(&data);
    
    data.perf_stats.preprocessing_time += (double)(clock() - preprocess_start) / CLOCKS_PER_SEC;
    
    /* PHASE 2: Fitting */
    clock_t fitting_start = clock();
    
    int fitted_ranges = FitACFOptimized(&data, &fit_prms, &fit_out);
    if (fitted_ranges < 0) {
        fprintf(stderr, "FitACF Optimized: Fitting failed\n");
        CleanupOptimizedStructures(&data, &fit_out);
        return -1;
    }
    
    data.perf_stats.fitting_time += (double)(clock() - fitting_start) / CLOCKS_PER_SEC;
    
    /* PHASE 3: Post-processing and output conversion */
    clock_t postprocess_start = clock();
    
    if (ConvertOptimizedToLegacy(&fit_out, &data, fit) != 0) {
        fprintf(stderr, "FitACF Optimized: Failed to convert output data\n");
        CleanupOptimizedStructures(&data, &fit_out);
        return -1;
    }
    
    data.perf_stats.postprocessing_time += (double)(clock() - postprocess_start) / CLOCKS_PER_SEC;
    
    /* Update total performance statistics */
    data.perf_stats.total_time = (double)(clock() - total_start) / CLOCKS_PER_SEC;
    data.perf_stats.ranges_processed = data.nranges;
    data.perf_stats.ranges_fitted = fitted_ranges;
    
    /* Print performance summary if enabled */
    if (g_config.enable_profiling) {
        PrintPerformanceSummary(&data.perf_stats, &data);
    }
    
    /* Update global statistics */
    UpdateGlobalStatistics(&data.perf_stats);
    
    /* Cleanup */
    CleanupOptimizedStructures(&data, &fit_out);
    
    printf("FitACF Optimized: Successfully processed %d ranges, fitted %d\n", 
           data.nranges, fitted_ranges);
    
    return fitted_ranges;
}

/*
 * Set default optimization configuration
 */
int SetDefaultOptimizationConfig(OPTIMIZATION_CONFIG *config)
{
    if (!config) return -1;
    
    memset(config, 0, sizeof(OPTIMIZATION_CONFIG));
    
    /* Processing settings */
    config->processing_mode = MODE_AUTO;
    config->use_openmp = 1;
    config->use_cuda = 1;
    config->use_simd = 1;
    config->memory_pool_size = 64 * 1024 * 1024; /* 64 MB */
    
    /* Thread configuration */
    #ifdef _OPENMP
    config->num_threads = omp_get_max_threads();
    #else
    config->num_threads = 1;
    #endif
    
    /* CUDA configuration */
    config->cuda_device = 0;
    config->cuda_streams = 4;
    config->cuda_block_size = 256;
    
    /* Performance settings */
    config->enable_profiling = 1;
    config->enable_validation = 0;
    config->batch_size = 256;
    config->prefetch_size = 1024;
    
    /* Algorithm settings */
    config->fitting_algorithm = FIT_ALGORITHM_STANDARD;
    config->convergence_tolerance = 1e-6;
    config->max_iterations = 100;
    
    /* Output settings */
    config->output_format = OUTPUT_FORMAT_LEGACY;
    config->compression_level = 6;
    
    return 0;
}

/*
 * Validate optimization configuration
 */
int ValidateOptimizationConfig(OPTIMIZATION_CONFIG *config)
{
    if (!config) return -1;
    
    /* Validate thread count */
    if (config->num_threads < 1 || config->num_threads > 256) {
        fprintf(stderr, "Invalid thread count: %d\n", config->num_threads);
        return -1;
    }
    
    /* Validate memory pool size */
    if (config->memory_pool_size < 1024 * 1024) { /* Minimum 1 MB */
        fprintf(stderr, "Memory pool size too small: %zu\n", config->memory_pool_size);
        return -1;
    }
    
    /* Validate CUDA settings */
    if (config->use_cuda) {
        if (config->cuda_device < 0 || config->cuda_device > 16) {
            fprintf(stderr, "Invalid CUDA device: %d\n", config->cuda_device);
            return -1;
        }
        
        if (config->cuda_streams < 1 || config->cuda_streams > 32) {
            fprintf(stderr, "Invalid CUDA streams: %d\n", config->cuda_streams);
            return -1;
        }
    }
    
    /* Validate batch size */
    if (config->batch_size < 1 || config->batch_size > 10000) {
        fprintf(stderr, "Invalid batch size: %d\n", config->batch_size);
        return -1;
    }
    
    return 0;
}

/*
 * Determine optimal processing mode based on data characteristics
 */
PROCESSING_MODE DetermineProcessingMode(OPTIMIZATION_CONFIG *config, FITACF_DATA_OPTIMIZED *data)
{
    if (config->processing_mode != MODE_AUTO) {
        return config->processing_mode;
    }
    
    /* Auto-select based on data size and available resources */
    int total_samples = data->nranges * data->mplgs;
    
    #ifdef __CUDACC__
    if (config->use_cuda && total_samples > 10000) {
        return MODE_CUDA;
    }
    #endif
    
    #ifdef _OPENMP
    if (config->use_openmp && total_samples > 1000) {
        if (config->use_cuda && total_samples > 50000) {
            return MODE_HYBRID;
        }
        return MODE_OPENMP;
    }
    #endif
    
    return MODE_SEQUENTIAL;
}

/*
 * Convert legacy structures to optimized format
 */
int ConvertLegacyToOptimized(struct RadarParm *prm, struct RawData *raw,
                             FITPRMS_OPTIMIZED *fit_prms, FITACF_DATA_OPTIMIZED *data)
{
    if (!prm || !raw || !fit_prms || !data) return -1;
    
    /* Copy basic parameters from RadarParm */
    fit_prms->nrang = prm->nrang;
    fit_prms->mplgs = prm->mplgs;
    fit_prms->lagfr = prm->lagfr;
    fit_prms->smsep = prm->smsep;
    fit_prms->txpl = prm->txpl;
    fit_prms->tfreq = prm->tfreq;
    fit_prms->noise = prm->noise;
    fit_prms->bm = prm->bmnum;
    fit_prms->cp = prm->cp;
    fit_prms->bmaz = prm->bmazm;
    fit_prms->scan = prm->scan;
    fit_prms->xcfd = (prm->xcf == 1) ? 1 : 0;
    
    /* Allocate lag table */
    fit_prms->lag_table = malloc(fit_prms->mplgs * 2 * sizeof(int));
    if (!fit_prms->lag_table) return -1;
    
    for (int i = 0; i < fit_prms->mplgs; i++) {
        fit_prms->lag_table[i][0] = prm->lag[i][0];
        fit_prms->lag_table[i][1] = prm->lag[i][1];
    }
    
    /* Allocate and copy ACF data */
    fit_prms->acfd = malloc(fit_prms->nrang * sizeof(double **));
    if (!fit_prms->acfd) return -1;
    
    for (int i = 0; i < fit_prms->nrang; i++) {
        fit_prms->acfd[i] = malloc(fit_prms->mplgs * sizeof(double *));
        if (!fit_prms->acfd[i]) return -1;
        
        for (int j = 0; j < fit_prms->mplgs; j++) {
            fit_prms->acfd[i][j] = malloc(2 * sizeof(double));
            if (!fit_prms->acfd[i][j]) return -1;
            
            if (raw->acfd && raw->acfd[i] && raw->acfd[i][j]) {
                fit_prms->acfd[i][j][0] = raw->acfd[i][j][0];
                fit_prms->acfd[i][j][1] = raw->acfd[i][j][1];
            } else {
                fit_prms->acfd[i][j][0] = 0.0;
                fit_prms->acfd[i][j][1] = 0.0;
            }
        }
    }
    
    /* Allocate and copy XCF data if available */
    if (fit_prms->xcfd && raw->xcfd) {
        fit_prms->xcfd_data = malloc(fit_prms->nrang * sizeof(double **));
        if (!fit_prms->xcfd_data) return -1;
        
        for (int i = 0; i < fit_prms->nrang; i++) {
            fit_prms->xcfd_data[i] = malloc(fit_prms->mplgs * sizeof(double *));
            if (!fit_prms->xcfd_data[i]) return -1;
            
            for (int j = 0; j < fit_prms->mplgs; j++) {
                fit_prms->xcfd_data[i][j] = malloc(2 * sizeof(double));
                if (!fit_prms->xcfd_data[i][j]) return -1;
                
                if (raw->xcfd && raw->xcfd[i] && raw->xcfd[i][j]) {
                    fit_prms->xcfd_data[i][j][0] = raw->xcfd[i][j][0];
                    fit_prms->xcfd_data[i][j][1] = raw->xcfd[i][j][1];
                } else {
                    fit_prms->xcfd_data[i][j][0] = 0.0;
                    fit_prms->xcfd_data[i][j][1] = 0.0;
                }
            }
        }
    }
    
    /* Copy power and noise data */
    if (raw->pwr0) {
        fit_prms->pwr0 = malloc(fit_prms->nrang * sizeof(double *));
        if (!fit_prms->pwr0) return -1;
        
        for (int i = 0; i < fit_prms->nrang; i++) {
            fit_prms->pwr0[i] = malloc(sizeof(double));
            if (!fit_prms->pwr0[i]) return -1;
            fit_prms->pwr0[i][0] = raw->pwr0[i];
        }
    }
    
    if (raw->noise) {
        fit_prms->noise_data = malloc(fit_prms->nrang * sizeof(double *));
        if (!fit_prms->noise_data) return -1;
        
        for (int i = 0; i < fit_prms->nrang; i++) {
            fit_prms->noise_data[i] = malloc(sizeof(double));
            if (!fit_prms->noise_data[i]) return -1;
            fit_prms->noise_data[i][0] = raw->noise[i];
        }
    }
    
    return 0;
}

/*
 * Convert optimized results back to legacy format
 */
int ConvertOptimizedToLegacy(FITDATA_OPTIMIZED *fit_out, FITACF_DATA_OPTIMIZED *data,
                             struct FitData *fit)
{
    if (!fit_out || !data || !fit) return -1;
    
    /* Allocate legacy arrays */
    fit->rng = malloc(fit_out->nranges * sizeof(struct FitRange));
    if (!fit->rng) return -1;
    
    fit->num = 0;
    
    /* Copy fitted data */
    for (int range = 0; range < fit_out->nranges; range++) {
        if (fit_out->quality_flag[range] > 0) {
            struct FitRange *rng = &fit->rng[fit->num];
            
            rng->v = fit_out->velocity[range];
            rng->p_0 = fit_out->power[range];
            rng->w_l = fit_out->spectral_width[range];
            rng->phi0 = fit_out->phi0[range];
            rng->sdev_v = 10.0; /* Default error */
            rng->sdev_p = 1.0;  /* Default error */
            rng->sdev_w = 10.0; /* Default error */
            rng->sdev_phi = 0.1; /* Default error */
            rng->qflg = fit_out->quality_flag[range];
            rng->gsct = 0; /* Ground scatter flag */
            rng->nump = data->mplgs;
            
            fit->num++;
        }
    }
    
    return 0;
}

/*
 * Print performance summary
 */
void PrintPerformanceSummary(PERFORMANCE_STATS *stats, FITACF_DATA_OPTIMIZED *data)
{
    if (!stats || !data) return;
    
    printf("\n=== FitACF Optimized Performance Summary ===\n");
    printf("Processing Mode: %s\n", GetProcessingModeString(data->processing_mode));
    printf("Total Time: %.3f seconds\n", stats->total_time);
    printf("  Preprocessing: %.3f seconds (%.1f%%)\n", 
           stats->preprocessing_time, 
           (stats->preprocessing_time / stats->total_time) * 100.0);
    printf("  Fitting: %.3f seconds (%.1f%%)\n", 
           stats->fitting_time, 
           (stats->fitting_time / stats->total_time) * 100.0);
    printf("  Postprocessing: %.3f seconds (%.1f%%)\n", 
           stats->postprocessing_time, 
           (stats->postprocessing_time / stats->total_time) * 100.0);
    
    printf("Ranges Processed: %d\n", stats->ranges_processed);
    printf("Ranges Fitted: %d (%.1f%% success rate)\n", 
           stats->ranges_fitted, 
           (double)stats->ranges_fitted / stats->ranges_processed * 100.0);
    printf("Samples Filtered: %d\n", stats->samples_filtered);
    printf("Memory Allocated: %.2f MB\n", stats->memory_allocated / (1024.0 * 1024.0));
    
    if (stats->ranges_processed > 0) {
        printf("Processing Rate: %.1f ranges/second\n", 
               stats->ranges_processed / stats->total_time);
    }
    
    #ifdef _OPENMP
    if (data->processing_mode == MODE_OPENMP || data->processing_mode == MODE_HYBRID) {
        printf("OpenMP Threads: %d\n", omp_get_max_threads());
    }
    #endif
    
    #ifdef __CUDACC__
    if (data->processing_mode == MODE_CUDA || data->processing_mode == MODE_HYBRID) {
        printf("CUDA Acceleration: Enabled\n");
    }
    #endif
    
    printf("===========================================\n\n");
}

/*
 * Get processing mode string for display
 */
const char* GetProcessingModeString(PROCESSING_MODE mode)
{
    switch (mode) {
        case MODE_SEQUENTIAL: return "Sequential";
        case MODE_OPENMP: return "OpenMP";
        case MODE_CUDA: return "CUDA";
        case MODE_HYBRID: return "Hybrid CPU+GPU";
        case MODE_AUTO: return "Auto";
        default: return "Unknown";
    }
}

/*
 * Initialize performance monitoring system
 */
int InitPerformanceMonitoring(OPTIMIZATION_CONFIG *config)
{
    /* Initialize global performance counters */
    static PERFORMANCE_STATS global_stats = {0};
    
    return 0;
}

/*
 * Update global performance statistics
 */
void UpdateGlobalStatistics(PERFORMANCE_STATS *stats)
{
    /* Update global counters for long-term monitoring */
    static PERFORMANCE_STATS global_stats = {0};
    
    global_stats.total_time += stats->total_time;
    global_stats.ranges_processed += stats->ranges_processed;
    global_stats.ranges_fitted += stats->ranges_fitted;
    global_stats.memory_allocated += stats->memory_allocated;
}

/*
 * Cleanup optimized data structures
 */
void CleanupOptimizedStructures(FITACF_DATA_OPTIMIZED *data, FITDATA_OPTIMIZED *fit_out)
{
    if (data) {
        CleanupMemoryPool(&data->memory_pool);
    }
    
    if (fit_out) {
        CleanupMemoryPool(&fit_out->memory_pool);
    }
}

/*
 * Cleanup the optimized FitACF engine
 */
void FitACFOptimizedCleanup(void)
{
    if (!g_initialized) return;
    
    #ifdef __CUDACC__
    if (g_config.use_cuda) {
        cudaDeviceReset();
    }
    #endif
    
    printf("FitACF Optimized: Cleanup completed\n");
    g_initialized = 0;
}
