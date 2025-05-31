/*
 * Top-level header for SuperDARN FitACF v3.0_optimized2
 * 
 * This header provides the main interface for the highly optimized
 * FitACF processing engine with full OpenMP and CUDA support.
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#ifndef _FITACF_TOPLEVEL_OPTIMIZED_H
#define _FITACF_TOPLEVEL_OPTIMIZED_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "fit_structures_optimized.h"
#include "rtypes.h"
#include "dmap.h"
#include "rprm.h"
#include "radar.h"
#include "rawdata.h"
#include "fitdata.h"
#include "fitblk.h"

#define MAJOR_OPTIMIZED 3
#define MINOR_OPTIMIZED 0
#define PATCH_OPTIMIZED 2
#define BUILD_OPTIMIZED 4

/* Performance configuration structure */
typedef struct fitacf_config_optimized {
    PROCESS_MODE processing_mode;
    int num_threads;
    int cuda_device_id;
    int enable_memory_pool;
    size_t memory_pool_size;
    int enable_simd;
    int enable_vectorization;
    double noise_threshold;
    int batch_processing_size;
    int cache_optimization_level;
    int debug_level;
    int profiling_enabled;
} FITACF_CONFIG_OPTIMIZED;

/* Main processing functions */
int FitacfOptimized(FITPRMS_OPTIMIZED *fit_prms, struct FitData *fit_data, 
                   int elv_version, FITACF_CONFIG_OPTIMIZED *config);

/* Configuration and setup functions */
int InitializeFitacfOptimized(FITACF_CONFIG_OPTIMIZED *config);
void FreeFitacfOptimized(FITPRMS_OPTIMIZED *fit_prms);
int CopyFittingPrmsOptimized(struct RadarSite *radar_site,
                            struct RadarParm *radar_prms, 
                            struct RawData *raw_data,
                            FITPRMS_OPTIMIZED *fit_prms);
int AllocateFitPrmOptimized(struct RadarParm *radar_prms, FITPRMS_OPTIMIZED *fit_prms);

/* Performance monitoring functions */
int StartPerformanceMonitoring(FITACF_DATA_OPTIMIZED *data);
int StopPerformanceMonitoring(FITACF_DATA_OPTIMIZED *data);
void PrintPerformanceReport(FITACF_DATA_OPTIMIZED *data, FILE *output);
double GetProcessingEfficiency(FITACF_DATA_OPTIMIZED *data);

/* Hardware detection and configuration */
int DetectOptimalConfiguration(FITACF_CONFIG_OPTIMIZED *config);
int DetectCudaCapabilities(void);
int DetectOpenMPCapabilities(void);
int DetectSIMDCapabilities(void);

/* Memory optimization functions */
int ConfigureMemoryOptimization(FITACF_CONFIG_OPTIMIZED *config);
int EnableLargePageSupport(void);
int ConfigureCacheOptimization(FITACF_CONFIG_OPTIMIZED *config);

/* Debugging and validation functions */
int ValidateOptimizedResults(FITACF_DATA_OPTIMIZED *data, struct FitData *reference);
int RunSelfTest(FITACF_CONFIG_OPTIMIZED *config);
void DumpProcessingState(FITACF_DATA_OPTIMIZED *data, const char *filename);

/* Default configuration presets */
extern const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_DEFAULT;
extern const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_PERFORMANCE;
extern const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_MEMORY_OPTIMIZED;
extern const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_CUDA_OPTIMIZED;

#endif /* _FITACF_TOPLEVEL_OPTIMIZED_H */
