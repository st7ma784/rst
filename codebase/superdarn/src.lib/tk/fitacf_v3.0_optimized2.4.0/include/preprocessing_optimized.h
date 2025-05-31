/*
 * Optimized preprocessing functions for SuperDARN FitACF v3.0_optimized2
 * 
 * This header defines highly optimized preprocessing algorithms that
 * operate on vectorized data structures for maximum performance.
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#ifndef _PREPROCESSING_OPTIMIZED_H
#define _PREPROCESSING_OPTIMIZED_H

#include "rtypes.h"
#include "fit_structures_optimized.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

/* Minimum requirements for processing */
#define MIN_LAGS_OPTIMIZED 3
#define MAX_ALPHA_POWER 6.0

/* Vectorized preprocessing functions */
int FillOptimizedStructures(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int ProcessRangesParallel(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);

/* Advanced filtering with vectorization */
int FilterTXOverlapOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int FilterLowPowerLagsOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int FilterBadFitsOptimized(FITACF_DATA_OPTIMIZED *data);
int NoiseFilteringOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);

/* Phase processing with SIMD optimization */
int PhaseUnwrapOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int XCFPhaseUnwrapOptimized(FITACF_DATA_OPTIMIZED *data);
int PhaseCorrectionsOptimized(FITACF_DATA_OPTIMIZED *data);

/* Alpha calculations with vectorization */
int FindAlphaOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int CalculateAlphaVectorized(double *power_array, double *alpha_array, int count);

/* CRI detection and handling */
int FindCRIOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int ProcessInterferenceOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);

/* Elevation angle processing */
int CalculateElevationOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int RefractiveIndexOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);

/* Data validation and quality control */
int ValidateDataQualityOptimized(FITACF_DATA_OPTIMIZED *data);
int MarkInvalidRangesOptimized(FITACF_DATA_OPTIMIZED *data, double noise_threshold);
int CheckDataConsistencyOptimized(FITACF_DATA_OPTIMIZED *data);

/* Memory and cache optimization */
int OptimizeDataLayout(FITACF_DATA_OPTIMIZED *data);
int PrefetchDataForProcessing(FITACF_DATA_OPTIMIZED *data, int range_start, int range_end);
int FlushCacheOptimized(void);

/* Parallel processing utilities */
int DetermineOptimalChunkSize(FITACF_DATA_OPTIMIZED *data, int num_threads);
int BalanceWorkload(FITACF_DATA_OPTIMIZED *data, int *range_assignments, int num_threads);
int SynchronizeThreads(void);

/* CUDA preprocessing kernels (if CUDA enabled) */
#ifdef __CUDACC__
__global__ void cuda_fill_structures_kernel(double *acf_data, double *xcf_data,
                                           FITACF_DATA_OPTIMIZED *gpu_data,
                                           int num_ranges, int num_lags);
__global__ void cuda_filter_tx_overlap_kernel(double *phase_matrix, double *power_matrix,
                                             uint8_t *flags, int num_ranges, int max_lags);
__global__ void cuda_phase_unwrap_kernel(double *phase_matrix, int num_ranges, int max_lags);
__global__ void cuda_alpha_calculation_kernel(double *power_matrix, double *alpha_matrix,
                                             int num_ranges, int max_lags);
__global__ void cuda_noise_filter_kernel(double *data_matrix, double *noise_levels,
                                        uint8_t *flags, int num_ranges, int max_lags);
__global__ void cuda_cri_detection_kernel(double *acf_matrix, double *cri_results,
                                         int num_ranges, int max_lags);
#endif

/* OpenMP optimized functions */
#ifdef _OPENMP
int ProcessRangesOpenMP(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int FilterDataOpenMP(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int PhaseProcessingOpenMP(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
#endif

/* SIMD optimized functions */
#ifdef __AVX2__
int VectorizedPhaseUnwrap(double *phase_data, int count);
int VectorizedPowerCalculation(double *acf_data, double *power_data, int count);
int VectorizedAlphaCalculation(double *power_data, double *alpha_data, int count);
#endif

/* Inline utility functions for performance */
static inline double fast_atan2_optimized(double y, double x) {
    /* Fast approximation of atan2 for phase calculations */
    const double abs_y = fabs(y) + 1e-10;
    const double angle = (x >= 0) ? (M_PI_4 * y / x) : (M_PI_2 - M_PI_4 * x / abs_y);
    return (y < 0) ? -angle : angle;
}

static inline double fast_log_optimized(double x) {
    /* Fast logarithm approximation for power calculations */
    if (x <= 0) return -HUGE_VAL;
    
    union { double d; long long i; } u;
    u.d = x;
    return (u.i - 4606921280493453312LL) * 1.539095918623324e-16;
}

static inline int fast_round_optimized(double x) {
    /* Fast rounding for integer conversions */
    return (int)(x + 0.5);
}

/* Memory prefetch hints for cache optimization */
static inline void prefetch_data(const void *addr) {
    #ifdef __builtin_prefetch
    __builtin_prefetch(addr, 0, 3);
    #endif
}

static inline void prefetch_data_write(void *addr) {
    #ifdef __builtin_prefetch
    __builtin_prefetch(addr, 1, 3);
    #endif
}

#endif /* _PREPROCESSING_OPTIMIZED_H */
