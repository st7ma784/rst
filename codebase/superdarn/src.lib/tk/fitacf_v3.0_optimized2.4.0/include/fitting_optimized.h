/*
 * Optimized fitting algorithms for SuperDARN FitACF v3.0_optimized2
 * 
 * This header defines high-performance fitting algorithms that operate
 * on vectorized data with full OpenMP and CUDA parallelization.
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#ifndef _FITTING_OPTIMIZED_H
#define _FITTING_OPTIMIZED_H

#include "rtypes.h"
#include "fit_structures_optimized.h"
#include "leastsquares.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

/* Phase types for fitting algorithms */
typedef enum {
    PHASE_TYPE_ACF = 0,
    PHASE_TYPE_XCF = 1
} PHASE_TYPE_OPTIMIZED;

/* Fitting algorithm types */
typedef enum {
    FIT_ALGORITHM_STANDARD = 0,
    FIT_ALGORITHM_ROBUST = 1,
    FIT_ALGORITHM_WEIGHTED = 2,
    FIT_ALGORITHM_ITERATIVE = 3
} FIT_ALGORITHM_TYPE;

/* Batch processing configuration */
typedef struct batch_fit_config {
    int batch_size;
    int num_batches;
    int parallel_batches;
    FIT_ALGORITHM_TYPE algorithm;
    double convergence_tolerance;
    int max_iterations;
} BATCH_FIT_CONFIG;

/* Main fitting functions with full parallelization */
int PowerFitsOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int PhaseFitsOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms, 
                      PHASE_TYPE_OPTIMIZED phase_type);
int ElevationFitsOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);

/* Batch processing for maximum throughput */
int BatchPowerFitting(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms,
                     BATCH_FIT_CONFIG *config);
int BatchPhaseFitting(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms,
                     PHASE_TYPE_OPTIMIZED phase_type, BATCH_FIT_CONFIG *config);

/* Vectorized fitting algorithms */
int VectorizedLinearFit(double *x_data, double *y_data, double *weights,
                       int count, FITDATA *result);
int VectorizedPolynomialFit(double *x_data, double *y_data, double *weights,
                           int count, int degree, FITDATA *result);
int VectorizedRobustFit(double *x_data, double *y_data, double *weights,
                       int count, FITDATA *result);

/* Matrix operations for parallel fitting */
int BatchMatrixMultiply(double **matrices_a, double **matrices_b, double **results,
                       int num_matrices, int rows, int cols);
int BatchMatrixInverse(double **matrices, double **results, int num_matrices, int size);
int BatchLeastSquares(double **design_matrices, double **response_vectors,
                     double **solutions, int num_problems, int rows, int cols);

/* Specialized fitting functions */
int CalculateLogPowerSigmaOptimized(FITACF_DATA_OPTIMIZED *data);
int PhaseFitForRangeOptimized(RANGENODE_OPTIMIZED *range, PHASE_TYPE_OPTIMIZED phase_type,
                             FITPRMS_OPTIMIZED *fit_prms);
int CalculatePhaseSigmaOptimized(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms,
                                PHASE_TYPE_OPTIMIZED phase_type);

/* Quality control and error analysis */
int ValidateFitResults(FITACF_DATA_OPTIMIZED *data);
int CalculateFitConfidence(FITDATA *fit_result, double *data_points, int count);
int DetectOutliers(double *data, int count, double *outlier_flags);
int RobustFitWithOutlierRejection(double *x_data, double *y_data, int count,
                                 FITDATA *result, int max_iterations);

/* Performance-critical inline fitting functions */
static inline double calculate_chi_squared_optimized(double *observed, double *expected,
                                                    double *weights, int count) {
    double chi_sq = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = observed[i] - expected[i];
        chi_sq += weights[i] * diff * diff;
    }
    return chi_sq;
}

static inline double calculate_r_squared_optimized(double *observed, double *fitted,
                                                  double mean_observed, int count) {
    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < count; i++) {
        double diff_res = observed[i] - fitted[i];
        double diff_tot = observed[i] - mean_observed;
        ss_res += diff_res * diff_res;
        ss_tot += diff_tot * diff_tot;
    }
    return (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 0.0;
}

/* OpenMP optimized fitting functions */
#ifdef _OPENMP
int PowerFitsOpenMP(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int PhaseFitsOpenMP(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms,
                   PHASE_TYPE_OPTIMIZED phase_type);
int ParallelMatrixOperations(double **matrices, double **results, int num_matrices,
                            int matrix_size, int num_threads);
#endif

/* CUDA fitting kernels and functions */
#ifdef __CUDACC__
/* CUDA kernel declarations */
__global__ void cuda_power_fit_kernel(double *power_matrix, double *time_matrix,
                                     double *weight_matrix, double *result_matrix,
                                     int num_ranges, int max_lags);
__global__ void cuda_phase_fit_kernel(double *phase_matrix, double *time_matrix,
                                     double *weight_matrix, double *result_matrix,
                                     int num_ranges, int max_lags);
__global__ void cuda_batch_linear_fit_kernel(double *x_matrices, double *y_matrices,
                                            double *weight_matrices, double *results,
                                            int num_fits, int data_points_per_fit);
__global__ void cuda_matrix_multiply_batch_kernel(double *matrices_a, double *matrices_b,
                                                 double *results, int num_matrices,
                                                 int rows, int cols);

/* CUDA host functions */
int InitializeCudaFitting(FITACF_DATA_OPTIMIZED *data);
int PowerFitsCUDA(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms);
int PhaseFitsCUDA(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms,
                 PHASE_TYPE_OPTIMIZED phase_type);
int BatchFittingCUDA(FITACF_DATA_OPTIMIZED *data, FITPRMS_OPTIMIZED *fit_prms,
                    BATCH_FIT_CONFIG *config);
int CleanupCudaFitting(FITACF_DATA_OPTIMIZED *data);

/* CUDA memory management for fitting */
int AllocateCudaFittingMemory(FITACF_DATA_OPTIMIZED *data);
int CopyDataToGPU(FITACF_DATA_OPTIMIZED *data);
int CopyResultsFromGPU(FITACF_DATA_OPTIMIZED *data);
int FreeCudaFittingMemory(FITACF_DATA_OPTIMIZED *data);
#endif

/* SIMD optimized fitting functions */
#ifdef __AVX2__
int VectorizedLeastSquares(double *x_data, double *y_data, double *weights,
                          int count, double *slope, double *intercept);
int VectorizedPolynomialEvaluation(double *coefficients, double *x_values,
                                  double *results, int degree, int count);
int VectorizedMatrixMultiply(double *matrix_a, double *matrix_b, double *result,
                            int rows, int cols);
#endif

/* Advanced fitting algorithms */
int IterativeReweightedLeastSquares(double *x_data, double *y_data, int count,
                                   FITDATA *result, int max_iterations);
int HuberRobustFit(double *x_data, double *y_data, int count, FITDATA *result);
int BiSquareRobustFit(double *x_data, double *y_data, int count, FITDATA *result);
int TheilSenRobustFit(double *x_data, double *y_data, int count, FITDATA *result);

/* Confidence interval calculations */
int CalculateConfidenceIntervals(FITDATA *fit_result, double *x_data, double *y_data,
                                int count, double confidence_level);
int BootstrapConfidenceIntervals(double *x_data, double *y_data, int count,
                                FITDATA *result, int num_bootstrap_samples);

/* Fitting optimization and tuning */
int OptimizeFittingParameters(FITACF_DATA_OPTIMIZED *data, BATCH_FIT_CONFIG *config);
int AdaptiveBatchSizing(FITACF_DATA_OPTIMIZED *data, int *optimal_batch_size);
int TuneFittingAlgorithm(FITACF_DATA_OPTIMIZED *data, FIT_ALGORITHM_TYPE *best_algorithm);

/* Debugging and validation for fitting */
int ValidateFittingResults(FITACF_DATA_OPTIMIZED *data, double tolerance);
int CompareFittingAlgorithms(FITACF_DATA_OPTIMIZED *data, FILE *report_file);
void DumpFittingDebugInfo(FITACF_DATA_OPTIMIZED *data, const char *filename);

#endif /* _FITTING_OPTIMIZED_H */
