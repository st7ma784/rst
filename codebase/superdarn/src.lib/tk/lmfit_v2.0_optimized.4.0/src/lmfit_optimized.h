/*
 LMFIT2 Optimized Headers and Performance Monitoring

 Copyright (c) 2016 University of Saskatchewan
 
 OPTIMIZATION FEATURES:
 - OpenMP parallelization support
 - SIMD vectorization with AVX2
 - Memory alignment and cache optimization
 - Performance monitoring and profiling
 - Enhanced error handling

 This file is part of the Radar Software Toolkit (RST).

 RST is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef LMFIT_OPTIMIZED_H
#define LMFIT_OPTIMIZED_H

#include <omp.h>
#include <immintrin.h>
#include "lmfit_fitting.h"
#include "lmfit_determinations.h"
#include "lmfit_preprocessing.h"
#include "lmfit_leastsquares.h"

// Optimization configuration
#define LMFIT_USE_OPENMP 1
#define LMFIT_USE_SIMD 1
#define LMFIT_CACHE_LINE_SIZE 64
#define LMFIT_SIMD_ALIGNMENT 32

// Performance monitoring macros
#ifdef LMFIT_PERFORMANCE_LOGGING
#define LMFIT_PERF_START(timer) double timer = omp_get_wtime()
#define LMFIT_PERF_END(timer, label) printf("%s: %.6f seconds\n", label, omp_get_wtime() - timer)
#else
#define LMFIT_PERF_START(timer)
#define LMFIT_PERF_END(timer, label)
#endif

// Memory alignment macros
#define LMFIT_ALIGN(x) __attribute__((aligned(x)))
#define LMFIT_CACHE_ALIGN LMFIT_ALIGN(LMFIT_CACHE_LINE_SIZE)

// SIMD utility macros
#define LMFIT_SIMD_WIDTH 4  // AVX2 processes 4 doubles
#define LMFIT_IS_ALIGNED(ptr) (((uintptr_t)(ptr) & (LMFIT_SIMD_ALIGNMENT - 1)) == 0)

// Performance statistics structure
typedef struct {
    double total_fitting_time;
    double total_determination_time;
    double total_preprocessing_time;
    long long total_ranges_processed;
    long long total_acf_fits;
    long long total_xcf_fits;
    int num_threads_used;
} lmfit_performance_stats_t;

// Optimized function declarations

// Fitting functions
void do_LMFIT_optimized(llist_node range, PHASETYPE *phasetype, FITPRMS *fitted_prms);
void ACF_Fit_optimized(llist ranges, FITPRMS *fitted_prms);
void XCF_Fit_optimized(llist ranges, FITPRMS *fitted_prms);
void enhanced_batch_fit(llist ranges, FITPRMS *fitted_prms, PHASETYPE phase_type);

// Determination functions
struct FitRange* new_range_array_optimized(FITPRMS* fit_prms);
void allocate_fit_data_optimized(struct FitData* fit_data, FITPRMS* fit_prms);
void ACF_Determinations_optimized(llist ranges, FITPRMS* fit_prms, 
                                 struct FitData* fit_data, double noise_pwr);
void batch_ACF_Determinations(llist* ranges_array, FITPRMS** fit_prms_array,
                             struct FitData** fit_data_array, double* noise_pwr_array,
                             int batch_size);
void cleanup_fit_data_optimized(struct FitData* fit_data);

// Preprocessing functions (from previously created files)
int lmfit_preprocessing_optimized(double complex *acf, int mplgs, LMFITPRM *lmfitprm);
void calc_initial_guesses_optimized(LMFITPRM *lmfitprm, double complex *acf, int mplgs);

// Least squares functions (from previously created files)
int lmfit_acf_optimized(LMFITPRM *lmfitprm, double complex *acf, double lambda, 
                       double mpinc, int goose, int print_level);

// Performance monitoring functions
void get_lmfit_performance_stats(double* acf_time, double* xcf_time, 
                                long long* acf_count, long long* xcf_count);
void reset_lmfit_performance_stats(void);
void get_determination_performance_metrics(double* alloc_time, double* det_time,
                                         double* proc_time, long long* range_count);
void reset_determination_performance_metrics(void);
void get_comprehensive_performance_stats(lmfit_performance_stats_t* stats);
void print_performance_summary(void);

// Memory management functions
void* lmfit_aligned_malloc(size_t size);
void lmfit_aligned_free(void* ptr);
void lmfit_memory_prefetch(void* ptr, size_t size);

// SIMD utility functions
static inline int lmfit_is_simd_aligned(const void* ptr) {
    return LMFIT_IS_ALIGNED(ptr);
}

static inline void lmfit_simd_memset_zero(void* ptr, size_t size) {
    if (size >= 32 && LMFIT_IS_ALIGNED(ptr)) {
        __m256i zero = _mm256_setzero_si256();
        size_t simd_size = (size / 32) * 32;
        
        for (size_t i = 0; i < simd_size; i += 32) {
            _mm256_store_si256((__m256i*)((char*)ptr + i), zero);
        }
        
        // Handle remaining bytes
        if (size > simd_size) {
            memset((char*)ptr + simd_size, 0, size - simd_size);
        }
    } else {
        memset(ptr, 0, size);
    }
}

// OpenMP configuration functions
void lmfit_set_num_threads(int num_threads);
int lmfit_get_num_threads(void);
int lmfit_get_max_threads(void);

// Cache optimization functions
void lmfit_prefetch_data(void* data, size_t size);
void lmfit_flush_cache_line(void* ptr);

// Error handling and logging
typedef enum {
    LMFIT_SUCCESS = 0,
    LMFIT_ERROR_MEMORY_ALLOCATION = -1,
    LMFIT_ERROR_INVALID_PARAMETERS = -2,
    LMFIT_ERROR_CONVERGENCE_FAILED = -3,
    LMFIT_ERROR_SIMD_ALIGNMENT = -4
} lmfit_error_code_t;

const char* lmfit_get_error_string(lmfit_error_code_t error_code);
void lmfit_log_error(lmfit_error_code_t error_code, const char* function, const char* message);

// Compile-time feature detection
#if defined(__AVX2__)
#define LMFIT_HAS_AVX2 1
#else
#define LMFIT_HAS_AVX2 0
#endif

#if defined(_OPENMP)
#define LMFIT_HAS_OPENMP 1
#else
#define LMFIT_HAS_OPENMP 0
#endif

// Version information
#define LMFIT_OPTIMIZED_VERSION_MAJOR 4
#define LMFIT_OPTIMIZED_VERSION_MINOR 0
#define LMFIT_OPTIMIZED_VERSION_PATCH 0

typedef struct {
    int major;
    int minor;
    int patch;
    const char* build_date;
    const char* compiler;
    int has_openmp;
    int has_avx2;
} lmfit_version_info_t;

void lmfit_get_version_info(lmfit_version_info_t* version_info);
void lmfit_print_version_info(void);

// Initialization and cleanup
void lmfit_optimized_init(void);
void lmfit_optimized_cleanup(void);

#endif // LMFIT_OPTIMIZED_H
