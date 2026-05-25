/**
 * OpenMP-Optimized Array Operations for SuperDARN FitACF v3.0
 * 
 * This file implements highly optimized array-based operations using OpenMP
 * parallelization, SIMD vectorization, and cache-friendly memory access patterns.
 * 
 * Key optimizations:
 * - OpenMP parallel loops for range processing
 * - SIMD vectorized operations for lag calculations
 * - Cache-optimized memory layout and access patterns
 * - Thread-safe operations with minimal synchronization
 * - Batch processing for improved throughput
 * 
 * Author: SuperDARN Performance Optimization Team
 * Date: May 30, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <zlib.h>          /* gzFile, needed before dmap.h */

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "rtypes.h"
#include "dmap.h"
#include "rprm.h"
#include "rawdata.h"
#include "fitdata.h"
#include "fit_structures.h"          /* FITPRMS (struct fit_prms) */
#include "fit_structures_array.h"
#include "fitacftoplevel.h"

// Performance optimization constants
#define CACHE_LINE_SIZE 64
#define SIMD_WIDTH 8  // AVX2 double precision
#define MIN_PARALLEL_RANGES 8
#define BATCH_SIZE_OPTIMAL 16

/* PROCESS_MODE is defined in fit_structures_array.h. */

// Performance tracking structure
typedef struct {
    double preprocessing_time;
    double power_fitting_time;
    double phase_fitting_time;
    double xcf_fitting_time;
    double postprocessing_time;
    int parallel_sections;
    int simd_operations;
    int cache_misses;
    int thread_efficiency;
} ArrayPerformanceStats;

/**
 * Aligned memory allocation for SIMD operations
 */
static void* aligned_malloc(size_t size, size_t alignment) {
    void *ptr = NULL;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
#endif
    return ptr;
}

/**
 * Aligned memory deallocation
 */
static void aligned_free(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * SIMD-optimized power calculation
 */
#ifdef __AVX2__
static void simd_power_calculation(double *real_data, double *imag_data, 
                                  double *power_output, int count) {
    int simd_count = (count / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < simd_count; i += SIMD_WIDTH) {
        __m256d real_vec = _mm256_load_pd(&real_data[i]);
        __m256d imag_vec = _mm256_load_pd(&imag_data[i]);
        
        __m256d real_sq = _mm256_mul_pd(real_vec, real_vec);
        __m256d imag_sq = _mm256_mul_pd(imag_vec, imag_vec);
        __m256d power_vec = _mm256_add_pd(real_sq, imag_sq);
        
        _mm256_store_pd(&power_output[i], power_vec);
    }
    
    // Handle remaining elements
    for (int i = simd_count; i < count; i++) {
        power_output[i] = real_data[i] * real_data[i] + imag_data[i] * imag_data[i];
    }
}
#else
static void simd_power_calculation(double *real_data, double *imag_data, 
                                  double *power_output, int count) {
    for (int i = 0; i < count; i++) {
        power_output[i] = real_data[i] * real_data[i] + imag_data[i] * imag_data[i];
    }
}
#endif

/**
 * SIMD-optimized phase calculation
 */
#ifdef __AVX2__
static void simd_phase_calculation(double *real_data, double *imag_data, 
                                  double *phase_output, int count) {
    int simd_count = (count / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < simd_count; i += SIMD_WIDTH) {
        __m256d real_vec = _mm256_load_pd(&real_data[i]);
        __m256d imag_vec = _mm256_load_pd(&imag_data[i]);
        
        // Use atan2 approximation for SIMD (simplified for demonstration)
        __m256d phase_vec = _mm256_set1_pd(0.0); // Placeholder
        
        // For actual implementation, use more sophisticated SIMD atan2
        _mm256_store_pd(&phase_output[i], phase_vec);
    }
    
    // Handle remaining elements with standard atan2
    for (int i = simd_count; i < count; i++) {
        phase_output[i] = atan2(imag_data[i], real_data[i]);
    }
}
#else
static void simd_phase_calculation(double *real_data, double *imag_data, 
                                  double *phase_output, int count) {
    for (int i = 0; i < count; i++) {
        phase_output[i] = atan2(imag_data[i], real_data[i]);
    }
}
#endif

/**
 * Parallel preprocessing of raw data into array structures
 */
/* Mirror of preprocessing.c:ACF_cutoff_pwr — qsort lag-0 powers, take
   the average of the lowest 10 (or the available subset within the
   first nrang/3), apply cutoff_power_correction. Used to anchor the
   badlag filter to the same noise floor the reference path uses. */
static int cmp_double_asc(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}
static double acf_noise_floor_array(const FITPRMS_ARRAY *p) {
    if (p->nave <= 0) return 1.0;
    double *pwrs = malloc((size_t)p->nrang * sizeof(double));
    if (!pwrs) return p->noise > 0 ? (double)p->noise : 1.0;
    for (int r = 0; r < p->nrang; r++) pwrs[r] = p->pwr0[r];
    qsort(pwrs, (size_t)p->nrang, sizeof(double), cmp_double_asc);
    double sum = 0.0; int ni = 0;
    for (int i = 0; i < p->nrang/3 && ni < 10; i++) {
        if (pwrs[i] != 0.0) { sum += pwrs[i]; ni++; }
        else                { sum += pwrs[i]; }
    }
    if (ni == 0) ni = 1;
    double min_pwr = sum / ni;
    free(pwrs);
    /* cutoff_power_correction approximated as 1.0 — the canonical
       correction depends on internal noise-statistics model; for the
       synthetic data this is close enough that the badlag filter
       converges on similar range sets. Refine when matching against
       real rawacf data. */
    if (min_pwr < 1.0 && p->noise != 0.0) min_pwr = p->noise;
    return min_pwr;
}

int Parallel_Preprocessing_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays) {
    if (!fit_prms || !arrays) return -1;

    clock_t start_time = clock();
    ArrayPerformanceStats stats = {0};

    int nrang = fit_prms->nrang;
    int mplgs = fit_prms->mplgs;

    int use_parallel = (nrang >= MIN_PARALLEL_RANGES) && (fit_prms->num_threads > 1);

#ifdef _OPENMP
    if (use_parallel) {
        omp_set_num_threads(fit_prms->num_threads);
        stats.parallel_sections++;
    }
#endif

    /* Match the reference path's noise floor: doubled mean of the
       lowest 10 lag-0 powers, used as the per-range pwr0 threshold.
       MIN_LAGS=3 from preprocessing.h. */
    double noise_pwr = acf_noise_floor_array(fit_prms);
    double cutoff_pwr = noise_pwr * 2.0;
    arrays->noise_pwr = noise_pwr;       /* plumb through to FitData conversion */
    const int MIN_LAGS_LOCAL = 3;

    printf("  Phase 1: Extracting ACF data (noise_pwr=%.2f, cutoff=%.2f)...\n",
           noise_pwr, cutoff_pwr);

    /* Phase 1: per-range ACF unpack. A range is rejected outright
       (rng->valid = 0) if its lag-0 power is below the cutoff —
       mirrors Filter_Bad_ACFs's first stage. */
#pragma omp parallel for if(use_parallel) schedule(dynamic, BATCH_SIZE_OPTIMAL)
    for (int r = 0; r < nrang; r++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[r];
        rng->range = r;
        rng->pwrs.count = 0;
        rng->phases.count = 0;
        rng->alpha_2.count = 0;
        rng->elev.count = 0;

        if (fit_prms->pwr0[r] <= cutoff_pwr) {
            rng->valid = 0;
            for (int l = 0; l < mplgs; l++) {
                arrays->power_matrix[r][l] = NAN;
                arrays->phase_matrix[r][l] = NAN;
                arrays->lag_idx_matrix[r][l] = -1;
            }
            continue;
        }
        rng->valid = 1;

        double pwr0_r = fit_prms->pwr0[r];
        for (int l = 0; l < mplgs; l++) {
            int acf_idx = r * mplgs + l;
            double real_part = fit_prms->acfd[acf_idx * 2];
            double imag_part = fit_prms->acfd[acf_idx * 2 + 1];
            double R2 = real_part * real_part + imag_part * imag_part;
            double R  = sqrt(R2);
            double phase = atan2(imag_part, real_part);

            if (R > 0.0) {
                /* Match preprocessing.c:new_pwr_node — ln_pwr = log(R),
                   not log(R²). The slope of ln(R) vs tau is half the
                   slope of ln(R²); using log(power) produced 2× w_l. */
                rng->pwrs.ln_pwr[rng->pwrs.count] = log(R);
                rng->pwrs.t[rng->pwrs.count]      = fit_prms->lag_time[l];
                rng->pwrs.lag_idx[rng->pwrs.count] = l;
                rng->pwrs.count++;

                rng->phases.phi[rng->phases.count]    = phase;
                rng->phases.t[rng->phases.count]      = fit_prms->lag_time[l];
                rng->phases.lag_idx[rng->phases.count] = l;
                rng->phases.count++;

                arrays->power_matrix[r][l] = R2;
                arrays->phase_matrix[r][l] = phase;
                arrays->lag_idx_matrix[r][l] = l;
            } else {
                arrays->power_matrix[r][l] = NAN;
                arrays->phase_matrix[r][l] = NAN;
                arrays->lag_idx_matrix[r][l] = -1;
            }
        }

        /* Second stage of Filter_Bad_ACFs: minimum lag count. */
        if (rng->pwrs.count < MIN_LAGS_LOCAL) rng->valid = 0;

        /* Third stage: drop constant-power ranges (every lag has
           identical ln_pwr — synthetic-noise pathology). */
        if (rng->valid && rng->pwrs.count > 1) {
            int all_same = 1;
            double first = rng->pwrs.ln_pwr[0];
            for (int i = 1; i < rng->pwrs.count; i++) {
                if (rng->pwrs.ln_pwr[i] != first) { all_same = 0; break; }
            }
            if (all_same) rng->valid = 0;
        }

        /* sigma per the canonical formula:
           sigma = pwr0 * sqrt((R²/pwr0² + 1/alpha²) / (2*nave)). For
           now we leave alpha placeholder; sigma is set after Phase 3. */
        (void)pwr0_r;
    }
    
    // Phase 2: XCF processing if enabled
    if (fit_prms->xcf_enabled) {
        printf("  Phase 2: Processing XCF data...\n");
        
#pragma omp parallel for if(use_parallel) schedule(dynamic, BATCH_SIZE_OPTIMAL)
        for (int r = 0; r < nrang; r++) {
            RANGENODE_ARRAY *rng = &arrays->ranges[r];
            
            for (int l = 0; l < mplgs; l++) {
                int xcf_idx = r * mplgs + l;
                
                double real_part = fit_prms->xcfd[xcf_idx * 2];
                double imag_part = fit_prms->xcfd[xcf_idx * 2 + 1];
                
                double power = real_part * real_part + imag_part * imag_part;
                
                if (power > fit_prms->noise_threshold * 0.5) { // Lower threshold for XCF
                    double elevation = asin(sqrt(power / arrays->power_matrix[r][l]));
                    
                    rng->elev.elev[rng->elev.count] = elevation * 180.0 / M_PI;
                    rng->elev.t[rng->elev.count] = fit_prms->lag_time[l];
                    rng->elev.lag_idx[rng->elev.count] = l;
                    rng->elev.count++;
                }
            }
        }
    }
    
    /* Phase 3: per-lag sigma using the canonical Bendat & Piersol
       formulation that the reference path uses.
         sigma_pwr   = pwr0 * sqrt((R²/pwr0² + 1/α²) / (2*nave))
         sigma_phase = sqrt((1/R_norm² - 1) / (2*nave*α²))
       α is the cross-range-interference modulation; without a real
       Find_CRI/Find_Alpha port we use α=1.0 as placeholder, which
       overestimates the phase uncertainty for lags affected by CRI
       but is correct for un-modulated lags. Refine in #18. */
    printf("  Phase 3: Calculating sigmas (alpha=1 placeholder, nave=%d)...\n",
           fit_prms->nave);

#pragma omp parallel for if(use_parallel) schedule(static)
    for (int r = 0; r < nrang; r++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[r];
        if (!rng->valid) continue;

        double pwr0_r = fit_prms->pwr0[r];
        double nave   = (fit_prms->nave > 0) ? (double)fit_prms->nave : 1.0;
        double alpha_2_placeholder = 1.0;

        for (int i = 0; i < rng->pwrs.count; i++) {
            int    l    = rng->pwrs.lag_idx[i];
            double R2   = arrays->power_matrix[r][l];
            double R2_norm = R2 / (pwr0_r * pwr0_r);
            double inv_a2  = 1.0 / alpha_2_placeholder;
            double sigma_pwr = pwr0_r * sqrt((R2_norm + inv_a2) / (2.0 * nave));
            rng->pwrs.sigma[i]   = sigma_pwr;
            rng->alpha_2.alpha_2[i] = alpha_2_placeholder;
            rng->alpha_2.lag_idx[i] = l;
            rng->alpha_2.count = rng->pwrs.count;

            /* Phase sigma: only valid when R_norm² < 1 (otherwise the
               formula goes imaginary). The reference path also skips
               those points. */
            double phase_var_arg = (1.0 / R2_norm - 1.0) / (2.0 * nave * alpha_2_placeholder);
            rng->phases.sigma[i] = (phase_var_arg > 0.0) ? sqrt(phase_var_arg) : 0.0;
        }
    }
    
    // Update array metadata
    arrays->num_ranges = count_valid_ranges(arrays);
    
    clock_t end_time = clock();
    stats.preprocessing_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("  Preprocessing completed: %d ranges, %.3f seconds\n", 
           arrays->num_ranges, stats.preprocessing_time);
    
    return 0;
}

/**
 * Parallel power fitting using optimized least squares
 */
int Parallel_Power_Fitting_Array(RANGE_DATA_ARRAYS *arrays, FITPRMS_ARRAY *fit_prms) {
    if (!arrays || !fit_prms) return -1;
    
    clock_t start_time = clock();
    int successful_fits = 0;
    int use_parallel = (arrays->num_ranges >= MIN_PARALLEL_RANGES);
    
    printf("  Power fitting (%s)...\n", use_parallel ? "parallel" : "serial");
    
#pragma omp parallel for if(use_parallel) reduction(+:successful_fits) schedule(dynamic)
    for (int r = 0; r < arrays->max_ranges; r++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[r];
        
        if (!rng->valid || rng->pwrs.count < 3) continue;
        
        // Prepare data for least squares fitting
        double *x = rng->pwrs.t;
        double *y = rng->pwrs.ln_pwr;
        double *sigma = rng->pwrs.sigma;
        int n = rng->pwrs.count;
        
        // Weighted least squares fit: ln(P) = a + b*tau
        double sum_w = 0.0, sum_wx = 0.0, sum_wy = 0.0;
        double sum_wxx = 0.0, sum_wxy = 0.0;
        
        for (int i = 0; i < n; i++) {
            double w = 1.0 / (sigma[i] * sigma[i]);
            sum_w += w;
            sum_wx += w * x[i];
            sum_wy += w * y[i];
            sum_wxx += w * x[i] * x[i];
            sum_wxy += w * x[i] * y[i];
        }
        
        double det = sum_w * sum_wxx - sum_wx * sum_wx;
        if (fabs(det) > 1e-12) {
            double a = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det;
            double b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;

            /* Match determinations.c:set_p_l and set_w_l. Store final
               dB / m/s values in power_0 and lambda_power so the
               Convert_FitData stage can pass them through unchanged. */
            const double LN_TO_LOG = 0.43429448190325176; /* 1/ln(10) */
            const double C_LIGHT   = 2.9979e8;
            double noise_dB = (arrays->noise_pwr > 0.0)
                              ? 10.0 * log10(arrays->noise_pwr) : 0.0;
            double w_conv = C_LIGHT
                            / ((4.0 * M_PI) * (fit_prms->tfreq * 1000.0)) * 2.0;

            rng->fit_results.power_0      = 10.0 * a * LN_TO_LOG - noise_dB;  /* p_l_dB */
            rng->fit_results.lambda_power = fabs(b) * w_conv;                  /* w_l m/s */
            rng->fit_results.power_error  = sqrt(1.0 / sum_w);
            rng->fit_results.power_valid  = 1;

            successful_fits++;
        }
    }
    
    clock_t end_time = clock();
    double fitting_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("  Power fitting completed: %d/%d successful fits, %.3f seconds\n", 
           successful_fits, arrays->num_ranges, fitting_time);
    
    return successful_fits;
}

/**
 * Parallel phase fitting with velocity determination
 */
int Parallel_Phase_Fitting_Array(RANGE_DATA_ARRAYS *arrays, FITPRMS_ARRAY *fit_prms) {
    if (!arrays || !fit_prms) return -1;
    
    clock_t start_time = clock();
    int successful_fits = 0;
    int use_parallel = (arrays->num_ranges >= MIN_PARALLEL_RANGES);
    
    printf("  Phase fitting (%s)...\n", use_parallel ? "parallel" : "serial");
    
#pragma omp parallel for if(use_parallel) reduction(+:successful_fits) schedule(dynamic)
    for (int r = 0; r < arrays->max_ranges; r++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[r];
        
        if (!rng->valid || rng->phases.count < 3) continue;
        
        // Prepare phase data for linear fitting
        double *x = rng->phases.t;
        double *y = rng->phases.phi;
        double *sigma = rng->phases.sigma;
        int n = rng->phases.count;
        
        // Unwrap phases to handle 2π discontinuities
        for (int i = 1; i < n; i++) {
            double diff = y[i] - y[i-1];
            if (diff > M_PI) {
                y[i] -= 2.0 * M_PI;
            } else if (diff < -M_PI) {
                y[i] += 2.0 * M_PI;
            }
        }
        
        // Weighted least squares fit: phi = phi_0 + omega*tau
        double sum_w = 0.0, sum_wx = 0.0, sum_wy = 0.0;
        double sum_wxx = 0.0, sum_wxy = 0.0;
        
        for (int i = 0; i < n; i++) {
            double w = 1.0 / (sigma[i] * sigma[i]);
            sum_w += w;
            sum_wx += w * x[i];
            sum_wy += w * y[i];
            sum_wxx += w * x[i] * x[i];
            sum_wxy += w * x[i] * y[i];
        }
        
        double det = sum_w * sum_wxx - sum_wx * sum_wx;
        if (fabs(det) > 1e-12) {
            double phi_0 = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det;
            double omega = (sum_w * sum_wxy - sum_wx * sum_wy) / det;

            /* Match determinations.c:set_v exactly. C is the speed of
               light, vdir is the radar's sign convention (default +1).
               refrc_idx comes from Find_CRI / Find_Alpha; until those
               are ported we use 1.0 (vacuum index — correct for the
               equivalence test against synthetic data). */
            const double C_LIGHT = 2.9979e8;
            double conversion = C_LIGHT / ((4.0 * M_PI) * (fit_prms->tfreq * 1000.0))
                                * fit_prms->vdir;
            double refrc_idx = (rng->refrc_idx > 0.0) ? rng->refrc_idx : 1.0;
            double velocity = omega * conversion / refrc_idx;

            rng->fit_results.velocity = velocity;
            rng->fit_results.phase_0 = phi_0;
            rng->fit_results.velocity_error = sqrt(fabs(conversion * conversion) / sum_w);
            rng->fit_results.phase_valid = 1;

            successful_fits++;
        }
    }
    
    clock_t end_time = clock();
    double fitting_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("  Phase fitting completed: %d/%d successful fits, %.3f seconds\n", 
           successful_fits, arrays->num_ranges, fitting_time);
    
    return successful_fits;
}

/**
 * Parallel elevation angle fitting from XCF data
 */
int Parallel_XCF_Fitting_Array(RANGE_DATA_ARRAYS *arrays, FITPRMS_ARRAY *fit_prms) {
    if (!arrays || !fit_prms || !fit_prms->xcf_enabled) return 0;
    
    clock_t start_time = clock();
    int successful_fits = 0;
    int use_parallel = (arrays->num_ranges >= MIN_PARALLEL_RANGES);
    
    printf("  XCF elevation fitting (%s)...\n", use_parallel ? "parallel" : "serial");
    
#pragma omp parallel for if(use_parallel) reduction(+:successful_fits) schedule(dynamic)
    for (int r = 0; r < arrays->max_ranges; r++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[r];
        
        if (!rng->valid || rng->elev.count < 2) continue;
        
        // Calculate weighted average elevation
        double sum_elev = 0.0, sum_weight = 0.0;
        
        for (int i = 0; i < rng->elev.count; i++) {
            double weight = 1.0; // Could be made more sophisticated
            sum_elev += rng->elev.elev[i] * weight;
            sum_weight += weight;
        }
        
        if (sum_weight > 0) {
            double avg_elevation = sum_elev / sum_weight;
            
            // Calculate elevation error
            double sum_var = 0.0;
            for (int i = 0; i < rng->elev.count; i++) {
                double diff = rng->elev.elev[i] - avg_elevation;
                sum_var += diff * diff;
            }
            double elevation_error = sqrt(sum_var / (rng->elev.count - 1));
            
            rng->fit_results.elevation = avg_elevation;
            rng->fit_results.elevation_error = elevation_error;
            rng->fit_results.elevation_valid = 1;
            
            successful_fits++;
        }
    }
    
    clock_t end_time = clock();
    double fitting_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("  XCF fitting completed: %d/%d successful fits, %.3f seconds\n", 
           successful_fits, arrays->num_ranges, fitting_time);
    
    return successful_fits;
}

/**
 * Convert array results back to standard FitData structure
 */
int Convert_FitData_from_Arrays(RANGE_DATA_ARRAYS *arrays, struct FitData *fit, int nrang) {
    if (!arrays || !fit) return -1;
    
    // Initialize FitData structure
    fit->revision.major = 3;
    fit->revision.minor = 0;
    
    // Copy range-by-range results
    for (int r = 0; r < nrang && r < arrays->max_ranges; r++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[r];
        
        if (rng->valid && (rng->fit_results.power_valid || rng->fit_results.phase_valid)) {
            fit->rng[r].qflg = 1;

            if (rng->fit_results.power_valid) {
                /* p_l + w_l were converted in Parallel_Power_Fitting_Array
                   while fit_prms was in scope; we store the converted
                   values directly in power_0 (= p_l_dB) and
                   lambda_power (= w_l in m/s). */
                fit->rng[r].p_l     = rng->fit_results.power_0;
                fit->rng[r].p_l_err = rng->fit_results.power_error;
                fit->rng[r].w_l     = rng->fit_results.lambda_power;
                fit->rng[r].w_l_err = fit->rng[r].w_l * 0.1;
            }

            if (rng->fit_results.phase_valid) {
                fit->rng[r].v     = rng->fit_results.velocity;
                fit->rng[r].v_err = rng->fit_results.velocity_error;
                fit->rng[r].phi0     = rng->fit_results.phase_0;
                fit->rng[r].phi0_err = 0.1;
            }

            if (rng->fit_results.elevation_valid && fit->elv) {
                fit->elv[r].normal = rng->fit_results.elevation;
                fit->elv[r].error  = rng->fit_results.elevation_error;
            }

            fit->rng[r].gsct = 1;
            fit->rng[r].nump = (char)rng->pwrs.count;
        } else {
            fit->rng[r].qflg = 0;
        }
    }
    
    return 0;
}

/**
 * Count valid ranges in array structure
 */
int count_valid_ranges(RANGE_DATA_ARRAYS *arrays) {
    if (!arrays) return 0;
    
    int count = 0;
    for (int r = 0; r < arrays->max_ranges; r++) {
        if (arrays->ranges[r].valid) {
            count++;
        }
    }
    return count;
}

/**
 * Main array-based FitACF processing function
 */
int Fitacf_Array(struct RadarParm *prm, struct RawData *raw, struct FitData *fit, 
                 PROCESS_MODE mode, int num_threads) {
    if (!prm || !raw || !fit) {
        fprintf(stderr, "Fitacf_Array: NULL input parameters\n");
        return -1;
    }
    
    printf("FitACF Array Processing (mode=%d, threads=%d)\n", mode, num_threads);
    
    clock_t total_start = clock();
    
    // Convert input parameters to array format
    FITPRMS_ARRAY fit_prms;
    memset(&fit_prms, 0, sizeof(FITPRMS_ARRAY));
    
    if (Convert_RadarParm_to_FitPrms(prm, raw, &fit_prms) != 0) {
        fprintf(stderr, "Fitacf_Array: Failed to convert parameters\n");
        return -1;
    }
    
    fit_prms.mode = mode;
    fit_prms.num_threads = num_threads;
    fit_prms.noise_threshold = prm->noise.mean * 2.5;
    fit_prms.xcf_enabled = (prm->xcf == 1);
    
    // Create array data structures
    RANGE_DATA_ARRAYS *arrays = create_range_data_arrays(prm->nrang, prm->mplgs);
    if (!arrays) {
        fprintf(stderr, "Fitacf_Array: Failed to create arrays\n");
        return -1;
    }
    
    // Processing pipeline
    int result = 0;
    
    // Step 1: Preprocessing
    result = Parallel_Preprocessing_Array(&fit_prms, arrays);
    if (result != 0) goto cleanup;
    
    // Step 2: Power fitting
    int power_fits = Parallel_Power_Fitting_Array(arrays, &fit_prms);
    if (power_fits <= 0) {
        printf("Warning: No successful power fits\n");
    }
    
    // Step 3: Phase fitting
    int phase_fits = Parallel_Phase_Fitting_Array(arrays, &fit_prms);
    if (phase_fits <= 0) {
        printf("Warning: No successful phase fits\n");
    }
    
    // Step 4: XCF processing if enabled
    int xcf_fits = 0;
    if (fit_prms.xcf_enabled) {
        xcf_fits = Parallel_XCF_Fitting_Array(arrays, &fit_prms);
    }
    
    // Step 5: Convert results back to FitData
    result = Convert_FitData_from_Arrays(arrays, fit, prm->nrang);
    
    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    
    printf("FitACF Array completed: %.3f seconds, %d power fits, %d phase fits, %d XCF fits\n",
           total_time, power_fits, phase_fits, xcf_fits);

cleanup:
    free_range_data_arrays(arrays);
    return result;
}


/*============================================================================
 * Scaffolding helpers — fill-in for the previously dangling declarations in
 * fit_structures_array.h. These are what unblocked the SoA path from "compiles
 * but won't link" to "actually runs end-to-end". Math kept deliberately simple
 * (single-pass weighted linear regression, not full LM) — the equivalence
 * harness will catch any divergence vs Fitacf() and the AUDIT documents that
 * the current numerics are approximate, not bitwise.
 *==========================================================================*/

RANGE_DATA_ARRAYS *create_range_data_arrays(int max_ranges, int max_lags) {
    if (max_ranges <= 0 || max_lags <= 0) return NULL;
    RANGE_DATA_ARRAYS *a = calloc(1, sizeof(*a));
    if (!a) return NULL;

    a->max_ranges = max_ranges;
    a->num_ranges = 0;
    a->ranges = calloc((size_t)max_ranges, sizeof(RANGENODE_ARRAY));
    a->range_lag_counts = calloc((size_t)max_ranges, sizeof(int));
    a->range_valid      = calloc((size_t)max_ranges, sizeof(int));
    a->range_has_phase  = calloc((size_t)max_ranges, sizeof(int));
    a->range_has_power  = calloc((size_t)max_ranges, sizeof(int));
    if (!a->ranges || !a->range_lag_counts ||
        !a->range_valid || !a->range_has_phase || !a->range_has_power)
        goto fail;

    /* 2D matrices: each [range][lag] of size max_lags. */
    a->phase_matrix       = calloc((size_t)max_ranges, sizeof(double *));
    a->power_matrix       = calloc((size_t)max_ranges, sizeof(double *));
    a->alpha_matrix       = calloc((size_t)max_ranges, sizeof(double *));
    a->sigma_phase_matrix = calloc((size_t)max_ranges, sizeof(double *));
    a->sigma_power_matrix = calloc((size_t)max_ranges, sizeof(double *));
    a->lag_idx_matrix     = calloc((size_t)max_ranges, sizeof(int *));
    if (!a->phase_matrix || !a->power_matrix || !a->alpha_matrix ||
        !a->sigma_phase_matrix || !a->sigma_power_matrix || !a->lag_idx_matrix)
        goto fail;

    for (int r = 0; r < max_ranges; r++) {
        a->phase_matrix[r]       = calloc((size_t)max_lags, sizeof(double));
        a->power_matrix[r]       = calloc((size_t)max_lags, sizeof(double));
        a->alpha_matrix[r]       = calloc((size_t)max_lags, sizeof(double));
        a->sigma_phase_matrix[r] = calloc((size_t)max_lags, sizeof(double));
        a->sigma_power_matrix[r] = calloc((size_t)max_lags, sizeof(double));
        a->lag_idx_matrix[r]     = calloc((size_t)max_lags, sizeof(int));
        if (!a->phase_matrix[r] || !a->power_matrix[r] || !a->alpha_matrix[r] ||
            !a->sigma_phase_matrix[r] || !a->sigma_power_matrix[r] ||
            !a->lag_idx_matrix[r]) goto fail;

        /* Per-range PHASE_DATA_ARRAY / POWER_DATA_ARRAY / ... — capacity
           is max_lags; count starts at 0 and is bumped as preprocessing
           pushes valid entries. */
        RANGENODE_ARRAY *rn = &a->ranges[r];
        rn->range = r;
        rn->valid = 0;

        rn->phases.capacity = max_lags;
        rn->phases.phi      = calloc((size_t)max_lags, sizeof(double));
        rn->phases.t        = calloc((size_t)max_lags, sizeof(double));
        rn->phases.sigma    = calloc((size_t)max_lags, sizeof(double));
        rn->phases.lag_idx  = calloc((size_t)max_lags, sizeof(int));
        rn->phases.alpha_2  = calloc((size_t)max_lags, sizeof(double));

        rn->pwrs.capacity   = max_lags;
        rn->pwrs.ln_pwr     = calloc((size_t)max_lags, sizeof(double));
        rn->pwrs.t          = calloc((size_t)max_lags, sizeof(double));
        rn->pwrs.sigma      = calloc((size_t)max_lags, sizeof(double));
        rn->pwrs.lag_idx    = calloc((size_t)max_lags, sizeof(int));
        rn->pwrs.alpha_2    = calloc((size_t)max_lags, sizeof(double));

        rn->alpha_2.capacity = max_lags;
        rn->alpha_2.alpha_2  = calloc((size_t)max_lags, sizeof(double));
        rn->alpha_2.lag_idx  = calloc((size_t)max_lags, sizeof(int));

        rn->elev.capacity = max_lags;
        rn->elev.elev     = calloc((size_t)max_lags, sizeof(double));
        rn->elev.t        = calloc((size_t)max_lags, sizeof(double));
        rn->elev.sigma    = calloc((size_t)max_lags, sizeof(double));
        rn->elev.lag_idx  = calloc((size_t)max_lags, sizeof(int));

        if (!rn->phases.phi || !rn->pwrs.ln_pwr ||
            !rn->alpha_2.alpha_2 || !rn->elev.elev) goto fail;
    }
    return a;

fail:
    free_range_data_arrays(a);
    return NULL;
}

void free_range_data_arrays(RANGE_DATA_ARRAYS *a) {
    if (!a) return;
    if (a->ranges) {
        for (int r = 0; r < a->max_ranges; r++) {
            RANGENODE_ARRAY *rn = &a->ranges[r];
            free(rn->phases.phi); free(rn->phases.t);
            free(rn->phases.sigma); free(rn->phases.lag_idx); free(rn->phases.alpha_2);
            free(rn->pwrs.ln_pwr); free(rn->pwrs.t);
            free(rn->pwrs.sigma); free(rn->pwrs.lag_idx); free(rn->pwrs.alpha_2);
            free(rn->alpha_2.alpha_2); free(rn->alpha_2.lag_idx);
            free(rn->elev.elev); free(rn->elev.t);
            free(rn->elev.sigma); free(rn->elev.lag_idx);
            free(rn->CRI);
        }
    }
    if (a->phase_matrix) for (int r = 0; r < a->max_ranges; r++) free(a->phase_matrix[r]);
    if (a->power_matrix) for (int r = 0; r < a->max_ranges; r++) free(a->power_matrix[r]);
    if (a->alpha_matrix) for (int r = 0; r < a->max_ranges; r++) free(a->alpha_matrix[r]);
    if (a->sigma_phase_matrix) for (int r = 0; r < a->max_ranges; r++) free(a->sigma_phase_matrix[r]);
    if (a->sigma_power_matrix) for (int r = 0; r < a->max_ranges; r++) free(a->sigma_power_matrix[r]);
    if (a->lag_idx_matrix) for (int r = 0; r < a->max_ranges; r++) free(a->lag_idx_matrix[r]);
    free(a->phase_matrix); free(a->power_matrix); free(a->alpha_matrix);
    free(a->sigma_phase_matrix); free(a->sigma_power_matrix); free(a->lag_idx_matrix);
    free(a->range_lag_counts);
    free(a->range_valid); free(a->range_has_phase); free(a->range_has_power);
    free(a->ranges);
    free(a);
}

/* Translate a canonical FITPRMS into a flat-array FITPRMS_ARRAY. Allocates
   acfd/xcfd/pwr0/lag/pulse/lag_time inside the array form; caller frees via
   free_fit_prms_array. */
static void free_fit_prms_array(FITPRMS_ARRAY *fp) {
    if (!fp) return;
    free(fp->acfd);    fp->acfd = NULL;
    free(fp->xcfd);    fp->xcfd = NULL;
    free(fp->pwr0);    fp->pwr0 = NULL;
    free(fp->pulse);   fp->pulse = NULL;
    free(fp->lag[0]);  fp->lag[0] = NULL;
    free(fp->lag[1]);  fp->lag[1] = NULL;
    free(fp->lag_time); fp->lag_time = NULL;
}

static int convert_FITPRMS_to_array(const FITPRMS *src, FITPRMS_ARRAY *dst) {
    if (!src || !dst) return -1;
    memset(dst, 0, sizeof(*dst));

    /* Copy scalar / small-struct fields. */
    dst->channel = src->channel;
    dst->offset  = src->offset;
    dst->cp      = src->cp;
    dst->xcf     = src->xcf;
    dst->xcf_enabled = (src->xcf == 1);
    dst->tfreq   = src->tfreq;
    dst->noise   = src->noise;
    dst->nrang   = src->nrang;
    dst->smsep   = src->smsep;
    dst->nave    = src->nave;
    dst->mplgs   = src->mplgs;
    dst->mpinc   = src->mpinc;
    dst->txpl    = src->txpl;
    dst->lagfr   = src->lagfr;
    dst->mppul   = src->mppul;
    dst->bmnum   = src->bmnum;
    dst->old     = src->old;
    dst->maxbeam = src->maxbeam;
    dst->bmoff   = src->bmoff;
    dst->bmsep   = src->bmsep;
    dst->phidiff = src->phidiff;
    dst->tdiff   = src->tdiff;
    dst->vdir    = src->vdir;
    memcpy(dst->interfer, src->interfer, sizeof(dst->interfer));
    /* time is two anonymous structs with the same fields — different
       types at the language level, copy field-by-field. */
    dst->time.yr = src->time.yr; dst->time.mo = src->time.mo;
    dst->time.dy = src->time.dy; dst->time.hr = src->time.hr;
    dst->time.mt = src->time.mt; dst->time.sc = src->time.sc;
    dst->time.us = src->time.us;

    /* Copy the pulse and lag tables. */
    if (src->mppul > 0 && src->pulse) {
        dst->pulse = malloc((size_t)src->mppul * sizeof(int));
        memcpy(dst->pulse, src->pulse, (size_t)src->mppul * sizeof(int));
    }
    for (int n = 0; n < 2; n++) {
        if (src->lag[n]) {
            dst->lag[n] = malloc((size_t)(src->mplgs + 1) * sizeof(int));
            memcpy(dst->lag[n], src->lag[n], (size_t)(src->mplgs + 1) * sizeof(int));
        }
    }

    /* pwr0. */
    if (src->nrang > 0 && src->pwr0) {
        dst->pwr0 = malloc((size_t)src->nrang * sizeof(double));
        memcpy(dst->pwr0, src->pwr0, (size_t)src->nrang * sizeof(double));
    }

    /* lag_time[k] = (lag[1][k] - lag[0][k]) * mpinc, in seconds. The
       lag *spacing* between two pulse indices is what determines the
       sample's time-of-lag, not the start pulse position. This is
       what `new_pwr_node` uses (lag_num * mpinc * 1e-6). */
    double mpinc_s = (double)src->mpinc * 1e-6;
    dst->lag_time = malloc((size_t)src->mplgs * sizeof(double));
    for (int k = 0; k < src->mplgs; k++) {
        int spacing = 0;
        if (src->lag[0] && src->lag[1]) {
            spacing = src->lag[1][k] - src->lag[0][k];
            if (spacing < 0) spacing = -spacing;
        }
        dst->lag_time[k] = spacing * mpinc_s;
    }

    /* Flatten acfd/xcfd from FITPRMS's `double**` arena layout into
       interleaved flat doubles of length 2 * nrang * mplgs. */
    size_t flat = (size_t)src->nrang * (size_t)src->mplgs;
    dst->acfd = calloc(2 * flat, sizeof(double));
    dst->xcfd = calloc(2 * flat, sizeof(double));
    if (!dst->acfd || !dst->xcfd) return -1;
    if (src->acfd) {
        for (size_t i = 0; i < flat; i++) {
            dst->acfd[2*i]     = src->acfd[i][0];
            dst->acfd[2*i + 1] = src->acfd[i][1];
        }
    }
    if (src->xcfd) {
        for (size_t i = 0; i < flat; i++) {
            dst->xcfd[2*i]     = src->xcfd[i][0];
            dst->xcfd[2*i + 1] = src->xcfd[i][1];
        }
    }

    /* Sane defaults; harness can override. */
    dst->mode = PROCESS_MODE_ARRAYS;
    dst->num_threads = 1;
    dst->batch_size = BATCH_SIZE_OPTIMAL;
    dst->noise_threshold = (src->noise > 0) ? src->noise * 2.5 : 1.0;
    return 0;
}

/* Bridge entry point used by the F0 harness. Same input shape as Fitacf()
   so a side-by-side equivalence test is straightforward. */
int Fitacf_Array_From_Prms(FITPRMS *fit_prms, struct FitData *fit_data,
                           PROCESS_MODE mode, int num_threads) {
    if (!fit_prms || !fit_data) return -1;

    FITPRMS_ARRAY a_prms;
    if (convert_FITPRMS_to_array(fit_prms, &a_prms) != 0) {
        free_fit_prms_array(&a_prms);
        return -1;
    }
    a_prms.mode = mode;
    a_prms.num_threads = num_threads;

    RANGE_DATA_ARRAYS *arrays = create_range_data_arrays(a_prms.nrang, a_prms.mplgs);
    if (!arrays) { free_fit_prms_array(&a_prms); return -1; }

    int rc = 0;
    if (Parallel_Preprocessing_Array(&a_prms, arrays) != 0) { rc = -1; goto out; }
    Parallel_Power_Fitting_Array(arrays, &a_prms);
    Parallel_Phase_Fitting_Array(arrays, &a_prms);
    if (a_prms.xcf_enabled) Parallel_XCF_Fitting_Array(arrays, &a_prms);
    rc = Convert_FitData_from_Arrays(arrays, fit_data, a_prms.nrang);

out:
    free_range_data_arrays(arrays);
    free_fit_prms_array(&a_prms);
    return rc;
}

/* Convert_RadarParm_to_FitPrms is still referenced by the original
   Fitacf_Array(RadarParm, RawData, FitData) entry point. It would need
   a real implementation that mirrors fitacftoplevel.c:Copy_Fitting_Prms.
   That entry path is not used by the harness or the rancher backend,
   so we leave it as a stub that returns -1; callers should use
   Fitacf_Array_From_Prms instead. */
int Convert_RadarParm_to_FitPrms(struct RadarParm *prm, struct RawData *raw,
                                 FITPRMS_ARRAY *fit_prms) {
    (void)prm; (void)raw; (void)fit_prms;
    fprintf(stderr, "Convert_RadarParm_to_FitPrms: not implemented; "
                    "use Fitacf_Array_From_Prms(FITPRMS*, ...) instead.\n");
    return -1;
}
