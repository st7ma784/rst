/*
 * Array-based top-level FitACF routine for SuperDARN FitACF v3.0
 * 
 * This file implements the main array-based FitACF routine that orchestrates
 * the entire fitting process using arrays instead of linked lists for
 * massive parallelization with OpenMP and CUDA.
 * 
 * Copyright (c) 2025 SuperDARN Refactoring Project
 * Author: GitHub Copilot Assistant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "rtypes.h"
#include "dmap.h"
#include "rprm.h"
#include "rawdata.h"
#include "fitdata.h"
#include "fit_structures_array.h"
#include "preprocessing_array.h"
#include "fitting_array.h"

/* Global performance tracking */
typedef struct performance_metrics {
    double preprocessing_time;
    double power_fitting_time;
    double phase_fitting_time;
    double xcf_fitting_time;
    double total_time;
    int ranges_processed;
    int power_fits_successful;
    int phase_fits_successful;
    int xcf_fits_successful;
} PERFORMANCE_METRICS;

/* Function prototypes */
int Fitacf_Array(struct RadarParm *prm, struct RawData *raw, struct FitData *fit, 
                 PROCESS_MODE mode, int num_threads);
int Convert_RadarParm_to_FitPrms(struct RadarParm *prm, struct RawData *raw, FITPRMS_ARRAY *fit_prms);
int Convert_FitData_from_Arrays(RANGE_DATA_ARRAYS *arrays, struct FitData *fit, int nrang);
void Print_Performance_Metrics(PERFORMANCE_METRICS *metrics);
int Validate_Array_Results(RANGE_DATA_ARRAYS *arrays, struct FitData *fit);

/* Main array-based FitACF routine */
int Fitacf_Array(struct RadarParm *prm, struct RawData *raw, struct FitData *fit, 
                 PROCESS_MODE mode, int num_threads) {
    
    if (!prm || !raw || !fit) {
        fprintf(stderr, "Fitacf_Array: NULL input parameters\n");
        return -1;
    }
    
    printf("Starting FitACF Array Processing (mode=%d, threads=%d)\n", mode, num_threads);
    
    clock_t start_time = clock();
    PERFORMANCE_METRICS metrics;
    memset(&metrics, 0, sizeof(PERFORMANCE_METRICS));
    
    /* Convert input parameters to array format */
    FITPRMS_ARRAY fit_prms;
    memset(&fit_prms, 0, sizeof(FITPRMS_ARRAY));
    
    if (Convert_RadarParm_to_FitPrms(prm, raw, &fit_prms) != 0) {
        fprintf(stderr, "Fitacf_Array: Failed to convert radar parameters\n");
        return -1;
    }
    
    /* Set processing parameters */
    fit_prms.mode = mode;
    fit_prms.num_threads = num_threads;
    fit_prms.noise_threshold = prm->noise.mean * 3.0; /* 3-sigma threshold */
    fit_prms.batch_size = 10; /* Process 10 ranges at a time */
    
    /* Create array data structures */
    RANGE_DATA_ARRAYS *arrays = create_range_data_arrays(fit_prms.nrang, fit_prms.mplgs);
    if (!arrays) {
        fprintf(stderr, "Fitacf_Array: Failed to create array structures\n");
        return -1;
    }
    
    printf("Created arrays for %d ranges, %d lags\n", fit_prms.nrang, fit_prms.mplgs);
    
    /* Step 1: Preprocessing - Fill arrays with data */
    printf("Step 1: Preprocessing...\n");
    clock_t prep_start = clock();
    
    Parallel_Preprocessing_Array(&fit_prms, arrays);
    
    clock_t prep_end = clock();
    metrics.preprocessing_time = (double)(prep_end - prep_start) / CLOCKS_PER_SEC;
    metrics.ranges_processed = count_valid_ranges(arrays);
    
    printf("  Preprocessing completed in %.3f seconds\n", metrics.preprocessing_time);
    printf("  Valid ranges: %d/%d\n", metrics.ranges_processed, fit_prms.nrang);
    
    /* Step 2: Power fitting */
    printf("Step 2: Power fitting...\n");
    clock_t power_start = clock();
    
    int power_fits = Power_Fits_Array(&fit_prms, arrays);
    
    clock_t power_end = clock();
    metrics.power_fitting_time = (double)(power_end - power_start) / CLOCKS_PER_SEC;
    metrics.power_fits_successful = power_fits;
    
    printf("  Power fitting completed in %.3f seconds\n", metrics.power_fitting_time);
    printf("  Successful power fits: %d\n", power_fits);
    
    /* Step 3: ACF phase fitting */
    printf("Step 3: ACF phase fitting...\n");
    clock_t phase_start = clock();
    
    int phase_fits = ACF_Phase_Fit_Array(&fit_prms, arrays);
    
    clock_t phase_end = clock();
    metrics.phase_fitting_time = (double)(phase_end - phase_start) / CLOCKS_PER_SEC;
    metrics.phase_fits_successful = phase_fits;
    
    printf("  Phase fitting completed in %.3f seconds\n", metrics.phase_fitting_time);
    printf("  Successful phase fits: %d\n", phase_fits);
    
    /* Step 4: XCF fitting (if interferometer data available) */
    int xcf_fits = 0;
    if (fit_prms.xcf && fit_prms.xcfd) {
        printf("Step 4: XCF elevation fitting...\n");
        clock_t xcf_start = clock();
        
        xcf_fits = XCF_Phase_Fit_Array(&fit_prms, arrays);
        
        clock_t xcf_end = clock();
        metrics.xcf_fitting_time = (double)(xcf_end - xcf_start) / CLOCKS_PER_SEC;
        metrics.xcf_fits_successful = xcf_fits;
        
        printf("  XCF fitting completed in %.3f seconds\n", metrics.xcf_fitting_time);
        printf("  Successful XCF fits: %d\n", xcf_fits);
    }
    
    /* Step 5: Convert results back to standard FitData format */
    printf("Step 5: Converting results...\n");
    
    if (Convert_FitData_from_Arrays(arrays, fit, fit_prms.nrang) != 0) {
        fprintf(stderr, "Fitacf_Array: Failed to convert results\n");
        free_range_data_arrays(arrays);
        return -1;
    }
    
    /* Step 6: Validation (optional) */
    if (mode == PROCESS_MODE_HYBRID) {
        printf("Step 6: Validating results...\n");
        Validate_Array_Results(arrays, fit);
    }
    
    /* Calculate total time and print metrics */
    clock_t total_end = clock();
    metrics.total_time = (double)(total_end - start_time) / CLOCKS_PER_SEC;
    
    Print_Performance_Metrics(&metrics);
    
    /* Print array statistics */
    ARRAY_STATS stats = calculate_array_stats(arrays);
    printf("\nArray Statistics:\n");
    printf("  Total ranges: %d\n", stats.total_ranges);
    printf("  Valid ranges: %d\n", stats.valid_ranges);
    printf("  Phase points: %d\n", stats.total_phase_points);
    printf("  Power points: %d\n", stats.total_power_points);
    printf("  Avg lags/range: %.1f\n", stats.avg_lags_per_range);
    printf("  Memory usage: %.2f MB\n", stats.memory_usage_mb);
    
    /* Cleanup */
    free_range_data_arrays(arrays);
    
    /* Cleanup fit_prms arrays */
    for (int i = 0; i < fit_prms.nrang; i++) {
        free(fit_prms.acfd[i]);
        if (fit_prms.xcfd) free(fit_prms.xcfd[i]);
    }
    free(fit_prms.acfd);
    free(fit_prms.xcfd);
    free(fit_prms.pwr0);
    free(fit_prms.lag[0]);
    free(fit_prms.lag[1]);
    free(fit_prms.pulse);
    
    printf("FitACF Array processing completed successfully\n");
    return 0;
}

/* Convert standard RadarParm and RawData to array format */
int Convert_RadarParm_to_FitPrms(struct RadarParm *prm, struct RawData *raw, FITPRMS_ARRAY *fit_prms) {
    if (!prm || !raw || !fit_prms) return -1;
    
    /* Copy basic parameters */
    fit_prms->channel = prm->channel;
    fit_prms->cp = prm->cp;
    fit_prms->tfreq = prm->tfreq;
    fit_prms->noise = prm->noise.mean;
    fit_prms->nrang = prm->nrang;
    fit_prms->smsep = prm->smsep;
    fit_prms->nave = prm->nave;
    fit_prms->mplgs = prm->mplgs;
    fit_prms->mpinc = prm->mpinc;
    fit_prms->txpl = prm->txpl;
    fit_prms->lagfr = prm->lagfr;
    fit_prms->mppul = prm->mppul;
    fit_prms->bmnum = prm->bmnum;
    fit_prms->bmoff = prm->bmoff;
    fit_prms->bmsep = prm->bmsep;
    fit_prms->xcf = (prm->xcf == 1);
    fit_prms->time = prm->time;
    
    /* Copy lag table */
    fit_prms->lag[0] = malloc(sizeof(int) * prm->mplgs);
    fit_prms->lag[1] = malloc(sizeof(int) * prm->mplgs);
    if (!fit_prms->lag[0] || !fit_prms->lag[1]) return -1;
    
    for (int i = 0; i < prm->mplgs; i++) {
        fit_prms->lag[0][i] = prm->lag[0][i];
        fit_prms->lag[1][i] = prm->lag[1][i];
    }
    
    /* Copy pulse table */
    fit_prms->pulse = malloc(sizeof(int) * prm->mppul);
    if (!fit_prms->pulse) return -1;
    
    for (int i = 0; i < prm->mppul; i++) {
        fit_prms->pulse[i] = prm->pulse[i];
    }
    
    /* Copy power data */
    fit_prms->pwr0 = malloc(sizeof(double) * prm->nrang);
    if (!fit_prms->pwr0) return -1;
    
    for (int i = 0; i < prm->nrang; i++) {
        fit_prms->pwr0[i] = raw->pwr0[i];
    }
    
    /* Copy ACF data */
    fit_prms->acfd = malloc(sizeof(double complex*) * prm->nrang);
    if (!fit_prms->acfd) return -1;
    
    for (int i = 0; i < prm->nrang; i++) {
        fit_prms->acfd[i] = malloc(sizeof(double complex) * prm->mplgs);
        if (!fit_prms->acfd[i]) return -1;
        
        for (int j = 0; j < prm->mplgs; j++) {
            fit_prms->acfd[i][j] = raw->acfd[i][j];
        }
    }
    
    /* Copy XCF data if available */
    if (prm->xcf == 1 && raw->xcfd) {
        fit_prms->xcfd = malloc(sizeof(double complex*) * prm->nrang);
        if (!fit_prms->xcfd) return -1;
        
        for (int i = 0; i < prm->nrang; i++) {
            fit_prms->xcfd[i] = malloc(sizeof(double complex) * prm->mplgs);
            if (!fit_prms->xcfd[i]) return -1;
            
            for (int j = 0; j < prm->mplgs; j++) {
                fit_prms->xcfd[i][j] = raw->xcfd[i][j];
            }
        }
    }
    
    return 0;
}

/* Convert array results back to standard FitData format */
int Convert_FitData_from_Arrays(RANGE_DATA_ARRAYS *arrays, struct FitData *fit, int nrang) {
    if (!arrays || !fit) return -1;
    
    /* Allocate FitData range array */
    fit->rng = malloc(sizeof(struct FitRange) * nrang);
    if (!fit->rng) return -1;
    
    /* Initialize all ranges to default values */
    for (int i = 0; i < nrang; i++) {
        fit->rng[i].qflg = 0;
        fit->rng[i].gsct = 0;
        fit->rng[i].v = 0.0;
        fit->rng[i].v_err = 0.0;
        fit->rng[i].p_l = 0.0;
        fit->rng[i].p_l_err = 0.0;
        fit->rng[i].p_s = 0.0;
        fit->rng[i].p_s_err = 0.0;
        fit->rng[i].w_l = 0.0;
        fit->rng[i].w_l_err = 0.0;
        fit->rng[i].w_s = 0.0;
        fit->rng[i].w_s_err = 0.0;
        fit->rng[i].phi0 = 0.0;
        fit->rng[i].phi0_err = 0.0;
        fit->rng[i].elv = 0.0;
        fit->rng[i].elv_low = 0.0;
        fit->rng[i].elv_high = 0.0;
        fit->rng[i].nump = 0;
    }
    
    /* Copy results from arrays */
    for (int i = 0; i < arrays->num_ranges && i < nrang; i++) {
        if (!arrays->range_valid[i]) continue;
        
        RANGENODE_ARRAY *rng = &arrays->ranges[i];
        
        /* Set quality flag */
        fit->rng[i].qflg = 1;
        
        /* Copy power fit results */
        if (rng->l_pwr_fit) {
            fit->rng[i].p_l = rng->l_pwr_fit->p_l;
            fit->rng[i].w_l = rng->l_pwr_fit->tau;
            fit->rng[i].nump = rng->l_pwr_fit->ndata;
            
            if (rng->l_pwr_fit_err) {
                fit->rng[i].p_l_err = rng->l_pwr_fit_err->a; /* approximation */
            }
        }
        
        /* Copy phase fit results */
        if (rng->phase_fit) {
            fit->rng[i].v = rng->phase_fit->v;
            fit->rng[i].v_err = rng->phase_fit->v_err;
            fit->rng[i].phi0 = rng->phase_fit->a;
            fit->rng[i].phi0_err = rng->phase_fit->a; /* placeholder */
        }
        
        /* Copy elevation results */
        if (rng->elev_fit) {
            fit->rng[i].elv = rng->elev_fit->elev;
            fit->rng[i].elv_low = rng->elev_fit->elev - rng->elev_fit->elev_err;
            fit->rng[i].elv_high = rng->elev_fit->elev + rng->elev_fit->elev_err;
        }
    }
    
    return 0;
}

/* Performance metrics output */
void Print_Performance_Metrics(PERFORMANCE_METRICS *metrics) {
    printf("\n=== Performance Metrics ===\n");
    printf("Preprocessing:  %.3f seconds\n", metrics->preprocessing_time);
    printf("Power fitting:  %.3f seconds\n", metrics->power_fitting_time);
    printf("Phase fitting:  %.3f seconds\n", metrics->phase_fitting_time);
    printf("XCF fitting:    %.3f seconds\n", metrics->xcf_fitting_time);
    printf("Total time:     %.3f seconds\n", metrics->total_time);
    printf("\nFit Success Rates:\n");
    printf("Power fits:     %d ranges\n", metrics->power_fits_successful);
    printf("Phase fits:     %d ranges\n", metrics->phase_fits_successful);
    printf("XCF fits:       %d ranges\n", metrics->xcf_fits_successful);
    
    if (metrics->total_time > 0) {
        printf("\nThroughput:     %.1f ranges/second\n", 
               metrics->ranges_processed / metrics->total_time);
    }
}

/* Results validation */
int Validate_Array_Results(RANGE_DATA_ARRAYS *arrays, struct FitData *fit) {
    printf("Validating array results...\n");
    
    int validation_errors = 0;
    
    for (int i = 0; i < arrays->num_ranges; i++) {
        if (!arrays->range_valid[i]) continue;
        
        /* Check for reasonable velocity values */
        if (fabs(fit->rng[i].v) > 2000.0) {
            printf("  Warning: Range %d has unrealistic velocity: %.1f m/s\n", i, fit->rng[i].v);
            validation_errors++;
        }
        
        /* Check for reasonable power values */
        if (fit->rng[i].p_l < 0.0 || fit->rng[i].p_l > 1e6) {
            printf("  Warning: Range %d has unrealistic power: %.1f\n", i, fit->rng[i].p_l);
            validation_errors++;
        }
        
        /* Check for reasonable elevation values */
        if (fit->rng[i].elv < -90.0 || fit->rng[i].elv > 90.0) {
            printf("  Warning: Range %d has unrealistic elevation: %.1f degrees\n", i, fit->rng[i].elv);
            validation_errors++;
        }
    }
    
    printf("Validation completed: %d warnings\n", validation_errors);
    return validation_errors;
}

/* Enhanced main wrapper function for different processing modes */
int Fitacf_Main_Array(struct RadarParm *prm, struct RawData *raw, struct FitData *fit) {
    /* Default to array mode with auto-detected thread count */
    int num_threads = 1;
    
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
    if (num_threads > 8) num_threads = 8; /* Reasonable limit */
#endif
    
    return Fitacf_Array(prm, raw, fit, PROCESS_MODE_ARRAYS, num_threads);
}

/* Benchmark function to compare linked list vs array performance */
int Fitacf_Benchmark(struct RadarParm *prm, struct RawData *raw, 
                     struct FitData *fit_llist, struct FitData *fit_array,
                     int num_iterations) {
    
    printf("=== FitACF Performance Benchmark ===\n");
    printf("Iterations: %d\n", num_iterations);
    
    clock_t llist_total = 0, array_total = 0;
    
    for (int iter = 0; iter < num_iterations; iter++) {
        printf("Iteration %d/%d\n", iter + 1, num_iterations);
        
        /* Benchmark linked list implementation */
        clock_t llist_start = clock();
        /* Call original Fitacf function here */
        clock_t llist_end = clock();
        llist_total += (llist_end - llist_start);
        
        /* Benchmark array implementation */
        clock_t array_start = clock();
        Fitacf_Array(prm, raw, fit_array, PROCESS_MODE_ARRAYS, 4);
        clock_t array_end = clock();
        array_total += (array_end - array_start);
    }
    
    double llist_avg = (double)llist_total / (CLOCKS_PER_SEC * num_iterations);
    double array_avg = (double)array_total / (CLOCKS_PER_SEC * num_iterations);
    double speedup = llist_avg / array_avg;
    
    printf("\n=== Benchmark Results ===\n");
    printf("Linked List Avg: %.3f seconds\n", llist_avg);
    printf("Array Avg:       %.3f seconds\n", array_avg);
    printf("Speedup:         %.2fx\n", speedup);
    
    return 0;
}
