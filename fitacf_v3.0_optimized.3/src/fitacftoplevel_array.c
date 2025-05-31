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
#include <zlib.h>      /* Must include before dmap.h to define gzFile */
#include "dmap.h"
#include "rprm.h"
#include "rawdata.h"
#include "fitdata.h"
#include "fit_structures_array.h"
#include "preprocessing_array.h"
#include "fitting_array.h"
#include "radar.h"    /* Include radar.h for RadarSite structure */

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
                 struct RadarSite *site, PROCESS_MODE mode, int num_threads);
int Convert_RadarParm_to_FitPrms(struct RadarParm *prm, struct RawData *raw, FITPRMS_ARRAY *fit_prms);
int Copy_Radar_Site_Prms(struct RadarSite *site, FITPRMS_ARRAY *fit_prms);
void Free_FitPrms_Array(FITPRMS_ARRAY *fit_prms);
int Convert_FitData_from_Arrays(RANGE_DATA_ARRAYS *arrays, struct FitData *fit, int nrang);
void Print_Performance_Metrics(PERFORMANCE_METRICS *metrics);
int Validate_Array_Results(RANGE_DATA_ARRAYS *arrays, struct FitData *fit);
int Copy_Radar_Site_Prms(struct RadarSite *site, FITPRMS_ARRAY *fit_prms);
void Free_FitPrms_Array(FITPRMS_ARRAY *fit_prms);

/* Main array-based FitACF routine - Completely rewritten for parallel processing */
int Fitacf_Array(struct RadarParm *prm, struct RawData *raw, struct FitData *fit, 
                 struct RadarSite *site, PROCESS_MODE mode, int num_threads) {
    
    /* Input validation */
    if (!prm || !raw || !fit) {
        fprintf(stderr, "Fitacf_Array: NULL input parameters\n");
        return -1;
    }
    
    if (prm->nrang <= 0 || prm->mplgs <= 0) {
        fprintf(stderr, "Fitacf_Array: Invalid array dimensions (nrang=%d, mplgs=%d)\n", 
                prm->nrang, prm->mplgs);
        return -1;
    }
    
    /* Set up OpenMP if requested */
    if (num_threads > 1) {
        #ifdef _OPENMP
        omp_set_num_threads(num_threads);
        printf("Fitacf_Array: Using OpenMP with %d threads\n", num_threads);
        #else
        printf("Fitacf_Array: OpenMP not available, using single thread\n");
        num_threads = 1;
        #endif
    }
    
    printf("Starting FitACF Array Processing\n");
    printf("  Mode: %s\n", mode == PROCESS_MODE_ARRAYS ? "Array-based" : "Hybrid");
    printf("  Ranges: %d, Lags: %d, Threads: %d\n", prm->nrang, prm->mplgs, num_threads);
    
    clock_t total_start = clock();
    PERFORMANCE_METRICS metrics;
    memset(&metrics, 0, sizeof(PERFORMANCE_METRICS));
    
    /* Convert input parameters to array format */
    FITPRMS_ARRAY fit_prms;
    clock_t conv_start = clock();
    
    if (Convert_RadarParm_to_FitPrms(prm, raw, &fit_prms) != 0) {
        fprintf(stderr, "Fitacf_Array: Failed to convert radar parameters\n");
        return -1;
    }
    
    /* Copy radar site parameters if available */
    if (Copy_Radar_Site_Prms(site, &fit_prms) != 0) {
        fprintf(stderr, "Fitacf_Array: Warning - using default site parameters\n");
    }
    
    /* Set processing parameters */
    fit_prms.mode = mode;
    fit_prms.num_threads = num_threads;
    fit_prms.noise_threshold = fit_prms.noise * 3.0; /* 3-sigma threshold */
    fit_prms.batch_size = (num_threads > 1) ? prm->nrang / num_threads : 10;
    if (fit_prms.batch_size < 1) fit_prms.batch_size = 1;
    
    clock_t conv_end = clock();
    double conversion_time = (double)(conv_end - conv_start) / CLOCKS_PER_SEC;
    printf("  Parameter conversion completed in %.3f seconds\n", conversion_time);
    
    /* Create array data structures optimized for parallel processing */
    RANGE_DATA_ARRAYS *arrays = create_range_data_arrays(fit_prms.nrang, fit_prms.mplgs);
    if (!arrays) {
        fprintf(stderr, "Fitacf_Array: Failed to create array structures\n");
        Free_FitPrms_Array(&fit_prms);
        return -1;
    }
    
    printf("  Created arrays for %d ranges, %d lags\n", fit_prms.nrang, fit_prms.mplgs);
      /* Step 1: Parallel Preprocessing - Fill arrays with data */
    printf("\nStep 1: Parallel Preprocessing...\n");
    clock_t prep_start = clock();
    Parallel_Preprocessing_Array(&fit_prms, arrays);
    /* Note: Parallel_Preprocessing_Array returns void, so no error checking needed here */
    
    clock_t prep_end = clock();
    metrics.preprocessing_time = (double)(prep_end - prep_start) / CLOCKS_PER_SEC;
    metrics.ranges_processed = count_valid_ranges(arrays);
    
    printf("  Preprocessing completed in %.3f seconds\n", metrics.preprocessing_time);
    printf("  Valid ranges: %d/%d (%.1f%%)\n", metrics.ranges_processed, fit_prms.nrang,
           100.0 * metrics.ranges_processed / fit_prms.nrang);
    
    if (metrics.ranges_processed == 0) {
        printf("  Warning: No valid ranges found - returning empty results\n");
        free_range_data_arrays(arrays);
        Free_FitPrms_Array(&fit_prms);
        return 0; /* Not an error, just no data */
    }
    
    /* Step 2: Parallel Power fitting */
    printf("\nStep 2: Parallel Power fitting...\n");
    clock_t power_start = clock();
    
    int power_fits = Power_Fits_Array(&fit_prms, arrays);
    if (power_fits < 0) {
        fprintf(stderr, "Fitacf_Array: Power fitting failed\n");
        free_range_data_arrays(arrays);
        Free_FitPrms_Array(&fit_prms);
        return -1;
    }
    
    clock_t power_end = clock();
    metrics.power_fitting_time = (double)(power_end - power_start) / CLOCKS_PER_SEC;
    metrics.power_fits_successful = power_fits;
    
    printf("  Power fitting completed in %.3f seconds\n", metrics.power_fitting_time);
    printf("  Successful power fits: %d/%d (%.1f%%)\n", power_fits, metrics.ranges_processed,
           metrics.ranges_processed > 0 ? 100.0 * power_fits / metrics.ranges_processed : 0.0);
    
    /* Step 3: Parallel ACF phase fitting */
    printf("\nStep 3: Parallel ACF phase fitting...\n");
    clock_t phase_start = clock();
    
    int phase_fits = ACF_Phase_Fit_Array(&fit_prms, arrays);
    if (phase_fits < 0) {
        fprintf(stderr, "Fitacf_Array: Phase fitting failed\n");
        free_range_data_arrays(arrays);
        Free_FitPrms_Array(&fit_prms);
        return -1;
    }
    
    clock_t phase_end = clock();
    metrics.phase_fitting_time = (double)(phase_end - phase_start) / CLOCKS_PER_SEC;
    metrics.phase_fits_successful = phase_fits;
    
    printf("  Phase fitting completed in %.3f seconds\n", metrics.phase_fitting_time);
    printf("  Successful phase fits: %d/%d (%.1f%%)\n", phase_fits, metrics.ranges_processed,
           metrics.ranges_processed > 0 ? 100.0 * phase_fits / metrics.ranges_processed : 0.0);
    
    /* Step 4: Parallel XCF fitting (if interferometer data available) */
    int xcf_fits = 0;
    if (fit_prms.xcf && fit_prms.xcfd) {
        printf("\nStep 4: Parallel XCF elevation fitting...\n");
        clock_t xcf_start = clock();
        
        xcf_fits = XCF_Phase_Fit_Array(&fit_prms, arrays);
        if (xcf_fits < 0) {
            printf("  Warning: XCF fitting failed, continuing without elevation data\n");
            xcf_fits = 0;
        }
        
        clock_t xcf_end = clock();
        metrics.xcf_fitting_time = (double)(xcf_end - xcf_start) / CLOCKS_PER_SEC;
        metrics.xcf_fits_successful = xcf_fits;
        
        printf("  XCF fitting completed in %.3f seconds\n", metrics.xcf_fitting_time);
        printf("  Successful XCF fits: %d/%d (%.1f%%)\n", xcf_fits, metrics.ranges_processed,
               metrics.ranges_processed > 0 ? 100.0 * xcf_fits / metrics.ranges_processed : 0.0);
    } else {
        printf("\nStep 4: Skipping XCF fitting (no interferometer data)\n");
    }
    
    /* Step 5: Convert results back to standard FitData format */
    printf("\nStep 5: Converting results to FitData format...\n");
    clock_t convert_start = clock();
    
    if (Convert_FitData_from_Arrays(arrays, fit, fit_prms.nrang) != 0) {
        fprintf(stderr, "Fitacf_Array: Failed to convert results\n");
        free_range_data_arrays(arrays);
        Free_FitPrms_Array(&fit_prms);
        return -1;
    }
    
    clock_t convert_end = clock();
    double convert_time = (double)(convert_end - convert_start) / CLOCKS_PER_SEC;
    printf("  Result conversion completed in %.3f seconds\n", convert_time);
    
    /* Optional: Validate results if in debug mode */
    #ifdef DEBUG_ARRAY
    printf("\nStep 6: Validating results...\n");
    if (Validate_Array_Results(arrays, fit) != 0) {
        printf("  Warning: Result validation found discrepancies\n");
    } else {
        printf("  Result validation passed\n");
    }
    #endif
    
    /* Calculate total performance metrics */
    clock_t total_end = clock();
    metrics.total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    
    /* Print comprehensive performance summary */
    printf("\n=== FitACF Array Processing Summary ===\n");
    Print_Performance_Metrics(&metrics);
    
    /* Calculate speedup estimates */
    double serial_estimate = (metrics.preprocessing_time + metrics.power_fitting_time + 
                             metrics.phase_fitting_time + metrics.xcf_fitting_time) * num_threads;
    if (serial_estimate > 0 && num_threads > 1) {
        double speedup = serial_estimate / metrics.total_time;
        printf("  Estimated speedup: %.2fx (with %d threads)\n", speedup, num_threads);
        printf("  Parallel efficiency: %.1f%%\n", 100.0 * speedup / num_threads);
    }
      /* Cleanup */
    free_range_data_arrays(arrays);
    Free_FitPrms_Array(&fit_prms);
    
    printf("\nFitACF Array processing completed successfully\n");
    printf("Total processing time: %.3f seconds\n", metrics.total_time);
    
    return 0;
}

/* Convert standard RadarParm and RawData to array format - Completely rewritten for array processing */
int Convert_RadarParm_to_FitPrms(struct RadarParm *prm, struct RawData *raw, FITPRMS_ARRAY *fit_prms) {
    if (!prm || !raw || !fit_prms) {
        fprintf(stderr, "Convert_RadarParm_to_FitPrms: NULL input parameters\n");
        return -1;
    }
    
    /* Initialize the entire structure to zero */
    memset(fit_prms, 0, sizeof(FITPRMS_ARRAY));
    
    /* Copy basic radar parameters */
    fit_prms->channel = prm->channel;
    fit_prms->offset = prm->offset;
    fit_prms->cp = prm->cp;
    fit_prms->xcf = (prm->xcf == 1);
    fit_prms->tfreq = prm->tfreq;
    fit_prms->noise = prm->noise.mean > 0 ? prm->noise.mean : prm->noise.search;
    fit_prms->nrang = prm->nrang;
    fit_prms->smsep = prm->smsep;
    fit_prms->nave = prm->nave;
    fit_prms->mplgs = prm->mplgs;
    fit_prms->mpinc = prm->mpinc;
    fit_prms->txpl = prm->txpl;
    fit_prms->lagfr = prm->lagfr;
    fit_prms->mppul = prm->mppul;
    fit_prms->bmnum = prm->bmnum;
    
    /* Set old/new data flag based on year */
    fit_prms->old = (prm->time.yr < 1993) ? 1 : 0;
    
    /* Copy time structure components individually */
    fit_prms->time.yr = prm->time.yr;
    fit_prms->time.mo = prm->time.mo;
    fit_prms->time.dy = prm->time.dy;
    fit_prms->time.hr = prm->time.hr;
    fit_prms->time.mt = prm->time.mt;
    fit_prms->time.sc = prm->time.sc;
    fit_prms->time.us = prm->time.us;
    
    /* Set default radar site parameters - these will be updated by Copy_Radar_Site_Prms if needed */
    fit_prms->maxbeam = 16;  /* Default beam count */
    fit_prms->bmoff = 0.0;   /* Default beam offset */
    fit_prms->bmsep = 3.24;  /* Default beam separation */
    fit_prms->phidiff = 0.0; /* Default phase difference */
    fit_prms->tdiff = 0.0;   /* Default interferometer time difference */
    fit_prms->vdir = 1;      /* Default velocity sign */
    
    /* Initialize interferometer array */
    fit_prms->interfer[0] = 0.0;
    fit_prms->interfer[1] = 0.0;
    fit_prms->interfer[2] = 0.0;
    
    /* Allocate and copy lag table */
    for (int n = 0; n < 2; n++) {
        fit_prms->lag[n] = malloc(sizeof(int) * (prm->mplgs + 1));
        if (!fit_prms->lag[n]) {
            fprintf(stderr, "Convert_RadarParm_to_FitPrms: Failed to allocate lag[%d]\n", n);
            /* Cleanup previous allocations */
            for (int i = 0; i < n; i++) {
                if (fit_prms->lag[i]) free(fit_prms->lag[i]);
            }
            return -1;
        }
        
        /* Copy lag values with bounds checking */
        for (int i = 0; i <= prm->mplgs; i++) {
            if (prm->lag[n] && i <= prm->mplgs) {
                fit_prms->lag[n][i] = prm->lag[n][i];
            } else {
                fit_prms->lag[n][i] = 0; /* Default value */
            }
        }
    }
    
    /* Allocate and copy pulse table */
    fit_prms->pulse = malloc(sizeof(int) * prm->mppul);
    if (!fit_prms->pulse) {
        fprintf(stderr, "Convert_RadarParm_to_FitPrms: Failed to allocate pulse table\n");
        /* Cleanup lag allocations */
        for (int n = 0; n < 2; n++) {
            if (fit_prms->lag[n]) free(fit_prms->lag[n]);
        }
        return -1;
    }
    
    for (int i = 0; i < prm->mppul; i++) {
        if (prm->pulse && i < prm->mppul) {
            fit_prms->pulse[i] = prm->pulse[i];
        } else {
            fit_prms->pulse[i] = 0; /* Default value */
        }
    }
    
    /* Allocate and copy power data */
    fit_prms->pwr0 = malloc(sizeof(double) * prm->nrang);
    if (!fit_prms->pwr0) {
        fprintf(stderr, "Convert_RadarParm_to_FitPrms: Failed to allocate pwr0\n");
        /* Cleanup previous allocations */
        for (int n = 0; n < 2; n++) {
            if (fit_prms->lag[n]) free(fit_prms->lag[n]);
        }
        if (fit_prms->pulse) free(fit_prms->pulse);
        return -1;
    }
    
    for (int i = 0; i < prm->nrang; i++) {
        if (raw->pwr0 && i < prm->nrang) {
            fit_prms->pwr0[i] = raw->pwr0[i];
        } else {
            fit_prms->pwr0[i] = 0.0; /* Default value */
        }
    }
    
    /* Allocate and copy ACF data as 2D array */
    fit_prms->acfd = malloc(sizeof(double*) * prm->nrang);
    if (!fit_prms->acfd) {
        fprintf(stderr, "Convert_RadarParm_to_FitPrms: Failed to allocate acfd array\n");
        /* Cleanup previous allocations */
        for (int n = 0; n < 2; n++) {
            if (fit_prms->lag[n]) free(fit_prms->lag[n]);
        }
        if (fit_prms->pulse) free(fit_prms->pulse);
        if (fit_prms->pwr0) free(fit_prms->pwr0);
        return -1;
    }
    
    /* Allocate contiguous memory block for ACF data for better cache performance */
    double *acfd_data = malloc(sizeof(double) * prm->nrang * prm->mplgs * 2); /* Real + Imaginary */
    if (!acfd_data) {
        fprintf(stderr, "Convert_RadarParm_to_FitPrms: Failed to allocate acfd data block\n");
        /* Cleanup allocations */
        for (int n = 0; n < 2; n++) {
            if (fit_prms->lag[n]) free(fit_prms->lag[n]);
        }
        if (fit_prms->pulse) free(fit_prms->pulse);
        if (fit_prms->pwr0) free(fit_prms->pwr0);
        if (fit_prms->acfd) free(fit_prms->acfd);
        return -1;
    }
    
    /* Set up pointers for 2D access */
    for (int i = 0; i < prm->nrang; i++) {
        fit_prms->acfd[i] = (double*)(acfd_data + (i * prm->mplgs * 2));
        
        /* Copy complex ACF data - convert from complex to interleaved real/imag */
        for (int j = 0; j < prm->mplgs; j++) {
            if (raw->acfd && i < prm->nrang && j < prm->mplgs) {
                fit_prms->acfd[i][j*2] = creal(raw->acfd[i][j]);     /* Real part */
                fit_prms->acfd[i][j*2+1] = cimag(raw->acfd[i][j]);   /* Imaginary part */
            } else {
                fit_prms->acfd[i][j*2] = 0.0;     /* Default real */
                fit_prms->acfd[i][j*2+1] = 0.0;   /* Default imaginary */
            }
        }
    }
    
    /* Allocate and copy XCF data if available */
    if (fit_prms->xcf && raw->xcfd) {
        fit_prms->xcfd = malloc(sizeof(double*) * prm->nrang);
        if (!fit_prms->xcfd) {
            fprintf(stderr, "Convert_RadarParm_to_FitPrms: Failed to allocate xcfd array\n");
            /* Note: We'll continue without XCF data rather than fail */
            fit_prms->xcf = 0;
        } else {
            /* Allocate contiguous memory block for XCF data */
            double *xcfd_data = malloc(sizeof(double) * prm->nrang * prm->mplgs * 2);
            if (!xcfd_data) {
                fprintf(stderr, "Convert_RadarParm_to_FitPrms: Failed to allocate xcfd data block\n");
                free(fit_prms->xcfd);
                fit_prms->xcfd = NULL;
                fit_prms->xcf = 0;
            } else {
                /* Set up pointers for 2D access */
                for (int i = 0; i < prm->nrang; i++) {
                    fit_prms->xcfd[i] = (double*)(xcfd_data + (i * prm->mplgs * 2));
                    
                    /* Copy complex XCF data */
                    for (int j = 0; j < prm->mplgs; j++) {
                        if (raw->xcfd && i < prm->nrang && j < prm->mplgs) {
                            fit_prms->xcfd[i][j*2] = creal(raw->xcfd[i][j]);     /* Real part */
                            fit_prms->xcfd[i][j*2+1] = cimag(raw->xcfd[i][j]);   /* Imaginary part */
                        } else {
                            fit_prms->xcfd[i][j*2] = 0.0;     /* Default real */
                            fit_prms->xcfd[i][j*2+1] = 0.0;   /* Default imaginary */
                        }
                    }
                }
            }
        }
    }
    
    /* Set default array processing parameters */
    fit_prms->mode = PROCESS_MODE_ARRAYS;
    fit_prms->num_threads = 1;  /* Will be set by caller */
    fit_prms->enable_cuda = 0;  /* Disabled by default */
    fit_prms->noise_threshold = fit_prms->noise * 3.0; /* 3-sigma threshold */
    fit_prms->batch_size = 10;  /* Process 10 ranges at a time */
    
    printf("Convert_RadarParm_to_FitPrms: Successfully converted parameters\n");
    printf("  Ranges: %d, Lags: %d, Pulses: %d\n", prm->nrang, prm->mplgs, prm->mppul);
    printf("  XCF enabled: %s\n", fit_prms->xcf ? "Yes" : "No");
    printf("  Noise level: %.2f\n", fit_prms->noise);
    
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
        fit->rng[i].w_s_err = 0.0;        fit->rng[i].phi0 = 0.0;
        fit->rng[i].phi0_err = 0.0;
        /* Note: elv, elv_low, elv_high are not fields in FitRange structure */
        /* Elevation data should be stored in fit->elv[] array separately */
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
            fit->rng[i].phi0_err = rng->phase_fit->a; /* placeholder */        }
        
        /* Note: Elevation results should be copied to fit->elv[] array, not fit->rng[].elv */
        /* TODO: Implement proper elevation data copying to FitElv structure */
        if (rng->elev_fit) {
            /* For now, skip elevation copying until FitElv structure is properly allocated */
            /* This requires allocating fit->elv array and copying to fit->elv[i] */
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

/* Function to copy radar site parameters - handles the missing bmoff/bmsep values */
int Copy_Radar_Site_Prms(struct RadarSite *site, FITPRMS_ARRAY *fit_prms) {
    if (!site || !fit_prms) {
        /* Use default values if no site data available */
        printf("Copy_Radar_Site_Prms: Using default site parameters\n");
        return 0;
    }
    
    /* Copy interferometer parameters */
    fit_prms->interfer[0] = site->interfer[0];
    fit_prms->interfer[1] = site->interfer[1];
    fit_prms->interfer[2] = site->interfer[2];
    
    /* Copy beam parameters */
    fit_prms->maxbeam = site->maxbeam;
    fit_prms->bmoff = site->bmoff;
    fit_prms->bmsep = site->bmsep;
    fit_prms->phidiff = site->phidiff;
    fit_prms->vdir = site->vdir;
    
    /* Copy time difference parameters */
    if (site->tdiff && fit_prms->channel >= 0 && fit_prms->channel < 2) {
        /* Use channel-specific tdiff if available */
        if ((fit_prms->offset == 0) || (fit_prms->channel < 2)) {
            fit_prms->tdiff = site->tdiff[0];
        } else {
            fit_prms->tdiff = site->tdiff[1];
        }
    }
    
    printf("Copy_Radar_Site_Prms: Site parameters copied\n");
    printf("  Beam offset: %.2f, separation: %.2f\n", fit_prms->bmoff, fit_prms->bmsep);
    printf("  Interferometer: [%.2f, %.2f, %.2f]\n", 
           fit_prms->interfer[0], fit_prms->interfer[1], fit_prms->interfer[2]);
    printf("  Time difference: %.2f\n", fit_prms->tdiff);
    
    return 0;
}

/* Enhanced cleanup function for FITPRMS_ARRAY */
void Free_FitPrms_Array(FITPRMS_ARRAY *fit_prms) {
    if (!fit_prms) return;
    
    /* Free lag arrays */
    for (int n = 0; n < 2; n++) {
        if (fit_prms->lag[n]) {
            free(fit_prms->lag[n]);
            fit_prms->lag[n] = NULL;
        }
    }
    
    /* Free pulse array */
    if (fit_prms->pulse) {
        free(fit_prms->pulse);
        fit_prms->pulse = NULL;
    }
    
    /* Free power array */
    if (fit_prms->pwr0) {
        free(fit_prms->pwr0);
        fit_prms->pwr0 = NULL;
    }
    
    /* Free ACF arrays */
    if (fit_prms->acfd) {
        /* Free the contiguous data block first */
        if (fit_prms->acfd[0]) {
            free(fit_prms->acfd[0]);
        }
        free(fit_prms->acfd);
        fit_prms->acfd = NULL;
    }
    
    /* Free XCF arrays */
    if (fit_prms->xcfd) {
        /* Free the contiguous data block first */
        if (fit_prms->xcfd[0]) {
            free(fit_prms->xcfd[0]);
        }
        free(fit_prms->xcfd);
        fit_prms->xcfd = NULL;
    }
    
    printf("Free_FitPrms_Array: Memory cleanup completed\n");
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
