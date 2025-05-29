/*
 * Array-based operations implementation for SuperDARN FitACF v3.0
 * 
 * This file implements the array-based data structures and operations
 * to replace linked lists for massive parallelization.
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

#include "fit_structures_array.h"
#include "llist.h"

/* Memory management functions */

RANGE_DATA_ARRAYS* create_range_data_arrays(int max_ranges, int max_lags) {
    RANGE_DATA_ARRAYS *arrays = malloc(sizeof(RANGE_DATA_ARRAYS));
    if (!arrays) return NULL;
    
    /* Initialize basic fields */
    arrays->num_ranges = 0;
    arrays->max_ranges = max_ranges;
    
    /* Allocate range array */
    arrays->ranges = malloc(sizeof(RANGENODE_ARRAY) * max_ranges);
    if (!arrays->ranges) {
        free(arrays);
        return NULL;
    }
    
    /* Initialize each range node */
    for (int i = 0; i < max_ranges; i++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[i];
        memset(rng, 0, sizeof(RANGENODE_ARRAY));
        
        /* Initialize phase data array */
        rng->phases.phi = malloc(sizeof(double) * max_lags);
        rng->phases.t = malloc(sizeof(double) * max_lags);
        rng->phases.sigma = malloc(sizeof(double) * max_lags);
        rng->phases.lag_idx = malloc(sizeof(int) * max_lags);
        rng->phases.alpha_2 = malloc(sizeof(double) * max_lags);
        rng->phases.count = 0;
        rng->phases.capacity = max_lags;
        
        /* Initialize power data array */
        rng->pwrs.ln_pwr = malloc(sizeof(double) * max_lags);
        rng->pwrs.t = malloc(sizeof(double) * max_lags);
        rng->pwrs.sigma = malloc(sizeof(double) * max_lags);
        rng->pwrs.lag_idx = malloc(sizeof(int) * max_lags);
        rng->pwrs.alpha_2 = malloc(sizeof(double) * max_lags);
        rng->pwrs.count = 0;
        rng->pwrs.capacity = max_lags;
        
        /* Initialize alpha data array */
        rng->alpha_2.lag_idx = malloc(sizeof(int) * max_lags);
        rng->alpha_2.alpha_2 = malloc(sizeof(double) * max_lags);
        rng->alpha_2.count = 0;
        rng->alpha_2.capacity = max_lags;
        
        /* Initialize elevation data array */
        rng->elev.elev = malloc(sizeof(double) * max_lags);
        rng->elev.t = malloc(sizeof(double) * max_lags);
        rng->elev.sigma = malloc(sizeof(double) * max_lags);
        rng->elev.lag_idx = malloc(sizeof(int) * max_lags);
        rng->elev.count = 0;
        rng->elev.capacity = max_lags;
    }
    
    /* Allocate 2D matrices for direct access */
    arrays->phase_matrix = malloc(sizeof(double*) * max_ranges);
    arrays->power_matrix = malloc(sizeof(double*) * max_ranges);
    arrays->alpha_matrix = malloc(sizeof(double*) * max_ranges);
    arrays->sigma_phase_matrix = malloc(sizeof(double*) * max_ranges);
    arrays->sigma_power_matrix = malloc(sizeof(double*) * max_ranges);
    arrays->lag_idx_matrix = malloc(sizeof(int*) * max_ranges);
    
    for (int i = 0; i < max_ranges; i++) {
        arrays->phase_matrix[i] = malloc(sizeof(double) * max_lags);
        arrays->power_matrix[i] = malloc(sizeof(double) * max_lags);
        arrays->alpha_matrix[i] = malloc(sizeof(double) * max_lags);
        arrays->sigma_phase_matrix[i] = malloc(sizeof(double) * max_lags);
        arrays->sigma_power_matrix[i] = malloc(sizeof(double) * max_lags);
        arrays->lag_idx_matrix[i] = malloc(sizeof(int) * max_lags);
        
        /* Initialize with invalid values */
        for (int j = 0; j < max_lags; j++) {
            arrays->phase_matrix[i][j] = NAN;
            arrays->power_matrix[i][j] = NAN;
            arrays->alpha_matrix[i][j] = NAN;
            arrays->sigma_phase_matrix[i][j] = NAN;
            arrays->sigma_power_matrix[i][j] = NAN;
            arrays->lag_idx_matrix[i][j] = -1;
        }
    }
    
    /* Allocate helper arrays */
    arrays->range_lag_counts = malloc(sizeof(int) * max_ranges);
    arrays->range_valid = malloc(sizeof(int) * max_ranges);
    arrays->range_has_phase = malloc(sizeof(int) * max_ranges);
    arrays->range_has_power = malloc(sizeof(int) * max_ranges);
    
    /* Initialize helper arrays */
    for (int i = 0; i < max_ranges; i++) {
        arrays->range_lag_counts[i] = 0;
        arrays->range_valid[i] = 0;
        arrays->range_has_phase[i] = 0;
        arrays->range_has_power[i] = 0;
    }
    
    return arrays;
}

void free_range_data_arrays(RANGE_DATA_ARRAYS *arrays) {
    if (!arrays) return;
    
    /* Free range node arrays */
    if (arrays->ranges) {
        for (int i = 0; i < arrays->max_ranges; i++) {
            RANGENODE_ARRAY *rng = &arrays->ranges[i];
            
            /* Free phase data */
            free(rng->phases.phi);
            free(rng->phases.t);
            free(rng->phases.sigma);
            free(rng->phases.lag_idx);
            free(rng->phases.alpha_2);
            
            /* Free power data */
            free(rng->pwrs.ln_pwr);
            free(rng->pwrs.t);
            free(rng->pwrs.sigma);
            free(rng->pwrs.lag_idx);
            free(rng->pwrs.alpha_2);
            
            /* Free alpha data */
            free(rng->alpha_2.lag_idx);
            free(rng->alpha_2.alpha_2);
            
            /* Free elevation data */
            free(rng->elev.elev);
            free(rng->elev.t);
            free(rng->elev.sigma);
            free(rng->elev.lag_idx);
            
            /* Free CRI array if allocated */
            free(rng->CRI);
        }
        free(arrays->ranges);
    }
    
    /* Free 2D matrices */
    if (arrays->phase_matrix) {
        for (int i = 0; i < arrays->max_ranges; i++) {
            free(arrays->phase_matrix[i]);
        }
        free(arrays->phase_matrix);
    }
    
    if (arrays->power_matrix) {
        for (int i = 0; i < arrays->max_ranges; i++) {
            free(arrays->power_matrix[i]);
        }
        free(arrays->power_matrix);
    }
    
    if (arrays->alpha_matrix) {
        for (int i = 0; i < arrays->max_ranges; i++) {
            free(arrays->alpha_matrix[i]);
        }
        free(arrays->alpha_matrix);
    }
    
    if (arrays->sigma_phase_matrix) {
        for (int i = 0; i < arrays->max_ranges; i++) {
            free(arrays->sigma_phase_matrix[i]);
        }
        free(arrays->sigma_phase_matrix);
    }
    
    if (arrays->sigma_power_matrix) {
        for (int i = 0; i < arrays->max_ranges; i++) {
            free(arrays->sigma_power_matrix[i]);
        }
        free(arrays->sigma_power_matrix);
    }
    
    if (arrays->lag_idx_matrix) {
        for (int i = 0; i < arrays->max_ranges; i++) {
            free(arrays->lag_idx_matrix[i]);
        }
        free(arrays->lag_idx_matrix);
    }
    
    /* Free helper arrays */
    free(arrays->range_lag_counts);
    free(arrays->range_valid);
    free(arrays->range_has_phase);
    free(arrays->range_has_power);
    
    free(arrays);
}

/* Data manipulation functions */

int add_phase_data(RANGE_DATA_ARRAYS *arrays, int range_idx, 
                   double phi, double t, double sigma, int lag_idx, double alpha_2) {
    if (!arrays || range_idx < 0 || range_idx >= arrays->max_ranges) {
        return -1;
    }
    
    PHASE_DATA_ARRAY *phase_data = &arrays->ranges[range_idx].phases;
    
    if (phase_data->count >= phase_data->capacity) {
        return -1; /* Array full */
    }
    
    int idx = phase_data->count;
    phase_data->phi[idx] = phi;
    phase_data->t[idx] = t;
    phase_data->sigma[idx] = sigma;
    phase_data->lag_idx[idx] = lag_idx;
    phase_data->alpha_2[idx] = alpha_2;
    phase_data->count++;
    
    arrays->range_has_phase[range_idx] = 1;
    
    return 0;
}

int add_power_data(RANGE_DATA_ARRAYS *arrays, int range_idx,
                   double ln_pwr, double t, double sigma, int lag_idx, double alpha_2) {
    if (!arrays || range_idx < 0 || range_idx >= arrays->max_ranges) {
        return -1;
    }
    
    POWER_DATA_ARRAY *power_data = &arrays->ranges[range_idx].pwrs;
    
    if (power_data->count >= power_data->capacity) {
        return -1; /* Array full */
    }
    
    int idx = power_data->count;
    power_data->ln_pwr[idx] = ln_pwr;
    power_data->t[idx] = t;
    power_data->sigma[idx] = sigma;
    power_data->lag_idx[idx] = lag_idx;
    power_data->alpha_2[idx] = alpha_2;
    power_data->count++;
    
    arrays->range_has_power[range_idx] = 1;
    
    return 0;
}

int add_alpha_data(RANGE_DATA_ARRAYS *arrays, int range_idx,
                   int lag_idx, double alpha_2) {
    if (!arrays || range_idx < 0 || range_idx >= arrays->max_ranges) {
        return -1;
    }
    
    ALPHA_DATA_ARRAY *alpha_data = &arrays->ranges[range_idx].alpha_2;
    
    if (alpha_data->count >= alpha_data->capacity) {
        return -1; /* Array full */
    }
    
    int idx = alpha_data->count;
    alpha_data->lag_idx[idx] = lag_idx;
    alpha_data->alpha_2[idx] = alpha_2;
    alpha_data->count++;
    
    return 0;
}

int add_elev_data(RANGE_DATA_ARRAYS *arrays, int range_idx,
                  double elev, double t, double sigma, int lag_idx) {
    if (!arrays || range_idx < 0 || range_idx >= arrays->max_ranges) {
        return -1;
    }
    
    ELEV_DATA_ARRAY *elev_data = &arrays->ranges[range_idx].elev;
    
    if (elev_data->count >= elev_data->capacity) {
        return -1; /* Array full */
    }
    
    int idx = elev_data->count;
    elev_data->elev[idx] = elev;
    elev_data->t[idx] = t;
    elev_data->sigma[idx] = sigma;
    elev_data->lag_idx[idx] = lag_idx;
    elev_data->count++;
    
    return 0;
}

/* Matrix operations for parallel processing */

int populate_matrices(RANGE_DATA_ARRAYS *arrays) {
    if (!arrays) return -1;
    
    for (int range_idx = 0; range_idx < arrays->num_ranges; range_idx++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[range_idx];
        
        /* Find maximum lag count for this range */
        int max_lags = 0;
        if (rng->phases.count > max_lags) max_lags = rng->phases.count;
        if (rng->pwrs.count > max_lags) max_lags = rng->pwrs.count;
        
        arrays->range_lag_counts[range_idx] = max_lags;
        
        /* Populate phase matrix */
        for (int lag = 0; lag < rng->phases.count; lag++) {
            int lag_idx = rng->phases.lag_idx[lag];
            if (lag_idx >= 0 && lag_idx < MAX_LAGS_PER_RANGE) {
                arrays->phase_matrix[range_idx][lag_idx] = rng->phases.phi[lag];
                arrays->sigma_phase_matrix[range_idx][lag_idx] = rng->phases.sigma[lag];
                arrays->lag_idx_matrix[range_idx][lag_idx] = lag_idx;
            }
        }
        
        /* Populate power matrix */
        for (int lag = 0; lag < rng->pwrs.count; lag++) {
            int lag_idx = rng->pwrs.lag_idx[lag];
            if (lag_idx >= 0 && lag_idx < MAX_LAGS_PER_RANGE) {
                arrays->power_matrix[range_idx][lag_idx] = rng->pwrs.ln_pwr[lag];
                arrays->sigma_power_matrix[range_idx][lag_idx] = rng->pwrs.sigma[lag];
            }
        }
        
        /* Populate alpha matrix */
        for (int lag = 0; lag < rng->alpha_2.count; lag++) {
            int lag_idx = rng->alpha_2.lag_idx[lag];
            if (lag_idx >= 0 && lag_idx < MAX_LAGS_PER_RANGE) {
                arrays->alpha_matrix[range_idx][lag_idx] = rng->alpha_2.alpha_2[lag];
            }
        }
    }
    
    return 0;
}

/* OpenMP parallel processing functions */

void parallel_power_fitting(RANGE_DATA_ARRAYS *arrays, int num_threads) {
    if (!arrays) return;
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for schedule(dynamic)
    for (int range_idx = 0; range_idx < arrays->num_ranges; range_idx++) {
        if (!arrays->range_valid[range_idx] || !arrays->range_has_power[range_idx]) {
            continue;
        }
        
        /* Power fitting algorithm can be parallelized here */
        /* This is where the original Power_Fits function logic would go */
        /* but operating on arrays instead of linked lists */
        
        RANGENODE_ARRAY *rng = &arrays->ranges[range_idx];
        
        /* Example: Simple linear fitting on power data */
        if (rng->pwrs.count >= 2) {
            /* Parallel-friendly power fitting algorithm */
            double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
            int n = rng->pwrs.count;
            
            for (int i = 0; i < n; i++) {
                double x = rng->pwrs.t[i];
                double y = rng->pwrs.ln_pwr[i];
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }
            
            double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            double intercept = (sum_y - slope * sum_x) / n;
            
            /* Store results in fit data structures */
            /* This would populate rng->l_pwr_fit, etc. */
        }
    }
#endif
}

void parallel_phase_fitting(RANGE_DATA_ARRAYS *arrays, int num_threads) {
    if (!arrays) return;
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for schedule(dynamic)
    for (int range_idx = 0; range_idx < arrays->num_ranges; range_idx++) {
        if (!arrays->range_valid[range_idx] || !arrays->range_has_phase[range_idx]) {
            continue;
        }
        
        /* Phase fitting algorithm can be parallelized here */
        /* This is where the original ACF_Phase_Fit function logic would go */
        
        RANGENODE_ARRAY *rng = &arrays->ranges[range_idx];
        
        /* Example: Phase unwrapping and fitting */
        if (rng->phases.count >= 2) {
            /* Parallel-friendly phase fitting algorithm */
            for (int i = 0; i < rng->phases.count; i++) {
                /* Phase processing that can be done in parallel */
                double phase = rng->phases.phi[i];
                /* Apply phase corrections, unwrapping, etc. */
            }
        }
    }
#endif
}

/* Utility functions */

int mark_valid_ranges(RANGE_DATA_ARRAYS *arrays, double noise_threshold) {
    if (!arrays) return -1;
    
    int valid_count = 0;
    
    for (int i = 0; i < arrays->num_ranges; i++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[i];
        
        /* Mark range as valid if it has sufficient data above noise threshold */
        int has_signal = 0;
        
        /* Check power data */
        for (int j = 0; j < rng->pwrs.count; j++) {
            if (rng->pwrs.ln_pwr[j] > noise_threshold) {
                has_signal = 1;
                break;
            }
        }
        
        /* Check if we have enough phase data */
        int has_enough_phase = rng->phases.count >= 2;
        
        arrays->range_valid[i] = has_signal && has_enough_phase;
        if (arrays->range_valid[i]) {
            valid_count++;
        }
    }
    
    return valid_count;
}

int count_valid_ranges(RANGE_DATA_ARRAYS *arrays) {
    if (!arrays) return 0;
    
    int count = 0;
    for (int i = 0; i < arrays->num_ranges; i++) {
        if (arrays->range_valid[i]) {
            count++;
        }
    }
    return count;
}

/* Statistics and validation */

ARRAY_STATS calculate_array_stats(RANGE_DATA_ARRAYS *arrays) {
    ARRAY_STATS stats;
    memset(&stats, 0, sizeof(ARRAY_STATS));
    
    if (!arrays) return stats;
    
    stats.total_ranges = arrays->num_ranges;
    stats.valid_ranges = count_valid_ranges(arrays);
    
    for (int i = 0; i < arrays->num_ranges; i++) {
        stats.total_phase_points += arrays->ranges[i].phases.count;
        stats.total_power_points += arrays->ranges[i].pwrs.count;
    }
    
    if (stats.total_ranges > 0) {
        stats.avg_lags_per_range = (double)(stats.total_phase_points + stats.total_power_points) / 
                                  (2.0 * stats.total_ranges);
    }
    
    /* Calculate memory usage */
    size_t total_memory = sizeof(RANGE_DATA_ARRAYS);
    total_memory += arrays->max_ranges * sizeof(RANGENODE_ARRAY);
    total_memory += arrays->max_ranges * MAX_LAGS_PER_RANGE * sizeof(double) * 5; /* matrices */
    total_memory += arrays->max_ranges * MAX_LAGS_PER_RANGE * sizeof(int); /* lag indices */
    
    stats.memory_usage_mb = total_memory / (1024.0 * 1024.0);
    
    return stats;
}
