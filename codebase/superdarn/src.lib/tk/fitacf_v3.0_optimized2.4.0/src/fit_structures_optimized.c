/*
 * Implementation of optimized data structures for SuperDARN FitACF v3.0_optimized2
 * 
 * This file implements high-performance data structures that completely eliminate
 * linked lists in favor of contiguous memory arrays for maximum parallelization.
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "fit_structures_optimized.h"

/* Global performance tracking */
static clock_t global_start_time = 0;
static double total_allocation_time = 0.0;
static size_t total_memory_allocated = 0;

/* Default configuration presets */
const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_DEFAULT = {
    .processing_mode = PROCESS_MODE_OPENMP,
    .num_threads = 0,  // Auto-detect
    .cuda_device_id = 0,
    .enable_memory_pool = 1,
    .memory_pool_size = MEMORY_POOL_SIZE,
    .enable_simd = 1,
    .enable_vectorization = 1,
    .noise_threshold = 1.0,
    .batch_processing_size = 64,
    .cache_optimization_level = 2,
    .debug_level = 0,
    .profiling_enabled = 0
};

const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_PERFORMANCE = {
    .processing_mode = PROCESS_MODE_HYBRID,
    .num_threads = 0,  // Use all available
    .cuda_device_id = 0,
    .enable_memory_pool = 1,
    .memory_pool_size = MEMORY_POOL_SIZE * 2,
    .enable_simd = 1,
    .enable_vectorization = 1,
    .noise_threshold = 0.5,
    .batch_processing_size = 128,
    .cache_optimization_level = 3,
    .debug_level = 0,
    .profiling_enabled = 1
};

const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_MEMORY_OPTIMIZED = {
    .processing_mode = PROCESS_MODE_OPENMP,
    .num_threads = 4,
    .cuda_device_id = -1,  // Disable CUDA
    .enable_memory_pool = 1,
    .memory_pool_size = MEMORY_POOL_SIZE / 2,
    .enable_simd = 1,
    .enable_vectorization = 1,
    .noise_threshold = 1.0,
    .batch_processing_size = 32,
    .cache_optimization_level = 1,
    .debug_level = 0,
    .profiling_enabled = 0
};

const FITACF_CONFIG_OPTIMIZED FITACF_CONFIG_CUDA_OPTIMIZED = {
    .processing_mode = PROCESS_MODE_CUDA,
    .num_threads = 2,  // Minimal CPU threads
    .cuda_device_id = 0,
    .enable_memory_pool = 1,
    .memory_pool_size = MEMORY_POOL_SIZE * 4,
    .enable_simd = 1,
    .enable_vectorization = 1,
    .noise_threshold = 0.1,
    .batch_processing_size = 256,
    .cache_optimization_level = 3,
    .debug_level = 0,
    .profiling_enabled = 1
};

/* Memory management functions */

FITACF_DATA_OPTIMIZED* create_fitacf_data_optimized(int max_ranges) {
    clock_t start = clock();
    
    if (max_ranges <= 0 || max_ranges > MAX_RANGES_OPTIMIZED) {
        fprintf(stderr, "Error: Invalid max_ranges %d (must be 1-%d)\n", 
                max_ranges, MAX_RANGES_OPTIMIZED);
        return NULL;
    }
    
    FITACF_DATA_OPTIMIZED *data = (FITACF_DATA_OPTIMIZED*)ALIGNED_MALLOC(sizeof(FITACF_DATA_OPTIMIZED));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate FITACF_DATA_OPTIMIZED structure\n");
        return NULL;
    }
    
    memset(data, 0, sizeof(FITACF_DATA_OPTIMIZED));
    
    /* Initialize basic parameters */
    data->max_ranges = max_ranges;
    data->num_ranges = 0;
    data->processing_mode = PROCESS_MODE_OPENMP;
    data->num_threads = 0;  // Auto-detect
    data->cuda_device_id = -1;
    
    /* Allocate range array with cache alignment */
    data->ranges = (RANGENODE_OPTIMIZED*)ALIGNED_MALLOC(max_ranges * sizeof(RANGENODE_OPTIMIZED));
    if (!data->ranges) {
        fprintf(stderr, "Error: Failed to allocate ranges array\n");
        ALIGNED_FREE(data);
        return NULL;
    }
    memset(data->ranges, 0, max_ranges * sizeof(RANGENODE_OPTIMIZED));
    
    /* Allocate flattened matrices for massive parallelization */
    size_t matrix_size = max_ranges * MAX_LAGS_PER_RANGE_OPTIMIZED;
    size_t matrix_bytes = matrix_size * sizeof(double);
    
    data->phase_matrix = (double*)ALIGNED_MALLOC(matrix_bytes);
    data->power_matrix = (double*)ALIGNED_MALLOC(matrix_bytes);
    data->alpha_matrix = (double*)ALIGNED_MALLOC(matrix_bytes);
    data->elev_matrix = (double*)ALIGNED_MALLOC(matrix_bytes);
    data->sigma_phase_matrix = (double*)ALIGNED_MALLOC(matrix_bytes);
    data->sigma_power_matrix = (double*)ALIGNED_MALLOC(matrix_bytes);
    data->t_matrix = (double*)ALIGNED_MALLOC(matrix_bytes);
    data->lag_idx_matrix = (int*)ALIGNED_MALLOC(matrix_size * sizeof(int));
    
    if (!data->phase_matrix || !data->power_matrix || !data->alpha_matrix ||
        !data->elev_matrix || !data->sigma_phase_matrix || !data->sigma_power_matrix ||
        !data->t_matrix || !data->lag_idx_matrix) {
        fprintf(stderr, "Error: Failed to allocate flattened matrices\n");
        destroy_fitacf_data_optimized(data);
        return NULL;
    }
    
    /* Initialize matrices to safe values */
    memset(data->phase_matrix, 0, matrix_bytes);
    memset(data->power_matrix, 0, matrix_bytes);
    memset(data->alpha_matrix, 0, matrix_bytes);
    memset(data->elev_matrix, 0, matrix_bytes);
    memset(data->sigma_phase_matrix, 0, matrix_bytes);
    memset(data->sigma_power_matrix, 0, matrix_bytes);
    memset(data->t_matrix, 0, matrix_bytes);
    memset(data->lag_idx_matrix, -1, matrix_size * sizeof(int));
    
    /* Allocate range metadata arrays */
    data->range_lag_counts = (int*)ALIGNED_MALLOC(max_ranges * sizeof(int));
    data->range_flags = (uint8_t*)ALIGNED_MALLOC(max_ranges * sizeof(uint8_t));
    data->range_noise_levels = (double*)ALIGNED_MALLOC(max_ranges * sizeof(double));
    
    if (!data->range_lag_counts || !data->range_flags || !data->range_noise_levels) {
        fprintf(stderr, "Error: Failed to allocate range metadata arrays\n");
        destroy_fitacf_data_optimized(data);
        return NULL;
    }
    
    memset(data->range_lag_counts, 0, max_ranges * sizeof(int));
    memset(data->range_flags, 0, max_ranges * sizeof(uint8_t));
    memset(data->range_noise_levels, 0, max_ranges * sizeof(double));
    
    /* Initialize performance tracking */
    data->total_processing_time = 0.0;
    data->preprocessing_time = 0.0;
    data->fitting_time = 0.0;
    data->postprocessing_time = 0.0;
    data->total_memory_used = sizeof(FITACF_DATA_OPTIMIZED) + 
                             max_ranges * sizeof(RANGENODE_OPTIMIZED) +
                             matrix_bytes * 7 + matrix_size * sizeof(int) +
                             max_ranges * (sizeof(int) + sizeof(uint8_t) + sizeof(double));
    
    total_memory_allocated += data->total_memory_used;
    
    clock_t end = clock();
    total_allocation_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Created FITACF_DATA_OPTIMIZED: %d ranges, %.2f MB allocated in %.3f ms\n",
           max_ranges, data->total_memory_used / (1024.0 * 1024.0),
           ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0);
    
    return data;
}

void destroy_fitacf_data_optimized(FITACF_DATA_OPTIMIZED *data) {
    if (!data) return;
    
    /* Free individual range data */
    if (data->ranges) {
        for (int i = 0; i < data->max_ranges; i++) {
            free_range_optimized(&data->ranges[i]);
        }
        ALIGNED_FREE(data->ranges);
    }
    
    /* Free flattened matrices */
    if (data->phase_matrix) ALIGNED_FREE(data->phase_matrix);
    if (data->power_matrix) ALIGNED_FREE(data->power_matrix);
    if (data->alpha_matrix) ALIGNED_FREE(data->alpha_matrix);
    if (data->elev_matrix) ALIGNED_FREE(data->elev_matrix);
    if (data->sigma_phase_matrix) ALIGNED_FREE(data->sigma_phase_matrix);
    if (data->sigma_power_matrix) ALIGNED_FREE(data->sigma_power_matrix);
    if (data->t_matrix) ALIGNED_FREE(data->t_matrix);
    if (data->lag_idx_matrix) ALIGNED_FREE(data->lag_idx_matrix);
    
    /* Free range metadata arrays */
    if (data->range_lag_counts) ALIGNED_FREE(data->range_lag_counts);
    if (data->range_flags) ALIGNED_FREE(data->range_flags);
    if (data->range_noise_levels) ALIGNED_FREE(data->range_noise_levels);
    
    /* Free memory pool if allocated */
    if (data->memory_pool) {
        ALIGNED_FREE(data->memory_pool);
    }
    
    total_memory_allocated -= data->total_memory_used;
    
    ALIGNED_FREE(data);
}

int initialize_memory_pool(FITACF_DATA_OPTIMIZED *data, size_t pool_size) {
    if (!data || pool_size == 0) return -1;
    
    if (data->memory_pool) {
        ALIGNED_FREE(data->memory_pool);
    }
    
    data->memory_pool = (char*)ALIGNED_MALLOC(pool_size);
    if (!data->memory_pool) {
        fprintf(stderr, "Error: Failed to allocate memory pool of size %zu\n", pool_size);
        return -1;
    }
    
    data->memory_pool_size = pool_size;
    data->memory_pool_offset = 0;
    
    printf("Initialized memory pool: %.2f MB\n", pool_size / (1024.0 * 1024.0));
    return 0;
}

void* allocate_from_pool(FITACF_DATA_OPTIMIZED *data, size_t size) {
    if (!data || !data->memory_pool || size == 0) return NULL;
    
    /* Align size to SIMD boundary */
    size_t aligned_size = (size + SIMD_ALIGNMENT - 1) & ~(SIMD_ALIGNMENT - 1);
    
    if (data->memory_pool_offset + aligned_size > data->memory_pool_size) {
        fprintf(stderr, "Warning: Memory pool exhausted, falling back to malloc\n");
        return ALIGNED_MALLOC(size);
    }
    
    void *ptr = data->memory_pool + data->memory_pool_offset;
    data->memory_pool_offset += aligned_size;
    
    return ptr;
}

void reset_memory_pool(FITACF_DATA_OPTIMIZED *data) {
    if (!data) return;
    data->memory_pool_offset = 0;
}

/* Vectorized array management functions */

int allocate_vectorized_array(VECTORIZED_DATA_ARRAY *array, int capacity) {
    if (!array || capacity <= 0) return -1;
    
    array->capacity = capacity;
    array->count = 0;
    array->stride = 1;  /* Default stride for contiguous access */
    
    size_t double_bytes = capacity * sizeof(double);
    size_t int_bytes = capacity * sizeof(int);
    
    array->values = (double*)ALIGNED_MALLOC(double_bytes);
    array->t_values = (double*)ALIGNED_MALLOC(double_bytes);
    array->sigma_values = (double*)ALIGNED_MALLOC(double_bytes);
    array->alpha_2_values = (double*)ALIGNED_MALLOC(double_bytes);
    array->lag_indices = (int*)ALIGNED_MALLOC(int_bytes);
    
    if (!array->values || !array->t_values || !array->sigma_values || 
        !array->alpha_2_values || !array->lag_indices) {
        free_vectorized_array(array);
        return -1;
    }
    
    /* Initialize to safe values */
    memset(array->values, 0, double_bytes);
    memset(array->t_values, 0, double_bytes);
    memset(array->sigma_values, 0, double_bytes);
    memset(array->alpha_2_values, 0, double_bytes);
    memset(array->lag_indices, -1, int_bytes);
    
    return 0;
}

void free_vectorized_array(VECTORIZED_DATA_ARRAY *array) {
    if (!array) return;
    
    if (array->values) ALIGNED_FREE(array->values);
    if (array->t_values) ALIGNED_FREE(array->t_values);
    if (array->sigma_values) ALIGNED_FREE(array->sigma_values);
    if (array->alpha_2_values) ALIGNED_FREE(array->alpha_2_values);
    if (array->lag_indices) ALIGNED_FREE(array->lag_indices);
    
    memset(array, 0, sizeof(VECTORIZED_DATA_ARRAY));
}

int resize_vectorized_array(VECTORIZED_DATA_ARRAY *array, int new_capacity) {
    if (!array || new_capacity <= array->capacity) return 0;
    
    size_t old_double_bytes = array->capacity * sizeof(double);
    size_t new_double_bytes = new_capacity * sizeof(double);
    size_t old_int_bytes = array->capacity * sizeof(int);
    size_t new_int_bytes = new_capacity * sizeof(int);
    
    /* Reallocate all arrays */
    double *new_values = (double*)ALIGNED_MALLOC(new_double_bytes);
    double *new_t_values = (double*)ALIGNED_MALLOC(new_double_bytes);
    double *new_sigma_values = (double*)ALIGNED_MALLOC(new_double_bytes);
    double *new_alpha_2_values = (double*)ALIGNED_MALLOC(new_double_bytes);
    int *new_lag_indices = (int*)ALIGNED_MALLOC(new_int_bytes);
    
    if (!new_values || !new_t_values || !new_sigma_values || 
        !new_alpha_2_values || !new_lag_indices) {
        if (new_values) ALIGNED_FREE(new_values);
        if (new_t_values) ALIGNED_FREE(new_t_values);
        if (new_sigma_values) ALIGNED_FREE(new_sigma_values);
        if (new_alpha_2_values) ALIGNED_FREE(new_alpha_2_values);
        if (new_lag_indices) ALIGNED_FREE(new_lag_indices);
        return -1;
    }
    
    /* Copy existing data */
    if (array->values) {
        memcpy(new_values, array->values, old_double_bytes);
        memcpy(new_t_values, array->t_values, old_double_bytes);
        memcpy(new_sigma_values, array->sigma_values, old_double_bytes);
        memcpy(new_alpha_2_values, array->alpha_2_values, old_double_bytes);
        memcpy(new_lag_indices, array->lag_indices, old_int_bytes);
    }
    
    /* Initialize new region */
    memset(new_values + array->capacity, 0, new_double_bytes - old_double_bytes);
    memset(new_t_values + array->capacity, 0, new_double_bytes - old_double_bytes);
    memset(new_sigma_values + array->capacity, 0, new_double_bytes - old_double_bytes);
    memset(new_alpha_2_values + array->capacity, 0, new_double_bytes - old_double_bytes);
    memset(new_lag_indices + array->capacity, -1, new_int_bytes - old_int_bytes);
    
    /* Free old arrays and update pointers */
    free_vectorized_array(array);
    
    array->values = new_values;
    array->t_values = new_t_values;
    array->sigma_values = new_sigma_values;
    array->alpha_2_values = new_alpha_2_values;
    array->lag_indices = new_lag_indices;
    array->capacity = new_capacity;
    
    return 0;
}

int add_data_to_vectorized_array(VECTORIZED_DATA_ARRAY *array, double value, 
                                 double t, double sigma, double alpha_2, int lag_idx) {
    if (!array) return -1;
    
    if (array->count >= array->capacity) {
        /* Auto-resize if needed */
        int new_capacity = array->capacity * 2;
        if (resize_vectorized_array(array, new_capacity) != 0) {
            return -1;
        }
    }
    
    int idx = array->count;
    array->values[idx] = value;
    array->t_values[idx] = t;
    array->sigma_values[idx] = sigma;
    array->alpha_2_values[idx] = alpha_2;
    array->lag_indices[idx] = lag_idx;
    
    array->count++;
    return idx;
}

/* Range node management functions */

int initialize_range_optimized(RANGENODE_OPTIMIZED *range, int range_num) {
    if (!range || range_num < 0) return -1;
    
    memset(range, 0, sizeof(RANGENODE_OPTIMIZED));
    
    range->range = range_num;
    range->refrc_idx = 1.0;  /* Default refractive index */
    range->processing_flags = 0;
    
    /* Allocate vectorized arrays with default capacity */
    int default_capacity = MAX_LAGS_PER_RANGE_OPTIMIZED;
    
    if (allocate_vectorized_array(&range->phases, default_capacity) != 0 ||
        allocate_vectorized_array(&range->powers, default_capacity) != 0 ||
        allocate_vectorized_array(&range->elevations, default_capacity) != 0 ||
        allocate_vectorized_array(&range->alphas, default_capacity) != 0) {
        free_range_optimized(range);
        return -1;
    }
    
    /* Allocate CRI array */
    range->CRI = (double*)ALIGNED_MALLOC(MAX_LAGS_PER_RANGE_OPTIMIZED * sizeof(double));
    if (!range->CRI) {
        free_range_optimized(range);
        return -1;
    }
    
    memset(range->CRI, 0, MAX_LAGS_PER_RANGE_OPTIMIZED * sizeof(double));
    range->CRI_count = 0;
    
    /* Initialize fit data structures */
    memset(&range->l_pwr_fit, 0, sizeof(FITDATA));
    memset(&range->q_pwr_fit, 0, sizeof(FITDATA));
    memset(&range->l_pwr_fit_err, 0, sizeof(FITDATA));
    memset(&range->q_pwr_fit_err, 0, sizeof(FITDATA));
    memset(&range->phase_fit, 0, sizeof(FITDATA));
    memset(&range->elev_fit, 0, sizeof(FITDATA));
    
    return 0;
}

void free_range_optimized(RANGENODE_OPTIMIZED *range) {
    if (!range) return;
    
    free_vectorized_array(&range->phases);
    free_vectorized_array(&range->powers);
    free_vectorized_array(&range->elevations);
    free_vectorized_array(&range->alphas);
    
    if (range->CRI) {
        ALIGNED_FREE(range->CRI);
        range->CRI = NULL;
    }
    
    memset(range, 0, sizeof(RANGENODE_OPTIMIZED));
}

int add_phase_data_optimized(RANGENODE_OPTIMIZED *range, double phi, double t, 
                            double sigma, int lag_idx, double alpha_2) {
    if (!range) return -1;
    
    return add_data_to_vectorized_array(&range->phases, phi, t, sigma, alpha_2, lag_idx);
}

int add_power_data_optimized(RANGENODE_OPTIMIZED *range, double ln_pwr, double t, 
                            double sigma, int lag_idx, double alpha_2) {
    if (!range) return -1;
    
    return add_data_to_vectorized_array(&range->powers, ln_pwr, t, sigma, alpha_2, lag_idx);
}

int add_elevation_data_optimized(RANGENODE_OPTIMIZED *range, double elev, double t, 
                                double sigma, int lag_idx) {
    if (!range) return -1;
    
    return add_data_to_vectorized_array(&range->elevations, elev, t, sigma, 0.0, lag_idx);
}

/* Matrix operations for parallel processing */

int populate_flat_matrices(FITACF_DATA_OPTIMIZED *data) {
    if (!data || !data->ranges) return -1;
    
    clock_t start = clock();
    
    /* Clear existing matrix data */
    size_t matrix_size = data->max_ranges * MAX_LAGS_PER_RANGE_OPTIMIZED;
    memset(data->phase_matrix, 0, matrix_size * sizeof(double));
    memset(data->power_matrix, 0, matrix_size * sizeof(double));
    memset(data->alpha_matrix, 0, matrix_size * sizeof(double));
    memset(data->elev_matrix, 0, matrix_size * sizeof(double));
    memset(data->sigma_phase_matrix, 0, matrix_size * sizeof(double));
    memset(data->sigma_power_matrix, 0, matrix_size * sizeof(double));
    memset(data->t_matrix, 0, matrix_size * sizeof(double));
    memset(data->lag_idx_matrix, -1, matrix_size * sizeof(int));
    memset(data->range_lag_counts, 0, data->max_ranges * sizeof(int));
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16) if(data->num_ranges > 100)
#endif
    for (int range_idx = 0; range_idx < data->num_ranges; range_idx++) {
        RANGENODE_OPTIMIZED *range = &data->ranges[range_idx];
        int base_offset = range_idx * MAX_LAGS_PER_RANGE_OPTIMIZED;
        
        /* Populate phase data */
        int phase_count = range->phases.count;
        for (int i = 0; i < phase_count && i < MAX_LAGS_PER_RANGE_OPTIMIZED; i++) {
            int matrix_idx = base_offset + i;
            data->phase_matrix[matrix_idx] = range->phases.values[i];
            data->sigma_phase_matrix[matrix_idx] = range->phases.sigma_values[i];
            data->t_matrix[matrix_idx] = range->phases.t_values[i];
            data->lag_idx_matrix[matrix_idx] = range->phases.lag_indices[i];
        }
        
        /* Populate power data */
        int power_count = range->powers.count;
        for (int i = 0; i < power_count && i < MAX_LAGS_PER_RANGE_OPTIMIZED; i++) {
            int matrix_idx = base_offset + i;
            data->power_matrix[matrix_idx] = range->powers.values[i];
            data->sigma_power_matrix[matrix_idx] = range->powers.sigma_values[i];
        }
        
        /* Populate alpha data */
        int alpha_count = range->alphas.count;
        for (int i = 0; i < alpha_count && i < MAX_LAGS_PER_RANGE_OPTIMIZED; i++) {
            int matrix_idx = base_offset + i;
            data->alpha_matrix[matrix_idx] = range->alphas.values[i];
        }
        
        /* Populate elevation data */
        int elev_count = range->elevations.count;
        for (int i = 0; i < elev_count && i < MAX_LAGS_PER_RANGE_OPTIMIZED; i++) {
            int matrix_idx = base_offset + i;
            data->elev_matrix[matrix_idx] = range->elevations.values[i];
        }
        
        /* Update range metadata */
        data->range_lag_counts[range_idx] = phase_count;
        
        /* Set range flags */
        data->range_flags[range_idx] = 0;
        if (phase_count > 0) data->range_flags[range_idx] |= RANGE_FLAG_HAS_PHASE;
        if (power_count > 0) data->range_flags[range_idx] |= RANGE_FLAG_HAS_POWER;
        if (elev_count > 0) data->range_flags[range_idx] |= RANGE_FLAG_HAS_ELEVATION;
        if (phase_count >= 3 || power_count >= 3) data->range_flags[range_idx] |= RANGE_FLAG_VALID;
    }
    
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Populated flat matrices for %d ranges in %.3f ms\n", 
           data->num_ranges, elapsed * 1000.0);
    
    return 0;
}

int validate_matrix_data(FITACF_DATA_OPTIMIZED *data) {
    if (!data) return -1;
    
    int validation_errors = 0;
    
    for (int range_idx = 0; range_idx < data->num_ranges; range_idx++) {
        int base_offset = range_idx * MAX_LAGS_PER_RANGE_OPTIMIZED;
        int lag_count = data->range_lag_counts[range_idx];
        
        for (int lag = 0; lag < lag_count; lag++) {
            int matrix_idx = base_offset + lag;
            
            /* Check for NaN values */
            if (isnan(data->phase_matrix[matrix_idx]) || 
                isnan(data->power_matrix[matrix_idx]) ||
                isnan(data->sigma_phase_matrix[matrix_idx]) ||
                isnan(data->sigma_power_matrix[matrix_idx])) {
                validation_errors++;
                if (validation_errors <= 10) {  /* Limit error output */
                    fprintf(stderr, "Warning: NaN detected at range %d, lag %d\n", 
                            range_idx, lag);
                }
            }
            
            /* Check for invalid lag indices */
            if (data->lag_idx_matrix[matrix_idx] < 0) {
                validation_errors++;
                if (validation_errors <= 10) {
                    fprintf(stderr, "Warning: Invalid lag index at range %d, lag %d\n", 
                            range_idx, lag);
                }
            }
        }
    }
    
    if (validation_errors > 0) {
        fprintf(stderr, "Matrix validation found %d errors\n", validation_errors);
        return -1;
    }
    
    return 0;
}

/* Performance monitoring functions */

void start_performance_timer(FITACF_DATA_OPTIMIZED *data) {
    if (!data) return;
    global_start_time = clock();
}

void record_phase_time(FITACF_DATA_OPTIMIZED *data, const char *phase_name) {
    if (!data) return;
    
    clock_t current_time = clock();
    double elapsed = ((double)(current_time - global_start_time)) / CLOCKS_PER_SEC;
    
    if (strcmp(phase_name, "preprocessing") == 0) {
        data->preprocessing_time = elapsed;
    } else if (strcmp(phase_name, "fitting") == 0) {
        data->fitting_time = elapsed - data->preprocessing_time;
    } else if (strcmp(phase_name, "postprocessing") == 0) {
        data->postprocessing_time = elapsed - data->preprocessing_time - data->fitting_time;
    }
    
    data->total_processing_time = elapsed;
    global_start_time = current_time;
}

void print_performance_report(FITACF_DATA_OPTIMIZED *data) {
    if (!data) return;
    
    printf("\n=== FitACF v3.0_optimized2 Performance Report ===\n");
    printf("Processing Mode: %s\n", 
           data->processing_mode == PROCESS_MODE_SEQUENTIAL ? "Sequential" :
           data->processing_mode == PROCESS_MODE_OPENMP ? "OpenMP" :
           data->processing_mode == PROCESS_MODE_CUDA ? "CUDA" : "Hybrid");
    printf("Number of Threads: %d\n", data->num_threads);
    printf("Ranges Processed: %d / %d\n", data->num_ranges, data->max_ranges);
    printf("Total Processing Time: %.3f ms\n", data->total_processing_time * 1000.0);
    printf("  - Preprocessing: %.3f ms (%.1f%%)\n", 
           data->preprocessing_time * 1000.0,
           (data->preprocessing_time / data->total_processing_time) * 100.0);
    printf("  - Fitting: %.3f ms (%.1f%%)\n", 
           data->fitting_time * 1000.0,
           (data->fitting_time / data->total_processing_time) * 100.0);
    printf("  - Postprocessing: %.3f ms (%.1f%%)\n", 
           data->postprocessing_time * 1000.0,
           (data->postprocessing_time / data->total_processing_time) * 100.0);
    printf("Memory Usage: %.2f MB\n", data->total_memory_used / (1024.0 * 1024.0));
    printf("Total Memory Allocated: %.2f MB\n", total_memory_allocated / (1024.0 * 1024.0));
    printf("Allocation Time: %.3f ms\n", total_allocation_time * 1000.0);
    
    if (data->total_processing_time > 0) {
        double ranges_per_second = data->num_ranges / data->total_processing_time;
        printf("Processing Rate: %.1f ranges/second\n", ranges_per_second);
    }
    
    printf("===============================================\n\n");
}
