/*
 * Optimized array-based data structures for SuperDARN FitACF v3.0_optimized2
 * 
 * This header defines high-performance data structures that completely eliminate
 * linked lists in favor of contiguous memory arrays for maximum parallelization
 * with OpenMP and CUDA. Features include:
 * - SIMD-aligned memory layouts
 * - Memory pooling for zero-allocation processing
 * - Cache-optimized data structures
 * - Vectorized processing patterns
 * 
 * Copyright (c) 2025 SuperDARN RST Optimization Project
 * Author: GitHub Copilot Assistant
 */

#ifndef _FIT_STRUCTURES_OPTIMIZED_H
#define _FIT_STRUCTURES_OPTIMIZED_H

#include "leastsquares.h"
#include "rtypes.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/* Configuration constants for maximum performance */
#define MAX_RANGES_OPTIMIZED 1000
#define MAX_LAGS_PER_RANGE_OPTIMIZED 50
#define MAX_PULSES_OPTIMIZED 32
#define CACHE_LINE_SIZE 64
#define SIMD_ALIGNMENT 32
#define MEMORY_POOL_SIZE (256 * 1024 * 1024)  /* 256MB pre-allocated pool */

/* Processing modes for different optimization levels */
typedef enum {
    PROCESS_MODE_SEQUENTIAL = 0,
    PROCESS_MODE_OPENMP = 1,
    PROCESS_MODE_CUDA = 2,
    PROCESS_MODE_HYBRID = 3
} PROCESS_MODE;

/* Memory alignment macros */
#define ALIGNED_MALLOC(size) _aligned_malloc(size, SIMD_ALIGNMENT)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)

/* Cache-aligned vectorized data arrays */
typedef struct vectorized_data_array {
    /* Core data arrays - SIMD aligned for vectorization */
    double *values __attribute__((aligned(SIMD_ALIGNMENT)));     /* Primary values (phi, ln_pwr, elev, etc.) */
    double *t_values __attribute__((aligned(SIMD_ALIGNMENT)));   /* Time values */
    double *sigma_values __attribute__((aligned(SIMD_ALIGNMENT))); /* Sigma values */
    double *alpha_2_values __attribute__((aligned(SIMD_ALIGNMENT))); /* Alpha-2 values */
    int *lag_indices __attribute__((aligned(SIMD_ALIGNMENT)));   /* Lag indices */
    
    /* Metadata */
    int count;          /* Number of valid entries */
    int capacity;       /* Allocated capacity */
    int stride;         /* Memory stride for cache optimization */
} VECTORIZED_DATA_ARRAY;

/* Optimized range node with contiguous memory layout */
typedef struct rangenode_optimized {
    /* Range metadata */
    int range;
    double refrc_idx;
    
    /* Vectorized data arrays for each data type */
    VECTORIZED_DATA_ARRAY phases;
    VECTORIZED_DATA_ARRAY powers;
    VECTORIZED_DATA_ARRAY elevations;
    VECTORIZED_DATA_ARRAY alphas;
    
    /* CRI data as contiguous array */
    double *CRI __attribute__((aligned(SIMD_ALIGNMENT)));
    int CRI_count;
    
    /* Fit results - pre-allocated structures */
    FITDATA l_pwr_fit;
    FITDATA q_pwr_fit;
    FITDATA l_pwr_fit_err;
    FITDATA q_pwr_fit_err;
    FITDATA phase_fit;
    FITDATA elev_fit;
    
    /* Processing flags for parallelization */
    uint32_t processing_flags;  /* Bitfield for various processing states */
    
} RANGENODE_OPTIMIZED;

/* Master optimized structure for entire dataset */
typedef struct fitacf_data_optimized {
    /* Primary range data array - contiguous allocation */
    RANGENODE_OPTIMIZED *ranges __attribute__((aligned(CACHE_LINE_SIZE)));
    int num_ranges;
    int max_ranges;
    
    /* Flattened matrices for massive parallelization */
    /* All matrices are [max_ranges * MAX_LAGS_PER_RANGE_OPTIMIZED] flat arrays */
    double *phase_matrix __attribute__((aligned(SIMD_ALIGNMENT)));      /* Flattened phase data */
    double *power_matrix __attribute__((aligned(SIMD_ALIGNMENT)));      /* Flattened power data */
    double *alpha_matrix __attribute__((aligned(SIMD_ALIGNMENT)));      /* Flattened alpha data */
    double *elev_matrix __attribute__((aligned(SIMD_ALIGNMENT)));       /* Flattened elevation data */
    double *sigma_phase_matrix __attribute__((aligned(SIMD_ALIGNMENT))); /* Flattened phase sigma */
    double *sigma_power_matrix __attribute__((aligned(SIMD_ALIGNMENT))); /* Flattened power sigma */
    double *t_matrix __attribute__((aligned(SIMD_ALIGNMENT)));          /* Flattened time data */
    int *lag_idx_matrix __attribute__((aligned(SIMD_ALIGNMENT)));       /* Flattened lag indices */
    
    /* Range metadata arrays for vectorized operations */
    int *range_lag_counts __attribute__((aligned(SIMD_ALIGNMENT)));     /* Valid lags per range */
    uint8_t *range_flags __attribute__((aligned(SIMD_ALIGNMENT)));      /* Processing flags per range */
    double *range_noise_levels __attribute__((aligned(SIMD_ALIGNMENT))); /* Noise level per range */
    
    /* Memory pool for zero-allocation processing */
    char *memory_pool;
    size_t memory_pool_size;
    size_t memory_pool_offset;
    
    /* Processing configuration */
    PROCESS_MODE processing_mode;
    int num_threads;
    int cuda_device_id;
    
    /* Performance monitoring */
    double total_processing_time;
    double preprocessing_time;
    double fitting_time;
    double postprocessing_time;
    size_t total_memory_used;
    
} FITACF_DATA_OPTIMIZED;

/* Processing flags bitfield definitions */
#define RANGE_FLAG_VALID           0x01
#define RANGE_FLAG_HAS_PHASE       0x02
#define RANGE_FLAG_HAS_POWER       0x04
#define RANGE_FLAG_HAS_ELEVATION   0x08
#define RANGE_FLAG_NOISE_FILTERED  0x10
#define RANGE_FLAG_TX_OVERLAP      0x20
#define RANGE_FLAG_FITTED          0x40
#define RANGE_FLAG_ERROR           0x80

/* Optimized fitting parameters structure */
typedef struct fitprms_optimized {
    /* Basic radar parameters */
    int channel;
    int offset;
    int cp;
    int xcf;
    int tfreq;
    float noise;
    int nrang;
    int smsep;
    int nave;
    int mplgs;
    int mpinc;
    int txpl;
    int lagfr;
    int mppul;
    int bmnum;
    int old;
    
    /* Array parameters - pre-allocated for performance */
    int lag[2][MAX_LAGS_PER_RANGE_OPTIMIZED];
    int pulse[MAX_PULSES_OPTIMIZED];
    double pwr0[MAX_RANGES_OPTIMIZED];
    
    /* ACF/XCF data as flat arrays for vectorization */
    double *acfd_flat __attribute__((aligned(SIMD_ALIGNMENT)));
    double *xcfd_flat __attribute__((aligned(SIMD_ALIGNMENT)));
    
    /* Processing parameters */
    int maxbeam;
    double bmoff;
    double bmsep;
    double interfer[3];
    double phidiff;
    double tdiff;
    double vdir;
    
    /* Time structure */
    struct {
        short yr;
        short mo;
        short dy;
        short hr;
        short mt;
        short sc;
        int us;
    } time;
    
    /* Optimization parameters */
    PROCESS_MODE mode;
    int num_threads;
    int enable_cuda;
    double noise_threshold;
    int batch_size;
    int enable_vectorization;
    int memory_pool_enabled;
    
} FITPRMS_OPTIMIZED;

/* Function declarations for optimized processing */

/* Memory management functions */
FITACF_DATA_OPTIMIZED* create_fitacf_data_optimized(int max_ranges);
void destroy_fitacf_data_optimized(FITACF_DATA_OPTIMIZED *data);
int initialize_memory_pool(FITACF_DATA_OPTIMIZED *data, size_t pool_size);
void* allocate_from_pool(FITACF_DATA_OPTIMIZED *data, size_t size);
void reset_memory_pool(FITACF_DATA_OPTIMIZED *data);

/* Data structure management */
int allocate_vectorized_array(VECTORIZED_DATA_ARRAY *array, int capacity);
void free_vectorized_array(VECTORIZED_DATA_ARRAY *array);
int resize_vectorized_array(VECTORIZED_DATA_ARRAY *array, int new_capacity);
int add_data_to_vectorized_array(VECTORIZED_DATA_ARRAY *array, double value, 
                                 double t, double sigma, double alpha_2, int lag_idx);

/* Range node management */
int initialize_range_optimized(RANGENODE_OPTIMIZED *range, int range_num);
void free_range_optimized(RANGENODE_OPTIMIZED *range);
int add_phase_data_optimized(RANGENODE_OPTIMIZED *range, double phi, double t, 
                            double sigma, int lag_idx, double alpha_2);
int add_power_data_optimized(RANGENODE_OPTIMIZED *range, double ln_pwr, double t, 
                            double sigma, int lag_idx, double alpha_2);
int add_elevation_data_optimized(RANGENODE_OPTIMIZED *range, double elev, double t, 
                                double sigma, int lag_idx);

/* Matrix operations for parallel processing */
int populate_flat_matrices(FITACF_DATA_OPTIMIZED *data);
int validate_matrix_data(FITACF_DATA_OPTIMIZED *data);
int flatten_range_data(FITACF_DATA_OPTIMIZED *data);

/* Parallel processing utilities */
int set_processing_mode(FITACF_DATA_OPTIMIZED *data, PROCESS_MODE mode);
int configure_openmp_threads(FITACF_DATA_OPTIMIZED *data, int num_threads);
int initialize_cuda_context(FITACF_DATA_OPTIMIZED *data, int device_id);

/* Performance monitoring */
void start_performance_timer(FITACF_DATA_OPTIMIZED *data);
void record_phase_time(FITACF_DATA_OPTIMIZED *data, const char *phase_name);
void print_performance_report(FITACF_DATA_OPTIMIZED *data);

/* CUDA kernel declarations (if CUDA enabled) */
#ifdef __CUDACC__
__global__ void cuda_phase_fitting_kernel(double *phase_matrix, double *result_matrix, 
                                         int num_ranges, int max_lags);
__global__ void cuda_power_fitting_kernel(double *power_matrix, double *result_matrix, 
                                         int num_ranges, int max_lags);
__global__ void cuda_noise_filtering_kernel(double *data_matrix, double *noise_levels, 
                                           uint8_t *flags, int num_ranges, int max_lags);
#endif

/* Inline utility functions for performance */
static inline int get_matrix_index(int range, int lag, int max_lags) {
    return range * max_lags + lag;
}

static inline void set_range_flag(FITACF_DATA_OPTIMIZED *data, int range, uint8_t flag) {
    data->range_flags[range] |= flag;
}

static inline int has_range_flag(FITACF_DATA_OPTIMIZED *data, int range, uint8_t flag) {
    return (data->range_flags[range] & flag) != 0;
}

static inline void clear_range_flag(FITACF_DATA_OPTIMIZED *data, int range, uint8_t flag) {
    data->range_flags[range] &= ~flag;
}

#endif /* _FIT_STRUCTURES_OPTIMIZED_H */
