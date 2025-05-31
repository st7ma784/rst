/*
 * Array-based data structures for SuperDARN FitACF v3.0
 * 
 * This header defines new data structures that replace linked lists
 * with 2D arrays/vectors to enable massive parallelization with OpenMP and CUDA.
 * 
 * Copyright (c) 2025 SuperDARN Refactoring Project
 * Author: GitHub Copilot Assistant
 */

#ifndef _FIT_STRUCTURES_ARRAY_H
#define _FIT_STRUCTURES_ARRAY_H

#include "leastsquares.h"
#include "rtypes.h"

/* Maximum expected values for array sizing */
#define MAX_RANGES 1000
#define MAX_LAGS_PER_RANGE 50
#define MAX_PULSES 32

/* Array-based phase data structure */
typedef struct phase_data_array {
    double *phi;        /* Phase values */
    double *t;          /* Time values */
    double *sigma;      /* Sigma values */
    int *lag_idx;       /* Lag indices */
    double *alpha_2;    /* Alpha-2 values */
    int count;          /* Number of valid entries */
    int capacity;       /* Allocated capacity */
} PHASE_DATA_ARRAY;

/* Array-based power data structure */
typedef struct power_data_array {
    double *ln_pwr;     /* Log power values */
    double *t;          /* Time values */
    double *sigma;      /* Sigma values */
    int *lag_idx;       /* Lag indices */
    double *alpha_2;    /* Alpha-2 values */
    int count;          /* Number of valid entries */
    int capacity;       /* Allocated capacity */
} POWER_DATA_ARRAY;

/* Array-based alpha data structure */
typedef struct alpha_data_array {
    int *lag_idx;       /* Lag indices */
    double *alpha_2;    /* Alpha-2 values */
    int count;          /* Number of valid entries */
    int capacity;       /* Allocated capacity */
} ALPHA_DATA_ARRAY;

/* Array-based elevation data structure */
typedef struct elev_data_array {
    double *elev;       /* Elevation values */
    double *t;          /* Time values */
    double *sigma;      /* Sigma values */
    int *lag_idx;       /* Lag indices */
    int count;          /* Number of valid entries */
    int capacity;       /* Allocated capacity */
} ELEV_DATA_ARRAY;

/* Array-based range node structure */
typedef struct rangenode_array {
    int range;
    double *CRI;
    double refrc_idx;
    
    /* Array-based data instead of linked lists */
    PHASE_DATA_ARRAY phases;
    POWER_DATA_ARRAY pwrs;
    ALPHA_DATA_ARRAY alpha_2;
    ELEV_DATA_ARRAY elev;
    
    /* Fit data (unchanged) */
    FITDATA *l_pwr_fit;
    FITDATA *q_pwr_fit;
    FITDATA *l_pwr_fit_err;
    FITDATA *q_pwr_fit_err;
    FITDATA *phase_fit;
    FITDATA *elev_fit;
} RANGENODE_ARRAY;

/* Master array structure for all ranges */
typedef struct range_data_arrays {
    RANGENODE_ARRAY *ranges;     /* Array of range nodes */
    int num_ranges;              /* Number of valid ranges */
    int max_ranges;              /* Maximum allocated ranges */
    
    /* 2D arrays for direct access - enables better parallelization */
    double **phase_matrix;       /* [range][lag] phase data */
    double **power_matrix;       /* [range][lag] power data */
    double **alpha_matrix;       /* [range][lag] alpha data */
    double **sigma_phase_matrix; /* [range][lag] phase sigma */
    double **sigma_power_matrix; /* [range][lag] power sigma */
    int **lag_idx_matrix;        /* [range][lag] lag indices */
    int *range_lag_counts;       /* Number of valid lags per range */
    
    /* Range validity flags for parallel processing */
    int *range_valid;            /* Boolean array: is range valid for processing */
    int *range_has_phase;        /* Boolean array: range has phase data */
    int *range_has_power;        /* Boolean array: range has power data */
} RANGE_DATA_ARRAYS;

/* Legacy compatibility structures */
typedef struct phasenode PHASENODE;  /* Keep for interface compatibility */
typedef struct pwrnode PWRNODE;      /* Keep for interface compatibility */
typedef struct lag_node LAGNODE;     /* Keep for interface compatibility */
typedef struct alpha ALPHANODE;      /* Keep for interface compatibility */

/* Enum for processing modes */
typedef enum {
    PROCESS_MODE_LINKED_LIST,   /* Use original linked list implementation */
    PROCESS_MODE_ARRAYS,        /* Use new array-based implementation */
    PROCESS_MODE_HYBRID         /* Use both for validation */
} PROCESS_MODE;

/* Function prototypes for array operations */

/* Memory management */
RANGE_DATA_ARRAYS* create_range_data_arrays(int max_ranges, int max_lags);
void free_range_data_arrays(RANGE_DATA_ARRAYS *arrays);
int resize_range_arrays(RANGE_DATA_ARRAYS *arrays, int new_max_ranges);

/* Data manipulation */
int add_phase_data(RANGE_DATA_ARRAYS *arrays, int range_idx, 
                   double phi, double t, double sigma, int lag_idx, double alpha_2);
int add_power_data(RANGE_DATA_ARRAYS *arrays, int range_idx,
                   double ln_pwr, double t, double sigma, int lag_idx, double alpha_2);
int add_alpha_data(RANGE_DATA_ARRAYS *arrays, int range_idx,
                   int lag_idx, double alpha_2);
int add_elev_data(RANGE_DATA_ARRAYS *arrays, int range_idx,
                  double elev, double t, double sigma, int lag_idx);

/* Matrix operations for parallel processing */
int populate_matrices(RANGE_DATA_ARRAYS *arrays);
int validate_matrix_data(RANGE_DATA_ARRAYS *arrays);

/* Conversion functions between linked lists and arrays */
int convert_llist_to_arrays(llist range_list, RANGE_DATA_ARRAYS *arrays);
int convert_arrays_to_llist(RANGE_DATA_ARRAYS *arrays, llist *range_list);

/* Parallel processing utilities */
int mark_valid_ranges(RANGE_DATA_ARRAYS *arrays, double noise_threshold);
int count_valid_ranges(RANGE_DATA_ARRAYS *arrays);
int get_range_batch(RANGE_DATA_ARRAYS *arrays, int start_idx, int batch_size, int *range_indices);

/* OpenMP-ready function prototypes */
void parallel_power_fitting(RANGE_DATA_ARRAYS *arrays, int num_threads);
void parallel_phase_fitting(RANGE_DATA_ARRAYS *arrays, int num_threads);
void parallel_preprocessing(RANGE_DATA_ARRAYS *arrays, int num_threads);

/* CUDA-ready function prototypes (when CUDA support is added) */
#ifdef ENABLE_CUDA
void cuda_power_fitting(RANGE_DATA_ARRAYS *arrays);
void cuda_phase_fitting(RANGE_DATA_ARRAYS *arrays);
#endif

/* Statistics and validation */
typedef struct array_stats {
    int total_ranges;
    int valid_ranges;
    int total_phase_points;
    int total_power_points;
    double avg_lags_per_range;
    double memory_usage_mb;
    double conversion_time_ms;
} ARRAY_STATS;

ARRAY_STATS calculate_array_stats(RANGE_DATA_ARRAYS *arrays);
int compare_llist_vs_arrays(llist range_list, RANGE_DATA_ARRAYS *arrays, double tolerance);

/* Enhanced fit parameters structure for array mode */
typedef struct fit_prms_array {
    /* Original parameters */
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
    int *lag[2];
    int *pulse;
    double *pwr0;
    double **acfd;
    double **xcfd;
    int maxbeam;
    double bmoff;
    double bmsep;
    double interfer[3];
    double phidiff;
    double tdiff;
    double vdir;
    struct {
        short yr;
        short mo;
        short dy;
        short hr;
        short mt;
        short sc;
        int us;
    } time;
    
    /* Array mode extensions */
    PROCESS_MODE mode;
    int num_threads;              /* For OpenMP */
    int enable_cuda;              /* CUDA flag */
    double noise_threshold;       /* For range validation */
    int batch_size;               /* For parallel processing */
    
} FITPRMS_ARRAY;

#endif /* _FIT_STRUCTURES_ARRAY_H */
