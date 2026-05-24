/* filtergrid_parallel.c
   =====================
   Author: R.J.Barnes (Original), Enhanced for Parallelization
   
   Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory
   
   This file is part of the Radar Software Toolkit (RST).
   
   Parallel implementation of grid filtering operations with advanced
   signal processing and statistical filtering techniques.
   
   Key Optimizations:
   - Parallel filter application with SIMD acceleration
   - Multi-criteria filtering with logical operations
   - Statistical outlier detection and removal
   - Spatial filtering with kernel operations
   - Real-time adaptive filtering parameters
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "rtypes.h"
#include "rfile.h"
#include "griddata_parallel.h"

/* Filter operation types */
typedef enum {
    FILTER_KEEP = 0,
    FILTER_REMOVE,
    FILTER_MODIFY
} FilterAction;

/* Statistical filter types */
typedef enum {
    STAT_FILTER_MEDIAN = 0,
    STAT_FILTER_MEAN,
    STAT_FILTER_SIGMA,
    STAT_FILTER_PERCENTILE,
    STAT_FILTER_INTERQUARTILE
} StatisticalFilterType;

/* Spatial filter kernels */
typedef enum {
    SPATIAL_KERNEL_GAUSSIAN = 0,
    SPATIAL_KERNEL_BOXCAR,
    SPATIAL_KERNEL_TRIANGLE,
    SPATIAL_KERNEL_HANN,
    SPATIAL_KERNEL_CUSTOM
} SpatialKernelType;

/* Filter criteria structure */
typedef struct {
    double velocity_min, velocity_max;
    double power_min, power_max;
    double width_min, width_max;
    double latitude_min, latitude_max;
    double longitude_min, longitude_max;
    int station_id;
    double error_threshold;
    int use_logical_and;  /* 1 for AND, 0 for OR */
} GridFilterCriteria;

/* Statistical filter parameters */
typedef struct {
    StatisticalFilterType type;
    double threshold_sigma;
    double percentile_lower;
    double percentile_upper;
    int window_size;
    int iterations;
} StatisticalFilterParams;

/* Spatial filter parameters */
typedef struct {
    SpatialKernelType kernel_type;
    double kernel_radius;
    double *custom_kernel;
    int kernel_size;
    double smoothing_factor;
} SpatialFilterParams;

/* Basic range filter for individual cell */
static FilterAction apply_basic_filter(const GridGVecOpt *cell, const GridFilterCriteria *criteria) {
    if (!cell || !criteria) return FILTER_KEEP;
    
    int velocity_ok = (cell->vel.median >= criteria->velocity_min && 
                      cell->vel.median <= criteria->velocity_max);
    int power_ok = (cell->pwr.median >= criteria->power_min && 
                   cell->pwr.median <= criteria->power_max);
    int width_ok = (cell->wdt.median >= criteria->width_min && 
                   cell->wdt.median <= criteria->width_max);
    int latitude_ok = (cell->mlat >= criteria->latitude_min && 
                      cell->mlat <= criteria->latitude_max);
    int longitude_ok = (cell->mlon >= criteria->longitude_min && 
                       cell->mlon <= criteria->longitude_max);
    int station_ok = (criteria->station_id < 0 || cell->st_id == criteria->station_id);
    int error_ok = (cell->vel.sd <= criteria->error_threshold);
    
    int passes_filter;
    if (criteria->use_logical_and) {
        passes_filter = velocity_ok && power_ok && width_ok && 
                       latitude_ok && longitude_ok && station_ok && error_ok;
    } else {
        passes_filter = velocity_ok || power_ok || width_ok || 
                       latitude_ok || longitude_ok || station_ok || error_ok;
    }
    
    return passes_filter ? FILTER_KEEP : FILTER_REMOVE;
}

/* Parallel basic filtering with multiple criteria */
int GridFilterBasicParallel(GridDataOpt *grid, const GridFilterCriteria *criteria,
                           GridProcessingConfig *config) {
    if (!grid || !criteria || !grid->data || grid->vcnum <= 0) return -1;
    
    clock_t start_time = clock();
    
    /* Configure threading */
    int num_threads = config ? config->num_threads : 1;
#ifdef _OPENMP
    if (num_threads > 1) {
        omp_set_num_threads(num_threads);
    }
#endif
    
    /* Create temporary array for filtered results */
    GridGVecOpt *filtered_data = (GridGVecOpt*)malloc(grid->vcnum * sizeof(GridGVecOpt));
    if (!filtered_data) return -1;
    
    int filtered_count = 0;
    int *keep_flags = (int*)calloc(grid->vcnum, sizeof(int));
    if (!keep_flags) {
        free(filtered_data);
        return -1;
    }
    
    /* Parallel filtering pass */
#ifdef __AVX2__
    /* AVX2 batched: 4 cells per iteration. Each scalar field is
       gathered across 4 cells via _mm256_set_pd, range checks run as
       parallel _mm256_cmp_pd, combined per criteria->use_logical_and,
       and extracted to a 4-bit mask via _mm256_movemask_pd. station_id
       (int32) and the tail are handled scalar. apply_basic_filter
       remains the source of truth for the tail. */
    const __m256d vmin = _mm256_set1_pd(criteria->velocity_min);
    const __m256d vmax = _mm256_set1_pd(criteria->velocity_max);
    const __m256d pmin = _mm256_set1_pd(criteria->power_min);
    const __m256d pmax = _mm256_set1_pd(criteria->power_max);
    const __m256d wmin = _mm256_set1_pd(criteria->width_min);
    const __m256d wmax = _mm256_set1_pd(criteria->width_max);
    const __m256d latmin = _mm256_set1_pd(criteria->latitude_min);
    const __m256d latmax = _mm256_set1_pd(criteria->latitude_max);
    const __m256d lonmin = _mm256_set1_pd(criteria->longitude_min);
    const __m256d lonmax = _mm256_set1_pd(criteria->longitude_max);
    const __m256d errmax = _mm256_set1_pd(criteria->error_threshold);
    const int use_and = criteria->use_logical_and;
    const int st_id_filter = criteria->station_id;

    const int simd_end = (grid->vcnum / 4) * 4;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < simd_end; i += 4) {
        const GridGVecOpt *c0 = &grid->data[i];
        const GridGVecOpt *c1 = &grid->data[i+1];
        const GridGVecOpt *c2 = &grid->data[i+2];
        const GridGVecOpt *c3 = &grid->data[i+3];

        __m256d v_vel  = _mm256_set_pd(c3->vel.median, c2->vel.median, c1->vel.median, c0->vel.median);
        __m256d v_pwr  = _mm256_set_pd(c3->pwr.median, c2->pwr.median, c1->pwr.median, c0->pwr.median);
        __m256d v_wdt  = _mm256_set_pd(c3->wdt.median, c2->wdt.median, c1->wdt.median, c0->wdt.median);
        __m256d v_mlat = _mm256_set_pd(c3->mlat,       c2->mlat,       c1->mlat,       c0->mlat);
        __m256d v_mlon = _mm256_set_pd(c3->mlon,       c2->mlon,       c1->mlon,       c0->mlon);
        __m256d v_sd   = _mm256_set_pd(c3->vel.sd,     c2->vel.sd,     c1->vel.sd,     c0->vel.sd);

        __m256d vel_ok = _mm256_and_pd(_mm256_cmp_pd(v_vel,  vmin,   _CMP_GE_OQ),
                                       _mm256_cmp_pd(v_vel,  vmax,   _CMP_LE_OQ));
        __m256d pwr_ok = _mm256_and_pd(_mm256_cmp_pd(v_pwr,  pmin,   _CMP_GE_OQ),
                                       _mm256_cmp_pd(v_pwr,  pmax,   _CMP_LE_OQ));
        __m256d wdt_ok = _mm256_and_pd(_mm256_cmp_pd(v_wdt,  wmin,   _CMP_GE_OQ),
                                       _mm256_cmp_pd(v_wdt,  wmax,   _CMP_LE_OQ));
        __m256d lat_ok = _mm256_and_pd(_mm256_cmp_pd(v_mlat, latmin, _CMP_GE_OQ),
                                       _mm256_cmp_pd(v_mlat, latmax, _CMP_LE_OQ));
        __m256d lon_ok = _mm256_and_pd(_mm256_cmp_pd(v_mlon, lonmin, _CMP_GE_OQ),
                                       _mm256_cmp_pd(v_mlon, lonmax, _CMP_LE_OQ));
        __m256d err_ok = _mm256_cmp_pd(v_sd, errmax, _CMP_LE_OQ);

        __m256d combined;
        if (use_and) {
            combined = _mm256_and_pd(vel_ok, pwr_ok);
            combined = _mm256_and_pd(combined, wdt_ok);
            combined = _mm256_and_pd(combined, lat_ok);
            combined = _mm256_and_pd(combined, lon_ok);
            combined = _mm256_and_pd(combined, err_ok);
        } else {
            combined = _mm256_or_pd(vel_ok, pwr_ok);
            combined = _mm256_or_pd(combined, wdt_ok);
            combined = _mm256_or_pd(combined, lat_ok);
            combined = _mm256_or_pd(combined, lon_ok);
            combined = _mm256_or_pd(combined, err_ok);
        }

        int mask = _mm256_movemask_pd(combined);

        int st0_ok = (st_id_filter < 0 || c0->st_id == st_id_filter);
        int st1_ok = (st_id_filter < 0 || c1->st_id == st_id_filter);
        int st2_ok = (st_id_filter < 0 || c2->st_id == st_id_filter);
        int st3_ok = (st_id_filter < 0 || c3->st_id == st_id_filter);

        if (use_and) {
            keep_flags[i]   = ((mask >> 0) & 1) & st0_ok;
            keep_flags[i+1] = ((mask >> 1) & 1) & st1_ok;
            keep_flags[i+2] = ((mask >> 2) & 1) & st2_ok;
            keep_flags[i+3] = ((mask >> 3) & 1) & st3_ok;
        } else {
            keep_flags[i]   = ((mask >> 0) & 1) | st0_ok;
            keep_flags[i+1] = ((mask >> 1) & 1) | st1_ok;
            keep_flags[i+2] = ((mask >> 2) & 1) | st2_ok;
            keep_flags[i+3] = ((mask >> 3) & 1) | st3_ok;
        }
    }

    for (int i = simd_end; i < grid->vcnum; i++) {
        FilterAction action = apply_basic_filter(&grid->data[i], criteria);
        keep_flags[i] = (action == FILTER_KEEP) ? 1 : 0;
    }
#else
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < grid->vcnum; i++) {
        FilterAction action = apply_basic_filter(&grid->data[i], criteria);
        keep_flags[i] = (action == FILTER_KEEP) ? 1 : 0;
    }
#endif
    
    /* Sequential copy of filtered cells */
    for (int i = 0; i < grid->vcnum; i++) {
        if (keep_flags[i]) {
            filtered_data[filtered_count++] = grid->data[i];
        }
    }
    
    /* Replace original data */
    free(grid->data);
    grid->data = filtered_data;
    grid->vcnum = filtered_count;
    
    /* Resize to actual filtered size */
    if (filtered_count > 0) {
        grid->data = (GridGVecOpt*)realloc(grid->data, filtered_count * sizeof(GridGVecOpt));
    }
    
    free(keep_flags);
    
    /* Update performance statistics */
    if (grid->perf_stats.processing_time == 0) {
        grid->perf_stats.processing_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        grid->perf_stats.operations_count = grid->vcnum;
        grid->perf_stats.parallel_threads = num_threads;
    }
    
    return 0;
}

/* Calculate statistical parameters for outlier detection */
static void calculate_statistics(const GridDataOpt *grid, const char *parameter,
                                double *mean, double *median, double *std_dev) {
    if (!grid || !grid->data || grid->vcnum <= 0) return;
    
    double *values = (double*)malloc(grid->vcnum * sizeof(double));
    if (!values) return;
    
    /* Extract parameter values */
    for (int i = 0; i < grid->vcnum; i++) {
        if (strcmp(parameter, "velocity") == 0) {
            values[i] = grid->data[i].vel.median;
        } else if (strcmp(parameter, "power") == 0) {
            values[i] = grid->data[i].pwr.median;
        } else if (strcmp(parameter, "width") == 0) {
            values[i] = grid->data[i].wdt.median;
        } else {
            values[i] = 0.0;
        }
    }
    
    /* Calculate mean */
    double sum = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum)
#endif
    for (int i = 0; i < grid->vcnum; i++) {
        sum += values[i];
    }
    *mean = sum / grid->vcnum;
    
    /* Calculate standard deviation */
    double variance = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:variance)
#endif
    for (int i = 0; i < grid->vcnum; i++) {
        double diff = values[i] - *mean;
        variance += diff * diff;
    }
    *std_dev = sqrt(variance / grid->vcnum);
    
    /* Calculate median (requires sorting) */
    qsort(values, grid->vcnum, sizeof(double), 
          (int(*)(const void*, const void*))strcmp);
    
    if (grid->vcnum % 2 == 0) {
        *median = (values[grid->vcnum/2 - 1] + values[grid->vcnum/2]) / 2.0;
    } else {
        *median = values[grid->vcnum/2];
    }
    
    free(values);
}

/* Statistical outlier filtering */
int GridFilterStatisticalParallel(GridDataOpt *grid, const char *parameter,
                                 const StatisticalFilterParams *params,
                                 GridProcessingConfig *config) {
    if (!grid || !parameter || !params || !grid->data || grid->vcnum <= 0) return -1;
    
    double mean, median, std_dev;
    calculate_statistics(grid, parameter, &mean, &median, &std_dev);
    
    /* Set up filter criteria based on statistical analysis */
    GridFilterCriteria criteria = {0};
    
    if (strcmp(parameter, "velocity") == 0) {
        switch (params->type) {
            case STAT_FILTER_SIGMA:
                criteria.velocity_min = mean - params->threshold_sigma * std_dev;
                criteria.velocity_max = mean + params->threshold_sigma * std_dev;
                break;
            case STAT_FILTER_MEDIAN:
                criteria.velocity_min = median - params->threshold_sigma * std_dev;
                criteria.velocity_max = median + params->threshold_sigma * std_dev;
                break;
            default:
                criteria.velocity_min = -1e6;
                criteria.velocity_max = 1e6;
        }
        criteria.power_min = -1e6;
        criteria.power_max = 1e6;
        criteria.width_min = -1e6;
        criteria.width_max = 1e6;
    } else if (strcmp(parameter, "power") == 0) {
        criteria.velocity_min = -1e6;
        criteria.velocity_max = 1e6;
        switch (params->type) {
            case STAT_FILTER_SIGMA:
                criteria.power_min = mean - params->threshold_sigma * std_dev;
                criteria.power_max = mean + params->threshold_sigma * std_dev;
                break;
            case STAT_FILTER_MEDIAN:
                criteria.power_min = median - params->threshold_sigma * std_dev;
                criteria.power_max = median + params->threshold_sigma * std_dev;
                break;
            default:
                criteria.power_min = -1e6;
                criteria.power_max = 1e6;
        }
        criteria.width_min = -1e6;
        criteria.width_max = 1e6;
    }
    
    criteria.latitude_min = -90.0;
    criteria.latitude_max = 90.0;
    criteria.longitude_min = -180.0;
    criteria.longitude_max = 360.0;
    criteria.station_id = -1;
    criteria.error_threshold = 1e6;
    criteria.use_logical_and = 1;
    
    /* Apply iterative filtering */
    for (int iter = 0; iter < params->iterations; iter++) {
        int status = GridFilterBasicParallel(grid, &criteria, config);
        if (status != 0) return status;
        
        /* Recalculate statistics for next iteration */
        if (iter < params->iterations - 1) {
            calculate_statistics(grid, parameter, &mean, &median, &std_dev);
        }
    }
    
    return 0;
}

/* Spatial filtering with kernel operations */
static double gaussian_kernel(double distance, double radius) {
    return exp(-(distance * distance) / (2.0 * radius * radius));
}

static double boxcar_kernel(double distance, double radius) {
    return (distance <= radius) ? 1.0 : 0.0;
}

static double triangle_kernel(double distance, double radius) {
    return (distance <= radius) ? (1.0 - distance / radius) : 0.0;
}

static double get_kernel_weight(double distance, const SpatialFilterParams *params) {
    switch (params->kernel_type) {
        case SPATIAL_KERNEL_GAUSSIAN:
            return gaussian_kernel(distance, params->kernel_radius);
        case SPATIAL_KERNEL_BOXCAR:
            return boxcar_kernel(distance, params->kernel_radius);
        case SPATIAL_KERNEL_TRIANGLE:
            return triangle_kernel(distance, params->kernel_radius);
        case SPATIAL_KERNEL_HANN:
            if (distance <= params->kernel_radius) {
                double x = M_PI * distance / params->kernel_radius;
                return 0.5 * (1.0 + cos(x));
            }
            return 0.0;
        default:
            return 1.0;
    }
}

/* Calculate distance between two lat/lon points */
static double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double dlat = (lat2 - lat1) * M_PI / 180.0;
    double dlon = (lon2 - lon1) * M_PI / 180.0;
    double a = sin(dlat/2) * sin(dlat/2) + 
               cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) * 
               sin(dlon/2) * sin(dlon/2);
    return 2 * atan2(sqrt(a), sqrt(1-a)) * 6371.0; // Earth radius in km
}

/* Spatial smoothing filter */
int GridFilterSpatialParallel(GridDataOpt *grid, const SpatialFilterParams *params,
                             GridProcessingConfig *config) {
    if (!grid || !params || !grid->data || grid->vcnum <= 0) return -1;
    
    clock_t start_time = clock();
    
    /* Create smoothed copy of the data */
    GridGVecOpt *smoothed_data = (GridGVecOpt*)malloc(grid->vcnum * sizeof(GridGVecOpt));
    if (!smoothed_data) return -1;
    
    /* Copy original data structure */
    memcpy(smoothed_data, grid->data, grid->vcnum * sizeof(GridGVecOpt));
    
    /* Configure threading */
    int num_threads = config ? config->num_threads : 1;
#ifdef _OPENMP
    if (num_threads > 1) {
        omp_set_num_threads(num_threads);
    }
#endif
    
    /* Apply spatial filtering */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < grid->vcnum; i++) {
        double lat_i = grid->data[i].mlat;
        double lon_i = grid->data[i].mlon;
        
        double weighted_vel = 0.0, weighted_pwr = 0.0, weighted_wdt = 0.0;
        double total_weight = 0.0;
        
        /* Find neighboring cells within kernel radius */
        for (int j = 0; j < grid->vcnum; j++) {
            double lat_j = grid->data[j].mlat;
            double lon_j = grid->data[j].mlon;
            
            double distance = haversine_distance(lat_i, lon_i, lat_j, lon_j);
            
            if (distance <= params->kernel_radius) {
                double weight = get_kernel_weight(distance, params);
                
                weighted_vel += grid->data[j].vel.median * weight;
                weighted_pwr += grid->data[j].pwr.median * weight;
                weighted_wdt += grid->data[j].wdt.median * weight;
                total_weight += weight;
            }
        }
        
        /* Apply smoothing with mixing factor */
        if (total_weight > 0) {
            double smooth_vel = weighted_vel / total_weight;
            double smooth_pwr = weighted_pwr / total_weight;
            double smooth_wdt = weighted_wdt / total_weight;
            
            smoothed_data[i].vel.median = grid->data[i].vel.median * (1.0 - params->smoothing_factor) +
                                         smooth_vel * params->smoothing_factor;
            smoothed_data[i].pwr.median = grid->data[i].pwr.median * (1.0 - params->smoothing_factor) +
                                         smooth_pwr * params->smoothing_factor;
            smoothed_data[i].wdt.median = grid->data[i].wdt.median * (1.0 - params->smoothing_factor) +
                                         smooth_wdt * params->smoothing_factor;
        }
    }
    
    /* Replace original data with smoothed data */
    free(grid->data);
    grid->data = smoothed_data;
    
    /* Update performance statistics */
    if (grid->perf_stats.processing_time == 0) {
        grid->perf_stats.processing_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        grid->perf_stats.operations_count = grid->vcnum;
        grid->perf_stats.parallel_threads = num_threads;
    }
    
    return 0;
}

/* Median filter for noise reduction */
int GridFilterMedianParallel(GridDataOpt *grid, int window_size, const char *parameter,
                            GridProcessingConfig *config) {
    if (!grid || !parameter || !grid->data || grid->vcnum <= 0 || window_size <= 0) return -1;
    
    /* Simple median filter on sorted data */
    GridGVecOpt *filtered_data = (GridGVecOpt*)malloc(grid->vcnum * sizeof(GridGVecOpt));
    if (!filtered_data) return -1;
    
    memcpy(filtered_data, grid->data, grid->vcnum * sizeof(GridGVecOpt));
    
    int half_window = window_size / 2;
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = half_window; i < grid->vcnum - half_window; i++) {
        double *window_values = (double*)malloc(window_size * sizeof(double));
        
        /* Extract values in window */
        for (int j = 0; j < window_size; j++) {
            int idx = i - half_window + j;
            if (strcmp(parameter, "velocity") == 0) {
                window_values[j] = grid->data[idx].vel.median;
            } else if (strcmp(parameter, "power") == 0) {
                window_values[j] = grid->data[idx].pwr.median;
            } else if (strcmp(parameter, "width") == 0) {
                window_values[j] = grid->data[idx].wdt.median;
            }
        }
        
        /* Sort window and find median */
        qsort(window_values, window_size, sizeof(double), 
              (int(*)(const void*, const void*))strcmp);
        
        double median_value = window_values[window_size / 2];
        
        /* Apply median filter */
        if (strcmp(parameter, "velocity") == 0) {
            filtered_data[i].vel.median = median_value;
        } else if (strcmp(parameter, "power") == 0) {
            filtered_data[i].pwr.median = median_value;
        } else if (strcmp(parameter, "width") == 0) {
            filtered_data[i].wdt.median = median_value;
        }
        
        free(window_values);
    }
    
    /* Replace original data */
    free(grid->data);
    grid->data = filtered_data;
    
    return 0;
}

/* Composite filter combining multiple filtering techniques */
int GridFilterCompositeParallel(GridDataOpt *grid, const GridFilterCriteria *basic_criteria,
                               const StatisticalFilterParams *stat_params,
                               const SpatialFilterParams *spatial_params,
                               GridProcessingConfig *config) {
    if (!grid) return -1;
    
    /* Apply basic filtering first */
    if (basic_criteria) {
        int status = GridFilterBasicParallel(grid, basic_criteria, config);
        if (status != 0) return status;
    }
    
    /* Apply statistical filtering */
    if (stat_params) {
        int status = GridFilterStatisticalParallel(grid, "velocity", stat_params, config);
        if (status != 0) return status;
    }
    
    /* Apply spatial filtering last */
    if (spatial_params) {
        int status = GridFilterSpatialParallel(grid, spatial_params, config);
        if (status != 0) return status;
    }
    
    return 0;
}

/* Legacy compatibility functions */
int GridFilterParallel(GridDataOpt *grid, double vel_min, double vel_max,
                      double pwr_min, double pwr_max, GridProcessingConfig *config) {
    GridFilterCriteria criteria = {
        .velocity_min = vel_min,
        .velocity_max = vel_max,
        .power_min = pwr_min,
        .power_max = pwr_max,
        .width_min = -1e6,
        .width_max = 1e6,
        .latitude_min = -90.0,
        .latitude_max = 90.0,
        .longitude_min = -180.0,
        .longitude_max = 360.0,
        .station_id = -1,
        .error_threshold = 1e6,
        .use_logical_and = 1
    };
    
    return GridFilterBasicParallel(grid, &criteria, config);
}

/* Simple quality filter */
int GridFilterQualityParallel(GridDataOpt *grid, double error_threshold,
                             GridProcessingConfig *config) {
    GridFilterCriteria criteria = {
        .velocity_min = -1e6,
        .velocity_max = 1e6,
        .power_min = -1e6,
        .power_max = 1e6,
        .width_min = -1e6,
        .width_max = 1e6,
        .latitude_min = -90.0,
        .latitude_max = 90.0,
        .longitude_min = -180.0,
        .longitude_max = 360.0,
        .station_id = -1,
        .error_threshold = error_threshold,
        .use_logical_and = 1
    };
    
    return GridFilterBasicParallel(grid, &criteria, config);
}
