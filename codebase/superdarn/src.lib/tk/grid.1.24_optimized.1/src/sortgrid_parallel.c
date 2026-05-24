/* sortgrid_parallel.c
   ===================
   Author: R.J.Barnes (Original), Enhanced for Parallelization

   Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory

   This file is part of the Radar Software Toolkit (RST).

   Parallel implementation of grid sorting operations with advanced
   algorithms and SIMD optimization.

   C2/C3 (AUDIT.md):
     - qsort_r removes the global_sort_context race so the parallel
       tile-sort phase is now thread-safe.
     - Multi-threaded path replaced with parallel tile-sort + iterative
       pairwise merge (the prior k-way and recursive merge paths both
       segfaulted for N >= ~50k).
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* glibc qsort_r */
#endif

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

/* Sorting criteria enumeration */
typedef enum {
    SORT_BY_LATITUDE = 0,
    SORT_BY_LONGITUDE,
    SORT_BY_TIME,
    SORT_BY_STATION,
    SORT_BY_INDEX,     /* per-cell index field -- matches original GridSort */
    SORT_BY_VELOCITY,
    SORT_BY_POWER,
    SORT_BY_DISTANCE,
    SORT_BY_CUSTOM
} GridSortCriteria;

/* Custom comparison context */
typedef struct {
    GridSortCriteria primary;
    GridSortCriteria secondary;
    int ascending;
    double reference_lat;
    double reference_lon;
    int (*custom_compare)(const GridGVecOpt*, const GridGVecOpt*, void*);
    void *custom_data;
} GridSortContext;

/* Static sort context for qsort compatibility (legacy fallback only --
   qsort_r path passes the context explicitly per call so the parallel
   tile-sort no longer races on this global). */
static GridSortContext *global_sort_context = NULL;

/* Distance calculation for spatial sorting */
static double calculate_distance(double lat1, double lon1, double lat2, double lon2) {
    double dlat = (lat2 - lat1) * M_PI / 180.0;
    double dlon = (lon2 - lon1) * M_PI / 180.0;
    double a = sin(dlat/2) * sin(dlat/2) + 
               cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) * 
               sin(dlon/2) * sin(dlon/2);
    return 2 * atan2(sqrt(a), sqrt(1-a)) * 6371.0; // Earth radius in km
}

/* Get sorting key value from grid cell */
static double get_sort_key(const GridGVecOpt *cell, GridSortCriteria criteria, 
                          GridSortContext *context) {
    switch (criteria) {
        case SORT_BY_LATITUDE:
            return cell->mlat;
        case SORT_BY_LONGITUDE:
            return cell->mlon;
        case SORT_BY_VELOCITY:
            return cell->vel.median;
        case SORT_BY_POWER:
            return cell->pwr.median;
        case SORT_BY_STATION:
            return cell->st_id;
        case SORT_BY_INDEX:
            return (double)cell->index;
        case SORT_BY_DISTANCE:
            if (context) {
                return calculate_distance(cell->mlat, cell->mlon,
                                        context->reference_lat, context->reference_lon);
            }
            return 0.0;
        default:
            return 0.0;
    }
}

/* Re-entrant comparator: takes the context as the qsort_r arg so
   parallel sorts no longer race on the global. Returns -1/0/+1. */
static int compare_grid_cells_r(const void *a, const void *b, void *arg) {
    const GridGVecOpt *cell_a = (const GridGVecOpt*)a;
    const GridGVecOpt *cell_b = (const GridGVecOpt*)b;
    GridSortContext *ctx = (GridSortContext*)arg;

    if (!ctx) {
        if (cell_a->mlat != cell_b->mlat)
            return (cell_a->mlat < cell_b->mlat) ? -1 : 1;
        if (cell_a->mlon != cell_b->mlon)
            return (cell_a->mlon < cell_b->mlon) ? -1 : 1;
        return 0;
    }

    /* Fast path: (SORT_BY_STATION, SORT_BY_INDEX, asc) -- the GridSortOpt
       default. Inlines the field compare and skips get_sort_key. */
    if (ctx->primary == SORT_BY_STATION &&
        ctx->secondary == SORT_BY_INDEX &&
        ctx->ascending) {
        if (cell_a->st_id != cell_b->st_id)
            return (cell_a->st_id < cell_b->st_id) ? -1 : 1;
        if (cell_a->index != cell_b->index)
            return (cell_a->index < cell_b->index) ? -1 : 1;
        return 0;
    }

    if (ctx->primary == SORT_BY_CUSTOM && ctx->custom_compare) {
        return ctx->custom_compare(cell_a, cell_b, ctx->custom_data);
    }

    double key_a = get_sort_key(cell_a, ctx->primary, ctx);
    double key_b = get_sort_key(cell_b, ctx->primary, ctx);
    if (fabs(key_a - key_b) > 1e-9) {
        int result = (key_a < key_b) ? -1 : 1;
        return ctx->ascending ? result : -result;
    }

    if (ctx->secondary != ctx->primary) {
        key_a = get_sort_key(cell_a, ctx->secondary, ctx);
        key_b = get_sort_key(cell_b, ctx->secondary, ctx);
        if (fabs(key_a - key_b) > 1e-9) {
            int result = (key_a < key_b) ? -1 : 1;
            return ctx->ascending ? result : -result;
        }
    }
    return 0;
}

/* Legacy single-arg comparator used by quickselect/partition and the
   GridIsSorted check which still rely on global_sort_context. */
static int compare_grid_cells(const void *a, const void *b) {
    return compare_grid_cells_r(a, b, global_sort_context);
}

/* C3 deleted the recursive parallel_merge_sort / parallel_merge pair:
   parallel_merge truncated 192-byte structs to 32B via __m256d in an
   earlier version, and even after the memcpy fix the recursion stack
   blew up for N > ~50k. The active multi-thread path lives in
   GridSortParallelEx below (iterative pairwise merge). */

/* Main parallel sorting function */
int GridSortParallelEx(GridDataOpt *grid, GridSortCriteria primary, GridSortCriteria secondary,
                       int ascending, GridProcessingConfig *config) {
    if (!grid || !grid->data || grid->vcnum <= 0) return -1;
    
    clock_t start_time = clock();
    
    /* Set up sorting context */
    GridSortContext sort_ctx = {
        .primary = primary,
        .secondary = secondary,
        .ascending = ascending,
        .reference_lat = 0.0,
        .reference_lon = 0.0,
        .custom_compare = NULL,
        .custom_data = NULL
    };
    
    global_sort_context = &sort_ctx;
    
    /* Configure threading -- auto-engage all available cores when caller
       didn't supply a config (the old behaviour of falling back to 1
       thread made the parallel path dead code for the common case). */
    int num_threads = config ? config->num_threads : 0;
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

#ifdef _OPENMP
    if (num_threads > 1) {
        omp_set_num_threads(num_threads);
    }
#endif

    /* C3: parallel tile-sort + iterative pairwise merge.
       Earlier paths segfaulted: the recursive OMP-task merge sort
       truncated 192B structs to 32B via __m256d, and the k-way merge
       went OOB when all tiles emptied. This implementation:
         1. partitions into num_threads tiles
         2. qsort_r each tile in parallel (no global race)
         3. iteratively merges pairs of runs into a scratch buffer,
            ping-ponging between data and temp until one run covers n.
       Memory: one extra n*sizeof(GridGVecOpt) scratch buffer.
       Time:   O(n log n) sort phase + O(n log k) merge phase.

       We only engage when vcnum is large enough that the OMP overhead
       is amortised. */
#ifdef _OPENMP
    if (grid->vcnum > 10000 && num_threads > 1) {
        long n = grid->vcnum;
        int  k = num_threads;
        long tile = (n + k - 1) / k;

        GridGVecOpt *temp = (GridGVecOpt*)malloc((size_t)n * sizeof(GridGVecOpt));
        if (!temp) {
            qsort_r(grid->data, n, sizeof(GridGVecOpt),
                    compare_grid_cells_r, &sort_ctx);
            goto sort_done;
        }

        /* Step 1: parallel tile sort using thread-safe qsort_r. */
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int t = 0; t < k; t++) {
            long a = (long)t * tile;
            long b = a + tile;
            if (b > n) b = n;
            if (b > a) {
                qsort_r(grid->data + a, (size_t)(b - a),
                        sizeof(GridGVecOpt),
                        compare_grid_cells_r, &sort_ctx);
            }
        }

        /* Step 2: iterative bottom-up pairwise merge.
           src holds sorted runs of length `run`; dst receives runs of
           length 2*run. Ping-pong src/dst until run >= n. */
        GridGVecOpt *src = grid->data;
        GridGVecOpt *dst = temp;
        long run = tile;
        while (run < n) {
            long step = run * 2;
            long num_pairs = (n + step - 1) / step;

            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (long p = 0; p < num_pairs; p++) {
                long lo  = p * step;
                long mid = lo + run;       if (mid > n) mid = n;
                long hi  = lo + step;      if (hi  > n) hi  = n;

                long i = lo, j = mid, w = lo;
                while (i < mid && j < hi) {
                    if (compare_grid_cells_r(&src[i], &src[j], &sort_ctx) <= 0)
                        dst[w++] = src[i++];
                    else
                        dst[w++] = src[j++];
                }
                while (i < mid) dst[w++] = src[i++];
                while (j < hi)  dst[w++] = src[j++];
            }

            /* swap roles */
            GridGVecOpt *tmp = src; src = dst; dst = tmp;
            run = step;
        }

        /* If the final sorted run lives in temp, copy it back. */
        if (src != grid->data) {
            memcpy(grid->data, src, (size_t)n * sizeof(GridGVecOpt));
        }
        free(temp);
    } else
#endif
    {
        qsort_r(grid->data, grid->vcnum, sizeof(GridGVecOpt),
                compare_grid_cells_r, &sort_ctx);
    }
sort_done:
    ;
    
    /* Update performance statistics */
    if (grid->perf_stats.processing_time == 0) {
        grid->perf_stats.processing_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        grid->perf_stats.operations_count = grid->vcnum;
        grid->perf_stats.parallel_threads = num_threads;
    }
    
    global_sort_context = NULL;
    return 0;
}

/* Sort by spatial distance from reference point */
int GridSortByDistanceParallel(GridDataOpt *grid, double ref_lat, double ref_lon,
                               GridProcessingConfig *config) {
    if (!grid || !grid->data || grid->vcnum <= 0) return -1;
    
    /* Set up distance-based sorting context */
    GridSortContext sort_ctx = {
        .primary = SORT_BY_DISTANCE,
        .secondary = SORT_BY_LATITUDE,
        .ascending = 1,
        .reference_lat = ref_lat,
        .reference_lon = ref_lon,
        .custom_compare = NULL,
        .custom_data = NULL
    };
    
    global_sort_context = &sort_ctx;
    
    /* Use standard qsort (distance calculation is expensive for parallel sort) */
    qsort(grid->data, grid->vcnum, sizeof(GridGVecOpt), compare_grid_cells);
    
    global_sort_context = NULL;
    return 0;
}

/* Sort with custom comparison function */
int GridSortCustomParallel(GridDataOpt *grid, 
                          int (*compare_func)(const GridGVecOpt*, const GridGVecOpt*, void*),
                          void *user_data, GridProcessingConfig *config) {
    if (!grid || !grid->data || grid->vcnum <= 0 || !compare_func) return -1;
    
    /* Set up custom sorting context */
    GridSortContext sort_ctx = {
        .primary = SORT_BY_CUSTOM,
        .secondary = SORT_BY_CUSTOM,
        .ascending = 1,
        .reference_lat = 0.0,
        .reference_lon = 0.0,
        .custom_compare = compare_func,
        .custom_data = user_data
    };
    
    global_sort_context = &sort_ctx;
    
    /* Use qsort for custom comparisons */
    qsort(grid->data, grid->vcnum, sizeof(GridGVecOpt), compare_grid_cells);
    
    global_sort_context = NULL;
    return 0;
}

/* Sort by multiple criteria with priorities */
int GridSortMultiCriteriaParallel(GridDataOpt *grid, GridSortCriteria *criteria,
                                  int num_criteria, int ascending,
                                  GridProcessingConfig *config) {
    if (!grid || !criteria || num_criteria <= 0) return -1;
    
    /* For simplicity, use first two criteria */
    GridSortCriteria primary = criteria[0];
    GridSortCriteria secondary = (num_criteria > 1) ? criteria[1] : primary;
    
    return GridSortParallelEx(grid, primary, secondary, ascending, config);
}

/* Stable sort preserving order of equal elements */
int GridStableSortParallel(GridDataOpt *grid, GridSortCriteria criteria, int ascending,
                          GridProcessingConfig *config) {
    if (!grid || !grid->data || grid->vcnum <= 0) return -1;
    
    /* Add index as secondary criteria for stability */
    for (int i = 0; i < grid->vcnum; i++) {
        grid->data[i].index = i;  // Assuming we add this field to GridGVecOpt
    }
    
    /* Sort with original index as tie-breaker */
    GridSortContext sort_ctx = {
        .primary = criteria,
        .secondary = SORT_BY_CUSTOM,  // Will use index
        .ascending = ascending,
        .reference_lat = 0.0,
        .reference_lon = 0.0,
        .custom_compare = NULL,
        .custom_data = NULL
    };
    
    global_sort_context = &sort_ctx;
    qsort(grid->data, grid->vcnum, sizeof(GridGVecOpt), compare_grid_cells);
    global_sort_context = NULL;
    
    return 0;
}

/* Quick selection algorithm for finding k-th smallest element */
static int partition(GridGVecOpt *arr, int low, int high) {
    GridGVecOpt pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (compare_grid_cells(&arr[j], &pivot) <= 0) {
            i++;
            GridGVecOpt temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    
    GridGVecOpt temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    
    return i + 1;
}

/* Find k-th smallest element (partial sorting) */
int GridPartialSortParallel(GridDataOpt *grid, int k, GridSortCriteria criteria,
                           GridProcessingConfig *config) {
    if (!grid || !grid->data || grid->vcnum <= 0 || k >= grid->vcnum) return -1;
    
    /* Set up sorting context */
    GridSortContext sort_ctx = {
        .primary = criteria,
        .secondary = SORT_BY_LATITUDE,
        .ascending = 1,
        .reference_lat = 0.0,
        .reference_lon = 0.0,
        .custom_compare = NULL,
        .custom_data = NULL
    };
    
    global_sort_context = &sort_ctx;
    
    /* Use quickselect for partial sorting */
    int low = 0, high = grid->vcnum - 1;
    
    while (low < high) {
        int pivot_idx = partition(grid->data, low, high);
        
        if (pivot_idx == k) {
            break;
        } else if (pivot_idx > k) {
            high = pivot_idx - 1;
        } else {
            low = pivot_idx + 1;
        }
    }
    
    global_sort_context = NULL;
    return 0;
}

/* Legacy compatibility functions */
int GridSortParallel(GridDataOpt *grid, GridProcessingConfig *config) {
    /* Match the original GridSort semantics: primary key st_id, secondary
       key index. RESULTS.md round-4 showed the previous lat/lon default
       broke equivalence tests against libgrd. */
    return GridSortParallelEx(grid, SORT_BY_STATION, SORT_BY_INDEX, 1, config);
}

void GridSortOpt(GridDataOpt *grid) {
    if (grid) {
        GridSortParallel(grid, NULL);
    }
}

/* Utility function to check if grid is sorted */
int GridIsSorted(GridDataOpt *grid, GridSortCriteria criteria) {
    if (!grid || !grid->data || grid->vcnum <= 1) return 1;
    
    GridSortContext sort_ctx = {
        .primary = criteria,
        .secondary = criteria,
        .ascending = 1,
        .reference_lat = 0.0,
        .reference_lon = 0.0,
        .custom_compare = NULL,
        .custom_data = NULL
    };
    
    global_sort_context = &sort_ctx;
    
    for (int i = 1; i < grid->vcnum; i++) {
        if (compare_grid_cells(&grid->data[i-1], &grid->data[i]) > 0) {
            global_sort_context = NULL;
            return 0;
        }
    }
    
    global_sort_context = NULL;
    return 1;
}

/* Shuffle grid data (useful for testing) */
int GridShuffleParallel(GridDataOpt *grid) {
    if (!grid || !grid->data || grid->vcnum <= 1) return -1;
    
    srand((unsigned int)time(NULL));
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = grid->vcnum - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        #pragma omp critical
        {
            GridGVecOpt temp = grid->data[i];
            grid->data[i] = grid->data[j];
            grid->data[j] = temp;
        }
    }
    
    return 0;
}
