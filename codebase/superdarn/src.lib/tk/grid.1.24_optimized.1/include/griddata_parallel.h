/* griddata_parallel.h
   ===================
   Author: R.J.Barnes (Original), Enhanced for Parallelization
   
   Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory
   
   This file is part of the Radar Software Toolkit (RST).
   
   RST is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.
   
   Modifications:
   - Added optimized data structures for CUDA/OpenMP parallelization
   - Replaced linked lists with 2D matrix operations
   - Enhanced memory alignment for SIMD operations
   - Added parallel processing function prototypes
*/

#ifndef _GRIDDATA_PARALLEL_H
#define _GRIDDATA_PARALLEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>      /* for M_PI when _GNU_SOURCE is set */
#include <zlib.h>      /* needed by dmap.h for gzFile */
#include "rtypes.h"    /* int16/int32/uint16/... used by dmap.h */
#include "dmap.h"      /* real DataMap definition */
#include "gridindex.h" /* real GridIndex definition */

/* Parallel processing configuration */
#ifndef MAX_GRID_CELLS
#define MAX_GRID_CELLS 65536
#endif

#ifndef MAX_STATIONS
#define MAX_STATIONS 256
#endif

#ifndef GRID_CHUNK_SIZE
#define GRID_CHUNK_SIZE 1024
#endif

/* Memory alignment for SIMD operations */
#ifdef __GNUC__
#define ALIGNED(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#define ALIGNED(x) __declspec(align(x))
#else
#define ALIGNED(x)
#endif

/* OpenMP and CUDA configuration */
#ifdef _OPENMP
#include <omp.h>
#define PARALLEL_FOR _Pragma("omp parallel for")
#define PARALLEL_FOR_REDUCTION(op, var) _Pragma("omp parallel for reduction(" #op ":" #var ")")
#define PARALLEL_SECTION _Pragma("omp parallel sections")
#else
#define PARALLEL_FOR
#define PARALLEL_FOR_REDUCTION(op, var)
#define PARALLEL_SECTION
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define CUDA_KERNEL __global__
#define CUDA_DEVICE __device__
#else
#define CUDA_CALLABLE
#define CUDA_KERNEL
#define CUDA_DEVICE
#endif

/* Enhanced statistics structure with vectorization support */
struct GridStats {
    double mean;
    double median;
    double weight;
    double sd;
    double min;
    double max;
    uint32_t count;
    uint32_t samples;
    double sum;
    double sum_sq;
} ALIGNED(32);

/* Optimized station vector structure */
struct GridSVecOpt {
    int32_t st_id;
    int32_t chn;
    int32_t npnt;
    double freq0;
    char major_revision;
    char minor_revision;
    int32_t prog_id;
    char gsct;
    
    struct GridStats noise;
    struct GridStats vel;
    struct GridStats pwr;
    struct GridStats wdt;
    struct GridStats verr;
    
    /* Padding for memory alignment */
    char _padding[16];
} ALIGNED(64);

/* Optimized grid vector structure.
   AUDIT C1: dropped ALIGNED(128) on outer struct, ALIGNED(32) on inner
   vel/pwr/wdt, the unused .flags field, the unused inner .weight and
   .samples fields, and the trailing _padding[32]. Result:
       sizeof(GridGVecOpt) == 88 bytes  (target: <= 1.2 * sizeof(GridGVec))
   Original libgrd GridGVec is ~88 bytes, so we are now layout-parity. */
struct GridGVecOpt {
    double mlat, mlon;
    double azm;

    struct {
        double median;
        double sd;
    } vel;

    struct {
        double median;
        double sd;
    } pwr;

    struct {
        double median;
        double sd;
    } wdt;

    int32_t st_id;
    int32_t chn;
    int32_t index;
};

/* Matrix-based grid data structure for parallel processing */
struct GridMatrix {
    uint32_t rows;
    uint32_t cols;
    uint32_t allocated_size;
    double *data;           /* Flattened 2D matrix */
    uint32_t *indices;      /* Cell indices matrix */
    uint32_t *counts;       /* Element count per cell */
    bool is_gpu_allocated;  /* CUDA memory flag */
} ALIGNED(64);

/* Enhanced grid data structure */
struct GridDataOpt {
    double st_time;
    double ed_time;
    int32_t stnum;
    int32_t vcnum;
    uint32_t max_cells;
    unsigned char xtd;
    
    /* Original data structures */
    struct GridSVecOpt *sdata;
    struct GridGVecOpt *data;
    
    /* Enhanced parallel processing structures */
    struct GridMatrix *velocity_matrix;
    struct GridMatrix *power_matrix;
    struct GridMatrix *width_matrix;
    struct GridMatrix *azimuth_matrix;
    
    /* Spatial indexing for fast lookup */
    uint32_t *spatial_index;
    uint32_t spatial_grid_size;
    
    /* Performance monitoring */
    struct {
        double processing_time;
        uint64_t operations_count;
        uint32_t parallel_threads;
        bool use_gpu;
    } perf_stats;

    /* Opaque pointers for parallel runtime, memory pool, and metrics
       handles. void* keeps the header light; the implementation casts
       to its own struct types. */
    void *parallel_ctx;
    void *memory_pool;
    void *metrics;

} ALIGNED(128);

/* Performance optimization structure */
struct GridProcessingConfig {
    uint32_t num_threads;
    uint32_t chunk_size;
    bool use_simd;
    bool use_gpu;
    bool enable_caching;
    double error_threshold[3];  /* vel, pwr, wdt */
    uint32_t max_iterations;
} ALIGNED(64);

/* Typedef aliases so source files can use bare names (the .c files were
   written against typedef'd names; add the typedefs here in one place). */
typedef struct GridStats GridStats;
typedef struct GridSVecOpt GridSVecOpt;
typedef struct GridGVecOpt GridGVecOpt;
typedef struct GridMatrix GridMatrix;
typedef struct GridDataOpt GridDataOpt;
typedef struct GridProcessingConfig GridProcessingConfig;

/* Function prototypes - Original API compatibility */
CUDA_CALLABLE struct GridDataOpt *GridMakeOpt();
CUDA_CALLABLE void GridFreeOpt(struct GridDataOpt *ptr);
CUDA_CALLABLE int GridLocateCellOpt(int npnt, struct GridGVecOpt *ptr, int index);
CUDA_CALLABLE void GridMergeOpt(struct GridDataOpt *mptr, struct GridDataOpt *ptr);
CUDA_CALLABLE void GridAverageOpt(struct GridDataOpt *mptr, struct GridDataOpt *ptr, int flg);
CUDA_CALLABLE void GridCopyOpt(struct GridDataOpt *a, struct GridDataOpt *b);
CUDA_CALLABLE void GridAddOpt(struct GridDataOpt *a, struct GridDataOpt *b, int recnum);
CUDA_CALLABLE void GridSortOpt(struct GridDataOpt *ptr);
CUDA_CALLABLE void GridIntegrateOpt(struct GridDataOpt *a, struct GridDataOpt *b, double *err);

/* Enhanced parallel processing functions */
CUDA_CALLABLE struct GridDataOpt *GridMakeParallel(uint32_t max_cells, struct GridProcessingConfig *config);
CUDA_CALLABLE void GridFreeParallel(struct GridDataOpt *ptr);

/* Matrix-based parallel operations */
CUDA_CALLABLE int GridMergeParallel(struct GridDataOpt *mptr, struct GridDataOpt *ptr, struct GridProcessingConfig *config);
CUDA_CALLABLE int GridAverageParallel(struct GridDataOpt *mptr, struct GridDataOpt *ptr, int flg, struct GridProcessingConfig *config);
CUDA_CALLABLE int GridIntegrateParallel(struct GridDataOpt *a, struct GridDataOpt *b, double *err, struct GridProcessingConfig *config);

/* Optimized search and indexing */
CUDA_CALLABLE int GridLocateCellParallel(struct GridDataOpt *grid, int index);
CUDA_CALLABLE int GridBuildSpatialIndex(struct GridDataOpt *grid);
CUDA_CALLABLE int GridSortParallel(struct GridDataOpt *ptr, struct GridProcessingConfig *config);

/* Memory management for parallel processing */
CUDA_CALLABLE int GridAllocateMatrices(struct GridDataOpt *grid, uint32_t max_cells);
CUDA_CALLABLE void GridDeallocateMatrices(struct GridDataOpt *grid);

/* GPU-specific functions */
#ifdef __CUDACC__
CUDA_KERNEL void GridMergeKernel(struct GridGVecOpt *input, struct GridGVecOpt *output, 
                                uint32_t *indices, uint32_t num_elements);
CUDA_KERNEL void GridAverageKernel(double *input_matrix, double *output_matrix,
                                  uint32_t *counts, uint32_t rows, uint32_t cols);
CUDA_KERNEL void GridIntegrateKernel(double *vel_matrix, double *pwr_matrix, double *wdt_matrix,
                                    double *errors, uint32_t num_elements);

/* CUDA memory management */
CUDA_CALLABLE int GridCopyToGPU(struct GridDataOpt *grid);
CUDA_CALLABLE int GridCopyFromGPU(struct GridDataOpt *grid);
CUDA_CALLABLE void GridFreeGPU(struct GridDataOpt *grid);
#endif

/* Vectorized mathematical operations */
CUDA_CALLABLE void GridVectorizedAdd(double *a, double *b, double *result, uint32_t size);
CUDA_CALLABLE void GridVectorizedMultiply(double *a, double *b, double *result, uint32_t size);
CUDA_CALLABLE void GridVectorizedDivide(double *a, double *b, double *result, uint32_t size);

/* Statistical operations optimized for parallel processing */
CUDA_CALLABLE double GridParallelMean(double *data, uint32_t size);
CUDA_CALLABLE double GridParallelStdDev(double *data, uint32_t size, double mean);
CUDA_CALLABLE void GridParallelMinMax(double *data, uint32_t size, double *min, double *max);

/* Linear regression for parallel merge operations */
CUDA_CALLABLE void GridLinRegParallel(struct GridGVecOpt **data, uint32_t num, double *vpar, double *vper);

/* Performance measurement and debugging */
CUDA_CALLABLE void GridStartTiming(struct GridDataOpt *grid);
CUDA_CALLABLE void GridEndTiming(struct GridDataOpt *grid);
CUDA_CALLABLE void GridPrintPerformanceStats(struct GridDataOpt *grid);

/* Configuration and initialization */
CUDA_CALLABLE struct GridProcessingConfig *GridCreateConfig();
CUDA_CALLABLE void GridDestroyConfig(struct GridProcessingConfig *config);
CUDA_CALLABLE int GridSetOptimalThreads(struct GridProcessingConfig *config);

/* Data validation and error checking */
CUDA_CALLABLE int GridValidateData(struct GridDataOpt *grid);
CUDA_CALLABLE int GridCheckMemoryAlignment(struct GridDataOpt *grid);

/* Memory management utilities */
void *aligned_malloc(size_t size, size_t alignment);
void aligned_free(void *ptr);

/* Grid merge modes */
#define GRID_MERGE_AVERAGE 0
#define GRID_MERGE_PREFER_FIRST 1
#define GRID_MERGE_PREFER_SECOND 2
#define GRID_MERGE_PREFER_HIGHER_POWER 3
#define GRID_MERGE_PREFER_LOWER_ERROR 4

/* Performance statistics structure -- broad superset of every field the
   parallel sources reference. Keeps counters and timers in one block so
   adding new probes doesn't require schema migrations. */
struct GridPerformanceStats {
    double processing_time;
    double memory_usage;
    int operations_count;
    int error_count;
    int cache_hits;
    int cache_misses;
    /* I/O counters */
    int read_errors;
    int write_errors;
    int grids_read;
    int grids_written;
    int cells_read;
    int cells_written;
    int stations_read;
    int stations_written;
    int indices_loaded;
    int index_entries;
    double read_time;
    double write_time;
    double index_load_time;
    /* Seek/index counters */
    int total_seeks;
    int sequential_seeks;
    int index_seeks;
    double total_seek_time;
    int cell_locates;
    int locate_hits;
    int locate_misses;
    double total_locate_time;
    int index_calculations;
    double total_index_time;
    int batch_locates;
    int batch_locate_hits;
    int batch_locate_misses;
    double batch_locate_time;
    /* Aggregation counters */
    int grid_merges;
    int grid_averages;
    int grid_integrations;
    int cells_merged;
    int cells_averaged;
    int merge_conflicts;
    double merge_time;
    double integration_time;
    /* Aggregate timing */
    double total_time;
    double average_time;
} ALIGNED(64);

/* Parallel grid vector structure */
struct GridGVecParallel {
    double mlat, mlon;
    double azm;
    double kvect;
    struct GridStats vel;
    struct GridStats pwr;
    struct GridStats wdt;
    int32_t st_id;
    int32_t chn;
    int32_t index;
    uint32_t quality_flag;
    uint32_t filter_flags;
    uint32_t avg_count;
    uint32_t merge_count;
} ALIGNED(64);

/* Parallel grid data structure */
struct GridDataParallel {
    struct {
        int yr, mo, dy, hr, mt, sc, us;
    } st_time, ed_time;
    
    int stnum;
    int vcnum;
    int xtd;
    
    struct GridSVecOpt *sdata;
    struct GridGVecParallel *data;
    
    /* Enhanced parallel processing structures */
    struct GridMatrix *velocity_matrix;
    struct GridMatrix *power_matrix;
    struct GridMatrix *width_matrix;
    struct GridMatrix *azimuth_matrix;
    
    /* Spatial indexing for fast lookup */
    uint32_t *spatial_index;
    uint32_t spatial_grid_size;
    
    /* Performance monitoring */
    struct GridPerformanceStats perf_stats;
} ALIGNED(128);

/* Enhanced grid index structure with parallel optimization */
struct GridIndexParallel {
    int num;
    double *tme;
    int *inx;
    int cache_valid;
    double last_search_time;
    int last_search_index;
} ALIGNED(64);

/* Grid integration parameters */
struct GridIntegrationParams {
    int prefer_quality;
    int apply_post_filter;
    double quality_threshold;
    int max_grids;
    int temporal_weighting;
} ALIGNED(32);

/* Grid filter parameters.
   Used by gridmerge_parallel.c (post-integration outlier filtering) and
   by the test_filtergrid_parallel.c test harness. Fields cover all
   filter modes that the test exercises: statistical outlier detection,
   percentile clipping, spatial smoothing, and median filtering. */
struct GridFilterParams {
    /* Per-channel enables. 0 = disabled, 1 = enabled. */
    int filter_velocity;
    int filter_power;
    int filter_width;

    /* Statistical outlier detection. */
    int outlier_detection;
    double outlier_threshold;       /* z-score cut-off in standard deviations. */

    /* Percentile-based clipping (0-100 inclusive). */
    double velocity_percentile_low;
    double velocity_percentile_high;
    double power_percentile_low;
    double power_percentile_high;
    double width_percentile_low;
    double width_percentile_high;

    /* Spatial / median smoothing parameters. */
    double spatial_radius;          /* degrees */
    double gaussian_sigma;          /* in cells */
    int    median_window_size;      /* must be odd */
} ALIGNED(32);
typedef struct GridFilterParams GridFilterParams;

/* Parallel grid processing functions */

/* Grid seeking and indexing functions */
double grid_parallel_get_time(struct DataMap *ptr);
int grid_parallel_seek(int fid, int yr, int mo, int dy, int hr, int mt, int sc,
                      double *atme, struct GridIndexParallel *inx,
                      struct GridPerformanceStats *stats);
int grid_parallel_fseek(FILE *fp, int yr, int mo, int dy, int hr, int mt, int sc,
                       double *atme, struct GridIndexParallel *inx,
                       struct GridPerformanceStats *stats);
int grid_parallel_locate_cell(int npnt, struct GridGVecParallel *ptr, 
                             int index, struct GridPerformanceStats *stats);
struct GridIndexParallel *grid_parallel_index_create(struct GridIndex *orig_inx);
void grid_parallel_index_free(struct GridIndexParallel *inx);
int grid_parallel_index_cell(struct GridDataParallel *grd, double mlat, double mlon,
                            struct GridPerformanceStats *stats);
int grid_parallel_locate_cells_batch(int npnt, struct GridGVecParallel *ptr,
                                    int *indices, int num_indices, int *results,
                                    struct GridPerformanceStats *stats);

/* Grid merging and integration functions */
int grid_parallel_average(struct GridDataParallel **src_grids, int num_grids,
                         struct GridDataParallel *dst_grid, int flags,
                         struct GridPerformanceStats *stats);
int grid_parallel_merge(struct GridDataParallel *grid1, 
                       struct GridDataParallel *grid2,
                       struct GridDataParallel *merged_grid,
                       int merge_mode,
                       struct GridPerformanceStats *stats);
int grid_parallel_integrate(struct GridDataParallel **grids, int num_grids,
                           struct GridDataParallel *integrated_grid,
                           struct GridIntegrationParams *params,
                           struct GridPerformanceStats *stats);

/* I/O functions */
int grid_parallel_read(int fid, struct GridDataParallel *grd,
                      struct GridPerformanceStats *stats);
int grid_parallel_fread(FILE *fp, struct GridDataParallel *grd,
                       struct GridPerformanceStats *stats);
int grid_parallel_write(int fid, struct GridDataParallel *grd,
                       struct GridPerformanceStats *stats);
int grid_parallel_fwrite(FILE *fp, struct GridDataParallel *grd,
                        struct GridPerformanceStats *stats);
struct GridIndexParallel *grid_parallel_load_index(int fid,
                                                   struct GridPerformanceStats *stats);
struct GridIndexParallel *grid_parallel_fload_index(FILE *fp,
                                                    struct GridPerformanceStats *stats);

#ifdef __cplusplus
}
#endif

#endif /* _GRIDDATA_PARALLEL_H */
