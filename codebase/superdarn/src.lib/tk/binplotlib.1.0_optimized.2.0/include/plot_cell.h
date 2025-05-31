/* plot_cell.h
   ===========
   Author: R.J.Barnes (Original)
   Optimized by: SuperDARN Optimization Framework
*/


/*
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

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Modifications:
- Added optimization structures and function prototypes
- Added performance statistics tracking
- Added SIMD and OpenMP optimization support
*/


#ifndef _PLOTCELL_H
#define _PLOTCELL_H

#include <stddef.h>  /* For size_t */
#include <time.h>    /* For clock_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Optimization constants */
#define PLOT_SIMD_WIDTH 8           /* AVX2 SIMD width for float operations */
#define PLOT_MIN_PARALLEL_SIZE 32   /* Minimum size for parallel processing */

/* Performance statistics structure */
typedef struct {
    size_t total_cells_processed;   /* Total number of cells processed */
    size_t vectorized_operations;   /* Number of SIMD operations performed */
    size_t parallel_operations;     /* Number of parallel operations performed */
    double processing_time;         /* Total processing time in seconds */
} PlotOptimizationStats;

/* Batch processing structures for SIMD optimization */
typedef struct {
    float lat[PLOT_SIMD_WIDTH];
    float lon[PLOT_SIMD_WIDTH];
} CoordBatch;

typedef struct {
    float x[PLOT_SIMD_WIDTH];
    float y[PLOT_SIMD_WIDTH];
} PointBatch;

/* Function prototypes for statistics */
void plot_optimization_stats_init(PlotOptimizationStats *stats);
void plot_optimization_stats_print(const PlotOptimizationStats *stats);
void plot_optimization_stats_reset(PlotOptimizationStats *stats);

/* Original function prototypes (for backward compatibility) */
void plot_field_cell(struct Plot *plot,struct RadarBeam *sbm,
                     struct GeoLocBeam *gbm,float latmin,int magflg,
                     float xoff,float yoff,float wdt,float hgt,
                     int (*trnf)(int,void *,int,void *,void *data),void *data,
                     unsigned int(*cfn)(double,void *),void *cdata,
                     int prm,unsigned int gscol,unsigned char gsflg);

void plot_grid_cell(struct Plot *plot,struct GridData *ptr,float latmin,int magflg,
                    float xoff,float yoff,float wdt,float hgt,
                    int (*trnf)(int,void *,int,void *,void *data),void *data,
                    unsigned int(*cfn)(double,void *),void *cdata,int cprm,
                    int old_aacgm);

/* Optimized function prototypes */
void plot_field_cell_optimized(struct Plot *plot,struct RadarBeam *sbm,
                               struct GeoLocBeam *gbm,float latmin,int magflg,
                               float xoff,float yoff,float wdt,float hgt,
                               int (*trnf)(int,void *,int,void *,void *data),void *data,
                               unsigned int(*cfn)(double,void *),void *cdata,
                               int prm,unsigned int gscol,unsigned char gsflg,
                               PlotOptimizationStats *stats);

void plot_grid_cell_optimized(struct Plot *plot,struct GridData *ptr,float latmin,int magflg,
                              float xoff,float yoff,float wdt,float hgt,
                              int (*trnf)(int,void *,int,void *,void *data),void *data,
                              unsigned int(*cfn)(double,void *),void *cdata,int cprm,
                              int old_aacgm, PlotOptimizationStats *stats);

/* Batch processing functions */
int plot_batch_coordinate_transform(const CoordBatch *coords, PointBatch *points,
                                   float xoff, float yoff, float wdt, float hgt,
                                   int (*trnf)(int,void *,int,void *,void *data),
                                   void *data, int count);

void plot_batch_polygons(struct Plot *plot, const PointBatch *points,
                        const unsigned int *colors, int count);

/* Utility function for coordinate conversion (original) */
int cell_convert(float xoff,float yoff,float wdt,float hgt,
                 float lat,float lon,float *px,float *py,int magflg,
                 int (*trnf)(int,void *,int,void *,void *data),
                 void *data, int old_aacgm);

/* Performance benchmarking functions */
double plot_benchmark_coordinate_transform(int iterations, int data_size);
double plot_benchmark_polygon_rendering(int iterations, int polygon_count);

#ifdef __cplusplus
}
#endif

#endif
