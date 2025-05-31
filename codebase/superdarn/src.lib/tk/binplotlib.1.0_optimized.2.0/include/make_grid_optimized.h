/* make_grid_optimized.h
   =====================
   Author: R.J.Barnes (Original)
   Optimized by: SuperDARN Optimization Framework
   
   Optimized version of make_grid.h with:
   - OpenMP parallelization support
   - SIMD vectorization capabilities
   - Batch processing for large grids
   - Memory optimization structures
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
- Added OpenMP support for parallel grid generation
- Added SIMD vectorization for coordinate calculations
- Added batch processing capabilities
- Added memory optimization structures
*/

#ifndef _MAKEGRID_OPTIMIZED_H
#define _MAKEGRID_OPTIMIZED_H

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Memory-aligned grid point structure */
typedef struct {
    float lat;
    float lon;
} __attribute__((aligned(8))) GridPoint;

/* Original API - maintained for compatibility */
struct PolygonData *make_grid(float lonspc, float latspc, int max);

/* Optimized API - enhanced performance functions */
struct PolygonData *make_grid_optimized(float lonspc, float latspc, int max);

struct PolygonData *make_grid_batch(float lonspc, float latspc, int max, 
                                   int batch_size);

/* SIMD-optimized grid point calculation */
#ifdef __AVX2__
void simd_generate_grid_points_avx2(float lat_start, float lon_start,
                                   float lat_step, float lon_step,
                                   GridPoint *points, int count);
#endif

#endif /* _MAKEGRID_OPTIMIZED_H */
