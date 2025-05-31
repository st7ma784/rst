/* make_fov_optimized.h
   ====================
   Author: R.J.Barnes (Original)
   Optimized by: SuperDARN Optimization Framework
   
   Optimized version of make_fov.h with:
   - OpenMP parallelization support
   - SIMD vectorization capabilities
   - Batch processing for multiple radars
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
- Added OpenMP support for parallel radar processing
- Added SIMD vectorization for position calculations
- Added batch processing capabilities
- Added memory optimization structures
*/

#ifndef _MAKEFOV_OPTIMIZED_H
#define _MAKEFOV_OPTIMIZED_H

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Memory-aligned position structure for SIMD operations */
typedef struct {
    double rho[8] __attribute__((aligned(64)));
    double lat[8] __attribute__((aligned(64)));
    double lon[8] __attribute__((aligned(64)));
} PositionBatch;

/* Original API - maintained for compatibility */
struct PolygonData *make_fov(double tval, struct RadarNetwork *network,
                             float alt, int chisham);

struct PolygonData *make_field_fov(double tval, struct RadarNetwork *network,
                                   int id, int chisham);

/* Optimized API - enhanced performance functions */
struct PolygonData *make_fov_optimized(double tval, struct RadarNetwork *network,
                                      float alt, int chisham);

struct PolygonData **make_fov_batch(double *tvals, int time_count,
                                   struct RadarNetwork *network,
                                   float alt, int chisham);

/* SIMD-optimized position calculation */
#ifdef __AVX__
void simd_calculate_positions_avx(int beam, const int *ranges, int range_count,
                                 struct RadarSite *site, int frang, int rsep,
                                 float recrise, float alt, PositionBatch *positions,
                                 int chisham);
#endif

#endif /* _MAKEFOV_OPTIMIZED_H */
