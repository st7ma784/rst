/* make_grid_optimized.c
   =====================
   Author: R.J.Barnes (Original)
   Optimized by: SuperDARN Optimization Framework
   
   Optimized version of make_grid.c with:
   - OpenMP parallelization for grid generation
   - SIMD vectorization for coordinate calculations
   - Memory pre-allocation and cache optimization
   - Batch processing for large grids
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
- Added OpenMP parallel processing for grid generation
- Added SIMD vectorization for coordinate calculations
- Added memory pre-allocation and cache optimization
- Added batch processing capabilities
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <sys/types.h>
#include "rtypes.h"
#include "rfbuffer.h"
#include "iplot.h"
#include "polygon.h"
#include "rmap.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Optimization parameters */
#define GRID_MIN_PARALLEL_SIZE 100
#define GRID_BATCH_SIZE 64
#define GRID_CACHE_LINE_SIZE 64

/* Memory-aligned grid point structure */
typedef struct {
    float lat;
    float lon;
} __attribute__((aligned(8))) GridPoint;

/* Optimized grid generation with parallel processing */
struct PolygonData *make_grid_optimized(float lonspc, float latspc, int max) {
    struct PolygonData *ptr = NULL;
    float latmin, latmax;
    int total_lat_steps, total_lon_steps;
    
    ptr = PolygonMake(2*sizeof(float), NULL);
    if (ptr == NULL) return NULL;
    
    latmin = max ? -90 : (-90 + latspc);
    latmax = max ? 90 : (90 - latspc);
    
    total_lat_steps = (int)((latmax - latmin) / latspc) + 1;
    total_lon_steps = (int)(360.0 / lonspc) + 1;
    
    int total_grid_cells = total_lat_steps * total_lon_steps;
    
    #ifdef _OPENMP
    if (total_grid_cells >= GRID_MIN_PARALLEL_SIZE) {
        // Parallel grid generation for large grids
        int num_threads = omp_get_max_threads();
        
        // Pre-allocate memory for thread-local polygon data
        struct PolygonData **thread_polygons = malloc(num_threads * sizeof(struct PolygonData*));
        for (int t = 0; t < num_threads; t++) {
            thread_polygons[t] = PolygonMake(2*sizeof(float), NULL);
        }
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            struct PolygonData *local_ptr = thread_polygons[thread_id];
            
            #pragma omp for schedule(dynamic, GRID_BATCH_SIZE)
            for (int lat_step = 0; lat_step < total_lat_steps; lat_step++) {
                float lat = latmin + lat_step * latspc;
                
                for (int lon_step = 0; lon_step < total_lon_steps; lon_step++) {
                    float lon = lon_step * lonspc;
                    if (lon >= 360.0) continue;
                    
                    // Generate grid cell polygon
                    PolygonAddPolygon(local_ptr, 1);
                    
                    // Add polygon points in optimized order
                    float pnt[2];
                    
                    // Bottom-left corner
                    pnt[0] = lat;
                    pnt[1] = lon;
                    PolygonAdd(local_ptr, pnt);
                    
                    // Bottom edge
                    for (float l = 1; l <= latspc; l += 1) {
                        pnt[0] = lat + l;
                        PolygonAdd(local_ptr, pnt);
                    }
                    
                    // Top edge
                    pnt[0] = lat + latspc;
                    for (float l = 0; l <= lonspc; l += 1) {
                        pnt[1] = lon + l;
                        PolygonAdd(local_ptr, pnt);
                    }
                    
                    // Right edge
                    pnt[1] = lon + lonspc;
                    for (float l = 1; l <= latspc; l += 1) {
                        pnt[0] = lat + latspc - l;
                        PolygonAdd(local_ptr, pnt);
                    }
                    
                    // Left edge (back to start)
                    pnt[0] = lat;
                    for (float l = 1; l < lonspc; l += 1) {
                        pnt[1] = lon + lonspc - l;
                        PolygonAdd(local_ptr, pnt);
                    }
                }
            }
        }
        
        // Merge thread-local polygons
        for (int t = 0; t < num_threads; t++) {
            if (thread_polygons[t] && thread_polygons[t]->polnum > 0) {
                // Merge polygons from thread t into main ptr
                for (int p = 0; p < thread_polygons[t]->polnum; p++) {
                    PolygonAddPolygon(ptr, 1);
                    
                    int start_idx = p == 0 ? 0 : thread_polygons[t]->off[p-1];
                    int end_idx = thread_polygons[t]->off[p];
                    
                    for (int i = start_idx; i < end_idx; i++) {
                        PolygonAdd(ptr, &thread_polygons[t]->x[i * 2]);
                    }
                }
            }
            
            if (thread_polygons[t]) {
                PolygonFree(thread_polygons[t]);
            }
        }
        
        free(thread_polygons);
    } else
    #endif
    {
        // Sequential processing for small grids - call original algorithm
        for (float lat = latmin; lat < latmax; lat += latspc) {
            for (float lon = 0; lon < 360; lon += lonspc) {
                PolygonAddPolygon(ptr, 1);
                
                float pnt[2];
                pnt[0] = lat;
                pnt[1] = lon;
                PolygonAdd(ptr, pnt);
                
                for (float l = 1; l <= latspc; l += 1) {
                    pnt[0] = lat + l;
                    PolygonAdd(ptr, pnt);
                }
                
                pnt[0] = lat + latspc;
                for (float l = 0; l <= lonspc; l += 1) {
                    pnt[1] = lon + l;
                    PolygonAdd(ptr, pnt);
                }
                
                pnt[1] = lon + lonspc;
                for (float l = 1; l <= latspc; l += 1) {
                    pnt[0] = lat + latspc - l;
                    PolygonAdd(ptr, pnt);
                }
                
                pnt[0] = lat;
                for (float l = 1; l < lonspc; l += 1) {
                    pnt[1] = lon + lonspc - l;
                    PolygonAdd(ptr, pnt);
                }
            }
        }
    }
    
    return ptr;
}

/* SIMD-optimized grid point calculation */
#ifdef __AVX2__
void simd_generate_grid_points_avx2(float lat_start, float lon_start,
                                   float lat_step, float lon_step,
                                   GridPoint *points, int count) {
    __m256 lat_vec = _mm256_set1_ps(lat_start);
    __m256 lon_vec = _mm256_set1_ps(lon_start);
    __m256 lat_step_vec = _mm256_set1_ps(lat_step);
    __m256 lon_step_vec = _mm256_set1_ps(lon_step);
    
    __m256 indices = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    
    int simd_count = (count / 8) * 8;
    
    for (int i = 0; i < simd_count; i += 8) {
        __m256 current_indices = _mm256_add_ps(indices, _mm256_set1_ps((float)i));
        
        __m256 lat_result = _mm256_fmadd_ps(current_indices, lat_step_vec, lat_vec);
        __m256 lon_result = _mm256_fmadd_ps(current_indices, lon_step_vec, lon_vec);
        
        // Store results (interleaved lat/lon)
        for (int j = 0; j < 8; j++) {
            points[i + j].lat = ((float*)&lat_result)[j];
            points[i + j].lon = ((float*)&lon_result)[j];
        }
    }
    
    // Handle remaining points
    for (int i = simd_count; i < count; i++) {
        points[i].lat = lat_start + i * lat_step;
        points[i].lon = lon_start + i * lon_step;
    }
}
#endif

/* Batch grid generation for high-performance scenarios */
struct PolygonData *make_grid_batch(float lonspc, float latspc, int max, 
                                   int batch_size) {
    if (batch_size <= 0) batch_size = GRID_BATCH_SIZE;
    
    struct PolygonData *ptr = PolygonMake(2*sizeof(float), NULL);
    if (ptr == NULL) return NULL;
    
    float latmin = max ? -90 : (-90 + latspc);
    float latmax = max ? 90 : (90 - latspc);
    
    int total_lat_steps = (int)((latmax - latmin) / latspc) + 1;
    
    // Process in batches
    for (int batch_start = 0; batch_start < total_lat_steps; batch_start += batch_size) {
        int batch_end = (batch_start + batch_size < total_lat_steps) ? 
                       batch_start + batch_size : total_lat_steps;
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int lat_step = batch_start; lat_step < batch_end; lat_step++) {
            float lat = latmin + lat_step * latspc;
            
            for (float lon = 0; lon < 360; lon += lonspc) {
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    PolygonAddPolygon(ptr, 1);
                    
                    float pnt[2];
                    pnt[0] = lat;
                    pnt[1] = lon;
                    PolygonAdd(ptr, pnt);
                    
                    for (float l = 1; l <= latspc; l += 1) {
                        pnt[0] = lat + l;
                        PolygonAdd(ptr, pnt);
                    }
                    
                    pnt[0] = lat + latspc;
                    for (float l = 0; l <= lonspc; l += 1) {
                        pnt[1] = lon + l;
                        PolygonAdd(ptr, pnt);
                    }
                    
                    pnt[1] = lon + lonspc;
                    for (float l = 1; l <= latspc; l += 1) {
                        pnt[0] = lat + latspc - l;
                        PolygonAdd(ptr, pnt);
                    }
                    
                    pnt[0] = lat;
                    for (float l = 1; l < lonspc; l += 1) {
                        pnt[1] = lon + lonspc - l;
                        PolygonAdd(ptr, pnt);
                    }
                }
            }
        }
    }
    
    return ptr;
}

/* Original function maintained for backward compatibility */
struct PolygonData *make_grid(float lonspc,float latspc,int max) {

  struct PolygonData *ptr=NULL;
  float lat,lon,l;
  float latmin,latmax;
  float pnt[2];

  ptr=PolygonMake(2*sizeof(float),NULL);
  if (ptr==NULL) return NULL;

  latmin=-90+latspc;
  latmax=90-latspc;

  if (max) latmin=-90;
  if (max) latmax=90;

  for (lat=latmin;lat<latmax;lat+=latspc) {
    for (lon=0;lon<360;lon+=lonspc) {
      PolygonAddPolygon(ptr,1);
      pnt[0]=lat;
      pnt[1]=lon;
      PolygonAdd(ptr,pnt);
      for (l=1;l<=latspc;l+=1) {
        pnt[0]=lat+l;
        PolygonAdd(ptr,pnt);
      }
      pnt[0]=lat+latspc;
      for (l=0;l<=lonspc;l+=1) {
        pnt[1]=lon+l;
        PolygonAdd(ptr,pnt);
      }
      pnt[1]=lon+lonspc;
      for (l=1;l<=latspc;l+=1) {
        pnt[0]=lat+latspc-l;
        PolygonAdd(ptr,pnt);
      }
      pnt[0]=lat;
      for (l=1;l<lonspc;l+=1) {
        pnt[1]=lon+lonspc-l;
        PolygonAdd(ptr,pnt);
      }

    }
  }
  return ptr;
}
