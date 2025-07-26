/* gridseek_optimized.h
   ====================
   Optimized grid search interface with AVX2/AVX-512 and CUDA support
*/

#ifndef GRID_SEEK_OPTIMIZED_H
#define GRID_SEEK_OPTIMIZED_H

#include "griddata_parallel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get time from DataMap with optimized access
 */
double grid_optimized_get_time(struct DataMap *ptr);

/**
 * Optimized grid seeking with vectorized operations
 */
int grid_optimized_seek(int fid, int yr, int mo, int dy, int hr, int mt, int sc,
                       double *atme, struct GridIndexParallel *inx,
                       struct GridPerformanceStats *stats);

/**
 * Optimized cell location with vectorized operations
 */
int grid_optimized_locate_cell(int npnt, struct GridGVecParallel *ptr, 
                              int index, struct GridPerformanceStats *stats);

/**
 * Batch cell location with vectorized operations
 */
int grid_optimized_locate_cells_batch(int npnt, struct GridGVecParallel *ptr,
                                     int *indices, int num_indices, int *results,
                                     struct GridPerformanceStats *stats);

#ifdef __cplusplus
}
#endif

#endif /* GRID_SEEK_OPTIMIZED_H */
