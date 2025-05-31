/**
 * copygrid_parallel.c
 * Parallel implementation of grid copying operations with memory optimization
 * 
 * This module provides high-performance grid copying functions using parallel
 * processing, SIMD instructions, and optimized memory management.
 * 
 * Author: SuperDARN Parallel Processing Team
 * Date: 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#ifdef AVX2_ENABLED
#include <immintrin.h>
#endif

#include "griddata_parallel.h"

/**
 * Fast memory copy using SIMD instructions
 */
static void fast_memcpy_simd(void *dest, const void *src, size_t size) {
#ifdef AVX2_ENABLED
    const size_t simd_size = 32; // AVX2 processes 32 bytes at a time
    size_t simd_chunks = size / simd_size;
    size_t remaining = size % simd_size;
    
    const char *src_ptr = (const char*)src;
    char *dest_ptr = (char*)dest;
    
    // Process 32-byte chunks with AVX2
    for (size_t i = 0; i < simd_chunks; i++) {
        __m256i data = _mm256_loadu_si256((const __m256i*)(src_ptr + i * simd_size));
        _mm256_storeu_si256((__m256i*)(dest_ptr + i * simd_size), data);
    }
    
    // Handle remaining bytes
    if (remaining > 0) {
        memcpy(dest_ptr + simd_chunks * simd_size, 
               src_ptr + simd_chunks * simd_size, 
               remaining);
    }
#else
    memcpy(dest, src, size);
#endif
}

/**
 * Parallel grid cell copying with SIMD optimization
 */
static int copy_grid_cells_parallel(GridGVec *dest, const GridGVec *src, int count) {
    if (!dest || !src || count <= 0) return -1;
    
#ifdef OPENMP_ENABLED
    // Use parallel copying for large datasets
    if (count > 1000) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; i++) {
            // Manual copy for better control over SIMD optimization
            dest[i].mlat = src[i].mlat;
            dest[i].mlon = src[i].mlon;
            dest[i].kvect = src[i].kvect;
            dest[i].st_id = src[i].st_id;
            
            // Copy statistical data structures
            dest[i].vel = src[i].vel;
            dest[i].pwr = src[i].pwr;
            dest[i].wdt = src[i].wdt;
        }
    } else {
        // Use SIMD-optimized memcpy for smaller datasets
        fast_memcpy_simd(dest, src, count * sizeof(GridGVec));
    }
#else
    fast_memcpy_simd(dest, src, count * sizeof(GridGVec));
#endif

    return 0;
}

/**
 * Parallel station data copying
 */
static int copy_station_data_parallel(GridSVec *dest, const GridSVec *src, int count) {
    if (!dest || !src || count <= 0) return -1;
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; i++) {
        dest[i] = src[i]; // Structure assignment
    }
#else
    fast_memcpy_simd(dest, src, count * sizeof(GridSVec));
#endif
    
    return 0;
}

/**
 * Deep copy grid data with parallel processing
 */
GridData* GridCopyParallel(const GridData *source) {
    if (!source) return NULL;
    
    // Allocate new grid structure
    GridData *dest = malloc(sizeof(GridData));
    if (!dest) return NULL;
    
    // Copy basic parameters
    dest->st_time = source->st_time;
    dest->ed_time = source->ed_time;
    dest->vcnum = source->vcnum;
    dest->stnum = source->stnum;
    
    // Allocate and copy velocity cell data
    if (source->vcnum > 0 && source->data) {
        dest->data = grid_aligned_malloc(source->vcnum * sizeof(GridGVec), 32);
        if (!dest->data) {
            free(dest);
            return NULL;
        }
        
        if (copy_grid_cells_parallel(dest->data, source->data, source->vcnum) != 0) {
            grid_aligned_free(dest->data);
            free(dest);
            return NULL;
        }
    } else {
        dest->data = NULL;
    }
    
    // Allocate and copy station data
    if (source->stnum > 0 && source->sdata) {
        dest->sdata = grid_aligned_malloc(source->stnum * sizeof(GridSVec), 32);
        if (!dest->sdata) {
            grid_aligned_free(dest->data);
            free(dest);
            return NULL;
        }
        
        if (copy_station_data_parallel(dest->sdata, source->sdata, source->stnum) != 0) {
            grid_aligned_free(dest->sdata);
            grid_aligned_free(dest->data);
            free(dest);
            return NULL;
        }
    } else {
        dest->sdata = NULL;
    }
    
    // Initialize parallel processing context if available
    dest->parallel_ctx = NULL;
    dest->spatial_index = NULL;
    dest->memory_pool = NULL;
    dest->metrics = NULL;
    
    return dest;
}

/**
 * Selective grid copying with filtering
 */
GridData* GridCopySelectiveParallel(const GridData *source, 
                                    int (*filter_func)(const GridGVec*, void*),
                                    void *filter_data) {
    if (!source || !filter_func) return NULL;
    
    // First pass: count filtered cells
    int filtered_count = 0;
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for reduction(+:filtered_count)
    for (int i = 0; i < source->vcnum; i++) {
        if (filter_func(&source->data[i], filter_data)) {
            filtered_count++;
        }
    }
#else
    for (int i = 0; i < source->vcnum; i++) {
        if (filter_func(&source->data[i], filter_data)) {
            filtered_count++;
        }
    }
#endif
    
    if (filtered_count == 0) return NULL;
    
    // Allocate destination grid
    GridData *dest = malloc(sizeof(GridData));
    if (!dest) return NULL;
    
    dest->st_time = source->st_time;
    dest->ed_time = source->ed_time;
    dest->vcnum = filtered_count;
    dest->stnum = source->stnum;
    
    // Allocate filtered cell array
    dest->data = grid_aligned_malloc(filtered_count * sizeof(GridGVec), 32);
    if (!dest->data) {
        free(dest);
        return NULL;
    }
    
    // Second pass: copy filtered cells
    int dest_idx = 0;
    
#ifdef OPENMP_ENABLED
    // Use critical section for thread-safe copying
    #pragma omp parallel for
    for (int i = 0; i < source->vcnum; i++) {
        if (filter_func(&source->data[i], filter_data)) {
            int local_idx;
            #pragma omp critical
            {
                local_idx = dest_idx++;
            }
            dest->data[local_idx] = source->data[i];
        }
    }
#else
    for (int i = 0; i < source->vcnum; i++) {
        if (filter_func(&source->data[i], filter_data)) {
            dest->data[dest_idx++] = source->data[i];
        }
    }
#endif
    
    // Copy station data
    if (source->stnum > 0 && source->sdata) {
        dest->sdata = grid_aligned_malloc(source->stnum * sizeof(GridSVec), 32);
        if (!dest->sdata) {
            grid_aligned_free(dest->data);
            free(dest);
            return NULL;
        }
        
        copy_station_data_parallel(dest->sdata, source->sdata, source->stnum);
    } else {
        dest->sdata = NULL;
    }
    
    // Initialize additional fields
    dest->parallel_ctx = NULL;
    dest->spatial_index = NULL;
    dest->memory_pool = NULL;
    dest->metrics = NULL;
    
    return dest;
}

/**
 * Parallel grid region extraction
 */
GridData* GridCopyRegionParallel(const GridData *source, 
                                 float min_lat, float max_lat,
                                 float min_lon, float max_lon) {
    if (!source) return NULL;
    
    // Create filter function for region extraction
    typedef struct {
        float min_lat, max_lat, min_lon, max_lon;
    } RegionFilter;
    
    RegionFilter filter_data = {min_lat, max_lat, min_lon, max_lon};
    
    // Region filter function
    int region_filter(const GridGVec *cell, void *data) {
        RegionFilter *region = (RegionFilter*)data;
        return (cell->mlat >= region->min_lat && cell->mlat <= region->max_lat &&
                cell->mlon >= region->min_lon && cell->mlon <= region->max_lon);
    }
    
    return GridCopySelectiveParallel(source, region_filter, &filter_data);
}

/**
 * Parallel grid time window extraction
 */
GridData* GridCopyTimeWindowParallel(const GridData *source,
                                     time_t start_time, time_t end_time) {
    if (!source) return NULL;
    
    // Create a new grid with the same structure but filtered time
    GridData *dest = GridCopyParallel(source);
    if (!dest) return NULL;
    
    // Update time boundaries
    if (start_time > source->st_time) dest->st_time = start_time;
    if (end_time < source->ed_time) dest->ed_time = end_time;
    
    // Filter could be implemented here for time-based cell filtering
    // For now, we copy the entire spatial data within the time window
    
    return dest;
}

/**
 * Parallel grid statistics copying
 */
int GridCopyStatsParallel(GridData *dest, const GridData *source) {
    if (!dest || !source) return -1;
    
    if (dest->vcnum != source->vcnum) return -2; // Size mismatch
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest->vcnum; i++) {
        // Copy only statistical fields, preserve spatial coordinates
        dest->data[i].vel = source->data[i].vel;
        dest->data[i].pwr = source->data[i].pwr;
        dest->data[i].wdt = source->data[i].wdt;
    }
#else
    for (int i = 0; i < dest->vcnum; i++) {
        dest->data[i].vel = source->data[i].vel;
        dest->data[i].pwr = source->data[i].pwr;
        dest->data[i].wdt = source->data[i].wdt;
    }
#endif
    
    return 0;
}

/**
 * Memory-efficient grid copying with compression
 */
GridData* GridCopyCompressedParallel(const GridData *source, float tolerance) {
    if (!source || tolerance < 0.0f) return NULL;
    
    // Create filter for compression based on data significance
    typedef struct {
        float tolerance;
    } CompressionFilter;
    
    CompressionFilter filter_data = {tolerance};
    
    // Compression filter - keep cells with significant data
    int compression_filter(const GridGVec *cell, void *data) {
        CompressionFilter *comp = (CompressionFilter*)data;
        
        // Keep cells with velocity above tolerance or high power
        return (fabs(cell->vel.median) > comp->tolerance || 
                cell->pwr.median > 20.0f || 
                cell->vel.sd < comp->tolerance * 0.1f); // Low uncertainty
    }
    
    return GridCopySelectiveParallel(source, compression_filter, &filter_data);
}

/**
 * Free grid data with parallel deallocation
 */
void GridFreeParallel(GridData *grid) {
    if (!grid) return;
    
    // Free data arrays
    if (grid->data) {
        grid_aligned_free(grid->data);
    }
    
    if (grid->sdata) {
        grid_aligned_free(grid->sdata);
    }
    
    // Free parallel processing context
    if (grid->parallel_ctx) {
        // Cleanup parallel context
        free(grid->parallel_ctx);
    }
    
    if (grid->spatial_index) {
        // Cleanup spatial index
        free(grid->spatial_index);
    }
    
    if (grid->memory_pool) {
        // Cleanup memory pool
        free(grid->memory_pool);
    }
    
    if (grid->metrics) {
        // Cleanup performance metrics
        free(grid->metrics);
    }
    
    // Free main structure
    free(grid);
}

/**
 * Validate grid data integrity
 */
int GridValidateParallel(const GridData *grid) {
    if (!grid) return -1;
    
    // Basic structure validation
    if (grid->vcnum < 0 || grid->stnum < 0) return -2;
    if (grid->ed_time < grid->st_time) return -3;
    
    if (grid->vcnum > 0 && !grid->data) return -4;
    if (grid->stnum > 0 && !grid->sdata) return -5;
    
    // Parallel validation of cell data
    int validation_errors = 0;
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for reduction(+:validation_errors)
    for (int i = 0; i < grid->vcnum; i++) {
        const GridGVec *cell = &grid->data[i];
        
        // Check coordinate ranges
        if (cell->mlat < -90.0f || cell->mlat > 90.0f ||
            cell->mlon < -180.0f || cell->mlon > 360.0f) {
            validation_errors++;
        }
        
        // Check for NaN values
        if (isnan(cell->vel.median) || isnan(cell->pwr.median) || 
            isnan(cell->wdt.median)) {
            validation_errors++;
        }
        
        // Check statistical consistency
        if (cell->vel.sd < 0.0f || cell->pwr.sd < 0.0f || cell->wdt.sd < 0.0f) {
            validation_errors++;
        }
    }
#else
    for (int i = 0; i < grid->vcnum; i++) {
        const GridGVec *cell = &grid->data[i];
        
        if (cell->mlat < -90.0f || cell->mlat > 90.0f ||
            cell->mlon < -180.0f || cell->mlon > 360.0f ||
            isnan(cell->vel.median) || isnan(cell->pwr.median) || 
            isnan(cell->wdt.median) ||
            cell->vel.sd < 0.0f || cell->pwr.sd < 0.0f || cell->wdt.sd < 0.0f) {
            validation_errors++;
        }
    }
#endif
    
    return validation_errors;
}
