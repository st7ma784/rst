/**
 * addgrid_parallel.c
 * Parallel implementation of grid addition and mathematical operations
 * 
 * This module provides high-performance grid arithmetic operations using
 * parallel processing, SIMD instructions, and optimized algorithms for
 * combining multiple SuperDARN grid datasets.
 * 
 * Author: SuperDARN Parallel Processing Team
 * Date: 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include "rtypes.h"
#include "rfile.h"
#include "griddata.h"
#include "griddata_parallel.h"

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#ifdef AVX2_ENABLED
#include <immintrin.h>
#endif

#include "griddata_parallel.h"

/**
 * SIMD-optimized vector addition for statistical data
 */
#ifdef AVX2_ENABLED
static void add_statistics_simd(GridStat *result, const GridStat *a, const GridStat *b, int count) {
    int simd_count = (count / 8) * 8;
    
    // Process 8 statistics at a time using AVX2
    for (int i = 0; i < simd_count; i += 8) {
        // Load median values
        __m256 a_median = _mm256_loadu_ps(&a[i].median);
        __m256 b_median = _mm256_loadu_ps(&b[i].median);
        __m256 a_sd = _mm256_loadu_ps(&a[i].sd);
        __m256 b_sd = _mm256_loadu_ps(&b[i].sd);
        
        // Weighted average of medians (simple addition for now)
        __m256 result_median = _mm256_add_ps(a_median, b_median);
        result_median = _mm256_mul_ps(result_median, _mm256_set1_ps(0.5f));
        
        // Combined standard deviation: sqrt(sd_a^2 + sd_b^2)
        __m256 a_variance = _mm256_mul_ps(a_sd, a_sd);
        __m256 b_variance = _mm256_mul_ps(b_sd, b_sd);
        __m256 combined_variance = _mm256_add_ps(a_variance, b_variance);
        __m256 result_sd = _mm256_sqrt_ps(combined_variance);
        
        // Store results
        _mm256_storeu_ps(&result[i].median, result_median);
        _mm256_storeu_ps(&result[i].sd, result_sd);
    }
    
    // Handle remaining elements
    for (int i = simd_count; i < count; i++) {
        result[i].median = (a[i].median + b[i].median) * 0.5f;
        result[i].sd = sqrtf(a[i].sd * a[i].sd + b[i].sd * b[i].sd);
    }
}
#else
static void add_statistics_simd(GridStat *result, const GridStat *a, const GridStat *b, int count) {
    for (int i = 0; i < count; i++) {
        result[i].median = (a[i].median + b[i].median) * 0.5f;
        result[i].sd = sqrtf(a[i].sd * a[i].sd + b[i].sd * b[i].sd);
    }
}
#endif

/**
 * Parallel spatial hash table for fast cell matching
 */
typedef struct GridAddHashEntry {
    int cell_index;
    float lat, lon;
    struct GridAddHashEntry *next;
} GridAddHashEntry;

typedef struct {
    GridAddHashEntry **buckets;
    int num_buckets;
    float spatial_tolerance;
} GridAddHashTable;

static GridAddHashTable* create_spatial_hash(const struct GridData *grid, float tolerance) {
    if (!grid || tolerance <= 0.0f) return NULL;
    
    GridAddHashTable *hash = malloc(sizeof(GridAddHashTable));
    if (!hash) return NULL;
    
    hash->num_buckets = grid->vcnum * 2 + 1; // Prime-like size
    hash->spatial_tolerance = tolerance;
    hash->buckets = calloc(hash->num_buckets, sizeof(GridAddHashEntry*));
    
    if (!hash->buckets) {
        free(hash);
        return NULL;
    }
    
    // Populate hash table
    for (int i = 0; i < grid->vcnum; i++) {
        int bucket = ((int)(grid->data[i].mlat * 100) + 
                      (int)(grid->data[i].mlon * 100)) % hash->num_buckets;
        if (bucket < 0) bucket += hash->num_buckets;
        
        GridAddHashEntry *entry = malloc(sizeof(GridAddHashEntry));
        if (entry) {
            entry->cell_index = i;
            entry->lat = grid->data[i].mlat;
            entry->lon = grid->data[i].mlon;
            entry->next = hash->buckets[bucket];
            hash->buckets[bucket] = entry;
        }
    }
    
    return hash;
}

static int find_matching_cell(GridAddHashTable *hash, float lat, float lon) {
    if (!hash) return -1;
    
    int bucket = ((int)(lat * 100) + (int)(lon * 100)) % hash->num_buckets;
    if (bucket < 0) bucket += hash->num_buckets;
    
    GridAddHashEntry *entry = hash->buckets[bucket];
    while (entry) {
        float lat_diff = fabsf(entry->lat - lat);
        float lon_diff = fabsf(entry->lon - lon);
        
        if (lat_diff <= hash->spatial_tolerance && lon_diff <= hash->spatial_tolerance) {
            return entry->cell_index;
        }
        entry = entry->next;
    }
    
    return -1;
}

static void free_spatial_hash(GridAddHashTable *hash) {
    if (!hash) return;
    
    for (int i = 0; i < hash->num_buckets; i++) {
        GridAddHashEntry *entry = hash->buckets[i];
        while (entry) {
            GridAddHashEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    
    free(hash->buckets);
    free(hash);
}

/**
 * Add two grids with spatial matching and statistical combination
 */
int GridAddParallel(struct GridData *target, const struct GridData *source, float spatial_tolerance) {
    if (!target || !source) return -1;
    if (spatial_tolerance <= 0.0f) spatial_tolerance = 0.1f; // Default 0.1 degree tolerance
    
    // Create spatial hash table for fast lookups
    GridAddHashTable *hash = create_spatial_hash(target, spatial_tolerance);
    if (!hash) return -2;
    
    // Track which cells were matched
    int *matched_cells = calloc(target->vcnum, sizeof(int));
    if (!matched_cells) {
        free_spatial_hash(hash);
        return -2;
    }
    
    int added_cells = 0;
    int updated_cells = 0;
    
    // Process source cells in parallel
#ifdef OPENMP_ENABLED
    #pragma omp parallel for reduction(+:added_cells,updated_cells)
    for (int i = 0; i < source->vcnum; i++) {
        const GridGVec *src_cell = &source->data[i];
        
        // Find matching cell in target grid
        int match_idx = find_matching_cell(hash, src_cell->mlat, src_cell->mlon);
        
        if (match_idx >= 0) {
            // Update existing cell with combined statistics
            #pragma omp critical
            {
                if (!matched_cells[match_idx]) {
                    GridGVec *tgt_cell = &target->data[match_idx];
                    
                    // Combine velocity statistics
                    float weight_target = 1.0f / (tgt_cell->vel.sd * tgt_cell->vel.sd + 1e-6f);
                    float weight_source = 1.0f / (src_cell->vel.sd * src_cell->vel.sd + 1e-6f);
                    float total_weight = weight_target + weight_source;
                    
                    tgt_cell->vel.median = (tgt_cell->vel.median * weight_target + 
                                           src_cell->vel.median * weight_source) / total_weight;
                    tgt_cell->vel.sd = sqrtf(1.0f / total_weight);
                    
                    // Combine power statistics
                    weight_target = 1.0f / (tgt_cell->pwr.sd * tgt_cell->pwr.sd + 1e-6f);
                    weight_source = 1.0f / (src_cell->pwr.sd * src_cell->pwr.sd + 1e-6f);
                    total_weight = weight_target + weight_source;
                    
                    tgt_cell->pwr.median = (tgt_cell->pwr.median * weight_target + 
                                           src_cell->pwr.median * weight_source) / total_weight;
                    tgt_cell->pwr.sd = sqrtf(1.0f / total_weight);
                    
                    // Combine width statistics
                    weight_target = 1.0f / (tgt_cell->wdt.sd * tgt_cell->wdt.sd + 1e-6f);
                    weight_source = 1.0f / (src_cell->wdt.sd * src_cell->wdt.sd + 1e-6f);
                    total_weight = weight_target + weight_source;
                    
                    tgt_cell->wdt.median = (tgt_cell->wdt.median * weight_target + 
                                           src_cell->wdt.median * weight_source) / total_weight;
                    tgt_cell->wdt.sd = sqrtf(1.0f / total_weight);
                    
                    matched_cells[match_idx] = 1;
                    updated_cells++;
                }
            }
        } else {
            // No matching cell found - will need to add new cell
            added_cells++;
        }
    }
#else
    for (int i = 0; i < source->vcnum; i++) {
        const GridGVec *src_cell = &source->data[i];
        int match_idx = find_matching_cell(hash, src_cell->mlat, src_cell->mlon);
        
        if (match_idx >= 0) {
            if (!matched_cells[match_idx]) {
                GridGVec *tgt_cell = &target->data[match_idx];
                
                // Simple average combination for sequential version
                tgt_cell->vel.median = (tgt_cell->vel.median + src_cell->vel.median) * 0.5f;
                tgt_cell->vel.sd = sqrtf(tgt_cell->vel.sd * tgt_cell->vel.sd + 
                                        src_cell->vel.sd * src_cell->vel.sd);
                
                tgt_cell->pwr.median = (tgt_cell->pwr.median + src_cell->pwr.median) * 0.5f;
                tgt_cell->pwr.sd = sqrtf(tgt_cell->pwr.sd * tgt_cell->pwr.sd + 
                                        src_cell->pwr.sd * src_cell->pwr.sd);
                
                tgt_cell->wdt.median = (tgt_cell->wdt.median + src_cell->wdt.median) * 0.5f;
                tgt_cell->wdt.sd = sqrtf(tgt_cell->wdt.sd * tgt_cell->wdt.sd + 
                                        src_cell->wdt.sd * src_cell->wdt.sd);
                
                matched_cells[match_idx] = 1;
                updated_cells++;
            }
        } else {
            added_cells++;
        }
    }
#endif
    
    // Expand grid if new cells need to be added
    if (added_cells > 0) {
        int new_vcnum = target->vcnum + added_cells;
        GridGVec *new_data = grid_aligned_malloc(new_vcnum * sizeof(GridGVec), 32);
        
        if (!new_data) {
            free(matched_cells);
            free_spatial_hash(hash);
            return -2;
        }
        
        // Copy existing data
        memcpy(new_data, target->data, target->vcnum * sizeof(GridGVec));
        
        // Add new cells
        int new_cell_idx = target->vcnum;
        for (int i = 0; i < source->vcnum; i++) {
            const GridGVec *src_cell = &source->data[i];
            int match_idx = find_matching_cell(hash, src_cell->mlat, src_cell->mlon);
            
            if (match_idx < 0) {
                // Add new cell
                new_data[new_cell_idx] = *src_cell;
                new_cell_idx++;
            }
        }
        
        // Replace target data
        grid_aligned_free(target->data);
        target->data = new_data;
        target->vcnum = new_vcnum;
    }
    
    // Update time range
    if (source->st_time < target->st_time) target->st_time = source->st_time;
    if (source->ed_time > target->ed_time) target->ed_time = source->ed_time;
    
    // Cleanup
    free(matched_cells);
    free_spatial_hash(hash);
    
    return 0;
}

/**
 * Weighted addition of grids
 */
int GridAddWeightedParallel(struct GridData *target, const struct GridData *source, 
                           float target_weight, float source_weight,
                           float spatial_tolerance) {
    if (!target || !source) return -1;
    if (target_weight < 0.0f || source_weight < 0.0f) return -1;
    
    float total_weight = target_weight + source_weight;
    if (total_weight == 0.0f) return -1;
    
    target_weight /= total_weight;
    source_weight /= total_weight;
    
    GridAddHashTable *hash = create_spatial_hash(target, spatial_tolerance);
    if (!hash) return -2;
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for
    for (int i = 0; i < source->vcnum; i++) {
        const GridGVec *src_cell = &source->data[i];
        int match_idx = find_matching_cell(hash, src_cell->mlat, src_cell->mlon);
        
        if (match_idx >= 0) {
            GridGVec *tgt_cell = &target->data[match_idx];
            
            #pragma omp critical
            {
                // Weighted combination
                tgt_cell->vel.median = tgt_cell->vel.median * target_weight + 
                                      src_cell->vel.median * source_weight;
                tgt_cell->pwr.median = tgt_cell->pwr.median * target_weight + 
                                      src_cell->pwr.median * source_weight;
                tgt_cell->wdt.median = tgt_cell->wdt.median * target_weight + 
                                      src_cell->wdt.median * source_weight;
                
                // Combined uncertainty
                tgt_cell->vel.sd = sqrtf(target_weight * target_weight * tgt_cell->vel.sd * tgt_cell->vel.sd +
                                        source_weight * source_weight * src_cell->vel.sd * src_cell->vel.sd);
                tgt_cell->pwr.sd = sqrtf(target_weight * target_weight * tgt_cell->pwr.sd * tgt_cell->pwr.sd +
                                        source_weight * source_weight * src_cell->pwr.sd * src_cell->pwr.sd);
                tgt_cell->wdt.sd = sqrtf(target_weight * target_weight * tgt_cell->wdt.sd * tgt_cell->wdt.sd +
                                        source_weight * source_weight * src_cell->wdt.sd * src_cell->wdt.sd);
            }
        }
    }
#else
    for (int i = 0; i < source->vcnum; i++) {
        const GridGVec *src_cell = &source->data[i];
        int match_idx = find_matching_cell(hash, src_cell->mlat, src_cell->mlon);
        
        if (match_idx >= 0) {
            GridGVec *tgt_cell = &target->data[match_idx];
            
            tgt_cell->vel.median = tgt_cell->vel.median * target_weight + 
                                  src_cell->vel.median * source_weight;
            tgt_cell->pwr.median = tgt_cell->pwr.median * target_weight + 
                                  src_cell->pwr.median * source_weight;
            tgt_cell->wdt.median = tgt_cell->wdt.median * target_weight + 
                                  src_cell->wdt.median * source_weight;
            
            tgt_cell->vel.sd = sqrtf(target_weight * target_weight * tgt_cell->vel.sd * tgt_cell->vel.sd +
                                    source_weight * source_weight * src_cell->vel.sd * src_cell->vel.sd);
            tgt_cell->pwr.sd = sqrtf(target_weight * target_weight * tgt_cell->pwr.sd * tgt_cell->pwr.sd +
                                    source_weight * source_weight * src_cell->pwr.sd * src_cell->pwr.sd);
            tgt_cell->wdt.sd = sqrtf(target_weight * target_weight * tgt_cell->wdt.sd * tgt_cell->wdt.sd +
                                    source_weight * source_weight * src_cell->wdt.sd * src_cell->wdt.sd);
        }
    }
#endif
    
    free_spatial_hash(hash);
    return 0;
}

/**
 * Subtract one grid from another
 */
int GridSubtractParallel(struct GridData *target, const struct GridData *source, float spatial_tolerance) {
    if (!target || !source) return -1;
    
    GridAddHashTable *hash = create_spatial_hash(target, spatial_tolerance);
    if (!hash) return -2;
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for
    for (int i = 0; i < source->vcnum; i++) {
        const GridGVec *src_cell = &source->data[i];
        int match_idx = find_matching_cell(hash, src_cell->mlat, src_cell->mlon);
        
        if (match_idx >= 0) {
            GridGVec *tgt_cell = &target->data[match_idx];
            
            #pragma omp critical
            {
                // Subtract values
                tgt_cell->vel.median -= src_cell->vel.median;
                tgt_cell->pwr.median -= src_cell->pwr.median;
                tgt_cell->wdt.median -= src_cell->wdt.median;
                
                // Add uncertainties in quadrature
                tgt_cell->vel.sd = sqrtf(tgt_cell->vel.sd * tgt_cell->vel.sd + 
                                        src_cell->vel.sd * src_cell->vel.sd);
                tgt_cell->pwr.sd = sqrtf(tgt_cell->pwr.sd * tgt_cell->pwr.sd + 
                                        src_cell->pwr.sd * src_cell->pwr.sd);
                tgt_cell->wdt.sd = sqrtf(tgt_cell->wdt.sd * tgt_cell->wdt.sd + 
                                        src_cell->wdt.sd * src_cell->wdt.sd);
            }
        }
    }
#else
    for (int i = 0; i < source->vcnum; i++) {
        const GridGVec *src_cell = &source->data[i];
        int match_idx = find_matching_cell(hash, src_cell->mlat, src_cell->mlon);
        
        if (match_idx >= 0) {
            GridGVec *tgt_cell = &target->data[match_idx];
            
            tgt_cell->vel.median -= src_cell->vel.median;
            tgt_cell->pwr.median -= src_cell->pwr.median;
            tgt_cell->wdt.median -= src_cell->wdt.median;
            
            tgt_cell->vel.sd = sqrtf(tgt_cell->vel.sd * tgt_cell->vel.sd + 
                                    src_cell->vel.sd * src_cell->vel.sd);
            tgt_cell->pwr.sd = sqrtf(tgt_cell->pwr.sd * tgt_cell->pwr.sd + 
                                    src_cell->pwr.sd * src_cell->pwr.sd);
            tgt_cell->wdt.sd = sqrtf(tgt_cell->wdt.sd * tgt_cell->wdt.sd + 
                                    src_cell->wdt.sd * src_cell->wdt.sd);
        }
    }
#endif
    
    free_spatial_hash(hash);
    return 0;
}

/**
 * Scale grid values by a constant factor
 */
int GridScaleParallel(GridData *grid, float scale_factor) {
    if (!grid || grid->vcnum <= 0) return -1;
    
    float abs_scale = fabsf(scale_factor);
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < grid->vcnum; i++) {
        GridGVec *cell = &grid->data[i];
        
        // Scale median values
        cell->vel.median *= scale_factor;
        cell->pwr.median *= scale_factor;
        cell->wdt.median *= scale_factor;
        
        // Scale standard deviations by absolute value
        cell->vel.sd *= abs_scale;
        cell->pwr.sd *= abs_scale;
        cell->wdt.sd *= abs_scale;
    }
#else
    for (int i = 0; i < grid->vcnum; i++) {
        GridGVec *cell = &grid->data[i];
        
        cell->vel.median *= scale_factor;
        cell->pwr.median *= scale_factor;
        cell->wdt.median *= scale_factor;
        
        cell->vel.sd *= abs_scale;
        cell->pwr.sd *= abs_scale;
        cell->wdt.sd *= abs_scale;
    }
#endif
    
    return 0;
}

/**
 * Apply mathematical function to grid values
 */
int GridApplyFunctionParallel(GridData *grid, float (*func)(float)) {
    if (!grid || !func || grid->vcnum <= 0) return -1;
    
#ifdef OPENMP_ENABLED
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < grid->vcnum; i++) {
        GridGVec *cell = &grid->data[i];
        
        // Apply function to median values
        cell->vel.median = func(cell->vel.median);
        cell->pwr.median = func(cell->pwr.median);
        cell->wdt.median = func(cell->wdt.median);
        
        // For standard deviations, apply function to absolute values
        cell->vel.sd = func(fabsf(cell->vel.sd));
        cell->pwr.sd = func(fabsf(cell->pwr.sd));
        cell->wdt.sd = func(fabsf(cell->wdt.sd));
    }
#else
    for (int i = 0; i < grid->vcnum; i++) {
        GridGVec *cell = &grid->data[i];
        
        cell->vel.median = func(cell->vel.median);
        cell->pwr.median = func(cell->pwr.median);
        cell->wdt.median = func(cell->wdt.median);
        
        cell->vel.sd = func(fabsf(cell->vel.sd));
        cell->pwr.sd = func(fabsf(cell->pwr.sd));
        cell->wdt.sd = func(fabsf(cell->wdt.sd));
    }
#endif
    
    return 0;
}
