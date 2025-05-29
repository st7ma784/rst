/* gridmerge_parallel.c
   ====================
   Author: RST Parallel Implementation

 Copyright (c) 2024 Parallel SuperDARN Grid Processing

This file is part of the Parallel SuperDARN Grid Library.

Parallel SuperDARN Grid Library is free software: you can redistribute it 
and/or modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Modifications:
- Added OpenMP parallelization for grid merging operations
- Enhanced memory management and error checking
- Implemented parallel data integration and averaging
- Added statistical analysis during merge operations
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "rtypes.h"
#include "rmath.h"
#include "griddata_parallel.h"

/**
 * Parallel grid averaging with enhanced statistics
 */
int grid_parallel_average(struct GridDataParallel **src_grids, int num_grids,
                         struct GridDataParallel *dst_grid, int flags,
                         struct GridPerformanceStats *stats) {
    if (!src_grids || !dst_grid || num_grids <= 0) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    double start_time = omp_get_wtime();
    
    // Initialize destination grid
    grid_parallel_free(dst_grid);
    grid_parallel_make(dst_grid, src_grids[0]->stnum);
    
    // Copy time information from first grid
    dst_grid->st_time = src_grids[0]->st_time;
    dst_grid->ed_time = src_grids[0]->ed_time;
    dst_grid->xtd = src_grids[0]->xtd;
    
    // Find all unique cell indices across grids
    int max_cells = 0;
    for (int g = 0; g < num_grids; g++) {
        if (src_grids[g]) max_cells += src_grids[g]->vcnum;
    }
    
    if (max_cells == 0) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    // Collect all unique indices
    int *all_indices = malloc(sizeof(int) * max_cells);
    int *index_counts = malloc(sizeof(int) * max_cells);
    int unique_count = 0;
    
    if (!all_indices || !index_counts) {
        free(all_indices);
        free(index_counts);
        if (stats) stats->error_count++;
        return -1;
    }
    
    // Phase 1: Collect all unique cell indices
    for (int g = 0; g < num_grids; g++) {
        if (!src_grids[g] || !src_grids[g]->data) continue;
        
        for (int i = 0; i < src_grids[g]->vcnum; i++) {
            int index = src_grids[g]->data[i].index;
            
            // Check if index already exists
            int found = -1;
            for (int j = 0; j < unique_count; j++) {
                if (all_indices[j] == index) {
                    found = j;
                    break;
                }
            }
            
            if (found >= 0) {
                index_counts[found]++;
            } else {
                all_indices[unique_count] = index;
                index_counts[unique_count] = 1;
                unique_count++;
            }
        }
    }
    
    // Allocate destination data
    dst_grid->vcnum = unique_count;
    if (unique_count > 0) {
        dst_grid->data = malloc(sizeof(struct GridGVecParallel) * unique_count);
        if (!dst_grid->data) {
            free(all_indices);
            free(index_counts);
            if (stats) stats->error_count++;
            return -1;
        }
    }
    
    // Phase 2: Parallel averaging for each unique cell
    #pragma omp parallel for
    for (int cell = 0; cell < unique_count; cell++) {
        int target_index = all_indices[cell];
        
        // Initialize averaged cell
        struct GridGVecParallel *avg_cell = &dst_grid->data[cell];
        memset(avg_cell, 0, sizeof(struct GridGVecParallel));
        avg_cell->index = target_index;
        
        // Collect data from all grids for this cell
        double *vel_values = malloc(sizeof(double) * num_grids);
        double *vel_errors = malloc(sizeof(double) * num_grids);
        double *pwr_values = malloc(sizeof(double) * num_grids);
        double *pwr_errors = malloc(sizeof(double) * num_grids);
        double *wdt_values = malloc(sizeof(double) * num_grids);
        double *wdt_errors = malloc(sizeof(double) * num_grids);
        int valid_count = 0;
        
        if (!vel_values || !vel_errors || !pwr_values || 
            !pwr_errors || !wdt_values || !wdt_errors) {
            free(vel_values); free(vel_errors);
            free(pwr_values); free(pwr_errors);
            free(wdt_values); free(wdt_errors);
            continue;
        }
        
        // Collect values from all grids
        for (int g = 0; g < num_grids; g++) {
            if (!src_grids[g] || !src_grids[g]->data) continue;
            
            int cell_idx = grid_parallel_locate_cell(src_grids[g]->vcnum,
                                                   src_grids[g]->data,
                                                   target_index, NULL);
            
            if (cell_idx < src_grids[g]->vcnum) {
                struct GridGVecParallel *src_cell = &src_grids[g]->data[cell_idx];
                
                // Copy position from first valid cell
                if (valid_count == 0) {
                    avg_cell->mlat = src_cell->mlat;
                    avg_cell->mlon = src_cell->mlon;
                    avg_cell->azm = src_cell->azm;
                    avg_cell->st_id = src_cell->st_id;
                    avg_cell->chn = src_cell->chn;
                }
                
                vel_values[valid_count] = src_cell->vel.median;
                vel_errors[valid_count] = src_cell->vel.sd;
                pwr_values[valid_count] = src_cell->pwr.median;
                pwr_errors[valid_count] = src_cell->pwr.sd;
                wdt_values[valid_count] = src_cell->wdt.median;
                wdt_errors[valid_count] = src_cell->wdt.sd;
                valid_count++;
            }
        }
        
        // Calculate averages and statistics
        if (valid_count > 0) {
            // Velocity statistics
            double vel_sum = 0, vel_sum_sq = 0;
            double vel_err_sum = 0;
            for (int i = 0; i < valid_count; i++) {
                vel_sum += vel_values[i];
                vel_sum_sq += vel_values[i] * vel_values[i];
                vel_err_sum += 1.0 / (vel_errors[i] * vel_errors[i]);
            }
            avg_cell->vel.median = vel_sum / valid_count;
            if (valid_count > 1) {
                avg_cell->vel.sd = sqrt((vel_sum_sq - vel_sum*vel_sum/valid_count) / 
                                      (valid_count - 1));
            } else {
                avg_cell->vel.sd = vel_errors[0];
            }
            
            // Power statistics
            double pwr_sum = 0, pwr_sum_sq = 0;
            for (int i = 0; i < valid_count; i++) {
                pwr_sum += pwr_values[i];
                pwr_sum_sq += pwr_values[i] * pwr_values[i];
            }
            avg_cell->pwr.median = pwr_sum / valid_count;
            if (valid_count > 1) {
                avg_cell->pwr.sd = sqrt((pwr_sum_sq - pwr_sum*pwr_sum/valid_count) / 
                                      (valid_count - 1));
            } else {
                avg_cell->pwr.sd = pwr_errors[0];
            }
            
            // Width statistics
            double wdt_sum = 0, wdt_sum_sq = 0;
            for (int i = 0; i < valid_count; i++) {
                wdt_sum += wdt_values[i];
                wdt_sum_sq += wdt_values[i] * wdt_values[i];
            }
            avg_cell->wdt.median = wdt_sum / valid_count;
            if (valid_count > 1) {
                avg_cell->wdt.sd = sqrt((wdt_sum_sq - wdt_sum*wdt_sum/valid_count) / 
                                      (valid_count - 1));
            } else {
                avg_cell->wdt.sd = wdt_errors[0];
            }
            
            // Store averaging statistics
            avg_cell->avg_count = valid_count;
            avg_cell->quality_flag = (valid_count >= 3) ? 1 : 0;
        }
        
        free(vel_values); free(vel_errors);
        free(pwr_values); free(pwr_errors);
        free(wdt_values); free(wdt_errors);
    }
    
    // Copy station data from first grid
    if (dst_grid->stnum > 0 && src_grids[0]->sdata) {
        for (int i = 0; i < dst_grid->stnum; i++) {
            dst_grid->sdata[i] = src_grids[0]->sdata[i];
        }
    }
    
    free(all_indices);
    free(index_counts);
    
    if (stats) {
        stats->grid_averages++;
        stats->average_time += omp_get_wtime() - start_time;
        stats->cells_averaged += unique_count;
    }
    
    return 0;
}

/**
 * Parallel grid merging with conflict resolution
 */
int grid_parallel_merge(struct GridDataParallel *grid1, 
                       struct GridDataParallel *grid2,
                       struct GridDataParallel *merged_grid,
                       int merge_mode,
                       struct GridPerformanceStats *stats) {
    if (!grid1 || !grid2 || !merged_grid) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    double start_time = omp_get_wtime();
    
    // Initialize merged grid
    grid_parallel_free(merged_grid);
    
    // Determine time range
    merged_grid->st_time = (grid1->st_time < grid2->st_time) ? 
                          grid1->st_time : grid2->st_time;
    merged_grid->ed_time = (grid1->ed_time > grid2->ed_time) ? 
                          grid1->ed_time : grid2->ed_time;
    merged_grid->xtd = grid1->xtd || grid2->xtd;
    merged_grid->stnum = grid1->stnum + grid2->stnum;
    
    // Allocate station data
    if (merged_grid->stnum > 0) {
        merged_grid->sdata = malloc(sizeof(struct GridSVecParallel) * 
                                   merged_grid->stnum);
        if (!merged_grid->sdata) {
            if (stats) stats->error_count++;
            return -1;
        }
        
        // Copy station data
        for (int i = 0; i < grid1->stnum; i++) {
            merged_grid->sdata[i] = grid1->sdata[i];
        }
        for (int i = 0; i < grid2->stnum; i++) {
            merged_grid->sdata[grid1->stnum + i] = grid2->sdata[i];
        }
    }
    
    // Find all unique cell indices
    int total_cells = grid1->vcnum + grid2->vcnum;
    int *all_indices = malloc(sizeof(int) * total_cells);
    int unique_count = 0;
    
    if (!all_indices) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    // Collect unique indices from both grids
    for (int i = 0; i < grid1->vcnum; i++) {
        int index = grid1->data[i].index;
        int found = 0;
        for (int j = 0; j < unique_count; j++) {
            if (all_indices[j] == index) {
                found = 1;
                break;
            }
        }
        if (!found) {
            all_indices[unique_count++] = index;
        }
    }
    
    for (int i = 0; i < grid2->vcnum; i++) {
        int index = grid2->data[i].index;
        int found = 0;
        for (int j = 0; j < unique_count; j++) {
            if (all_indices[j] == index) {
                found = 1;
                break;
            }
        }
        if (!found) {
            all_indices[unique_count++] = index;
        }
    }
    
    // Allocate merged data
    merged_grid->vcnum = unique_count;
    if (unique_count > 0) {
        merged_grid->data = malloc(sizeof(struct GridGVecParallel) * unique_count);
        if (!merged_grid->data) {
            free(all_indices);
            if (stats) stats->error_count++;
            return -1;
        }
    }
    
    // Parallel merging of cells
    int conflicts = 0;
    
    #pragma omp parallel for reduction(+:conflicts)
    for (int cell = 0; cell < unique_count; cell++) {
        int target_index = all_indices[cell];
        struct GridGVecParallel *merged_cell = &merged_grid->data[cell];
        
        // Find cells in both grids
        int idx1 = grid_parallel_locate_cell(grid1->vcnum, grid1->data, 
                                           target_index, NULL);
        int idx2 = grid_parallel_locate_cell(grid2->vcnum, grid2->data, 
                                           target_index, NULL);
        
        int has_cell1 = (idx1 < grid1->vcnum);
        int has_cell2 = (idx2 < grid2->vcnum);
        
        if (has_cell1 && has_cell2) {
            // Conflict - both grids have this cell
            conflicts++;
            struct GridGVecParallel *cell1 = &grid1->data[idx1];
            struct GridGVecParallel *cell2 = &grid2->data[idx2];
            
            // Copy position from first cell
            *merged_cell = *cell1;
            
            switch (merge_mode) {
                case GRID_MERGE_AVERAGE:
                    // Average the values
                    merged_cell->vel.median = (cell1->vel.median + cell2->vel.median) / 2.0;
                    merged_cell->vel.sd = sqrt((cell1->vel.sd*cell1->vel.sd + 
                                              cell2->vel.sd*cell2->vel.sd) / 2.0);
                    merged_cell->pwr.median = (cell1->pwr.median + cell2->pwr.median) / 2.0;
                    merged_cell->pwr.sd = sqrt((cell1->pwr.sd*cell1->pwr.sd + 
                                              cell2->pwr.sd*cell2->pwr.sd) / 2.0);
                    merged_cell->wdt.median = (cell1->wdt.median + cell2->wdt.median) / 2.0;
                    merged_cell->wdt.sd = sqrt((cell1->wdt.sd*cell1->wdt.sd + 
                                              cell2->wdt.sd*cell2->wdt.sd) / 2.0);
                    break;
                    
                case GRID_MERGE_PREFER_FIRST:
                    // Keep cell1 values (already copied)
                    break;
                    
                case GRID_MERGE_PREFER_SECOND:
                    // Use cell2 values
                    merged_cell->vel = cell2->vel;
                    merged_cell->pwr = cell2->pwr;
                    merged_cell->wdt = cell2->wdt;
                    break;
                    
                case GRID_MERGE_PREFER_HIGHER_POWER:
                    // Use cell with higher power
                    if (cell2->pwr.median > cell1->pwr.median) {
                        merged_cell->vel = cell2->vel;
                        merged_cell->pwr = cell2->pwr;
                        merged_cell->wdt = cell2->wdt;
                    }
                    break;
                    
                case GRID_MERGE_PREFER_LOWER_ERROR:
                    // Use cell with lower velocity error
                    if (cell2->vel.sd < cell1->vel.sd) {
                        merged_cell->vel = cell2->vel;
                        merged_cell->pwr = cell2->pwr;
                        merged_cell->wdt = cell2->wdt;
                    }
                    break;
                    
                default:
                    // Default to average
                    merged_cell->vel.median = (cell1->vel.median + cell2->vel.median) / 2.0;
                    merged_cell->vel.sd = sqrt((cell1->vel.sd*cell1->vel.sd + 
                                              cell2->vel.sd*cell2->vel.sd) / 2.0);
                    break;
            }
            
            merged_cell->merge_count = 2;
            merged_cell->quality_flag = (cell1->quality_flag && cell2->quality_flag) ? 1 : 0;
            
        } else if (has_cell1) {
            // Only grid1 has this cell
            *merged_cell = grid1->data[idx1];
            merged_cell->merge_count = 1;
        } else if (has_cell2) {
            // Only grid2 has this cell
            *merged_cell = grid2->data[idx2];
            merged_cell->merge_count = 1;
        }
    }
    
    free(all_indices);
    
    if (stats) {
        stats->grid_merges++;
        stats->merge_time += omp_get_wtime() - start_time;
        stats->merge_conflicts += conflicts;
        stats->cells_merged += unique_count;
    }
    
    return 0;
}

/**
 * Integrate multiple grids into a single composite grid
 */
int grid_parallel_integrate(struct GridDataParallel **grids, int num_grids,
                           struct GridDataParallel *integrated_grid,
                           struct GridIntegrationParams *params,
                           struct GridPerformanceStats *stats) {
    if (!grids || !integrated_grid || num_grids <= 0) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    double start_time = omp_get_wtime();
    
    // Start with first grid
    if (grid_parallel_copy(integrated_grid, grids[0]) != 0) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    // Merge each subsequent grid
    for (int i = 1; i < num_grids; i++) {
        if (!grids[i]) continue;
        
        struct GridDataParallel temp_grid;
        grid_parallel_make(&temp_grid, 0);
        
        int merge_mode = (params && params->prefer_quality) ? 
                        GRID_MERGE_PREFER_LOWER_ERROR : GRID_MERGE_AVERAGE;
        
        if (grid_parallel_merge(integrated_grid, grids[i], &temp_grid, 
                               merge_mode, stats) != 0) {
            grid_parallel_free(&temp_grid);
            continue;
        }
        
        // Replace integrated grid with merged result
        grid_parallel_free(integrated_grid);
        *integrated_grid = temp_grid;
    }
    
    // Apply post-integration filtering if requested
    if (params && params->apply_post_filter) {
        struct GridFilterParams filter_params = {0};
        filter_params.outlier_detection = 1;
        filter_params.outlier_threshold = 2.0;
        
        grid_parallel_filter_outliers(integrated_grid, &filter_params, stats);
    }
    
    if (stats) {
        stats->grid_integrations++;
        stats->integration_time += omp_get_wtime() - start_time;
    }
    
    return 0;
}
