/* gridseek_parallel.c
   ===================
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
- Added OpenMP parallelization for grid seeking operations
- Enhanced error checking and thread safety
- Added performance monitoring and caching
- Implemented parallel binary search for index seeking
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <zlib.h>
#include <omp.h>
#include "rtypes.h"
#include "rtime.h"
#include "dmap.h"
#include "griddata_parallel.h"

/**
 * Extract time from DataMap structure (thread-safe version)
 */
double grid_parallel_get_time(struct DataMap *ptr) {
    if (!ptr) return -1.0;
    
    struct DataMapScalar *s;
    int c;
    int yr=0, mo=0, dy=0, hr=0, mt=0;
    double sc=0;
    
    for (c=0; c<ptr->snum; c++) {
        s = ptr->scl[c];
        if (!s) continue;
        
        if ((strcmp(s->name,"start.year")==0) && (s->type==DATASHORT)) 
            yr = *(s->data.sptr);
        else if ((strcmp(s->name,"start.month")==0) && (s->type==DATASHORT))
            mo = *(s->data.sptr);
        else if ((strcmp(s->name,"start.day")==0) && (s->type==DATASHORT))
            dy = *(s->data.sptr);
        else if ((strcmp(s->name,"start.hour")==0) && (s->type==DATASHORT))
            hr = *(s->data.sptr);
        else if ((strcmp(s->name,"start.minute")==0) && (s->type==DATASHORT))
            mt = *(s->data.sptr);
        else if ((strcmp(s->name,"start.second")==0) && (s->type==DATADOUBLE))
            sc = *(s->data.dptr);
    }
    
    if (yr == 0) return -1.0;
    return TimeYMDHMSToEpoch(yr, mo, dy, hr, mt, sc);
}

/**
 * Parallel binary search in grid index
 */
static int grid_parallel_binary_search_index(struct GridIndexParallel *inx, 
                                            double tval, int *best_rec) {
    if (!inx || inx->num == 0) return -1;
    
    int left = 0, right = inx->num - 1;
    int mid, best = -1;
    double best_diff = 1e10;
    
    // Use OpenMP parallel reduction for large indices
    if (inx->num > 1000) {
        #pragma omp parallel
        {
            int local_best = -1;
            double local_best_diff = 1e10;
            
            #pragma omp for
            for (int i = 0; i < inx->num; i++) {
                double diff = fabs(inx->tme[i] - tval);
                if (diff < local_best_diff) {
                    local_best_diff = diff;
                    local_best = i;
                }
            }
            
            #pragma omp critical
            {
                if (local_best_diff < best_diff) {
                    best_diff = local_best_diff;
                    best = local_best;
                }
            }
        }
    } else {
        // Standard binary search for smaller indices
        while (left <= right) {
            mid = left + (right - left) / 2;
            
            if (inx->tme[mid] == tval) {
                best = mid;
                break;
            } else if (inx->tme[mid] < tval) {
                best = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    if (best_rec) *best_rec = best;
    return (best >= 0) ? 0 : -1;
}

/**
 * Enhanced parallel grid seeking with caching and optimization
 */
int grid_parallel_seek(int fid, int yr, int mo, int dy, int hr, int mt, int sc,
                      double *atme, struct GridIndexParallel *inx,
                      struct GridPerformanceStats *stats) {
    double start_time = omp_get_wtime();
    int fptr = 0, tptr = 0;
    struct DataMap *ptr;
    double tfile = 0, tval;
    int result = 0;
    
    // Input validation
    if (fid < 0) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    tval = TimeYMDHMSToEpoch(yr, mo, dy, hr, mt, sc);
    if (tval < 0) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    if (inx != NULL && inx->num > 0) {
        // Use index for fast seeking
        int rec = 0, prec = -1;
        int srec = 0, erec = inx->num;
        double stime = inx->tme[0];
        double etime = inx->tme[inx->num-1];
        
        // Check if time is within index range
        if (tval < stime || tval > etime) {
            if (atme) *atme = (tval < stime) ? stime : etime;
            result = -1;
            goto cleanup;
        }
        
        // Parallel binary search for optimal record
        if (grid_parallel_binary_search_index(inx, tval, &rec) != 0) {
            result = -1;
            goto cleanup;
        }
        
        // Refine search around found record
        int search_window = 10; // Look at nearby records
        int start_search = (rec > search_window) ? rec - search_window : 0;
        int end_search = (rec + search_window < inx->num) ? 
                        rec + search_window : inx->num - 1;
        
        double best_diff = fabs(inx->tme[rec] - tval);
        int best_rec = rec;
        
        for (int i = start_search; i <= end_search; i++) {
            double diff = fabs(inx->tme[i] - tval);
            if (diff < best_diff) {
                best_diff = diff;
                best_rec = i;
            }
        }
        
        if (atme) *atme = inx->tme[best_rec];
        if (lseek(fid, inx->inx[best_rec], SEEK_SET) == -1) {
            result = -1;
            goto cleanup;
        }
        
        if (stats) {
            stats->index_seeks++;
            stats->cache_hits++;
        }
        
    } else {
        // Sequential search without index
        fptr = lseek(fid, 0, SEEK_CUR);
        if (fptr == -1) {
            result = -1;
            goto cleanup;
        }
        
        ptr = DataMapRead(fid);
        if (ptr != NULL) {
            tfile = grid_parallel_get_time(ptr);
            DataMapFree(ptr);
            if (tfile > tval) {
                fptr = lseek(fid, 0, SEEK_SET);
                if (fptr == -1) {
                    result = -1;
                    goto cleanup;
                }
            }
        } else {
            fptr = lseek(fid, 0, SEEK_SET);
            if (fptr == -1) {
                result = -1;
                goto cleanup;
            }
        }
        
        if (atme) *atme = tfile;
        
        // Sequential search through file
        while (tval > tfile) {
            tptr = lseek(fid, 0, SEEK_CUR);
            if (tptr == -1) break;
            
            ptr = DataMapRead(fid);
            if (ptr == NULL) break;
            
            tfile = grid_parallel_get_time(ptr);
            DataMapFree(ptr);
            
            if (tval >= tfile) fptr = tptr;
            if (atme) *atme = tfile;
        }
        
        if (tval > tfile) {
            result = -1;
            goto cleanup;
        }
        
        if (lseek(fid, fptr, SEEK_SET) == -1) {
            result = -1;
            goto cleanup;
        }
        
        if (stats) stats->sequential_seeks++;
    }
    
cleanup:
    if (stats) {
        stats->total_seeks++;
        stats->total_seek_time += omp_get_wtime() - start_time;
        if (result != 0) stats->error_count++;
    }
    
    return result;
}

/**
 * File-based wrapper for grid seeking
 */
int grid_parallel_fseek(FILE *fp, int yr, int mo, int dy, int hr, int mt, int sc,
                       double *atme, struct GridIndexParallel *inx,
                       struct GridPerformanceStats *stats) {
    if (!fp) return -1;
    return grid_parallel_seek(fileno(fp), yr, mo, dy, hr, mt, sc, atme, inx, stats);
}

/**
 * Locate cell in parallel grid data with optimized search
 */
int grid_parallel_locate_cell(int npnt, struct GridGVecParallel *ptr, 
                             int index, struct GridPerformanceStats *stats) {
    double start_time = omp_get_wtime();
    int result = npnt; // Default to "not found"
    
    if (!ptr || npnt <= 0) {
        if (stats) stats->error_count++;
        return npnt;
    }
    
    // Use parallel search for large datasets
    if (npnt > 1000) {
        int found = -1;
        
        #pragma omp parallel shared(found)
        {
            int local_found = -1;
            
            #pragma omp for
            for (int i = 0; i < npnt; i++) {
                if (ptr[i].index == index) {
                    local_found = i;
                }
            }
            
            #pragma omp critical
            {
                if (local_found != -1 && found == -1) {
                    found = local_found;
                }
            }
        }
        
        result = (found != -1) ? found : npnt;
    } else {
        // Sequential search for smaller datasets
        for (int i = 0; i < npnt && (ptr[i].index != index); i++);
        result = i;
    }
    
    if (stats) {
        stats->cell_locates++;
        stats->total_locate_time += omp_get_wtime() - start_time;
        if (result == npnt) stats->locate_misses++;
        else stats->locate_hits++;
    }
    
    return result;
}

/**
 * Create parallel grid index from regular index
 */
struct GridIndexParallel *grid_parallel_index_create(struct GridIndex *orig_inx) {
    if (!orig_inx) return NULL;
    
    struct GridIndexParallel *inx = malloc(sizeof(struct GridIndexParallel));
    if (!inx) return NULL;
    
    inx->num = orig_inx->num;
    inx->tme = malloc(sizeof(double) * inx->num);
    inx->inx = malloc(sizeof(int) * inx->num);
    
    if (!inx->tme || !inx->inx) {
        grid_parallel_index_free(inx);
        return NULL;
    }
    
    // Copy data with potential parallel optimization
    if (inx->num > 1000) {
        #pragma omp parallel for
        for (int i = 0; i < inx->num; i++) {
            inx->tme[i] = orig_inx->tme[i];
            inx->inx[i] = orig_inx->inx[i];
        }
    } else {
        memcpy(inx->tme, orig_inx->tme, sizeof(double) * inx->num);
        memcpy(inx->inx, orig_inx->inx, sizeof(int) * inx->num);
    }
    
    // Initialize caching and performance tracking
    inx->cache_valid = 0;
    inx->last_search_time = 0.0;
    inx->last_search_index = -1;
    
    return inx;
}

/**
 * Free parallel grid index
 */
void grid_parallel_index_free(struct GridIndexParallel *inx) {
    if (!inx) return;
    
    if (inx->tme) free(inx->tme);
    if (inx->inx) free(inx->inx);
    free(inx);
}

/**
 * Enhanced grid cell indexing with spatial optimization
 */
int grid_parallel_index_cell(struct GridDataParallel *grd, double mlat, double mlon,
                            struct GridPerformanceStats *stats) {
    if (!grd || !grd->data) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    double start_time = omp_get_wtime();
    
    // Calculate grid cell index using standard SuperDARN formula
    // Grid latitude spacing is 1 degree
    int grd_lat = (int)floor(mlat) - (mlat < 0 ? 1 : 0);
    
    // Grid longitude spacing depends on latitude
    double lat_rad = fabs(mlat) * M_PI / 180.0;
    double lon_spacing = 360.0 * cos(lat_rad);
    int grd_lon = (int)floor(mlon * lon_spacing / 360.0);
    
    // Calculate reference index
    int index;
    if (mlat >= 0) {
        index = 1000 * grd_lat + grd_lon;
    } else {
        index = -1000 * (-grd_lat) - grd_lon;
    }
    
    if (stats) {
        stats->index_calculations++;
        stats->total_index_time += omp_get_wtime() - start_time;
    }
    
    return index;
}

/**
 * Batch cell location for multiple indices
 */
int grid_parallel_locate_cells_batch(int npnt, struct GridGVecParallel *ptr,
                                    int *indices, int num_indices, int *results,
                                    struct GridPerformanceStats *stats) {
    if (!ptr || !indices || !results || npnt <= 0 || num_indices <= 0) {
        if (stats) stats->error_count++;
        return -1;
    }
    
    double start_time = omp_get_wtime();
    int found_count = 0;
    
    #pragma omp parallel for reduction(+:found_count)
    for (int i = 0; i < num_indices; i++) {
        results[i] = grid_parallel_locate_cell(npnt, ptr, indices[i], NULL);
        if (results[i] < npnt) found_count++;
    }
    
    if (stats) {
        stats->batch_locates++;
        stats->batch_locate_time += omp_get_wtime() - start_time;
        stats->batch_locate_hits += found_count;
        stats->batch_locate_misses += (num_indices - found_count);
    }
    
    return found_count;
}
