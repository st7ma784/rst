/**
 * test_copygrid_parallel.c
 * Comprehensive test suite for parallel copy grid operations
 * 
 * Tests the correctness and performance of parallel grid copying
 * algorithms with memory optimization and selective filtering.
 * 
 * Author: SuperDARN Parallel Processing Team
 * Date: 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "griddata_parallel.h"

// Test configuration
#define MAX_COPY_CELLS 8000
#define MAX_COPY_STATIONS 25
#define NUM_COPY_RUNS 5
#define COPY_TOLERANCE 1e-6

// Test statistics
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
    double avg_sequential_time;
    double avg_parallel_time;
    double total_test_time;
} CopyTestStats;

static CopyTestStats test_stats = {0, 0, 0, 0.0, 0.0, 0.0};

/**
 * Get high-resolution timestamp
 */
static double get_precise_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/**
 * Generate synthetic grid data for copy testing
 */
static GridData* create_copy_test_grid(int num_cells, int num_stations, int pattern) {
    GridData* grid = malloc(sizeof(GridData));
    if (!grid) return NULL;
    
    grid->st_time = time(NULL) - 7200; // 2 hours ago
    grid->ed_time = time(NULL);
    grid->vcnum = num_cells;
    grid->stnum = num_stations;
    grid->xtd = 0;
    
    // Allocate and populate velocity cells
    grid->data = malloc(num_cells * sizeof(GridGVec));
    for (int i = 0; i < num_cells; i++) {
        switch (pattern) {
            case 0: // Linear distribution
                grid->data[i].mlat = -80.0 + (i * 160.0) / num_cells;
                grid->data[i].mlon = -180.0 + (i * 360.0) / num_cells;
                break;
                
            case 1: // Arctic focus
                grid->data[i].mlat = 50.0 + (i * 40.0) / num_cells;
                grid->data[i].mlon = -150.0 + (i * 60.0) / num_cells;
                break;
                
            case 2: // High latitude sparse
                grid->data[i].mlat = 60.0 + (i % 30);
                grid->data[i].mlon = -180.0 + (i * 7 % 360);
                break;
                
            default: // Random distribution
                grid->data[i].mlat = -90.0 + (rand() % 180);
                grid->data[i].mlon = -180.0 + (rand() % 360);
        }
        
        grid->data[i].kvect = 3 + (i % 18);
        grid->data[i].vel.median = -600.0 + (i % 1200);
        grid->data[i].vel.sd = 8.0 + (i % 40);
        grid->data[i].pwr.median = 3.0 + (i % 55);
        grid->data[i].pwr.sd = 1.5 + (i % 10);
        grid->data[i].wdt.median = 25.0 + (i % 200);
        grid->data[i].wdt.sd = 3.0 + (i % 20);
        grid->data[i].st_id = i % num_stations;
        
        // Add some noise for realistic testing
        double noise = ((double)rand() / RAND_MAX - 0.5) * 15.0;
        grid->data[i].vel.median += noise;
    }
    
    // Allocate station data
    grid->sdata = malloc(num_stations * sizeof(GridSVec));
    for (int i = 0; i < num_stations; i++) {
        grid->sdata[i].st_id = i;
        grid->sdata[i].chn = i % 2;
        grid->sdata[i].npnt = num_cells / num_stations + (i % 7);
        grid->sdata[i].freq0 = 8.0 + i * 2;
        grid->sdata[i].major_revision = 3;
        grid->sdata[i].minor_revision = 0;
        grid->sdata[i].prog_id = 1;
        
        // Station noise characteristics
        grid->sdata[i].noise.mean = 3.0 + (i % 8);
        grid->sdata[i].noise.sd = 0.8 + (i % 4);
        
        // Parameter ranges
        grid->sdata[i].vel.min = -1200.0;
        grid->sdata[i].vel.max = 1200.0;
        grid->sdata[i].pwr.min = 0.0;
        grid->sdata[i].pwr.max = 65.0;
        grid->sdata[i].wdt.min = 0.0;
        grid->sdata[i].wdt.max = 400.0;
    }
    
    return grid;
}

/**
 * Free test grid
 */
static void free_copy_test_grid(GridData* grid) {
    if (grid) {
        free(grid->data);
        free(grid->sdata);
        free(grid);
    }
}

/**
 * Compare two grids for exact equality
 */
static int compare_copy_grids(const GridData* original, const GridData* copy) {
    if (!original || !copy) return 0;
    
    // Check basic parameters
    if (original->vcnum != copy->vcnum || 
        original->stnum != copy->stnum ||
        original->st_time != copy->st_time ||
        original->ed_time != copy->ed_time) {
        printf("Grid metadata mismatch\n");
        return 0;
    }
    
    // Check velocity data
    for (int i = 0; i < original->vcnum; i++) {
        const GridGVec *orig_cell = &original->data[i];
        const GridGVec *copy_cell = &copy->data[i];
        
        if (fabs(orig_cell->mlat - copy_cell->mlat) > COPY_TOLERANCE ||
            fabs(orig_cell->mlon - copy_cell->mlon) > COPY_TOLERANCE ||
            fabs(orig_cell->vel.median - copy_cell->vel.median) > COPY_TOLERANCE) {
            printf("Cell %d data mismatch\n", i);
            return 0;
        }
    }
    
    // Check station data
    for (int i = 0; i < original->stnum; i++) {
        if (original->sdata[i].st_id != copy->sdata[i].st_id ||
            original->sdata[i].chn != copy->sdata[i].chn) {
            printf("Station %d metadata mismatch\n", i);
            return 0;
        }
    }
    
    return 1;
}

/**
 * Test filter function for selective copying
 */
static int latitude_filter(const GridGVec *cell, void *data) {
    double *threshold = (double*)data;
    return (cell->mlat >= *threshold);
}

static int power_filter(const GridGVec *cell, void *data) {
    double *min_power = (double*)data;
    return (cell->pwr.median >= *min_power);
}

/**
 * Test basic grid copying functionality
 */
static void test_basic_copy(void) {
    printf("Testing basic grid copying...\n");
    test_stats.total_tests++;
    
    GridData* original = create_copy_test_grid(500, 12, 0);
    if (!original) {
        printf("FAIL: Could not create test grid for copying\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Perform parallel copy
    double start_time = get_precise_time();
    GridData* copy = GridCopyParallel(original);
    test_stats.avg_parallel_time += get_precise_time() - start_time;
    
    if (!copy) {
        printf("FAIL: Parallel copy returned NULL\n");
        test_stats.failed_tests++;
        free_copy_test_grid(original);
        return;
    }
    
    // Verify exact copy
    if (compare_copy_grids(original, copy)) {
        printf("PASS: Basic grid copy completed successfully\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Grid copy is not identical to original\n");
        test_stats.failed_tests++;
    }
    
    free_copy_test_grid(original);
    free_copy_test_grid(copy);
}

/**
 * Test selective copying with filters
 */
static void test_selective_copy(void) {
    printf("Testing selective grid copying...\n");
    test_stats.total_tests++;
    
    GridData* original = create_copy_test_grid(800, 15, 1);
    if (!original) {
        printf("FAIL: Could not create test grid for selective copying\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Test latitude filter (northern hemisphere only)
    double lat_threshold = 0.0;
    GridData* filtered_grid = GridCopySelectiveParallel(original, latitude_filter, &lat_threshold);
    
    if (!filtered_grid) {
        printf("FAIL: Selective copy with latitude filter returned NULL\n");
        test_stats.failed_tests++;
        free_copy_test_grid(original);
        return;
    }
    
    // Verify all cells meet filter criteria
    int filter_correct = 1;
    for (int i = 0; i < filtered_grid->vcnum; i++) {
        if (filtered_grid->data[i].mlat < lat_threshold) {
            filter_correct = 0;
            break;
        }
    }
    
    if (filter_correct && filtered_grid->vcnum < original->vcnum) {
        printf("PASS: Selective copy with latitude filter succeeded\n");
        printf("  Original cells: %d, Filtered cells: %d\n", 
               original->vcnum, filtered_grid->vcnum);
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Selective copy filter not applied correctly\n");
        test_stats.failed_tests++;
    }
    
    free_copy_test_grid(original);
    free_copy_test_grid(filtered_grid);
}

/**
 * Test region-based copying
 */
static void test_region_copy(void) {
    printf("Testing region-based grid copying...\n");
    test_stats.total_tests++;
    
    GridData* original = create_copy_test_grid(1000, 18, 2);
    if (!original) {
        printf("FAIL: Could not create test grid for region copying\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Test Arctic region extraction
    GridData* arctic_grid = GridCopyRegionParallel(original, 60.0, 90.0, -180.0, 180.0);
    
    if (!arctic_grid) {
        printf("FAIL: Region copy returned NULL\n");
        test_stats.failed_tests++;
        free_copy_test_grid(original);
        return;
    }
    
    // Verify all cells are in specified region
    int region_correct = 1;
    for (int i = 0; i < arctic_grid->vcnum; i++) {
        if (arctic_grid->data[i].mlat < 60.0 || arctic_grid->data[i].mlat > 90.0) {
            region_correct = 0;
            break;
        }
    }
    
    if (region_correct) {
        printf("PASS: Region-based copy succeeded\n");
        printf("  Original cells: %d, Arctic cells: %d\n", 
               original->vcnum, arctic_grid->vcnum);
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Region-based copy included cells outside region\n");
        test_stats.failed_tests++;
    }
    
    free_copy_test_grid(original);
    free_copy_test_grid(arctic_grid);
}

/**
 * Test copy edge cases
 */
static void test_copy_edge_cases(void) {
    printf("Testing copy edge cases...\n");
    
    // Test NULL pointer
    test_stats.total_tests++;
    GridData* null_copy = GridCopyParallel(NULL);
    
    if (null_copy == NULL) {
        printf("PASS: NULL pointer handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: NULL pointer not handled properly\n");
        test_stats.failed_tests++;
        GridFreeParallel(null_copy);
    }
    
    // Test empty grid
    test_stats.total_tests++;
    GridData* empty_grid = create_copy_test_grid(0, 0, 0);
    GridData* empty_copy = GridCopyParallel(empty_grid);
    
    if (empty_copy && empty_copy->vcnum == 0 && empty_copy->stnum == 0) {
        printf("PASS: Empty grid copy handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Empty grid copy failed\n");
        test_stats.failed_tests++;
    }
    
    free_copy_test_grid(empty_grid);
    free_copy_test_grid(empty_copy);
    
    // Test single cell grid
    test_stats.total_tests++;
    GridData* single_grid = create_copy_test_grid(1, 1, 0);
    GridData* single_copy = GridCopyParallel(single_grid);
    
    if (single_copy && compare_copy_grids(single_grid, single_copy)) {
        printf("PASS: Single cell copy handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Single cell copy failed\n");
        test_stats.failed_tests++;
    }
    
    free_copy_test_grid(single_grid);
    free_copy_test_grid(single_copy);
}

/**
 * Performance benchmarking for copying
 */
static void test_copy_performance(void) {
    printf("Running copy performance benchmarks...\n");
    
    int test_sizes[] = {1000, 2500, 5000, 8000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("Grid Size\tTime(ms)\tThroughput(cells/sec)\tMemory(MB)\n");
    printf("=========\t========\t====================\t==========\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int size = test_sizes[i];
        int stations = size / 80 + 3;
        
        double total_time = 0.0;
        double total_throughput = 0.0;
        
        for (int run = 0; run < NUM_COPY_RUNS; run++) {
            GridData* grid = create_copy_test_grid(size, stations, run % 4);
            if (!grid) continue;
            
            double start_time = get_precise_time();
            GridData* copy = GridCopyParallel(grid);
            double elapsed = get_precise_time() - start_time;
            
            if (copy) {
                total_time += elapsed * 1000.0;
                total_throughput += size / elapsed;
                GridFreeParallel(copy);
            }
            
            free_copy_test_grid(grid);
        }
        
        double avg_time = total_time / NUM_COPY_RUNS;
        double avg_throughput = total_throughput / NUM_COPY_RUNS;
        double memory_mb = (size * sizeof(GridGVec) + stations * sizeof(GridSVec)) / (1024.0 * 1024.0);
        
        printf("%8d\t%7.2f\t%19.0f\t%9.2f\n", 
               size, avg_time, avg_throughput, memory_mb);
    }
}

/**
 * Test memory optimization
 */
static void test_memory_optimization(void) {
    printf("Testing memory optimization features...\n");
    test_stats.total_tests++;
    
    GridData* large_grid = create_copy_test_grid(MAX_COPY_CELLS, MAX_COPY_STATIONS, 0);
    if (!large_grid) {
        printf("FAIL: Could not create large test grid\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Test power threshold filter (should significantly reduce size)
    double power_threshold = 25.0;
    GridData* optimized_grid = GridCopySelectiveParallel(large_grid, power_filter, &power_threshold);
    
    if (optimized_grid && optimized_grid->vcnum < large_grid->vcnum) {
        double reduction = 100.0 * (1.0 - (double)optimized_grid->vcnum / large_grid->vcnum);
        printf("PASS: Memory optimization achieved %.1f%% size reduction\n", reduction);
        printf("  Original: %d cells, Optimized: %d cells\n", 
               large_grid->vcnum, optimized_grid->vcnum);
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Memory optimization failed\n");
        test_stats.failed_tests++;
    }
    
    free_copy_test_grid(large_grid);
    if (optimized_grid) free_copy_test_grid(optimized_grid);
}

/**
 * Main test runner for copying
 */
int main(int argc, char* argv[]) {
    printf("=== SuperDARN Grid Parallel Copy Test Suite ===\n\n");
    
    // Parse command line arguments
    int run_benchmarks = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmarks = 1;
        }
    }
    
    double start_time = get_precise_time();
    
    // Initialize random seed for reproducible tests
    srand(54321);
    
    // Run test suite
    test_basic_copy();
    test_selective_copy();
    test_region_copy();
    test_copy_edge_cases();
    test_memory_optimization();
    
    if (run_benchmarks) {
        test_copy_performance();
    }
    
    test_stats.total_test_time = get_precise_time() - start_time;
    
    // Print results summary
    printf("\n=== Copy Test Results ===\n");
    printf("Total tests:     %d\n", test_stats.total_tests);
    printf("Tests passed:    %d\n", test_stats.passed_tests);
    printf("Tests failed:    %d\n", test_stats.failed_tests);
    printf("Success rate:    %.1f%%\n", 
           (100.0 * test_stats.passed_tests) / test_stats.total_tests);
    printf("Total test time: %.3f seconds\n", test_stats.total_test_time);
    
    if (test_stats.avg_parallel_time > 0) {
        printf("Avg parallel time: %.3f ms\n", test_stats.avg_parallel_time * 1000.0);
    }
    
#ifdef OPENMP_ENABLED
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif

    if (test_stats.failed_tests > 0) {
        printf("\nSome tests failed. Check output above for details.\n");
        return 1;
    }
    
    printf("\nAll copy tests passed successfully!\n");
    return 0;
}
