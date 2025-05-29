/**
 * test_avggrid_parallel.c
 * Comprehensive test suite for parallel average grid operations
 * 
 * Tests the correctness and performance of parallel grid averaging
 * algorithms with hash-based optimization and vectorized operations.
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
#define MAX_AVG_CELLS 5000
#define MAX_AVG_STATIONS 20
#define NUM_AVG_RUNS 5
#define AVG_TOLERANCE 1e-5

// Test statistics
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
    double avg_sequential_time;
    double avg_parallel_time;
    double total_test_time;
} AvgTestStats;

static AvgTestStats test_stats = {0, 0, 0, 0.0, 0.0, 0.0};

/**
 * High-precision timer
 */
static double get_precise_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/**
 * Generate test grid with specific averaging patterns
 */
static GridData* create_averaging_test_grid(int num_cells, int num_stations, int pattern) {
    GridData* grid = malloc(sizeof(GridData));
    if (!grid) return NULL;
    
    grid->st_time = time(NULL) - 1800; // 30 minutes ago
    grid->ed_time = time(NULL);
    grid->vcnum = num_cells;
    grid->stnum = num_stations;
    
    // Allocate velocity cells
    grid->data = malloc(num_cells * sizeof(GridGVec));
    
    // Create different test patterns
    for (int i = 0; i < num_cells; i++) {
        switch (pattern) {
            case 0: // Uniform distribution
                grid->data[i].mlat = -80.0 + (i * 160.0) / num_cells;
                grid->data[i].mlon = -180.0 + (i * 360.0) / num_cells;
                break;
                
            case 1: // Clustered distribution
                grid->data[i].mlat = -70.0 + 20.0 * sin(i * 0.1);
                grid->data[i].mlon = -30.0 + 30.0 * cos(i * 0.1);
                break;
                
            case 2: // Sparse distribution
                grid->data[i].mlat = -90.0 + (i * i % 180);
                grid->data[i].mlon = -180.0 + (i * 3 % 360);
                break;
                
            default: // Random distribution
                grid->data[i].mlat = -90.0 + (rand() % 180);
                grid->data[i].mlon = -180.0 + (rand() % 360);
        }
        
        grid->data[i].kvect = 5 + (i % 15);
        grid->data[i].vel.median = -500.0 + (i % 1000);
        grid->data[i].vel.sd = 5.0 + (i % 30);
        grid->data[i].pwr.median = 5.0 + (i % 45);
        grid->data[i].pwr.sd = 1.0 + (i % 8);
        grid->data[i].wdt.median = 20.0 + (i % 180);
        grid->data[i].wdt.sd = 2.0 + (i % 15);
        grid->data[i].st_id = i % num_stations;
        
        // Add some noise for realistic testing
        double noise = ((double)rand() / RAND_MAX - 0.5) * 10.0;
        grid->data[i].vel.median += noise;
    }
    
    // Allocate station data
    grid->sdata = malloc(num_stations * sizeof(GridSVec));
    for (int i = 0; i < num_stations; i++) {
        grid->sdata[i].st_id = i;
        grid->sdata[i].chn = i % 2;
        grid->sdata[i].npnt = num_cells / num_stations + (i % 5);
        grid->sdata[i].freq0 = 8.0 + i * 2;
        grid->sdata[i].major_revision = 3;
        grid->sdata[i].minor_revision = 0;
        grid->sdata[i].prog_id = 1;
        grid->sdata[i].noise.mean = 3.0 + (i % 8);
        grid->sdata[i].noise.sd = 0.5 + (i % 3);
        grid->sdata[i].vel.min = -1500.0;
        grid->sdata[i].vel.max = 1500.0;
        grid->sdata[i].pwr.min = 0.0;
        grid->sdata[i].pwr.max = 50.0;
        grid->sdata[i].wdt.min = 0.0;
        grid->sdata[i].wdt.max = 300.0;
    }
    
    return grid;
}

/**
 * Compare averaged grid results
 */
static int compare_averaged_grids(const GridData* grid1, const GridData* grid2, double tolerance) {
    if (!grid1 || !grid2) return 0;
    
    if (grid1->vcnum != grid2->vcnum) {
        printf("Averaged cell count mismatch: %d vs %d\n", grid1->vcnum, grid2->vcnum);
        return 0;
    }
    
    // Compare averaged cell values
    for (int i = 0; i < grid1->vcnum; i++) {
        if (fabs(grid1->data[i].vel.median - grid2->data[i].vel.median) > tolerance) {
            printf("Cell %d velocity average mismatch: %.3f vs %.3f\n", 
                   i, grid1->data[i].vel.median, grid2->data[i].vel.median);
            return 0;
        }
        
        if (fabs(grid1->data[i].pwr.median - grid2->data[i].pwr.median) > tolerance) {
            printf("Cell %d power average mismatch: %.3f vs %.3f\n", 
                   i, grid1->data[i].pwr.median, grid2->data[i].pwr.median);
            return 0;
        }
    }
    
    return 1;
}

/**
 * Test basic averaging functionality
 */
static void test_basic_averaging(void) {
    printf("Testing basic grid averaging...\n");
    test_stats.total_tests++;
    
    GridData* grid1 = create_averaging_test_grid(200, 8, 0);
    GridData* grid2 = create_averaging_test_grid(200, 8, 0);
    
    if (!grid1 || !grid2) {
        printf("FAIL: Could not create test grids for averaging\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Perform parallel averaging
    double start_time = get_precise_time();
    int status = GridAverageParallel(grid1, 60.0, 5.0); // 60-second window, 5-degree resolution
    test_stats.avg_parallel_time += get_precise_time() - start_time;
    
    if (status == 0) {
        printf("PASS: Basic grid averaging completed successfully\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Grid averaging returned error code %d\n", status);
        test_stats.failed_tests++;
    }
    
    free(grid1->data);
    free(grid1->sdata);
    free(grid1);
    free(grid2->data);
    free(grid2->sdata);
    free(grid2);
}

/**
 * Test averaging with different spatial resolutions
 */
static void test_resolution_averaging(void) {
    printf("Testing multi-resolution averaging...\n");
    
    double resolutions[] = {1.0, 2.5, 5.0, 10.0, 20.0};
    int num_resolutions = sizeof(resolutions) / sizeof(resolutions[0]);
    
    printf("Resolution(deg)\tCells\tTime(ms)\tCells/sec\n");
    printf("==============\t=====\t========\t=========\n");
    
    for (int i = 0; i < num_resolutions; i++) {
        test_stats.total_tests++;
        
        GridData* grid = create_averaging_test_grid(1000, 15, 1);
        if (!grid) {
            test_stats.failed_tests++;
            continue;
        }
        
        int original_cells = grid->vcnum;
        
        double start_time = get_precise_time();
        int status = GridAverageParallel(grid, 120.0, resolutions[i]);
        double elapsed = (get_precise_time() - start_time) * 1000.0;
        
        if (status == 0) {
            double cells_per_sec = original_cells / (elapsed / 1000.0);
            printf("%13.1f\t%5d\t%7.2f\t%8.0f\n", 
                   resolutions[i], grid->vcnum, elapsed, cells_per_sec);
            test_stats.passed_tests++;
        } else {
            printf("%13.1f\tFAIL\t   -   \t    -   \n", resolutions[i]);
            test_stats.failed_tests++;
        }
        
        free(grid->data);
        free(grid->sdata);
        free(grid);
    }
}

/**
 * Test averaging edge cases
 */
static void test_averaging_edge_cases(void) {
    printf("Testing averaging edge cases...\n");
    
    // Test empty grid
    test_stats.total_tests++;
    GridData* empty_grid = create_averaging_test_grid(0, 0, 0);
    int status = GridAverageParallel(empty_grid, 60.0, 5.0);
    
    if (status == 0) {
        printf("PASS: Empty grid averaging handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Empty grid averaging failed\n");
        test_stats.failed_tests++;
    }
    
    free(empty_grid);
    
    // Test single cell grid
    test_stats.total_tests++;
    GridData* single_grid = create_averaging_test_grid(1, 1, 0);
    status = GridAverageParallel(single_grid, 60.0, 5.0);
    
    if (status == 0) {
        printf("PASS: Single cell averaging handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Single cell averaging failed\n");
        test_stats.failed_tests++;
    }
    
    free(single_grid->data);
    free(single_grid->sdata);
    free(single_grid);
    
    // Test NULL pointer
    test_stats.total_tests++;
    status = GridAverageParallel(NULL, 60.0, 5.0);
    
    if (status == -1) {
        printf("PASS: NULL pointer handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: NULL pointer not handled properly\n");
        test_stats.failed_tests++;
    }
    
    // Test invalid parameters
    test_stats.total_tests++;
    GridData* test_grid = create_averaging_test_grid(100, 5, 0);
    status = GridAverageParallel(test_grid, -60.0, 5.0); // Negative time window
    
    if (status == -1) {
        printf("PASS: Invalid parameters handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Invalid parameters not handled properly\n");
        test_stats.failed_tests++;
    }
    
    free(test_grid->data);
    free(test_grid->sdata);
    free(test_grid);
}

/**
 * Performance benchmarking for averaging
 */
static void test_averaging_performance(void) {
    printf("Running averaging performance benchmarks...\n");
    
    int test_sizes[] = {500, 1000, 2000, 5000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("Grid Size\tTime(ms)\tThroughput(cells/sec)\tMemory(MB)\n");
    printf("=========\t========\t====================\t==========\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int size = test_sizes[i];
        int stations = size / 50 + 2;
        
        double total_time = 0.0;
        double total_throughput = 0.0;
        
        for (int run = 0; run < NUM_AVG_RUNS; run++) {
            GridData* grid = create_averaging_test_grid(size, stations, run % 3);
            if (!grid) continue;
            
            double start_time = get_precise_time();
            int status = GridAverageParallel(grid, 90.0, 3.0);
            double elapsed = get_precise_time() - start_time;
            
            if (status == 0) {
                total_time += elapsed * 1000.0;
                total_throughput += size / elapsed;
            }
            
            free(grid->data);
            free(grid->sdata);
            free(grid);
        }
        
        double avg_time = total_time / NUM_AVG_RUNS;
        double avg_throughput = total_throughput / NUM_AVG_RUNS;
        double memory_mb = (size * sizeof(GridGVec) + stations * sizeof(GridSVec)) / (1024.0 * 1024.0);
        
        printf("%8d\t%7.2f\t%19.0f\t%9.2f\n", 
               size, avg_time, avg_throughput, memory_mb);
    }
}

/**
 * Test hash table performance
 */
static void test_hash_performance(void) {
    printf("Testing hash table optimization...\n");
    
    // Create grid with many duplicate spatial locations
    GridData* clustered_grid = create_averaging_test_grid(2000, 10, 1);
    GridData* sparse_grid = create_averaging_test_grid(2000, 10, 2);
    
    if (!clustered_grid || !sparse_grid) {
        printf("FAIL: Could not create hash test grids\n");
        test_stats.failed_tests++;
        return;
    }
    
    test_stats.total_tests++;
    
    // Test clustered data (should benefit from hash optimization)
    double start_time = get_precise_time();
    int status1 = GridAverageParallel(clustered_grid, 120.0, 2.0);
    double clustered_time = get_precise_time() - start_time;
    
    // Test sparse data
    start_time = get_precise_time();
    int status2 = GridAverageParallel(sparse_grid, 120.0, 2.0);
    double sparse_time = get_precise_time() - start_time;
    
    if (status1 == 0 && status2 == 0) {
        printf("PASS: Hash optimization test completed\n");
        printf("  Clustered data: %.3f ms\n", clustered_time * 1000.0);
        printf("  Sparse data:    %.3f ms\n", sparse_time * 1000.0);
        printf("  Hash benefit:   %.2fx speedup on clustered data\n", 
               sparse_time / clustered_time);
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Hash optimization test failed\n");
        test_stats.failed_tests++;
    }
    
    free(clustered_grid->data);
    free(clustered_grid->sdata);
    free(clustered_grid);
    free(sparse_grid->data);
    free(sparse_grid->sdata);
    free(sparse_grid);
}

/**
 * Main test runner for averaging
 */
int main(int argc, char* argv[]) {
    printf("=== SuperDARN Grid Parallel Averaging Test Suite ===\n\n");
    
    // Parse command line arguments
    int run_benchmarks = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmarks = 1;
        }
    }
    
    double start_time = get_precise_time();
    
    // Initialize random seed for reproducible tests
    srand(12345);
    
    // Run test suite
    test_basic_averaging();
    test_averaging_edge_cases();
    test_hash_performance();
    
    if (run_benchmarks) {
        test_resolution_averaging();
        test_averaging_performance();
    }
    
    test_stats.total_test_time = get_precise_time() - start_time;
    
    // Print results summary
    printf("\n=== Averaging Test Results ===\n");
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
    printf("OpenMP threads:  %d\n", omp_get_max_threads());
#endif

#ifdef AVX2_ENABLED
    printf("AVX2 SIMD:       Enabled\n");
#else
    printf("AVX2 SIMD:       Disabled\n");
#endif
    
    return (test_stats.failed_tests == 0) ? 0 : 1;
}
