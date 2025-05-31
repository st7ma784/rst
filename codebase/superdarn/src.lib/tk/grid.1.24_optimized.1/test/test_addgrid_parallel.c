/**
 * test_addgrid_parallel.c
 * Comprehensive test suite for parallel grid addition operations
 * 
 * Tests the correctness and performance of parallel grid arithmetic
 * operations including addition, subtraction, scaling, and statistical
 * uncertainty propagation.
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
#define MAX_ADD_CELLS 6000
#define MAX_ADD_STATIONS 20
#define NUM_ADD_RUNS 5
#define ADD_TOLERANCE 1e-5

// Test statistics
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
    double avg_sequential_time;
    double avg_parallel_time;
    double total_test_time;
} AddTestStats;

static AddTestStats test_stats = {0, 0, 0, 0.0, 0.0, 0.0};

/**
 * Get high-resolution timestamp
 */
static double get_precise_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/**
 * Generate synthetic grid data for addition testing
 */
static GridData* create_add_test_grid(int num_cells, int num_stations, int pattern, float scale) {
    GridData* grid = malloc(sizeof(GridData));
    if (!grid) return NULL;
    
    grid->st_time = time(NULL) - 3600; // 1 hour ago
    grid->ed_time = time(NULL);
    grid->vcnum = num_cells;
    grid->stnum = num_stations;
    grid->xtd = 0;
    
    // Allocate and populate velocity cells
    grid->data = malloc(num_cells * sizeof(GridGVec));
    for (int i = 0; i < num_cells; i++) {
        switch (pattern) {
            case 0: // Linear distribution
                grid->data[i].mlat = -75.0 + (i * 150.0) / num_cells;
                grid->data[i].mlon = -150.0 + (i * 300.0) / num_cells;
                break;
                
            case 1: // Polar regions
                grid->data[i].mlat = 70.0 + (i * 20.0) / num_cells;
                grid->data[i].mlon = -120.0 + (i * 240.0) / num_cells;
                break;
                
            case 2: // Mid latitudes
                grid->data[i].mlat = -45.0 + (i * 90.0) / num_cells;
                grid->data[i].mlon = -60.0 + (i * 120.0) / num_cells;
                break;
                
            default: // Random distribution
                grid->data[i].mlat = -90.0 + (rand() % 180);
                grid->data[i].mlon = -180.0 + (rand() % 360);
        }
        
        grid->data[i].kvect = 4 + (i % 16);
        
        // Scale values by pattern multiplier
        grid->data[i].vel.median = scale * (50.0 + (i % 400));
        grid->data[i].vel.sd = scale * (5.0 + (i % 25));
        grid->data[i].pwr.median = scale * (10.0 + (i % 35));
        grid->data[i].pwr.sd = scale * (2.0 + (i % 8));
        grid->data[i].wdt.median = scale * (30.0 + (i % 170));
        grid->data[i].wdt.sd = scale * (4.0 + (i % 15));
        grid->data[i].st_id = i % num_stations;
        
        // Add some noise for realistic testing
        double noise = ((double)rand() / RAND_MAX - 0.5) * 10.0 * scale;
        grid->data[i].vel.median += noise;
    }
    
    // Allocate station data
    grid->sdata = malloc(num_stations * sizeof(GridSVec));
    for (int i = 0; i < num_stations; i++) {
        grid->sdata[i].st_id = i;
        grid->sdata[i].chn = i % 2;
        grid->sdata[i].npnt = num_cells / num_stations + (i % 6);
        grid->sdata[i].freq0 = 9.0 + i * 2;
        grid->sdata[i].major_revision = 3;
        grid->sdata[i].minor_revision = 0;
        grid->sdata[i].prog_id = 1;
        
        // Station noise characteristics
        grid->sdata[i].noise.mean = 4.0 + (i % 7);
        grid->sdata[i].noise.sd = 1.0 + (i % 3);
        
        // Parameter ranges
        grid->sdata[i].vel.min = -800.0;
        grid->sdata[i].vel.max = 800.0;
        grid->sdata[i].pwr.min = 0.0;
        grid->sdata[i].pwr.max = 50.0;
        grid->sdata[i].wdt.min = 0.0;
        grid->sdata[i].wdt.max = 300.0;
    }
    
    return grid;
}

/**
 * Free test grid
 */
static void free_add_test_grid(GridData* grid) {
    if (grid) {
        free(grid->data);
        free(grid->sdata);
        free(grid);
    }
}

/**
 * Verify grid addition result
 */
static int verify_addition(const GridData* grid1, const GridData* grid2, const GridData* result) {
    if (!grid1 || !grid2 || !result) return 0;
    
    // Basic structure checks
    if (result->vcnum != grid1->vcnum) {
        printf("Addition result has wrong number of cells: %d vs %d\n", 
               result->vcnum, grid1->vcnum);
        return 0;
    }
    
    // Check some cells for correct addition
    int cells_to_check = result->vcnum < 100 ? result->vcnum : 100;
    for (int i = 0; i < cells_to_check; i++) {
        const GridGVec *c1 = &grid1->data[i];
        const GridGVec *cr = &result->data[i];
        
        // Find corresponding cell in grid2 by location
        const GridGVec *c2 = NULL;
        for (int j = 0; j < grid2->vcnum; j++) {
            if (fabs(grid2->data[j].mlat - c1->mlat) < ADD_TOLERANCE &&
                fabs(grid2->data[j].mlon - c1->mlon) < ADD_TOLERANCE) {
                c2 = &grid2->data[j];
                break;
            }
        }
        
        if (c2) {
            // Verify weighted addition
            double w1 = 1.0 / (c1->vel.sd * c1->vel.sd + 1e-6);
            double w2 = 1.0 / (c2->vel.sd * c2->vel.sd + 1e-6);
            double total_weight = w1 + w2;
            double expected_vel = (c1->vel.median * w1 + c2->vel.median * w2) / total_weight;
            
            if (fabs(cr->vel.median - expected_vel) > ADD_TOLERANCE) {
                printf("Cell %d velocity addition incorrect: %.3f vs %.3f\n", 
                       i, cr->vel.median, expected_vel);
                return 0;
            }
        }
    }
    
    return 1;
}

/**
 * Test basic grid addition functionality
 */
static void test_basic_addition(void) {
    printf("Testing basic grid addition...\n");
    test_stats.total_tests++;
    
    GridData* grid1 = create_add_test_grid(300, 8, 0, 1.0);
    GridData* grid2 = create_add_test_grid(300, 8, 0, 0.8);
    
    if (!grid1 || !grid2) {
        printf("FAIL: Could not create test grids for addition\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Perform parallel addition
    double start_time = get_precise_time();
    int status = GridAddParallel(grid1, grid2);
    test_stats.avg_parallel_time += get_precise_time() - start_time;
    
    if (status == 0) {
        printf("PASS: Basic grid addition completed successfully\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Grid addition failed with status %d\n", status);
        test_stats.failed_tests++;
    }
    
    free_add_test_grid(grid1);
    free_add_test_grid(grid2);
}

/**
 * Test grid subtraction functionality
 */
static void test_subtraction(void) {
    printf("Testing grid subtraction...\n");
    test_stats.total_tests++;
    
    GridData* grid1 = create_add_test_grid(400, 10, 1, 2.0);
    GridData* grid2 = create_add_test_grid(400, 10, 1, 1.0);
    
    if (!grid1 || !grid2) {
        printf("FAIL: Could not create test grids for subtraction\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Store original values for verification
    double orig_vel = grid1->data[0].vel.median;
    double subtract_vel = grid2->data[0].vel.median;
    
    // Perform parallel subtraction
    int status = GridSubtractParallel(grid1, grid2);
    
    if (status == 0) {
        // Verify subtraction (check first cell)
        double expected_vel = orig_vel - subtract_vel;
        if (fabs(grid1->data[0].vel.median - expected_vel) < ADD_TOLERANCE) {
            printf("PASS: Grid subtraction completed correctly\n");
            test_stats.passed_tests++;
        } else {
            printf("FAIL: Grid subtraction result incorrect: %.3f vs %.3f\n",
                   grid1->data[0].vel.median, expected_vel);
            test_stats.failed_tests++;
        }
    } else {
        printf("FAIL: Grid subtraction failed with status %d\n", status);
        test_stats.failed_tests++;
    }
    
    free_add_test_grid(grid1);
    free_add_test_grid(grid2);
}

/**
 * Test grid scaling functionality
 */
static void test_scaling(void) {
    printf("Testing grid scaling operations...\n");
    test_stats.total_tests++;
    
    GridData* grid = create_add_test_grid(500, 12, 2, 1.0);
    if (!grid) {
        printf("FAIL: Could not create test grid for scaling\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Store original values
    double orig_vel = grid->data[0].vel.median;
    double orig_pwr = grid->data[0].pwr.median;
    
    // Test scaling by factor of 2.5
    float scale_factor = 2.5f;
    int status = GridScaleParallel(grid, scale_factor);
    
    if (status == 0) {
        // Verify scaling
        double expected_vel = orig_vel * scale_factor;
        double expected_pwr = orig_pwr * scale_factor;
        
        if (fabs(grid->data[0].vel.median - expected_vel) < ADD_TOLERANCE &&
            fabs(grid->data[0].pwr.median - expected_pwr) < ADD_TOLERANCE) {
            printf("PASS: Grid scaling completed correctly (%.1fx)\n", scale_factor);
            test_stats.passed_tests++;
        } else {
            printf("FAIL: Grid scaling result incorrect\n");
            test_stats.failed_tests++;
        }
    } else {
        printf("FAIL: Grid scaling failed with status %d\n", status);
        test_stats.failed_tests++;
    }
    
    free_add_test_grid(grid);
}

/**
 * Test mathematical function application
 */
static float test_square_function(float x) {
    return x * x;
}

static float test_sqrt_function(float x) {
    return x >= 0 ? sqrtf(x) : 0.0f;
}

static void test_function_application(void) {
    printf("Testing mathematical function application...\n");
    test_stats.total_tests++;
    
    GridData* grid = create_add_test_grid(300, 8, 0, 1.0);
    if (!grid) {
        printf("FAIL: Could not create test grid for function application\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Store original value
    double orig_vel = fabs(grid->data[0].vel.median);
    
    // Apply square root function
    int status = GridApplyFunctionParallel(grid, test_sqrt_function);
    
    if (status == 0) {
        // Verify function application
        double expected_vel = sqrt(orig_vel);
        if (fabs(grid->data[0].vel.median - expected_vel) < ADD_TOLERANCE) {
            printf("PASS: Function application completed correctly\n");
            test_stats.passed_tests++;
        } else {
            printf("FAIL: Function application result incorrect: %.3f vs %.3f\n",
                   grid->data[0].vel.median, expected_vel);
            test_stats.failed_tests++;
        }
    } else {
        printf("FAIL: Function application failed with status %d\n", status);
        test_stats.failed_tests++;
    }
    
    free_add_test_grid(grid);
}

/**
 * Test addition edge cases
 */
static void test_addition_edge_cases(void) {
    printf("Testing addition edge cases...\n");
    
    // Test NULL pointer
    test_stats.total_tests++;
    int status = GridAddParallel(NULL, NULL);
    
    if (status == -1) {
        printf("PASS: NULL pointer handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: NULL pointer not handled properly\n");
        test_stats.failed_tests++;
    }
    
    // Test empty grids
    test_stats.total_tests++;
    GridData* empty_grid1 = create_add_test_grid(0, 0, 0, 1.0);
    GridData* empty_grid2 = create_add_test_grid(0, 0, 0, 1.0);
    
    status = GridAddParallel(empty_grid1, empty_grid2);
    
    if (status == 0) {
        printf("PASS: Empty grid addition handled correctly\n");
        test_stats.passed_tests++;
    } else {
        printf("FAIL: Empty grid addition failed\n");
        test_stats.failed_tests++;
    }
    
    free_add_test_grid(empty_grid1);
    free_add_test_grid(empty_grid2);
    
    // Test single cell grids
    test_stats.total_tests++;
    GridData* single_grid1 = create_add_test_grid(1, 1, 0, 2.0);
    GridData* single_grid2 = create_add_test_grid(1, 1, 0, 1.5);
    
    if (single_grid1 && single_grid2) {
        // Make them have the same location for proper addition
        single_grid2->data[0].mlat = single_grid1->data[0].mlat;
        single_grid2->data[0].mlon = single_grid1->data[0].mlon;
        
        status = GridAddParallel(single_grid1, single_grid2);
        
        if (status == 0) {
            printf("PASS: Single cell addition handled correctly\n");
            test_stats.passed_tests++;
        } else {
            printf("FAIL: Single cell addition failed\n");
            test_stats.failed_tests++;
        }
    } else {
        printf("FAIL: Could not create single cell test grids\n");
        test_stats.failed_tests++;
    }
    
    free_add_test_grid(single_grid1);
    free_add_test_grid(single_grid2);
}

/**
 * Performance benchmarking for addition operations
 */
static void test_addition_performance(void) {
    printf("Running addition performance benchmarks...\n");
    
    int test_sizes[] = {800, 2000, 4000, 6000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("Grid Size\tAdd Time(ms)\tSub Time(ms)\tScale Time(ms)\tMemory(MB)\n");
    printf("=========\t============\t============\t==============\t==========\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int size = test_sizes[i];
        int stations = size / 100 + 2;
        
        double add_time = 0.0, sub_time = 0.0, scale_time = 0.0;
        
        for (int run = 0; run < NUM_ADD_RUNS; run++) {
            // Test addition
            GridData* grid1 = create_add_test_grid(size, stations, 0, 1.0);
            GridData* grid2 = create_add_test_grid(size, stations, 0, 0.8);
            
            if (grid1 && grid2) {
                double start_time = get_precise_time();
                GridAddParallel(grid1, grid2);
                add_time += (get_precise_time() - start_time) * 1000.0;
            }
            
            free_add_test_grid(grid1);
            free_add_test_grid(grid2);
            
            // Test subtraction
            grid1 = create_add_test_grid(size, stations, 1, 2.0);
            grid2 = create_add_test_grid(size, stations, 1, 1.0);
            
            if (grid1 && grid2) {
                double start_time = get_precise_time();
                GridSubtractParallel(grid1, grid2);
                sub_time += (get_precise_time() - start_time) * 1000.0;
            }
            
            free_add_test_grid(grid1);
            free_add_test_grid(grid2);
            
            // Test scaling
            GridData* grid = create_add_test_grid(size, stations, 2, 1.0);
            if (grid) {
                double start_time = get_precise_time();
                GridScaleParallel(grid, 1.5f);
                scale_time += (get_precise_time() - start_time) * 1000.0;
            }
            
            free_add_test_grid(grid);
        }
        
        double avg_add = add_time / NUM_ADD_RUNS;
        double avg_sub = sub_time / NUM_ADD_RUNS;
        double avg_scale = scale_time / NUM_ADD_RUNS;
        double memory_mb = (size * sizeof(GridGVec) + stations * sizeof(GridSVec)) / (1024.0 * 1024.0);
        
        printf("%8d\t%11.2f\t%11.2f\t%13.2f\t%9.2f\n", 
               size, avg_add, avg_sub, avg_scale, memory_mb);
    }
}

/**
 * Test uncertainty propagation
 */
static void test_uncertainty_propagation(void) {
    printf("Testing statistical uncertainty propagation...\n");
    test_stats.total_tests++;
    
    GridData* grid1 = create_add_test_grid(200, 6, 0, 1.0);
    GridData* grid2 = create_add_test_grid(200, 6, 0, 1.0);
    
    if (!grid1 || !grid2) {
        printf("FAIL: Could not create test grids for uncertainty testing\n");
        test_stats.failed_tests++;
        return;
    }
    
    // Set up known uncertainty values
    grid1->data[0].vel.sd = 10.0;
    grid2->data[0].vel.sd = 15.0;
    grid2->data[0].mlat = grid1->data[0].mlat;  // Same location
    grid2->data[0].mlon = grid1->data[0].mlon;
    
    // Perform addition
    int status = GridAddParallel(grid1, grid2);
    
    if (status == 0) {
        // Check uncertainty propagation (should be weighted combination)
        double w1 = 1.0 / (10.0 * 10.0);
        double w2 = 1.0 / (15.0 * 15.0);
        double expected_sd = sqrt(1.0 / (w1 + w2));
        
        if (fabs(grid1->data[0].vel.sd - expected_sd) < ADD_TOLERANCE) {
            printf("PASS: Statistical uncertainty propagation correct\n");
            test_stats.passed_tests++;
        } else {
            printf("FAIL: Uncertainty propagation incorrect: %.3f vs %.3f\n",
                   grid1->data[0].vel.sd, expected_sd);
            test_stats.failed_tests++;
        }
    } else {
        printf("FAIL: Addition for uncertainty test failed\n");
        test_stats.failed_tests++;
    }
    
    free_add_test_grid(grid1);
    free_add_test_grid(grid2);
}

/**
 * Main test runner for addition operations
 */
int main(int argc, char* argv[]) {
    printf("=== SuperDARN Grid Parallel Addition Test Suite ===\n\n");
    
    // Parse command line arguments
    int run_benchmarks = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmarks = 1;
        }
    }
    
    double start_time = get_precise_time();
    
    // Initialize random seed for reproducible tests
    srand(98765);
    
    // Run test suite
    test_basic_addition();
    test_subtraction();
    test_scaling();
    test_function_application();
    test_addition_edge_cases();
    test_uncertainty_propagation();
    
    if (run_benchmarks) {
        test_addition_performance();
    }
    
    test_stats.total_test_time = get_precise_time() - start_time;
    
    // Print results summary
    printf("\n=== Addition Test Results ===\n");
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
    
    printf("\nAll addition tests passed successfully!\n");
    return 0;
}
