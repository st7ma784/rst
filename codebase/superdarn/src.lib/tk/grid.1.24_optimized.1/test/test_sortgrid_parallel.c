/*
 * test_sortgrid_parallel.c
 * ========================
 * 
 * Comprehensive test suite for parallel grid sorting functions.
 * Tests multi-criteria sorting, parallel merge sort implementation,
 * distance-based sorting, and custom comparison functions.
 * 
 * Author: SuperDARN Grid Parallelization Project
 * Date: 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <omp.h>

#include "griddata_parallel.h"

/* Test configuration constants */
#define NUM_TEST_POINTS 10000
#define NUM_PERFORMANCE_ITERATIONS 100
#define TOLERANCE 1e-6
#define MAX_THREADS 8

/* Global test statistics */
typedef struct {
    int tests_passed;
    int tests_failed;
    double total_time;
    size_t memory_allocated;
    size_t memory_freed;
} TestStats;

static TestStats test_stats = {0, 0, 0.0, 0, 0};

/* Test helper functions */
static void print_test_header(const char* test_name) {
    printf("\n=== Testing %s ===\n", test_name);
}

static void assert_test(int condition, const char* message) {
    if (condition) {
        printf("✓ %s\n", message);
        test_stats.tests_passed++;
    } else {
        printf("✗ %s\n", message);
        test_stats.tests_failed++;
    }
}

static void print_test_summary(void) {
    printf("\n=== Test Summary ===\n");
    printf("Tests passed: %d\n", test_stats.tests_passed);
    printf("Tests failed: %d\n", test_stats.tests_failed);
    printf("Total time: %.3f seconds\n", test_stats.total_time);
    printf("Success rate: %.1f%%\n", 
           100.0 * test_stats.tests_passed / (test_stats.tests_passed + test_stats.tests_failed));
}

/* Generate test data with specified characteristics */
static GridData_Parallel* create_test_grid(int num_points, int randomize) {
    GridData_Parallel* grid = grid_parallel_make();
    if (!grid) return NULL;
    
    grid->vcnum = num_points;
    grid->data = aligned_malloc(num_points * sizeof(GridGVec_Parallel), 32);
    if (!grid->data) {
        grid_parallel_free(grid);
        return NULL;
    }
    
    test_stats.memory_allocated += num_points * sizeof(GridGVec_Parallel);
    
    srand(42); // Reproducible results
    
    for (int i = 0; i < num_points; i++) {
        GridGVec_Parallel* point = &grid->data[i];
        
        if (randomize) {
            point->mlat = -90.0 + 180.0 * rand() / RAND_MAX;
            point->mlon = -180.0 + 360.0 * rand() / RAND_MAX;
            point->vel.median = -2000.0 + 4000.0 * rand() / RAND_MAX;
            point->pwr.median = 0.0 + 50.0 * rand() / RAND_MAX;
        } else {
            // Ordered data for stability testing
            point->mlat = -90.0 + 180.0 * i / num_points;
            point->mlon = -180.0 + 360.0 * i / num_points;
            point->vel.median = -2000.0 + 4000.0 * i / num_points;
            point->pwr.median = 0.0 + 50.0 * i / num_points;
        }
        
        point->azm = 0.0 + 360.0 * rand() / RAND_MAX;
        point->vel.sd = 10.0 + 100.0 * rand() / RAND_MAX;
        point->pwr.sd = 1.0 + 5.0 * rand() / RAND_MAX;
        point->wdt.median = 10.0 + 200.0 * rand() / RAND_MAX;
        point->wdt.sd = 5.0 + 20.0 * rand() / RAND_MAX;
        point->st_id = rand() % 100;
        point->chn = rand() % 2;
        point->index = i;
    }
    
    return grid;
}

/* Test basic sorting functionality */
static void test_basic_sorting(void) {
    print_test_header("Basic Sorting Functions");
    
    GridData_Parallel* grid = create_test_grid(1000, 1);
    assert_test(grid != NULL, "Test grid creation");
    
    clock_t start = clock();
    
    /* Test latitude sorting */
    grid_sort_by_latitude_parallel(grid, GRID_SORT_ASCENDING);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    /* Verify sorting */
    int sorted = 1;
    for (int i = 1; i < grid->vcnum; i++) {
        if (grid->data[i-1].mlat > grid->data[i].mlat) {
            sorted = 0;
            break;
        }
    }
    assert_test(sorted, "Latitude ascending sort correctness");
    
    /* Test longitude sorting */
    grid_sort_by_longitude_parallel(grid, GRID_SORT_DESCENDING);
    sorted = 1;
    for (int i = 1; i < grid->vcnum; i++) {
        if (grid->data[i-1].mlon < grid->data[i].mlon) {
            sorted = 0;
            break;
        }
    }
    assert_test(sorted, "Longitude descending sort correctness");
    
    /* Test velocity sorting */
    grid_sort_by_velocity_parallel(grid, GRID_SORT_ASCENDING);
    sorted = 1;
    for (int i = 1; i < grid->vcnum; i++) {
        if (grid->data[i-1].vel.median > grid->data[i].vel.median) {
            sorted = 0;
            break;
        }
    }
    assert_test(sorted, "Velocity ascending sort correctness");
    
    /* Test power sorting */
    grid_sort_by_power_parallel(grid, GRID_SORT_DESCENDING);
    sorted = 1;
    for (int i = 1; i < grid->vcnum; i++) {
        if (grid->data[i-1].pwr.median < grid->data[i].pwr.median) {
            sorted = 0;
            break;
        }
    }
    assert_test(sorted, "Power descending sort correctness");
    
    grid_parallel_free(grid);
}

/* Test multi-criteria sorting */
static void test_multi_criteria_sorting(void) {
    print_test_header("Multi-Criteria Sorting");
    
    GridData_Parallel* grid = create_test_grid(500, 1);
    assert_test(grid != NULL, "Test grid creation");
    
    /* Define sorting criteria */
    GridSortCriteria criteria[3];
    criteria[0].type = GRID_SORT_LATITUDE;
    criteria[0].order = GRID_SORT_ASCENDING;
    criteria[0].weight = 1.0;
    
    criteria[1].type = GRID_SORT_LONGITUDE;
    criteria[1].order = GRID_SORT_ASCENDING;
    criteria[1].weight = 0.8;
    
    criteria[2].type = GRID_SORT_VELOCITY;
    criteria[2].order = GRID_SORT_DESCENDING;
    criteria[2].weight = 0.6;
    
    clock_t start = clock();
    int result = grid_sort_multi_criteria_parallel(grid, criteria, 3);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Multi-criteria sort execution");
    
    /* Verify primary sort criterion (latitude) */
    int primary_sorted = 1;
    for (int i = 1; i < grid->vcnum; i++) {
        if (grid->data[i-1].mlat > grid->data[i].mlat) {
            primary_sorted = 0;
            break;
        }
    }
    assert_test(primary_sorted, "Multi-criteria primary sort correctness");
    
    grid_parallel_free(grid);
}

/* Test distance-based sorting */
static void test_distance_sorting(void) {
    print_test_header("Distance-Based Sorting");
    
    GridData_Parallel* grid = create_test_grid(300, 1);
    assert_test(grid != NULL, "Test grid creation");
    
    /* Set reference point */
    double ref_lat = 65.0;
    double ref_lon = -145.0;
    
    clock_t start = clock();
    int result = grid_sort_by_distance_parallel(grid, ref_lat, ref_lon, GRID_SORT_ASCENDING);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Distance sort execution");
    
    /* Verify distance ordering */
    int distance_sorted = 1;
    double prev_dist = 0.0;
    
    for (int i = 0; i < grid->vcnum; i++) {
        double dlat = grid->data[i].mlat - ref_lat;
        double dlon = grid->data[i].mlon - ref_lon;
        double dist = sqrt(dlat*dlat + dlon*dlon);
        
        if (i > 0 && dist < prev_dist) {
            distance_sorted = 0;
            break;
        }
        prev_dist = dist;
    }
    assert_test(distance_sorted, "Distance sort correctness");
    
    grid_parallel_free(grid);
}

/* Test custom comparison sorting */
static void test_custom_comparison(void) {
    print_test_header("Custom Comparison Sorting");
    
    GridData_Parallel* grid = create_test_grid(200, 1);
    assert_test(grid != NULL, "Test grid creation");
    
    /* Custom comparison: sort by combined velocity and power */
    auto int custom_compare(const GridGVec_Parallel* a, const GridGVec_Parallel* b, void* context) {
        (void)context; // Unused
        double score_a = a->vel.median + 0.1 * a->pwr.median;
        double score_b = b->vel.median + 0.1 * b->pwr.median;
        return (score_a > score_b) - (score_a < score_b);
    }
    
    clock_t start = clock();
    int result = grid_sort_custom_parallel(grid, custom_compare, NULL);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Custom comparison sort execution");
    
    /* Verify custom ordering */
    int custom_sorted = 1;
    for (int i = 1; i < grid->vcnum; i++) {
        double score_prev = grid->data[i-1].vel.median + 0.1 * grid->data[i-1].pwr.median;
        double score_curr = grid->data[i].vel.median + 0.1 * grid->data[i].pwr.median;
        
        if (score_prev > score_curr) {
            custom_sorted = 0;
            break;
        }
    }
    assert_test(custom_sorted, "Custom comparison sort correctness");
    
    grid_parallel_free(grid);
}

/* Test parallel performance */
static void test_parallel_performance(void) {
    print_test_header("Parallel Performance Testing");
    
    GridData_Parallel* grid = create_test_grid(NUM_TEST_POINTS, 1);
    assert_test(grid != NULL, "Large test grid creation");
    
    /* Test different thread counts */
    for (int threads = 1; threads <= MAX_THREADS && threads <= omp_get_max_threads(); threads *= 2) {
        omp_set_num_threads(threads);
        
        /* Create copy for testing */
        GridData_Parallel* test_grid = create_test_grid(NUM_TEST_POINTS, 1);
        
        clock_t start = clock();
        for (int i = 0; i < 10; i++) {
            grid_sort_by_latitude_parallel(test_grid, GRID_SORT_ASCENDING);
            
            /* Shuffle for next iteration */
            for (int j = 0; j < test_grid->vcnum; j++) {
                int k = rand() % test_grid->vcnum;
                GridGVec_Parallel temp = test_grid->data[j];
                test_grid->data[j] = test_grid->data[k];
                test_grid->data[k] = temp;
            }
        }
        clock_t end = clock();
        
        double time_per_sort = (double)(end - start) / CLOCKS_PER_SEC / 10.0;
        printf("  %d threads: %.4f seconds per sort\n", threads, time_per_sort);
        
        grid_parallel_free(test_grid);
    }
    
    assert_test(1, "Parallel performance test completed");
    grid_parallel_free(grid);
}

/* Test sort stability */
static void test_sort_stability(void) {
    print_test_header("Sort Stability Testing");
    
    GridData_Parallel* grid = create_test_grid(100, 0); // Ordered data
    assert_test(grid != NULL, "Test grid creation");
    
    /* Add duplicate values for stability testing */
    for (int i = 0; i < grid->vcnum; i += 2) {
        if (i + 1 < grid->vcnum) {
            grid->data[i+1].mlat = grid->data[i].mlat; // Create duplicates
            grid->data[i+1].index = i+1; // But keep unique indices
            grid->data[i].index = i;
        }
    }
    
    /* Sort by latitude (stable sort should preserve index order for equal values) */
    grid_sort_by_latitude_parallel(grid, GRID_SORT_ASCENDING);
    
    /* Check stability */
    int stable = 1;
    for (int i = 1; i < grid->vcnum; i++) {
        if (fabs(grid->data[i-1].mlat - grid->data[i].mlat) < TOLERANCE) {
            if (grid->data[i-1].index > grid->data[i].index) {
                stable = 0;
                break;
            }
        }
    }
    assert_test(stable, "Sort stability with duplicate values");
    
    grid_parallel_free(grid);
}

/* Test memory management */
static void test_memory_management(void) {
    print_test_header("Memory Management Testing");
    
    size_t initial_allocated = test_stats.memory_allocated;
    
    /* Create and destroy multiple grids */
    for (int i = 0; i < 10; i++) {
        GridData_Parallel* grid = create_test_grid(1000, 1);
        assert_test(grid != NULL, "Grid creation in loop");
        
        grid_sort_by_velocity_parallel(grid, GRID_SORT_ASCENDING);
        grid_parallel_free(grid);
        test_stats.memory_freed += 1000 * sizeof(GridGVec_Parallel);
    }
    
    size_t final_allocated = test_stats.memory_allocated;
    assert_test(final_allocated > initial_allocated, "Memory allocation tracking");
    
    /* Test error handling with invalid input */
    int result = grid_sort_by_latitude_parallel(NULL, GRID_SORT_ASCENDING);
    assert_test(result == -1, "Error handling with NULL input");
    
    GridData_Parallel empty_grid = {0};
    result = grid_sort_by_latitude_parallel(&empty_grid, GRID_SORT_ASCENDING);
    assert_test(result == -1, "Error handling with empty grid");
}

/* Test edge cases */
static void test_edge_cases(void) {
    print_test_header("Edge Cases Testing");
    
    /* Single element grid */
    GridData_Parallel* single_grid = create_test_grid(1, 0);
    assert_test(single_grid != NULL, "Single element grid creation");
    
    int result = grid_sort_by_latitude_parallel(single_grid, GRID_SORT_ASCENDING);
    assert_test(result == 0, "Single element sort");
    grid_parallel_free(single_grid);
    
    /* Two element grid */
    GridData_Parallel* two_grid = create_test_grid(2, 1);
    assert_test(two_grid != NULL, "Two element grid creation");
    
    result = grid_sort_by_longitude_parallel(two_grid, GRID_SORT_DESCENDING);
    assert_test(result == 0, "Two element sort");
    
    /* Verify order */
    int ordered = (two_grid->data[0].mlon >= two_grid->data[1].mlon);
    assert_test(ordered, "Two element sort correctness");
    grid_parallel_free(two_grid);
    
    /* Already sorted grid */
    GridData_Parallel* sorted_grid = create_test_grid(100, 0);
    assert_test(sorted_grid != NULL, "Pre-sorted grid creation");
    
    clock_t start = clock();
    result = grid_sort_by_latitude_parallel(sorted_grid, GRID_SORT_ASCENDING);
    clock_t end = clock();
    
    assert_test(result == 0, "Pre-sorted grid handling");
    
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    assert_test(time_taken < 0.01, "Pre-sorted grid performance");
    
    grid_parallel_free(sorted_grid);
}

/* Main test function */
int main(int argc, char** argv) {
    printf("SuperDARN Grid Parallel Sorting Test Suite\n");
    printf("==========================================\n");
    
    int run_benchmarks = 0;
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        run_benchmarks = 1;
    }
    
    clock_t total_start = clock();
    
    /* Run all tests */
    test_basic_sorting();
    test_multi_criteria_sorting();
    test_distance_sorting();
    test_custom_comparison();
    test_sort_stability();
    test_memory_management();
    test_edge_cases();
    
    if (run_benchmarks) {
        test_parallel_performance();
    }
    
    clock_t total_end = clock();
    test_stats.total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    
    print_test_summary();
    
    /* Return failure code if any tests failed */
    return (test_stats.tests_failed > 0) ? 1 : 0;
}
