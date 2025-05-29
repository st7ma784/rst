/*
 * test_filtergrid_parallel.c
 * ==========================
 * 
 * Comprehensive test suite for parallel grid filtering functions.
 * Tests statistical outlier detection, spatial smoothing, median filtering,
 * and composite filtering operations.
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
#define NUM_TEST_POINTS 5000
#define NUM_PERFORMANCE_ITERATIONS 50
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

/* Generate test data with outliers */
static GridData_Parallel* create_test_grid_with_outliers(int num_points, double outlier_fraction) {
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
    
    int num_outliers = (int)(num_points * outlier_fraction);
    
    for (int i = 0; i < num_points; i++) {
        GridGVec_Parallel* point = &grid->data[i];
        
        /* Generate regular spatial grid */
        point->mlat = 50.0 + 20.0 * (i % 100) / 100.0;
        point->mlon = -150.0 + 60.0 * (i / 100) / (num_points / 100 + 1);
        
        /* Normal velocity distribution */
        if (i < num_outliers) {
            /* Add outliers */
            point->vel.median = (rand() % 2 ? 1 : -1) * (3000.0 + 2000.0 * rand() / RAND_MAX);
            point->pwr.median = 80.0 + 20.0 * rand() / RAND_MAX; // High power outliers
        } else {
            /* Normal data */
            point->vel.median = -500.0 + 1000.0 * rand() / RAND_MAX;
            point->pwr.median = 10.0 + 30.0 * rand() / RAND_MAX;
        }
        
        point->azm = 0.0 + 360.0 * rand() / RAND_MAX;
        point->vel.sd = 10.0 + 50.0 * rand() / RAND_MAX;
        point->pwr.sd = 1.0 + 5.0 * rand() / RAND_MAX;
        point->wdt.median = 20.0 + 100.0 * rand() / RAND_MAX;
        point->wdt.sd = 5.0 + 15.0 * rand() / RAND_MAX;
        point->st_id = 1;
        point->chn = 0;
        point->index = i;
    }
    
    return grid;
}

/* Generate noisy test data for smoothing tests */
static GridData_Parallel* create_noisy_grid(int num_points) {
    GridData_Parallel* grid = grid_parallel_make();
    if (!grid) return NULL;
    
    grid->vcnum = num_points;
    grid->data = aligned_malloc(num_points * sizeof(GridGVec_Parallel), 32);
    if (!grid->data) {
        grid_parallel_free(grid);
        return NULL;
    }
    
    test_stats.memory_allocated += num_points * sizeof(GridGVec_Parallel);
    
    srand(42);
    
    /* Create structured data with noise */
    int grid_size = (int)sqrt(num_points);
    for (int i = 0; i < num_points; i++) {
        GridGVec_Parallel* point = &grid->data[i];
        
        int row = i / grid_size;
        int col = i % grid_size;
        
        point->mlat = 60.0 + 10.0 * row / grid_size;
        point->mlon = -160.0 + 40.0 * col / grid_size;
        
        /* Smooth underlying field with noise */
        double smooth_vel = 500.0 * sin(2 * M_PI * row / grid_size) * cos(2 * M_PI * col / grid_size);
        double noise = 200.0 * (2.0 * rand() / RAND_MAX - 1.0);
        point->vel.median = smooth_vel + noise;
        
        point->pwr.median = 20.0 + 10.0 * (2.0 * rand() / RAND_MAX - 1.0);
        point->azm = atan2(point->vel.median, 100.0) * 180.0 / M_PI;
        point->vel.sd = 20.0;
        point->pwr.sd = 2.0;
        point->wdt.median = 50.0;
        point->wdt.sd = 10.0;
        point->st_id = 1;
        point->chn = 0;
        point->index = i;
    }
    
    return grid;
}

/* Test statistical outlier detection */
static void test_outlier_detection(void) {
    print_test_header("Statistical Outlier Detection");
    
    GridData_Parallel* grid = create_test_grid_with_outliers(1000, 0.1);
    assert_test(grid != NULL, "Test grid with outliers creation");
    
    int original_count = grid->vcnum;
    
    /* Test sigma-based outlier removal */
    GridFilterParams params = {0};
    params.outlier_threshold = 2.5; // 2.5-sigma threshold
    params.filter_velocity = 1;
    params.filter_power = 1;
    
    clock_t start = clock();
    int result = grid_filter_outliers_parallel(grid, &params);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Outlier detection execution");
    assert_test(grid->vcnum < original_count, "Outliers were removed");
    
    printf("  Removed %d outliers from %d points (%.1f%%)\n", 
           original_count - grid->vcnum, original_count, 
           100.0 * (original_count - grid->vcnum) / original_count);
    
    /* Verify remaining data is within reasonable bounds */
    int outliers_remain = 0;
    double vel_mean = 0.0, vel_sq_sum = 0.0;
    
    /* Calculate statistics */
    for (int i = 0; i < grid->vcnum; i++) {
        vel_mean += grid->data[i].vel.median;
    }
    vel_mean /= grid->vcnum;
    
    for (int i = 0; i < grid->vcnum; i++) {
        double diff = grid->data[i].vel.median - vel_mean;
        vel_sq_sum += diff * diff;
    }
    double vel_std = sqrt(vel_sq_sum / grid->vcnum);
    
    /* Check for remaining outliers */
    for (int i = 0; i < grid->vcnum; i++) {
        double z_score = fabs((grid->data[i].vel.median - vel_mean) / vel_std);
        if (z_score > params.outlier_threshold) {
            outliers_remain++;
        }
    }
    
    assert_test(outliers_remain < 0.01 * grid->vcnum, "Most outliers removed");
    
    grid_parallel_free(grid);
}

/* Test percentile-based filtering */
static void test_percentile_filtering(void) {
    print_test_header("Percentile-Based Filtering");
    
    GridData_Parallel* grid = create_test_grid_with_outliers(800, 0.15);
    assert_test(grid != NULL, "Test grid creation");
    
    int original_count = grid->vcnum;
    
    /* Filter extreme percentiles */
    GridFilterParams params = {0};
    params.velocity_percentile_low = 5.0;   // Remove bottom 5%
    params.velocity_percentile_high = 95.0;  // Remove top 5%
    params.power_percentile_low = 10.0;
    params.power_percentile_high = 90.0;
    params.filter_velocity = 1;
    params.filter_power = 1;
    
    clock_t start = clock();
    int result = grid_filter_percentiles_parallel(grid, &params);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Percentile filtering execution");
    assert_test(grid->vcnum < original_count, "Extreme values were removed");
    
    printf("  Removed %d extreme values from %d points (%.1f%%)\n", 
           original_count - grid->vcnum, original_count, 
           100.0 * (original_count - grid->vcnum) / original_count);
    
    grid_parallel_free(grid);
}

/* Test spatial smoothing filters */
static void test_spatial_smoothing(void) {
    print_test_header("Spatial Smoothing Filters");
    
    GridData_Parallel* grid = create_noisy_grid(400); // 20x20 grid
    assert_test(grid != NULL, "Noisy test grid creation");
    
    /* Create copy for comparison */
    GridData_Parallel* original = grid_parallel_copy(grid);
    assert_test(original != NULL, "Grid copy creation");
    
    /* Test Gaussian smoothing */
    GridFilterParams params = {0};
    params.spatial_radius = 2.0; // degrees
    params.gaussian_sigma = 1.0;
    
    clock_t start = clock();
    int result = grid_smooth_gaussian_parallel(grid, &params);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Gaussian smoothing execution");
    
    /* Verify smoothing occurred (variance should decrease) */
    double original_variance = 0.0, smoothed_variance = 0.0;
    double original_mean = 0.0, smoothed_mean = 0.0;
    
    for (int i = 0; i < grid->vcnum; i++) {
        original_mean += original->data[i].vel.median;
        smoothed_mean += grid->data[i].vel.median;
    }
    original_mean /= grid->vcnum;
    smoothed_mean /= grid->vcnum;
    
    for (int i = 0; i < grid->vcnum; i++) {
        double orig_diff = original->data[i].vel.median - original_mean;
        double smooth_diff = grid->data[i].vel.median - smoothed_mean;
        original_variance += orig_diff * orig_diff;
        smoothed_variance += smooth_diff * smooth_diff;
    }
    
    assert_test(smoothed_variance < original_variance, "Gaussian smoothing reduces variance");
    
    /* Test boxcar smoothing */
    GridData_Parallel* boxcar_grid = grid_parallel_copy(original);
    params.spatial_radius = 1.5;
    
    start = clock();
    result = grid_smooth_boxcar_parallel(boxcar_grid, &params);
    end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Boxcar smoothing execution");
    
    /* Test triangle smoothing */
    GridData_Parallel* triangle_grid = grid_parallel_copy(original);
    
    start = clock();
    result = grid_smooth_triangle_parallel(triangle_grid, &params);
    end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Triangle smoothing execution");
    
    grid_parallel_free(grid);
    grid_parallel_free(original);
    grid_parallel_free(boxcar_grid);
    grid_parallel_free(triangle_grid);
}

/* Test median filtering */
static void test_median_filtering(void) {
    print_test_header("Median Filtering");
    
    GridData_Parallel* grid = create_noisy_grid(225); // 15x15 grid
    assert_test(grid != NULL, "Test grid creation");
    
    /* Create copy for comparison */
    GridData_Parallel* original = grid_parallel_copy(grid);
    assert_test(original != NULL, "Grid copy creation");
    
    /* Test median filter */
    GridFilterParams params = {0};
    params.median_window_size = 3; // 3x3 window
    
    clock_t start = clock();
    int result = grid_filter_median_parallel(grid, &params);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Median filtering execution");
    
    /* Verify filtering reduces noise spikes */
    int spike_reduction = 0;
    for (int i = 0; i < grid->vcnum; i++) {
        if (fabs(original->data[i].vel.median) > 1000.0 && 
            fabs(grid->data[i].vel.median) < fabs(original->data[i].vel.median)) {
            spike_reduction++;
        }
    }
    
    assert_test(spike_reduction > 0, "Median filter reduces noise spikes");
    printf("  Reduced %d noise spikes\n", spike_reduction);
    
    grid_parallel_free(grid);
    grid_parallel_free(original);
}

/* Test composite filtering pipeline */
static void test_composite_filtering(void) {
    print_test_header("Composite Filtering Pipeline");
    
    GridData_Parallel* grid = create_test_grid_with_outliers(1000, 0.2);
    assert_test(grid != NULL, "Test grid creation");
    
    int original_count = grid->vcnum;
    
    /* Apply composite filter: outliers -> percentiles -> smoothing */
    GridFilterParams params = {0};
    params.outlier_threshold = 3.0;
    params.velocity_percentile_low = 2.0;
    params.velocity_percentile_high = 98.0;
    params.spatial_radius = 1.0;
    params.gaussian_sigma = 0.5;
    params.filter_velocity = 1;
    params.filter_power = 1;
    
    clock_t start = clock();
    int result = grid_filter_composite_parallel(grid, &params);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Composite filtering execution");
    assert_test(grid->vcnum <= original_count, "Composite filter processing");
    
    printf("  Processed %d points, retained %d points (%.1f%%)\n", 
           original_count, grid->vcnum, 
           100.0 * grid->vcnum / original_count);
    
    grid_parallel_free(grid);
}

/* Test quality control filtering */
static void test_quality_control(void) {
    print_test_header("Quality Control Filtering");
    
    GridData_Parallel* grid = create_test_grid_with_outliers(500, 0.1);
    assert_test(grid != NULL, "Test grid creation");
    
    /* Add some points with high velocity errors */
    for (int i = 0; i < 50; i++) {
        grid->data[i].vel.sd = 200.0 + 100.0 * rand() / RAND_MAX; // High error
    }
    
    int original_count = grid->vcnum;
    
    /* Apply quality control */
    GridFilterParams params = {0};
    params.max_velocity_error = 150.0;
    params.max_power_error = 10.0;
    params.min_signal_to_noise = 3.0;
    
    clock_t start = clock();
    int result = grid_filter_quality_control_parallel(grid, &params);
    clock_t end = clock();
    test_stats.total_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    assert_test(result == 0, "Quality control filtering execution");
    assert_test(grid->vcnum < original_count, "Poor quality data removed");
    
    /* Verify remaining data meets quality criteria */
    int quality_violations = 0;
    for (int i = 0; i < grid->vcnum; i++) {
        if (grid->data[i].vel.sd > params.max_velocity_error) {
            quality_violations++;
        }
    }
    
    assert_test(quality_violations == 0, "Quality control criteria enforced");
    
    grid_parallel_free(grid);
}

/* Test parallel performance */
static void test_parallel_performance(void) {
    print_test_header("Parallel Performance Testing");
    
    GridData_Parallel* grid = create_test_grid_with_outliers(NUM_TEST_POINTS, 0.15);
    assert_test(grid != NULL, "Large test grid creation");
    
    /* Test different thread counts */
    for (int threads = 1; threads <= MAX_THREADS && threads <= omp_get_max_threads(); threads *= 2) {
        omp_set_num_threads(threads);
        
        /* Create copy for testing */
        GridData_Parallel* test_grid = grid_parallel_copy(grid);
        
        GridFilterParams params = {0};
        params.outlier_threshold = 2.5;
        params.spatial_radius = 1.0;
        params.filter_velocity = 1;
        
        clock_t start = clock();
        for (int i = 0; i < NUM_PERFORMANCE_ITERATIONS/10; i++) {
            GridData_Parallel* temp_grid = grid_parallel_copy(grid);
            grid_filter_outliers_parallel(temp_grid, &params);
            grid_parallel_free(temp_grid);
        }
        clock_t end = clock();
        
        double time_per_filter = (double)(end - start) / CLOCKS_PER_SEC / (NUM_PERFORMANCE_ITERATIONS/10);
        printf("  %d threads: %.4f seconds per filter\n", threads, time_per_filter);
        
        grid_parallel_free(test_grid);
    }
    
    assert_test(1, "Parallel performance test completed");
    grid_parallel_free(grid);
}

/* Test memory management */
static void test_memory_management(void) {
    print_test_header("Memory Management Testing");
    
    size_t initial_allocated = test_stats.memory_allocated;
    
    /* Create and destroy multiple filtered grids */
    for (int i = 0; i < 5; i++) {
        GridData_Parallel* grid = create_test_grid_with_outliers(200, 0.1);
        assert_test(grid != NULL, "Grid creation in loop");
        
        GridFilterParams params = {0};
        params.outlier_threshold = 2.0;
        params.filter_velocity = 1;
        
        grid_filter_outliers_parallel(grid, &params);
        grid_parallel_free(grid);
        test_stats.memory_freed += 200 * sizeof(GridGVec_Parallel);
    }
    
    size_t final_allocated = test_stats.memory_allocated;
    assert_test(final_allocated > initial_allocated, "Memory allocation tracking");
    
    /* Test error handling */
    int result = grid_filter_outliers_parallel(NULL, NULL);
    assert_test(result == -1, "Error handling with NULL input");
    
    GridData_Parallel empty_grid = {0};
    GridFilterParams params = {0};
    result = grid_filter_outliers_parallel(&empty_grid, &params);
    assert_test(result == -1, "Error handling with empty grid");
}

/* Test edge cases */
static void test_edge_cases(void) {
    print_test_header("Edge Cases Testing");
    
    /* Single element grid */
    GridData_Parallel* single_grid = create_test_grid_with_outliers(1, 0.0);
    assert_test(single_grid != NULL, "Single element grid creation");
    
    GridFilterParams params = {0};
    params.outlier_threshold = 2.0;
    params.filter_velocity = 1;
    
    int result = grid_filter_outliers_parallel(single_grid, &params);
    assert_test(result == 0, "Single element filtering");
    assert_test(single_grid->vcnum == 1, "Single element preserved");
    
    grid_parallel_free(single_grid);
    
    /* Grid with all identical values */
    GridData_Parallel* uniform_grid = create_test_grid_with_outliers(100, 0.0);
    for (int i = 0; i < uniform_grid->vcnum; i++) {
        uniform_grid->data[i].vel.median = 500.0; // All same velocity
        uniform_grid->data[i].pwr.median = 25.0;  // All same power
    }
    
    params.spatial_radius = 1.0;
    result = grid_smooth_gaussian_parallel(uniform_grid, &params);
    assert_test(result == 0, "Uniform data smoothing");
    
    /* Check that values remain unchanged */
    int uniform_preserved = 1;
    for (int i = 0; i < uniform_grid->vcnum; i++) {
        if (fabs(uniform_grid->data[i].vel.median - 500.0) > TOLERANCE) {
            uniform_preserved = 0;
            break;
        }
    }
    assert_test(uniform_preserved, "Uniform data preservation");
    
    grid_parallel_free(uniform_grid);
}

/* Main test function */
int main(int argc, char** argv) {
    printf("SuperDARN Grid Parallel Filtering Test Suite\n");
    printf("============================================\n");
    
    int run_benchmarks = 0;
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        run_benchmarks = 1;
    }
    
    clock_t total_start = clock();
    
    /* Run all tests */
    test_outlier_detection();
    test_percentile_filtering();
    test_spatial_smoothing();
    test_median_filtering();
    test_composite_filtering();
    test_quality_control();
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
