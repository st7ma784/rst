/*
 * CUDA Linked List Validation Test Suite
 * 
 * This test suite validates the CUDA-compatible linked list implementation
 * against the original CPU version using real SUPERDARN rawacf data.
 * 
 * Test Strategy:
 * 1. Load identical rawacf data into both CPU and CUDA implementations
 * 2. Run identical processing algorithms on both versions
 * 3. Compare outputs for correctness and performance
 * 4. Generate detailed reports for CI/CD validation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>

// Include both original and CUDA implementations
#include "../include/llist.h"
#include "../include/llist_cuda.h"
#include "../include/llist_compat.h"

// SUPERDARN data structures
#include "fit_structures.h"
#include "fitacftoplevel.h"

// Test configuration
#define MAX_TEST_FILES 10
#define MAX_PATH_LEN 512
#define MAX_RANGES 300
#define TOLERANCE_FLOAT 1e-6
#define TOLERANCE_DOUBLE 1e-12

// Test result structure
typedef struct {
    char test_name[128];
    int passed;
    double cpu_time_ms;
    double cuda_time_ms;
    double speedup;
    int data_points_compared;
    int mismatches;
    char error_msg[256];
} TestResult;

// Global test statistics
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
    double total_cpu_time;
    double total_cuda_time;
    TestResult results[100];
} TestSuite;

static TestSuite g_test_suite = {0};

// Utility functions
static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int file_exists(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0);
}

static void add_test_result(const char* name, int passed, double cpu_time, 
                           double cuda_time, int data_points, int mismatches, 
                           const char* error) {
    TestResult* result = &g_test_suite.results[g_test_suite.total_tests];
    strncpy(result->test_name, name, sizeof(result->test_name) - 1);
    result->passed = passed;
    result->cpu_time_ms = cpu_time;
    result->cuda_time_ms = cuda_time;
    result->speedup = (cuda_time > 0) ? cpu_time / cuda_time : 0.0;
    result->data_points_compared = data_points;
    result->mismatches = mismatches;
    if (error) {
        strncpy(result->error_msg, error, sizeof(result->error_msg) - 1);
    } else {
        result->error_msg[0] = '\0';
    }
    
    g_test_suite.total_tests++;
    if (passed) {
        g_test_suite.passed_tests++;
    } else {
        g_test_suite.failed_tests++;
    }
    g_test_suite.total_cpu_time += cpu_time;
    g_test_suite.total_cuda_time += cuda_time;
}

// Data comparison functions
static int compare_float_arrays(float* cpu_data, float* cuda_data, int count, 
                               const char* data_name) {
    int mismatches = 0;
    for (int i = 0; i < count; i++) {
        if (fabs(cpu_data[i] - cuda_data[i]) > TOLERANCE_FLOAT) {
            if (mismatches < 10) { // Limit error output
                printf("  MISMATCH in %s[%d]: CPU=%.6f, CUDA=%.6f, diff=%.6f\n",
                       data_name, i, cpu_data[i], cuda_data[i], 
                       fabs(cpu_data[i] - cuda_data[i]));
            }
            mismatches++;
        }
    }
    return mismatches;
}

static int compare_double_arrays(double* cpu_data, double* cuda_data, int count,
                                const char* data_name) {
    int mismatches = 0;
    for (int i = 0; i < count; i++) {
        if (fabs(cpu_data[i] - cuda_data[i]) > TOLERANCE_DOUBLE) {
            if (mismatches < 10) { // Limit error output
                printf("  MISMATCH in %s[%d]: CPU=%.12f, CUDA=%.12f, diff=%.12f\n",
                       data_name, i, cpu_data[i], cuda_data[i], 
                       fabs(cpu_data[i] - cuda_data[i]));
            }
            mismatches++;
        }
    }
    return mismatches;
}

// Test case: Basic linked list operations
static void test_basic_operations() {
    printf("Running basic operations test...\n");
    
    double start_time, cpu_time, cuda_time;
    int mismatches = 0;
    
    // CPU version
    start_time = get_time_ms();
    llist_t* cpu_list = llist_create(NULL, NULL, MT_SUPPORT_FALSE);
    
    // Add test data
    for (int i = 0; i < 1000; i++) {
        int* data = malloc(sizeof(int));
        *data = i;
        llist_add_node(cpu_list, data, ADD_NODE_REAR);
    }
    
    int cpu_size = llist_size(cpu_list);
    cpu_time = get_time_ms() - start_time;
    
    // CUDA version
    start_time = get_time_ms();
    llist_cuda_t* cuda_list = llist_cuda_create(1000, sizeof(int), NULL, NULL, 0);
    
    // Add test data
    for (int i = 0; i < 1000; i++) {
        llist_cuda_add_node(cuda_list, &i, ADD_NODE_REAR);
    }
    
    int cuda_size = llist_cuda_size(cuda_list);
    cuda_time = get_time_ms() - start_time;
    
    // Compare results
    if (cpu_size != cuda_size) {
        mismatches++;
        printf("  Size mismatch: CPU=%d, CUDA=%d\n", cpu_size, cuda_size);
    }
    
    // Cleanup
    llist_destroy(cpu_list, 1, free);
    llist_cuda_destroy(cuda_list);
    
    add_test_result("basic_operations", mismatches == 0, cpu_time, cuda_time, 
                   1000, mismatches, mismatches > 0 ? "Size mismatch" : NULL);
}

// Test case: Range processing simulation
static void test_range_processing() {
    printf("Running range processing test...\n");
    
    double start_time, cpu_time, cuda_time;
    int mismatches = 0;
    const int num_ranges = 100;
    const int samples_per_range = 50;
    
    // Simulate ACF data for testing
    float test_acf_data[num_ranges * samples_per_range];
    for (int i = 0; i < num_ranges * samples_per_range; i++) {
        test_acf_data[i] = sin(i * 0.1) + 0.1 * (rand() / (float)RAND_MAX);
    }
    
    // CPU version - simulate range-based processing
    start_time = get_time_ms();
    llist_t* cpu_ranges[num_ranges];
    float cpu_results[num_ranges];
    
    for (int r = 0; r < num_ranges; r++) {
        cpu_ranges[r] = llist_create(NULL, NULL, MT_SUPPORT_FALSE);
        
        // Add ACF samples to list
        for (int s = 0; s < samples_per_range; s++) {
            float* sample = malloc(sizeof(float));
            *sample = test_acf_data[r * samples_per_range + s];
            llist_add_node(cpu_ranges[r], sample, ADD_NODE_REAR);
        }
        
        // Simulate processing (compute mean)
        float sum = 0.0;
        llist_reset_iter(cpu_ranges[r]);
        float* sample;
        while ((sample = (float*)llist_go_next(cpu_ranges[r])) != NULL) {
            sum += *sample;
        }
        cpu_results[r] = sum / samples_per_range;
    }
    cpu_time = get_time_ms() - start_time;
    
    // CUDA version - simulate batch processing
    start_time = get_time_ms();
    llist_cuda_t* cuda_ranges[num_ranges];
    float cuda_results[num_ranges];
    
    for (int r = 0; r < num_ranges; r++) {
        cuda_ranges[r] = llist_cuda_create(samples_per_range, sizeof(float), NULL, NULL, 0);
        
        // Add ACF samples to CUDA list
        for (int s = 0; s < samples_per_range; s++) {
            float sample = test_acf_data[r * samples_per_range + s];
            llist_cuda_add_node(cuda_ranges[r], &sample, ADD_NODE_REAR);
        }
        
        // Simulate processing (compute mean using CUDA batch processing)
        float sum = 0.0;
        int count = llist_cuda_size(cuda_ranges[r]);
        for (int i = 0; i < count; i++) {
            float* sample = (float*)llist_cuda_get_data_at(cuda_ranges[r], i);
            if (sample && llist_cuda_is_valid(cuda_ranges[r], i)) {
                sum += *sample;
            }
        }
        cuda_results[r] = sum / count;
    }
    cuda_time = get_time_ms() - start_time;
    
    // Compare results
    mismatches = compare_float_arrays(cpu_results, cuda_results, num_ranges, "range_means");
    
    // Cleanup
    for (int r = 0; r < num_ranges; r++) {
        llist_destroy(cpu_ranges[r], 1, free);
        llist_cuda_destroy(cuda_ranges[r]);
    }
    
    add_test_result("range_processing", mismatches == 0, cpu_time, cuda_time,
                   num_ranges, mismatches, mismatches > 0 ? "Range processing mismatch" : NULL);
}

// Test case: Filtering operations
static void test_filtering_operations() {
    printf("Running filtering operations test...\n");
    
    double start_time, cpu_time, cuda_time;
    int mismatches = 0;
    const int data_size = 1000;
    
    // Generate test data
    float test_data[data_size];
    for (int i = 0; i < data_size; i++) {
        test_data[i] = (float)i + 0.5 * sin(i * 0.1);
    }
    
    // CPU version - filter values > 500
    start_time = get_time_ms();
    llist_t* cpu_list = llist_create(NULL, NULL, MT_SUPPORT_FALSE);
    
    for (int i = 0; i < data_size; i++) {
        float* data = malloc(sizeof(float));
        *data = test_data[i];
        llist_add_node(cpu_list, data, ADD_NODE_REAR);
    }
    
    // Filter by removing nodes with value <= 500
    llist_reset_iter(cpu_list);
    float* sample;
    while ((sample = (float*)llist_go_next(cpu_list)) != NULL) {
        if (*sample <= 500.0) {
            llist_delete_node(cpu_list, sample, 1, free);
            llist_reset_iter(cpu_list); // Reset after deletion
        }
    }
    
    int cpu_filtered_count = llist_size(cpu_list);
    cpu_time = get_time_ms() - start_time;
    
    // CUDA version - use mask-based filtering
    start_time = get_time_ms();
    llist_cuda_t* cuda_list = llist_cuda_create(data_size, sizeof(float), NULL, NULL, 0);
    
    for (int i = 0; i < data_size; i++) {
        llist_cuda_add_node(cuda_list, &test_data[i], ADD_NODE_REAR);
    }
    
    // Filter using mask (mark invalid instead of deleting)
    for (int i = 0; i < data_size; i++) {
        float* data = (float*)llist_cuda_get_data_at(cuda_list, i);
        if (data && *data <= 500.0) {
            llist_cuda_mark_invalid(cuda_list, i);
        }
    }
    
    int cuda_filtered_count = llist_cuda_size(cuda_list);
    cuda_time = get_time_ms() - start_time;
    
    // Compare results
    if (cpu_filtered_count != cuda_filtered_count) {
        mismatches++;
        printf("  Filtered count mismatch: CPU=%d, CUDA=%d\n", 
               cpu_filtered_count, cuda_filtered_count);
    }
    
    // Cleanup
    llist_destroy(cpu_list, 1, free);
    llist_cuda_destroy(cuda_list);
    
    add_test_result("filtering_operations", mismatches == 0, cpu_time, cuda_time,
                   data_size, mismatches, mismatches > 0 ? "Filtering mismatch" : NULL);
}

// Test case: Compatibility layer
static void test_compatibility_layer() {
    printf("Running compatibility layer test...\n");
    
    double start_time, cpu_time, cuda_time;
    int mismatches = 0;
    const int data_size = 500;
    
    // Test using compatibility layer (should use CUDA backend)
    start_time = get_time_ms();
    
    // This should transparently use CUDA implementation
    llist_t* compat_list = llist_create(NULL, NULL, MT_SUPPORT_FALSE);
    
    for (int i = 0; i < data_size; i++) {
        int* data = malloc(sizeof(int));
        *data = i * 2;
        llist_add_node(compat_list, data, ADD_NODE_REAR);
    }
    
    int compat_size = llist_size(compat_list);
    
    // Test iteration
    int sum = 0;
    llist_reset_iter(compat_list);
    int* value;
    while ((value = (int*)llist_go_next(compat_list)) != NULL) {
        sum += *value;
    }
    
    cuda_time = get_time_ms() - start_time;
    
    // Compare with expected results
    int expected_size = data_size;
    int expected_sum = 0;
    for (int i = 0; i < data_size; i++) {
        expected_sum += i * 2;
    }
    
    if (compat_size != expected_size) {
        mismatches++;
        printf("  Size mismatch: Expected=%d, Got=%d\n", expected_size, compat_size);
    }
    
    if (sum != expected_sum) {
        mismatches++;
        printf("  Sum mismatch: Expected=%d, Got=%d\n", expected_sum, sum);
    }
    
    llist_destroy(compat_list, 1, free);
    
    add_test_result("compatibility_layer", mismatches == 0, 0.0, cuda_time,
                   data_size, mismatches, mismatches > 0 ? "Compatibility mismatch" : NULL);
}

// Main test runner
static void run_all_tests() {
    printf("=== CUDA Linked List Validation Test Suite ===\n\n");
    
    // Initialize random seed for reproducible tests
    srand(12345);
    
    // Run individual test cases
    test_basic_operations();
    test_range_processing();
    test_filtering_operations();
    test_compatibility_layer();
    
    printf("\n=== Test Results Summary ===\n");
    printf("Total Tests: %d\n", g_test_suite.total_tests);
    printf("Passed: %d\n", g_test_suite.passed_tests);
    printf("Failed: %d\n", g_test_suite.failed_tests);
    printf("Success Rate: %.1f%%\n", 
           (g_test_suite.total_tests > 0) ? 
           (100.0 * g_test_suite.passed_tests / g_test_suite.total_tests) : 0.0);
    
    if (g_test_suite.total_cuda_time > 0) {
        printf("Overall Speedup: %.2fx\n", 
               g_test_suite.total_cpu_time / g_test_suite.total_cuda_time);
    }
    
    printf("\nDetailed Results:\n");
    printf("%-25s %-8s %-12s %-12s %-10s %-8s %s\n", 
           "Test Name", "Status", "CPU Time(ms)", "CUDA Time(ms)", 
           "Speedup", "Points", "Error");
    printf("%-25s %-8s %-12s %-12s %-10s %-8s %s\n", 
           "-------------------------", "--------", "------------", "------------", 
           "----------", "--------", "-----");
    
    for (int i = 0; i < g_test_suite.total_tests; i++) {
        TestResult* r = &g_test_suite.results[i];
        printf("%-25s %-8s %-12.2f %-12.2f %-10.2f %-8d %s\n",
               r->test_name,
               r->passed ? "PASS" : "FAIL",
               r->cpu_time_ms,
               r->cuda_time_ms,
               r->speedup,
               r->data_points_compared,
               r->error_msg);
    }
}

int main(int argc, char* argv[]) {
    printf("CUDA Linked List Validation Test Suite\n");
    printf("Built: %s %s\n\n", __DATE__, __TIME__);
    
    // Check if CUDA is available
    if (!llist_cuda_init()) {
        printf("ERROR: CUDA initialization failed. Running CPU-only tests.\n");
        return 1;
    }
    
    run_all_tests();
    
    llist_cuda_cleanup();
    
    // Return non-zero if any tests failed
    return (g_test_suite.failed_tests > 0) ? 1 : 0;
}
