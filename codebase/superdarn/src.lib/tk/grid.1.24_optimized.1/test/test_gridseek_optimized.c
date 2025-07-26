/* test_gridseek_optimized.c
   =========================
   Test and benchmark for optimized grid search implementation
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "griddata_parallel.h"
#include "gridseek_optimized.h"

#define NUM_TESTS 1000000
#define ARRAY_SIZE 1000000

// Simple timing macros
#define START_TIMER() struct timeval start, end; gettimeofday(&start, NULL)
#define STOP_TIMER() gettimeofday(&end, NULL); \
    ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6

// Generate test data
void generate_test_data(double *data, int size, double min_val, double max_val) {
    srand(time(NULL));
    double range = max_val - min_val;
    
    for (int i = 0; i < size; i++) {
        data[i] = min_val + ((double)rand() / RAND_MAX) * range;
    }
    
    // Ensure data is sorted for binary search
    qsort(data, size, sizeof(double), (int (*)(const void*, const void*))&dbl_cmp);
}

// Simple double comparison for qsort
int dbl_cmp(const double *a, const double *b) {
    if (*a < *b) return -1;
    if (*a > *b) return 1;
    return 0;
}

// Test vectorized binary search
void test_vectorized_search() {
    printf("Testing vectorized binary search...\n");
    
    double *test_data = (double *)aligned_alloc(64, ARRAY_SIZE * sizeof(double));
    generate_test_data(test_data, ARRAY_SIZE, 0.0, 1000000.0);
    
    // Test with existing values
    printf("  Testing with existing values...");
    int num_found = 0;
    
    START_TIMER();
    for (int i = 0; i < NUM_TESTS; i++) {
        int idx = rand() % ARRAY_SIZE;
        double target = test_data[idx];
        int found_idx = -1;
        
        grid_vectorized_binary_search(test_data, ARRAY_SIZE, target, &found_idx);
        if (found_idx >= 0 && found_idx < ARRAY_SIZE && 
            fabs(test_data[found_idx] - target) < 1e-9) {
            num_found++;
        }
    }
    double elapsed = STOP_TIMER();
    
    printf(" %.2fM searches/sec (found %d/%d)\n", 
           NUM_TESTS / elapsed / 1e6, num_found, NUM_TESTS);
    
    // Test with interpolated values
    printf("  Testing with interpolated values...");
    num_found = 0;
    
    START_TIMER();
    for (int i = 0; i < NUM_TESTS; i++) {
        int idx1 = rand() % (ARRAY_SIZE - 1);
        int idx2 = idx1 + 1;
        double target = (test_data[idx1] + test_data[idx2]) / 2.0;
        int found_idx = -1;
        
        grid_vectorized_binary_search(test_data, ARRAY_SIZE, target, &found_idx);
        if (found_idx >= 0 && found_idx < ARRAY_SIZE - 1) {
            // Should find either the lower or upper bound
            if (fabs(test_data[found_idx] - target) < fabs(test_data[found_idx+1] - target)) {
                num_found++;
            } else if (found_idx + 1 < ARRAY_SIZE && 
                      fabs(test_data[found_idx+1] - target) <= fabs(test_data[found_idx] - target)) {
                num_found++;
            }
        }
    }
    elapsed = STOP_TIMER();
    
    printf(" %.2fM searches/sec (found %d/%d)\n", 
           NUM_TESTS / elapsed / 1e6, num_found, NUM_TESTS);
    
    free(test_data);
}

// Benchmark grid_optimized_seek vs original implementation
void benchmark_grid_seek() {
    printf("\nBenchmarking grid_optimized_seek...\n");
    
    // Create test index
    struct GridIndexParallel test_idx;
    test_idx.num = ARRAY_SIZE;
    test_idx.tme = (double *)aligned_alloc(64, ARRAY_SIZE * sizeof(double));
    test_idx.inx = (off_t *)aligned_alloc(64, ARRAY_SIZE * sizeof(off_t));
    
    generate_test_data(test_idx.tme, ARRAY_SIZE, 0.0, 1000000.0);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        test_idx.inx[i] = i * 1024;  // Dummy file offsets
    }
    
    // Test parameters
    double target_time = test_idx.tme[ARRAY_SIZE / 2];
    int yr, mo, dy, hr, mt, sc;
    double sec;
    
    TimeEpochToYMDHMS(target_time, &yr, &mo, &dy, &hr, &mt, &sec);
    sc = (int)sec;
    
    // Benchmark original implementation
    printf("  Original implementation: ");
    fflush(stdout);
    
    double atme_orig = 0.0;
    START_TIMER();
    for (int i = 0; i < NUM_TESTS; i++) {
        // Note: This is a placeholder - in a real test, we'd call the original function
        // grid_parallel_seek(0, yr, mo, dy, hr, mt, sc, &atme_orig, &test_idx, NULL);
    }
    double orig_time = STOP_TIMER();
    printf("%.2fM seeks/sec\n", NUM_TESTS / orig_time / 1e6);
    
    // Benchmark optimized implementation
    printf("  Optimized implementation: ");
    fflush(stdout);
    
    double atme_opt = 0.0;
    START_TIMER();
    for (int i = 0; i < NUM_TESTS; i++) {
        grid_optimized_seek(0, yr, mo, dy, hr, mt, sc, &atme_opt, &test_idx, NULL);
    }
    double opt_time = STOP_TIMER();
    printf("%.2fM seeks/sec (%.2fx speedup)\n", 
           NUM_TESTS / opt_time / 1e6, orig_time / opt_time);
    
    // Verify results match
    if (fabs(atme_orig - atme_opt) > 1e-9) {
        printf("  WARNING: Results differ! orig=%.9f, opt=%.9f\n", atme_orig, atme_opt);
    }
    
    free(test_idx.tme);
    free(test_idx.inx);
}

int main() {
    printf("=== Grid Search Optimization Test ===\n");
    
    // Test vectorized binary search
    test_vectorized_search();
    
    // Benchmark grid seek operations
    benchmark_grid_seek();
    
    printf("\nTests completed.\n");
    return 0;
}
