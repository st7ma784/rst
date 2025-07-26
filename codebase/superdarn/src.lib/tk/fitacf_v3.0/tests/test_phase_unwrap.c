#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "cuda_phase_unwrap.h"

#define NUM_RANGES 1000
#define NUM_LAGS 256
#define NUM_ITERATIONS 100
#define PHASE_THRESHOLD (M_PI * 0.9f) // 90% of pi

// Simple random number generator
float rand_float(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

// Generate test data with wrapped phases
void generate_test_data(float* data, int num_ranges, int num_lags) {
    for (int r = 0; r < num_ranges; r++) {
        float phase = 0.0f;
        for (int l = 0; l < num_lags; l++) {
            // Add random phase step
            phase += rand_float(-PHASE_THRESHOLD, PHASE_THRESHOLD);
            
            // Wrap to [-π, π]
            while (phase > M_PI) phase -= 2 * M_PI;
            while (phase < -M_PI) phase += 2 * M_PI;
            
            data[r * num_lags + l] = phase;
        }
    }
}

// Verify unwrapped phases
int verify_results(const float* wrapped, const float* unwrapped, int num_ranges, int num_lags) {
    int errors = 0;
    
    for (int r = 0; r < num_ranges; r++) {
        float expected = 0.0f;
        
        for (int l = 0; l < num_lags; l++) {
            int idx = r * num_lags + l;
            float delta = wrapped[idx] - (l > 0 ? wrapped[idx - 1] : 0.0f);
            
            // Handle wrapping in the reference implementation
            if (l > 0) {
                if (delta > PHASE_THRESHOLD) delta -= 2 * M_PI;
                else if (delta < -PHASE_THRESHOLD) delta += 2 * M_PI;
                expected += delta;
            } else {
                expected = wrapped[idx];
            }
            
            // Compare with tolerance
            if (fabsf(unwrapped[idx] - expected) > 1e-5) {
                if (errors < 10) { // Only print first few errors
                    printf("Mismatch at range %d, lag %d: got %f, expected %f\n", 
                           r, l, unwrapped[idx], expected);
                } else if (errors == 10) {
                    printf("Too many errors, suppressing further output...\n");
                }
                errors++;
            }
        }
    }
    
    return errors;
}

// Benchmark function
double benchmark_phase_unwrap(
    void (*unwrap_func)(const float*, float*, float*, int, int, float),
    const char* name,
    const float* input,
    float* output,
    float* quality,
    int num_ranges,
    int num_lags,
    int iterations
) {
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        unwrap_func(input, output, quality, num_ranges, num_lags, PHASE_THRESHOLD);
    }
    
    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    double time_per_iteration = total_time / iterations;
    
    printf("%s: %d iterations, %.3f ms/iter, %.1f ranges/ms\n",
           name, iterations, time_per_iteration * 1000,
           num_ranges / (time_per_iteration * 1000));
    
    return time_per_iteration;
}

int main() {
    printf("=== Phase Unwrapping Benchmark ===\n");
    printf("Ranges: %d, Lags: %d, Iterations: %d\n\n", 
           NUM_RANGES, NUM_LAGS, NUM_ITERATIONS);
    
    // Allocate memory
    size_t data_size = NUM_RANGES * NUM_LAGS * sizeof(float);
    float* input = (float*)malloc(data_size);
    float* output_cpu = (float*)malloc(data_size);
    float* output_gpu = (float*)malloc(data_size);
    float* quality_cpu = (float*)calloc(NUM_RANGES, sizeof(float));
    float* quality_gpu = (float*)calloc(NUM_RANGES, sizeof(float));
    
    if (!input || !output_cpu || !output_gpu || !quality_cpu || !quality_gpu) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Generate test data
    srand(42); // Fixed seed for reproducibility
    generate_test_data(input, NUM_RANGES, NUM_LAGS);
    
    // Run benchmarks
    printf("\nBenchmarking...\n");
    double cpu_time = benchmark_phase_unwrap(
        omp_phase_unwrap, "CPU (OpenMP)", 
        input, output_cpu, quality_cpu, 
        NUM_RANGES, NUM_LAGS, NUM_ITERATIONS
    );
    
    double gpu_time = benchmark_phase_unwrap(
        cuda_phase_unwrap, "GPU (CUDA)", 
        input, output_gpu, quality_gpu, 
        NUM_RANGES, NUM_LAGS, NUM_ITERATIONS
    );
    
    printf("\nSpeedup: %.2fx\n\n", cpu_time / gpu_time);
    
    // Verify results
    printf("Verifying results...\n");
    int errors = verify_results(input, output_gpu, NUM_RANGES, NUM_LAGS);
    
    if (errors == 0) {
        printf("Verification PASSED - All phase values unwrapped correctly\n");
    } else {
        printf("Verification FAILED - %d phase errors detected\n", errors);
    }
    
    // Clean up
    free(input);
    free(output_cpu);
    free(output_gpu);
    free(quality_cpu);
    free(quality_gpu);
    
    return errors > 0 ? 1 : 0;
}
