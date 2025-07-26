#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// CUDA function declaration
extern void cuda_phase_unwrap(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
);

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple random number generator
float rand_float(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

// Generate test data with wrapped phases
void generate_test_data(float* data, int num_ranges, int num_lags, float threshold) {
    for (int r = 0; r < num_ranges; r++) {
        float phase = 0.0f;
        for (int l = 0; l < num_lags; l++) {
            // Add random phase step
            phase += rand_float(-threshold, threshold);
            
            // Wrap to [-π, π]
            while (phase > M_PI) phase -= 2 * M_PI;
            while (phase < -M_PI) phase += 2 * M_PI;
            
            data[r * num_lags + l] = phase;
        }
    }
}

// CPU implementation of phase unwrapping (single-threaded)
void cpu_phase_unwrap_st(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
) {
    for (int r = 0; r < num_ranges; r++) {
        int base_idx = r * num_lags;
        
        if (num_lags > 0) {
            phase_out[base_idx] = phase_in[base_idx];
            
            for (int i = 1; i < num_lags; i++) {
                int curr_idx = base_idx + i;
                float delta = phase_in[curr_idx] - phase_in[curr_idx - 1];
                
                // Handle phase wrapping
                if (delta > phase_threshold) {
                    delta -= 2.0f * M_PI;
                } else if (delta < -phase_threshold) {
                    delta += 2.0f * M_PI;
                }
                
                phase_out[curr_idx] = phase_out[curr_idx - 1] + delta;
                
                if (quality_metric) {
                    quality_metric[r] += fabsf(delta);
                }
            }
        }
    }
}

// CPU implementation of phase unwrapping (OpenMP parallel)
void cpu_phase_unwrap_omp(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
) {
    #pragma omp parallel for
    for (int r = 0; r < num_ranges; r++) {
        int base_idx = r * num_lags;
        
        if (num_lags > 0) {
            phase_out[base_idx] = phase_in[base_idx];
            float quality = 0.0f;
            
            for (int i = 1; i < num_lags; i++) {
                int curr_idx = base_idx + i;
                float delta = phase_in[curr_idx] - phase_in[curr_idx - 1];
                
                // Handle phase wrapping
                if (delta > phase_threshold) {
                    delta -= 2.0f * M_PI;
                } else if (delta < -phase_threshold) {
                    delta += 2.0f * M_PI;
                }
                
                phase_out[curr_idx] = phase_out[curr_idx - 1] + delta;
                quality += fabsf(delta);
            }
            
            if (quality_metric) {
                quality_metric[r] = quality;
            }
        }
    }
}

// Verify unwrapped phases
int verify_results(const float* wrapped, const float* unwrapped, 
                  int num_ranges, int num_lags, float threshold) {
    int errors = 0;
    
    for (int r = 0; r < num_ranges; r++) {
        float expected = 0.0f;
        
        for (int l = 0; l < num_lags; l++) {
            int idx = r * num_lags + l;
            float delta = wrapped[idx] - (l > 0 ? wrapped[idx - 1] : 0.0f);
            
            // Handle wrapping in the reference implementation
            if (l > 0) {
                if (delta > threshold) delta -= 2 * M_PI;
                else if (delta < -threshold) delta += 2 * M_PI;
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
    float threshold,
    int iterations,
    int warmup_iterations
) {
    // Warm-up runs
    for (int i = 0; i < warmup_iterations; i++) {
        unwrap_func(input, output, quality, num_ranges, num_lags, threshold);
    }
    
    // Actual benchmark
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        unwrap_func(input, output, quality, num_ranges, num_lags, threshold);
    }
    
    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    double time_per_iteration = total_time / iterations;
    
    printf("%-20s: %4d iterations, %6.3f ms/iter, %8.1f ranges/ms\n",
           name, iterations, time_per_iteration * 1000,
           num_ranges / (time_per_iteration * 1000));
    
    return time_per_iteration;
}

int main(int argc, char** argv) {
    // Test parameters
    int num_ranges = 1000;
    int num_lags = 256;
    int iterations = 100;
    int warmup_iterations = 10;
    
    // Parse command line arguments
    if (argc > 1) num_ranges = atoi(argv[1]);
    if (argc > 2) num_lags = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    if (argc > 4) warmup_iterations = atoi(argv[4]);
    
    const float PHASE_THRESHOLD = (float)(M_PI * 0.9); // 90% of pi
    
    printf("=== Phase Unwrapping Benchmark ===\n");
    printf("Ranges: %d, Lags: %d, Iterations: %d, Warmup: %d\n\n", 
           num_ranges, num_lags, iterations, warmup_iterations);
    
    // Allocate memory
    size_t data_size = num_ranges * num_lags * sizeof(float);
    size_t quality_size = num_ranges * sizeof(float);
    
    float* input = (float*)malloc(data_size);
    float* output_st = (float*)malloc(data_size);
    float* output_omp = (float*)malloc(data_size);
    float* output_cuda = (float*)malloc(data_size);
    
    float* quality_st = (float*)calloc(num_ranges, sizeof(float));
    float* quality_omp = (float*)calloc(num_ranges, sizeof(float));
    float* quality_cuda = (float*)calloc(num_ranges, sizeof(float));
    
    if (!input || !output_st || !output_omp || !output_cuda || 
        !quality_st || !quality_omp || !quality_cuda) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Generate test data
    srand(42); // Fixed seed for reproducibility
    generate_test_data(input, num_ranges, num_lags, PHASE_THRESHOLD);
    
    // Initialize OpenMP
    #ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    #else
    int num_threads = 1;
    #endif
    
    printf("Using %d OpenMP threads\n", num_threads);
    
    // Run benchmarks
    printf("\nBenchmarking implementations...\n");
    printf("%-20s %12s %12s %12s\n", "Implementation", "Time (ms)", "Ranges/ms", "Speedup");
    printf("--------------------------------------------------\n");
    
    // Single-threaded CPU
    double st_time = benchmark_phase_unwrap(
        cpu_phase_unwrap_st, "CPU (Single-thread)", 
        input, output_st, quality_st,
        num_ranges, num_lags, PHASE_THRESHOLD, 
        iterations, warmup_iterations
    );
    
    // OpenMP CPU
    double omp_time = benchmark_phase_unwrap(
        cpu_phase_unwrap_omp, "CPU (OpenMP)", 
        input, output_omp, quality_omp,
        num_ranges, num_lags, PHASE_THRESHOLD, 
        iterations, warmup_iterations
    );
    
    // CUDA GPU
    double cuda_time = 0;
    #ifdef __CUDACC__
    cuda_time = benchmark_phase_unwrap(
        cuda_phase_unwrap, "GPU (CUDA)", 
        input, output_cuda, quality_cuda,
        num_ranges, num_lags, PHASE_THRESHOLD, 
        iterations, warmup_iterations
    );
    
    printf("--------------------------------------------------\n");
    printf("Speedup (CUDA/ST): %.2fx\n", st_time / cuda_time);
    printf("Speedup (CUDA/OMP): %.2fx\n", omp_time / cuda_time);
    printf("Speedup (OMP/ST): %.2fx\n", st_time / omp_time);
    #else
    printf("--------------------------------------------------\n");
    printf("Speedup (OMP/ST): %.2fx\n", st_time / omp_time);
    #endif
    
    // Verify results
    printf("\nVerifying results...\n");
    
    // Verify single-threaded results
    int errors_st = verify_results(input, output_st, num_ranges, num_lags, PHASE_THRESHOLD);
    printf("Single-threaded: %s\n", 
           errors_st == 0 ? "PASSED" : "FAILED");
    
    // Verify OpenMP results
    int errors_omp = 0;
    for (int i = 0; i < num_ranges * num_lags; i++) {
        if (fabsf(output_st[i] - output_omp[i]) > 1e-5) {
            errors_omp++;
            if (errors_omp <= 5) {
                printf("Mismatch at index %d: ST=%.6f, OMP=%.6f\n", 
                       i, output_st[i], output_omp[i]);
            } else if (errors_omp == 6) {
                printf("Additional mismatches not shown...\n");
            }
        }
    }
    printf("OpenMP: %s\n", 
           errors_omp == 0 ? "PASSED" : "FAILED");
    
    #ifdef __CUDACC__
    // Verify CUDA results
    int errors_cuda = 0;
    for (int i = 0; i < num_ranges * num_lags; i++) {
        if (fabsf(output_st[i] - output_cuda[i]) > 1e-5) {
            errors_cuda++;
            if (errors_cuda <= 5) {
                printf("Mismatch at index %d: ST=%.6f, CUDA=%.6f\n", 
                       i, output_st[i], output_cuda[i]);
            } else if (errors_cuda == 6) {
                printf("Additional mismatches not shown...\n");
            }
        }
    }
    printf("CUDA: %s\n", 
           errors_cuda == 0 ? "PASSED" : "FAILED");
    #endif
    
    // Clean up
    free(input);
    free(output_st);
    free(output_omp);
    free(output_cuda);
    free(quality_st);
    free(quality_omp);
    free(quality_cuda);
    
    int total_errors = errors_st + errors_omp;
    #ifdef __CUDACC__
    total_errors += errors_cuda;
    #endif
    
    return total_errors > 0 ? 1 : 0;
}
