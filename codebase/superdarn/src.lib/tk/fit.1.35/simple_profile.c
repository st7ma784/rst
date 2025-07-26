#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// Simple structure for profiling
struct TimingResult {
    const char *name;
    double total_time;    // in microseconds
    double avg_time;      // in microseconds
    double operations_per_sec;
};

// Function to get current time in microseconds
long long current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000000LL + te.tv_usec;
}

// Function to run a benchmark and record timing
void run_benchmark(const char *name, void (*benchmark_func)(int, void*), 
                  int num_iterations, void *user_data) {
    printf("Running benchmark: %s (%d iterations)\n", name, num_iterations);
    
    // Warm-up
    benchmark_func(10, user_data);
    
    // Run benchmark
    long long start = current_timestamp();
    benchmark_func(num_iterations, user_data);
    long long end = current_timestamp();
    
    // Calculate results
    double total_time = (end - start) / 1000.0;  // Convert to milliseconds
    double avg_time = total_time / num_iterations;
    double operations_per_sec = (num_iterations * 1000.0) / total_time;
    
    printf("  Total time: %.2f ms\n", total_time);
    printf("  Avg time: %.6f ms\n", avg_time);
    printf("  Operations/sec: %.2f\n\n", operations_per_sec);
}

// Benchmark function for matrix operations
void benchmark_matrix_ops(int iterations, void *user_data) {
    int size = 100;  // Matrix size (N x N)
    double *A = (double *)malloc(size * size * sizeof(double));
    double *B = (double *)malloc(size * size * sizeof(double));
    double *C = (double *)malloc(size * size * sizeof(double));
    
    // Initialize matrices with random values
    for (int i = 0; i < size * size; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
    
    // Perform matrix multiplication
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double sum = 0.0;
                for (int k = 0; k < size; k++) {
                    sum += A[i * size + k] * B[k * size + j];
                }
                C[i * size + j] = sum;
            }
        }
    }
    
    free(A);
    free(B);
    free(C);
}

// Benchmark function for FFT operations
void benchmark_fft(int iterations, void *user_data) {
    int n = 1024;  // FFT size
    double *real = (double *)malloc(n * sizeof(double));
    double *imag = (double *)malloc(n * sizeof(double));
    
    // Initialize with random data
    for (int i = 0; i < n; i++) {
        real[i] = (double)rand() / RAND_MAX;
        imag[i] = 0.0;
    }
    
    // Simple FFT implementation (Cooley-Tukey algorithm)
    for (int iter = 0; iter < iterations; iter++) {
        int m, i, j, k;
        double c, s, t1, t2;
        
        // Bit-reversal permutation
        j = 0;
        for (i = 0; i < n - 1; i++) {
            if (i < j) {
                double temp = real[i];
                real[i] = real[j];
                real[j] = temp;
                
                temp = imag[i];
                imag[i] = imag[j];
                imag[j] = temp;
            }
            
            k = n / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }
        
        // Danielson-Lanczos section
        int mmax = 1;
        while (n > mmax) {
            int istep = 2 * mmax;
            double theta = M_PI / mmax;
            
            for (m = 0; m < mmax; m++) {
                double w = -m * theta;
                c = cos(w);
                s = sin(w);
                
                for (i = m; i < n; i += istep) {
                    j = i + mmax;
                    t1 = c * real[j] - s * imag[j];
                    t2 = s * real[j] + c * imag[j];
                    real[j] = real[i] - t1;
                    imag[j] = imag[i] - t2;
                    real[i] += t1;
                    imag[i] += t2;
                }
            }
            mmax = istep;
        }
    }
    
    free(real);
    free(imag);
}

int main() {
    printf("=== Fit.1.35 Performance Profiler ===\n\n");
    
    // Seed the random number generator
    srand(time(NULL));
    
    // Run benchmarks
    run_benchmark("Matrix Multiplication (100x100)", benchmark_matrix_ops, 1000, NULL);
    run_benchmark("FFT (1024 points)", benchmark_fft, 1000, NULL);
    
    printf("=== Profiling Complete ===\n");
    return 0;
}
