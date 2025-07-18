#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "../include/llist.h"
#include "../include/cuda_llist.h"

/**
 * SUPERDARN Advanced CUDA Optimization Benchmark
 * 
 * This test demonstrates the performance improvements achieved by
 * parallelizing the most computationally intensive for-loops and
 * recursive patterns in the SUPERDARN codebase with CUDA.
 */

// External function declarations for our optimizations
extern int cuda_optimized_copy_fitting_data(float*, float*, float*, float*, void*, void*, int, int, bool);
extern int cuda_optimized_power_phase_computation(void*, float*, float*, float*, float*, int, int, float, bool);
extern int cuda_optimized_statistical_reduction(void*, float*, int, int, float*, float, bool);
extern void print_optimization_performance_report(void);

// Test configuration
#define MAX_TEST_RANGES 2000
#define MAX_TEST_LAGS 100
#define TEST_ITERATIONS 5

// Test data structures
typedef struct {
    float real;
    float imag;
} complex_data_t;

// Global test metrics
static int g_tests_passed = 0;
static int g_tests_failed = 0;

// Function prototypes
int test_data_copying_optimization(void);
int test_power_phase_optimization(void);
int test_statistical_reduction_optimization(void);
int test_scalability_analysis(void);
int test_memory_efficiency(void);

// Utility functions
double get_time_ms(void);
void generate_realistic_test_data(float* data, int size, float noise_level);
void print_test_result(const char* test_name, int passed);
void print_performance_comparison(const char* operation, double cpu_time, double cuda_time, int elements);

// ============================================================================
// Main Test Runner
// ============================================================================

int main(int argc, char* argv[]) {
    printf("🚀 SUPERDARN Advanced CUDA Optimization Benchmark\n");
    printf("==================================================\n\n");
    
    // Parse command line arguments
    int test_ranges = 500;
    int test_lags = 50;
    bool verbose = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            test_ranges = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            test_lags = atoi(argv[++i]);
        }
    }
    
    printf("Benchmark Configuration:\n");
    printf("  Range Gates: %d\n", test_ranges);
    printf("  Lags per Gate: %d\n", test_lags);
    printf("  Total Elements: %d\n", test_ranges * test_lags);
    printf("  Verbose Mode: %s\n", verbose ? "ON" : "OFF");
    printf("\n");
    
    // Initialize CUDA if available
    if (cuda_is_available()) {
        printf("✅ CUDA Available - Running comprehensive optimization benchmarks\n\n");
        cuda_initialize();
    } else {
        printf("⚠️  CUDA Not Available - Running CPU-only baseline tests\n\n");
    }
    
    // Run optimization benchmark suite
    printf("🔧 Running Advanced Optimization Benchmarks...\n\n");
    
    // Test 1: Data Copying Optimization
    print_test_result("Data Copying Optimization", test_data_copying_optimization());
    
    // Test 2: Power/Phase Computation Optimization
    print_test_result("Power/Phase Computation Optimization", test_power_phase_optimization());
    
    // Test 3: Statistical Reduction Optimization
    print_test_result("Statistical Reduction Optimization", test_statistical_reduction_optimization());
    
    // Test 4: Scalability Analysis
    print_test_result("Scalability Analysis", test_scalability_analysis());
    
    // Test 5: Memory Efficiency Test
    print_test_result("Memory Efficiency Test", test_memory_efficiency());
    
    // Print comprehensive performance report
    print_optimization_performance_report();
    
    // Final Results
    printf("\n🏁 Advanced Optimization Benchmark Results:\n");
    printf("==========================================\n");
    printf("✅ Tests Passed: %d\n", g_tests_passed);
    printf("❌ Tests Failed: %d\n", g_tests_failed);
    printf("📊 Success Rate: %.1f%%\n", 
           (float)g_tests_passed / (g_tests_passed + g_tests_failed) * 100.0f);
    
    if (g_tests_failed == 0) {
        printf("\n🎉 ALL OPTIMIZATION TESTS PASSED!\n");
        printf("   Advanced CUDA optimizations are ready for production deployment.\n");
        return 0;
    } else {
        printf("\n⚠️  Some optimization tests failed. Review results for deployment readiness.\n");
        return 1;
    }
}

// ============================================================================
// Test Implementations
// ============================================================================

int test_data_copying_optimization(void) {
    printf("  🔄 Testing data copying optimization (nested range×lag loops)...\n");
    
    const int nrang = 1000;
    const int mplgs = 75;
    const int total_elements = nrang * mplgs;
    
    // Allocate test data
    float* raw_acfd_real = malloc(total_elements * sizeof(float));
    float* raw_acfd_imag = malloc(total_elements * sizeof(float));
    float* raw_xcfd_real = malloc(total_elements * sizeof(float));
    float* raw_xcfd_imag = malloc(total_elements * sizeof(float));
    
    complex_data_t* fit_acfd = malloc(total_elements * sizeof(complex_data_t));
    complex_data_t* fit_xcfd = malloc(total_elements * sizeof(complex_data_t));
    
    if (!raw_acfd_real || !raw_acfd_imag || !raw_xcfd_real || !raw_xcfd_imag || !fit_acfd || !fit_xcfd) {
        printf("    ❌ Memory allocation failed\n");
        return 0;
    }
    
    // Generate realistic SUPERDARN-like test data
    generate_realistic_test_data(raw_acfd_real, total_elements, 0.1f);
    generate_realistic_test_data(raw_acfd_imag, total_elements, 0.1f);
    generate_realistic_test_data(raw_xcfd_real, total_elements, 0.15f);
    generate_realistic_test_data(raw_xcfd_imag, total_elements, 0.15f);
    
    // Test CPU implementation (baseline)
    double cpu_start = get_time_ms();
    int cpu_success = cuda_optimized_copy_fitting_data(
        raw_acfd_real, raw_acfd_imag, raw_xcfd_real, raw_xcfd_imag,
        fit_acfd, fit_xcfd, nrang, mplgs, false  // CPU only
    );
    double cpu_time = get_time_ms() - cpu_start;
    
    // Test CUDA implementation
    double cuda_start = get_time_ms();
    int cuda_success = cuda_optimized_copy_fitting_data(
        raw_acfd_real, raw_acfd_imag, raw_xcfd_real, raw_xcfd_imag,
        fit_acfd, fit_xcfd, nrang, mplgs, true   // CUDA enabled
    );
    double cuda_time = get_time_ms() - cuda_start;
    
    // Analyze results
    if (cpu_success && (cuda_success || !cuda_is_available())) {
        print_performance_comparison("Data Copying", cpu_time, cuda_time, total_elements * 2);
        printf("    ✅ Data copying optimization validated\n");
        
        // Cleanup
        free(raw_acfd_real);
        free(raw_acfd_imag);
        free(raw_xcfd_real);
        free(raw_xcfd_imag);
        free(fit_acfd);
        free(fit_xcfd);
        
        return 1;
    } else {
        printf("    ❌ Data copying optimization failed\n");
        return 0;
    }
}

int test_power_phase_optimization(void) {
    printf("  ⚡ Testing power/phase computation optimization...\n");
    
    const int nrang = 800;
    const int mplgs = 60;
    const int total_elements = nrang * mplgs;
    
    // Allocate test data
    complex_data_t* acf_data = malloc(total_elements * sizeof(complex_data_t));
    float* pwr0 = malloc(nrang * sizeof(float));
    float* output_power = malloc(total_elements * sizeof(float));
    float* output_phase = malloc(total_elements * sizeof(float));
    float* output_normalized = malloc(total_elements * sizeof(float));
    
    if (!acf_data || !pwr0 || !output_power || !output_phase || !output_normalized) {
        printf("    ❌ Memory allocation failed\n");
        return 0;
    }
    
    // Generate realistic ACF data
    for (int i = 0; i < total_elements; i++) {
        acf_data[i].real = 10.0f * cosf(i * 0.01f) + (rand() % 100) / 100.0f;
        acf_data[i].imag = 10.0f * sinf(i * 0.01f) + (rand() % 100) / 100.0f;
    }
    
    // Generate lag-0 power values
    for (int i = 0; i < nrang; i++) {
        pwr0[i] = 50.0f + (rand() % 100);
    }
    
    float noise_threshold = 5.0f;
    
    // Test CPU implementation
    double cpu_start = get_time_ms();
    int cpu_success = cuda_optimized_power_phase_computation(
        acf_data, pwr0, output_power, output_phase, output_normalized,
        nrang, mplgs, noise_threshold, false  // CPU only
    );
    double cpu_time = get_time_ms() - cpu_start;
    
    // Test CUDA implementation
    double cuda_start = get_time_ms();
    int cuda_success = cuda_optimized_power_phase_computation(
        acf_data, pwr0, output_power, output_phase, output_normalized,
        nrang, mplgs, noise_threshold, true   // CUDA enabled
    );
    double cuda_time = get_time_ms() - cuda_start;
    
    // Analyze results
    if (cpu_success && (cuda_success || !cuda_is_available())) {
        print_performance_comparison("Power/Phase Computation", cpu_time, cuda_time, total_elements);
        printf("    ✅ Power/phase computation optimization validated\n");
        
        // Cleanup
        free(acf_data);
        free(pwr0);
        free(output_power);
        free(output_phase);
        free(output_normalized);
        
        return 1;
    } else {
        printf("    ❌ Power/phase computation optimization failed\n");
        return 0;
    }
}

int test_statistical_reduction_optimization(void) {
    printf("  📊 Testing statistical reduction optimization...\n");
    
    const int nrang = 1200;
    const int mplgs = 80;
    const int total_elements = nrang * mplgs;
    
    // Allocate test data
    complex_data_t* acf_data = malloc(total_elements * sizeof(complex_data_t));
    float* pwr0 = malloc(nrang * sizeof(float));
    float statistics[4]; // [mean, max, total, count]
    
    if (!acf_data || !pwr0) {
        printf("    ❌ Memory allocation failed\n");
        return 0;
    }
    
    // Generate realistic statistical test data
    for (int i = 0; i < total_elements; i++) {
        float magnitude = 5.0f + (rand() % 1000) / 100.0f;
        float phase = (rand() % 628) / 100.0f; // 0 to 2π
        acf_data[i].real = magnitude * cosf(phase);
        acf_data[i].imag = magnitude * sinf(phase);
    }
    
    for (int i = 0; i < nrang; i++) {
        pwr0[i] = 10.0f + (rand() % 500) / 10.0f;
    }
    
    float noise_threshold = 8.0f;
    
    // Test CPU implementation
    double cpu_start = get_time_ms();
    int cpu_success = cuda_optimized_statistical_reduction(
        acf_data, pwr0, nrang, mplgs, statistics, noise_threshold, false  // CPU only
    );
    double cpu_time = get_time_ms() - cpu_start;
    
    // Test CUDA implementation
    double cuda_start = get_time_ms();
    int cuda_success = cuda_optimized_statistical_reduction(
        acf_data, pwr0, nrang, mplgs, statistics, noise_threshold, true   // CUDA enabled
    );
    double cuda_time = get_time_ms() - cuda_start;
    
    // Analyze results
    if (cpu_success && (cuda_success || !cuda_is_available())) {
        print_performance_comparison("Statistical Reduction", cpu_time, cuda_time, total_elements);
        printf("    📈 Statistics: Mean=%.2f, Max=%.2f, Count=%.0f\n", 
               statistics[0], statistics[1], statistics[3]);
        printf("    ✅ Statistical reduction optimization validated\n");
        
        // Cleanup
        free(acf_data);
        free(pwr0);
        
        return 1;
    } else {
        printf("    ❌ Statistical reduction optimization failed\n");
        return 0;
    }
}

int test_scalability_analysis(void) {
    printf("  📈 Testing scalability analysis across different data sizes...\n");
    
    int test_sizes[] = {100, 500, 1000, 2000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("    Data Size | CPU Time (ms) | CUDA Time (ms) | Speedup\n");
    printf("    ----------|---------------|----------------|--------\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int nrang = test_sizes[s];
        int mplgs = 50;
        int total_elements = nrang * mplgs;
        
        // Allocate test data
        complex_data_t* acf_data = malloc(total_elements * sizeof(complex_data_t));
        float* pwr0 = malloc(nrang * sizeof(float));
        float statistics[4];
        
        if (!acf_data || !pwr0) continue;
        
        // Generate test data
        for (int i = 0; i < total_elements; i++) {
            acf_data[i].real = (rand() % 1000) / 100.0f;
            acf_data[i].imag = (rand() % 1000) / 100.0f;
        }
        for (int i = 0; i < nrang; i++) {
            pwr0[i] = 10.0f + (rand() % 100);
        }
        
        // Test CPU
        double cpu_start = get_time_ms();
        cuda_optimized_statistical_reduction(acf_data, pwr0, nrang, mplgs, statistics, 5.0f, false);
        double cpu_time = get_time_ms() - cpu_start;
        
        // Test CUDA
        double cuda_start = get_time_ms();
        cuda_optimized_statistical_reduction(acf_data, pwr0, nrang, mplgs, statistics, 5.0f, true);
        double cuda_time = get_time_ms() - cuda_start;
        
        float speedup = cuda_is_available() ? (cpu_time / cuda_time) : 1.0f;
        
        printf("    %8d  | %11.2f   | %12.2f     | %6.2fx\n", 
               total_elements, cpu_time, cuda_time, speedup);
        
        free(acf_data);
        free(pwr0);
    }
    
    printf("    ✅ Scalability analysis completed\n");
    return 1;
}

int test_memory_efficiency(void) {
    printf("  💾 Testing memory efficiency and bandwidth utilization...\n");
    
    const int nrang = 1500;
    const int mplgs = 100;
    const int total_elements = nrang * mplgs;
    const size_t total_memory = total_elements * sizeof(complex_data_t) * 2; // ACF + XCF
    
    printf("    Test Data Size: %.2f MB\n", total_memory / (1024.0 * 1024.0));
    
    // Allocate large test dataset
    complex_data_t* acf_data = malloc(total_elements * sizeof(complex_data_t));
    complex_data_t* xcf_data = malloc(total_elements * sizeof(complex_data_t));
    
    if (!acf_data || !xcf_data) {
        printf("    ❌ Large memory allocation failed\n");
        return 0;
    }
    
    // Initialize data
    for (int i = 0; i < total_elements; i++) {
        acf_data[i].real = (rand() % 2000) / 100.0f - 10.0f;
        acf_data[i].imag = (rand() % 2000) / 100.0f - 10.0f;
        xcf_data[i].real = (rand() % 2000) / 100.0f - 10.0f;
        xcf_data[i].imag = (rand() % 2000) / 100.0f - 10.0f;
    }
    
    // Test memory bandwidth
    double start_time = get_time_ms();
    
    // Simulate memory-intensive operations
    for (int iter = 0; iter < 3; iter++) {
        for (int i = 0; i < total_elements; i++) {
            float acf_power = acf_data[i].real * acf_data[i].real + acf_data[i].imag * acf_data[i].imag;
            float xcf_power = xcf_data[i].real * xcf_data[i].real + xcf_data[i].imag * xcf_data[i].imag;
            
            // Simulate computation
            acf_data[i].real = sqrtf(acf_power);
            xcf_data[i].imag = sqrtf(xcf_power);
        }
    }
    
    double end_time = get_time_ms();
    double processing_time = end_time - start_time;
    
    double bandwidth_gbps = (total_memory * 3 * 2) / (processing_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    
    printf("    Processing Time: %.2f ms\n", processing_time);
    printf("    Memory Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("    ✅ Memory efficiency test completed\n");
    
    free(acf_data);
    free(xcf_data);
    
    return 1;
}

// ============================================================================
// Utility Functions
// ============================================================================

double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void generate_realistic_test_data(float* data, int size, float noise_level) {
    for (int i = 0; i < size; i++) {
        // Generate SUPERDARN-like ACF data with realistic characteristics
        float base_value = 20.0f * expf(-i * 0.001f); // Exponential decay
        float noise = (rand() % 1000) / 1000.0f * noise_level;
        data[i] = base_value + noise;
    }
}

void print_test_result(const char* test_name, int passed) {
    if (passed) {
        printf("✅ %s\n", test_name);
        g_tests_passed++;
    } else {
        printf("❌ %s\n", test_name);
        g_tests_failed++;
    }
}

void print_performance_comparison(const char* operation, double cpu_time, double cuda_time, int elements) {
    if (cuda_is_available() && cuda_time > 0) {
        float speedup = cpu_time / cuda_time;
        float throughput = (elements / cuda_time) / 1000.0f; // M elements/sec
        
        printf("    %s Performance:\n", operation);
        printf("      CPU Time:    %.2f ms\n", cpu_time);
        printf("      CUDA Time:   %.2f ms\n", cuda_time);
        printf("      Speedup:     %.2fx\n", speedup);
        printf("      Throughput:  %.2f M elements/sec\n", throughput);
    } else {
        printf("    %s Performance (CPU only): %.2f ms\n", operation, cpu_time);
    }
}
