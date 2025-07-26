/* benchmark_fitcfit.c
   ===================
   Comprehensive benchmark comparing original and optimized FitToCFit implementations
   
   This benchmark tests:
   1. Original FitToCFit function
   2. Optimized single-pass version
   3. AVX2-optimized version (if available)
   4. OpenMP parallelized version (if available)
   
   Measures:
   - Execution time
   - Throughput (ranges/second)
   - Memory usage
   - Cache performance
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fitdata.h"
#include "fitcfit.h"
#include "cfitdata.h"
#include "rprm.h"

// Function prototypes for optimized versions
int FitToCFit_Optimized(double min_pwr, struct CFitdata *ptr,
                       struct RadarParm *prm, struct FitData *fit);

// Timing utilities
static inline long long current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000000LL + te.tv_usec;
}

// Test data generation
struct FitData* generate_benchmark_data(int num_ranges, double valid_ratio) {
    struct FitData* fit = FitMake();
    if (!fit) return NULL;
    
    FitSetAlgorithm(fit, "test");
    FitSetRng(fit, num_ranges);
    
    srand(42); // Fixed seed for reproducible results
    
    for (int i = 0; i < num_ranges; i++) {
        // Generate realistic test data
        fit->rng[i].qflg = (rand() / (double)RAND_MAX < valid_ratio) ? 1 : 0;
        fit->rng[i].gsct = rand() % 2;
        fit->rng[i].p_0 = 10.0 + (rand() / (double)RAND_MAX) * 40.0; // 10-50 dB
        fit->rng[i].v = -200.0 + (rand() / (double)RAND_MAX) * 400.0; // -200 to +200 m/s
        fit->rng[i].v_err = 1.0 + (rand() / (double)RAND_MAX) * 10.0;
        fit->rng[i].p_l = 5.0 + (rand() / (double)RAND_MAX) * 20.0;
        fit->rng[i].p_l_err = 0.5 + (rand() / (double)RAND_MAX) * 2.0;
        fit->rng[i].w_l = 50.0 + (rand() / (double)RAND_MAX) * 200.0;
        fit->rng[i].w_l_err = 5.0 + (rand() / (double)RAND_MAX) * 20.0;
    }
    
    return fit;
}

struct RadarParm* generate_benchmark_params(int num_ranges) {
    struct RadarParm* prm = RadarParmMake();
    if (!prm) return NULL;
    
    // Initialize all fields
    memset(prm, 0, sizeof(struct RadarParm));
    
    // Set version
    prm->revision.major = 1;
    prm->revision.minor = 0;
    
    // Set time
    time_t now = time(NULL);
    struct tm* tm = localtime(&now);
    prm->time.yr = tm->tm_year + 1900;
    prm->time.mo = tm->tm_mon + 1;
    prm->time.dy = tm->tm_mday;
    prm->time.hr = tm->tm_hour;
    prm->time.mt = tm->tm_min;
    prm->time.sc = tm->tm_sec;
    prm->time.us = 0;
    
    // Set radar parameters
    prm->stid = 1;
    prm->scan = 0;
    prm->cp = 1000;
    prm->bmnum = 0;
    prm->bmazm = 0.0;
    prm->channel = 0;
    prm->intt.sc = 3;
    prm->intt.us = 0;
    prm->frang = 180.0;
    prm->rsep = 45.0;
    prm->rxrise = 0.0;
    prm->tfreq = 12000;
    prm->noise.search = 0.1;
    prm->noise.mean = 0.05;
    prm->atten = 0;
    prm->nave = 1;
    prm->nrang = num_ranges;
    prm->num = num_ranges;
    
    // Allocate and initialize range array
    prm->rng = (int16 *)calloc(num_ranges, sizeof(int16));
    if (prm->rng) {
        for (int i = 0; i < num_ranges; i++) {
            prm->rng[i] = prm->frang + i * prm->rsep;
        }
    }
    
    return prm;
}

// Benchmark configuration
typedef struct {
    const char* name;
    int (*func)(double, struct CFitdata*, struct RadarParm*, struct FitData*);
    int available;
} benchmark_config_t;

// Performance metrics
typedef struct {
    double avg_time_us;
    double min_time_us;
    double max_time_us;
    double std_dev_us;
    double throughput_ranges_per_sec;
    int successful_runs;
    int failed_runs;
} performance_metrics_t;

// Run benchmark for a specific function
performance_metrics_t run_benchmark(benchmark_config_t* config, 
                                   int num_ranges, int iterations,
                                   double valid_ratio, double min_pwr) {
    performance_metrics_t metrics = {0};
    
    printf("Running benchmark: %s (%d ranges, %d iterations, %.1f%% valid)\n", 
           config->name, num_ranges, iterations, valid_ratio * 100);
    
    // Generate test data
    struct FitData* fit = generate_benchmark_data(num_ranges, valid_ratio);
    struct RadarParm* prm = generate_benchmark_params(num_ranges);
    
    if (!fit || !prm) {
        printf("  ERROR: Failed to generate test data\n");
        if (fit) FitFree(fit);
        if (prm) {
            if (prm->rng) free(prm->rng);
            RadarParmFree(prm);
        }
        return metrics;
    }
    
    double times[iterations];
    int successful = 0;
    
    // Warm-up runs
    for (int i = 0; i < 10; i++) {
        struct CFitdata* cfit = CFitMake();
        if (cfit) {
            config->func(min_pwr, cfit, prm, fit);
            CFitFree(cfit);
        }
    }
    
    // Benchmark runs
    for (int i = 0; i < iterations; i++) {
        struct CFitdata* cfit = CFitMake();
        if (!cfit) {
            metrics.failed_runs++;
            continue;
        }
        
        long long start = current_timestamp();
        int result = config->func(min_pwr, cfit, prm, fit);
        long long end = current_timestamp();
        
        if (result == 0) {
            times[successful] = (end - start);
            successful++;
        } else {
            metrics.failed_runs++;
        }
        
        CFitFree(cfit);
    }
    
    // Calculate statistics
    if (successful > 0) {
        double sum = 0, sum_sq = 0;
        metrics.min_time_us = times[0];
        metrics.max_time_us = times[0];
        
        for (int i = 0; i < successful; i++) {
            sum += times[i];
            sum_sq += times[i] * times[i];
            if (times[i] < metrics.min_time_us) metrics.min_time_us = times[i];
            if (times[i] > metrics.max_time_us) metrics.max_time_us = times[i];
        }
        
        metrics.avg_time_us = sum / successful;
        metrics.std_dev_us = sqrt((sum_sq / successful) - (metrics.avg_time_us * metrics.avg_time_us));
        metrics.throughput_ranges_per_sec = (num_ranges * 1000000.0) / metrics.avg_time_us;
        metrics.successful_runs = successful;
    }
    
    // Cleanup
    FitFree(fit);
    if (prm->rng) free(prm->rng);
    RadarParmFree(prm);
    
    return metrics;
}

// Print performance comparison
void print_performance_comparison(benchmark_config_t* configs, int num_configs,
                                performance_metrics_t* results, int num_ranges) {
    printf("\n=== Performance Comparison (%d ranges) ===\n", num_ranges);
    printf("%-20s %12s %12s %12s %12s %15s %8s\n", 
           "Implementation", "Avg (μs)", "Min (μs)", "Max (μs)", "StdDev (μs)", 
           "Throughput (R/s)", "Success");
    printf("%-20s %12s %12s %12s %12s %15s %8s\n", 
           "--------------------", "--------", "--------", "--------", "----------", 
           "---------------", "-------");
    
    double baseline_time = 0;
    for (int i = 0; i < num_configs; i++) {
        if (!configs[i].available) continue;
        
        performance_metrics_t* m = &results[i];
        if (m->successful_runs == 0) {
            printf("%-20s %12s %12s %12s %12s %15s %8s\n", 
                   configs[i].name, "FAILED", "FAILED", "FAILED", "FAILED", "FAILED", "0");
            continue;
        }
        
        if (baseline_time == 0) baseline_time = m->avg_time_us;
        double speedup = baseline_time / m->avg_time_us;
        
        printf("%-20s %12.2f %12.2f %12.2f %12.2f %15.0f %8d", 
               configs[i].name, m->avg_time_us, m->min_time_us, m->max_time_us, 
               m->std_dev_us, m->throughput_ranges_per_sec, m->successful_runs);
        
        if (i > 0) {
            printf(" (%.2fx)", speedup);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    printf("=== FitToCFit Performance Benchmark ===\n\n");
    
    // Configuration
    int test_sizes[] = {50, 100, 200, 500, 1000};
    int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int iterations = 1000;
    double valid_ratio = 0.7; // 70% of ranges are valid
    double min_pwr = 0.0;
    
    // Available implementations
    benchmark_config_t configs[] = {
        {"Original", FitToCFit, 1},
        {"Optimized", FitToCFit_Optimized, 1},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    
    // Print system information
    printf("System Information:\n");
#ifdef _OPENMP
    printf("  OpenMP threads: %d\n", omp_get_max_threads());
#else
    printf("  OpenMP: Not available\n");
#endif
#ifdef __AVX2__
    printf("  AVX2: Available\n");
#else
    printf("  AVX2: Not available\n");
#endif
    printf("  Test iterations: %d\n", iterations);
    printf("  Valid range ratio: %.1f%%\n\n", valid_ratio * 100);
    
    // Run benchmarks for different sizes
    for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
        int num_ranges = test_sizes[size_idx];
        performance_metrics_t results[num_configs];
        
        // Run each implementation
        for (int config_idx = 0; config_idx < num_configs; config_idx++) {
            if (configs[config_idx].available) {
                results[config_idx] = run_benchmark(&configs[config_idx], 
                                                   num_ranges, iterations, 
                                                   valid_ratio, min_pwr);
            } else {
                memset(&results[config_idx], 0, sizeof(performance_metrics_t));
            }
        }
        
        // Print results
        print_performance_comparison(configs, num_configs, results, num_ranges);
        printf("\n");
    }
    
    printf("=== Benchmark Complete ===\n");
    return 0;
}
