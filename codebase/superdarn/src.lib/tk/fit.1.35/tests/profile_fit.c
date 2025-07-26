#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "fitdata.h"
#include "fitcfit.h"
#include "cfitdata.h"
#include "rprm.h"

// Function to get current time in microseconds
long long current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000000LL + te.tv_usec;
}

// Function to generate random FitData with specified number of ranges
struct FitData* generate_test_data(int num_ranges) {
    struct FitData* fit = FitMake();
    if (!fit) return NULL;
    
    // Set algorithm and revision
    FitSetAlgorithm(fit, "test");
    
    // Allocate ranges
    if (FitSetRng(fit, num_ranges) == -1) {
        FitFree(fit);
        return NULL;
    }
    
    // Fill with random data
    for (int i = 0; i < num_ranges; i++) {
        fit->rng[i].qflg = 1;  // Mark as valid
        fit->rng[i].p_0 = (double)rand() / RAND_MAX * 100.0;  // Random power 0-100
        fit->rng[i].v = (double)rand() / RAND_MAX * 2000.0 - 1000.0;  // Random velocity -1000 to 1000
        fit->rng[i].v_err = (double)rand() / RAND_MAX * 100.0;  // Random error 0-100
        fit->rng[i].p_l = (double)rand() / RAND_MAX * 100.0;  // Random power 0-100
        fit->rng[i].p_l_err = (double)rand() / RAND_MAX * 10.0;  // Random error 0-10
        fit->rng[i].gsct = rand() % 10;  // Random ground scatter flag
    }
    
    return fit;
}

// Function to generate test radar parameters
void generate_radar_params(struct RadarParm* prm, int num_ranges) {
    // Initialize all fields to zero first
    memset(prm, 0, sizeof(struct RadarParm));
    
    // Set version information
    prm->revision.major = 1;
    prm->revision.minor = 0;
    
    // Set time information
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
    if (!prm->rng) {
        fprintf(stderr, "Failed to allocate range array\n");
        return;
    }
    
    // Fill range array with test data
    for (int i = 0; i < num_ranges; i++) {
        prm->rng[i] = prm->frang + i * prm->rsep;
    }
}

// Profile FitToCFit function
void profile_fit_to_cfit(int num_ranges, int num_iterations) {
    printf("\n=== Profiling FitToCFit (%d ranges, %d iterations) ===\n", num_ranges, num_iterations);
    
    // Generate test data
    struct FitData* fit = generate_test_data(num_ranges);
    if (!fit) {
        printf("Failed to generate test data\n");
        return;
    }
    
    struct RadarParm prm;
    generate_radar_params(&prm, num_ranges);
    if (!prm.rng) {
        printf("Failed to initialize radar parameters\n");
        FitFree(fit);
        return;
    }
    
    // Allocate CFitData
    struct CFitdata* cfit = CFitMake();
    if (!cfit) {
        printf("Failed to allocate CFitData\n");
        free(prm.rng);
        FitFree(fit);
        return;
    }
    
    // Warm-up run
    printf("Performing warm-up run...\n");
    FitToCFit(0.0, cfit, &prm, fit);
    
    // Profile
    printf("Starting profiling...\n");
    long long start = current_timestamp();
    for (int i = 0; i < num_iterations; i++) {
        FitToCFit(0.0, cfit, &prm, fit);
    }
    long long end = current_timestamp();
    
    // Calculate and print results
    double total_time = (end - start) / 1000000.0; // Convert to seconds
    double avg_time = (total_time * 1000.0) / num_iterations; // Convert to milliseconds
    
    printf("Total time: %.6f seconds\n", total_time);
    printf("Average time per iteration: %.6f ms\n", avg_time);
    
    // Clean up
    printf("Cleaning up...\n");
    CFitFree(cfit);
    free(prm.rng);
    FitFree(fit);
    printf("Done.\n");
}

int main(int argc, char* argv[]) {
    // Test with different range sizes
    int test_sizes[] = {100, 1000, 10000, 50000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("=== Starting Fit.1.35 Profiling ===\n");
    
    for (int i = 0; i < num_tests; i++) {
        int num_ranges = test_sizes[i];
        int num_iterations = (num_ranges < 1000) ? 1000 : 100;
        
        printf("\n--- Test %d: %d ranges, %d iterations ---\n", 
               i + 1, num_ranges, num_iterations);
        
        profile_fit_to_cfit(num_ranges, num_iterations);
    }
    
    printf("\n=== Profiling Complete ===\n");
    return 0;
}
