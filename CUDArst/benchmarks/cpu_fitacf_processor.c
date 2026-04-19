/*
 * CPU FITACF Processor - Traditional SuperDARN processing
 * Processes FITACF data using original CPU algorithms
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// SuperDARN constants
#define MAX_RANGE 75
#define MAX_LAGS 17
#define MAX_BEAMS 16
#define C_LIGHT 299792458.0  // Speed of light

// Data structures
typedef struct {
    float real;
    float imag;
} complex_t;

typedef struct {
    int beam_num;
    int date;
    int time;
    float power[MAX_RANGE];
    complex_t acf[MAX_RANGE][MAX_LAGS];
} beam_data_t;

typedef struct {
    int num_beams;
    int num_ranges;
    int num_lags;
    beam_data_t beams[MAX_BEAMS];
} fitacf_data_t;

typedef struct {
    float velocity[MAX_RANGE];
    float width[MAX_RANGE];
    float power[MAX_RANGE];
    float velocity_error[MAX_RANGE];
    float width_error[MAX_RANGE];
    float power_error[MAX_RANGE];
    int quality_flag[MAX_RANGE];
} fit_results_t;

// Function prototypes
int load_fitacf_data(const char* filename, fitacf_data_t* data);
void process_beam_cpu(const beam_data_t* beam_data, fit_results_t* results, int num_ranges, int num_lags);
void fit_acf_cpu(const complex_t* acf, int num_lags, float* velocity, float* width, float* power, float* vel_error, float* width_error, float* power_error, int* quality);
void save_results(const char* filename, const fit_results_t* results, int num_beams, int num_ranges);
double get_time_ms();

// Load FITACF data from file
int load_fitacf_data(const char* filename, fitacf_data_t* data) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    // Read header
    char header[16];
    fread(header, 1, 16, file);
    
    if (strncmp(header, "FITACF_TEST_DATA", 16) != 0) {
        printf("Error: Invalid file format\n");
        fclose(file);
        return -1;
    }
    
    // Read parameters
    fread(&data->num_beams, sizeof(int), 1, file);
    fread(&data->num_ranges, sizeof(int), 1, file);
    fread(&data->num_lags, sizeof(int), 1, file);
    
    printf("Loading data: %d beams, %d ranges, %d lags\n", 
           data->num_beams, data->num_ranges, data->num_lags);
    
    // Read beam data
    for (int beam = 0; beam < data->num_beams; beam++) {
        beam_data_t* beam_data = &data->beams[beam];
        
        // Read beam header
        fread(&beam_data->beam_num, sizeof(int), 1, file);
        fread(&beam_data->date, sizeof(int), 1, file);
        fread(&beam_data->time, sizeof(int), 1, file);
        
        // Read range data
        for (int rng = 0; rng < data->num_ranges; rng++) {
            // Read power
            fread(&beam_data->power[rng], sizeof(float), 1, file);
            
            // Read ACF data
            for (int lag = 0; lag < data->num_lags; lag++) {
                fread(&beam_data->acf[rng][lag].real, sizeof(float), 1, file);
                fread(&beam_data->acf[rng][lag].imag, sizeof(float), 1, file);
            }
        }
    }
    
    fclose(file);
    return 0;
}

// CPU ACF fitting algorithm (simplified)
void fit_acf_cpu(const complex_t* acf, int num_lags, float* velocity, float* width, 
                 float* power, float* vel_error, float* width_error, float* power_error, int* quality) {
    
    // Initialize outputs
    *velocity = 0.0;
    *width = 50.0;
    *power = acf[0].real;  // Lag 0 power
    *vel_error = 10.0;
    *width_error = 5.0;
    *power_error = *power * 0.1;
    *quality = 1;
    
    // Simple ACF fitting using first few lags
    if (num_lags >= 3 && *power > 100.0) {
        
        // Phase difference method for velocity
        float phase1 = atan2(acf[1].imag, acf[1].real);
        float phase2 = atan2(acf[2].imag, acf[2].real);
        
        // Velocity from phase progression
        float phase_diff = phase2 - phase1;
        
        // Unwrap phase
        while (phase_diff > M_PI) phase_diff -= 2.0 * M_PI;
        while (phase_diff < -M_PI) phase_diff += 2.0 * M_PI;
        
        // Convert to velocity (simplified)
        *velocity = phase_diff * 300.0 / (2.0 * M_PI * 0.0024);  // 2.4 ms lag separation
        
        // Width from amplitude decay
        float amp1 = sqrt(acf[1].real * acf[1].real + acf[1].imag * acf[1].imag);
        float amp2 = sqrt(acf[2].real * acf[2].real + acf[2].imag * acf[2].imag);
        
        if (amp1 > 0 && amp2 > 0) {
            float decay_rate = log(amp1 / amp2) / 0.0024;  // Decay per second
            *width = decay_rate * 50.0;  // Convert to spectral width
            
            if (*width < 10.0) *width = 10.0;   // Minimum width
            if (*width > 500.0) *width = 500.0; // Maximum width
        }
        
        // Error estimates based on SNR
        float snr = *power / 100.0;  // Assume 100 is noise level
        *vel_error = 50.0 / sqrt(snr);
        *width_error = 20.0 / sqrt(snr);
        
    } else {
        *quality = 0;  // Poor quality
    }
}

// Process a single beam using CPU
void process_beam_cpu(const beam_data_t* beam_data, fit_results_t* results, int num_ranges, int num_lags) {
    
    for (int rng = 0; rng < num_ranges; rng++) {
        
        // Fit ACF for this range gate
        fit_acf_cpu(beam_data->acf[rng], num_lags,
                   &results->velocity[rng],
                   &results->width[rng], 
                   &results->power[rng],
                   &results->velocity_error[rng],
                   &results->width_error[rng],
                   &results->power_error[rng],
                   &results->quality_flag[rng]);
    }
}

// Save results to file
void save_results(const char* filename, const fit_results_t* results, int num_beams, int num_ranges) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    fprintf(file, "# FITACF Processing Results\n");
    fprintf(file, "# Beams: %d, Ranges: %d\n", num_beams, num_ranges);
    fprintf(file, "# Format: beam range velocity width power vel_error width_error power_error quality\n");
    
    for (int beam = 0; beam < num_beams; beam++) {
        for (int rng = 0; rng < num_ranges; rng++) {
            fprintf(file, "%d %d %.2f %.2f %.2f %.2f %.2f %.2f %d\n",
                   beam, rng,
                   results[beam].velocity[rng],
                   results[beam].width[rng],
                   results[beam].power[rng],
                   results[beam].velocity_error[rng],
                   results[beam].width_error[rng],
                   results[beam].power_error[rng],
                   results[beam].quality_flag[rng]);
        }
    }
    
    fclose(file);
    printf("Results saved to: %s\n", filename);
}

// Get current time in milliseconds
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Main processing function
int main(int argc, char* argv[]) {
    
    printf("SuperDARN CPU FITACF Processor\n");
    printf("==============================\n");
    
    if (argc != 2) {
        printf("Usage: %s <fitacf_file>\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    
    // Load FITACF data
    printf("Loading FITACF data from: %s\n", input_file);
    
    fitacf_data_t data;
    if (load_fitacf_data(input_file, &data) != 0) {
        return 1;
    }
    
    // Process each beam
    printf("Processing %d beams...\n", data.num_beams);
    
    fit_results_t results[MAX_BEAMS];
    
    double start_time = get_time_ms();
    
    for (int beam = 0; beam < data.num_beams; beam++) {
        printf("  Processing beam %d/%d\n", beam + 1, data.num_beams);
        process_beam_cpu(&data.beams[beam], &results[beam], data.num_ranges, data.num_lags);
    }
    
    double end_time = get_time_ms();
    double processing_time = end_time - start_time;
    
    printf("CPU Processing complete!\n");
    printf("Processing time: %.2f ms\n", processing_time);
    printf("Throughput: %.2f ranges/sec\n", (data.num_beams * data.num_ranges * 1000.0) / processing_time);
    
    // Save results
    save_results("cpu_fitacf_results.txt", results, data.num_beams, data.num_ranges);
    
    // Print summary statistics
    printf("\nSummary Statistics:\n");
    printf("===================\n");
    
    int total_ranges = 0;
    int good_ranges = 0;
    float avg_velocity = 0.0;
    float avg_width = 0.0;
    float avg_power = 0.0;
    
    for (int beam = 0; beam < data.num_beams; beam++) {
        for (int rng = 0; rng < data.num_ranges; rng++) {
            total_ranges++;
            if (results[beam].quality_flag[rng] > 0) {
                good_ranges++;
                avg_velocity += results[beam].velocity[rng];
                avg_width += results[beam].width[rng];
                avg_power += results[beam].power[rng];
            }
        }
    }
    
    if (good_ranges > 0) {
        avg_velocity /= good_ranges;
        avg_width /= good_ranges;
        avg_power /= good_ranges;
        
        printf("Total ranges processed: %d\n", total_ranges);
        printf("Good quality ranges: %d (%.1f%%)\n", good_ranges, 100.0 * good_ranges / total_ranges);
        printf("Average velocity: %.2f m/s\n", avg_velocity);
        printf("Average spectral width: %.2f m/s\n", avg_width);
        printf("Average power: %.2f dB\n", 10.0 * log10(avg_power));
    }
    
    return 0;
}