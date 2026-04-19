/*
 * CUDA FITACF Processor - GPU-accelerated SuperDARN processing
 * Processes FITACF data using CUDA parallel algorithms
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// SuperDARN constants
#define MAX_RANGE 75
#define MAX_LAGS 17
#define MAX_BEAMS 16
#define C_LIGHT 299792458.0  // Speed of light

// CUDA constants
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 65535

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

// CUDA kernel for ACF fitting
__global__ void cuda_fit_acf_kernel(const complex_t* acf_data, 
                                   float* velocity, float* width, float* power,
                                   float* vel_error, float* width_error, float* power_error,
                                   int* quality, int num_ranges, int num_lags) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_ranges) return;
    
    // Get ACF data for this range gate
    const complex_t* acf = &acf_data[idx * num_lags];
    
    // Initialize outputs
    velocity[idx] = 0.0f;
    width[idx] = 50.0f;
    power[idx] = acf[0].real;  // Lag 0 power
    vel_error[idx] = 10.0f;
    width_error[idx] = 5.0f;
    power_error[idx] = power[idx] * 0.1f;
    quality[idx] = 1;
    
    // Simple ACF fitting using first few lags
    if (num_lags >= 3 && power[idx] > 100.0f) {
        
        // Phase difference method for velocity
        float phase1 = atan2f(acf[1].imag, acf[1].real);
        float phase2 = atan2f(acf[2].imag, acf[2].real);
        
        // Velocity from phase progression
        float phase_diff = phase2 - phase1;
        
        // Unwrap phase
        while (phase_diff > M_PI) phase_diff -= 2.0f * M_PI;
        while (phase_diff < -M_PI) phase_diff += 2.0f * M_PI;
        
        // Convert to velocity (simplified)
        velocity[idx] = phase_diff * 300.0f / (2.0f * M_PI * 0.0024f);  // 2.4 ms lag separation
        
        // Width from amplitude decay
        float amp1 = sqrtf(acf[1].real * acf[1].real + acf[1].imag * acf[1].imag);
        float amp2 = sqrtf(acf[2].real * acf[2].real + acf[2].imag * acf[2].imag);
        
        if (amp1 > 0 && amp2 > 0) {
            float decay_rate = logf(amp1 / amp2) / 0.0024f;  // Decay per second
            width[idx] = decay_rate * 50.0f;  // Convert to spectral width
            
            if (width[idx] < 10.0f) width[idx] = 10.0f;   // Minimum width
            if (width[idx] > 500.0f) width[idx] = 500.0f; // Maximum width
        }
        
        // Error estimates based on SNR
        float snr = power[idx] / 100.0f;  // Assume 100 is noise level
        vel_error[idx] = 50.0f / sqrtf(snr);
        width_error[idx] = 20.0f / sqrtf(snr);
        
    } else {
        quality[idx] = 0;  // Poor quality
    }
}

// CUDA kernel for statistical reduction
__global__ void cuda_statistics_kernel(const float* velocity, const float* width, const float* power,
                                      const int* quality, int num_ranges,
                                      float* sum_vel, float* sum_width, float* sum_power, int* good_count) {
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    sdata[tid] = 0.0f;
    sdata[tid + blockDim.x] = 0.0f;
    sdata[tid + 2 * blockDim.x] = 0.0f;
    sdata[tid + 3 * blockDim.x] = 0.0f;
    
    // Load data
    if (idx < num_ranges && quality[idx] > 0) {
        sdata[tid] = velocity[idx];
        sdata[tid + blockDim.x] = width[idx];
        sdata[tid + 2 * blockDim.x] = power[idx];
        sdata[tid + 3 * blockDim.x] = 1.0f;  // Count
    }
    
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
            sdata[tid + 2 * blockDim.x] += sdata[tid + s + 2 * blockDim.x];
            sdata[tid + 3 * blockDim.x] += sdata[tid + s + 3 * blockDim.x];
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        atomicAdd(sum_vel, sdata[0]);
        atomicAdd(sum_width, sdata[blockDim.x]);
        atomicAdd(sum_power, sdata[2 * blockDim.x]);
        atomicAdd((float*)good_count, sdata[3 * blockDim.x]);
    }
}

// Function prototypes
int load_fitacf_data(const char* filename, fitacf_data_t* data);
void process_beam_cuda(const beam_data_t* beam_data, fit_results_t* results, int num_ranges, int num_lags);
void save_results(const char* filename, const fit_results_t* results, int num_beams, int num_ranges);
double get_time_ms();

// Load FITACF data from file (same as CPU version)
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

// Process a single beam using CUDA
void process_beam_cuda(const beam_data_t* beam_data, fit_results_t* results, int num_ranges, int num_lags) {
    
    // Allocate GPU memory
    complex_t *d_acf;
    float *d_velocity, *d_width, *d_power;
    float *d_vel_error, *d_width_error, *d_power_error;
    int *d_quality;
    
    size_t acf_size = num_ranges * num_lags * sizeof(complex_t);
    size_t float_size = num_ranges * sizeof(float);
    size_t int_size = num_ranges * sizeof(int);
    
    cudaMalloc(&d_acf, acf_size);
    cudaMalloc(&d_velocity, float_size);
    cudaMalloc(&d_width, float_size);
    cudaMalloc(&d_power, float_size);
    cudaMalloc(&d_vel_error, float_size);
    cudaMalloc(&d_width_error, float_size);
    cudaMalloc(&d_power_error, float_size);
    cudaMalloc(&d_quality, int_size);
    
    // Copy ACF data to GPU
    cudaMemcpy(d_acf, beam_data->acf, acf_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    int threads_per_block = THREADS_PER_BLOCK;
    int blocks = (num_ranges + threads_per_block - 1) / threads_per_block;
    
    // Launch CUDA kernel
    cuda_fit_acf_kernel<<<blocks, threads_per_block>>>(
        d_acf, d_velocity, d_width, d_power,
        d_vel_error, d_width_error, d_power_error, d_quality,
        num_ranges, num_lags);
    
    // Wait for kernel completion
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(results->velocity, d_velocity, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->width, d_width, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->power, d_power, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->velocity_error, d_vel_error, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->width_error, d_width_error, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->power_error, d_power_error, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->quality_flag, d_quality, int_size, cudaMemcpyDeviceToHost);
    
    // Clean up GPU memory
    cudaFree(d_acf);
    cudaFree(d_velocity);
    cudaFree(d_width);
    cudaFree(d_power);
    cudaFree(d_vel_error);
    cudaFree(d_width_error);
    cudaFree(d_power_error);
    cudaFree(d_quality);
}

// Save results to file (same as CPU version)
void save_results(const char* filename, const fit_results_t* results, int num_beams, int num_ranges) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    fprintf(file, "# FITACF Processing Results (CUDA)\n");
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
    
    printf("SuperDARN CUDA FITACF Processor\n");
    printf("===============================\n");
    
    if (argc != 2) {
        printf("Usage: %s <fitacf_file>\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    
    // Initialize CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("Error: No CUDA devices found\n");
        return 1;
    }
    
    printf("CUDA devices found: %d\n", device_count);
    
    // Load FITACF data
    printf("Loading FITACF data from: %s\n", input_file);
    
    fitacf_data_t data;
    if (load_fitacf_data(input_file, &data) != 0) {
        return 1;
    }
    
    // Process each beam
    printf("Processing %d beams with CUDA...\n", data.num_beams);
    
    fit_results_t results[MAX_BEAMS];
    
    double start_time = get_time_ms();
    
    for (int beam = 0; beam < data.num_beams; beam++) {
        printf("  Processing beam %d/%d\n", beam + 1, data.num_beams);
        process_beam_cuda(&data.beams[beam], &results[beam], data.num_ranges, data.num_lags);
    }
    
    double end_time = get_time_ms();
    double processing_time = end_time - start_time;
    
    printf("CUDA Processing complete!\n");
    printf("Processing time: %.2f ms\n", processing_time);
    printf("Throughput: %.2f ranges/sec\n", (data.num_beams * data.num_ranges * 1000.0) / processing_time);
    
    // Save results
    save_results("cuda_fitacf_results.txt", results, data.num_beams, data.num_ranges);
    
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