/*
 * CUDA FITACF Processor - Debug Version with Error Checking
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_RANGE 75
#define MAX_LAGS 17
#define MAX_BEAMS 16
#define THREADS_PER_BLOCK 256

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

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

// Simple CUDA kernel for testing
__global__ void cuda_fit_acf_kernel_debug(const complex_t* acf_data, 
                                         float* velocity, float* width, float* power,
                                         float* vel_error, float* width_error, float* power_error,
                                         int* quality, int num_ranges, int num_lags) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_ranges) return;
    
    // Get ACF data for this range gate
    const complex_t* acf = &acf_data[idx * num_lags];
    
    // Initialize with lag-0 power
    power[idx] = acf[0].real;
    
    // Simple processing
    if (power[idx] > 100.0f) {
        // Calculate velocity from phase of lag-1
        if (num_lags > 1) {
            velocity[idx] = atan2f(acf[1].imag, acf[1].real) * 300.0f / (2.0f * 3.14159f * 0.0024f);
        } else {
            velocity[idx] = 0.0f;
        }
        
        // Simple width estimate
        width[idx] = 50.0f + fabs(velocity[idx]) * 0.3f;
        
        // Error estimates
        vel_error[idx] = 10.0f;
        width_error[idx] = 5.0f;
        power_error[idx] = power[idx] * 0.1f;
        quality[idx] = 1;
        
    } else {
        velocity[idx] = 0.0f;
        width[idx] = 0.0f;
        vel_error[idx] = 0.0f;
        width_error[idx] = 0.0f;
        power_error[idx] = 0.0f;
        quality[idx] = 0;
    }
}

int load_fitacf_data(const char* filename, fitacf_data_t* data) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    char header[16];
    fread(header, 1, 16, file);
    
    if (strncmp(header, "FITACF_TEST_DATA", 16) != 0) {
        printf("Error: Invalid file format\n");
        fclose(file);
        return -1;
    }
    
    fread(&data->num_beams, sizeof(int), 1, file);
    fread(&data->num_ranges, sizeof(int), 1, file);
    fread(&data->num_lags, sizeof(int), 1, file);
    
    printf("Loading data: %d beams, %d ranges, %d lags\n", 
           data->num_beams, data->num_ranges, data->num_lags);
    
    for (int beam = 0; beam < data->num_beams; beam++) {
        beam_data_t* beam_data = &data->beams[beam];
        
        fread(&beam_data->beam_num, sizeof(int), 1, file);
        fread(&beam_data->date, sizeof(int), 1, file);
        fread(&beam_data->time, sizeof(int), 1, file);
        
        for (int rng = 0; rng < data->num_ranges; rng++) {
            fread(&beam_data->power[rng], sizeof(float), 1, file);
            
            for (int lag = 0; lag < data->num_lags; lag++) {
                fread(&beam_data->acf[rng][lag].real, sizeof(float), 1, file);
                fread(&beam_data->acf[rng][lag].imag, sizeof(float), 1, file);
            }
        }
    }
    
    fclose(file);
    return 0;
}

void process_beam_cuda_debug(const beam_data_t* beam_data, fit_results_t* results, int num_ranges, int num_lags) {
    
    printf("  Debug: Processing beam with %d ranges, %d lags\n", num_ranges, num_lags);
    
    // Check first few ACF values
    printf("  Debug: First ACF values: %.2f + %.2f*i, %.2f + %.2f*i\n",
           beam_data->acf[0][0].real, beam_data->acf[0][0].imag,
           beam_data->acf[0][1].real, beam_data->acf[0][1].imag);
    
    // Allocate GPU memory
    complex_t *d_acf;
    float *d_velocity, *d_width, *d_power;
    float *d_vel_error, *d_width_error, *d_power_error;
    int *d_quality;
    
    size_t acf_size = num_ranges * num_lags * sizeof(complex_t);
    size_t float_size = num_ranges * sizeof(float);
    size_t int_size = num_ranges * sizeof(int);
    
    printf("  Debug: Allocating GPU memory - ACF: %zu bytes\n", acf_size);
    
    CUDA_CHECK(cudaMalloc(&d_acf, acf_size));
    CUDA_CHECK(cudaMalloc(&d_velocity, float_size));
    CUDA_CHECK(cudaMalloc(&d_width, float_size));
    CUDA_CHECK(cudaMalloc(&d_power, float_size));
    CUDA_CHECK(cudaMalloc(&d_vel_error, float_size));
    CUDA_CHECK(cudaMalloc(&d_width_error, float_size));
    CUDA_CHECK(cudaMalloc(&d_power_error, float_size));
    CUDA_CHECK(cudaMalloc(&d_quality, int_size));
    
    // Copy ACF data to GPU
    printf("  Debug: Copying data to GPU\n");
    CUDA_CHECK(cudaMemcpy(d_acf, beam_data->acf, acf_size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    int threads_per_block = THREADS_PER_BLOCK;
    int blocks = (num_ranges + threads_per_block - 1) / threads_per_block;
    
    printf("  Debug: Launching kernel with %d blocks, %d threads\n", blocks, threads_per_block);
    
    // Launch CUDA kernel
    cuda_fit_acf_kernel_debug<<<blocks, threads_per_block>>>(
        d_acf, d_velocity, d_width, d_power,
        d_vel_error, d_width_error, d_power_error, d_quality,
        num_ranges, num_lags);
    
    // Check for kernel launch error
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  Debug: Kernel completed, copying results back\n");
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(results->velocity, d_velocity, float_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->width, d_width, float_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->power, d_power, float_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->velocity_error, d_vel_error, float_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->width_error, d_width_error, float_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->power_error, d_power_error, float_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->quality_flag, d_quality, int_size, cudaMemcpyDeviceToHost));
    
    printf("  Debug: First result - Power: %.2f, Velocity: %.2f, Quality: %d\n",
           results->power[0], results->velocity[0], results->quality_flag[0]);
    
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

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(int argc, char* argv[]) {
    
    printf("SuperDARN CUDA FITACF Processor (Debug)\n");
    printf("========================================\n");
    
    if (argc != 2) {
        printf("Usage: %s <fitacf_file>\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    
    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
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
    
    // Process first beam only for debugging
    printf("Processing first beam for debugging...\n");
    
    fit_results_t result;
    
    double start_time = get_time_ms();
    
    process_beam_cuda_debug(&data.beams[0], &result, data.num_ranges, data.num_lags);
    
    double end_time = get_time_ms();
    double processing_time = end_time - start_time;
    
    printf("CUDA Processing complete!\n");
    printf("Processing time: %.2f ms\n", processing_time);
    
    // Print first few results
    printf("\nFirst 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("Range %d: Power=%.2f, Velocity=%.2f, Width=%.2f, Quality=%d\n",
               i, result.power[i], result.velocity[i], result.width[i], result.quality_flag[i]);
    }
    
    return 0;
}