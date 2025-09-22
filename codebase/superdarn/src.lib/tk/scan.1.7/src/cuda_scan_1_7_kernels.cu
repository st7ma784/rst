/**
 * @file cuda_scan_1_7_kernels.cu
 * @brief CUDA kernel implementations for scan.1.7 module
 * 
 * Provides GPU acceleration for SuperDARN scan data processing operations
 * including beam processing, range gate computations, and scan management.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/copy_if.h>
#include <stdio.h>
#include <float.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

/* CUDA-compatible scan data structures */
typedef struct {
    int gsct;           // Ground scatter flag
    float p_0;          // Lag-0 power
    float p_0_e;        // Lag-0 power error
    float v;            // Velocity
    float v_e;          // Velocity error  
    float w_l;          // Lambda spectral width
    float w_l_e;        // Lambda spectral width error
    float p_l;          // Lambda power
    float p_l_e;        // Lambda power error
    float phi0;         // Phase
    float elv;          // Elevation angle
} cuda_radar_cell_t;

typedef struct {
    int scan, bm;               // Scan and beam numbers
    float bmazm;                // Beam azimuth
    double time;                // Timestamp
    int cpid, nave, frang, rsep, rxrise, freq, noise, atten, channel, nrang;
    int intt_sc, intt_us;       // Integration time
    unsigned char *sct;         // Scatter flags array
    cuda_radar_cell_t *rng;     // Range cell data array
    bool valid;                 // Validity flag for filtering
} cuda_radar_beam_t;

typedef struct {
    int stid;                   // Station ID
    int version_major, version_minor;  // Version info
    double st_time, ed_time;    // Start and end times
    int num;                    // Number of beams
    cuda_radar_beam_t *bm;      // Beam array
    int capacity;               // Allocated capacity
} cuda_radar_scan_t;

/**
 * Parallel beam processing kernel
 * Processes multiple radar beams simultaneously
 */
__global__ void cuda_scan_process_beams_kernel(cuda_radar_beam_t *beams,
                                              int num_beams,
                                              int max_range_gates) {
    int beam_idx = blockIdx.x;
    int range_idx = threadIdx.x;
    
    if (beam_idx >= num_beams || range_idx >= max_range_gates) return;
    if (range_idx >= beams[beam_idx].nrang) return;
    
    cuda_radar_beam_t *beam = &beams[beam_idx];
    cuda_radar_cell_t *cell = &beam->rng[range_idx];
    
    // Apply noise filtering based on power threshold
    float noise_threshold = beam->noise * 1.5f;  // 1.5x noise threshold
    
    if (cell->p_0 < noise_threshold) {
        cell->gsct = 0;         // Mark as invalid
        cell->v = 0.0f;         // Zero velocity
        cell->v_e = FLT_MAX;    // High error
        cell->p_l = 0.0f;       // Zero power
    } else {
        // Apply ground scatter classification
        if (cell->w_l < 50.0f && fabsf(cell->v) < 100.0f) {
            cell->gsct = 1;     // Ground scatter
        } else {
            cell->gsct = 0;     // Ionospheric scatter
        }
        
        // Velocity error estimation based on power
        if (cell->p_0 > 0.0f) {
            float snr = cell->p_0 / beam->noise;
            cell->v_e = 50.0f / sqrtf(snr);  // Error inversely proportional to SNR
        }
    }
}

/**
 * Beam filtering kernel based on beam number
 * Implements parallel version of RadarScanResetBeam
 */
__global__ void cuda_scan_filter_beams_kernel(cuda_radar_beam_t *beams,
                                              int num_beams,
                                              int target_beam,
                                              bool *valid_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_beams) return;
    
    // Mark beam as valid if it matches target beam or if target_beam < 0 (keep all)
    valid_mask[idx] = (target_beam < 0) || (beams[idx].bm == target_beam);
}

/**
 * Scan validation kernel
 * Parallel implementation of exclude_outofscan functionality
 */
__global__ void cuda_scan_validate_kernel(cuda_radar_beam_t *beams,
                                          int num_beams,
                                          bool *valid_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_beams) return;
    
    cuda_radar_beam_t *beam = &beams[idx];
    
    // Check if beam has valid scan number (scan >= 0)
    bool is_valid_scan = (beam->scan >= 0);
    
    // Check if beam has reasonable parameters
    bool is_valid_params = (beam->nrang > 0 && beam->nrang <= 1000) &&
                          (beam->frang >= 0 && beam->frang <= 5000) &&
                          (beam->rsep > 0 && beam->rsep <= 500);
    
    // Check if beam has valid time
    bool is_valid_time = (beam->time > 0.0);
    
    valid_mask[idx] = is_valid_scan && is_valid_params && is_valid_time;
}

/**
 * Range gate statistics kernel
 * Computes statistics across range gates for quality assessment
 */
__global__ void cuda_scan_range_statistics_kernel(cuda_radar_beam_t *beams,
                                                  int num_beams,
                                                  float *power_mean,
                                                  float *velocity_mean,
                                                  float *width_mean,
                                                  int *valid_count) {
    int beam_idx = blockIdx.x;
    int range_idx = threadIdx.x;
    
    if (beam_idx >= num_beams) return;
    
    cuda_radar_beam_t *beam = &beams[beam_idx];
    if (range_idx >= beam->nrang) return;
    
    cuda_radar_cell_t *cell = &beam->rng[range_idx];
    
    // Use shared memory for reduction
    extern __shared__ float sdata[];
    float *s_power = sdata;
    float *s_velocity = sdata + blockDim.x;
    float *s_width = sdata + 2 * blockDim.x;
    int *s_count = (int*)(sdata + 3 * blockDim.x);
    
    // Initialize shared memory
    s_power[range_idx] = (cell->gsct == 0 && cell->p_0 > 0) ? cell->p_0 : 0.0f;
    s_velocity[range_idx] = (cell->gsct == 0 && cell->p_0 > 0) ? cell->v : 0.0f;
    s_width[range_idx] = (cell->gsct == 0 && cell->p_0 > 0) ? cell->w_l : 0.0f;
    s_count[range_idx] = (cell->gsct == 0 && cell->p_0 > 0) ? 1 : 0;
    
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (range_idx < s) {
            s_power[range_idx] += s_power[range_idx + s];
            s_velocity[range_idx] += s_velocity[range_idx + s];
            s_width[range_idx] += s_width[range_idx + s];
            s_count[range_idx] += s_count[range_idx + s];
        }
        __syncthreads();
    }
    
    // Write results
    if (range_idx == 0) {
        int count = s_count[0];
        power_mean[beam_idx] = (count > 0) ? s_power[0] / count : 0.0f;
        velocity_mean[beam_idx] = (count > 0) ? s_velocity[0] / count : 0.0f;
        width_mean[beam_idx] = (count > 0) ? s_width[0] / count : 0.0f;
        valid_count[beam_idx] = count;
    }
}

/**
 * Scatter classification kernel
 * Advanced ground scatter vs ionospheric scatter classification
 */
__global__ void cuda_scan_scatter_classification_kernel(cuda_radar_beam_t *beams,
                                                        int num_beams,
                                                        int max_range_gates) {
    int beam_idx = blockIdx.x;
    int range_idx = threadIdx.x;
    
    if (beam_idx >= num_beams || range_idx >= max_range_gates) return;
    if (range_idx >= beams[beam_idx].nrang) return;
    
    cuda_radar_beam_t *beam = &beams[beam_idx];
    cuda_radar_cell_t *cell = &beam->rng[range_idx];
    
    if (cell->p_0 <= 0.0f) return;  // Skip invalid cells
    
    // Multi-parameter ground scatter classification
    float velocity_threshold = 150.0f;  // m/s
    float width_threshold = 200.0f;     // m/s
    float power_ratio_threshold = 0.3f;
    
    // Calculate range-dependent parameters
    float range_km = beam->frang + range_idx * beam->rsep;
    float range_factor = expf(-range_km / 1000.0f);  // Exponential decay with range
    
    // Velocity-based classification
    bool low_velocity = (fabsf(cell->v) < velocity_threshold * range_factor);
    
    // Width-based classification  
    bool narrow_width = (cell->w_l < width_threshold);
    
    // Power-based classification (ground scatter often has consistent power)
    bool steady_power = (cell->p_l / cell->p_0 > power_ratio_threshold);
    
    // Elevation-based classification (ground scatter typically low elevation)
    bool low_elevation = (cell->elv < 10.0f);  // degrees
    
    // Combined classification
    int gs_score = low_velocity + narrow_width + steady_power + low_elevation;
    
    // Classify as ground scatter if 3 or more criteria met
    cell->gsct = (gs_score >= 3) ? 1 : 0;
}

/**
 * Beam compaction kernel
 * Removes invalid beams from the scan array
 */
__global__ void cuda_scan_compact_beams_kernel(cuda_radar_beam_t *input_beams,
                                               cuda_radar_beam_t *output_beams,
                                               bool *valid_mask,
                                               int *scan_indices,
                                               int num_beams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_beams) return;
    
    if (valid_mask[idx]) {
        int output_idx = scan_indices[idx];
        output_beams[output_idx] = input_beams[idx];
    }
}

/**
 * Time-based beam sorting kernel preparation
 * Prepares sorting keys for time-based beam ordering
 */
__global__ void cuda_scan_prepare_sort_keys_kernel(cuda_radar_beam_t *beams,
                                                   double *sort_keys,
                                                   int num_beams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_beams) return;
    
    // Create composite sort key: time * 1000 + beam_number
    // This ensures chronological order with beam number as secondary sort
    sort_keys[idx] = beams[idx].time * 1000.0 + beams[idx].bm;
}

/**
 * Noise level estimation kernel
 * Estimates noise levels across all beams for quality control
 */
__global__ void cuda_scan_noise_estimation_kernel(cuda_radar_beam_t *beams,
                                                  int num_beams,
                                                  float *noise_estimates) {
    int beam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (beam_idx >= num_beams) return;
    
    cuda_radar_beam_t *beam = &beams[beam_idx];
    
    // Estimate noise from far-range gates (typically quieter)
    float noise_sum = 0.0f;
    int noise_count = 0;
    int far_range_start = beam->nrang * 3 / 4;  // Start from 75% of max range
    
    for (int r = far_range_start; r < beam->nrang; r++) {
        cuda_radar_cell_t *cell = &beam->rng[r];
        if (cell->p_0 > 0.0f && cell->p_0 < beam->noise * 3.0f) {  // Low power gates
            noise_sum += cell->p_0;
            noise_count++;
        }
    }
    
    noise_estimates[beam_idx] = (noise_count > 0) ? noise_sum / noise_count : beam->noise;
}

/* Host wrapper functions */

/**
 * CUDA-accelerated beam processing
 */
extern "C" cudaError_t cuda_scan_process_beams(cuda_radar_beam_t *beams,
                                               int num_beams,
                                               int max_range_gates) {
    if (!beams || num_beams <= 0) return cudaErrorInvalidValue;
    
    dim3 grid(num_beams);
    dim3 block(max_range_gates);
    
    cuda_scan_process_beams_kernel<<<grid, block>>>(beams, num_beams, max_range_gates);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated beam filtering
 */
extern "C" cudaError_t cuda_scan_filter_beams(cuda_radar_beam_t *beams,
                                              int num_beams,
                                              int target_beam,
                                              bool *valid_mask,
                                              int *new_count) {
    if (!beams || !valid_mask) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_beams + threads_per_block - 1) / threads_per_block;
    
    cuda_scan_filter_beams_kernel<<<blocks, threads_per_block>>>(
        beams, num_beams, target_beam, valid_mask);
    
    // Count valid beams using Thrust
    thrust::device_ptr<bool> valid_ptr(valid_mask);
    int count = thrust::count(valid_ptr, valid_ptr + num_beams, true);
    
    if (new_count) {
        cudaMemcpy(new_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated scan validation
 */
extern "C" cudaError_t cuda_scan_validate(cuda_radar_beam_t *beams,
                                          int num_beams,
                                          bool *valid_mask) {
    if (!beams || !valid_mask) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_beams + threads_per_block - 1) / threads_per_block;
    
    cuda_scan_validate_kernel<<<blocks, threads_per_block>>>(
        beams, num_beams, valid_mask);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated range gate statistics
 */
extern "C" cudaError_t cuda_scan_range_statistics(cuda_radar_beam_t *beams,
                                                  int num_beams,
                                                  float *power_mean,
                                                  float *velocity_mean,
                                                  float *width_mean,
                                                  int *valid_count) {
    if (!beams || !power_mean || !velocity_mean || !width_mean) {
        return cudaErrorInvalidValue;
    }
    
    int max_range_gates = 0;
    for (int i = 0; i < num_beams; i++) {
        if (beams[i].nrang > max_range_gates) {
            max_range_gates = beams[i].nrang;
        }
    }
    
    dim3 grid(num_beams);
    dim3 block(max_range_gates);
    int shared_mem_size = 3 * max_range_gates * sizeof(float) + max_range_gates * sizeof(int);
    
    cuda_scan_range_statistics_kernel<<<grid, block, shared_mem_size>>>(
        beams, num_beams, power_mean, velocity_mean, width_mean, valid_count);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated scatter classification
 */
extern "C" cudaError_t cuda_scan_scatter_classification(cuda_radar_beam_t *beams,
                                                       int num_beams,
                                                       int max_range_gates) {
    if (!beams || num_beams <= 0) return cudaErrorInvalidValue;
    
    dim3 grid(num_beams);
    dim3 block(max_range_gates);
    
    cuda_scan_scatter_classification_kernel<<<grid, block>>>(
        beams, num_beams, max_range_gates);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}