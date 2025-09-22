/* iq.1.7_cuda.cu
   ===============
   CUDA-accelerated IQ data processing implementation
   Author: R.J.Barnes (CUDA implementation)
*/

/*
  Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory
 
This file is part of the Radar Software Toolkit (RST).

RST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Modifications:
*/ 

#include "iq.1.7_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/* Global CUDA context */
static cublasHandle_t cublas_handle = NULL;
static bool cuda_initialized = false;
static bool profiling_enabled = false;
static cudaEvent_t start_event, stop_event;

/* CUDA Initialization */
__host__ cudaError_t iq_1_7_cuda_init(void) {
    if (cuda_initialized) return cudaSuccess;
    
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorInitializationError;
    }
    
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cuda_initialized = true;
    return cudaSuccess;
}

/* CUDA cleanup */
__host__ void iq_1_7_cuda_cleanup(void) {
    if (cuda_initialized) {
        cublasDestroy(cublas_handle);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_initialized = false;
    }
}

/* CUDA-compatible IQ data structures */
typedef struct {
    int major, minor;       // Revision numbers
} cuda_iq_revision_t;

typedef struct {
    cuda_iq_revision_t revision;
    int chnnum;            // Number of channels
    int smpnum;            // Number of samples per sequence
    int skpnum;            // Number of samples to skip
    int seqnum;            // Number of sequences
    int tbadtr;            // Total bad transmit samples
    long *tv_sec;          // Time seconds array
    long *tv_nsec;         // Time nanoseconds array
    int *atten;            // Attenuation array
    float *noise;          // Noise array
    int *offset;           // Offset array
    int *size;             // Size array
    int *badtr;            // Bad transmit array
} cuda_iq_data_t;

/* CUDA Kernels for IQ Processing */

/**
 * Time conversion kernel
 * Converts between different time formats in parallel
 */
__global__ void cuda_iq_time_convert_kernel(const double *input_time,
                                            long *tv_sec,
                                            long *tv_nsec,
                                            int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_samples) return;
    
    double time_val = input_time[idx];
    tv_sec[idx] = (long)time_val;
    tv_nsec[idx] = (long)((time_val - (double)tv_sec[idx]) * 1e9);
}

/**
 * Array copy kernel with validation
 * Performs parallel array copying with bounds checking
 */
__global__ void cuda_iq_array_copy_kernel(const int *input,
                                          int *output,
                                          bool *valid_mask,
                                          int num_elements,
                                          int min_val, int max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_elements) return;
    
    int val = input[idx];
    bool is_valid = (val >= min_val && val <= max_val);
    
    output[idx] = val;
    if (valid_mask) {
        valid_mask[idx] = is_valid;
    }
}

/**
 * Float array copy kernel with validation
 * Performs parallel float array copying with NaN/infinity checking
 */
__global__ void cuda_iq_float_copy_kernel(const float *input,
                                          float *output,
                                          bool *valid_mask,
                                          int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_elements) return;
    
    float val = input[idx];
    bool is_valid = isfinite(val);
    
    output[idx] = val;
    if (valid_mask) {
        valid_mask[idx] = is_valid;
    }
}

/**
 * IQ data flattening kernel
 * Efficiently packs IQ data into contiguous memory layout
 */
__global__ void cuda_iq_flatten_kernel(const cuda_iq_data_t *iq_data,
                                       char *buffer,
                                       int *offsets,
                                       int buffer_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= iq_data->seqnum) return;
    
    // Calculate memory offsets for each sequence
    int base_offset = offsets[idx];
    
    // Copy sequence data to buffer (simplified version)
    if (base_offset + sizeof(int) * 6 < buffer_size) {
        int *int_ptr = (int*)(buffer + base_offset);
        int_ptr[0] = iq_data->chnnum;
        int_ptr[1] = iq_data->smpnum;
        int_ptr[2] = iq_data->skpnum;
        int_ptr[3] = iq_data->atten[idx];
        int_ptr[4] = iq_data->offset[idx];
        int_ptr[5] = iq_data->size[idx];
        
        // Copy noise data
        float *float_ptr = (float*)(buffer + base_offset + 6 * sizeof(int));
        float_ptr[0] = iq_data->noise[idx];
    }
}

/**
 * IQ data expansion kernel
 * Unpacks flattened IQ data back into structured format
 */
__global__ void cuda_iq_expand_kernel(const char *buffer,
                                      cuda_iq_data_t *iq_data,
                                      const int *offsets,
                                      int num_sequences) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_sequences) return;
    
    int base_offset = offsets[idx];
    
    // Read sequence data from buffer
    const int *int_ptr = (const int*)(buffer + base_offset);
    iq_data->atten[idx] = int_ptr[3];
    iq_data->offset[idx] = int_ptr[4];
    iq_data->size[idx] = int_ptr[5];
    
    // Read noise data
    const float *float_ptr = (const float*)(buffer + base_offset + 6 * sizeof(int));
    iq_data->noise[idx] = float_ptr[0];
}

/**
 * IQ data encoding kernel
 * Converts IQ data to DataMap format in parallel
 */
__global__ void cuda_iq_encode_kernel(const cuda_iq_data_t *iq_data,
                                      float *encoded_data,
                                      int *metadata,
                                      int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_samples) return;
    
    // Determine which sequence this sample belongs to
    int seq_idx = 0;
    int cumulative_size = 0;
    
    for (int i = 0; i < iq_data->seqnum; i++) {
        if (idx < cumulative_size + iq_data->size[i]) {
            seq_idx = i;
            break;
        }
        cumulative_size += iq_data->size[i];
    }
    
    // Apply attenuation and noise correction
    int local_idx = idx - cumulative_size;
    float atten_factor = (float)iq_data->atten[seq_idx] / 1000.0f;  // Convert to dB
    float noise_level = iq_data->noise[seq_idx];
    
    // Store encoded data with corrections applied
    encoded_data[idx] = encoded_data[idx] * atten_factor - noise_level;
    
    // Store metadata
    if (local_idx == 0) {  // First sample of sequence
        metadata[seq_idx * 4 + 0] = iq_data->atten[seq_idx];
        metadata[seq_idx * 4 + 1] = iq_data->offset[seq_idx];
        metadata[seq_idx * 4 + 2] = iq_data->size[seq_idx];
        metadata[seq_idx * 4 + 3] = iq_data->badtr[seq_idx];
    }
}

/**
 * IQ data decoding kernel
 * Converts DataMap format to IQ data in parallel
 */
__global__ void cuda_iq_decode_kernel(const float *encoded_data,
                                      const int *metadata,
                                      cuda_iq_data_t *iq_data,
                                      int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_samples) return;
    
    // Determine sequence for this sample
    int seq_idx = 0;
    int cumulative_size = 0;
    
    for (int i = 0; i < iq_data->seqnum; i++) {
        if (idx < cumulative_size + iq_data->size[i]) {
            seq_idx = i;
            break;
        }
        cumulative_size += iq_data->size[i];
    }
    
    // Decode metadata (first thread of each sequence)
    int local_idx = idx - cumulative_size;
    if (local_idx == 0) {
        iq_data->atten[seq_idx] = metadata[seq_idx * 4 + 0];
        iq_data->offset[seq_idx] = metadata[seq_idx * 4 + 1];
        iq_data->size[seq_idx] = metadata[seq_idx * 4 + 2];
        iq_data->badtr[seq_idx] = metadata[seq_idx * 4 + 3];
    }
}

/**
 * Bad transmit sample detection kernel
 * Identifies corrupted or invalid IQ samples
 */
__global__ void cuda_iq_badtr_detect_kernel(const int16_t *iq_samples,
                                            bool *badtr_mask,
                                            int num_samples,
                                            int16_t threshold_min,
                                            int16_t threshold_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_samples) return;
    
    // Check I and Q components (interleaved format)
    int i_sample = iq_samples[2 * idx];
    int q_sample = iq_samples[2 * idx + 1];
    
    // Detect saturated or corrupted samples
    bool is_bad = false;
    
    // Check for saturation
    if (i_sample <= threshold_min || i_sample >= threshold_max ||
        q_sample <= threshold_min || q_sample >= threshold_max) {
        is_bad = true;
    }
    
    // Check for stuck bits (consecutive identical values)
    if (idx > 0) {
        int prev_i = iq_samples[2 * (idx - 1)];
        int prev_q = iq_samples[2 * (idx - 1) + 1];
        
        if (i_sample == prev_i && q_sample == prev_q) {
            is_bad = true;
        }
    }
    
    badtr_mask[idx] = is_bad;
}

/**
 * Statistical analysis kernel for IQ data
 * Computes mean, variance, and other statistics in parallel
 */
__global__ void cuda_iq_statistics_kernel(const int16_t *iq_samples,
                                          float *statistics,
                                          int num_samples,
                                          int stats_type) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    if (idx < num_samples) {
        float i_val = (float)iq_samples[2 * idx];
        float q_val = (float)iq_samples[2 * idx + 1];
        float power = i_val * i_val + q_val * q_val;
        
        switch (stats_type) {
            case 0: // Mean power
                local_sum = power;
                break;
            case 1: // Variance calculation
                local_sum = power;
                local_sum_sq = power * power;
                break;
            case 2: // I channel mean
                local_sum = i_val;
                break;
            case 3: // Q channel mean
                local_sum = q_val;
                break;
        }
    }
    
    // Store in shared memory
    sdata[tid] = local_sum;
    if (stats_type == 1) {
        sdata[tid + blockDim.x] = local_sum_sq;
    }
    
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            if (stats_type == 1) {
                sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
            }
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        atomicAdd(&statistics[0], sdata[0]);
        if (stats_type == 1) {
            atomicAdd(&statistics[1], sdata[blockDim.x]);
        }
    }
}

/* Host wrapper functions */

/**
 * CUDA-accelerated time conversion
 */
extern "C" cudaError_t cuda_iq_convert_time(const double *input_time,
                                            long *tv_sec,
                                            long *tv_nsec,
                                            int num_samples) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!input_time || !tv_sec || !tv_nsec) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_samples + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_time_convert_kernel<<<blocks, threads_per_block>>>(
        input_time, tv_sec, tv_nsec, num_samples);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated array copying with validation
 */
extern "C" cudaError_t cuda_iq_copy_array(const int *input,
                                          int *output,
                                          bool *valid_mask,
                                          int num_elements,
                                          int min_val, int max_val) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!input || !output) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_array_copy_kernel<<<blocks, threads_per_block>>>(
        input, output, valid_mask, num_elements, min_val, max_val);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated float array copying
 */
extern "C" cudaError_t cuda_iq_copy_float_array(const float *input,
                                                float *output,
                                                bool *valid_mask,
                                                int num_elements) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!input || !output) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_float_copy_kernel<<<blocks, threads_per_block>>>(
        input, output, valid_mask, num_elements);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated IQ data flattening
 */
extern "C" cudaError_t cuda_iq_flatten(const cuda_iq_data_t *iq_data,
                                       char *buffer,
                                       int *offsets,
                                       int buffer_size) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!iq_data || !buffer || !offsets) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (iq_data->seqnum + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_flatten_kernel<<<blocks, threads_per_block>>>(
        iq_data, buffer, offsets, buffer_size);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated IQ data expansion
 */
extern "C" cudaError_t cuda_iq_expand(const char *buffer,
                                      cuda_iq_data_t *iq_data,
                                      const int *offsets,
                                      int num_sequences) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!buffer || !iq_data || !offsets) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_sequences + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_expand_kernel<<<blocks, threads_per_block>>>(
        buffer, iq_data, offsets, num_sequences);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated IQ data encoding
 */
extern "C" cudaError_t cuda_iq_encode(const cuda_iq_data_t *iq_data,
                                      float *encoded_data,
                                      int *metadata,
                                      int total_samples) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!iq_data || !encoded_data || !metadata) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (total_samples + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_encode_kernel<<<blocks, threads_per_block>>>(
        iq_data, encoded_data, metadata, total_samples);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated IQ data decoding
 */
extern "C" cudaError_t cuda_iq_decode(const float *encoded_data,
                                      const int *metadata,
                                      cuda_iq_data_t *iq_data,
                                      int total_samples) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!encoded_data || !metadata || !iq_data) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (total_samples + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_decode_kernel<<<blocks, threads_per_block>>>(
        encoded_data, metadata, iq_data, total_samples);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated bad transmit detection
 */
extern "C" cudaError_t cuda_iq_detect_badtr(const int16_t *iq_samples,
                                            bool *badtr_mask,
                                            int num_samples,
                                            int16_t threshold_min,
                                            int16_t threshold_max) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!iq_samples || !badtr_mask) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_samples + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_badtr_detect_kernel<<<blocks, threads_per_block>>>(
        iq_samples, badtr_mask, num_samples, threshold_min, threshold_max);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated IQ statistics calculation
 */
extern "C" cudaError_t cuda_iq_calculate_statistics(const int16_t *iq_samples,
                                                    float *statistics,
                                                    int num_samples,
                                                    int stats_type) {
    if (!cuda_initialized) {
        cudaError_t init_error = iq_1_7_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!iq_samples || !statistics) return cudaErrorInvalidValue;
    
    // Initialize statistics array
    cudaMemset(statistics, 0, 2 * sizeof(float));
    
    int threads_per_block = 256;
    int blocks = (num_samples + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_iq_statistics_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        iq_samples, statistics, num_samples, stats_type);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/* Utility Functions */
extern "C" bool iq_1_7_cuda_is_available(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

extern "C" int iq_1_7_cuda_get_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

extern "C" void iq_1_7_cuda_enable_profiling(bool enable) {
    profiling_enabled = enable;
}

extern "C" float iq_1_7_cuda_get_last_kernel_time(void) {
    if (!profiling_enabled) return 0.0f;
    
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    return milliseconds;
}