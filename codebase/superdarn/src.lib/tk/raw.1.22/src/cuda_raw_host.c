/**
 * @file cuda_raw_host.c
 * @brief Host-side implementations for raw.1.22 CUDA functions
 * 
 * Provides CPU/GPU bridge and memory management for raw data processing
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#include "cuda_raw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Global state */
static bool cuda_raw_initialized = false;
static bool profiling_enabled = false;
static cudaEvent_t start_event, stop_event;
static cuda_raw_profile_t current_profile = {0};

/* Forward declarations for CUDA kernels */
extern cudaError_t cuda_raw_interleave_complex(const float *real_data,
                                               const float *imag_data,
                                               float *output,
                                               int nrang, int mplgs);

extern cudaError_t cuda_raw_threshold_filter(const float *pwr0,
                                             int *slist, int *snum,
                                             float threshold, int nrang);

extern cudaError_t cuda_raw_sparse_gather(const float *input,
                                          const int *slist,
                                          float *output,
                                          int snum, int mplgs);

extern cudaError_t cuda_raw_data_reorganize(const float *acfd_real,
                                            const float *acfd_imag,
                                            const float *xcfd_real,
                                            const float *xcfd_imag,
                                            const int *slist,
                                            float *encoded_acfd,
                                            float *encoded_xcfd,
                                            int snum, int mplgs);

extern cudaError_t cuda_raw_time_search(const double *time_array,
                                        const double *search_times,
                                        int *result_indices,
                                        int num_records,
                                        int num_searches);

/* Initialization and cleanup */
cudaError_t cuda_raw_init(void) {
    if (cuda_raw_initialized) return cudaSuccess;
    
    cudaError_t err = cudaEventCreate(&start_event);
    if (err != cudaSuccess) return err;
    
    err = cudaEventCreate(&stop_event);
    if (err != cudaSuccess) {
        cudaEventDestroy(start_event);
        return err;
    }
    
    cuda_raw_initialized = true;
    memset(&current_profile, 0, sizeof(current_profile));
    
    return cudaSuccess;
}

void cuda_raw_cleanup(void) {
    if (!cuda_raw_initialized) return;
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cuda_raw_initialized = false;
}

/* Utility functions */
bool cuda_raw_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

int cuda_raw_get_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

/* Memory management */
cuda_raw_data_t* cuda_raw_data_alloc(int nrang, int mplgs) {
    cuda_raw_data_t *raw_data = (cuda_raw_data_t*)malloc(sizeof(cuda_raw_data_t));
    if (!raw_data) return NULL;
    
    raw_data->nrang = nrang;
    raw_data->mplgs = mplgs;
    raw_data->threshold = 0.0f;
    
    size_t pwr_size = nrang * sizeof(float);
    size_t acf_size = nrang * mplgs * sizeof(float);
    
    // Allocate unified memory for seamless CPU/GPU access
    cudaError_t err = cudaMallocManaged(&raw_data->pwr0, pwr_size);
    if (err != cudaSuccess) {
        free(raw_data);
        return NULL;
    }
    
    err = cudaMallocManaged(&raw_data->acfd_real, acf_size);
    if (err != cudaSuccess) {
        cudaFree(raw_data->pwr0);
        free(raw_data);
        return NULL;
    }
    
    err = cudaMallocManaged(&raw_data->acfd_imag, acf_size);
    if (err != cudaSuccess) {
        cudaFree(raw_data->pwr0);
        cudaFree(raw_data->acfd_real);
        free(raw_data);
        return NULL;
    }
    
    err = cudaMallocManaged(&raw_data->xcfd_real, acf_size);
    if (err != cudaSuccess) {
        cudaFree(raw_data->pwr0);
        cudaFree(raw_data->acfd_real);
        cudaFree(raw_data->acfd_imag);
        free(raw_data);
        return NULL;
    }
    
    err = cudaMallocManaged(&raw_data->xcfd_imag, acf_size);
    if (err != cudaSuccess) {
        cudaFree(raw_data->pwr0);
        cudaFree(raw_data->acfd_real);
        cudaFree(raw_data->acfd_imag);
        cudaFree(raw_data->xcfd_real);
        free(raw_data);
        return NULL;
    }
    
    // Initialize to zero
    memset(raw_data->pwr0, 0, pwr_size);
    memset(raw_data->acfd_real, 0, acf_size);
    memset(raw_data->acfd_imag, 0, acf_size);
    memset(raw_data->xcfd_real, 0, acf_size);
    memset(raw_data->xcfd_imag, 0, acf_size);
    
    return raw_data;
}

void cuda_raw_data_free(cuda_raw_data_t *raw_data) {
    if (!raw_data) return;
    
    cudaFree(raw_data->pwr0);
    cudaFree(raw_data->acfd_real);
    cudaFree(raw_data->acfd_imag);
    cudaFree(raw_data->xcfd_real);
    cudaFree(raw_data->xcfd_imag);
    free(raw_data);
}

cuda_raw_index_t* cuda_raw_index_alloc(int num_records) {
    cuda_raw_index_t *raw_index = (cuda_raw_index_t*)malloc(sizeof(cuda_raw_index_t));
    if (!raw_index) return NULL;
    
    raw_index->num_records = num_records;
    
    size_t time_size = num_records * sizeof(double);
    size_t index_size = num_records * sizeof(int);
    
    cudaError_t err = cudaMallocManaged(&raw_index->tme, time_size);
    if (err != cudaSuccess) {
        free(raw_index);
        return NULL;
    }
    
    err = cudaMallocManaged(&raw_index->inx, index_size);
    if (err != cudaSuccess) {
        cudaFree(raw_index->tme);
        free(raw_index);
        return NULL;
    }
    
    return raw_index;
}

void cuda_raw_index_free(cuda_raw_index_t *raw_index) {
    if (!raw_index) return;
    
    cudaFree(raw_index->tme);
    cudaFree(raw_index->inx);
    free(raw_index);
}

/* Performance profiling */
cudaError_t cuda_raw_enable_profiling(bool enable) {
    if (!cuda_raw_initialized) {
        cudaError_t err = cuda_raw_init();
        if (err != cudaSuccess) return err;
    }
    
    profiling_enabled = enable;
    if (enable) {
        cuda_raw_reset_profile();
    }
    return cudaSuccess;
}

cudaError_t cuda_raw_get_profile(cuda_raw_profile_t *profile) {
    if (!profile) return cudaErrorInvalidValue;
    *profile = current_profile;
    return cudaSuccess;
}

void cuda_raw_reset_profile(void) {
    memset(&current_profile, 0, sizeof(current_profile));
}

/* High-level wrapper functions */
cudaError_t cuda_raw_encode(cuda_raw_data_t *raw_data,
                           const int *slist, int snum,
                           float *encoded_data) {
    if (!cuda_raw_initialized) {
        cudaError_t err = cuda_raw_init();
        if (err != cudaSuccess) return err;
    }
    
    if (!raw_data || !slist || !encoded_data) return cudaErrorInvalidValue;
    
    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    // Allocate temporary device memory for encoded data
    size_t encoded_size = 2 * snum * raw_data->mplgs * sizeof(float);
    float *d_encoded_acfd, *d_encoded_xcfd;
    
    cudaError_t err = cudaMalloc(&d_encoded_acfd, encoded_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_encoded_xcfd, encoded_size);
    if (err != cudaSuccess) {
        cudaFree(d_encoded_acfd);
        return err;
    }
    
    // Perform CUDA-accelerated data reorganization
    err = cuda_raw_data_reorganize(raw_data->acfd_real, raw_data->acfd_imag,
                                  raw_data->xcfd_real, raw_data->xcfd_imag,
                                  slist, d_encoded_acfd, d_encoded_xcfd,
                                  snum, raw_data->mplgs);
    
    if (err == cudaSuccess) {
        // Copy results back to host
        err = cudaMemcpy(encoded_data, d_encoded_acfd, encoded_size, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
            err = cudaMemcpy(encoded_data + encoded_size/sizeof(float), 
                           d_encoded_xcfd, encoded_size, cudaMemcpyDeviceToHost);
        }
    }
    
    cudaFree(d_encoded_acfd);
    cudaFree(d_encoded_xcfd);
    
    if (profiling_enabled) {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start_event, stop_event);
        current_profile.gpu_time_ms += gpu_time;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
    double total_time = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 + 
                       (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000000.0;
    current_profile.cpu_time_ms += total_time;
    
    if (current_profile.cpu_time_ms > 0 && current_profile.gpu_time_ms > 0) {
        current_profile.speedup_factor = current_profile.cpu_time_ms / current_profile.gpu_time_ms;
    }
    
    return err;
}

cudaError_t cuda_raw_decode(const float *encoded_data,
                           cuda_raw_data_t *raw_data,
                           const int *slist, int snum) {
    if (!cuda_raw_initialized) {
        cudaError_t err = cuda_raw_init();
        if (err != cudaSuccess) return err;
    }
    
    if (!encoded_data || !raw_data || !slist) return cudaErrorInvalidValue;
    
    // This would implement the reverse of cuda_raw_encode
    // For now, return success as placeholder
    return cudaSuccess;
}

cudaError_t cuda_raw_seek(cuda_raw_index_t *index,
                         double target_time,
                         int *result_position) {
    if (!cuda_raw_initialized) {
        cudaError_t err = cuda_raw_init();
        if (err != cudaSuccess) return err;
    }
    
    if (!index || !result_position) return cudaErrorInvalidValue;
    
    // Use CUDA-accelerated time search
    double *d_target_time;
    int *d_result;
    
    cudaError_t err = cudaMalloc(&d_target_time, sizeof(double));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_result, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_target_time);
        return err;
    }
    
    err = cudaMemcpy(d_target_time, &target_time, sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_target_time);
        cudaFree(d_result);
        return err;
    }
    
    err = cuda_raw_time_search(index->tme, d_target_time, d_result, index->num_records, 1);
    
    if (err == cudaSuccess) {
        err = cudaMemcpy(result_position, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_target_time);
    cudaFree(d_result);
    
    return err;
}

/* Statistics calculation wrapper */
cudaError_t cuda_raw_calculate_statistics(const float *data,
                                          cuda_raw_statistics_t *stats,
                                          int num_elements) {
    if (!cuda_raw_initialized) {
        cudaError_t err = cuda_raw_init();
        if (err != cudaSuccess) return err;
    }
    
    if (!data || !stats) return cudaErrorInvalidValue;
    
    // Allocate device memory for results
    float *d_min, *d_max, *d_sum;
    cudaError_t err = cudaMalloc(&d_min, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_max, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_min);
        return err;
    }
    
    err = cudaMalloc(&d_sum, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_min);
        cudaFree(d_max);
        return err;
    }
    
    // Initialize values
    float init_min = FLT_MAX, init_max = -FLT_MAX, init_sum = 0.0f;
    cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice);
    
    // For now, implement simple host-side calculation
    // In full implementation, would call CUDA statistics kernel
    float min_val = FLT_MAX, max_val = -FLT_MAX, sum_val = 0.0f;
    
    for (int i = 0; i < num_elements; i++) {
        float val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
    }
    
    stats->min_val = min_val;
    stats->max_val = max_val;
    stats->sum_val = sum_val;
    stats->mean_val = (num_elements > 0) ? sum_val / num_elements : 0.0f;
    stats->count = num_elements;
    
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
    
    return cudaSuccess;
}