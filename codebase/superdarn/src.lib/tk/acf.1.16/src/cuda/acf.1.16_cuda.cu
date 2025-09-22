#include "acf.1.16_cuda.h"
#include <stdio.h>
#include <stdlib.h>

/* Global CUDA context */
static cublasHandle_t cublas_handle = NULL;
static bool cuda_initialized = false;
static bool profiling_enabled = false;
static cudaEvent_t start_event, stop_event;

/* CUDA Initialization */
__host__ cudaError_t acf_1_16_cuda_init(void) {
    if (cuda_initialized) return cudaSuccess;
    
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorInitializationError;
    }
    
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cuda_initialized = true;
    return cudaSuccess;
}

/* Memory Management Implementation */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type) {
    cuda_array_t *array = (cuda_array_t*)malloc(sizeof(cuda_array_t));
    if (!array) return NULL;
    
    array->size = size;
    array->element_size = element_size;
    array->type = type;
    array->device_id = 0;
    array->is_on_device = false;
    
    cudaError_t error = cudaMallocManaged(&array->data, size * element_size);
    if (error != cudaSuccess) {
        free(array);
        return NULL;
    }
    
    return array;
}

void cuda_array_destroy(cuda_array_t *array) {
    if (!array) return;
    if (array->data) cudaFree(array->data);
    free(array);
}

/* CUDA-compatible ACF data structures */
typedef struct {
    int16_t *inbuf;     // Input I/Q samples
    float *acfbuf;      // ACF output buffer (complex, interleaved)
    float *xcfbuf;      // XCF output buffer (complex, interleaved)
    float *pwr0;        // Lag-0 power array
    int nrang;          // Number of range gates
    int mplgs;          // Number of lags
    int nave;           // Number of averages
    int *lagfr;         // Lag to first range table
    int *smsep;         // Sample separation
    int *pat;           // Pulse pattern
    int mpinc;          // Multi-pulse increment
    float attn;         // Attenuation factor
    bool xcf_enabled;   // Cross-correlation enabled
} cuda_acf_data_t;

typedef struct {
    float real, imag;   // Complex number
} cuda_complex_t;

/* CUDA Kernels for ACF Processing */

/**
 * Core ACF correlation kernel
 * Computes auto-correlation functions from I/Q samples
 */
__global__ void cuda_acf_calculate_kernel(const int16_t *inbuf,
                                          float *acfbuf,
                                          float *xcfbuf,
                                          const int *lagfr,
                                          const int *smsep,
                                          const int *pat,
                                          int nrang, int mplgs,
                                          int mpinc, int nave,
                                          int offset, bool xcf_enabled) {
    int range = blockIdx.x;
    int lag = threadIdx.x;
    
    if (range >= nrang || lag >= mplgs) return;
    
    // Shared memory for frequently accessed data
    extern __shared__ int s_pat[];
    
    if (threadIdx.x < mpinc) {
        s_pat[threadIdx.x] = pat[threadIdx.x];
    }
    __syncthreads();
    
    // Calculate correlation for this (range, lag) pair
    float real_sum = 0.0f, imag_sum = 0.0f;
    float xcf_real_sum = 0.0f, xcf_imag_sum = 0.0f;
    
    // Pulse sequence processing
    for (int pulse = 0; pulse < nave; pulse++) {
        int tau = lagfr[lag];
        int sample1 = offset + pulse * smsep[0] + range * smsep[0];
        int sample2 = sample1 + tau * smsep[0];
        
        // Check bounds
        if (sample2 + 1 < offset + (nave + 1) * smsep[0]) {
            // Get I/Q samples
            float i1 = (float)inbuf[sample1 * 2];     // I component, pulse 1
            float q1 = (float)inbuf[sample1 * 2 + 1]; // Q component, pulse 1
            float i2 = (float)inbuf[sample2 * 2];     // I component, pulse 2
            float q2 = (float)inbuf[sample2 * 2 + 1]; // Q component, pulse 2
            
            // Complex correlation: (a+bi)*(c-di) = (ac+bd) + (bc-ad)i
            real_sum += i1 * i2 + q1 * q2;
            imag_sum += q1 * i2 - i1 * q2;
            
            // Cross-correlation (if enabled)
            if (xcf_enabled && sample1 + 1 < offset + (nave + 1) * smsep[0]) {
                float i1_next = (float)inbuf[(sample1 + 1) * 2];
                float q1_next = (float)inbuf[(sample1 + 1) * 2 + 1];
                
                xcf_real_sum += i1_next * i2 + q1_next * q2;
                xcf_imag_sum += q1_next * i2 - i1_next * q2;
            }
        }
    }
    
    // Store results in interleaved format
    int output_idx = range * (2 * mplgs) + 2 * lag;
    acfbuf[output_idx] = real_sum;
    acfbuf[output_idx + 1] = imag_sum;
    
    if (xcf_enabled && xcfbuf) {
        xcfbuf[output_idx] = xcf_real_sum;
        xcfbuf[output_idx + 1] = xcf_imag_sum;
    }
}

/**
 * Power calculation kernel (lag-0 power)
 * Computes |ACF[0]|² for each range gate
 */
__global__ void cuda_acf_power_kernel(const float *acfbuf,
                                      float *pwr0,
                                      int nrang, int mplgs) {
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range >= nrang) return;
    
    // Get lag-0 (autocorrelation at zero lag)
    int lag0_idx = range * (2 * mplgs);  // Real component at lag 0
    float real = acfbuf[lag0_idx];
    float imag = acfbuf[lag0_idx + 1];
    
    // Power = |z|² = real² + imag²
    pwr0[range] = real * real + imag * imag;
}

/**
 * ACF averaging kernel
 * Averages ACF data across multiple integrations
 */
__global__ void cuda_acf_average_kernel(float *acfbuf,
                                        float *xcfbuf,
                                        int nrang, int mplgs,
                                        int nave, bool xcf_enabled) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = nrang * mplgs * 2;  // Complex data (real + imag)
    
    if (idx >= total_elements) return;
    
    if (nave > 0) {
        acfbuf[idx] /= (float)nave;
        
        if (xcf_enabled && xcfbuf) {
            xcfbuf[idx] /= (float)nave;
        }
    }
}

/**
 * ACF normalization kernel
 * Normalizes ACF data by attenuation factor
 */
__global__ void cuda_acf_normalize_kernel(float *acfbuf,
                                          float *xcfbuf,
                                          int nrang, int mplgs,
                                          float attn, bool xcf_enabled) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = nrang * mplgs * 2;  // Complex data
    
    if (idx >= total_elements) return;
    
    if (attn != 0.0f) {
        acfbuf[idx] /= attn;
        
        if (xcf_enabled && xcfbuf) {
            xcfbuf[idx] /= attn;
        }
    }
}

/**
 * Bad lag detection kernel
 * Identifies and marks corrupted lag values
 */
__global__ void cuda_acf_badlag_kernel(const float *acfbuf,
                                       bool *badlag_mask,
                                       int nrang, int mplgs,
                                       float noise_threshold) {
    int range = blockIdx.x;
    int lag = threadIdx.x;
    
    if (range >= nrang || lag >= mplgs) return;
    
    int idx = range * (2 * mplgs) + 2 * lag;
    float real = acfbuf[idx];
    float imag = acfbuf[idx + 1];
    float power = real * real + imag * imag;
    
    // Simple bad lag detection based on power anomalies
    bool is_bad = false;
    
    // Check for NaN or infinite values
    if (!isfinite(real) || !isfinite(imag)) {
        is_bad = true;
    }
    
    // Check for excessive power (interference)
    if (power > noise_threshold * 1000.0f) {
        is_bad = true;
    }
    
    // Check for zero power (missing data)
    if (power < noise_threshold * 0.001f) {
        is_bad = true;
    }
    
    badlag_mask[range * mplgs + lag] = is_bad;
}

/**
 * Sum power calculation kernel
 * Computes power statistics across ranges
 */
__global__ void cuda_acf_sum_power_kernel(const int16_t *inbuf,
                                          float *power_sum,
                                          float *power_max,
                                          int nrang, int nave,
                                          int offset, int smsep) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int range = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    float local_max = 0.0f;
    
    if (range < nrang) {
        // Calculate power for this range across all samples
        for (int sample = 0; sample < nave; sample++) {
            int idx = offset + sample * smsep + range * 2;
            float i_val = (float)inbuf[idx];
            float q_val = (float)inbuf[idx + 1];
            float power = i_val * i_val + q_val * q_val;
            
            local_sum += power;
            local_max = fmaxf(local_max, power);
        }
    }
    
    // Store in shared memory
    sdata[tid] = local_sum;
    sdata[tid + blockDim.x] = local_max;
    
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];                    // Sum
            sdata[tid + blockDim.x] = fmaxf(sdata[tid + blockDim.x], 
                                           sdata[tid + s + blockDim.x]); // Max
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        atomicAdd(power_sum, sdata[0]);
        atomicMaxFloat(power_max, sdata[blockDim.x]);
    }
}

/**
 * Complex magnitude calculation kernel
 * Computes |ACF[lag]| for each range and lag
 */
__global__ void cuda_acf_magnitude_kernel(const float *acfbuf,
                                          float *magnitude,
                                          int nrang, int mplgs) {
    int range = blockIdx.x;
    int lag = threadIdx.x;
    
    if (range >= nrang || lag >= mplgs) return;
    
    int idx = range * (2 * mplgs) + 2 * lag;
    float real = acfbuf[idx];
    float imag = acfbuf[idx + 1];
    
    magnitude[range * mplgs + lag] = sqrtf(real * real + imag * imag);
}

/* Atomic operations for float */
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/* Host wrapper functions */

/**
 * CUDA-accelerated ACF calculation
 */
extern "C" cudaError_t cuda_acf_calculate(const int16_t *inbuf,
                                          float *acfbuf,
                                          float *xcfbuf,
                                          const int *lagfr,
                                          const int *smsep,
                                          const int *pat,
                                          int nrang, int mplgs,
                                          int mpinc, int nave,
                                          int offset, bool xcf_enabled) {
    if (!cuda_initialized) {
        cudaError_t init_error = acf_1_16_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!inbuf || !acfbuf || !lagfr || !smsep || !pat) return cudaErrorInvalidValue;
    
    dim3 grid(nrang);
    dim3 block(mplgs);
    int shared_mem_size = mpinc * sizeof(int);
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_acf_calculate_kernel<<<grid, block, shared_mem_size>>>(
        inbuf, acfbuf, xcfbuf, lagfr, smsep, pat,
        nrang, mplgs, mpinc, nave, offset, xcf_enabled);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated power calculation
 */
extern "C" cudaError_t cuda_acf_power(const float *acfbuf,
                                      float *pwr0,
                                      int nrang, int mplgs) {
    if (!cuda_initialized) {
        cudaError_t init_error = acf_1_16_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!acfbuf || !pwr0) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (nrang + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_acf_power_kernel<<<blocks, threads_per_block>>>(
        acfbuf, pwr0, nrang, mplgs);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated ACF averaging
 */
extern "C" cudaError_t cuda_acf_average(float *acfbuf,
                                        float *xcfbuf,
                                        int nrang, int mplgs,
                                        int nave, bool xcf_enabled) {
    if (!cuda_initialized) {
        cudaError_t init_error = acf_1_16_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!acfbuf) return cudaErrorInvalidValue;
    
    int total_elements = nrang * mplgs * 2;  // Complex data
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_acf_average_kernel<<<blocks, threads_per_block>>>(
        acfbuf, xcfbuf, nrang, mplgs, nave, xcf_enabled);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated ACF normalization
 */
extern "C" cudaError_t cuda_acf_normalize(float *acfbuf,
                                          float *xcfbuf,
                                          int nrang, int mplgs,
                                          float attn, bool xcf_enabled) {
    if (!cuda_initialized) {
        cudaError_t init_error = acf_1_16_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!acfbuf || attn == 0.0f) return cudaErrorInvalidValue;
    
    int total_elements = nrang * mplgs * 2;  // Complex data
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_acf_normalize_kernel<<<blocks, threads_per_block>>>(
        acfbuf, xcfbuf, nrang, mplgs, attn, xcf_enabled);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated bad lag detection
 */
extern "C" cudaError_t cuda_acf_badlag_detect(const float *acfbuf,
                                              bool *badlag_mask,
                                              int nrang, int mplgs,
                                              float noise_threshold) {
    if (!cuda_initialized) {
        cudaError_t init_error = acf_1_16_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!acfbuf || !badlag_mask) return cudaErrorInvalidValue;
    
    dim3 grid(nrang);
    dim3 block(mplgs);
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_acf_badlag_kernel<<<grid, block>>>(
        acfbuf, badlag_mask, nrang, mplgs, noise_threshold);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated sum power calculation
 */
extern "C" cudaError_t cuda_acf_sum_power(const int16_t *inbuf,
                                          float *power_sum,
                                          float *power_max,
                                          int nrang, int nave,
                                          int offset, int smsep) {
    if (!cuda_initialized) {
        cudaError_t init_error = acf_1_16_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!inbuf || !power_sum || !power_max) return cudaErrorInvalidValue;
    
    // Initialize output values
    cudaMemset(power_sum, 0, sizeof(float));
    cudaMemset(power_max, 0, sizeof(float));
    
    int threads_per_block = 256;
    int blocks = (nrang + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_acf_sum_power_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        inbuf, power_sum, power_max, nrang, nave, offset, smsep);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated magnitude calculation
 */
extern "C" cudaError_t cuda_acf_magnitude(const float *acfbuf,
                                          float *magnitude,
                                          int nrang, int mplgs) {
    if (!cuda_initialized) {
        cudaError_t init_error = acf_1_16_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!acfbuf || !magnitude) return cudaErrorInvalidValue;
    
    dim3 grid(nrang);
    dim3 block(mplgs);
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_acf_magnitude_kernel<<<grid, block>>>(
        acfbuf, magnitude, nrang, mplgs);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * High-level ACF processing pipeline
 */
extern "C" cudaError_t cuda_acf_process_pipeline(cuda_acf_data_t *acf_data,
                                                 bool calculate_power,
                                                 bool detect_badlags,
                                                 float noise_threshold) {
    if (!acf_data) return cudaErrorInvalidValue;
    
    cudaError_t err;
    
    // Step 1: Calculate ACF
    err = cuda_acf_calculate(acf_data->inbuf, acf_data->acfbuf, acf_data->xcfbuf,
                            acf_data->lagfr, acf_data->smsep, acf_data->pat,
                            acf_data->nrang, acf_data->mplgs, acf_data->mpinc,
                            acf_data->nave, 0, acf_data->xcf_enabled);
    if (err != cudaSuccess) return err;
    
    // Step 2: Average ACF data
    if (acf_data->nave > 1) {
        err = cuda_acf_average(acf_data->acfbuf, acf_data->xcfbuf,
                              acf_data->nrang, acf_data->mplgs,
                              acf_data->nave, acf_data->xcf_enabled);
        if (err != cudaSuccess) return err;
    }
    
    // Step 3: Normalize by attenuation
    if (acf_data->attn != 0.0f && acf_data->attn != 1.0f) {
        err = cuda_acf_normalize(acf_data->acfbuf, acf_data->xcfbuf,
                                acf_data->nrang, acf_data->mplgs,
                                acf_data->attn, acf_data->xcf_enabled);
        if (err != cudaSuccess) return err;
    }
    
    // Step 4: Calculate power (if requested)
    if (calculate_power && acf_data->pwr0) {
        err = cuda_acf_power(acf_data->acfbuf, acf_data->pwr0,
                            acf_data->nrang, acf_data->mplgs);
        if (err != cudaSuccess) return err;
    }
    
    // Step 5: Detect bad lags (if requested)
    if (detect_badlags) {
        bool *badlag_mask;
        cudaMalloc(&badlag_mask, acf_data->nrang * acf_data->mplgs * sizeof(bool));
        
        err = cuda_acf_badlag_detect(acf_data->acfbuf, badlag_mask,
                                    acf_data->nrang, acf_data->mplgs,
                                    noise_threshold);
        
        // For now, just detect but don't apply corrections
        // In a full implementation, would mark or correct bad lags
        
        cudaFree(badlag_mask);
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

/* Utility Functions */
bool acf_1_16_cuda_is_available(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

int acf_1_16_cuda_get_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}
