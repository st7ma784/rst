/**
 * @file cuda_raw_kernels.cu
 * @brief CUDA kernel implementations for raw.1.22 module
 * 
 * Provides GPU acceleration for SuperDARN raw data processing operations
 * including data reorganization, filtering, and memory operations.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <cub/cub.cuh>
#include <stdio.h>

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

/* Raw data CUDA structures */
typedef struct {
    float *pwr0;         // Power array (nrang elements)
    float *acfd_real;    // ACF real data
    float *acfd_imag;    // ACF imaginary data  
    float *xcfd_real;    // XCF real data
    float *xcfd_imag;    // XCF imaginary data
    int nrang;           // Number of range gates
    int mplgs;           // Number of lags
    float threshold;     // Power threshold
} cuda_raw_data_t;

typedef struct {
    double *tme;         // Time array
    int *inx;           // Index array
    int num_records;    // Number of records
} cuda_raw_index_t;

/**
 * Complex data interleaving kernel
 * Combines real and imaginary components into interleaved format
 */
__global__ void cuda_raw_interleave_complex_kernel(const float *real_data, 
                                                   const float *imag_data,
                                                   float *output,
                                                   int nrang, int mplgs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = nrang * mplgs;
    
    if (idx >= total_elements) return;
    
    int range = idx / mplgs;
    int lag = idx % mplgs;
    int output_idx = 2 * idx; // Interleaved index
    
    output[output_idx] = real_data[range * mplgs + lag];       // Real component
    output[output_idx + 1] = imag_data[range * mplgs + lag];   // Imaginary component
}

/**
 * Threshold-based filtering kernel
 * Generates sample list based on power threshold
 */
__global__ void cuda_raw_threshold_filter_kernel(const float *pwr0,
                                                 int *slist,
                                                 float threshold,
                                                 int nrang,
                                                 int *valid_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nrang) return;
    
    // Mark valid ranges above threshold
    valid_mask[idx] = (pwr0[idx] >= threshold) ? 1 : 0;
}

/**
 * Sparse gather operation kernel
 * Gathers data from sparse sample list
 */
__global__ void cuda_raw_sparse_gather_kernel(const float *input,
                                              const int *slist,
                                              float *output,
                                              int snum, int mplgs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= snum * mplgs) return;
    
    int sample_idx = idx / mplgs;
    int lag_idx = idx % mplgs;
    int range_gate = slist[sample_idx];
    
    // Gather operation: output[sample][lag] = input[range_gate][lag]
    output[sample_idx * mplgs + lag_idx] = input[range_gate * mplgs + lag_idx];
}

/**
 * Data reorganization kernel for encoding
 * Reorganizes raw data for efficient storage/transmission
 */
__global__ void cuda_raw_data_reorganize_kernel(const float *acfd_real,
                                                const float *acfd_imag,
                                                const float *xcfd_real, 
                                                const float *xcfd_imag,
                                                const int *slist,
                                                float *encoded_acfd,
                                                float *encoded_xcfd,
                                                int snum, int mplgs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= snum * mplgs) return;
    
    int sample_idx = idx / mplgs;
    int lag_idx = idx % mplgs;
    int range_gate = slist[sample_idx];
    int input_idx = range_gate * mplgs + lag_idx;
    int output_base = sample_idx * mplgs + lag_idx;
    
    // Interleave complex data: [real0, imag0, real1, imag1, ...]
    encoded_acfd[2 * output_base] = acfd_real[input_idx];       // Real
    encoded_acfd[2 * output_base + 1] = acfd_imag[input_idx];   // Imaginary
    
    if (xcfd_real && xcfd_imag) {
        encoded_xcfd[2 * output_base] = xcfd_real[input_idx];     // Real
        encoded_xcfd[2 * output_base + 1] = xcfd_imag[input_idx]; // Imaginary
    }
}

/**
 * Power-based range filtering kernel
 * Applies power threshold filtering and updates statistics
 */
__global__ void cuda_raw_power_filter_kernel(const float *pwr0,
                                             float *filtered_pwr,
                                             int *valid_indices,
                                             float threshold,
                                             int nrang) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nrang) return;
    
    if (pwr0[idx] >= threshold) {
        filtered_pwr[idx] = pwr0[idx];
        valid_indices[idx] = 1;  // Mark as valid
    } else {
        filtered_pwr[idx] = 0.0f;
        valid_indices[idx] = 0;  // Mark as invalid
    }
}

/**
 * Binary search kernel for time-based indexing
 * Performs parallel binary search in sorted time arrays
 */
__global__ void cuda_raw_time_search_kernel(const double *time_array,
                                            const double *search_times,
                                            int *result_indices,
                                            int num_records,
                                            int num_searches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_searches) return;
    
    double target_time = search_times[idx];
    int left = 0, right = num_records - 1;
    int result = -1;
    
    // Binary search
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (time_array[mid] == target_time) {
            result = mid;
            break;
        } else if (time_array[mid] < target_time) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    // If exact match not found, return closest lower index
    if (result == -1 && right >= 0) {
        result = right;
    }
    
    result_indices[idx] = result;
}

/**
 * Data deinterleaving kernel
 * Separates interleaved complex data back to real/imaginary arrays
 */
__global__ void cuda_raw_deinterleave_complex_kernel(const float *interleaved_data,
                                                     float *real_data,
                                                     float *imag_data,
                                                     int nrang, int mplgs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = nrang * mplgs;
    
    if (idx >= total_elements) return;
    
    int range = idx / mplgs;
    int lag = idx % mplgs;
    int input_idx = 2 * idx; // Interleaved index
    
    real_data[range * mplgs + lag] = interleaved_data[input_idx];     // Real component
    imag_data[range * mplgs + lag] = interleaved_data[input_idx + 1]; // Imaginary component
}

/**
 * Statistics calculation kernel
 * Computes basic statistics for raw data arrays
 */
__global__ void cuda_raw_statistics_kernel(const float *data,
                                           float *min_val, float *max_val,
                                           float *mean_val, float *sum_val,
                                           int num_elements) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    float local_min = FLT_MAX, local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    // Load data and compute local statistics
    if (idx < num_elements) {
        float val = data[idx];
        local_min = local_max = local_sum = val;
    }
    
    // Store in shared memory
    sdata[tid * 3 + 0] = local_min;
    sdata[tid * 3 + 1] = local_max;
    sdata[tid * 3 + 2] = local_sum;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 3 + 0] = fminf(sdata[tid * 3 + 0], sdata[(tid + s) * 3 + 0]); // min
            sdata[tid * 3 + 1] = fmaxf(sdata[tid * 3 + 1], sdata[(tid + s) * 3 + 1]); // max
            sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];                           // sum
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicMinFloat(min_val, sdata[0]);
        atomicMaxFloat(max_val, sdata[1]);
        atomicAdd(sum_val, sdata[2]);
    }
}

/* Atomic operations for float */
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val) {
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
 * CUDA-accelerated complex data interleaving
 */
extern "C" cudaError_t cuda_raw_interleave_complex(const float *real_data,
                                                   const float *imag_data,
                                                   float *output,
                                                   int nrang, int mplgs) {
    int total_elements = nrang * mplgs;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    cuda_raw_interleave_complex_kernel<<<blocks, threads_per_block>>>(
        real_data, imag_data, output, nrang, mplgs);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated threshold filtering with sample list generation
 */
extern "C" cudaError_t cuda_raw_threshold_filter(const float *pwr0,
                                                 int *slist,
                                                 int *snum,
                                                 float threshold,
                                                 int nrang) {
    // Allocate temporary arrays
    int *valid_mask;
    CUDA_CHECK(cudaMalloc(&valid_mask, nrang * sizeof(int)));
    
    int threads_per_block = 256;
    int blocks = (nrang + threads_per_block - 1) / threads_per_block;
    
    // Filter based on threshold
    cuda_raw_threshold_filter_kernel<<<blocks, threads_per_block>>>(
        pwr0, slist, threshold, nrang, valid_mask);
    
    // Use Thrust to compact the sample list
    thrust::device_ptr<int> valid_ptr(valid_mask);
    thrust::device_ptr<int> slist_ptr(slist);
    
    // Create index sequence
    thrust::sequence(slist_ptr, slist_ptr + nrang);
    
    // Compact to get only valid indices
    auto end_ptr = thrust::copy_if(slist_ptr, slist_ptr + nrang, valid_ptr, slist_ptr,
                                   thrust::identity<int>());
    
    // Get count of valid samples
    int host_snum = end_ptr - slist_ptr;
    CUDA_CHECK(cudaMemcpy(snum, &host_snum, sizeof(int), cudaMemcpyHostToDevice));
    
    cudaFree(valid_mask);
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated sparse data gathering
 */
extern "C" cudaError_t cuda_raw_sparse_gather(const float *input,
                                              const int *slist,
                                              float *output,
                                              int snum, int mplgs) {
    int total_elements = snum * mplgs;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    cuda_raw_sparse_gather_kernel<<<blocks, threads_per_block>>>(
        input, slist, output, snum, mplgs);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated data reorganization for encoding
 */
extern "C" cudaError_t cuda_raw_data_reorganize(const float *acfd_real,
                                                const float *acfd_imag,
                                                const float *xcfd_real,
                                                const float *xcfd_imag,
                                                const int *slist,
                                                float *encoded_acfd,
                                                float *encoded_xcfd,
                                                int snum, int mplgs) {
    int total_elements = snum * mplgs;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    cuda_raw_data_reorganize_kernel<<<blocks, threads_per_block>>>(
        acfd_real, acfd_imag, xcfd_real, xcfd_imag, slist,
        encoded_acfd, encoded_xcfd, snum, mplgs);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated time-based binary search
 */
extern "C" cudaError_t cuda_raw_time_search(const double *time_array,
                                            const double *search_times,
                                            int *result_indices,
                                            int num_records,
                                            int num_searches) {
    int threads_per_block = 256;
    int blocks = (num_searches + threads_per_block - 1) / threads_per_block;
    
    cuda_raw_time_search_kernel<<<blocks, threads_per_block>>>(
        time_array, search_times, result_indices, num_records, num_searches);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}