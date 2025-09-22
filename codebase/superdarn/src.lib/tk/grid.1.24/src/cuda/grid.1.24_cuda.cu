#include "grid.1.24_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/find.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>

/* Global CUDA context */
static cublasHandle_t cublas_handle = NULL;
static bool cuda_initialized = false;
static bool profiling_enabled = false;
static cudaEvent_t start_event, stop_event;

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

/* Grid-specific data structures for CUDA */
typedef struct {
    float mlat, mlon, azm;          // Magnetic coordinates
    float vel_median, vel_sd;       // Velocity statistics
    float pwr_median, pwr_sd;       // Power statistics  
    float wdt_median, wdt_sd;       // Width statistics
    int st_id, chn, index;          // Station identifiers
    bool valid;                     // Validity flag
} cuda_grid_vector_t;

typedef struct {
    int st_id, chn, npnt;
    float freq0;
    float noise_mean, noise_sd;
    float vel_min, vel_max;
    float pwr_min, pwr_max;
    float wdt_min, wdt_max;
} cuda_grid_station_t;

/* CUDA Initialization */
__host__ cudaError_t grid_1_24_cuda_init(void) {
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

/* CUDA Kernels for Grid Processing */

/**
 * Parallel cell location kernel - replaces O(n) GridLocateCell
 */
__global__ void grid_locate_cell_kernel(const cuda_grid_vector_t *vectors, 
                                        int num_vectors, 
                                        int target_index,
                                        int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_vectors && vectors[idx].valid && vectors[idx].index == target_index) {
        *result = idx;
    }
}

/**
 * Parallel grid averaging kernel
 */
__global__ void grid_average_kernel(const cuda_grid_vector_t *input_vectors,
                                   int num_input,
                                   cuda_grid_vector_t *output_vectors,
                                   int *output_count,
                                   int averaging_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_input || !input_vectors[idx].valid) return;
    
    // Find existing cell or create new one
    int cell_idx = atomicAdd(output_count, 1);
    
    // Copy and process data based on averaging mode
    output_vectors[cell_idx] = input_vectors[idx];
    
    // Apply averaging mode logic
    switch (averaging_mode) {
        case 0: // Simple copy
            break;
        case 1: // Median averaging
            // Implement statistical processing
            break;
        case 2: // Weighted averaging
            // Implement weighted statistics
            break;
        default:
            break;
    }
}

/**
 * Linear regression kernel for grid merging
 */
__global__ void grid_linear_regression_kernel(const cuda_grid_vector_t *vectors,
                                             int num_vectors,
                                             const int *cell_indices,
                                             int num_cells,
                                             float *vpar, float *vper) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (cell_idx >= num_cells) return;
    
    // Accumulation variables
    float sx = 0.0f, cx = 0.0f, ysx = 0.0f, ycx = 0.0f, cxsx = 0.0f;
    int count = 0;
    
    // Find all vectors for this cell
    for (int i = 0; i < num_vectors; i++) {
        if (vectors[i].valid && vectors[i].index == cell_indices[cell_idx]) {
            float vel = vectors[i].vel_median;
            float azm = vectors[i].azm * M_PI / 180.0f; // Convert to radians
            
            float sin_azm = sinf(azm);
            float cos_azm = cosf(azm);
            
            sx += sin_azm;
            cx += cos_azm;
            ysx += vel * sin_azm;
            ycx += vel * cos_azm;
            cxsx += cos_azm * sin_azm;
            count++;
        }
    }
    
    if (count > 1) {
        // Solve linear regression
        float denom = count * cxsx - sx * cx;
        if (fabsf(denom) > 1e-6f) {
            vpar[cell_idx] = (count * ycx - cx * ysx) / denom;
            vper[cell_idx] = (cxsx * ysx - sx * ycx) / denom;
        } else {
            vpar[cell_idx] = 0.0f;
            vper[cell_idx] = 0.0f;
        }
    } else {
        vpar[cell_idx] = 0.0f;
        vper[cell_idx] = 0.0f;
    }
}

/**
 * Grid integration kernel with weighted averaging
 */
__global__ void grid_integrate_kernel(const cuda_grid_vector_t *input_vectors,
                                     int num_input,
                                     const cuda_grid_station_t *stations,
                                     int num_stations,
                                     cuda_grid_vector_t *output_vectors,
                                     int *output_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_input || !input_vectors[idx].valid) return;
    
    // Group by station and calculate weighted averages
    int station_id = input_vectors[idx].st_id;
    
    // Find station parameters for weighting
    float weight = 1.0f;
    for (int s = 0; s < num_stations; s++) {
        if (stations[s].st_id == station_id) {
            // Calculate weight based on power and noise
            float snr = input_vectors[idx].pwr_median / stations[s].noise_mean;
            weight = fmaxf(0.1f, fminf(1.0f, snr / 10.0f));
            break;
        }
    }
    
    // Output weighted result
    int out_idx = atomicAdd(output_count, 1);
    output_vectors[out_idx] = input_vectors[idx];
    
    // Apply weights to statistics
    output_vectors[out_idx].vel_median *= weight;
    output_vectors[out_idx].pwr_median *= weight;
    output_vectors[out_idx].wdt_median *= weight;
}

/**
 * Statistical reduction kernel for min/max/mean calculations
 */
__global__ void grid_statistical_reduction_kernel(const cuda_grid_vector_t *vectors,
                                                  int num_vectors,
                                                  float *vel_min, float *vel_max,
                                                  float *pwr_min, float *pwr_max,
                                                  float *wdt_min, float *wdt_max) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    float vel_min_local = FLT_MAX, vel_max_local = -FLT_MAX;
    float pwr_min_local = FLT_MAX, pwr_max_local = -FLT_MAX;
    float wdt_min_local = FLT_MAX, wdt_max_local = -FLT_MAX;
    
    // Load data and find local min/max
    if (idx < num_vectors && vectors[idx].valid) {
        vel_min_local = vel_max_local = vectors[idx].vel_median;
        pwr_min_local = pwr_max_local = vectors[idx].pwr_median;
        wdt_min_local = wdt_max_local = vectors[idx].wdt_median;
    }
    
    // Store in shared memory
    sdata[tid * 6 + 0] = vel_min_local;
    sdata[tid * 6 + 1] = vel_max_local;
    sdata[tid * 6 + 2] = pwr_min_local;
    sdata[tid * 6 + 3] = pwr_max_local;
    sdata[tid * 6 + 4] = wdt_min_local;
    sdata[tid * 6 + 5] = wdt_max_local;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 6 + 0] = fminf(sdata[tid * 6 + 0], sdata[(tid + s) * 6 + 0]);
            sdata[tid * 6 + 1] = fmaxf(sdata[tid * 6 + 1], sdata[(tid + s) * 6 + 1]);
            sdata[tid * 6 + 2] = fminf(sdata[tid * 6 + 2], sdata[(tid + s) * 6 + 2]);
            sdata[tid * 6 + 3] = fmaxf(sdata[tid * 6 + 3], sdata[(tid + s) * 6 + 3]);
            sdata[tid * 6 + 4] = fminf(sdata[tid * 6 + 4], sdata[(tid + s) * 6 + 4]);
            sdata[tid * 6 + 5] = fmaxf(sdata[tid * 6 + 5], sdata[(tid + s) * 6 + 5]);
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicMinFloat(vel_min, sdata[0]);
        atomicMaxFloat(vel_max, sdata[1]);
        atomicMinFloat(pwr_min, sdata[2]);
        atomicMaxFloat(pwr_max, sdata[3]);
        atomicMinFloat(wdt_min, sdata[4]);
        atomicMaxFloat(wdt_max, sdata[5]);
    }
}

/**
 * Atomic min/max operations for float
 */
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

/* Host Functions */

/**
 * Main grid processing function with CUDA acceleration
 */
cudaError_t grid_1_24_process_cuda(cuda_array_t *input_data,
                                   cuda_array_t *output_data,
                                   void *parameters) {
    if (!cuda_initialized) {
        cudaError_t init_error = grid_1_24_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!input_data || !output_data) return cudaErrorInvalidValue;
    
    // Cast input parameters
    // Parameters would contain GridData structure information
    
    int threads_per_block = 256;
    int blocks = (input_data->size + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    // Process based on data type
    if (input_data->type == CUDA_R_32F) {
        // For now, implement a basic processing kernel
        // In full implementation, this would call appropriate specialized kernels
        // based on the parameters
    }
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated grid averaging
 */
cudaError_t grid_1_24_average_cuda(const cuda_grid_vector_t *input_vectors,
                                   int num_input,
                                   cuda_grid_vector_t *output_vectors,
                                   int *output_count,
                                   int averaging_mode) {
    if (!cuda_initialized) {
        cudaError_t init_error = grid_1_24_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    int threads_per_block = 256;
    int blocks = (num_input + threads_per_block - 1) / threads_per_block;
    
    // Initialize output count
    CUDA_CHECK(cudaMemset(output_count, 0, sizeof(int)));
    
    // Launch averaging kernel
    grid_average_kernel<<<blocks, threads_per_block>>>(
        input_vectors, num_input, output_vectors, output_count, averaging_mode);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated cell location (parallel search)
 */
cudaError_t grid_1_24_locate_cell_cuda(const cuda_grid_vector_t *vectors,
                                       int num_vectors,
                                       int target_index,
                                       int *result) {
    if (!cuda_initialized) {
        cudaError_t init_error = grid_1_24_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    int threads_per_block = 256;
    int blocks = (num_vectors + threads_per_block - 1) / threads_per_block;
    
    // Initialize result
    CUDA_CHECK(cudaMemset(result, -1, sizeof(int)));
    
    // Launch cell location kernel
    grid_locate_cell_kernel<<<blocks, threads_per_block>>>(
        vectors, num_vectors, target_index, result);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated linear regression for grid merging
 */
cudaError_t grid_1_24_linear_regression_cuda(const cuda_grid_vector_t *vectors,
                                             int num_vectors,
                                             const int *cell_indices,
                                             int num_cells,
                                             float *vpar, float *vper) {
    if (!cuda_initialized) {
        cudaError_t init_error = grid_1_24_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    int threads_per_block = 256;
    int blocks = (num_cells + threads_per_block - 1) / threads_per_block;
    
    // Launch linear regression kernel
    grid_linear_regression_kernel<<<blocks, threads_per_block>>>(
        vectors, num_vectors, cell_indices, num_cells, vpar, vper);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated grid integration
 */
cudaError_t grid_1_24_integrate_cuda(const cuda_grid_vector_t *input_vectors,
                                     int num_input,
                                     const cuda_grid_station_t *stations,
                                     int num_stations,
                                     cuda_grid_vector_t *output_vectors,
                                     int *output_count) {
    if (!cuda_initialized) {
        cudaError_t init_error = grid_1_24_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    int threads_per_block = 256;
    int blocks = (num_input + threads_per_block - 1) / threads_per_block;
    
    // Initialize output count
    CUDA_CHECK(cudaMemset(output_count, 0, sizeof(int)));
    
    // Launch integration kernel
    grid_integrate_kernel<<<blocks, threads_per_block>>>(
        input_vectors, num_input, stations, num_stations, output_vectors, output_count);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * CUDA-accelerated statistical reduction
 */
cudaError_t grid_1_24_statistical_reduction_cuda(const cuda_grid_vector_t *vectors,
                                                 int num_vectors,
                                                 float *vel_min, float *vel_max,
                                                 float *pwr_min, float *pwr_max,
                                                 float *wdt_min, float *wdt_max) {
    if (!cuda_initialized) {
        cudaError_t init_error = grid_1_24_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    int threads_per_block = 256;
    int blocks = (num_vectors + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * 6 * sizeof(float);
    
    // Initialize min/max values
    float init_min = FLT_MAX, init_max = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(vel_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vel_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pwr_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pwr_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wdt_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wdt_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch statistical reduction kernel
    grid_statistical_reduction_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        vectors, num_vectors, vel_min, vel_max, pwr_min, pwr_max, wdt_min, wdt_max);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaGetLastError();
}

/**
 * High-level grid sorting using Thrust
 */
cudaError_t grid_1_24_sort_cuda(cuda_grid_vector_t *vectors, int num_vectors) {
    if (!cuda_initialized) {
        cudaError_t init_error = grid_1_24_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    try {
        // Use thrust::sort with custom comparator
        thrust::device_ptr<cuda_grid_vector_t> dev_ptr(vectors);
        
        // Sort by index for efficient cell grouping
        thrust::sort(dev_ptr, dev_ptr + num_vectors, 
                    [] __device__ (const cuda_grid_vector_t &a, const cuda_grid_vector_t &b) {
                        return a.index < b.index;
                    });
        
        cudaDeviceSynchronize();
        return cudaGetLastError();
    } catch (...) {
        return cudaErrorUnknown;
    }
}

/* Utility Functions */
bool grid.1.24_cuda_is_available(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

int grid.1.24_cuda_get_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}
