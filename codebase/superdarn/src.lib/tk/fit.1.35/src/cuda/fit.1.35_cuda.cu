#include "fit.1.35_cuda.h"
#include <stdio.h>
#include <stdlib.h>

/* Global CUDA context */
static cublasHandle_t cublas_handle = NULL;
static bool cuda_initialized = false;
static bool profiling_enabled = false;
static cudaEvent_t start_event, stop_event;

/* CUDA Initialization */
__host__ cudaError_t fit.1.35_cuda_init(void) {
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

/* CUDA-compatible FIT data structures */
typedef struct {
    float v;        // Line-of-sight velocity (m/s)
    float v_e;      // Velocity error (m/s)
    float p_l;      // Lambda power (dB)
    float p_l_e;    // Lambda power error (dB)
    float p_s;      // Sigma power (dB)
    float p_s_e;    // Sigma power error (dB)
    float w_l;      // Lambda spectral width (m/s)
    float w_l_e;    // Lambda spectral width error (m/s)
    float w_s;      // Sigma spectral width (m/s)
    float w_s_e;    // Sigma spectral width error (m/s)
    float sd_l;     // Lambda standard deviation
    float sd_s;     // Sigma standard deviation
    float sd_phi;   // Phase standard deviation
    float phi0;     // Phase of lag zero (degrees)
    float phi0_e;   // Phase error (degrees)
    float elv;      // Elevation angle (degrees)
    float elv_low;  // Lower elevation angle (degrees)
    float elv_high; // Upper elevation angle (degrees)
    int qflg;       // Quality flag
    int gsct;       // Ground scatter flag
    bool valid;     // Data validity flag
} cuda_fit_range_t;

typedef struct {
    int stid;       // Station ID
    int bmnum;      // Beam number
    int scan;       // Scan flag
    int cp;         // Control program ID
    int channel;    // Channel number
    double time;    // Time
    int nrang;      // Number of range gates
    int *slist;     // Sample list
    cuda_fit_range_t *ranges;  // Range data
} cuda_fit_data_t;

typedef struct {
    float v;        // Velocity (m/s)
    float p_l;      // Lambda power (dB)
    float w_l;      // Lambda spectral width (m/s)
    float phi0;     // Phase (degrees)
    float elv;      // Elevation angle (degrees)
    int qflg;       // Quality flag
    int gsct;       // Ground scatter flag
} cuda_cfit_cell_t;

/* CUDA Kernels for FIT processing */

/**
 * Range validation kernel
 * Validates range gates based on power thresholds and quality criteria
 */
__global__ void cuda_fit_validate_ranges_kernel(const cuda_fit_range_t *ranges,
                                                int *valid_indices,
                                                int *valid_count,
                                                int total_ranges,
                                                float min_power_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_ranges) return;
    
    const cuda_fit_range_t *range = &ranges[idx];
    
    // Validation criteria
    bool has_valid_power = (range->p_l > min_power_threshold);
    bool has_reasonable_velocity = (fabsf(range->v) < 2000.0f);  // |v| < 2000 m/s
    bool has_reasonable_width = (range->w_l > 0.0f && range->w_l < 1000.0f);
    bool has_quality_flag = (range->qflg > 0);
    bool is_marked_valid = range->valid;
    
    // Mark as valid if all criteria met
    int is_valid = has_valid_power && has_reasonable_velocity && 
                   has_reasonable_width && has_quality_flag && is_marked_valid;
    
    if (is_valid) {
        int pos = atomicAdd(valid_count, 1);
        valid_indices[pos] = idx;
    }
}

/**
 * FIT to CFIT conversion kernel
 * Converts full FIT data to compact CFIT format
 */
__global__ void cuda_fit_to_cfit_kernel(const cuda_fit_range_t *fit_ranges,
                                        cuda_cfit_cell_t *cfit_cells,
                                        const int *valid_indices,
                                        int num_valid_ranges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_valid_ranges) return;
    
    int range_idx = valid_indices[idx];
    const cuda_fit_range_t *fit_range = &fit_ranges[range_idx];
    cuda_cfit_cell_t *cfit_cell = &cfit_cells[idx];
    
    // Copy essential parameters
    cfit_cell->v = fit_range->v;
    cfit_cell->p_l = fit_range->p_l;
    cfit_cell->w_l = fit_range->w_l;
    cfit_cell->phi0 = fit_range->phi0;
    cfit_cell->elv = fit_range->elv;
    cfit_cell->qflg = fit_range->qflg;
    cfit_cell->gsct = fit_range->gsct;
}

/**
 * Range data processing kernel
 * Applies quality control and data conditioning
 */
__global__ void cuda_fit_process_ranges_kernel(cuda_fit_range_t *ranges,
                                               int num_ranges,
                                               float noise_level,
                                               float velocity_limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_ranges) return;
    
    cuda_fit_range_t *range = &ranges[idx];
    
    // Skip invalid ranges
    if (!range->valid || range->p_l <= 0.0f) {
        range->qflg = 0;
        return;
    }
    
    // Signal-to-noise ratio calculation
    float snr = powf(10.0f, range->p_l / 10.0f) / noise_level;
    
    // Quality flag determination based on SNR and error bounds
    if (snr > 3.0f && range->v_e < fabsf(range->v) * 0.5f) {
        range->qflg = 1;
    } else {
        range->qflg = 0;
    }
    
    // Velocity limiting
    if (fabsf(range->v) > velocity_limit) {
        range->v = copysignf(velocity_limit, range->v);
        range->qflg = 0;  // Mark as low quality
    }
    
    // Ground scatter classification
    if (range->w_l < 100.0f && fabsf(range->v) < 200.0f && range->elv < 15.0f) {
        range->gsct = 1;  // Ground scatter
    } else {
        range->gsct = 0;  // Ionospheric scatter
    }
    
    // Error estimation refinement
    if (snr > 0.0f) {
        range->v_e = fminf(range->v_e, 100.0f / sqrtf(snr));
        range->p_l_e = fminf(range->p_l_e, 5.0f / sqrtf(snr));
        range->w_l_e = fminf(range->w_l_e, range->w_l * 0.3f);
    }
}

/**
 * Statistics calculation kernel
 * Computes statistical parameters for quality assessment
 */
__global__ void cuda_fit_statistics_kernel(const cuda_fit_range_t *ranges,
                                           int num_ranges,
                                           float *velocity_mean,
                                           float *power_mean,
                                           float *width_mean,
                                           int *valid_count) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    float local_v = 0.0f, local_p = 0.0f, local_w = 0.0f;
    int local_count = 0;
    
    // Load data and compute local sums
    if (idx < num_ranges && ranges[idx].valid && ranges[idx].qflg > 0) {
        local_v = ranges[idx].v;
        local_p = ranges[idx].p_l;
        local_w = ranges[idx].w_l;
        local_count = 1;
    }
    
    // Store in shared memory
    sdata[tid * 4 + 0] = local_v;
    sdata[tid * 4 + 1] = local_p;
    sdata[tid * 4 + 2] = local_w;
    sdata[tid * 4 + 3] = (float)local_count;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 4 + 0] += sdata[(tid + s) * 4 + 0];  // velocity sum
            sdata[tid * 4 + 1] += sdata[(tid + s) * 4 + 1];  // power sum
            sdata[tid * 4 + 2] += sdata[(tid + s) * 4 + 2];  // width sum
            sdata[tid * 4 + 3] += sdata[(tid + s) * 4 + 3];  // count sum
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        int count = (int)sdata[3];
        if (count > 0) {
            atomicAdd(velocity_mean, sdata[0] / count);
            atomicAdd(power_mean, sdata[1] / count);
            atomicAdd(width_mean, sdata[2] / count);
        }
        atomicAdd(valid_count, count);
    }
}

/**
 * Elevation angle calculation kernel
 * Computes elevation angles based on range and phase measurements
 */
__global__ void cuda_fit_elevation_kernel(cuda_fit_range_t *ranges,
                                          int num_ranges,
                                          float antenna_separation,
                                          float operating_frequency) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_ranges) return;
    
    cuda_fit_range_t *range = &ranges[idx];
    
    if (!range->valid || range->qflg <= 0) return;
    
    // Calculate elevation angle from phase difference
    float wavelength = 299792458.0f / (operating_frequency * 1000.0f);  // m
    float phase_rad = range->phi0 * M_PI / 180.0f;
    
    // Elevation calculation (simplified)
    float sin_elv = (phase_rad * wavelength) / (2.0f * M_PI * antenna_separation);
    sin_elv = fmaxf(-1.0f, fminf(1.0f, sin_elv));  // Clamp to valid range
    
    range->elv = asinf(sin_elv) * 180.0f / M_PI;  // Convert to degrees
    
    // Error estimation based on phase error
    float elv_error = range->phi0_e * wavelength / (2.0f * M_PI * antenna_separation);
    range->elv_low = range->elv - elv_error;
    range->elv_high = range->elv + elv_error;
}

/* Host wrapper functions */

/**
 * CUDA-accelerated range validation
 */
extern "C" cudaError_t cuda_fit_validate_ranges(const cuda_fit_range_t *ranges,
                                                int *valid_indices,
                                                int *valid_count,
                                                int total_ranges,
                                                float min_power_threshold) {
    if (!cuda_initialized) {
        cudaError_t init_error = fit_1_35_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!ranges || !valid_indices || !valid_count) return cudaErrorInvalidValue;
    
    // Initialize valid count
    cudaMemset(valid_count, 0, sizeof(int));
    
    int threads_per_block = 256;
    int blocks = (total_ranges + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_fit_validate_ranges_kernel<<<blocks, threads_per_block>>>(
        ranges, valid_indices, valid_count, total_ranges, min_power_threshold);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated FIT to CFIT conversion
 */
extern "C" cudaError_t cuda_fit_to_cfit(const cuda_fit_range_t *fit_ranges,
                                        cuda_cfit_cell_t *cfit_cells,
                                        const int *valid_indices,
                                        int num_valid_ranges) {
    if (!cuda_initialized) {
        cudaError_t init_error = fit_1_35_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!fit_ranges || !cfit_cells || !valid_indices) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_valid_ranges + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_fit_to_cfit_kernel<<<blocks, threads_per_block>>>(
        fit_ranges, cfit_cells, valid_indices, num_valid_ranges);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated range data processing
 */
extern "C" cudaError_t cuda_fit_process_ranges(cuda_fit_range_t *ranges,
                                               int num_ranges,
                                               float noise_level,
                                               float velocity_limit) {
    if (!cuda_initialized) {
        cudaError_t init_error = fit_1_35_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!ranges) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_ranges + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_fit_process_ranges_kernel<<<blocks, threads_per_block>>>(
        ranges, num_ranges, noise_level, velocity_limit);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated statistics calculation
 */
extern "C" cudaError_t cuda_fit_calculate_statistics(const cuda_fit_range_t *ranges,
                                                     int num_ranges,
                                                     float *velocity_mean,
                                                     float *power_mean,
                                                     float *width_mean,
                                                     int *valid_count) {
    if (!cuda_initialized) {
        cudaError_t init_error = fit_1_35_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!ranges || !velocity_mean || !power_mean || !width_mean || !valid_count) {
        return cudaErrorInvalidValue;
    }
    
    // Initialize results
    cudaMemset(velocity_mean, 0, sizeof(float));
    cudaMemset(power_mean, 0, sizeof(float));
    cudaMemset(width_mean, 0, sizeof(float));
    cudaMemset(valid_count, 0, sizeof(int));
    
    int threads_per_block = 256;
    int blocks = (num_ranges + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * 4 * sizeof(float);
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_fit_statistics_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        ranges, num_ranges, velocity_mean, power_mean, width_mean, valid_count);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated elevation angle calculation
 */
extern "C" cudaError_t cuda_fit_calculate_elevation(cuda_fit_range_t *ranges,
                                                    int num_ranges,
                                                    float antenna_separation,
                                                    float operating_frequency) {
    if (!cuda_initialized) {
        cudaError_t init_error = fit_1_35_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!ranges) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (num_ranges + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_fit_elevation_kernel<<<blocks, threads_per_block>>>(
        ranges, num_ranges, antenna_separation, operating_frequency);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * High-level FIT processing pipeline
 */
extern "C" cudaError_t cuda_fit_process_pipeline(cuda_fit_data_t *fit_data,
                                                 cuda_cfit_cell_t *cfit_cells,
                                                 int *num_valid_ranges,
                                                 float min_power_threshold,
                                                 float noise_level,
                                                 float velocity_limit,
                                                 float antenna_separation,
                                                 float operating_frequency) {
    if (!fit_data || !cfit_cells || !num_valid_ranges) return cudaErrorInvalidValue;
    
    cudaError_t err;
    
    // Step 1: Process ranges (quality control and conditioning)
    err = cuda_fit_process_ranges(fit_data->ranges, fit_data->nrang, 
                                 noise_level, velocity_limit);
    if (err != cudaSuccess) return err;
    
    // Step 2: Calculate elevation angles
    err = cuda_fit_calculate_elevation(fit_data->ranges, fit_data->nrang,
                                      antenna_separation, operating_frequency);
    if (err != cudaSuccess) return err;
    
    // Step 3: Validate ranges and create compact list
    int *valid_indices;
    int *valid_count;
    
    cudaMalloc(&valid_indices, fit_data->nrang * sizeof(int));
    cudaMalloc(&valid_count, sizeof(int));
    
    err = cuda_fit_validate_ranges(fit_data->ranges, valid_indices, valid_count,
                                  fit_data->nrang, min_power_threshold);
    if (err != cudaSuccess) {
        cudaFree(valid_indices);
        cudaFree(valid_count);
        return err;
    }
    
    // Get number of valid ranges
    int host_valid_count;
    cudaMemcpy(&host_valid_count, valid_count, sizeof(int), cudaMemcpyDeviceToHost);
    *num_valid_ranges = host_valid_count;
    
    // Step 4: Convert to CFIT format
    if (host_valid_count > 0) {
        err = cuda_fit_to_cfit(fit_data->ranges, cfit_cells, valid_indices, host_valid_count);
    }
    
    cudaFree(valid_indices);
    cudaFree(valid_count);
    
    return err;
}

/* Utility Functions */
bool fit.1.35_cuda_is_available(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

int fit.1.35_cuda_get_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}
