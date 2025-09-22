#ifndef GRID.1.24_CUDA_H
#define GRID.1.24_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cufft.h>
#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Native CUDA Data Structures */

// CUDA-native array structure with unified memory
typedef struct {
    void *data;
    size_t size;
    size_t element_size;
    cudaDataType_t type;
    int device_id;
    bool is_on_device;
} cuda_array_t;

// CUDA-native matrix structure
typedef struct {
    void *data;
    int rows;
    int cols;
    int ld;
    cudaDataType_t type;
    int device_id;
} cuda_matrix_t;

// CUDA-native complex arrays
typedef struct {
    cuFloatComplex *data;
    size_t size;
    int device_id;
} cuda_complex_array_t;

// Range processing structure (SuperDARN specific)
typedef struct {
    int *ranges;
    float *powers;
    cuFloatComplex *phases;
    float *velocities;
    int num_ranges;
    int device_id;
} cuda_range_data_t;

/* Memory Management Functions */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type);
void cuda_array_destroy(cuda_array_t *array);
cudaError_t cuda_array_copy_to_device(cuda_array_t *array);
cudaError_t cuda_array_copy_to_host(cuda_array_t *array);

cuda_matrix_t* cuda_matrix_create(int rows, int cols, cudaDataType_t type);
void cuda_matrix_destroy(cuda_matrix_t *matrix);

cuda_complex_array_t* cuda_complex_array_create(size_t size);
void cuda_complex_array_destroy(cuda_complex_array_t *array);

cuda_range_data_t* cuda_range_data_create(int num_ranges);
void cuda_range_data_destroy(cuda_range_data_t *data);

/* Grid-specific CUDA data structures */
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

/* Core grid.1.24 CUDA Functions */
cudaError_t grid_1_24_process_cuda(
    cuda_array_t *input_data,
    cuda_array_t *output_data,
    void *parameters
);

cudaError_t grid_1_24_average_cuda(
    const cuda_grid_vector_t *input_vectors,
    int num_input,
    cuda_grid_vector_t *output_vectors,
    int *output_count,
    int averaging_mode
);

cudaError_t grid_1_24_locate_cell_cuda(
    const cuda_grid_vector_t *vectors,
    int num_vectors,
    int target_index,
    int *result
);

cudaError_t grid_1_24_linear_regression_cuda(
    const cuda_grid_vector_t *vectors,
    int num_vectors,
    const int *cell_indices,
    int num_cells,
    float *vpar, float *vper
);

cudaError_t grid_1_24_integrate_cuda(
    const cuda_grid_vector_t *input_vectors,
    int num_input,
    const cuda_grid_station_t *stations,
    int num_stations,
    cuda_grid_vector_t *output_vectors,
    int *output_count
);

cudaError_t grid_1_24_statistical_reduction_cuda(
    const cuda_grid_vector_t *vectors,
    int num_vectors,
    float *vel_min, float *vel_max,
    float *pwr_min, float *pwr_max,
    float *wdt_min, float *wdt_max
);

cudaError_t grid_1_24_sort_cuda(
    cuda_grid_vector_t *vectors, 
    int num_vectors
);

/* Utility Functions */
bool grid.1.24_cuda_is_available(void);
int grid.1.24_cuda_get_device_count(void);
const char* grid.1.24_cuda_get_error_string(cudaError_t error);

/* Performance Profiling */
typedef struct {
    float cpu_time_ms;
    float gpu_time_ms;
    float speedup_factor;
    size_t memory_used_bytes;
} grid.1.24_cuda_profile_t;

cudaError_t grid.1.24_cuda_enable_profiling(bool enable);
cudaError_t grid.1.24_cuda_get_profile(grid.1.24_cuda_profile_t *profile);

#ifdef __cplusplus
}
#endif

#endif /* GRID.1.24_CUDA_H */
