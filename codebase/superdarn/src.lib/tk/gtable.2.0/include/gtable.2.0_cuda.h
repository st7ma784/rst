#ifndef GTABLE.2.0_CUDA_H
#define GTABLE.2.0_CUDA_H

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

/* Core gtable.2.0 CUDA Functions */
cudaError_t gtable.2.0_process_cuda(
    cuda_array_t *input_data,
    cuda_array_t *output_data,
    void *parameters
);

cudaError_t gtable.2.0_process_ranges_cuda(
    cuda_range_data_t *range_data,
    void *parameters,
    cuda_array_t *results
);

/* Utility Functions */
bool gtable.2.0_cuda_is_available(void);
int gtable.2.0_cuda_get_device_count(void);
const char* gtable.2.0_cuda_get_error_string(cudaError_t error);

/* Performance Profiling */
typedef struct {
    float cpu_time_ms;
    float gpu_time_ms;
    float speedup_factor;
    size_t memory_used_bytes;
} gtable.2.0_cuda_profile_t;

cudaError_t gtable.2.0_cuda_enable_profiling(bool enable);
cudaError_t gtable.2.0_cuda_get_profile(gtable.2.0_cuda_profile_t *profile);

#ifdef __cplusplus
}
#endif

#endif /* GTABLE.2.0_CUDA_H */
