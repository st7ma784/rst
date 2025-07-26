/*
 * Universal CUDA Header for channel.1.0
 * Provides CUDA acceleration for any SuperDARN module type
 */

#ifndef CHANNEL.1.0_CUDA_H
#define CHANNEL.1.0_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cufft.h>
#include <curand.h>
#include <cuComplex.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Universal CUDA Data Structures */

// Generic CUDA buffer for any data type
typedef struct {
    void *data;
    size_t size;
    size_t element_size;
    cudaDataType_t type;
    int device_id;
    bool is_managed;
} channel.1.0_cuda_buffer_t;

// Generic CUDA array structure
typedef struct {
    void *data;
    size_t num_elements;
    size_t element_size;
    cudaDataType_t type;
    int device_id;
} channel.1.0_cuda_array_t;

// I/O acceleration structure
typedef struct {
    void *input_buffer;
    void *output_buffer;
    size_t buffer_size;
    cudaStream_t stream;
    int device_id;
} channel.1.0_cuda_io_t;

// Processing context
typedef struct {
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    cufftHandle cufft_plan;
    cudaStream_t compute_stream;
    cudaStream_t memory_stream;
    bool initialized;
} channel.1.0_cuda_context_t;

/* Memory Management */
channel.1.0_cuda_buffer_t* channel.1.0_cuda_buffer_create(size_t size, cudaDataType_t type);
void channel.1.0_cuda_buffer_destroy(channel.1.0_cuda_buffer_t *buffer);
cudaError_t channel.1.0_cuda_buffer_copy_to_device(channel.1.0_cuda_buffer_t *buffer);
cudaError_t channel.1.0_cuda_buffer_copy_to_host(channel.1.0_cuda_buffer_t *buffer);

channel.1.0_cuda_array_t* channel.1.0_cuda_array_create(size_t num_elements, size_t element_size);
void channel.1.0_cuda_array_destroy(channel.1.0_cuda_array_t *array);

channel.1.0_cuda_io_t* channel.1.0_cuda_io_create(size_t buffer_size);
void channel.1.0_cuda_io_destroy(channel.1.0_cuda_io_t *io);

/* Context Management */
cudaError_t channel.1.0_cuda_init(channel.1.0_cuda_context_t *ctx);
void channel.1.0_cuda_cleanup(channel.1.0_cuda_context_t *ctx);

/* Core Processing Functions */
cudaError_t channel.1.0_process_cuda(
    channel.1.0_cuda_buffer_t *input,
    channel.1.0_cuda_buffer_t *output,
    void *parameters
);

cudaError_t channel.1.0_process_async_cuda(
    channel.1.0_cuda_buffer_t *input,
    channel.1.0_cuda_buffer_t *output,
    void *parameters,
    cudaStream_t stream
);

/* I/O Acceleration */
cudaError_t channel.1.0_read_cuda(
    const char *filename,
    channel.1.0_cuda_buffer_t *buffer
);

cudaError_t channel.1.0_write_cuda(
    const char *filename,
    channel.1.0_cuda_buffer_t *buffer
);

/* Utility Functions */
bool channel.1.0_cuda_is_available(void);
int channel.1.0_cuda_get_device_count(void);
cudaError_t channel.1.0_cuda_set_device(int device_id);
const char* channel.1.0_cuda_get_error_string(cudaError_t error);

/* Performance Monitoring */
typedef struct {
    float processing_time_ms;
    float memory_transfer_time_ms;
    float total_time_ms;
    size_t memory_used_bytes;
    float speedup_factor;
} channel.1.0_cuda_perf_t;

cudaError_t channel.1.0_cuda_enable_profiling(bool enable);
cudaError_t channel.1.0_cuda_get_performance(channel.1.0_cuda_perf_t *perf);

/* Compatibility Layer */
int channel.1.0_process_auto(void *input, void *output, void *params);
bool channel.1.0_is_cuda_enabled(void);
const char* channel.1.0_get_compute_mode(void);

#ifdef __cplusplus
}
#endif

#endif /* CHANNEL.1.0_CUDA_H */
