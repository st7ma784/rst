#include "cuda_cfit.h"
#include <cuda_runtime.h>

__global__ void cuda_cfit_compress_kernel(const cuda_cfit_data_t *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const cuda_cfit_data_t *data = &input[idx];
    output[idx * 4 + 0] = data->power;
    output[idx * 4 + 1] = data->velocity;
    output[idx * 4 + 2] = data->width;
    output[idx * 4 + 3] = data->phi0;
}

extern "C" {
cuda_error_t cuda_cfit_compress_data(cuda_array_t *input_data, cuda_array_t *output_data) {
    if (!input_data || !output_data) return CUDA_ERROR_INVALID_ARGUMENT;
    
    size_t n = input_data->count;
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    cuda_cfit_compress_kernel<<<grid_size, block_size>>>(
        (cuda_cfit_data_t*)input_data->memory.device_ptr,
        (float*)output_data->memory.device_ptr, n);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return CUDA_SUCCESS;
}
}
