#include "cuda_acf.h"
#include <cuda_runtime.h>

__global__ void cuda_acf_power_kernel(const float2 *acf_data, float *power_output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    float2 acf_val = acf_data[idx];
    power_output[idx] = sqrtf(acf_val.x * acf_val.x + acf_val.y * acf_val.y);
}

extern "C" {
cuda_error_t cuda_acf_calculate_power(cuda_array_t *acf_data, cuda_array_t *power_output) {
    if (!acf_data || !power_output) return CUDA_ERROR_INVALID_ARGUMENT;
    
    size_t num_elements = acf_data->count;
    dim3 block_size(256);
    dim3 grid_size((num_elements + block_size.x - 1) / block_size.x);
    
    cuda_acf_power_kernel<<<grid_size, block_size>>>(
        (float2*)acf_data->memory.device_ptr,
        (float*)power_output->memory.device_ptr,
        num_elements
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return CUDA_SUCCESS;
}
}
