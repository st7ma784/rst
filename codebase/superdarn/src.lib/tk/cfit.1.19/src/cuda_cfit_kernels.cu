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

/* -----------------------------------------------------------------------
 * Kernel 2: decompress — packed float4 → cuda_cfit_data_t
 */
__global__ void cuda_cfit_decompress_kernel(const float *input,
                                            cuda_cfit_data_t *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx].power    = input[idx * 4 + 0];
    output[idx].velocity = input[idx * 4 + 1];
    output[idx].width    = input[idx * 4 + 2];
    output[idx].phi0     = input[idx * 4 + 3];
}

extern "C" {
cuda_error_t cuda_cfit_decompress_data(cuda_array_t *packed, cuda_array_t *output) {
    if (!packed || !output) return CUDA_ERROR_INVALID_ARGUMENT;
    size_t n = output->count;
    dim3 block(256), grid((n + 255) / 256);
    cuda_cfit_decompress_kernel<<<grid, block>>>(
        (const float *)packed->memory.device_ptr,
        (cuda_cfit_data_t *)output->memory.device_ptr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return CUDA_SUCCESS;
}
}

/* -----------------------------------------------------------------------
 * Kernel 3: quality filter — produce validity mask (1=keep, 0=reject)
 * Rejects cells with gsct set OR power below snr_threshold.
 */
__global__ void cuda_cfit_quality_filter_kernel(const cuda_cfit_data_t *data,
                                                unsigned char *mask,
                                                float snr_threshold, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    mask[idx] = (data[idx].quality_flag == 0 &&
                 data[idx].power >= snr_threshold) ? 1u : 0u;
}

extern "C" {
cuda_error_t cuda_cfit_filter_quality(cuda_array_t *data, cuda_array_t *mask,
                                      float snr_threshold) {
    if (!data || !mask) return CUDA_ERROR_INVALID_ARGUMENT;
    size_t n = data->count;
    dim3 block(256), grid((n + 255) / 256);
    cuda_cfit_quality_filter_kernel<<<grid, block>>>(
        (const cuda_cfit_data_t *)data->memory.device_ptr,
        (unsigned char *)mask->memory.device_ptr,
        snr_threshold, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return CUDA_SUCCESS;
}
}

/* -----------------------------------------------------------------------
 * Kernel 4: double → float conversion
 * Converts CPU-side CFitCell double arrays to the GPU float struct.
 * Called after cudaMemcpy of individual double arrays from CFitCell.
 */
__global__ void cuda_cfit_double_to_float_kernel(
    const double *d_v,    const double *d_p_l,  const double *d_w_l,
    const double *d_p_0,  const int    *d_gsct,
    cuda_cfit_data_t *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx].velocity     = (float)d_v[idx];
    out[idx].power        = (float)d_p_l[idx];
    out[idx].width        = (float)d_w_l[idx];
    out[idx].phi0         = (float)d_p_0[idx];
    out[idx].quality_flag = d_gsct[idx];
    out[idx].range        = idx;
}

extern "C" {
cuda_error_t cuda_cfit_convert_from_double(
    cuda_array_t *d_v, cuda_array_t *d_p_l, cuda_array_t *d_w_l,
    cuda_array_t *d_p_0, cuda_array_t *d_gsct, cuda_array_t *out) {
    if (!d_v || !d_p_l || !d_w_l || !d_p_0 || !d_gsct || !out)
        return CUDA_ERROR_INVALID_ARGUMENT;
    size_t n = out->count;
    dim3 block(256), grid((n + 255) / 256);
    cuda_cfit_double_to_float_kernel<<<grid, block>>>(
        (const double *)d_v->memory.device_ptr,
        (const double *)d_p_l->memory.device_ptr,
        (const double *)d_w_l->memory.device_ptr,
        (const double *)d_p_0->memory.device_ptr,
        (const int    *)d_gsct->memory.device_ptr,
        (cuda_cfit_data_t *)out->memory.device_ptr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return CUDA_SUCCESS;
}
}
