/**
 * @file cuda_lmfit_kernels.cu
 * @brief CUDA kernels for Levenberg-Marquardt fitting acceleration
 * 
 * This file contains GPU kernels for the most computationally intensive
 * parts of the LMFIT v2.0 algorithm, providing significant speedup
 * for large datasets.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#include "cuda_lmfit.h"
#include "cuda_lmfit_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <math.h>

// =============================================================================
// CUDA KERNEL IMPLEMENTATIONS
// =============================================================================

/**
 * @brief GPU kernel for parallel power calculation
 */
__global__ void cuda_calculate_power_kernel(
    const float2 *acf_data,     ///< Input ACF data (complex)
    float *power_output,        ///< Output power values
    float *sigma_output,        ///< Output sigma values
    const int *lag_nums,        ///< Lag numbers
    const float *lag_times,     ///< Lag times
    int num_ranges,             ///< Number of range gates
    int num_lags,               ///< Number of lags
    float nave                  ///< Number of averages
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lag_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (range_idx >= num_ranges || lag_idx >= num_lags) return;
    
    int data_idx = range_idx * num_lags + lag_idx;
    float2 acf_val = acf_data[data_idx];
    
    // Calculate power: P = sqrt(real^2 + imag^2)
    float power = sqrtf(acf_val.x * acf_val.x + acf_val.y * acf_val.y);
    
    // Check for valid power
    if (power <= 0.0f) {
        power_output[data_idx] = 0.0f;
        sigma_output[data_idx] = 0.0f;
        return;
    }
    
    // Calculate error: sigma = P / sqrt(nave)
    float sigma = power / sqrtf(nave);
    
    power_output[data_idx] = power;
    sigma_output[data_idx] = sigma;
}

/**
 * @brief GPU kernel for parallel ACF/XCF node creation
 */
__global__ void cuda_create_acf_nodes_kernel(
    const float2 *acf_data,     ///< Input ACF data (complex)
    cuda_acf_node_t *acf_nodes, ///< Output ACF nodes
    const int *lag_nums,        ///< Lag numbers
    const float *lag_times,     ///< Lag times
    int num_ranges,             ///< Number of range gates
    int num_lags,               ///< Number of lags
    float nave,                 ///< Number of averages
    bool is_xcf                 ///< Whether this is XCF data
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lag_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (range_idx >= num_ranges || lag_idx >= num_lags) return;
    
    int data_idx = range_idx * num_lags + lag_idx;
    float2 acf_val = acf_data[data_idx];
    
    // Calculate power for error estimation
    float power = sqrtf(acf_val.x * acf_val.x + acf_val.y * acf_val.y);
    
    if (power <= 0.0f) {
        // Mark as invalid
        acf_nodes[data_idx].lag_num = -1;
        return;
    }
    
    // Fill ACF node
    acf_nodes[data_idx].lag_num = lag_nums[lag_idx];
    acf_nodes[data_idx].re = acf_val.x;
    acf_nodes[data_idx].im = acf_val.y;
    acf_nodes[data_idx].sigma_re = power / sqrtf(nave);
    acf_nodes[data_idx].sigma_im = power / sqrtf(nave);
    acf_nodes[data_idx].t = lag_times[lag_idx];
}

/**
 * @brief GPU kernel for self-clutter estimation
 */
__global__ void cuda_estimate_self_clutter_kernel(
    const float *lag0_powers,   ///< Lag-0 powers for all ranges
    float *self_clutter,        ///< Output self-clutter estimates
    const int *range_indices,   ///< Range indices
    int num_ranges,             ///< Number of ranges
    int num_lags,               ///< Number of lags
    float clutter_factor        ///< Clutter estimation factor
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lag_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (range_idx >= num_ranges || lag_idx >= num_lags) return;
    
    int current_range = range_indices[range_idx];
    int data_idx = range_idx * num_lags + lag_idx;
    
    // Simple self-clutter model: estimate based on nearby ranges
    float clutter_sum = 0.0f;
    int clutter_count = 0;
    
    // Look at neighboring ranges (within ±2 range gates)
    for (int offset = -2; offset <= 2; offset++) {
        int neighbor_range = current_range + offset;
        if (neighbor_range >= 0 && neighbor_range < num_ranges) {
            int neighbor_idx = -1;
            
            // Find the neighbor range in our list
            for (int i = 0; i < num_ranges; i++) {
                if (range_indices[i] == neighbor_range) {
                    neighbor_idx = i;
                    break;
                }
            }
            
            if (neighbor_idx >= 0) {
                float neighbor_power = lag0_powers[neighbor_idx];
                if (neighbor_power > 0.0f) {
                    clutter_sum += neighbor_power;
                    clutter_count++;
                }
            }
        }
    }
    
    // Estimate self-clutter as a fraction of average neighbor power
    if (clutter_count > 0) {
        float avg_neighbor_power = clutter_sum / clutter_count;
        self_clutter[data_idx] = clutter_factor * avg_neighbor_power;
    } else {
        self_clutter[data_idx] = 0.0f;
    }
}

/**
 * @brief GPU kernel for Jacobian matrix calculation
 */
__global__ void cuda_calculate_jacobian_kernel(
    const float *parameters,    ///< Current parameter estimates [P, vel, wid, phi0]
    const float *times,         ///< Time values for each lag
    const float2 *observations, ///< Observed ACF values
    float *jacobian,           ///< Output Jacobian matrix
    int num_observations,      ///< Number of data points
    int num_parameters,        ///< Number of parameters (4)
    float lambda,              ///< Radar wavelength
    float dt                   ///< Time step
) {
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int param_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (obs_idx >= num_observations || param_idx >= num_parameters) return;
    
    float P = parameters[0];
    float vel = parameters[1];
    float wid = parameters[2];
    float phi0 = parameters[3];
    
    float t = times[obs_idx];
    float jacobian_val = 0.0f;
    
    // Calculate partial derivatives for ACF model:
    // ACF(t) = P * exp(-2π*wid*t/λ) * exp(i*4π*vel*t/λ + i*phi0)
    
    float decay = expf(-2.0f * M_PI * wid * t / lambda);
    float phase = 4.0f * M_PI * vel * t / lambda + phi0;
    float cos_phase = cosf(phase);
    float sin_phase = sinf(phase);
    
    switch (param_idx) {
        case 0: // ∂/∂P
            jacobian_val = decay * cos_phase; // For real part
            if (obs_idx >= num_observations / 2) {
                jacobian_val = decay * sin_phase; // For imaginary part
            }
            break;
            
        case 1: // ∂/∂vel
            jacobian_val = P * decay * (-4.0f * M_PI * t / lambda) * sin_phase;
            if (obs_idx >= num_observations / 2) {
                jacobian_val = P * decay * (4.0f * M_PI * t / lambda) * cos_phase;
            }
            break;
            
        case 2: // ∂/∂wid
            jacobian_val = P * (-2.0f * M_PI * t / lambda) * decay * cos_phase;
            if (obs_idx >= num_observations / 2) {
                jacobian_val = P * (-2.0f * M_PI * t / lambda) * decay * sin_phase;
            }
            break;
            
        case 3: // ∂/∂phi0
            jacobian_val = P * decay * (-sin_phase);
            if (obs_idx >= num_observations / 2) {
                jacobian_val = P * decay * cos_phase;
            }
            break;
    }
    
    int jacobian_idx = obs_idx * num_parameters + param_idx;
    jacobian[jacobian_idx] = jacobian_val;
}

/**
 * @brief GPU kernel for residual calculation
 */
__global__ void cuda_calculate_residuals_kernel(
    const float *parameters,    ///< Current parameter estimates
    const float *times,         ///< Time values for each lag
    const float2 *observations, ///< Observed ACF values
    float *residuals,          ///< Output residuals
    float *weights,            ///< Data weights (1/sigma^2)
    int num_observations,      ///< Number of data points
    float lambda               ///< Radar wavelength
) {
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (obs_idx >= num_observations) return;
    
    float P = parameters[0];
    float vel = parameters[1];
    float wid = parameters[2];
    float phi0 = parameters[3];
    
    float t = times[obs_idx];
    
    // Calculate model prediction
    float decay = expf(-2.0f * M_PI * wid * t / lambda);
    float phase = 4.0f * M_PI * vel * t / lambda + phi0;
    
    float2 model;
    model.x = P * decay * cosf(phase);
    model.y = P * decay * sinf(phase);
    
    // Calculate weighted residuals
    float2 obs = observations[obs_idx];
    float weight = sqrtf(weights[obs_idx]);
    
    // Store residuals for real and imaginary parts
    int real_idx = obs_idx;
    int imag_idx = obs_idx + num_observations;
    
    residuals[real_idx] = weight * (obs.x - model.x);
    residuals[imag_idx] = weight * (obs.y - model.y);
}

/**
 * @brief GPU kernel for error estimation
 */
__global__ void cuda_estimate_errors_kernel(
    const float *fitted_params, ///< Fitted parameters
    const float *times,         ///< Time values
    const float *lag0_powers,   ///< Lag-0 powers
    const float *self_clutter,  ///< Self-clutter estimates
    float *sigma_re,           ///< Output real part errors
    float *sigma_im,           ///< Output imaginary part errors
    int num_observations,      ///< Number of observations
    float noise_power,         ///< Noise power level
    float nave,                ///< Number of averages
    float lambda               ///< Radar wavelength
) {
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (obs_idx >= num_observations) return;
    
    float P = fitted_params[0];
    float vel = fitted_params[1];
    float wid = fitted_params[2];
    
    float t = times[obs_idx];
    float lag0_pwr = lag0_powers[0]; // Assuming first element is lag-0
    float clutter = self_clutter[obs_idx];
    
    // Calculate correlation coefficient
    float rho = expf(-2.0f * M_PI * wid * t / lambda);
    if (rho > 0.999f) rho = 0.999f;
    
    rho = rho * P / (P + noise_power + clutter);
    
    float phase = 4.0f * M_PI * vel * t / lambda;
    float rho_r = rho * cosf(phase);
    float rho_i = rho * sinf(phase);
    
    // First-order error estimation
    float base_error = (P + noise_power + clutter) / sqrtf(nave);
    
    sigma_re[obs_idx] = base_error * sqrtf((1.0f - rho * rho) / 2.0f + rho_r * rho_r);
    sigma_im[obs_idx] = base_error * sqrtf((1.0f - rho * rho) / 2.0f + rho_i * rho_i);
}

/**
 * @brief GPU kernel for convergence checking
 */
__global__ void cuda_check_convergence_kernel(
    const float *current_params,  ///< Current parameter estimates
    const float *previous_params, ///< Previous parameter estimates
    bool *converged_flags,        ///< Output convergence flags
    int num_ranges,              ///< Number of ranges
    int num_params,              ///< Number of parameters per range
    float tolerance              ///< Convergence tolerance
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (range_idx >= num_ranges) return;
    
    bool range_converged = true;
    
    for (int param_idx = 0; param_idx < num_params; param_idx++) {
        int idx = range_idx * num_params + param_idx;
        float current = current_params[idx];
        float previous = previous_params[idx];
        
        // Check relative change
        float relative_change = fabsf(current - previous) / (fabsf(previous) + 1e-10f);
        
        if (relative_change > tolerance) {
            range_converged = false;
            break;
        }
    }
    
    converged_flags[range_idx] = range_converged;
}

// =============================================================================
// CUDA UTILITY KERNELS
// =============================================================================

/**
 * @brief GPU kernel for parallel data filtering
 */
__global__ void cuda_filter_data_kernel(
    const float2 *input_data,   ///< Input data
    float2 *output_data,        ///< Filtered output data
    const bool *valid_flags,    ///< Validity flags
    const int *input_indices,   ///< Input to output index mapping
    int num_input,             ///< Number of input elements
    int num_output             ///< Number of output elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_input) return;
    
    if (valid_flags[idx]) {
        int output_idx = input_indices[idx];
        if (output_idx >= 0 && output_idx < num_output) {
            output_data[output_idx] = input_data[idx];
        }
    }
}

/**
 * @brief GPU kernel for parallel array reduction (sum)
 */
__global__ void cuda_reduce_sum_kernel(
    const float *input,         ///< Input array
    float *output,             ///< Output sum
    int num_elements           ///< Number of elements
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < num_elements) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// CUDA KERNEL LAUNCH WRAPPERS
// =============================================================================

extern "C" {

cuda_error_t cuda_launch_power_calculation(
    const float2 *acf_data,
    float *power_output,
    float *sigma_output,
    const int *lag_nums,
    const float *lag_times,
    int num_ranges,
    int num_lags,
    float nave
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (num_ranges + block_size.x - 1) / block_size.x,
        (num_lags + block_size.y - 1) / block_size.y
    );
    
    cuda_calculate_power_kernel<<<grid_size, block_size>>>(
        acf_data, power_output, sigma_output, lag_nums, lag_times,
        num_ranges, num_lags, nave
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_launch_acf_creation(
    const float2 *acf_data,
    cuda_acf_node_t *acf_nodes,
    const int *lag_nums,
    const float *lag_times,
    int num_ranges,
    int num_lags,
    float nave,
    bool is_xcf
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (num_ranges + block_size.x - 1) / block_size.x,
        (num_lags + block_size.y - 1) / block_size.y
    );
    
    cuda_create_acf_nodes_kernel<<<grid_size, block_size>>>(
        acf_data, acf_nodes, lag_nums, lag_times,
        num_ranges, num_lags, nave, is_xcf
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_launch_self_clutter_estimation(
    const float *lag0_powers,
    float *self_clutter,
    const int *range_indices,
    int num_ranges,
    int num_lags,
    float clutter_factor
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (num_ranges + block_size.x - 1) / block_size.x,
        (num_lags + block_size.y - 1) / block_size.y
    );
    
    cuda_estimate_self_clutter_kernel<<<grid_size, block_size>>>(
        lag0_powers, self_clutter, range_indices,
        num_ranges, num_lags, clutter_factor
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_launch_jacobian_calculation(
    const float *parameters,
    const float *times,
    const float2 *observations,
    float *jacobian,
    int num_observations,
    int num_parameters,
    float lambda,
    float dt
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (num_observations + block_size.x - 1) / block_size.x,
        (num_parameters + block_size.y - 1) / block_size.y
    );
    
    cuda_calculate_jacobian_kernel<<<grid_size, block_size>>>(
        parameters, times, observations, jacobian,
        num_observations, num_parameters, lambda, dt
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_launch_residual_calculation(
    const float *parameters,
    const float *times,
    const float2 *observations,
    float *residuals,
    float *weights,
    int num_observations,
    float lambda
) {
    dim3 block_size(256);
    dim3 grid_size((num_observations + block_size.x - 1) / block_size.x);
    
    cuda_calculate_residuals_kernel<<<grid_size, block_size>>>(
        parameters, times, observations, residuals, weights,
        num_observations, lambda
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_launch_error_estimation(
    const float *fitted_params,
    const float *times,
    const float *lag0_powers,
    const float *self_clutter,
    float *sigma_re,
    float *sigma_im,
    int num_observations,
    float noise_power,
    float nave,
    float lambda
) {
    dim3 block_size(256);
    dim3 grid_size((num_observations + block_size.x - 1) / block_size.x);
    
    cuda_estimate_errors_kernel<<<grid_size, block_size>>>(
        fitted_params, times, lag0_powers, self_clutter,
        sigma_re, sigma_im, num_observations,
        noise_power, nave, lambda
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return CUDA_SUCCESS;
}

cuda_error_t cuda_launch_convergence_check(
    const float *current_params,
    const float *previous_params,
    bool *converged_flags,
    int num_ranges,
    int num_params,
    float tolerance
) {
    dim3 block_size(256);
    dim3 grid_size((num_ranges + block_size.x - 1) / block_size.x);
    
    cuda_check_convergence_kernel<<<grid_size, block_size>>>(
        current_params, previous_params, converged_flags,
        num_ranges, num_params, tolerance
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return CUDA_SUCCESS;
}

} // extern "C"
