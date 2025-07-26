#ifndef CUDA_LMFIT_KERNELS_H
#define CUDA_LMFIT_KERNELS_H

/**
 * @file cuda_lmfit_kernels.h
 * @brief CUDA kernel declarations for LMFIT v2.0 acceleration
 * 
 * This header declares the CUDA kernel launch functions for the
 * Levenberg-Marquardt fitting algorithm acceleration.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_lmfit.h"
#include <cuda_runtime.h>

// =============================================================================
// CUDA KERNEL LAUNCH FUNCTIONS
// =============================================================================

/**
 * @brief Launch power calculation kernel
 * @param acf_data Input ACF data (complex)
 * @param power_output Output power values
 * @param sigma_output Output sigma values
 * @param lag_nums Lag numbers
 * @param lag_times Lag times
 * @param num_ranges Number of range gates
 * @param num_lags Number of lags
 * @param nave Number of averages
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_launch_power_calculation(
    const float2 *acf_data,
    float *power_output,
    float *sigma_output,
    const int *lag_nums,
    const float *lag_times,
    int num_ranges,
    int num_lags,
    float nave
);

/**
 * @brief Launch ACF node creation kernel
 * @param acf_data Input ACF data (complex)
 * @param acf_nodes Output ACF nodes
 * @param lag_nums Lag numbers
 * @param lag_times Lag times
 * @param num_ranges Number of range gates
 * @param num_lags Number of lags
 * @param nave Number of averages
 * @param is_xcf Whether this is XCF data
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_launch_acf_creation(
    const float2 *acf_data,
    cuda_acf_node_t *acf_nodes,
    const int *lag_nums,
    const float *lag_times,
    int num_ranges,
    int num_lags,
    float nave,
    bool is_xcf
);

/**
 * @brief Launch self-clutter estimation kernel
 * @param lag0_powers Lag-0 powers for all ranges
 * @param self_clutter Output self-clutter estimates
 * @param range_indices Range indices
 * @param num_ranges Number of ranges
 * @param num_lags Number of lags
 * @param clutter_factor Clutter estimation factor
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_launch_self_clutter_estimation(
    const float *lag0_powers,
    float *self_clutter,
    const int *range_indices,
    int num_ranges,
    int num_lags,
    float clutter_factor
);

/**
 * @brief Launch Jacobian calculation kernel
 * @param parameters Current parameter estimates
 * @param times Time values for each lag
 * @param observations Observed ACF values
 * @param jacobian Output Jacobian matrix
 * @param num_observations Number of data points
 * @param num_parameters Number of parameters
 * @param lambda Radar wavelength
 * @param dt Time step
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_launch_jacobian_calculation(
    const float *parameters,
    const float *times,
    const float2 *observations,
    float *jacobian,
    int num_observations,
    int num_parameters,
    float lambda,
    float dt
);

/**
 * @brief Launch residual calculation kernel
 * @param parameters Current parameter estimates
 * @param times Time values for each lag
 * @param observations Observed ACF values
 * @param residuals Output residuals
 * @param weights Data weights
 * @param num_observations Number of data points
 * @param lambda Radar wavelength
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_launch_residual_calculation(
    const float *parameters,
    const float *times,
    const float2 *observations,
    float *residuals,
    float *weights,
    int num_observations,
    float lambda
);

/**
 * @brief Launch error estimation kernel
 * @param fitted_params Fitted parameters
 * @param times Time values
 * @param lag0_powers Lag-0 powers
 * @param self_clutter Self-clutter estimates
 * @param sigma_re Output real part errors
 * @param sigma_im Output imaginary part errors
 * @param num_observations Number of observations
 * @param noise_power Noise power level
 * @param nave Number of averages
 * @param lambda Radar wavelength
 * @return CUDA_SUCCESS on success, error code otherwise
 */
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
);

/**
 * @brief Launch convergence checking kernel
 * @param current_params Current parameter estimates
 * @param previous_params Previous parameter estimates
 * @param converged_flags Output convergence flags
 * @param num_ranges Number of ranges
 * @param num_params Number of parameters per range
 * @param tolerance Convergence tolerance
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_launch_convergence_check(
    const float *current_params,
    const float *previous_params,
    bool *converged_flags,
    int num_ranges,
    int num_params,
    float tolerance
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_LMFIT_KERNELS_H
