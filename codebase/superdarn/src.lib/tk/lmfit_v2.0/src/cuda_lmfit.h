#ifndef CUDA_LMFIT_H
#define CUDA_LMFIT_H

/**
 * @file cuda_lmfit.h
 * @brief CUDA-accelerated Levenberg-Marquardt fitting for SuperDARN data
 * 
 * This header provides CUDA implementations of the LMFIT v2.0 algorithms
 * using the standardized CUDA datatypes framework for optimal performance.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_datatypes.h"
#include "lmfit_preprocessing.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

// =============================================================================
// CUDA-COMPATIBLE DATA STRUCTURES
// =============================================================================

/**
 * @brief CUDA-compatible range node structure
 */
typedef struct {
    int range;
    float *SC_pow;              ///< Self-clutter power array (GPU memory)
    int refrc_idx;
    cuda_list_t acf_list;       ///< ACF data using CUDA lists
    cuda_list_t xcf_list;       ///< XCF data using CUDA lists  
    cuda_list_t phases_list;    ///< Phase data using CUDA lists
    cuda_list_t pwrs_list;      ///< Power data using CUDA lists
    cuda_list_t elev_list;      ///< Elevation data using CUDA lists
    cuda_list_t scpwr_list;     ///< Self-clutter power data using CUDA lists
    
    // Fitting results (GPU memory)
    cuda_memory_t l_acf_fit_mem;
    cuda_memory_t q_acf_fit_mem;
    cuda_memory_t l_xcf_fit_mem;
    cuda_memory_t q_xcf_fit_mem;
    
    // Previous values for convergence checking
    float prev_pow;
    float prev_phase;
    float prev_width;
} cuda_range_node_t;

/**
 * @brief CUDA-compatible power node structure
 */
typedef struct {
    int lag_num;
    float pwr;
    float sigma;
    float t;
} cuda_pwr_node_t;

/**
 * @brief CUDA-compatible ACF node structure
 */
typedef struct {
    int lag_num;
    float re;
    float im;
    float sigma_re;
    float sigma_im;
    float t;
} cuda_acf_node_t;

/**
 * @brief CUDA-compatible self-clutter node structure
 */
typedef struct {
    int lag_num;
    float clutter;
    float sigma;
    float t;
} cuda_sc_node_t;

/**
 * @brief CUDA-compatible phase node structure
 */
typedef struct {
    int lag_num;
    float phase;
    float sigma;
    float t;
} cuda_phase_node_t;

/**
 * @brief CUDA-compatible lag node structure
 */
typedef struct {
    int lag_num;
    int pulse_diff;
    int *pulses;
    int num_pulses;
} cuda_lag_node_t;

/**
 * @brief CUDA-compatible LMFIT data structure
 */
typedef struct {
    float P;        ///< Power
    float vel;      ///< Velocity
    float wid;      ///< Spectral width
    float phi0;     ///< Phase offset
    float sigma_P;  ///< Power error
    float sigma_vel;///< Velocity error
    float sigma_wid;///< Width error
    float sigma_phi0;///< Phase error
    float chi2;     ///< Chi-squared goodness of fit
    int ndf;        ///< Number of degrees of freedom
    bool converged; ///< Convergence flag
} cuda_lmfit_data_t;

// =============================================================================
// CUDA PREPROCESSING FUNCTIONS
// =============================================================================

/**
 * @brief Create a new CUDA range node
 * @param range Range gate number
 * @param fit_prms Fitting parameters
 * @param range_node Pointer to store the created range node
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_new_range_node(int range, FITPRMS *fit_prms, cuda_range_node_t **range_node);

/**
 * @brief Free a CUDA range node and all associated memory
 * @param range_node Range node to free
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_free_range_node(cuda_range_node_t *range_node);

/**
 * @brief Create a new CUDA power node
 * @param range Range gate number
 * @param lag Lag information
 * @param fit_prms Fitting parameters
 * @param pwr_node Pointer to store the created power node
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_new_pwr_node(int range, cuda_lag_node_t *lag, FITPRMS *fit_prms, cuda_pwr_node_t **pwr_node);

/**
 * @brief Create a new CUDA ACF node
 * @param range Range gate number
 * @param lag Lag information
 * @param fit_prms Fitting parameters
 * @param acf_node Pointer to store the created ACF node
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_new_acf_node(int range, cuda_lag_node_t *lag, FITPRMS *fit_prms, cuda_acf_node_t **acf_node);

/**
 * @brief Create a new CUDA XCF node
 * @param range Range gate number
 * @param lag Lag information
 * @param fit_prms Fitting parameters
 * @param xcf_node Pointer to store the created XCF node
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_new_xcf_node(int range, cuda_lag_node_t *lag, FITPRMS *fit_prms, cuda_acf_node_t **xcf_node);

/**
 * @brief Fill range list with CUDA range nodes
 * @param fit_prms Fitting parameters
 * @param ranges CUDA list to fill with range nodes
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_fill_range_list(FITPRMS *fit_prms, cuda_list_t *ranges);

/**
 * @brief Determine lags using CUDA acceleration
 * @param lags CUDA list to fill with lag information
 * @param fit_prms Fitting parameters
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_determine_lags(cuda_list_t *lags, FITPRMS *fit_prms);

/**
 * @brief Fill data lists for a specific range using CUDA
 * @param range_node Range node to process
 * @param lags List of lags
 * @param fit_prms Fitting parameters
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_fill_data_lists_for_range(cuda_range_node_t *range_node, 
                                           cuda_list_t *lags, FITPRMS *fit_prms);

/**
 * @brief Filter TX overlap using CUDA acceleration
 * @param ranges List of range nodes
 * @param lags List of lags
 * @param fit_prms Fitting parameters
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_filter_tx_overlap(cuda_list_t *ranges, cuda_list_t *lags, FITPRMS *fit_prms);

/**
 * @brief Estimate self-clutter using CUDA acceleration
 * @param ranges List of range nodes
 * @param fit_prms Fitting parameters
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_estimate_self_clutter(cuda_list_t *ranges, FITPRMS *fit_prms);

// =============================================================================
// CUDA LEAST SQUARES FITTING
// =============================================================================

/**
 * @brief Perform Levenberg-Marquardt fitting using CUDA
 * @param range_node Range node containing data to fit
 * @param fit_type Type of fit (linear or quadratic)
 * @param data_type Type of data (ACF or XCF)
 * @param result Pointer to store fitting results
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_lmfit_solve(cuda_range_node_t *range_node, 
                             int fit_type, int data_type, 
                             cuda_lmfit_data_t *result);

/**
 * @brief Batch process multiple ranges using CUDA
 * @param ranges List of range nodes to process
 * @param fit_prms Fitting parameters
 * @param profile Performance profiling structure (optional)
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_lmfit_batch_process(cuda_list_t *ranges, FITPRMS *fit_prms, 
                                     cuda_profile_t *profile);

/**
 * @brief Calculate Jacobian matrix on GPU
 * @param x Parameter vector
 * @param jacobian Output Jacobian matrix
 * @param data_points Number of data points
 * @param parameters Number of parameters
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_calculate_jacobian(cuda_matrix_t *x, cuda_matrix_t *jacobian, 
                                    size_t data_points, size_t parameters);

/**
 * @brief Solve normal equations using CUDA linear algebra
 * @param jacobian Jacobian matrix
 * @param residuals Residual vector
 * @param delta Output parameter update vector
 * @param lambda Levenberg-Marquardt damping parameter
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_solve_normal_equations(cuda_matrix_t *jacobian, 
                                        cuda_array_t *residuals,
                                        cuda_array_t *delta, float lambda);

// =============================================================================
// CUDA ERROR ESTIMATION
// =============================================================================

/**
 * @brief Estimate first-order errors using CUDA
 * @param ranges List of range nodes
 * @param fit_prms Fitting parameters
 * @param noise_pwr Noise power level
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_estimate_first_order_error(cuda_list_t *ranges, 
                                            FITPRMS *fit_prms, float noise_pwr);

/**
 * @brief Estimate real/imaginary errors using CUDA
 * @param ranges List of range nodes
 * @param fit_prms Fitting parameters
 * @param noise_pwr Noise power level
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_estimate_re_im_error(cuda_list_t *ranges, 
                                      FITPRMS *fit_prms, float noise_pwr);

// =============================================================================
// CUDA UTILITY FUNCTIONS
// =============================================================================

/**
 * @brief Convert CPU range list to CUDA format
 * @param cpu_ranges CPU-based linked list
 * @param cuda_ranges CUDA list to populate
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_convert_range_list(llist cpu_ranges, cuda_list_t *cuda_ranges);

/**
 * @brief Convert CUDA range list back to CPU format
 * @param cuda_ranges CUDA list
 * @param cpu_ranges CPU-based linked list to populate
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_convert_range_list_to_cpu(cuda_list_t *cuda_ranges, llist cpu_ranges);

/**
 * @brief Check convergence across all ranges
 * @param ranges List of range nodes
 * @param tolerance Convergence tolerance
 * @param converged Output convergence flag
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_check_convergence(cuda_list_t *ranges, float tolerance, bool *converged);

/**
 * @brief Calculate ACF cutoff power using CUDA
 * @param fit_prms Fitting parameters
 * @param cutoff_power Output cutoff power
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_acf_cutoff_power(FITPRMS *fit_prms, float *cutoff_power);

// =============================================================================
// COMPATIBILITY LAYER
// =============================================================================

/**
 * @brief Initialize CUDA LMFIT system
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_lmfit_init(void);

/**
 * @brief Cleanup CUDA LMFIT system
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_lmfit_cleanup(void);

/**
 * @brief Set CUDA LMFIT compute mode
 * @param mode Compute mode (AUTO, CPU, CUDA)
 * @return CUDA_SUCCESS on success, error code otherwise
 */
cuda_error_t cuda_lmfit_set_mode(compute_mode_t mode);

/**
 * @brief Get current CUDA LMFIT compute mode
 * @return Current compute mode
 */
compute_mode_t cuda_lmfit_get_mode(void);

/**
 * @brief Drop-in replacement for original LMFIT preprocessing
 * 
 * This function provides the same interface as the original CPU version
 * but automatically uses CUDA acceleration when available and beneficial.
 */
cuda_error_t cuda_lmfit_preprocess_ranges(llist ranges, llist lags, FITPRMS *fit_prms);

/**
 * @brief Drop-in replacement for original LMFIT fitting
 * 
 * This function provides the same interface as the original CPU version
 * but automatically uses CUDA acceleration when available and beneficial.
 */
cuda_error_t cuda_lmfit_fit_ranges(llist ranges, FITPRMS *fit_prms);

#ifdef __cplusplus
}
#endif

#endif // CUDA_LMFIT_H
