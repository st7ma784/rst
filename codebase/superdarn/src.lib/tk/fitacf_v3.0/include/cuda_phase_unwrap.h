#ifndef CUDA_PHASE_UNWRAP_H
#define CUDA_PHASE_UNWRAP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Unwraps phase data using CUDA acceleration
 * 
 * @param phase_in Input phase data (wrapped, in radians)
 * @param phase_out Output phase data (unwrapped)
 * @param quality_metric Optional output quality metric array (one per range)
 * @param num_ranges Number of range gates
 * @param num_lags Number of lags per range gate
 * @param phase_threshold Threshold for phase jumps (typically ~π)
 */
void cuda_phase_unwrap(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
);

/**
 * @brief Unwraps phase data using OpenMP multi-threading
 * 
 * @param phase_in Input phase data (wrapped, in radians)
 * @param phase_out Output phase data (unwrapped)
 * @param quality_metric Optional output quality metric array (one per range)
 * @param num_ranges Number of range gates
 * @param num_lags Number of lags per range gate
 * @param phase_threshold Threshold for phase jumps (typically ~π)
 */
void omp_phase_unwrap(
    const float* phase_in,
    float* phase_out,
    float* quality_metric,
    int num_ranges,
    int num_lags,
    float phase_threshold
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PHASE_UNWRAP_H
