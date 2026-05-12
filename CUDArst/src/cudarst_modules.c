/**
 * @file cudarst_modules.c
 * @brief Implementation of CUDArst module wrapper functions
 * 
 * Provides C wrapper functions for all CUDA-accelerated modules
 */

#include "cudarst.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* External CUDA kernel function declarations */
extern int cuda_acf_process(const int16_t *inbuf, float *acfbuf, float *xcfbuf,
                           const int *lagfr, const int *smsep, const int *pat,
                           int nrang, int mplgs, int mpinc, int nave,
                           int offset, bool xcf_enabled);

extern int cuda_iq_process_time_series(const double *input_time, const float *iq_data,
                                      long *tv_sec, long *tv_nsec, int16_t *encoded_iq,
                                      int num_samples, float scale_factor);

extern int cuda_cnvmap_spherical_harmonic_fit(const double *theta, const double *phi,
                                              const double *v_los, int n_points,
                                              double *coefficients, int lmax);

extern int cuda_grid_interpolate_data(const float *x_data, const float *y_data,
                                     const float *values, int n_points,
                                     const float *grid_x, const float *grid_y,
                                     float *grid_values, int grid_nx, int grid_ny,
                                     float cell_size);

/* ====================================================================
 * ACF v1.16 Interface Implementation
 * ====================================================================*/

cudarst_error_t cudarst_acf_process(const int16_t *inbuf, float *acfbuf, float *xcfbuf,
                                   const int *lagfr, const int *smsep, const int *pat,
                                   int nrang, int mplgs, int mpinc, int nave,
                                   int offset, bool xcf_enabled)
{
    if (!inbuf || !acfbuf || !lagfr || !smsep || !pat) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    if (nrang <= 0 || mplgs <= 0 || mpinc <= 0 || nave <= 0) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    /* Try CUDA processing first */
    if (cudarst_is_cuda_available()) {
        int result = cuda_acf_process(inbuf, acfbuf, xcfbuf, lagfr, smsep, pat,
                                     nrang, mplgs, mpinc, nave, offset, xcf_enabled);
        if (result == 0) {
            return CUDARST_SUCCESS;
        }
    }
    
    /* CPU fallback: compute ACF from inbuf IQ samples.
     * inbuf layout: [nave * nrang * mpinc] complex int16 pairs (I,Q interleaved).
     * lagfr[lag*2], lagfr[lag*2+1] are the two pulse indices for each lag. */
    fprintf(stderr, "ACF: CUDA unavailable — using CPU fallback\n");

    for (int r = 0; r < nrang; r++) {
        for (int lag = 0; lag < mplgs; lag++) {
            int out_idx = (r * mplgs + lag) * 2;
            int p1 = lagfr[lag * 2];
            int p2 = lagfr[lag * 2 + 1];

            float real_sum = 0.0f, imag_sum = 0.0f;
            float xreal_sum = 0.0f, ximag_sum = 0.0f;
            int   valid = 0;

            for (int n = 0; n < nave; n++) {
                /* Each pulse sequence has mpinc samples per range */
                int base = (n * nrang * mpinc + r * mpinc);
                int i1 = base + p1, i2 = base + p2;

                /* Bounds check */
                if (i1 < 0 || i2 < 0) continue;

                float I1 = (float)inbuf[i1 * 2];
                float Q1 = (float)inbuf[i1 * 2 + 1];
                float I2 = (float)inbuf[i2 * 2];
                float Q2 = (float)inbuf[i2 * 2 + 1];

                /* ACF: s1 * conj(s2) */
                real_sum += I1 * I2 + Q1 * Q2;
                imag_sum += Q1 * I2 - I1 * Q2;

                if (xcf_enabled && xcfbuf) {
                    /* XCF uses different sign convention for cross-correlation */
                    xreal_sum += I1 * I2 - Q1 * Q2;
                    ximag_sum += I1 * Q2 + Q1 * I2;
                }
                valid++;
            }

            float scale = (valid > 0) ? 1.0f / valid : 0.0f;
            acfbuf[out_idx]     = real_sum * scale;
            acfbuf[out_idx + 1] = imag_sum * scale;

            if (xcf_enabled && xcfbuf) {
                xcfbuf[out_idx]     = xreal_sum * scale;
                xcfbuf[out_idx + 1] = ximag_sum * scale;
            }
        }
    }

    return CUDARST_SUCCESS;
}

/* ====================================================================
 * IQ v1.7 Interface Implementation
 * ====================================================================*/

cudarst_error_t cudarst_iq_process_time_series(const double *input_time, const float *iq_data,
                                              long *tv_sec, long *tv_nsec, int16_t *encoded_iq,
                                              int num_samples, float scale_factor)
{
    if (!input_time || !iq_data || !tv_sec || !tv_nsec || !encoded_iq) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    if (num_samples <= 0 || scale_factor <= 0.0f) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    /* Try CUDA processing first */
    if (cudarst_is_cuda_available()) {
        int result = cuda_iq_process_time_series(input_time, iq_data, tv_sec, tv_nsec,
                                                encoded_iq, num_samples, scale_factor);
        if (result == 0) {
            return CUDARST_SUCCESS;
        }
    }
    
    /* CPU fallback implementation */
    printf("IQ: Falling back to CPU implementation\n");
    
    for (int i = 0; i < num_samples; i++) {
        /* Convert time */
        tv_sec[i] = (long)input_time[i];
        tv_nsec[i] = (long)((input_time[i] - (double)tv_sec[i]) * 1e9);
        
        /* Encode I/Q data */
        float i_val = iq_data[i * 2] * scale_factor;
        float q_val = iq_data[i * 2 + 1] * scale_factor;
        
        /* Clamp to int16 range */
        i_val = (i_val > 32767.0f) ? 32767.0f : ((i_val < -32768.0f) ? -32768.0f : i_val);
        q_val = (q_val > 32767.0f) ? 32767.0f : ((q_val < -32768.0f) ? -32768.0f : q_val);
        
        encoded_iq[i * 2] = (int16_t)i_val;
        encoded_iq[i * 2 + 1] = (int16_t)q_val;
    }
    
    return CUDARST_SUCCESS;
}

/* ====================================================================
 * CNVMAP v1.17 Interface Implementation
 * ====================================================================*/

cudarst_error_t cudarst_cnvmap_spherical_harmonic_fit(const double *theta, const double *phi,
                                                     const double *v_los, int n_points,
                                                     double *coefficients, int lmax)
{
    if (!theta || !phi || !v_los || !coefficients) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    if (n_points <= 0 || lmax < 1) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    /* Try CUDA processing first */
    if (cudarst_is_cuda_available()) {
        int result = cuda_cnvmap_spherical_harmonic_fit(theta, phi, v_los, n_points,
                                                       coefficients, lmax);
        if (result == 0) {
            return CUDARST_SUCCESS;
        }
    }
    
    /* CPU fallback: delegate to the same host-side least-squares solver used
     * by the CUDA path (implemented at the bottom of cudarst_kernels.cu).
     * That function already handles cos(colatitude) conversion and normal equations. */
    fprintf(stderr, "CNVMAP: CUDA unavailable — using CPU least-squares solver\n");
    extern int cuda_cnvmap_spherical_harmonic_fit(const double*, const double*,
                                                  const double*, int, double*, int);
    int result = cuda_cnvmap_spherical_harmonic_fit(theta, phi, v_los, n_points,
                                                    coefficients, lmax);
    return (result == 0) ? CUDARST_SUCCESS : CUDARST_ERROR_PROCESSING_FAILED;
}

/* ====================================================================
 * GRID v1.24 Interface Implementation
 * ====================================================================*/

cudarst_error_t cudarst_grid_interpolate_data(const float *x_data, const float *y_data,
                                             const float *values, int n_points,
                                             const float *grid_x, const float *grid_y,
                                             float *grid_values, int grid_nx, int grid_ny,
                                             float cell_size)
{
    if (!x_data || !y_data || !values || !grid_x || !grid_y || !grid_values) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    if (n_points <= 0 || grid_nx <= 0 || grid_ny <= 0 || cell_size <= 0.0f) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    /* Try CUDA processing first */
    if (cudarst_is_cuda_available()) {
        int result = cuda_grid_interpolate_data(x_data, y_data, values, n_points,
                                               grid_x, grid_y, grid_values,
                                               grid_nx, grid_ny, cell_size);
        if (result == 0) {
            return CUDARST_SUCCESS;
        }
    }
    
    /* CPU fallback implementation */
    printf("GRID: Falling back to CPU implementation\n");
    
    /* Simple nearest neighbor interpolation */
    for (int gy = 0; gy < grid_ny; gy++) {
        for (int gx = 0; gx < grid_nx; gx++) {
            int grid_idx = gy * grid_nx + gx;
            float gx_pos = grid_x[gx];
            float gy_pos = grid_y[gy];
            
            /* Find nearest data point */
            float min_dist = 1e30f;
            float best_value = 0.0f;
            
            for (int i = 0; i < n_points; i++) {
                float dx = x_data[i] - gx_pos;
                float dy = y_data[i] - gy_pos;
                float dist = dx * dx + dy * dy;
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_value = values[i];
                }
            }
            
            grid_values[grid_idx] = best_value;
        }
    }
    
    return CUDARST_SUCCESS;
}