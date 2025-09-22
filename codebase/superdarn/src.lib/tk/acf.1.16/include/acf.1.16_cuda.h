/* acf.1.16_cuda.h
   ================
   CUDA-accelerated ACF processing library
   Author: R.J.Barnes (CUDA implementation)
*/

/*
  Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory
 
This file is part of the Radar Software Toolkit (RST).

RST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Modifications:
*/ 

#ifndef ACF_1_16_CUDA_H
#define ACF_1_16_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA data structures */
#ifndef CUDA_DATA_TYPE_DEFINED
#define CUDA_DATA_TYPE_DEFINED
typedef enum {
    CUDA_R_32F_CUSTOM = 0,
    CUDA_C_32F_CUSTOM = 1,
    CUDA_R_64F_CUSTOM = 2,
    CUDA_C_64F_CUSTOM = 3
} cudaDataType_t_custom;
#endif

typedef struct {
    void *data;
    size_t size;
    size_t element_size;
    cudaDataType_t type;
    int device_id;
    bool is_on_device;
} cuda_array_t;

/* CUDA-compatible ACF data structures */
typedef struct {
    int16_t *inbuf;     // Input I/Q samples
    float *acfbuf;      // ACF output buffer (complex, interleaved)
    float *xcfbuf;      // XCF output buffer (complex, interleaved)
    float *pwr0;        // Lag-0 power array
    int nrang;          // Number of range gates
    int mplgs;          // Number of lags
    int nave;           // Number of averages
    int *lagfr;         // Lag to first range table
    int *smsep;         // Sample separation
    int *pat;           // Pulse pattern
    int mpinc;          // Multi-pulse increment
    float attn;         // Attenuation factor
    bool xcf_enabled;   // Cross-correlation enabled
} cuda_acf_data_t;

typedef struct {
    float real, imag;   // Complex number
} cuda_complex_t;

/* Initialization and cleanup */
cudaError_t acf_1_16_cuda_init(void);
void acf_1_16_cuda_cleanup(void);

/* Memory management */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type);
void cuda_array_destroy(cuda_array_t *array);

/* Core CUDA functions */
cudaError_t cuda_acf_calculate(const int16_t *inbuf,
                              float *acfbuf,
                              float *xcfbuf,
                              const int *lagfr,
                              const int *smsep,
                              const int *pat,
                              int nrang, int mplgs,
                              int mpinc, int nave,
                              int offset, bool xcf_enabled);

cudaError_t cuda_acf_power(const float *acfbuf,
                          float *pwr0,
                          int nrang, int mplgs);

cudaError_t cuda_acf_average(float *acfbuf,
                            float *xcfbuf,
                            int nrang, int mplgs,
                            int nave, bool xcf_enabled);

cudaError_t cuda_acf_normalize(float *acfbuf,
                              float *xcfbuf,
                              int nrang, int mplgs,
                              float attn, bool xcf_enabled);

cudaError_t cuda_acf_badlag_detect(const float *acfbuf,
                                  bool *badlag_mask,
                                  int nrang, int mplgs,
                                  float noise_threshold);

cudaError_t cuda_acf_sum_power(const int16_t *inbuf,
                              float *power_sum,
                              float *power_max,
                              int nrang, int nave,
                              int offset, int smsep);

cudaError_t cuda_acf_magnitude(const float *acfbuf,
                              float *magnitude,
                              int nrang, int mplgs);

/* High-level processing pipeline */
cudaError_t cuda_acf_process_pipeline(cuda_acf_data_t *acf_data,
                                     bool calculate_power,
                                     bool detect_badlags,
                                     float noise_threshold);

/* Backward compatibility bridge functions for original ACF API */
void ACFCalculate(int16_t *inbuf, float *acfbuf, float *xcfbuf,
                  int *lagfr, int *smsep, int *pat,
                  int nrang, int mplgs, int mpinc, int nave, int offset);

void ACFAverage(float *acfbuf, float *xcfbuf, int nrang, int mplgs, int nave);

void ACFNormalize(float *pwr0, float *acfd, float *xcfd,
                  int nrang, int mplgs, float atten);

int ACFBadLagZero(struct TSGprm *prm, int mplgs, int *lagtable[2]);

void ACFSumPower(int16_t *inbuf, float *power, int nrang, int nave,
                 int offset, int smsep);

void ACFSumProduct(int16_t *inbuf, float *sum, int nrang, int nave,
                   int offset, int smsep);

/* Utility functions */
bool acf_1_16_cuda_is_available(void);
int acf_1_16_cuda_get_device_count(void);

/* Performance monitoring */
void acf_1_16_cuda_enable_profiling(bool enable);
float acf_1_16_cuda_get_last_kernel_time(void);

#ifdef __cplusplus
}
#endif

#endif /* ACF_1_16_CUDA_H */
