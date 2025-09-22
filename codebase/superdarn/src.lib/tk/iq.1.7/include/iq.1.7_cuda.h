/* iq.1.7_cuda.h
   ==============
   CUDA-accelerated IQ data processing library
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

#ifndef IQ_1_7_CUDA_H
#define IQ_1_7_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA-compatible IQ data structures */
typedef struct {
    int major, minor;       // Revision numbers
} cuda_iq_revision_t;

typedef struct {
    cuda_iq_revision_t revision;
    int chnnum;            // Number of channels
    int smpnum;            // Number of samples per sequence
    int skpnum;            // Number of samples to skip
    int seqnum;            // Number of sequences
    int tbadtr;            // Total bad transmit samples
    long *tv_sec;          // Time seconds array
    long *tv_nsec;         // Time nanoseconds array
    int *atten;            // Attenuation array
    float *noise;          // Noise array
    int *offset;           // Offset array
    int *size;             // Size array
    int *badtr;            // Bad transmit array
} cuda_iq_data_t;

/* Statistics types for IQ analysis */
typedef enum {
    IQ_STATS_MEAN_POWER = 0,
    IQ_STATS_VARIANCE = 1,
    IQ_STATS_I_MEAN = 2,
    IQ_STATS_Q_MEAN = 3
} iq_stats_type_t;

/* Initialization and cleanup */
cudaError_t iq_1_7_cuda_init(void);
void iq_1_7_cuda_cleanup(void);

/* Core CUDA functions */
cudaError_t cuda_iq_convert_time(const double *input_time,
                                long *tv_sec,
                                long *tv_nsec,
                                int num_samples);

cudaError_t cuda_iq_copy_array(const int *input,
                              int *output,
                              bool *valid_mask,
                              int num_elements,
                              int min_val, int max_val);

cudaError_t cuda_iq_copy_float_array(const float *input,
                                    float *output,
                                    bool *valid_mask,
                                    int num_elements);

cudaError_t cuda_iq_flatten(const cuda_iq_data_t *iq_data,
                           char *buffer,
                           int *offsets,
                           int buffer_size);

cudaError_t cuda_iq_expand(const char *buffer,
                          cuda_iq_data_t *iq_data,
                          const int *offsets,
                          int num_sequences);

cudaError_t cuda_iq_encode(const cuda_iq_data_t *iq_data,
                          float *encoded_data,
                          int *metadata,
                          int total_samples);

cudaError_t cuda_iq_decode(const float *encoded_data,
                          const int *metadata,
                          cuda_iq_data_t *iq_data,
                          int total_samples);

cudaError_t cuda_iq_detect_badtr(const int16_t *iq_samples,
                                bool *badtr_mask,
                                int num_samples,
                                int16_t threshold_min,
                                int16_t threshold_max);

cudaError_t cuda_iq_calculate_statistics(const int16_t *iq_samples,
                                        float *statistics,
                                        int num_samples,
                                        int stats_type);

/* High-level processing functions */
cudaError_t cuda_iq_process_sequences(cuda_iq_data_t *iq_data,
                                     const int16_t *raw_samples,
                                     bool enable_badtr_detection,
                                     int16_t saturation_threshold);

/* Backward compatibility bridge functions for original IQ API */
struct IQ* IQMake(void);
void IQFree(struct IQ *ptr);
struct IQ* IQCopy(struct IQ *ptr);
int IQSetTime(struct IQ *ptr, int seq, struct timespec *tval);
int IQSetAtten(struct IQ *ptr, int seq, int atten);
int IQSetNoise(struct IQ *ptr, int seq, float noise);
int IQSetOffset(struct IQ *ptr, int seq, int offset);
int IQSetSize(struct IQ *ptr, int seq, int size);
int IQSetBadTR(struct IQ *ptr, int seq, int badtr);
char *IQFlatten(struct IQ *ptr, int num, int *size);
int IQExpand(struct IQ **ptr, int num, void *buffer);
int IQDecode(struct DataMap *ptr, struct IQ **iq);
int IQEncode(struct DataMap *ptr, struct IQ *iq);

/* Memory management helpers */
cuda_iq_data_t* cuda_iq_data_create(int seqnum);
void cuda_iq_data_destroy(cuda_iq_data_t *iq_data);
cudaError_t cuda_iq_data_allocate_arrays(cuda_iq_data_t *iq_data);
void cuda_iq_data_free_arrays(cuda_iq_data_t *iq_data);

/* Utility functions */
bool iq_1_7_cuda_is_available(void);
int iq_1_7_cuda_get_device_count(void);

/* Performance monitoring */
void iq_1_7_cuda_enable_profiling(bool enable);
float iq_1_7_cuda_get_last_kernel_time(void);

/* Data validation helpers */
bool cuda_iq_validate_data(const cuda_iq_data_t *iq_data);
cudaError_t cuda_iq_check_memory_requirements(const cuda_iq_data_t *iq_data,
                                             size_t *required_bytes);

/* CUDA-accelerated versions of original functions */
int IQSetTime_CUDA(struct IQ *ptr, int seq, struct timespec *tval);
int IQSetAtten_CUDA(struct IQ *ptr, int seq, int atten);
int IQSetNoise_CUDA(struct IQ *ptr, int seq, float noise);
int IQSetOffset_CUDA(struct IQ *ptr, int seq, int offset);
int IQSetSize_CUDA(struct IQ *ptr, int seq, int size);
int IQSetBadTR_CUDA(struct IQ *ptr, int seq, int badtr);
char *IQFlatten_CUDA(struct IQ *ptr, int num, int *size);
int IQExpand_CUDA(struct IQ **ptr, int num, void *buffer);

/* Control functions */
void iq_1_7_cuda_enable_acceleration(bool enable);
bool iq_1_7_cuda_acceleration_enabled(void);
void iq_1_7_cuda_set_debug_mode(bool enable);

/* Conversion functions */
int iq_to_cuda_iq(const struct IQ *iq, cuda_iq_data_t **cuda_iq);
int cuda_iq_to_iq(const cuda_iq_data_t *cuda_iq, struct IQ *iq);

#ifdef __cplusplus
}
#endif

#endif /* IQ_1_7_CUDA_H */