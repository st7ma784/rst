#ifndef CUDA_ACF_H
#define CUDA_ACF_H

#include "cuda_datatypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float real, imag;
    float power, phase;
    int lag_num;
} cuda_acf_data_t;

cuda_error_t cuda_acf_calculate_power(cuda_array_t *acf_data, cuda_array_t *power_output);
cuda_error_t cuda_acf_process_ranges(void *ranges, void *params);

#ifdef __cplusplus
}
#endif

#endif
