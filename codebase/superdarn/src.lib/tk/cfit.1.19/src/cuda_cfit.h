#ifndef CUDA_CFIT_H
#define CUDA_CFIT_H

#include "cuda_datatypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float power, velocity, width, phi0;
    int range, quality_flag;
} cuda_cfit_data_t;

cuda_error_t cuda_cfit_compress_data(cuda_array_t *input_data, cuda_array_t *output_data);
cuda_error_t cuda_cfit_process_scan(void *scan_data, void *params);

#ifdef __cplusplus
}
#endif

#endif
