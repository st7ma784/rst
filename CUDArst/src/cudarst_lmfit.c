/**
 * @file cudarst_lmfit.c
 * @brief LMFIT v2.0 compatible interface implementation
 * 
 * Provides backward-compatible LMFIT processing with CUDA acceleration
 */

#include "cudarst.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* CUDA kernel declarations (implemented in .cu file) */
#ifdef __NVCC__
extern "C" {
    int cuda_lmfit_solve(float *y, float *x, float *sig, int ndata,
                         float *a, int ma, float **covar, float *chisq,
                         int max_iter, float tolerance);
}
#endif

/* CPU fallback implementation */
static cudarst_error_t cpu_lmfit_solve(cudarst_lmfit_data_t *data,
                                       const cudarst_lmfit_config_t *config);

cudarst_error_t cudarst_lmfit_solve(cudarst_lmfit_data_t *data,
                                    const cudarst_lmfit_config_t *config)
{
    if (!data || !config) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    if (!data->y || !data->x || !data->a || data->ndata <= 0 || data->ma <= 0) {
        return CUDARST_ERROR_INVALID_ARGS;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    cudarst_error_t result = CUDARST_ERROR_PROCESSING_FAILED;
    
    /* Attempt CUDA processing if available and requested */
    if (config->use_cuda && cudarst_is_cuda_available()) {
#ifdef __NVCC__
        struct timespec cuda_start, cuda_end;
        clock_gettime(CLOCK_MONOTONIC, &cuda_start);
        
        int cuda_result = cuda_lmfit_solve(
            data->y, data->x, data->sig, data->ndata,
            data->a, data->ma, data->covar, &data->chisq,
            config->max_iterations, config->tolerance
        );
        
        clock_gettime(CLOCK_MONOTONIC, &cuda_end);
        double cuda_time = (cuda_end.tv_sec - cuda_start.tv_sec) * 1000.0 + 
                          (cuda_end.tv_nsec - cuda_start.tv_nsec) / 1000000.0;
        
        /* Update performance counters */
        cudarst_performance_t perf;
        cudarst_get_performance(&perf);
        perf.cuda_time_ms += cuda_time;
        perf.cuda_used = true;
        
        if (cuda_result == 0) {
            result = CUDARST_SUCCESS;
        } else {
            /* CUDA processing failed, fall back to CPU */
            fprintf(stderr, "CUDArst: CUDA LMFIT processing failed, falling back to CPU\n");
            result = cpu_lmfit_solve(data, config);
        }
#endif
    } else {
        /* CPU-only processing */
        result = cpu_lmfit_solve(data, config);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    /* Update performance counters */
    cudarst_performance_t perf;
    cudarst_get_performance(&perf);
    perf.total_time_ms += total_time;
    
    return result;
}

cudarst_lmfit_data_t* cudarst_lmfit_data_alloc(int ndata, int ma)
{
    if (ndata <= 0 || ma <= 0) {
        return NULL;
    }
    
    cudarst_lmfit_data_t *data = malloc(sizeof(cudarst_lmfit_data_t));
    if (!data) return NULL;
    
    data->ndata = ndata;
    data->ma = ma;
    data->chisq = 0.0f;
    data->alamda = 0.001f;  /* Initial lambda value */
    
    /* Allocate arrays */
    data->y = cudarst_malloc(ndata * sizeof(float));
    data->x = cudarst_malloc(ndata * sizeof(float));
    data->sig = cudarst_malloc(ndata * sizeof(float));
    data->a = cudarst_malloc(ma * sizeof(float));
    data->alpha = cudarst_malloc(ma * ma * sizeof(float));
    
    /* Allocate 2D covariance matrix */
    data->covar = malloc(ma * sizeof(float*));
    if (data->covar) {
        float *covar_data = cudarst_malloc(ma * ma * sizeof(float));
        if (covar_data) {
            for (int i = 0; i < ma; i++) {
                data->covar[i] = covar_data + i * ma;
            }
        } else {
            free(data->covar);
            data->covar = NULL;
        }
    }
    
    /* Check allocation success */
    if (!data->y || !data->x || !data->sig || !data->a || !data->alpha || !data->covar) {
        cudarst_lmfit_data_free(data);
        return NULL;
    }
    
    /* Initialize arrays */
    memset(data->y, 0, ndata * sizeof(float));
    memset(data->x, 0, ndata * sizeof(float));
    
    /* Initialize standard deviations to 1.0 */
    for (int i = 0; i < ndata; i++) {
        data->sig[i] = 1.0f;
    }
    
    /* Initialize parameters */
    memset(data->a, 0, ma * sizeof(float));
    memset(data->alpha, 0, ma * ma * sizeof(float));
    
    /* Initialize covariance matrix */
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < ma; j++) {
            data->covar[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    return data;
}

void cudarst_lmfit_data_free(cudarst_lmfit_data_t *data)
{
    if (!data) return;
    
    cudarst_free(data->y);
    cudarst_free(data->x);
    cudarst_free(data->sig);
    cudarst_free(data->a);
    cudarst_free(data->alpha);
    
    if (data->covar) {
        if (data->covar[0]) {
            cudarst_free(data->covar[0]); /* Free the contiguous data block */
        }
        free(data->covar); /* Free the pointer array */
    }
    
    free(data);
}

/* CPU fallback implementation using simplified Levenberg-Marquardt */
static cudarst_error_t cpu_lmfit_solve(cudarst_lmfit_data_t *data,
                                       const cudarst_lmfit_config_t *config)
{
    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);
    
    /* Simplified Levenberg-Marquardt implementation */
    const int max_iter = config->max_iterations > 0 ? config->max_iterations : 100;
    const float tolerance = config->tolerance > 0.0f ? config->tolerance : 1e-6f;
    
    float lambda = data->alamda;
    float old_chisq = 1e10f;
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* Calculate chi-squared and derivatives */
        float chisq = 0.0f;
        
        /* Reset alpha matrix and beta vector */
        memset(data->alpha, 0, data->ma * data->ma * sizeof(float));
        float *beta = alloca(data->ma * sizeof(float));
        memset(beta, 0, data->ma * sizeof(float));
        
        /* Accumulate chi-squared and build matrices */
        for (int i = 0; i < data->ndata; i++) {
            /* Calculate model value and derivatives */
            float y_model = 0.0f;
            float *dyda = alloca(data->ma * sizeof(float));
            
            /* Simple polynomial model: y = a[0] + a[1]*x + a[2]*x^2 + ... */
            if (config->func) {
                y_model = config->func(data->x[i], data->a, data->ma);
                if (config->funcs) {
                    config->funcs(data->x[i], data->a, dyda, data->ma);
                }
            } else {
                /* Default polynomial model */
                float x_power = 1.0f;
                for (int j = 0; j < data->ma; j++) {
                    y_model += data->a[j] * x_power;
                    dyda[j] = x_power;
                    x_power *= data->x[i];
                }
            }
            
            float dy = data->y[i] - y_model;
            float sig2_inv = 1.0f / (data->sig[i] * data->sig[i]);
            
            chisq += dy * dy * sig2_inv;
            
            /* Build alpha matrix and beta vector */
            for (int j = 0; j < data->ma; j++) {
                float wt = dyda[j] * sig2_inv;
                for (int k = 0; k <= j; k++) {
                    data->alpha[j * data->ma + k] += wt * dyda[k];
                }
                beta[j] += dy * wt;
            }
        }
        
        /* Fill symmetric elements */
        for (int j = 1; j < data->ma; j++) {
            for (int k = 0; k < j; k++) {
                data->alpha[k * data->ma + j] = data->alpha[j * data->ma + k];
            }
        }
        
        data->chisq = chisq;
        
        /* Check for convergence */
        if (iter > 0 && fabs(old_chisq - chisq) < tolerance * chisq) {
            break;
        }
        
        /* Augment diagonal elements */
        for (int j = 0; j < data->ma; j++) {
            data->alpha[j * data->ma + j] *= (1.0f + lambda);
        }
        
        /* Solve for parameter increments using Gaussian elimination */
        float *da = alloca(data->ma * sizeof(float));
        memcpy(da, beta, data->ma * sizeof(float));
        
        /* Simple Gaussian elimination (for small matrices) */
        for (int i = 0; i < data->ma; i++) {
            /* Find pivot */
            int pivot = i;
            float max_val = fabs(data->alpha[i * data->ma + i]);
            for (int j = i + 1; j < data->ma; j++) {
                if (fabs(data->alpha[j * data->ma + i]) > max_val) {
                    max_val = fabs(data->alpha[j * data->ma + i]);
                    pivot = j;
                }
            }
            
            /* Swap rows if needed */
            if (pivot != i) {
                for (int k = 0; k < data->ma; k++) {
                    float temp = data->alpha[i * data->ma + k];
                    data->alpha[i * data->ma + k] = data->alpha[pivot * data->ma + k];
                    data->alpha[pivot * data->ma + k] = temp;
                }
                float temp = da[i];
                da[i] = da[pivot];
                da[pivot] = temp;
            }
            
            /* Eliminate column */
            for (int j = i + 1; j < data->ma; j++) {
                if (data->alpha[i * data->ma + i] != 0.0f) {
                    float factor = data->alpha[j * data->ma + i] / data->alpha[i * data->ma + i];
                    for (int k = i; k < data->ma; k++) {
                        data->alpha[j * data->ma + k] -= factor * data->alpha[i * data->ma + k];
                    }
                    da[j] -= factor * da[i];
                }
            }
        }
        
        /* Back substitution */
        for (int i = data->ma - 1; i >= 0; i--) {
            if (data->alpha[i * data->ma + i] != 0.0f) {
                da[i] /= data->alpha[i * data->ma + i];
                for (int j = i - 1; j >= 0; j--) {
                    da[j] -= data->alpha[j * data->ma + i] * da[i];
                }
            }
        }
        
        /* Update parameters */
        for (int j = 0; j < data->ma; j++) {
            data->a[j] += da[j];
        }
        
        /* Adjust lambda for next iteration */
        if (chisq < old_chisq) {
            lambda *= 0.1f;  /* Decrease lambda (more Gauss-Newton) */
        } else {
            lambda *= 10.0f; /* Increase lambda (more gradient descent) */
        }
        
        old_chisq = chisq;
    }
    
    /* Calculate final covariance matrix */
    /* (This is a simplified version - in practice would need matrix inversion) */
    for (int i = 0; i < data->ma; i++) {
        for (int j = 0; j < data->ma; j++) {
            data->covar[i][j] = (i == j) ? 1.0f / data->alpha[i * data->ma + i] : 0.0f;
        }
    }
    
    data->alamda = lambda;
    
    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
    double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 + 
                     (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000000.0;
    
    /* Update performance counters */
    cudarst_performance_t perf;
    cudarst_get_performance(&perf);
    perf.cpu_fallback_ms += cpu_time;
    
    return CUDARST_SUCCESS;
}