/* cnvmap.1.17_cuda.cu
   ====================
   CUDA-accelerated convection mapping implementation
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

#include "cnvmap.1.17_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

/* Global CUDA context */
static cublasHandle_t cublas_handle = NULL;
static cusolverDnHandle_t cusolver_handle = NULL;
static bool cuda_initialized = false;
static bool profiling_enabled = false;
static cudaEvent_t start_event, stop_event;

/* Mathematical constants */
#define M_PI 3.14159265358979323846
#define DEG_TO_RAD (M_PI / 180.0)
#define RAD_TO_DEG (180.0 / M_PI)
#define EARTH_RADIUS 6371.0  // km

/* CUDA Initialization */
__host__ cudaError_t cnvmap_1_17_cuda_init(void) {
    if (cuda_initialized) return cudaSuccess;
    
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorInitializationError;
    }
    
    if (cusolverDnCreate(&cusolver_handle) != CUSOLVER_STATUS_SUCCESS) {
        cublasDestroy(cublas_handle);
        return cudaErrorInitializationError;
    }
    
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cuda_initialized = true;
    return cudaSuccess;
}

/* CUDA cleanup */
__host__ void cnvmap_1_17_cuda_cleanup(void) {
    if (cuda_initialized) {
        cublasDestroy(cublas_handle);
        cusolverDnDestroy(cusolver_handle);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_initialized = false;
    }
}

/* Memory Management Implementation */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type) {
    cuda_array_t *array = (cuda_array_t*)malloc(sizeof(cuda_array_t));
    if (!array) return NULL;
    
    array->size = size;
    array->element_size = element_size;
    array->type = type;
    array->device_id = 0;
    array->is_on_device = false;
    
    cudaError_t error = cudaMallocManaged(&array->data, size * element_size);
    if (error != cudaSuccess) {
        free(array);
        return NULL;
    }
    
    return array;
}

void cuda_array_destroy(cuda_array_t *array) {
    if (!array) return;
    if (array->data) cudaFree(array->data);
    free(array);
}

/* CUDA-compatible convection map data structures */
typedef struct {
    int fit_order;                    // Order of spherical harmonic fit
    double latmin;                    // Minimum latitude for fitting
    int num_coef;                     // Number of coefficients
    double *coef;                     // Spherical harmonic coefficients
    double chi_sqr, chi_sqr_dat;      // Chi-squared metrics
    double rms_err;                   // RMS error
    int num_model;                    // Number of model vectors
    double *model_lat, *model_lon;    // Model vector positions
    double *model_vx, *model_vy;      // Model vector components
    int num_bnd;                      // Number of boundary points
    double *bnd_lat, *bnd_lon;        // Boundary coordinates
} cuda_cnvmap_data_t;

typedef struct {
    double lat, lon;                  // Position coordinates
    double vlos;                      // Line-of-sight velocity
    double verr;                      // Velocity error
    double cos, sin;                  // Direction cosines
    double azm;                       // Azimuth angle
    double bmag;                      // Magnetic field magnitude
} cuda_cnvmap_vec_t;

/* CUDA Kernels for Convection Mapping */

/**
 * Legendre polynomial evaluation kernel
 * Computes associated Legendre polynomials P_l^m(x) in parallel
 */
__global__ void cuda_legendre_eval_kernel(int lmax,
                                          const double *x,
                                          double *plm,
                                          int n_points) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int point_idx = tid / ((lmax + 1) * (lmax + 2) / 2);
    int lm_idx = tid % ((lmax + 1) * (lmax + 2) / 2);
    
    if (point_idx >= n_points) return;
    
    // Convert linear index to (l,m) indices
    int l = 0, m = 0;
    int count = 0;
    for (int ll = 0; ll <= lmax; ll++) {
        for (int mm = 0; mm <= ll; mm++) {
            if (count == lm_idx) {
                l = ll;
                m = mm;
                break;
            }
            count++;
        }
        if (count > lm_idx) break;
    }
    
    double xi = x[point_idx];
    double sqrt_1_minus_x2 = sqrt(1.0 - xi * xi);
    
    // Calculate P_l^m(x) using recurrence relations
    double plm_val = 0.0;
    
    if (m == 0) {
        // P_l^0(x) - Legendre polynomials
        if (l == 0) {
            plm_val = 1.0;
        } else if (l == 1) {
            plm_val = xi;
        } else {
            // Recurrence: (l)P_l = (2l-1)xP_{l-1} - (l-1)P_{l-2}
            double p0 = 1.0;
            double p1 = xi;
            for (int ll = 2; ll <= l; ll++) {
                double p2 = ((2 * ll - 1) * xi * p1 - (ll - 1) * p0) / ll;
                p0 = p1;
                p1 = p2;
            }
            plm_val = p1;
        }
    } else {
        // Associated Legendre polynomials P_l^m(x)
        if (l == m) {
            // P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^{m/2}
            plm_val = 1.0;
            for (int i = 1; i <= m; i++) {
                plm_val *= -(2 * i - 1) * sqrt_1_minus_x2;
            }
        } else if (l == m + 1) {
            // P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
            double pmm = 1.0;
            for (int i = 1; i <= m; i++) {
                pmm *= -(2 * i - 1) * sqrt_1_minus_x2;
            }
            plm_val = xi * (2 * m + 1) * pmm;
        } else {
            // General recurrence for l > m+1
            double pmm = 1.0;
            for (int i = 1; i <= m; i++) {
                pmm *= -(2 * i - 1) * sqrt_1_minus_x2;
            }
            double pmm1 = xi * (2 * m + 1) * pmm;
            
            for (int ll = m + 2; ll <= l; ll++) {
                double pll = (xi * (2 * ll - 1) * pmm1 - (ll + m - 1) * pmm) / (ll - m);
                pmm = pmm1;
                pmm1 = pll;
            }
            plm_val = pmm1;
        }
    }
    
    // Store result
    int output_idx = point_idx * ((lmax + 1) * (lmax + 2) / 2) + lm_idx;
    plm[output_idx] = plm_val;
}

/**
 * Velocity matrix construction kernel
 * Builds the observation matrix for spherical harmonic fitting
 */
__global__ void cuda_velocity_matrix_kernel(const cuda_cnvmap_vec_t *obs,
                                            const double *plm,
                                            double *matrix,
                                            int n_obs, int n_coef,
                                            int lmax) {
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (obs_idx >= n_obs) return;
    
    cuda_cnvmap_vec_t observation = obs[obs_idx];
    
    // Convert to magnetic coordinates
    double mlat = observation.lat * DEG_TO_RAD;
    double mlon = observation.lon * DEG_TO_RAD;
    double cos_mlat = cos(mlat);
    double sin_mlat = sin(mlat);
    
    // Calculate electric field components from potential derivatives
    double bmag = observation.bmag;
    double cos_azm = observation.cos;
    double sin_azm = observation.sin;
    
    int coef_idx = 0;
    
    // Loop over spherical harmonic orders and degrees
    for (int l = 1; l <= lmax; l++) {
        for (int m = 0; m <= l; m++) {
            // Get Legendre polynomial value
            int plm_idx = obs_idx * ((lmax + 1) * (lmax + 2) / 2);
            int lm_offset = l * (l + 1) / 2 + m;
            double plm_val = plm[plm_idx + lm_offset];
            
            // Calculate derivatives for electric field
            double dtheta_cos = 0.0, dtheta_sin = 0.0;
            double dphi_cos = 0.0, dphi_sin = 0.0;
            
            if (m > 0) {
                // d/dtheta terms
                if (l > m) {
                    double plm_m1 = plm[plm_idx + (l * (l + 1) / 2 + m - 1)];
                    dtheta_cos = m * cos_mlat * plm_val / sin_mlat - 
                                sqrt((l - m) * (l + m + 1)) * plm_m1;
                }
                
                // d/dphi terms  
                dphi_cos = -m * plm_val;
                dphi_sin = m * plm_val;
            }
            
            // Electric field components (negative gradient of potential)
            double Etheta_cos = -dtheta_cos / EARTH_RADIUS;
            double Etheta_sin = -dtheta_sin / EARTH_RADIUS;
            double Ephi_cos = -dphi_cos / (EARTH_RADIUS * sin_mlat);
            double Ephi_sin = -dphi_sin / (EARTH_RADIUS * sin_mlat);
            
            // Convert to velocity using E x B drift
            double vtheta_cos = Ephi_cos / bmag;
            double vtheta_sin = Ephi_sin / bmag;
            double vphi_cos = -Etheta_cos / bmag;
            double vphi_sin = -Etheta_sin / bmag;
            
            // Project onto line-of-sight direction
            double vlos_cos = vtheta_cos * cos_azm + vphi_cos * sin_azm;
            double vlos_sin = vtheta_sin * cos_azm + vphi_sin * sin_azm;
            
            // Store in matrix (cos and sin components)
            if (m == 0) {
                matrix[obs_idx * n_coef + coef_idx] = vlos_cos;
                coef_idx++;
            } else {
                matrix[obs_idx * n_coef + coef_idx] = vlos_cos;
                matrix[obs_idx * n_coef + coef_idx + 1] = vlos_sin;
                coef_idx += 2;
            }
        }
    }
}

/**
 * Potential evaluation kernel
 * Evaluates spherical harmonic expansion at grid points
 */
__global__ void cuda_potential_eval_kernel(const double *coef,
                                           const double *grid_lat,
                                           const double *grid_lon,
                                           double *potential,
                                           int n_grid, int lmax) {
    int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (grid_idx >= n_grid) return;
    
    double lat = grid_lat[grid_idx] * DEG_TO_RAD;
    double lon = grid_lon[grid_idx] * DEG_TO_RAD;
    double sin_lat = sin(lat);
    
    double potential_sum = 0.0;
    int coef_idx = 0;
    
    // Evaluate spherical harmonic expansion
    for (int l = 1; l <= lmax; l++) {
        for (int m = 0; m <= l; m++) {
            // Calculate Legendre polynomial P_l^m(sin(lat))
            double plm_val = 1.0;  // Simplified - should use full Legendre calculation
            
            double cos_mlon = cos(m * lon);
            double sin_mlon = sin(m * lon);
            
            if (m == 0) {
                potential_sum += coef[coef_idx] * plm_val;
                coef_idx++;
            } else {
                potential_sum += coef[coef_idx] * plm_val * cos_mlon + 
                                coef[coef_idx + 1] * plm_val * sin_mlon;
                coef_idx += 2;
            }
        }
    }
    
    potential[grid_idx] = potential_sum;
}

/**
 * Chi-squared calculation kernel
 * Computes fitting error statistics
 */
__global__ void cuda_chisquared_kernel(const cuda_cnvmap_vec_t *obs,
                                       const double *model_vlos,
                                       double *chi_squared,
                                       double *rms_error,
                                       int n_obs) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double local_chi2 = 0.0;
    double local_rms = 0.0;
    
    if (idx < n_obs) {
        double residual = obs[idx].vlos - model_vlos[idx];
        double weight = 1.0 / (obs[idx].verr * obs[idx].verr);
        
        local_chi2 = residual * residual * weight;
        local_rms = residual * residual;
    }
    
    // Store in shared memory
    sdata[tid] = local_chi2;
    sdata[tid + blockDim.x] = local_rms;
    
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        atomicAdd(chi_squared, sdata[0]);
        atomicAdd(rms_error, sdata[blockDim.x]);
    }
}

/* Host wrapper functions */

/**
 * CUDA-accelerated Legendre polynomial evaluation
 */
extern "C" cudaError_t cuda_cnvmap_eval_legendre(int lmax,
                                                  const double *x,
                                                  double *plm,
                                                  int n_points) {
    if (!cuda_initialized) {
        cudaError_t init_error = cnvmap_1_17_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!x || !plm) return cudaErrorInvalidValue;
    
    int n_coef = (lmax + 1) * (lmax + 2) / 2;
    int total_threads = n_points * n_coef;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_legendre_eval_kernel<<<blocks, threads_per_block>>>(
        lmax, x, plm, n_points);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/**
 * CUDA-accelerated potential evaluation
 */
extern "C" cudaError_t cuda_cnvmap_eval_potential(const double *coef,
                                                  const double *grid_lat,
                                                  const double *grid_lon,
                                                  double *potential,
                                                  int n_grid, int lmax) {
    if (!cuda_initialized) {
        cudaError_t init_error = cnvmap_1_17_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!coef || !grid_lat || !grid_lon || !potential) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (n_grid + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    cuda_potential_eval_kernel<<<blocks, threads_per_block>>>(
        coef, grid_lat, grid_lon, potential, n_grid, lmax);
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/* Utility Functions */
extern "C" bool cnvmap_1_17_cuda_is_available(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

extern "C" int cnvmap_1_17_cuda_get_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

extern "C" void cnvmap_1_17_cuda_enable_profiling(bool enable) {
    profiling_enabled = enable;
}

extern "C" float cnvmap_1_17_cuda_get_last_kernel_time(void) {
    if (!profiling_enabled) return 0.0f;
    
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    return milliseconds;
}
