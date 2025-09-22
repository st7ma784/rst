/* cnvmap.1.17_cuda.h
   ===================
   CUDA-accelerated convection mapping library
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

#ifndef CNVMAP_1_17_CUDA_H
#define CNVMAP_1_17_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA data structures */
typedef enum {
    CUDA_R_32F = 0,
    CUDA_C_32F = 1,
    CUDA_R_64F = 2,
    CUDA_C_64F = 3
} cudaDataType_t;

typedef struct {
    void *data;
    size_t size;
    size_t element_size;
    cudaDataType_t type;
    int device_id;
    bool is_on_device;
} cuda_array_t;

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

/* Initialization and cleanup */
cudaError_t cnvmap_1_17_cuda_init(void);
void cnvmap_1_17_cuda_cleanup(void);

/* Memory management */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type);
void cuda_array_destroy(cuda_array_t *array);

/* Core CUDA functions for convection mapping */
cudaError_t cuda_cnvmap_eval_legendre(int lmax,
                                      const double *x,
                                      double *plm,
                                      int n_points);

cudaError_t cuda_cnvmap_build_velocity_matrix(const cuda_cnvmap_vec_t *obs,
                                              double *matrix,
                                              int n_obs, int lmax);

cudaError_t cuda_cnvmap_eval_potential(const double *coef,
                                       const double *grid_lat,
                                       const double *grid_lon,
                                       double *potential,
                                       int n_grid, int lmax);

cudaError_t cuda_cnvmap_eval_velocity(const double *coef,
                                      const double *grid_lat,
                                      const double *grid_lon,
                                      const double *bmag,
                                      double *vel_x, double *vel_y,
                                      int n_grid, int lmax);

cudaError_t cuda_cnvmap_calculate_chisquared(const cuda_cnvmap_vec_t *obs,
                                             const double *model_vlos,
                                             double *chi_squared,
                                             double *rms_error,
                                             int n_obs);

/* High-level processing functions */
cudaError_t cuda_cnvmap_fit_spherical_harmonics(const cuda_cnvmap_vec_t *obs,
                                                int n_obs, int fit_order,
                                                cuda_cnvmap_data_t *result);

cudaError_t cuda_cnvmap_solve_least_squares(const double *matrix,
                                            const double *rhs,
                                            double *solution,
                                            int m, int n);

/* Backward compatibility bridge functions */
int CnvMapEvalLegendre(int Lmax, double *x, int n, double *plm);
int CnvMapFitVector(struct GridGVec *obs, int n_obs, 
                   struct CnvMapData *fit, int fit_order);
int CnvMapSolvePotential(struct CnvMapData *fit, double *grid_lat, 
                        double *grid_lon, double *potential, int n_grid);
int CnvMapSolveVelocity(struct CnvMapData *fit, double *grid_lat, 
                       double *grid_lon, double *vel_x, double *vel_y, int n_grid);

/* Memory management helpers */
cuda_cnvmap_data_t* cuda_cnvmap_data_create(int fit_order);
void cuda_cnvmap_data_destroy(cuda_cnvmap_data_t *cnvmap_data);
cuda_cnvmap_vec_t* cuda_cnvmap_vec_create(int n_obs);
void cuda_cnvmap_vec_destroy(cuda_cnvmap_vec_t *obs);

/* Utility functions */
bool cnvmap_1_17_cuda_is_available(void);
int cnvmap_1_17_cuda_get_device_count(void);

/* Performance monitoring */
void cnvmap_1_17_cuda_enable_profiling(bool enable);
float cnvmap_1_17_cuda_get_last_kernel_time(void);

/* Data validation helpers */
bool cuda_cnvmap_validate_data(const cuda_cnvmap_data_t *cnvmap_data);
bool cuda_cnvmap_validate_observations(const cuda_cnvmap_vec_t *obs, int n_obs);

#ifdef __cplusplus
}
#endif

#endif /* CNVMAP_1_17_CUDA_H */
