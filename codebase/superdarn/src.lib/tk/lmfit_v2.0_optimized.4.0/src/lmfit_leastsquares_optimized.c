/*
 Non-Linear Least squares fitting using Levenburg-Marquardt 
 Algorithm implements in C (cmpfit) - OPTIMIZED VERSION

 Copyright (c) 2016 University of Saskatchewan
 Adapted by: Ashton Reimer
 From code by: Keith Kotyk
 Optimized by: SuperDARN Optimization Framework v4.0

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
     Added OpenMP parallelization and SIMD vectorization for performance
*/

#include "lmfit_leastsquares.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include "mpfit.h"
#include "lmfit_structures.h"

#define ITMAX 100 
#define EPS 3.0e-7 
#define FPMIN 1.0e-30
#define CACHE_LINE_SIZE 64
#define ALIGN_SIZE 32

struct vars_struct {
  double *x;
  double *y;
  double *ey;
  double lambda;
};

// Performance counters
static double total_model_time = 0.0;
static double total_simd_time = 0.0;
static int model_calls = 0;

/**
Returns a pointer to a new FITDATA structure
*/
LMFITDATA *new_lmfit_data(){
    LMFITDATA *new_lmfit_data;

    new_lmfit_data = malloc(sizeof(*new_lmfit_data));
    new_lmfit_data->P = 0.0;
    new_lmfit_data->wid = 0.0;
    new_lmfit_data->vel = 0.0;
    new_lmfit_data->phi0 = 0.0;
    new_lmfit_data->sigma_2_P = 0.0;
    new_lmfit_data->sigma_2_wid = 0.0;
    new_lmfit_data->sigma_2_vel = 0.0;
    new_lmfit_data->sigma_2_phi0 = 0.0;
    new_lmfit_data->chi_2 = 0.0;

    return new_lmfit_data;
}

void free_lmfit_data(LMFITDATA *lmfit_data){
    if(lmfit_data != NULL){
        free(lmfit_data);
    }
}

/**
prints the contents of a LMFITDATA structure
*/
void print_lmfit_data(LMFITDATA *fit_data, FILE* fp){
    fprintf(fp,"P: %e\n",fit_data->P);
    fprintf(fp,"wid: %e\n",fit_data->wid);
    fprintf(fp,"vel: %e\n",fit_data->vel);
    fprintf(fp,"phi0: %e\n",fit_data->phi0);
    fprintf(fp,"sigma_2_P: %e\n",fit_data->sigma_2_P);
    fprintf(fp,"sigma_2_wid: %e\n",fit_data->sigma_2_wid);
    fprintf(fp,"sigma_2_vel: %e\n",fit_data->sigma_2_vel);
    fprintf(fp,"sigma_2_phi0: %e\n",fit_data->sigma_2_phi0);
    fprintf(fp,"chi_2: %f\n",fit_data->chi_2);
}

// SIMD-optimized exponential model calculation
static inline void calculate_model_simd(double *x, double *y_real, double *y_imag,
                                       double *ey_real, double *ey_imag,
                                       double *deviates, double **derivs,
                                       double P, double wid, double vel, double lambda,
                                       int num_lags) {
    
    const __m256d pi_vec = _mm256_set1_pd(M_PI);
    const __m256d two_pi_vec = _mm256_set1_pd(2.0 * M_PI);
    const __m256d four_pi_vec = _mm256_set1_pd(4.0 * M_PI);
    const __m256d P_vec = _mm256_set1_pd(P);
    const __m256d wid_vec = _mm256_set1_pd(wid);
    const __m256d vel_vec = _mm256_set1_pd(vel);
    const __m256d lambda_vec = _mm256_set1_pd(lambda);
    const __m256d neg_two_pi_lambda = _mm256_div_pd(_mm256_mul_pd(two_pi_vec, _mm256_set1_pd(-1.0)), lambda_vec);
    const __m256d four_pi_lambda = _mm256_div_pd(four_pi_vec, lambda_vec);
    
    // Process 4 lags at a time with AVX2
    int vec_len = num_lags - (num_lags % 4);
    
    for (int i = 0; i < vec_len; i += 4) {
        // Load time values
        __m256d t_vec = _mm256_loadu_pd(&x[i]);
        
        // Load data values
        __m256d acf_real_vec = _mm256_loadu_pd(&y_real[i]);
        __m256d acf_imag_vec = _mm256_loadu_pd(&y_imag[i]);
        __m256d sigma_real_vec = _mm256_loadu_pd(&ey_real[i]);
        __m256d sigma_imag_vec = _mm256_loadu_pd(&ey_imag[i]);
        
        // Calculate exponential: exp(-2*pi*wid*t/lambda)
        __m256d exp_arg = _mm256_mul_pd(_mm256_mul_pd(neg_two_pi_lambda, wid_vec), t_vec);
        
        // Use fast approximation for exp() - replace with _mm256_exp_pd if available
        __m256d exponential;
        double exp_vals[4];
        _mm256_storeu_pd(exp_vals, exp_arg);
        for (int j = 0; j < 4; j++) {
            exp_vals[j] = exp(exp_vals[j]);
        }
        exponential = _mm256_loadu_pd(exp_vals);
        
        // Calculate cosine and sine: cos/sin(4*pi*vel*t/lambda)
        __m256d trig_arg = _mm256_mul_pd(_mm256_mul_pd(four_pi_lambda, vel_vec), t_vec);
        
        __m256d cosine, sine;
        double trig_vals[4];
        _mm256_storeu_pd(trig_vals, trig_arg);
        double cos_vals[4], sin_vals[4];
        for (int j = 0; j < 4; j++) {
            cos_vals[j] = cos(trig_vals[j]);
            sin_vals[j] = sin(trig_vals[j]);
        }
        cosine = _mm256_loadu_pd(cos_vals);
        sine = _mm256_loadu_pd(sin_vals);
        
        // Calculate model values
        __m256d P_exp = _mm256_mul_pd(P_vec, exponential);
        __m256d model_real = _mm256_mul_pd(P_exp, cosine);
        __m256d model_imag = _mm256_mul_pd(P_exp, sine);
        
        // Calculate deviates: (data - model) / sigma
        __m256d dev_real = _mm256_div_pd(_mm256_sub_pd(acf_real_vec, model_real), sigma_real_vec);
        __m256d dev_imag = _mm256_div_pd(_mm256_sub_pd(acf_imag_vec, model_imag), sigma_imag_vec);
        
        // Store deviates
        _mm256_storeu_pd(&deviates[i], dev_real);
        _mm256_storeu_pd(&deviates[i + num_lags], dev_imag);
        
        // Calculate derivatives if needed
        if (derivs) {
            // dP derivatives
            __m256d dP_real = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), 
                                          _mm256_mul_pd(exponential, cosine)), sigma_real_vec);
            __m256d dP_imag = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), 
                                          _mm256_mul_pd(exponential, sine)), sigma_imag_vec);
            
            _mm256_storeu_pd(&derivs[0][i], dP_real);
            _mm256_storeu_pd(&derivs[0][i + num_lags], dP_imag);
            
            // dwid derivatives
            __m256d dwid_factor = _mm256_mul_pd(_mm256_mul_pd(neg_two_pi_lambda, t_vec), P_exp);
            __m256d dwid_real = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), 
                                            _mm256_mul_pd(dwid_factor, cosine)), sigma_real_vec);
            __m256d dwid_imag = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), 
                                            _mm256_mul_pd(dwid_factor, sine)), sigma_imag_vec);
            
            _mm256_storeu_pd(&derivs[1][i], dwid_real);
            _mm256_storeu_pd(&derivs[1][i + num_lags], dwid_imag);
            
            // dvel derivatives
            __m256d dvel_factor = _mm256_mul_pd(_mm256_mul_pd(four_pi_lambda, t_vec), P_exp);
            __m256d dvel_real = _mm256_div_pd(_mm256_mul_pd(dvel_factor, sine), sigma_real_vec);
            __m256d dvel_imag = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), 
                                            _mm256_mul_pd(dvel_factor, cosine)), sigma_imag_vec);
            
            _mm256_storeu_pd(&derivs[2][i], dvel_real);
            _mm256_storeu_pd(&derivs[2][i + num_lags], dvel_imag);
        }
    }
    
    // Handle remaining elements
    for (int i = vec_len; i < num_lags; i++) {
        double t = x[i];
        double exponential = exp(-2.0 * M_PI * wid * t / lambda);
        double cosine = cos(4 * M_PI * vel * t / lambda);
        double sine = sin(4 * M_PI * vel * t / lambda);
        
        // Real component
        deviates[i] = (y_real[i] - P * exponential * cosine) / ey_real[i];
        
        // Imaginary component  
        deviates[i + num_lags] = (y_imag[i] - P * exponential * sine) / ey_imag[i];
        
        if (derivs) {
            // Real derivatives
            derivs[0][i] = -exponential * cosine / ey_real[i];
            derivs[1][i] = -(-2.0 * M_PI * t / lambda) * P * exponential * cosine / ey_real[i];
            derivs[2][i] = (4 * M_PI * t / lambda) * P * exponential * sine / ey_real[i];
            
            // Imaginary derivatives
            derivs[0][i + num_lags] = -exponential * sine / ey_imag[i];
            derivs[1][i + num_lags] = -(-2.0 * M_PI * t / lambda) * P * exponential * sine / ey_imag[i];
            derivs[2][i + num_lags] = -(4 * M_PI * t / lambda) * P * exponential * cosine / ey_imag[i];
        }
    }
}

/*function to calculate residuals for MPFIT - OPTIMIZED VERSION*/
int exp_acf_model_optimized(int m, int n, double *params, double *deviates, double **derivs, void *private)
{
    model_calls++;
    double start_time = omp_get_wtime();
    
    struct vars_struct *v = (struct vars_struct *) private;
    double *x = v->x;
    double *y = v->y;
    double *ey = v->ey;
    double lambda = v->lambda;

    /* Parameters we are fitting for */
    double P = params[0];     /* lag0 power */
    double wid = params[1];   /* spectral width */
    double vel = params[2];   /* Doppler velocity */

    int num_lags = m / 2;
    
    // Split data arrays for better cache locality
    double *y_real = y;
    double *y_imag = y + num_lags;
    double *ey_real = ey;
    double *ey_imag = ey + num_lags;
    
    double simd_start = omp_get_wtime();
    
    // Use SIMD-optimized calculation
    calculate_model_simd(x, y_real, y_imag, ey_real, ey_imag, deviates, derivs,
                        P, wid, vel, lambda, num_lags);
    
    total_simd_time += omp_get_wtime() - simd_start;
    total_model_time += omp_get_wtime() - start_time;

    return 0;
}

// Batch processing for multiple initial velocities with OpenMP
void lmfit_acf_optimized(LMFITDATA *fit_data, llist data, double lambda, int mpinc, int confidence, int model)
{
    const static int num_init_vel = 30;  /* Number of initial velocities to try */
    int i;
    double min_chi = 1e200;
    int num_lags = llist_size(data);
    ACFNODE* data_node;

    double nyquist_velocity = lambda / (4.0 * (double)(mpinc) * 1.e-6);
    double v_step = (nyquist_velocity - (-nyquist_velocity)) / ((double)(num_init_vel) - 1);

    // Arrays for parallel processing results
    double chi2s[num_init_vel];
    double pows[num_init_vel], wids[num_init_vel], vels[num_init_vel];
    double pows_e[num_init_vel], wids_e[num_init_vel], vels_e[num_init_vel];
    int statuses[num_init_vel];

    /* set up data array */
    struct vars_struct *lmdata = malloc(sizeof(struct vars_struct));
    lmdata->x = (double*)aligned_alloc(ALIGN_SIZE, num_lags * 2 * sizeof(double));
    lmdata->y = (double*)aligned_alloc(ALIGN_SIZE, 2 * num_lags * sizeof(double));
    lmdata->ey = (double*)aligned_alloc(ALIGN_SIZE, num_lags * 2 * sizeof(double));
    lmdata->lambda = lambda;

    /* Re-structure the data for mpfit to use with better cache locality */
    llist_reset_iter(data);
    i = 0;
    do {
        llist_get_iter(data, (void**)&data_node);
        lmdata->x[i] = data_node->t;
        lmdata->x[i + num_lags] = data_node->t;
        lmdata->y[i] = data_node->re;
        lmdata->y[i + num_lags] = data_node->im;
        lmdata->ey[i] = data_node->sigma_re;
        lmdata->ey[i + num_lags] = data_node->sigma_im;
        i++;
    } while(llist_go_next(data) != LLIST_END_OF_LIST);

    // Parallel processing of different initial velocities
    #pragma omp parallel for schedule(dynamic) if(num_init_vel > 8)
    for (i = 0; i < num_init_vel; i++) {
        mp_par params_info[3];
        mp_result result;
        mp_config config;
        double best_fit_params[3];
        double paramerrors[3];

        memset(&params_info[0], 0, sizeof(params_info));
        memset(&config, 0, sizeof(config));
        memset(&result, 0, sizeof(result));
        result.xerror = paramerrors;

        /*limit values to prevent fit from going to +- inf and breaking*/
        params_info[0].limited[0] = 1;
        params_info[0].limits[0] = 0;
        params_info[0].side = 3;
        params_info[0].deriv_debug = 0;

        params_info[1].limited[0] = 1;
        params_info[1].limits[0] = -100.0;
        params_info[1].limited[1] = 0;
        params_info[1].side = 3;
        params_info[1].deriv_debug = 0;

        params_info[2].limited[0] = 1;
        params_info[2].limits[0] = -nyquist_velocity / 2.;
        params_info[2].limited[1] = 1;
        params_info[2].limits[1] = nyquist_velocity / 2.;
        params_info[2].side = 3;
        params_info[2].deriv_debug = 0;

        /* CONFIGURE LMFIT */
        config.maxiter = 200;
        config.maxfev = 200;
        config.ftol = .0001;
        config.gtol = .0001;
        config.nofinitecheck = 0;

        /* Starting guess */
        best_fit_params[0] = 10000.0;
        best_fit_params[1] = 200.0;
        best_fit_params[2] = -nyquist_velocity / 2. + i * v_step;

        /*run a single-component fit using optimized model*/
        int status = mpfit(exp_acf_model_optimized, num_lags * 2, 3, best_fit_params, 
                          params_info, &config, (void *)lmdata, &result);

        // Store results
        chi2s[i] = result.bestnorm;
        pows[i] = best_fit_params[0];
        wids[i] = best_fit_params[1];
        vels[i] = best_fit_params[2];
        pows_e[i] = paramerrors[0];
        wids_e[i] = paramerrors[1];
        vels_e[i] = paramerrors[2];
        statuses[i] = status;
    }

    // Find best fit result
    int best_idx = 0;
    for (i = 1; i < num_init_vel; i++) {
        if (chi2s[i] < min_chi) {
            min_chi = chi2s[i];
            best_idx = i;
        }
    }

    // Store best results
    fit_data->P = pows[best_idx];
    fit_data->wid = wids[best_idx];
    fit_data->vel = vels[best_idx];
    fit_data->sigma_2_P = pows_e[best_idx] * pows_e[best_idx];
    fit_data->sigma_2_wid = wids_e[best_idx] * wids_e[best_idx];
    fit_data->sigma_2_vel = vels_e[best_idx] * vels_e[best_idx];
    fit_data->chi_2 = min_chi;

    // Cleanup
    free(lmdata->x);
    free(lmdata->y);
    free(lmdata->ey);
    free(lmdata);
}

// Wrapper function for backward compatibility
int exp_acf_model(int m, int n, double *params, double *deviates, double **derivs, void *private) {
    return exp_acf_model_optimized(m, n, params, deviates, derivs, private);
}

void lmfit_acf(LMFITDATA *fit_data, llist data, double lambda, int mpinc, int confidence, int model) {
    lmfit_acf_optimized(fit_data, data, lambda, mpinc, confidence, model);
}

// Performance monitoring functions
void get_lmfit_performance_stats(double *total_time, double *simd_time, int *calls) {
    *total_time = total_model_time;
    *simd_time = total_simd_time;
    *calls = model_calls;
}

void reset_lmfit_performance_stats() {
    total_model_time = 0.0;
    total_simd_time = 0.0;
    model_calls = 0;
}
