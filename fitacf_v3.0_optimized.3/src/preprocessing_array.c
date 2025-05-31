/*
 * Array-based preprocessing functions for SuperDARN FitACF v3.0
 * 
 * This file implements array-based versions of the preprocessing functions
 * to enable massive parallelization with OpenMP and CUDA.
 * 
 * Copyright (c) 2025 SuperDARN Refactoring Project
 * Author: GitHub Copilot Assistant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fit_structures_array.h"
#include "preprocessing.h"
#include "leastsquares.h"

/* Array-based preprocessing functions */

int Fill_Range_List_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays) {
    if (!fit_prms || !arrays) return -1;
    
    /* Initialize the range array */
    arrays->num_ranges = fit_prms->nrang;
    
    /* Set up range nodes */
    for (int i = 0; i < fit_prms->nrang; i++) {
        RANGENODE_ARRAY *rng = &arrays->ranges[i];
        rng->range = i;
        rng->refrc_idx = 1.0; /* Default refractive index */
        
        /* Allocate CRI array */
        rng->CRI = malloc(sizeof(double) * fit_prms->mplgs);
        if (!rng->CRI) return -1;
        
        /* Initialize CRI values */
        for (int j = 0; j < fit_prms->mplgs; j++) {
            rng->CRI[j] = 0.0;
        }
    }
    
    return 0;
}

int Fill_Data_Lists_For_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx) {
    if (!fit_prms || !arrays || range_idx < 0 || range_idx >= arrays->num_ranges) {
        return -1;
    }
    
    RANGENODE_ARRAY *rng = &arrays->ranges[range_idx];
    
    /* Process each lag */
    for (int lag = 0; lag < fit_prms->mplgs; lag++) {
        if (fit_prms->lag[0][lag] == -1) continue; /* Invalid lag */
        
        /* Calculate lag time */
        double lag_time = fit_prms->lag[0][lag] * fit_prms->mpinc * 1.0e-6;
        
        /* Get ACF value for this range and lag */
        double complex acf_val = fit_prms->acfd[range_idx][lag];
        double power = cabs(acf_val);
        double phase = carg(acf_val);
        
        /* Calculate alpha-2 value */
        double alpha_2 = Calculate_Alpha_2(fit_prms, range_idx, lag);
        
        /* Add power data if above threshold */
        if (power > fit_prms->noise_threshold) {
            double ln_pwr = log(power);
            double sigma = Calculate_Power_Sigma(fit_prms, range_idx, lag, power);
            
            add_power_data(arrays, range_idx, ln_pwr, lag_time, sigma, lag, alpha_2);
        }
        
        /* Add phase data if power is sufficient */
        if (power > fit_prms->noise_threshold * 2.0) {
            double sigma = Calculate_Phase_Sigma(fit_prms, range_idx, lag, power);
            
            add_phase_data(arrays, range_idx, phase, lag_time, sigma, lag, alpha_2);
        }
        
        /* Add alpha data */
        add_alpha_data(arrays, range_idx, lag, alpha_2);
        
        /* Handle XCF data for elevation if available */
        if (fit_prms->xcf && fit_prms->xcfd) {
            double complex xcf_val = fit_prms->xcfd[range_idx][lag];
            double xcf_power = cabs(xcf_val);
            
            if (xcf_power > fit_prms->noise_threshold) {
                double elev = Calculate_Elevation(fit_prms, acf_val, xcf_val);
                double elev_sigma = Calculate_Elevation_Sigma(fit_prms, range_idx, lag, xcf_power);
                
                add_elev_data(arrays, range_idx, elev, lag_time, elev_sigma, lag);
            }
        }
    }
    
    return 0;
}

/* Parallel preprocessing function */
void Parallel_Preprocessing_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays) {
    if (!fit_prms || !arrays) return;
    
    /* First, set up the range list */
    Fill_Range_List_Array(fit_prms, arrays);
    
#ifdef _OPENMP
    if (fit_prms->num_threads > 1) {
        omp_set_num_threads(fit_prms->num_threads);
        
        #pragma omp parallel for schedule(dynamic)
        for (int range_idx = 0; range_idx < fit_prms->nrang; range_idx++) {
            Fill_Data_Lists_For_Range_Array(fit_prms, arrays, range_idx);
        }
    } else {
#endif
        /* Sequential processing */
        for (int range_idx = 0; range_idx < fit_prms->nrang; range_idx++) {
            Fill_Data_Lists_For_Range_Array(fit_prms, arrays, range_idx);
        }
#ifdef _OPENMP
    }
#endif
    
    /* Populate matrices for efficient access */
    populate_matrices(arrays);
    
    /* Mark valid ranges */
    mark_valid_ranges(arrays, fit_prms->noise_threshold);
}

/* Array-based filtering functions */

int Filter_TX_Overlap_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays) {
    if (!fit_prms || !arrays) return -1;
    
    int filtered_count = 0;
    
#ifdef _OPENMP
    if (fit_prms->num_threads > 1) {
        #pragma omp parallel for reduction(+:filtered_count)
        for (int range_idx = 0; range_idx < arrays->num_ranges; range_idx++) {
            filtered_count += Filter_TX_Overlap_Range_Array(fit_prms, arrays, range_idx);
        }
    } else {
#endif
        for (int range_idx = 0; range_idx < arrays->num_ranges; range_idx++) {
            filtered_count += Filter_TX_Overlap_Range_Array(fit_prms, arrays, range_idx);
        }
#ifdef _OPENMP
    }
#endif
    
    return filtered_count;
}

int Filter_TX_Overlap_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx) {
    if (!fit_prms || !arrays || range_idx < 0 || range_idx >= arrays->num_ranges) {
        return 0;
    }
    
    RANGENODE_ARRAY *rng = &arrays->ranges[range_idx];
    int filtered_count = 0;
    
    /* Calculate TX overlap window */
    double tx_end_time = fit_prms->txpl * 1.0e-6;
    double range_time = (fit_prms->frang + range_idx * fit_prms->rsep) * 2.0 / 3.0e8;
    
    if (range_time < tx_end_time) {
        /* This range overlaps with TX pulse - filter out early lags */
        double overlap_lag_time = tx_end_time - range_time;
        
        /* Filter power data */
        int new_power_count = 0;
        for (int i = 0; i < rng->pwrs.count; i++) {
            if (rng->pwrs.t[i] > overlap_lag_time) {
                if (new_power_count != i) {
                    rng->pwrs.ln_pwr[new_power_count] = rng->pwrs.ln_pwr[i];
                    rng->pwrs.t[new_power_count] = rng->pwrs.t[i];
                    rng->pwrs.sigma[new_power_count] = rng->pwrs.sigma[i];
                    rng->pwrs.lag_idx[new_power_count] = rng->pwrs.lag_idx[i];
                    rng->pwrs.alpha_2[new_power_count] = rng->pwrs.alpha_2[i];
                }
                new_power_count++;
            } else {
                filtered_count++;
            }
        }
        rng->pwrs.count = new_power_count;
        
        /* Filter phase data */
        int new_phase_count = 0;
        for (int i = 0; i < rng->phases.count; i++) {
            if (rng->phases.t[i] > overlap_lag_time) {
                if (new_phase_count != i) {
                    rng->phases.phi[new_phase_count] = rng->phases.phi[i];
                    rng->phases.t[new_phase_count] = rng->phases.t[i];
                    rng->phases.sigma[new_phase_count] = rng->phases.sigma[i];
                    rng->phases.lag_idx[new_phase_count] = rng->phases.lag_idx[i];
                    rng->phases.alpha_2[new_phase_count] = rng->phases.alpha_2[i];
                }
                new_phase_count++;
            } else {
                filtered_count++;
            }
        }
        rng->phases.count = new_phase_count;
    }
    
    return filtered_count;
}

int Find_CRI_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays) {
    if (!fit_prms || !arrays) return -1;
    
#ifdef _OPENMP
    if (fit_prms->num_threads > 1) {
        #pragma omp parallel for
        for (int range_idx = 0; range_idx < arrays->num_ranges; range_idx++) {
            Find_CRI_Range_Array(fit_prms, arrays, range_idx);
        }
    } else {
#endif
        for (int range_idx = 0; range_idx < arrays->num_ranges; range_idx++) {
            Find_CRI_Range_Array(fit_prms, arrays, range_idx);
        }
#ifdef _OPENMP
    }
#endif
    
    return 0;
}

int Find_CRI_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx) {
    if (!fit_prms || !arrays || range_idx < 0 || range_idx >= arrays->num_ranges) {
        return -1;
    }
    
    RANGENODE_ARRAY *rng = &arrays->ranges[range_idx];
    
    /* Calculate CRI (Complex Radio Interferometry) values for each lag */
    for (int lag = 0; lag < fit_prms->mplgs; lag++) {
        if (fit_prms->lag[0][lag] == -1) {
            rng->CRI[lag] = 0.0;
            continue;
        }
        
        /* Get main array and interferometer array data */
        double complex main_acf = fit_prms->acfd[range_idx][lag];
        double complex xcf = 0.0;
        
        if (fit_prms->xcfd) {
            xcf = fit_prms->xcfd[range_idx][lag];
        }
        
        /* Calculate cross-correlation phase difference */
        if (cabs(main_acf) > 0 && cabs(xcf) > 0) {
            double phase_diff = carg(xcf) - carg(main_acf);
            
            /* Wrap phase difference to [-π, π] */
            while (phase_diff > M_PI) phase_diff -= 2.0 * M_PI;
            while (phase_diff < -M_PI) phase_diff += 2.0 * M_PI;
            
            rng->CRI[lag] = phase_diff;
        } else {
            rng->CRI[lag] = 0.0;
        }
    }
    
    return 0;
}

/* Helper calculation functions */

double Calculate_Alpha_2(FITPRMS_ARRAY *fit_prms, int range_idx, int lag) {
    if (!fit_prms) return 1.0;
    
    /* Alpha-2 calculation based on radar parameters */
    double tau = fit_prms->mpinc * 1.0e-6; /* pulse increment in seconds */
    double lag_time = fit_prms->lag[0][lag] * tau;
    
    /* Simple alpha-2 model - can be enhanced */
    double alpha_2 = 1.0 + 0.1 * lag_time;
    
    return alpha_2;
}

double Calculate_Power_Sigma(FITPRMS_ARRAY *fit_prms, int range_idx, int lag, double power) {
    if (!fit_prms) return 1.0;
    
    /* Power sigma calculation */
    double noise_level = fit_prms->noise;
    double snr = power / noise_level;
    
    /* Simple sigma model based on SNR */
    double sigma = 1.0 / sqrt(snr);
    
    return sigma;
}

double Calculate_Phase_Sigma(FITPRMS_ARRAY *fit_prms, int range_idx, int lag, double power) {
    if (!fit_prms) return 1.0;
    
    /* Phase sigma calculation */
    double noise_level = fit_prms->noise;
    double snr = power / noise_level;
    
    /* Phase sigma is inversely related to SNR */
    double sigma = 1.0 / snr;
    
    return sigma;
}

double Calculate_Elevation(FITPRMS_ARRAY *fit_prms, double complex acf, double complex xcf) {
    if (!fit_prms) return 0.0;
    
    /* Calculate elevation angle from interferometry */
    double phase_diff = carg(xcf) - carg(acf);
    
    /* Wrap phase difference */
    while (phase_diff > M_PI) phase_diff -= 2.0 * M_PI;
    while (phase_diff < -M_PI) phase_diff += 2.0 * M_PI;
    
    /* Convert phase difference to elevation angle */
    double wavelength = 3.0e8 / (fit_prms->tfreq * 1000.0); /* wavelength in meters */
    double baseline = 100.0; /* interferometer baseline in meters - should come from hdw.dat */
    
    double elevation = asin(phase_diff * wavelength / (2.0 * M_PI * baseline));
    
    return elevation * 180.0 / M_PI; /* Convert to degrees */
}

double Calculate_Elevation_Sigma(FITPRMS_ARRAY *fit_prms, int range_idx, int lag, double power) {
    if (!fit_prms) return 1.0;
    
    /* Elevation sigma calculation */
    double noise_level = fit_prms->noise;
    double snr = power / noise_level;
    
    /* Elevation sigma depends on SNR and geometry */
    double sigma = 5.0 / sqrt(snr); /* 5 degrees base uncertainty */
    
    return sigma;
}
