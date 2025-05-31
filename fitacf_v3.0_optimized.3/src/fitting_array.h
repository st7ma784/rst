/*
 * fitting_array.h
 * ===============
 * 
 * Header file for array-based fitting functions for SuperDARN FitACF v3.0
 * 
 * This header defines the array-based versions of the fitting functions
 * to enable massive parallelization with OpenMP.
 * 
 * Copyright (c) 2025 SuperDARN Refactoring Project
 * Author: GitHub Copilot Assistant
 */

#ifndef _FITTING_ARRAY_H
#define _FITTING_ARRAY_H

#include "fit_structures_array.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Array-based fitting function prototypes */

/**
 * Perform power fits for all ranges in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @return Number of successful fits
 */
int Power_Fits_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays);

/**
 * Perform power fit for a specific range in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @param range_idx Index of the range to fit
 * @return 0 on success, -1 on error
 */
int Power_Fit_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx);

/**
 * Perform linear power fit for array-based processing
 * @param rng Pointer to range node array
 * @return 0 on success, -1 on error
 */
int Linear_Power_Fit_Array(RANGENODE_ARRAY *rng);

/**
 * Perform quadratic power fit for array-based processing
 * @param rng Pointer to range node array
 * @return 0 on success, -1 on error
 */
int Quadratic_Power_Fit_Array(RANGENODE_ARRAY *rng);

/**
 * Perform ACF phase fits for all ranges in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @return Number of successful fits
 */
int ACF_Phase_Fit_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays);

/**
 * Perform phase fit for a specific range in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @param range_idx Index of the range to fit
 * @return 0 on success, -1 on error
 */
int Phase_Fit_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx);

/**
 * Perform phase unwrapping for array-based processing
 * @param rng Pointer to range node array
 * @return 0 on success, -1 on error
 */
int Phase_Unwrap_Array(RANGENODE_ARRAY *rng);

/**
 * Perform linear phase fit for array-based processing
 * @param rng Pointer to range node array
 * @return 0 on success, -1 on error
 */
int Linear_Phase_Fit_Array(RANGENODE_ARRAY *rng);

/**
 * Perform XCF phase fits for all ranges in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @return Number of successful fits
 */
int XCF_Phase_Fit_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays);

/**
 * Perform elevation fit for a specific range in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @param range_idx Index of the range to fit
 * @return 0 on success, -1 on error
 */
int Elevation_Fit_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx);

/**
 * Perform matrix-based power fitting with parallelization
 * @param arrays Pointer to range data arrays structure
 * @param num_threads Number of OpenMP threads to use
 */
void Matrix_Power_Fitting_Array(RANGE_DATA_ARRAYS *arrays, int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* _FITTING_ARRAY_H */
