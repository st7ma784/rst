/*
 * preprocessing_array.h
 * =====================
 * 
 * Header file for array-based preprocessing functions for SuperDARN FitACF v3.0
 * 
 * This header defines the array-based versions of the preprocessing functions
 * to enable massive parallelization with OpenMP.
 * 
 * Copyright (c) 2025 SuperDARN Refactoring Project
 * Author: GitHub Copilot Assistant
 */

#ifndef _PREPROCESSING_ARRAY_H
#define _PREPROCESSING_ARRAY_H

#include <complex.h>
#include "fit_structures_array.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Array-based preprocessing function prototypes */

/**
 * Fill the range list for array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @return 0 on success, -1 on error
 */
int Fill_Range_List_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays);

/**
 * Fill data lists for a specific range in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @param range_idx Index of the range to process
 * @return 0 on success, -1 on error
 */
int Fill_Data_Lists_For_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx);

/**
 * Parallel preprocessing using OpenMP for array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 */
void Parallel_Preprocessing_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays);

/**
 * Filter TX overlap for all ranges in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @return Number of ranges processed
 */
int Filter_TX_Overlap_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays);

/**
 * Filter TX overlap for a specific range in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @param range_idx Index of the range to process
 * @return 0 on success, -1 on error
 */
int Filter_TX_Overlap_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx);

/**
 * Find CRI (Close Range Interference) for all ranges in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @return Number of ranges processed
 */
int Find_CRI_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays);

/**
 * Find CRI for a specific range in array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param arrays Pointer to range data arrays structure
 * @param range_idx Index of the range to process
 * @return 0 on success, -1 on error
 */
int Find_CRI_Range_Array(FITPRMS_ARRAY *fit_prms, RANGE_DATA_ARRAYS *arrays, int range_idx);

/**
 * Calculate alpha-2 parameter for array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param range_idx Index of the range
 * @param lag Lag number
 * @return Alpha-2 value
 */
double Calculate_Alpha_2(FITPRMS_ARRAY *fit_prms, int range_idx, int lag);

/**
 * Calculate power sigma for array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param range_idx Index of the range
 * @param lag Lag number
 * @param power Power value
 * @return Power sigma value
 */
double Calculate_Power_Sigma(FITPRMS_ARRAY *fit_prms, int range_idx, int lag, double power);

/**
 * Calculate phase sigma for array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param range_idx Index of the range
 * @param lag Lag number
 * @param power Power value
 * @return Phase sigma value
 */
double Calculate_Phase_Sigma(FITPRMS_ARRAY *fit_prms, int range_idx, int lag, double power);

/**
 * Calculate elevation angle for array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param acf ACF value (complex)
 * @param xcf XCF value (complex)
 * @return Elevation angle in degrees
 */
double Calculate_Elevation(FITPRMS_ARRAY *fit_prms, double complex acf, double complex xcf);

/**
 * Calculate elevation sigma for array-based processing
 * @param fit_prms Pointer to array-based fit parameters structure
 * @param range_idx Index of the range
 * @param lag Lag number
 * @param power Power value
 * @return Elevation sigma value
 */
double Calculate_Elevation_Sigma(FITPRMS_ARRAY *fit_prms, int range_idx, int lag, double power);

#ifdef __cplusplus
}
#endif

#endif /* _PREPROCESSING_ARRAY_H */
