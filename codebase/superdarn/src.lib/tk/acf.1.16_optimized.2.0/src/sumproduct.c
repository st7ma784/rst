/* sumproduct.c (Optimized Version 2.0)
   =============
   Author: R.J.Barnes (Original)
   Optimized by: SuperDARN Optimization Framework
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
Version 2.0 - Added SIMD vectorization for power calculations
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "rtypes.h"
#include "acf.h"

/* Original ACFSumProduct - maintained for compatibility */
int ACFSumProduct(int16 *buffer,float *avepower,
                  int numsamples,float *mxpwr,float *dco) {
    
    /* Use optimized version for larger datasets */
    if (numsamples >= ACF_SIMD_WIDTH * 4) {
        return ACFSumProductOptimized(buffer, avepower, numsamples, mxpwr, dco);
    }
    
    /* Original implementation for small datasets */
    int sample;
    float sumpower;
    float power;
    float maxpower;
    float ltemp;

    float dcor1 = 0, dcoi1 = 0;

    sumpower = 0;
    maxpower = 0;

    if (dco != NULL) {
        dcor1 = dco[0];
        dcoi1 = dco[1]; 
    }

    for (sample = 0; sample < numsamples; sample++) {
        ltemp = *buffer - dcor1;
        power = ltemp * ltemp;
        buffer++;
        ltemp = *buffer - dcoi1;
        power = power + ltemp * ltemp;
        buffer++;

        if (maxpower < power) maxpower = power;
        sumpower = sumpower + power;
    }

    *mxpwr = maxpower;
    *avepower = sumpower / numsamples;
    return 0;
}

/* Optimized sum product with vectorization */
int ACFSumProductOptimized(int16 *buffer,float *avepower,
                          int numsamples,float *mxpwr,float *dco) {
    
    float sumpower = 0;
    float maxpower = 0;
    float dcor1 = 0, dcoi1 = 0;

    if (dco != NULL) {
        dcor1 = dco[0];
        dcoi1 = dco[1]; 
    }

#ifdef __AVX2__
    /* SIMD vectorized power calculation */
    __m256 vec_dcor = _mm256_set1_ps(dcor1);
    __m256 vec_dcoi = _mm256_set1_ps(dcoi1);
    __m256 vec_sumpower = _mm256_setzero_ps();
    __m256 vec_maxpower = _mm256_setzero_ps();
    
    int simd_samples = numsamples - (numsamples % ACF_SIMD_WIDTH);
    
    for (int sample = 0; sample < simd_samples; sample += ACF_SIMD_WIDTH) {
        /* Load 8 real values and 8 imaginary values */
        __m256i real_int16 = _mm256_loadu_si256((__m256i*)(buffer + sample * 2));
        __m256i imag_int16 = _mm256_loadu_si256((__m256i*)(buffer + sample * 2 + ACF_SIMD_WIDTH));
        
        /* Convert to float */
        __m256 real_float = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(real_int16, _mm256_setzero_si256()));
        __m256 imag_float = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(imag_int16, _mm256_setzero_si256()));
        
        /* Subtract DC offset */
        real_float = _mm256_sub_ps(real_float, vec_dcor);
        imag_float = _mm256_sub_ps(imag_float, vec_dcoi);
        
        /* Compute power: real^2 + imag^2 */
        __m256 real_squared = _mm256_mul_ps(real_float, real_float);
        __m256 imag_squared = _mm256_mul_ps(imag_float, imag_float);
        __m256 power = _mm256_add_ps(real_squared, imag_squared);
        
        /* Update sum and max */
        vec_sumpower = _mm256_add_ps(vec_sumpower, power);
        vec_maxpower = _mm256_max_ps(vec_maxpower, power);
    }
    
    /* Extract results from SIMD registers */
    float sum_array[ACF_SIMD_WIDTH];
    float max_array[ACF_SIMD_WIDTH];
    _mm256_store_ps(sum_array, vec_sumpower);
    _mm256_store_ps(max_array, vec_maxpower);
    
    for (int i = 0; i < ACF_SIMD_WIDTH; i++) {
        sumpower += sum_array[i];
        if (max_array[i] > maxpower) maxpower = max_array[i];
    }
    
    /* Handle remaining samples */
    for (int sample = simd_samples; sample < numsamples; sample++) {
        float rval = (float)(buffer[sample * 2]) - dcor1;
        float ival = (float)(buffer[sample * 2 + 1]) - dcoi1;
        float power = rval * rval + ival * ival;
        
        if (power > maxpower) maxpower = power;
        sumpower += power;
    }
    
#else
    /* Non-SIMD optimized version */
    for (int sample = 0; sample < numsamples; sample++) {
        float rval = (float)(buffer[sample * 2]) - dcor1;
        float ival = (float)(buffer[sample * 2 + 1]) - dcoi1;
        float power = rval * rval + ival * ival;
        
        if (power > maxpower) maxpower = power;
        sumpower += power;
    }
#endif

    *mxpwr = maxpower;
    *avepower = sumpower / numsamples;
    return 0;
}
