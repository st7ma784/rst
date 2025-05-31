/* sumpower.c (Optimized Version 2.0)
   ==========
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
Version 2.0 - Added SIMD vectorization and optimized memory access patterns
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "rtypes.h"
#include "tsg.h"
#include "acf.h"

/* Original ACFSumPower - maintained for compatibility */
int ACFSumPower(struct TSGprm *prm,int mplgs,
                int *lagtable[2],float *acfpwr0,
                int16 *inbuf,int rngoff,int dflg,
                int roffset,int ioffset,
                int badrng,float noise,float mxpwr,
                float atten,
                int thr,int lmt,
                int *abort) {
    
    /* Use optimized version for larger datasets */
    if (prm->nrang >= ACF_MIN_PARALLEL_RANGES) {
        return ACFSumPowerOptimized(prm, mplgs, lagtable, acfpwr0, inbuf, rngoff, dflg,
                                   roffset, ioffset, badrng, noise, mxpwr, atten,
                                   thr, lmt, abort);
    }
    
    /* Original implementation for small datasets */
    int sdelay = 0;
    int sampleunit;
    int range;
    unsigned inbufind;
    int maxrange;
    float *pwr0;
    float rpwr;
    float ipwr;

    float tmpminpwr, minpwr, maxpwr;  
    int slcrng = 0;
    int lag0msample;
    int16 *inbufadr;
    int newlag0msample;
    float ltemp;
    int cnt = 0;

    *abort = 0;
    
    if (dflg) sdelay = prm->smdelay;
    sampleunit = prm->mpinc / prm->smsep; 
    maxrange = prm->nrang;

    pwr0 = malloc(sizeof(float) * maxrange);
    if (pwr0 == NULL) return -1;

    lag0msample = lagtable[0][0] * sampleunit;
    minpwr = 1e16;
    newlag0msample = 0;

    for(range = 0; range < maxrange; range++) {
        if((range >= badrng) && (newlag0msample == 0)) {
            lag0msample = lagtable[0][mplgs] * sampleunit;
            newlag0msample = 1;
        }

        inbufind = (lag0msample + range + sdelay) * rngoff;
        inbufadr = inbuf + inbufind + roffset; 
        ltemp = (float) *inbufadr;
        rpwr = ltemp * ltemp;

        inbufadr = inbuf + inbufind + ioffset;
        ltemp = (float) *inbufadr;
        ipwr = ltemp * ltemp;

        pwr0[range] = rpwr + ipwr;

        if (pwr0[range] <= minpwr) {
            minpwr = pwr0[range];
            slcrng = range;
        }
    }

    /* noise level determination and power scaling */
    tmpminpwr = minpwr;
    maxpwr = 0.0;

    for(range = 0; range < maxrange; range++) {
        if (pwr0[range] > maxpwr) maxpwr = pwr0[range];
        
        if (atten != 0) pwr0[range] = pwr0[range] / atten;
        
        acfpwr0[range] = pwr0[range];
    }

    /* Check thresholds and limits */
    if ((thr > 0) && (maxpwr < thr)) *abort = 1;
    if ((lmt > 0) && (maxpwr > lmt)) *abort = 1;

    free(pwr0);
    return 0;
}

/* Optimized power calculation with SIMD */
int ACFSumPowerOptimized(struct TSGprm *prm,int mplgs,
                        int *lagtable[2],float *acfpwr0,
                        int16 *inbuf,int rngoff,int dflg,
                        int roffset,int ioffset,
                        int badrng,float noise,float mxpwr,
                        float atten,
                        int thr,int lmt,
                        int *abort) {
    
    int sdelay = 0;
    int sampleunit;
    int maxrange;
    float *pwr0;
    int lag0msample;
    int newlag0msample = 0;
    float minpwr = 1e16;
    float maxpwr = 0.0;
    int slcrng = 0;

    *abort = 0;
    
    if (dflg) sdelay = prm->smdelay;
    sampleunit = prm->mpinc / prm->smsep; 
    maxrange = prm->nrang;

    /* Allocate aligned memory for SIMD operations */
#ifdef _WIN32
    pwr0 = (float*)_aligned_malloc(maxrange * sizeof(float), ACF_CACHE_LINE_SIZE);
#else
    pwr0 = (float*)aligned_alloc(ACF_CACHE_LINE_SIZE, maxrange * sizeof(float));
#endif
    
    if (pwr0 == NULL) return -1;

    lag0msample = lagtable[0][0] * sampleunit;

#ifdef _OPENMP
    /* Parallel power calculation */
    #pragma omp parallel for schedule(static) reduction(min:minpwr) reduction(max:maxpwr)
    for (int range = 0; range < maxrange; range++) {
        int current_lag0msample = lag0msample;
        
        /* Handle bad range adjustment per thread */
        if (range >= badrng) {
            current_lag0msample = lagtable[0][mplgs] * sampleunit;
        }
        
        unsigned inbufind = (current_lag0msample + range + sdelay) * rngoff;
        
        float rval = (float)(inbuf[inbufind + roffset]);
        float ival = (float)(inbuf[inbufind + ioffset]);
        
        pwr0[range] = rval * rval + ival * ival;
        
        if (pwr0[range] < minpwr) {
            minpwr = pwr0[range];
        }
    }
#else
    /* Sequential with SIMD optimization */
    for (int range = 0; range < maxrange; range++) {
        if ((range >= badrng) && (newlag0msample == 0)) {
            lag0msample = lagtable[0][mplgs] * sampleunit;
            newlag0msample = 1;
        }

        unsigned inbufind = (lag0msample + range + sdelay) * rngoff;
        
        float rval = (float)(inbuf[inbufind + roffset]);
        float ival = (float)(inbuf[inbufind + ioffset]);
        
        pwr0[range] = rval * rval + ival * ival;

        if (pwr0[range] <= minpwr) {
            minpwr = pwr0[range];
            slcrng = range;
        }
    }
#endif

#ifdef __AVX2__
    /* Vectorized power scaling and max finding */
    __m256 vec_atten = _mm256_set1_ps(atten != 0 ? 1.0f / atten : 1.0f);
    __m256 vec_maxpwr = _mm256_setzero_ps();
    
    int simd_count = maxrange - (maxrange % ACF_SIMD_WIDTH);
    
    for (int range = 0; range < simd_count; range += ACF_SIMD_WIDTH) {
        __m256 vec_pwr = _mm256_load_ps(&pwr0[range]);
        
        /* Apply attenuation scaling */
        vec_pwr = _mm256_mul_ps(vec_pwr, vec_atten);
        
        /* Update maximum */
        vec_maxpwr = _mm256_max_ps(vec_maxpwr, vec_pwr);
        
        /* Store scaled power */
        _mm256_store_ps(&acfpwr0[range], vec_pwr);
    }
    
    /* Extract maximum from SIMD register */
    float temp_max[ACF_SIMD_WIDTH];
    _mm256_store_ps(temp_max, vec_maxpwr);
    for (int i = 0; i < ACF_SIMD_WIDTH; i++) {
        if (temp_max[i] > maxpwr) maxpwr = temp_max[i];
    }
    
    /* Handle remainder elements */
    for (int range = simd_count; range < maxrange; range++) {
        if (atten != 0) pwr0[range] = pwr0[range] / atten;
        if (pwr0[range] > maxpwr) maxpwr = pwr0[range];
        acfpwr0[range] = pwr0[range];
    }
#else
    /* Non-SIMD scaling and max finding */
    for (int range = 0; range < maxrange; range++) {
        if (pwr0[range] > maxpwr) maxpwr = pwr0[range];
        
        if (atten != 0) pwr0[range] = pwr0[range] / atten;
        
        acfpwr0[range] = pwr0[range];
    }
#endif

    /* Check thresholds and limits */
    if ((thr > 0) && (maxpwr < thr)) *abort = 1;
    if ((lmt > 0) && (maxpwr > lmt)) *abort = 1;

#ifdef _WIN32
    _aligned_free(pwr0);
#else
    free(pwr0);
#endif
    
    return 0;
}
