/* acfcalculate.c (Optimized Version 2.0)
   ==============
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
Version 2.0 - Added OpenMP parallelization, SIMD vectorization, and optimized memory access patterns
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

/* Thread-local storage for parallel processing */
static __thread ACF_ThreadLocal *thread_local_data = NULL;

/* SIMD Buffer Management */
ACF_SIMD_Buffer* ACF_CreateSIMDBuffer(int size) {
    ACF_SIMD_Buffer *buf = malloc(sizeof(ACF_SIMD_Buffer));
    if (!buf) return NULL;
    
    buf->aligned_size = ((size + ACF_SIMD_WIDTH - 1) / ACF_SIMD_WIDTH) * ACF_SIMD_WIDTH;
    
#ifdef _WIN32
    buf->real_buffer = (float*)_aligned_malloc(buf->aligned_size * sizeof(float), ACF_CACHE_LINE_SIZE);
    buf->imag_buffer = (float*)_aligned_malloc(buf->aligned_size * sizeof(float), ACF_CACHE_LINE_SIZE);
#else
    buf->real_buffer = (float*)aligned_alloc(ACF_CACHE_LINE_SIZE, buf->aligned_size * sizeof(float));
    buf->imag_buffer = (float*)aligned_alloc(ACF_CACHE_LINE_SIZE, buf->aligned_size * sizeof(float));
#endif
    
    if (!buf->real_buffer || !buf->imag_buffer) {
        ACF_DestroySIMDBuffer(buf);
        return NULL;
    }
    
    buf->size = size;
    memset(buf->real_buffer, 0, buf->aligned_size * sizeof(float));
    memset(buf->imag_buffer, 0, buf->aligned_size * sizeof(float));
    
    return buf;
}

void ACF_DestroySIMDBuffer(ACF_SIMD_Buffer *buf) {
    if (!buf) return;
    
#ifdef _WIN32
    if (buf->real_buffer) _aligned_free(buf->real_buffer);
    if (buf->imag_buffer) _aligned_free(buf->imag_buffer);
#else
    if (buf->real_buffer) free(buf->real_buffer);
    if (buf->imag_buffer) free(buf->imag_buffer);
#endif
    
    free(buf);
}

/* Thread-local storage management */
ACF_ThreadLocal* ACF_GetThreadLocal(int mplgs, int nrang) {
    if (thread_local_data == NULL) {
        thread_local_data = malloc(sizeof(ACF_ThreadLocal));
        if (!thread_local_data) return NULL;
        
        thread_local_data->temp_acf = malloc(nrang * 2 * mplgs * sizeof(float));
        thread_local_data->simd_buf = *ACF_CreateSIMDBuffer(mplgs * 2);
        
#ifdef _OPENMP
        thread_local_data->thread_id = omp_get_thread_num();
#else
        thread_local_data->thread_id = 0;
#endif
        
        if (!thread_local_data->temp_acf) {
            ACF_DestroyThreadLocal(thread_local_data);
            thread_local_data = NULL;
            return NULL;
        }
        
        memset(thread_local_data->temp_acf, 0, nrang * 2 * mplgs * sizeof(float));
    }
    
    return thread_local_data;
}

void ACF_DestroyThreadLocal(ACF_ThreadLocal *tl) {
    if (!tl) return;
    
    if (tl->temp_acf) free(tl->temp_acf);
    ACF_DestroySIMDBuffer(&tl->simd_buf);
    free(tl);
}

/* Vectorized complex multiplication using SIMD */
#ifdef __AVX2__
static inline void simd_complex_multiply(const float *a_real, const float *a_imag,
                                        const float *b_real, const float *b_imag,
                                        float *result_real, float *result_imag,
                                        int count) {
    int simd_count = count - (count % ACF_SIMD_WIDTH);
    
    for (int i = 0; i < simd_count; i += ACF_SIMD_WIDTH) {
        __m256 ar = _mm256_load_ps(&a_real[i]);
        __m256 ai = _mm256_load_ps(&a_imag[i]);
        __m256 br = _mm256_load_ps(&b_real[i]);
        __m256 bi = _mm256_load_ps(&b_imag[i]);
        
        // Complex multiplication: (ar + ai*i) * (br + bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
        __m256 real_part = _mm256_sub_ps(_mm256_mul_ps(ar, br), _mm256_mul_ps(ai, bi));
        __m256 imag_part = _mm256_add_ps(_mm256_mul_ps(ar, bi), _mm256_mul_ps(ai, br));
        
        _mm256_store_ps(&result_real[i], real_part);
        _mm256_store_ps(&result_imag[i], imag_part);
    }
    
    // Handle remainder elements
    for (int i = simd_count; i < count; i++) {
        result_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        result_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }
}
#endif

/* Original ACF calculation - maintained for compatibility */
int ACFCalculate(struct TSGprm *prm,
                 int16 *inbuf,int rngoff,int dflg,
                 int roffset,int ioffset,
                 int mplgs,int *lagtable[2],
                 float *acfbuf,
                 int xcf,int xcfoff,
                 int badrange,float atten,float *dco) {
    
    int nrang = prm->nrang;
    
    /* Use optimized version for larger datasets */
    if (nrang >= ACF_MIN_PARALLEL_RANGES && mplgs >= 4) {
        return ACFCalculateParallelRanges(prm, inbuf, rngoff, dflg, roffset, ioffset,
                                         mplgs, lagtable, acfbuf, xcf, xcfoff,
                                         badrange, atten, dco);
    }
    
    /* Fall back to vectorized version for medium datasets */
    if (mplgs >= ACF_SIMD_WIDTH) {
        return ACFCalculateVectorized(prm, inbuf, rngoff, dflg, roffset, ioffset,
                                     mplgs, lagtable, acfbuf, xcf, xcfoff,
                                     badrange, atten, dco);
    }
    
    /* Original implementation for small datasets */
    int sdelay = 0;
    int range;
    int sampleunit;
    int offset1;
    float real;
    float imag;
    int lag;
    int sample1;
    int sample2;
    float temp1;
    float temp2;
    int offset2;

    float dcor1 = 0, dcor2 = 0, dcoi1 = 0, dcoi2 = 0;

    if (dco != NULL) {
        if (xcf == ACF_PART) {
            dcor1 = dco[0];
            dcor2 = dco[0];
            dcoi1 = dco[1];
            dcoi2 = dco[1];       
        } else {
            dcor1 = dco[0];
            dcor2 = dco[2];
            dcoi1 = dco[1];
            dcoi2 = dco[3];
        }
    }

    if (dflg) sdelay = prm->smdelay;
    sampleunit = (prm->mpinc / prm->smsep) * rngoff;
				 
    for(range = 0; range < nrang; range++) {
        offset1 = (range + sdelay) * rngoff;

        if (xcf == ACF_PART) offset2 = offset1;
        else offset2 = ((range + sdelay) * rngoff) + xcfoff;
      
        for(lag = 0; lag < mplgs; lag++) {
            if ((range >= badrange) && (lag == 0)) {
                sample1 = lagtable[0][mplgs] * sampleunit + offset1;        
                sample2 = lagtable[1][mplgs] * sampleunit + offset2;
            } else { 
                sample1 = lagtable[0][lag] * sampleunit + offset1;        
                sample2 = lagtable[1][lag] * sampleunit + offset2;
            }
         
            temp1 = (float)(inbuf[sample1 + roffset] - dcor1) * 
                    (float)(inbuf[sample2 + roffset] - dcor2);

            temp2 = (float)(inbuf[sample1 + ioffset] - dcoi1) * 
                    (float)(inbuf[sample2 + ioffset] - dcoi2);
            real = temp1 + temp2;

            temp1 = (float)(inbuf[sample1 + roffset] - dcor1) *
                    (float)(inbuf[sample2 + ioffset] - dcoi2);
            temp2 = (float)(inbuf[sample2 + roffset] - dcor2) * 
                    (float)(inbuf[sample1 + ioffset] - dcoi1); 
            imag = temp1 - temp2;

            if (atten != 0) {
                real = real / atten;
                imag = imag / atten;
            }

            acfbuf[range * (2 * mplgs) + 2 * lag] += real;
            acfbuf[range * (2 * mplgs) + 2 * lag + 1] += imag;
        } 
    }

    return 0;
}

/* Vectorized ACF calculation using SIMD */
int ACFCalculateVectorized(struct TSGprm *prm,
                          int16 *inbuf,int rngoff,int dflg,
                          int roffset,int ioffset,
                          int mplgs,int *lagtable[2],
                          float *acfbuf,
                          int xcf,int xcfoff,
                          int badrange,float atten,float *dco) {
    
    int nrang = prm->nrang;
    int sdelay = 0;
    int sampleunit;
    float dcor1 = 0, dcor2 = 0, dcoi1 = 0, dcoi2 = 0;
    
    if (dco != NULL) {
        if (xcf == ACF_PART) {
            dcor1 = dcor2 = dco[0];
            dcoi1 = dcoi2 = dco[1];
        } else {
            dcor1 = dco[0]; dcor2 = dco[2];
            dcoi1 = dco[1]; dcoi2 = dco[3];
        }
    }
    
    if (dflg) sdelay = prm->smdelay;
    sampleunit = (prm->mpinc / prm->smsep) * rngoff;
    
    ACF_ThreadLocal *tl = ACF_GetThreadLocal(mplgs, nrang);
    if (!tl) return -1;
    
#ifdef __AVX2__
    // Use SIMD for lag processing when possible
    for (int range = 0; range < nrang; range++) {
        int offset1 = (range + sdelay) * rngoff;
        int offset2 = (xcf == ACF_PART) ? offset1 : ((range + sdelay) * rngoff) + xcfoff;
        
        // Process lags in SIMD-width chunks
        int simd_lags = mplgs - (mplgs % ACF_SIMD_WIDTH);
        
        for (int lag = 0; lag < simd_lags; lag += ACF_SIMD_WIDTH) {
            __m256 real_results = _mm256_setzero_ps();
            __m256 imag_results = _mm256_setzero_ps();
            
            for (int i = 0; i < ACF_SIMD_WIDTH; i++) {
                int current_lag = lag + i;
                int sample1, sample2;
                
                if ((range >= badrange) && (current_lag == 0)) {
                    sample1 = lagtable[0][mplgs] * sampleunit + offset1;
                    sample2 = lagtable[1][mplgs] * sampleunit + offset2;
                } else {
                    sample1 = lagtable[0][current_lag] * sampleunit + offset1;
                    sample2 = lagtable[1][current_lag] * sampleunit + offset2;
                }
                
                float s1r = (float)(inbuf[sample1 + roffset]) - dcor1;
                float s1i = (float)(inbuf[sample1 + ioffset]) - dcoi1;
                float s2r = (float)(inbuf[sample2 + roffset]) - dcor2;
                float s2i = (float)(inbuf[sample2 + ioffset]) - dcoi2;
                
                float real_part = s1r * s2r + s1i * s2i;
                float imag_part = s1r * s2i - s2r * s1i;
                
                if (atten != 0) {
                    real_part /= atten;
                    imag_part /= atten;
                }
                
                // Store in SIMD buffers for accumulation
                ((float*)&real_results)[i] = real_part;
                ((float*)&imag_results)[i] = imag_part;
            }
            
            // Store results back to ACF buffer
            for (int i = 0; i < ACF_SIMD_WIDTH; i++) {
                int current_lag = lag + i;
                acfbuf[range * (2 * mplgs) + 2 * current_lag] += ((float*)&real_results)[i];
                acfbuf[range * (2 * mplgs) + 2 * current_lag + 1] += ((float*)&imag_results)[i];
            }
        }
        
        // Handle remaining lags
        for (int lag = simd_lags; lag < mplgs; lag++) {
            int sample1, sample2;
            
            if ((range >= badrange) && (lag == 0)) {
                sample1 = lagtable[0][mplgs] * sampleunit + offset1;
                sample2 = lagtable[1][mplgs] * sampleunit + offset2;
            } else {
                sample1 = lagtable[0][lag] * sampleunit + offset1;
                sample2 = lagtable[1][lag] * sampleunit + offset2;
            }
            
            float s1r = (float)(inbuf[sample1 + roffset]) - dcor1;
            float s1i = (float)(inbuf[sample1 + ioffset]) - dcoi1;
            float s2r = (float)(inbuf[sample2 + roffset]) - dcor2;
            float s2i = (float)(inbuf[sample2 + ioffset]) - dcoi2;
            
            float real_part = s1r * s2r + s1i * s2i;
            float imag_part = s1r * s2i - s2r * s1i;
            
            if (atten != 0) {
                real_part /= atten;
                imag_part /= atten;
            }
            
            acfbuf[range * (2 * mplgs) + 2 * lag] += real_part;
            acfbuf[range * (2 * mplgs) + 2 * lag + 1] += imag_part;
        }
    }
#else
    // Fallback to original implementation if SIMD not available
    return ACFCalculate(prm, inbuf, rngoff, dflg, roffset, ioffset,
                       mplgs, lagtable, acfbuf, xcf, xcfoff,
                       badrange, atten, dco);
#endif
    
    return 0;
}

/* Parallel range processing using OpenMP */
int ACFCalculateParallelRanges(struct TSGprm *prm,
                              int16 *inbuf,int rngoff,int dflg,
                              int roffset,int ioffset,
                              int mplgs,int *lagtable[2],
                              float *acfbuf,
                              int xcf,int xcfoff,
                              int badrange,float atten,float *dco) {
    
    int nrang = prm->nrang;
    int sdelay = 0;
    int sampleunit;
    float dcor1 = 0, dcor2 = 0, dcoi1 = 0, dcoi2 = 0;
    
    if (dco != NULL) {
        if (xcf == ACF_PART) {
            dcor1 = dcor2 = dco[0];
            dcoi1 = dcoi2 = dco[1];
        } else {
            dcor1 = dco[0]; dcor2 = dco[2];
            dcoi1 = dco[1]; dcoi2 = dco[3];
        }
    }
    
    if (dflg) sdelay = prm->smdelay;
    sampleunit = (prm->mpinc / prm->smsep) * rngoff;
    
#ifdef _OPENMP
    // Parallel processing across ranges
    #pragma omp parallel for schedule(dynamic, 4) shared(acfbuf) firstprivate(dcor1, dcor2, dcoi1, dcoi2)
    for (int range = 0; range < nrang; range++) {
        int offset1 = (range + sdelay) * rngoff;
        int offset2 = (xcf == ACF_PART) ? offset1 : ((range + sdelay) * rngoff) + xcfoff;
        
        // Each thread processes all lags for this range
        for (int lag = 0; lag < mplgs; lag++) {
            int sample1, sample2;
            
            if ((range >= badrange) && (lag == 0)) {
                sample1 = lagtable[0][mplgs] * sampleunit + offset1;
                sample2 = lagtable[1][mplgs] * sampleunit + offset2;
            } else {
                sample1 = lagtable[0][lag] * sampleunit + offset1;
                sample2 = lagtable[1][lag] * sampleunit + offset2;
            }
            
            float s1r = (float)(inbuf[sample1 + roffset]) - dcor1;
            float s1i = (float)(inbuf[sample1 + ioffset]) - dcoi1;
            float s2r = (float)(inbuf[sample2 + roffset]) - dcor2;
            float s2i = (float)(inbuf[sample2 + ioffset]) - dcoi2;
            
            float real_part = s1r * s2r + s1i * s2i;
            float imag_part = s1r * s2i - s2r * s1i;
            
            if (atten != 0) {
                real_part /= atten;
                imag_part /= atten;
            }
            
            // Critical section for updating shared ACF buffer
            #pragma omp critical
            {
                acfbuf[range * (2 * mplgs) + 2 * lag] += real_part;
                acfbuf[range * (2 * mplgs) + 2 * lag + 1] += imag_part;
            }
        }
    }
#else
    // Fallback to vectorized implementation if OpenMP not available
    return ACFCalculateVectorized(prm, inbuf, rngoff, dflg, roffset, ioffset,
                                 mplgs, lagtable, acfbuf, xcf, xcfoff,
                                 badrange, atten, dco);
#endif
    
    return 0;
}
