/* acf.h (Optimized Version 2.0)
   =====
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
Version 2.0 - Optimized with OpenMP parallelization and SIMD vectorization
*/ 

#ifndef _ACF_OPTIMIZED_H
#define _ACF_OPTIMIZED_H

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#define ACF_PART 0
#define XCF_PART 1

/* Optimization parameters */
#define ACF_CACHE_LINE_SIZE 64
#define ACF_SIMD_WIDTH 8  /* AVX2 processes 8 floats at once */
#define ACF_MIN_PARALLEL_RANGES 16  /* Minimum ranges for parallelization */

/* Cache-aligned data structure for SIMD operations */
typedef struct {
    float *real_buffer;
    float *imag_buffer;
    int size;
    int aligned_size;
} ACF_SIMD_Buffer;

/* Thread-local storage for parallel processing */
typedef struct {
    float *temp_acf;
    ACF_SIMD_Buffer simd_buf;
    int thread_id;
} ACF_ThreadLocal;

/* Optimized ACF calculation functions */
int ACFCalculate(struct TSGprm *prm,
                 int16 *inbuf,int rngoff,int dflg,
                 int roffset,int ioffset,
                 int mplgs,int *lagtable[2],
                 float *acfbuf,
                 int xcf,int xcfoff,
                 int badrange,float atten,float *dco);

/* Vectorized ACF calculation for SIMD optimization */
int ACFCalculateVectorized(struct TSGprm *prm,
                          int16 *inbuf,int rngoff,int dflg,
                          int roffset,int ioffset,
                          int mplgs,int *lagtable[2],
                          float *acfbuf,
                          int xcf,int xcfoff,
                          int badrange,float atten,float *dco);

/* Parallel range processing */
int ACFCalculateParallelRanges(struct TSGprm *prm,
                              int16 *inbuf,int rngoff,int dflg,
                              int roffset,int ioffset,
                              int mplgs,int *lagtable[2],
                              float *acfbuf,
                              int xcf,int xcfoff,
                              int badrange,float atten,float *dco);

/* SIMD buffer management */
ACF_SIMD_Buffer* ACF_CreateSIMDBuffer(int size);
void ACF_DestroySIMDBuffer(ACF_SIMD_Buffer *buf);

/* Thread-local storage management */
ACF_ThreadLocal* ACF_GetThreadLocal(int mplgs, int nrang);
void ACF_DestroyThreadLocal(ACF_ThreadLocal *tl);

/* Original function signatures maintained for compatibility */
int ACFAverage(float *pwr0,float *acfd,
               float *xcfd,
               int nave,int nrang,int mplgs);

int ACFBadLagZero(struct TSGprm *prm,int mplgs,int *lagtable[2]);

void ACFNormalize(float *pwr0,float *acfd,float *xcfd,
                  int nrang,int mplgs,float atten);

int ACFSumPower(struct TSGprm *prm,int mplgs,
                int *lagtable[2],float *acfpwr0,
                int16 *inbuf,int rngoff,int dflg,
                int roffset,int ioffset,
                int badrng,float noise,float mxpwr,
                float atten,
                int thr,int lmt,
                int *abort);

/* Optimized power calculation with SIMD */
int ACFSumPowerOptimized(struct TSGprm *prm,int mplgs,
                        int *lagtable[2],float *acfpwr0,
                        int16 *inbuf,int rngoff,int dflg,
                        int roffset,int ioffset,
                        int badrng,float noise,float mxpwr,
                        float atten,
                        int thr,int lmt,
                        int *abort);

int ACFSumProduct(int16 *buffer,float *avepower,
                  int numsamples,float *mxpwr,float *dco);

/* Optimized sum product with vectorization */
int ACFSumProductOptimized(int16 *buffer,float *avepower,
                          int numsamples,float *mxpwr,float *dco);

#endif
