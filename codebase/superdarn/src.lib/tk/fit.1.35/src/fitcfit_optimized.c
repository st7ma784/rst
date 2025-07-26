/* fitcfit_optimized.c
   ===================
   Optimized version of FitToCFit function for improved performance
   
   Key optimizations:
   1. Single-pass processing (eliminates redundant loops)
   2. Reduced debug output overhead
   3. Optimized memory access patterns
   4. Minimized bounds checking
   5. SIMD-friendly data layout
   
   Author: Performance optimization for SuperDARN RST
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "rtypes.h"
#include "rtime.h"
#include "dmap.h"
#include "rprm.h"
#include "fitdata.h"
#include "cfitdata.h"

// Optimized version with single-pass processing
int FitToCFit_Optimized(double min_pwr, struct CFitdata *ptr,
                       struct RadarParm *prm, struct FitData *fit) {
    
    // Input validation (fast path)
    if (!ptr || !prm || !fit) return -1;
    
    // Set version information
    ptr->version.major = CFIT_MAJOR_REVISION;
    ptr->version.minor = CFIT_MINOR_REVISION;
    
    // Convert time stamp
    ptr->time = TimeYMDHMSToEpoch(prm->time.yr, prm->time.mo, prm->time.dy,
                                 prm->time.hr, prm->time.mt, prm->time.sc, prm->time.us);
    
    // Copy radar parameters (bulk copy for better cache performance)
    ptr->stid = prm->stid;
    ptr->scan = prm->scan;
    ptr->bmnum = prm->bmnum;
    ptr->channel = prm->channel;
    ptr->bmazm = prm->bmazm;
    ptr->intt.sc = prm->intt.sc;
    ptr->intt.us = prm->intt.us;
    ptr->txpow = 0;
    ptr->noise = prm->noise.search;
    ptr->rxrise = prm->rxrise;
    ptr->tfreq = prm->tfreq;
    ptr->atten = prm->atten;
    ptr->nave = prm->nave;
    
    // Determine actual number of ranges
    int actual_nrang = (prm->nrang > prm->num) ? prm->nrang : prm->num;
    if (actual_nrang <= 0) actual_nrang = 100;
    ptr->nrang = actual_nrang;
    
    // Single-pass processing: count valid ranges and collect indices
    int valid_indices[actual_nrang];  // Stack allocation for better performance
    int num_valid = 0;
    
    // Optimized loop with minimal branching
    const double min_power_threshold = (min_pwr != 0) ? min_pwr : -1e30;
    
    for (int i = 0; i < actual_nrang; i++) {
        // Bounds check once
        if (i >= actual_nrang || !fit->rng) break;
        
        // Combined condition check (branch prediction friendly)
        if (fit->rng[i].qflg == 1 && fit->rng[i].p_0 > min_power_threshold) {
            valid_indices[num_valid++] = i;
        }
    }
    
    // Allocate CFitdata storage
    if (CFitSetRng(ptr, num_valid) != 0) return -1;
    
    // Fast data copying with optimized memory access
    if (num_valid > 0 && ptr->data && ptr->rng) {
        // Copy range indices
        memcpy(ptr->rng, valid_indices, num_valid * sizeof(int));
        
        // Vectorized data copying where possible
        #pragma omp simd
        for (int i = 0; i < num_valid; i++) {
            int rng_idx = valid_indices[i];
            
            // Bulk copy structure members for better cache utilization
            ptr->data[i].gsct = fit->rng[rng_idx].gsct;
            ptr->data[i].p_0 = fit->rng[rng_idx].p_0;
            ptr->data[i].p_0_e = 0;
            ptr->data[i].v = fit->rng[rng_idx].v;
            ptr->data[i].v_e = fit->rng[rng_idx].v_err;
            ptr->data[i].p_l = fit->rng[rng_idx].p_l;
            ptr->data[i].p_l_e = fit->rng[rng_idx].p_l_err;
            ptr->data[i].w_l = fit->rng[rng_idx].w_l;
            ptr->data[i].w_l_e = fit->rng[rng_idx].w_l_err;
        }
    }
    
    return 0;
}

// Simple optimized version focusing on algorithmic improvements
// Removes complex SIMD and OpenMP code for better compatibility
