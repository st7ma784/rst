/* fitcfit.c
   =========
   Author: R.J.Barnes
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


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <zlib.h>
#include "rtypes.h"
#include "rtime.h"
#include "dmap.h"
#include "rprm.h"
#include "fitdata.h"
#include "cfitdata.h"



int FitToCFit(double min_pwr,struct CFitdata *ptr,
              struct RadarParm *prm,
              struct FitData *fit) {
  printf("Debug: FitToCFit - Start\n");
  printf("Debug: min_pwr=%f, ptr=%p, prm=%p, fit=%p\n", min_pwr, (void*)ptr, (void*)prm, (void*)fit);
  
  if (ptr == NULL) {
    printf("Error: ptr is NULL\n");
    return -1;
  }
  if (prm == NULL) {
    printf("Error: prm is NULL\n");
    return -1;
  }
  if (fit == NULL) {
    printf("Error: fit is NULL\n");
    return -1;
  }
  
  int i,num=0,rng;
  printf("Debug: Initialized variables\n");
  printf("Debug: Setting version\n");
  ptr->version.major=CFIT_MAJOR_REVISION;
  ptr->version.minor=CFIT_MINOR_REVISION;

  /* time stamp the record */
  printf("Debug: Converting time - yr=%d, mo=%d, dy=%d, hr=%d, mt=%d, sc=%d, us=%d\n", 
         prm->time.yr, prm->time.mo, prm->time.dy, prm->time.hr, prm->time.mt, prm->time.sc, prm->time.us);
  ptr->time = TimeYMDHMSToEpoch(prm->time.yr, prm->time.mo, prm->time.dy,
                              prm->time.hr, prm->time.mt, prm->time.sc, prm->time.us);
  printf("Debug: Time converted to epoch: %f\n", ptr->time);
  
  /* Debug print RadarParm structure */
  printf("Debug: RadarParm - stid=%d, channel=%d, bmnum=%d, bmazm=%.2f\n", 
         prm->stid, prm->channel, prm->bmnum, prm->bmazm);
  printf("Debug: RadarParm - nrang=%d, num=%d, rng=%p\n", 
         prm->nrang, prm->num, (void*)prm->rng);
  
  /* copy the radar parameters */
  ptr->stid = prm->stid;
  ptr->scan = prm->scan;
  ptr->bmnum = prm->bmnum;
  ptr->channel = prm->channel;
  ptr->bmazm = prm->bmazm;
  
  /* Set integration time */
  ptr->intt.sc = prm->intt.sc;
  ptr->intt.us = prm->intt.us;
  ptr->txpow = 0;  /* txpow not available in the new structure */
  ptr->noise = prm->noise.search;
  ptr->rxrise = prm->rxrise;
  ptr->tfreq = prm->tfreq;
  ptr->atten = prm->atten;
  ptr->nave = prm->nave;
  
  // Use the larger of nrang and num to ensure we don't miss any ranges
  int actual_nrang = (prm->nrang > prm->num) ? prm->nrang : prm->num;
  if (actual_nrang <= 0) {
    printf("Warning: Invalid number of ranges: nrang=%d, num=%d. Using default 100.\n", 
           prm->nrang, prm->num);
    actual_nrang = 100;  // Default value if both are invalid
  }
  ptr->nrang = actual_nrang;
  
  printf("Debug: Using %d ranges (nrang=%d, num=%d)\n", 
         actual_nrang, prm->nrang, prm->num);
  
  // Validate range array
  if (prm->rng == NULL) {
    printf("Warning: prm->rng is NULL, using sequential range gates\n");
  } else {
    printf("Debug: First few range gates: %d, %d, %d, ...\n", 
           prm->rng[0], prm->rng[1], prm->rng[2]);
  }
  
  // Count valid ranges
  printf("Debug: Counting valid ranges (nrang=%d, actual_nrang=%d)\n", 
         prm->nrang, actual_nrang);
         
  for (i = 0; i < actual_nrang; i++) {
    if (i < 0 || i >= actual_nrang) {
      printf("Error: Invalid index i=%d, actual_nrang=%d\n", i, actual_nrang);
      return -1;
    }
    if (fit->rng == NULL) {
      printf("Error: fit->rng is NULL at index %d\n", i);
      return -1;
    }
    if (fit->rng[i].qflg!=1) continue; 
    if ((min_pwr !=0) && (fit->rng[i].p_0 <= min_pwr)) continue;
    num++;
  }
  printf("Debug: Found %d valid ranges\n", num);

  printf("Debug: Setting CFitdata range to %d\n", num);
  if (CFitSetRng(ptr,num) != 0) {
    printf("Error: Failed to set CFitdata range\n");
    return -1;
  }
  num=0;
  printf("Debug: Filling range indices (nrang=%d)\n", prm->nrang);
  for (i=0;i<prm->nrang;i++) {
    if (i < 0 || i >= prm->nrang) {
      printf("Error: Invalid index i=%d, nrang=%d\n", i, prm->nrang);
      return -1;
    }
    if (fit->rng == NULL) {
      printf("Error: fit->rng is NULL at index %d\n", i);
      return -1;
    }
    if (fit->rng[i].qflg!=1) continue;
    if ((min_pwr !=0) && (fit->rng[i].p_0 <= min_pwr)) continue;
    
    if (ptr->rng == NULL) {
      printf("Error: ptr->rng is NULL at index %d\n", num);
      return -1;
    }
    
    ptr->rng[num]=i;
    printf("Debug: ptr->rng[%d] = %d\n", num, i);
    num++;
  }
  if (num>0) {
    printf("Debug: Copying data for %d ranges\n", num);
    if (ptr->data == NULL) {
      printf("Error: ptr->data is NULL\n");
      return -1;
    }
    
    for (i=0;i<num;i++) {
      printf("Debug: Processing range %d/%d\n", i+1, num);
      
      if (i < 0 || i >= num) {
        printf("Error: Invalid index i=%d, num=%d\n", i, num);
        return -1;
      }
      
      if (ptr->rng == NULL) {
        printf("Error: ptr->rng is NULL at index %d\n", i);
        return -1;
      }
      
      rng = ptr->rng[i];
      printf("Debug: Range index %d: rng=%d\n", i, rng);
      
      if (rng < 0 || rng >= prm->nrang) {
        printf("Error: Invalid range index rng=%d, nrang=%d\n", rng, prm->nrang);
        return -1;
      }
      
      if (fit->rng == NULL) {
        printf("Error: fit->rng is NULL at rng=%d\n", rng);
        return -1;
      }
      
      printf("Debug: Copying data for range %d\n", rng);
      
      ptr->data[i].gsct = fit->rng[rng].gsct;
      ptr->data[i].p_0 = fit->rng[rng].p_0;
      ptr->data[i].p_0_e = 0;
      ptr->data[i].v = fit->rng[rng].v;
      ptr->data[i].v_e = fit->rng[rng].v_err;
      ptr->data[i].p_l = fit->rng[rng].p_l;
      ptr->data[i].p_l_e = fit->rng[rng].p_l_err;
      ptr->data[i].w_l = fit->rng[rng].w_l;
      ptr->data[i].w_l_e = fit->rng[rng].w_l_err;
    }
  }
  
  printf("Debug: FitToCFit - Done\n");
  return 0;
}
 






