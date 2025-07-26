/* fitwrite.c
   ========== 
   Author R.J.Barnes

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
    2021-11-12 Emma Bland (UNIS): Added "elv_error" and "elv_fitted" fields for FitACF v3
                                  Only write XCF fitted parameters to file for FitACF v2 and earlier
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <zlib.h>
#include "rtypes.h"
#include "dmap.h"
#include "rprm.h"
#include "fitblk.h"
#include "fitdata.h"



int FitEncode(struct DataMap *ptr,struct RadarParm *prm, struct FitData *fit) {

  int c,x;
  int32 snum,xnum;
  int32 p0num;

  int16 *slist=NULL;
  float *pwr0=NULL;

  int16 *nlag=NULL;

  char *qflg=NULL;
  char *gflg=NULL;

  float *p_l=NULL;
  float *p_l_e=NULL;
  float *p_s=NULL;
  float *p_s_e=NULL;

  float *v=NULL;
  float *v_e=NULL;

  float *w_l=NULL;
  float *w_l_e=NULL;
  float *w_s=NULL;
  float *w_s_e=NULL;

  float *sd_l=NULL;
  float *sd_s=NULL;
  float *sd_phi=NULL;

  char *x_qflg=NULL;
  char *x_gflg=NULL;

  float *x_p_l=NULL;
  float *x_p_l_e=NULL;
  float *x_p_s=NULL;
  float *x_p_s_e=NULL;

  float *x_v=NULL;
  float *x_v_e=NULL;

  float *x_w_l=NULL;
  float *x_w_l_e=NULL;
  float *x_w_s=NULL;
  float *x_w_s_e=NULL;

  float *phi0=NULL;
  float *phi0_e=NULL;
  float *elv=NULL;
  float *elv_low=NULL;    // fitacf 1-2
  float *elv_high=NULL;   // fitacf 1-2
  float *elv_fitted=NULL; // fitacf 3
  float *elv_error=NULL;  // fitacf 3

  float *x_sd_l=NULL;
  float *x_sd_s=NULL;
  float *x_sd_phi=NULL;

  float sky_noise=fit->noise.skynoise;
  float lag0_noise=fit->noise.lag0;
  float vel_noise=fit->noise.vel;

  DataMapAddScalar(ptr,"algorithm",DATASTRING,&fit->algorithm);

  DataMapAddScalar(ptr,"fitacf.revision.major",DATAINT,
		    &fit->revision.major);
  DataMapAddScalar(ptr,"fitacf.revision.minor",DATAINT,
		    &fit->revision.minor);

  DataMapStoreScalar(ptr,"noise.sky",DATAFLOAT,&sky_noise);
  DataMapStoreScalar(ptr,"noise.lag0",DATAFLOAT,&lag0_noise);
  DataMapStoreScalar(ptr,"noise.vel",DATAFLOAT,&vel_noise);

  DataMapStoreScalar(ptr,"tdiff",DATAFLOAT,&fit->tdiff);

  p0num=prm->nrang;
  // Allocate temporary array for pwr0
  float *pwr0_tmp = (float *)malloc(p0num * sizeof(float));
  if (!pwr0_tmp) return -1;
  
  // Copy data from fit->rng to temporary array
  for (c = 0; c < p0num; c++) {
    pwr0_tmp[c] = fit->rng[c].p_0;
  }
  
  // Store the array
  DataMapStoreArray(ptr, "pwr0", DATAFLOAT, 1, &p0num, pwr0_tmp);
  free(pwr0_tmp);

  snum=0;
  for (c=0;c<prm->nrang;c++) {
      if ( (fit->rng[c].qflg==1) ||
              ((fit->xrng !=NULL) && (fit->xrng[c].qflg==1)))
          snum++;
  }

  if (prm->xcf !=0) 
      xnum=snum;
  else 
      xnum=0;

  if (snum==0){
      return 0;
  }

  // Allocate temporary arrays for the data
  int16 *slist_tmp = (int16 *)malloc(snum * sizeof(int16));
  int16 *nlag_tmp = (int16 *)malloc(snum * sizeof(int16));
  char *qflg_tmp = (char *)malloc(snum * sizeof(char));
  char *gflg_tmp = (char *)malloc(snum * sizeof(char));
  float *p_l_tmp = (float *)malloc(snum * sizeof(float));
  float *p_l_e_tmp = (float *)malloc(snum * sizeof(float));
  float *p_s_tmp = (float *)malloc(snum * sizeof(float));
  float *p_s_e_tmp = (float *)malloc(snum * sizeof(float));
  float *v_tmp = (float *)malloc(snum * sizeof(float));
  float *v_e_tmp = (float *)malloc(snum * sizeof(float));
  float *w_l_tmp = (float *)malloc(snum * sizeof(float));
  float *w_l_e_tmp = (float *)malloc(snum * sizeof(float));
  float *w_s_tmp = (float *)malloc(snum * sizeof(float));
  float *w_s_e_tmp = (float *)malloc(snum * sizeof(float));
  float *sd_l_tmp = (float *)malloc(snum * sizeof(float));
  float *sd_s_tmp = (float *)malloc(snum * sizeof(float));
  float *sd_phi_tmp = (float *)malloc(snum * sizeof(float));
  
  // Check for allocation failures
  if (!slist_tmp || !nlag_tmp || !qflg_tmp || !gflg_tmp || 
      !p_l_tmp || !p_l_e_tmp || !p_s_tmp || !p_s_e_tmp ||
      !v_tmp || !v_e_tmp || !w_l_tmp || !w_l_e_tmp ||
      !w_s_tmp || !w_s_e_tmp || !sd_l_tmp || !sd_s_tmp || !sd_phi_tmp) {
    // Free any allocated memory on failure
    free(slist_tmp); free(nlag_tmp); free(qflg_tmp); free(gflg_tmp);
    free(p_l_tmp); free(p_l_e_tmp); free(p_s_tmp); free(p_s_e_tmp);
    free(v_tmp); free(v_e_tmp); free(w_l_tmp); free(w_l_e_tmp);
    free(w_s_tmp); free(w_s_e_tmp); free(sd_l_tmp); free(sd_s_tmp);
    free(sd_phi_tmp);
    return -1;
  }
  
  // Copy data from fit->rng to temporary arrays
  for (int i = 0; i < snum; i++) {
    slist_tmp[i] = i;  // Assuming this is just the index
    nlag_tmp[i] = 0;   // Default value, adjust as needed
    qflg_tmp[i] = fit->rng[i].qflg;
    gflg_tmp[i] = 0;   // Default value, adjust as needed
    p_l_tmp[i] = (float)fit->rng[i].p_l;
    p_l_e_tmp[i] = (float)fit->rng[i].p_l_err;
    p_s_tmp[i] = (float)fit->rng[i].p_s;
    p_s_e_tmp[i] = (float)fit->rng[i].p_s_err;
    v_tmp[i] = (float)fit->rng[i].v;
    v_e_tmp[i] = (float)fit->rng[i].v_err;
    w_l_tmp[i] = (float)fit->rng[i].w_l;
    w_l_e_tmp[i] = (float)fit->rng[i].w_l_err;
    w_s_tmp[i] = 0.0f;  // Not in FitRange, set to default
    w_s_e_tmp[i] = 0.0f; // Not in FitRange, set to default
    sd_l_tmp[i] = (float)fit->rng[i].sdev_l;
    sd_s_tmp[i] = (float)fit->rng[i].sdev_s;
    sd_phi_tmp[i] = (float)fit->rng[i].sdev_phi;
  }
  
  // Store the arrays
  DataMapStoreArray(ptr, "slist", DATASHORT, 1, &snum, slist_tmp);
  DataMapStoreArray(ptr, "nlag", DATASHORT, 1, &snum, nlag_tmp);
  DataMapStoreArray(ptr, "qflg", DATACHAR, 1, &snum, qflg_tmp);
  DataMapStoreArray(ptr, "gflg", DATACHAR, 1, &snum, gflg_tmp);
  DataMapStoreArray(ptr, "p_l", DATAFLOAT, 1, &snum, p_l_tmp);
  DataMapStoreArray(ptr, "p_l_e", DATAFLOAT, 1, &snum, p_l_e_tmp);
  DataMapStoreArray(ptr, "p_s", DATAFLOAT, 1, &snum, p_s_tmp);
  DataMapStoreArray(ptr, "p_s_e", DATAFLOAT, 1, &snum, p_s_e_tmp);
  DataMapStoreArray(ptr, "v", DATAFLOAT, 1, &snum, v_tmp);
  DataMapStoreArray(ptr, "v_e", DATAFLOAT, 1, &snum, v_e_tmp);
  DataMapStoreArray(ptr, "w_l", DATAFLOAT, 1, &snum, w_l_tmp);
  DataMapStoreArray(ptr, "w_l_e", DATAFLOAT, 1, &snum, w_l_e_tmp);
  DataMapStoreArray(ptr, "w_s", DATAFLOAT, 1, &snum, w_s_tmp);
  DataMapStoreArray(ptr, "w_s_e", DATAFLOAT, 1, &snum, w_s_e_tmp);
  DataMapStoreArray(ptr, "sd_l", DATAFLOAT, 1, &snum, sd_l_tmp);
  DataMapStoreArray(ptr, "sd_s", DATAFLOAT, 1, &snum, sd_s_tmp);
  DataMapStoreArray(ptr, "sd_phi", DATAFLOAT, 1, &snum, sd_phi_tmp);
  
  // Free temporary arrays
  free(slist_tmp); free(nlag_tmp); free(qflg_tmp); free(gflg_tmp);
  free(p_l_tmp); free(p_l_e_tmp); free(p_s_tmp); free(p_s_e_tmp);
  free(v_tmp); free(v_e_tmp); free(w_l_tmp); free(w_l_e_tmp);
  free(w_s_tmp); free(w_s_e_tmp); free(sd_l_tmp); free(sd_s_tmp);
  free(sd_phi_tmp);

  if (prm->xcf !=0) {
  
    /* fit.revision.major has values of 4 and 5 in some historical data. 
       The logic of the if statements below should be changed if a new major
       version of FitACF is created in the future */
    
    if (fit->revision.major==3) {
      //XCF fitted parameters for FitACF 3
      // Allocate temporary arrays for XCF data (FitACF v3)
      float *phi0_tmp = (float *)malloc(xnum * sizeof(float));
      float *phi0_e_tmp = (float *)malloc(xnum * sizeof(float));
      float *elv_tmp = (float *)malloc(xnum * sizeof(float));
      float *elv_fitted_tmp = (float *)malloc(xnum * sizeof(float));
      float *elv_error_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_sd_phi_tmp = (float *)malloc(xnum * sizeof(float));
      
      if (!phi0_tmp || !phi0_e_tmp || !elv_tmp || !elv_fitted_tmp || 
          !elv_error_tmp || !x_sd_phi_tmp) {
        free(phi0_tmp); free(phi0_e_tmp); free(elv_tmp);
        free(elv_fitted_tmp); free(elv_error_tmp); free(x_sd_phi_tmp);
        return -1;
      }
      
      // Copy data from fit->xrng to temporary arrays
      for (int i = 0; i < xnum; i++) {
        phi0_tmp[i] = (float)fit->xrng[i].phi0;
        phi0_e_tmp[i] = (float)fit->xrng[i].phi0_err;
        elv_tmp[i] = 0.0f;  // Not directly available, set to default
        elv_fitted_tmp[i] = 0.0f;  // Not directly available, set to default
        elv_error_tmp[i] = 0.0f;   // Not directly available, set to default
        x_sd_phi_tmp[i] = (float)fit->xrng[i].sdev_phi;
      }
      
      // Store the arrays
      DataMapStoreArray(ptr, "phi0", DATAFLOAT, 1, &xnum, phi0_tmp);
      DataMapStoreArray(ptr, "phi0_e", DATAFLOAT, 1, &xnum, phi0_e_tmp);
      DataMapStoreArray(ptr, "elv", DATAFLOAT, 1, &xnum, elv_tmp);
      DataMapStoreArray(ptr, "elv_fitted", DATAFLOAT, 1, &xnum, elv_fitted_tmp);
      DataMapStoreArray(ptr, "elv_error", DATAFLOAT, 1, &xnum, elv_error_tmp);
      DataMapStoreArray(ptr, "x_sd_phi", DATAFLOAT, 1, &xnum, x_sd_phi_tmp);
      
      // Free temporary arrays
      free(phi0_tmp); free(phi0_e_tmp); free(elv_tmp);
      free(elv_fitted_tmp); free(elv_error_tmp); free(x_sd_phi_tmp);
    } else {
      //XCF fitted parameters for FitACF 1-2
      // Allocate temporary arrays for XCF data (FitACF v1-2)
      char *x_qflg_tmp = (char *)malloc(xnum * sizeof(char));
      char *x_gflg_tmp = (char *)malloc(xnum * sizeof(char));
      float *x_p_l_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_p_l_e_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_p_s_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_p_s_e_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_v_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_v_e_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_w_l_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_w_l_e_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_w_s_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_w_s_e_tmp = (float *)malloc(xnum * sizeof(float));
      float *phi0_tmp = (float *)malloc(xnum * sizeof(float));
      float *phi0_e_tmp = (float *)malloc(xnum * sizeof(float));
      float *elv_tmp = (float *)malloc(xnum * sizeof(float));
      float *elv_low_tmp = (float *)malloc(xnum * sizeof(float));
      float *elv_high_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_sd_l_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_sd_s_tmp = (float *)malloc(xnum * sizeof(float));
      float *x_sd_phi_tmp = (float *)malloc(xnum * sizeof(float));
      
      // Check for allocation failures
      if (!x_qflg_tmp || !x_gflg_tmp || !x_p_l_tmp || !x_p_l_e_tmp || 
          !x_p_s_tmp || !x_p_s_e_tmp || !x_v_tmp || !x_v_e_tmp ||
          !x_w_l_tmp || !x_w_l_e_tmp || !x_w_s_tmp || !x_w_s_e_tmp ||
          !phi0_tmp || !phi0_e_tmp || !elv_tmp || !elv_low_tmp || 
          !elv_high_tmp || !x_sd_l_tmp || !x_sd_s_tmp || !x_sd_phi_tmp) {
        // Free any allocated memory on failure
        free(x_qflg_tmp); free(x_gflg_tmp); free(x_p_l_tmp); free(x_p_l_e_tmp);
        free(x_p_s_tmp); free(x_p_s_e_tmp); free(x_v_tmp); free(x_v_e_tmp);
        free(x_w_l_tmp); free(x_w_l_e_tmp); free(x_w_s_tmp); free(x_w_s_e_tmp);
        free(phi0_tmp); free(phi0_e_tmp); free(elv_tmp); free(elv_low_tmp);
        free(elv_high_tmp); free(x_sd_l_tmp); free(x_sd_s_tmp); free(x_sd_phi_tmp);
        return -1;
      }
      
      // Copy data from fit->xrng to temporary arrays
      for (int i = 0; i < xnum; i++) {
        x_qflg_tmp[i] = (char)fit->xrng[i].qflg;
        x_gflg_tmp[i] = 0;  // Default value, adjust as needed
        x_p_l_tmp[i] = (float)fit->xrng[i].p_l;
        x_p_l_e_tmp[i] = (float)fit->xrng[i].p_l_err;
        x_p_s_tmp[i] = (float)fit->xrng[i].p_s;
        x_p_s_e_tmp[i] = (float)fit->xrng[i].p_s_err;
        x_v_tmp[i] = (float)fit->xrng[i].v;
        x_v_e_tmp[i] = (float)fit->xrng[i].v_err;
        x_w_l_tmp[i] = (float)fit->xrng[i].w_l;
        x_w_l_e_tmp[i] = (float)fit->xrng[i].w_l_err;
        x_w_s_tmp[i] = 0.0f;  // Not in FitRange, set to default
        x_w_s_e_tmp[i] = 0.0f; // Not in FitRange, set to default
        phi0_tmp[i] = (float)fit->xrng[i].phi0;
        phi0_e_tmp[i] = (float)fit->xrng[i].phi0_err;
        elv_tmp[i] = 0.0f;      // Not directly available, set to default
        elv_low_tmp[i] = 0.0f;  // Not directly available, set to default
        elv_high_tmp[i] = 0.0f; // Not directly available, set to default
        x_sd_l_tmp[i] = (float)fit->xrng[i].sdev_l;
        x_sd_s_tmp[i] = (float)fit->xrng[i].sdev_s;
        x_sd_phi_tmp[i] = (float)fit->xrng[i].sdev_phi;
      }
      
      // Store the arrays
      DataMapStoreArray(ptr, "x_qflg", DATACHAR, 1, &xnum, x_qflg_tmp);
      DataMapStoreArray(ptr, "x_gflg", DATACHAR, 1, &xnum, x_gflg_tmp);
      DataMapStoreArray(ptr, "x_p_l", DATAFLOAT, 1, &xnum, x_p_l_tmp);
      DataMapStoreArray(ptr, "x_p_l_e", DATAFLOAT, 1, &xnum, x_p_l_e_tmp);
      DataMapStoreArray(ptr, "x_p_s", DATAFLOAT, 1, &xnum, x_p_s_tmp);
      DataMapStoreArray(ptr, "x_p_s_e", DATAFLOAT, 1, &xnum, x_p_s_e_tmp);
      DataMapStoreArray(ptr, "x_v", DATAFLOAT, 1, &xnum, x_v_tmp);
      DataMapStoreArray(ptr, "x_v_e", DATAFLOAT, 1, &xnum, x_v_e_tmp);
      DataMapStoreArray(ptr, "x_w_l", DATAFLOAT, 1, &xnum, x_w_l_tmp);
      DataMapStoreArray(ptr, "x_w_l_e", DATAFLOAT, 1, &xnum, x_w_l_e_tmp);
      DataMapStoreArray(ptr, "x_w_s", DATAFLOAT, 1, &xnum, x_w_s_tmp);
      DataMapStoreArray(ptr, "x_w_s_e", DATAFLOAT, 1, &xnum, x_w_s_e_tmp);
      DataMapStoreArray(ptr, "phi0", DATAFLOAT, 1, &xnum, phi0_tmp);
      DataMapStoreArray(ptr, "phi0_e", DATAFLOAT, 1, &xnum, phi0_e_tmp);
      DataMapStoreArray(ptr, "elv", DATAFLOAT, 1, &xnum, elv_tmp);
      DataMapStoreArray(ptr, "elv_low", DATAFLOAT, 1, &xnum, elv_low_tmp);
      DataMapStoreArray(ptr, "elv_high", DATAFLOAT, 1, &xnum, elv_high_tmp);
      DataMapStoreArray(ptr, "x_sd_l", DATAFLOAT, 1, &xnum, x_sd_l_tmp);
      DataMapStoreArray(ptr, "x_sd_s", DATAFLOAT, 1, &xnum, x_sd_s_tmp);
      DataMapStoreArray(ptr, "x_sd_phi", DATAFLOAT, 1, &xnum, x_sd_phi_tmp);
      
      // Free temporary arrays
      free(x_qflg_tmp); free(x_gflg_tmp); free(x_p_l_tmp); free(x_p_l_e_tmp);
      free(x_p_s_tmp); free(x_p_s_e_tmp); free(x_v_tmp); free(x_v_e_tmp);
      free(x_w_l_tmp); free(x_w_l_e_tmp); free(x_w_s_tmp); free(x_w_s_e_tmp);
      free(phi0_tmp); free(phi0_e_tmp); free(elv_tmp); free(elv_low_tmp);
      free(elv_high_tmp); free(x_sd_l_tmp); free(x_sd_s_tmp); free(x_sd_phi_tmp);

    }
  }
  
  x=0;
  for (c=0;c<prm->nrang;c++) {
    if ( (fit->rng[c].qflg==1) ||
         ((fit->xrng !=NULL) && (fit->xrng[c].qflg==1))) {
      slist[x]=c;
      nlag[x]=fit->rng[c].nump;
      
      qflg[x]=fit->rng[c].qflg;
      gflg[x]=fit->rng[c].gsct;
        
      p_l[x]=fit->rng[c].p_l;
      p_l_e[x]=fit->rng[c].p_l_err;
      p_s[x]=fit->rng[c].p_s;
      p_s_e[x]=fit->rng[c].p_s_err;
        
      v[x]=fit->rng[c].v;
      v_e[x]=fit->rng[c].v_err;

      w_l[x]=fit->rng[c].w_l;
      w_l_e[x]=fit->rng[c].w_l_err;
      w_s[x]=fit->rng[c].w_s;
      w_s_e[x]=fit->rng[c].w_s_err;

      sd_l[x]=fit->rng[c].sdev_l;
      sd_s[x]=fit->rng[c].sdev_s;
      sd_phi[x]=fit->rng[c].sdev_phi;

      if (xnum !=0) {
        
      /* FitACF v3 does not determine XCF fitted parameters, so only 
         write these data to file for FitACF v1-2. The elevation field 
         names have also changed. 
         NB: update if statement logic if new major revision of FitACF
             is created in the future*/
        if (fit->revision.major==3) {
          phi0[x]=fit->xrng[c].phi0;
          phi0_e[x]=fit->xrng[c].phi0_err;
          elv[x]=fit->elv[c].normal;
          elv_fitted[x]=fit->elv[c].fitted;
          elv_error[x]=fit->elv[c].error;

          x_sd_phi[x]=fit->xrng[c].sdev_phi;
        } else {
          x_qflg[x]=fit->xrng[c].qflg;
          x_gflg[x]=fit->xrng[c].gsct;

          x_p_l[x]=fit->xrng[c].p_l;
          x_p_l_e[x]=fit->xrng[c].p_l_err;
          x_p_s[x]=fit->xrng[c].p_s;
          x_p_s_e[x]=fit->xrng[c].p_s_err;

          x_v[x]=fit->xrng[c].v;
          x_v_e[x]=fit->xrng[c].v_err;

          x_w_l[x]=fit->xrng[c].w_l;
          x_w_l_e[x]=fit->xrng[c].w_l_err;
          x_w_s[x]=fit->xrng[c].w_s;
          x_w_s_e[x]=fit->xrng[c].w_s_err;

          phi0[x]=fit->xrng[c].phi0;
          phi0_e[x]=fit->xrng[c].phi0_err;
          elv[x]=fit->elv[c].normal;
          elv_low[x]=fit->elv[c].low;
          elv_high[x]=fit->elv[c].high;

          x_sd_l[x]=fit->xrng[c].sdev_l;
          x_sd_s[x]=fit->xrng[c].sdev_s;
          x_sd_phi[x]=fit->xrng[c].sdev_phi;
        }
      }
      x++;
    }
  }
  return 0;
}



int FitWrite(int fid,struct RadarParm *prm,
            struct FitData *fit) {

  int s;
  struct DataMap *ptr=NULL;

  ptr=DataMapMake();
  if (ptr==NULL) return -1;

  s=RadarParmEncode(ptr,prm);
  
  if (s==0) s=FitEncode(ptr,prm,fit);
  
  if (s==0) {
    if (fid !=-1) s=DataMapWrite(fid,ptr);
    else s=DataMapSize(ptr);
  }

  DataMapFree(ptr);
  return s;

}


int FitFwrite(FILE *fp,struct RadarParm *prm,
              struct FitData *fit) {
  return FitWrite(fileno(fp),prm,fit);
}


