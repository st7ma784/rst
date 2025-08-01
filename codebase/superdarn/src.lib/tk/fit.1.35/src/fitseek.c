/* fitseek.c
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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <zlib.h>
#include "rtypes.h"
#include "rtime.h"
#include "dmap.h"
#include "fitindex.h"




double FitGetTime(struct DataMap *ptr) {
  struct DataMapScalar *s;
  int c;
  int yr=0,mo=0,dy=0,hr=0,mt=0,sc=0;
  int32 us=0;
  for (c=0;c<ptr->snum;c++) {
    s=ptr->scl[c];
    if ((strcmp(s->name,"time.yr")==0) && (s->type==DATASHORT)) 
      yr=*(s->data.sptr);
    if ((strcmp(s->name,"time.mo")==0) && (s->type==DATASHORT))
      mo=*(s->data.sptr);
    if ((strcmp(s->name,"time.dy")==0) && (s->type==DATASHORT))
      dy=*(s->data.sptr);
    if ((strcmp(s->name,"time.hr")==0) && (s->type==DATASHORT))
      hr=*(s->data.sptr);
    if ((strcmp(s->name,"time.mt")==0) && (s->type==DATASHORT))
      mt=*(s->data.sptr);
    if ((strcmp(s->name,"time.sc")==0) && (s->type==DATASHORT))
      sc=*(s->data.sptr);
    if ((strcmp(s->name,"time.us")==0) && (s->type==DATAINT))
      us=*(s->data.iptr);
   }
   if (yr==0) return -1;
   int usec = (int)(us % 1000000);
   return TimeYMDHMSToEpoch(yr, mo, dy, hr, mt, sc, usec);
}


int FitSeek(int fid,
	    int yr,int mo,int dy,int hr,int mt,int sc,double *atme,
            struct FitIndex *inx) {

  int fptr=0,tptr=0;
  struct DataMap *ptr;
  double tfile=0,tval;

  tval=TimeYMDHMSToEpoch(yr, mo, dy, hr, mt, sc, 0);

  if (inx !=NULL) {
    int rec=0,prec=-1;
    int srec;
    int erec;

    /* search using index records */

    double stime,etime;
    stime=inx->tme[0];
    etime=inx->tme[inx->num-1];

    srec=0;
    erec=inx->num;
    if (tval<stime) { /* before start of file */
      if (atme !=NULL) *atme=stime;
      lseek(fid,inx->inx[srec],SEEK_SET);
      return 0;
    } else if (tval>etime) { /* after end of file */
      if (atme !=NULL) *atme=stime;
      lseek(fid,inx->inx[erec-1],SEEK_SET);
      return -1;
    }
    do {
      prec=rec;
      rec=srec+(int) ((tval-stime)*(erec-srec)/(etime-stime));
      if (inx->tme[rec]==tval) break;
      if (inx->tme[rec]<tval) {
         srec=rec;
         stime=inx->tme[rec];
      } else {
         erec=rec;
         etime=inx->tme[rec];
      }
    } while (prec != rec);
    if (atme !=NULL) *atme=inx->tme[rec]; 
    lseek(fid,inx->inx[rec],SEEK_SET);
    return 0;
  } else {
    fptr=lseek(fid,0,SEEK_CUR);
    if (DataMapRead(fid, &ptr) != 0 || ptr == NULL) {
      return -1;
    } else {
      tfile=FitGetTime(ptr);
      DataMapFree(ptr);
      if (tfile>tval) {
        fptr=lseek(fid,0,SEEK_SET);
      }
    }
    if (atme!=NULL) *atme=tfile;
    while (tval>=tfile) {
      tptr=lseek(fid,0,SEEK_CUR);
      if (DataMapRead(fid, &ptr) != 0) {
        return -1;
      }
      if (ptr==NULL) break;
      tfile=FitGetTime(ptr);
      DataMapFree(ptr);
      /*if (tval>=tfile) fptr=tptr;*/
      fptr=tptr;
      if (atme !=NULL) *atme=tfile;
    } 
    if (tval>tfile) return -1;
    lseek(fid,fptr,SEEK_SET);
  }
  return 0;
}



int FitFseek(FILE *fp,
	    int yr,int mo,int dy,int hr,int mt,int sc,double *atme,
            struct FitIndex *inx) {
  return FitSeek(fileno(fp),yr,mo,dy,hr,mt,sc,atme,inx);
}
