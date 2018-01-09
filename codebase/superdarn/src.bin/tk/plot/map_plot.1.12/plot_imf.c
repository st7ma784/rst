/* plot_imf.c
   ========== 
   Author: R.J.Barnes
*/


/*
 LICENSE AND DISCLAIMER
 
 Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory
 
 This file is part of the Radar Software Toolkit (RST).
 
 RST is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 any later version.
 
 RST is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public License
 along with RST.  If not, see <http://www.gnu.org/licenses/>.
 
 
 
*/



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rmath.h"
#include "rfbuffer.h"
#include "iplot.h"
#include "rtime.h"
#include "text_box.h"


void plot_imf(struct Plot *plot, float xoff,float yoff,float rad,
              float bx,float by,float bz,float mxval,
              unsigned int color,unsigned char mask,float width,
              char *fontname,float fontsize, void *txtdata)
{
  char txt[256];
  float txbox[3];

  float ang;
  float mag;
  float px,py;
  mag=sqrt(bz*bz+by*by);
  ang=atan(bz/by);
  if (by<0) ang+=PI;  

  sprintf(txt,"%.2d nT",(int) mag);
 
  mag=mag*rad/mxval;
  
  txtbox(fontname,fontsize,strlen(txt),txt,txbox,txtdata);
 
  PlotText(plot,NULL,fontname,fontsize,xoff-rad-4-txbox[0],yoff-0.8*rad,
            strlen(txt),txt,color,mask,1);

  px=xoff+mag*cos(ang);
  py=yoff-mag*sin(ang);
  
  PlotLine(plot,xoff,yoff,px,py,color,0x0f,width,NULL);

}

