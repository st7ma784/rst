/* graphic.h
   ======
   Author: R.J.Barnes
*/


/*
 LICENSE AND DISCLAIMER
 
 Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory
 
 This file is part of the Radar Software Toolkit (RST).
 
 RST is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with RST.  If not, see <http://www.gnu.org/licenses/>.
 
 
 
*/




int txtbox(char *fntname,float sze,int num,char *txt,float *box,void *data);

void plot_frame(struct Grplot *grplot,int pmax,double stime,
                double etime,int dflg,
                unsigned int color,float width,char *fontname,
                float fontsize);

void plot_pos(struct Grplot *grplot,int p,
              struct posdata *pos,double stime,double etime,int mode,
              unsigned int color,float width,char *fontname,
              float fontsize);

void plot_plasma(struct Grplot *grplot,int p,struct plasmadata *plasma,
              double stime,double etime,int mode,int *pflg,
              unsigned int color,float width,char *fontname,
	      float fontsize);

void plot_imf(struct Grplot *grplot,int p,struct imfdata *imf,
              double stime,double etime,int mode,int *pflg,
              unsigned int color,float width,char *fontname,
              float fontsize);

void plot_title(struct Grplot *grplot,double stime,double etime,
                char *sat,char *plot,int mode,
                unsigned int color,float width,char *fontname,
                float fontsize);








