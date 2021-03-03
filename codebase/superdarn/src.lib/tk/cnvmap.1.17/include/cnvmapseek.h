/* cnvmapseek.h
   ============
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
 
 RST is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with RST.  If not, see <http://www.gnu.org/licenses/>.
 
 
 
*/




#ifndef _CNVMAPSEEK_H
#define _CNVMAPSEEK_H

int CnvMapSeek(int fid,
	     int yr,int mo,int dy,int hr,int mt,int sc,double *atme,
	     struct CnvMapIndex *inx);
int CnvMapFseek(FILE *fp,
	      int yr,int mo,int dy,int hr,int mt,int sc,double *atme,
	      struct CnvMapIndex *inx);

#endif



