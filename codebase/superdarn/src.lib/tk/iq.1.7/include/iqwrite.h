/* iqwrite.h
   ======== 
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




#ifndef _IQWRITE_H
#define _IQWRITE_H


int IQEncode(struct DataMap *ptr,struct IQ *iq,unsigned int *badtr,int16 *samples);

int IQWrite(int fid,struct RadarParm *prm,
	    struct IQ *iq,unsigned int *badtr,int16 *samples);

int IQFwrite(FILE *fp,struct RadarParm *prm,
	     struct IQ *iq,unsigned int *badtr,int16 *samples);

#endif

