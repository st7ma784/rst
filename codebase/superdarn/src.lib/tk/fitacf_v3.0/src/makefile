# makefile for fitacf 3.0
#
#
include $(MAKECFG).$(SYSTEM)



INCLUDE= -I$(IPATH)/base -I$(IPATH)/general  -I$(IPATH)/superdarn/fit -I$(IPATH)/superdarn
CFLAGS += -fopenmp
SRC = determinations.c fitacftoplevel.c fitting.c leastsquares.c llist.c \
	preprocessing.c fitacf_array_optimized.c \
	determinations.h fit_structures.h fit_structures_array.h \
	leastsquares.h preprocessing.h fitacftoplevel.h fitting.h llist.h
OBJS = determinations.o  fitacftoplevel.o  fitting.o  leastsquares.o  llist.o \
	preprocessing.o fitacf_array_optimized.o
INC=$(IPATH)/superdarn
DSTPATH=$(LIBPATH)
OUTPUT=fitacf
LINK="3.0"



include $(MAKELIB).$(SYSTEM)
