# makefile for lmfit 2.0 - OPTIMIZED VERSION
# Added OpenMP and SIMD optimization flags

include $(MAKECFG).$(SYSTEM)

# Optimization flags
CFLAGS += -fopenmp -mavx2 -mfma -O3 -funroll-loops -ffast-math -march=native
CFLAGS += -DLMFIT_PERFORMANCE_LOGGING -DLMFIT_USE_OPTIMIZED

INCLUDE= -I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn -I$(IPATH)/analysis

# Updated to use optimized versions where available
SRC = lmfit_determinations_optimized.c error_estimates.c lmfit_fitting_optimized.c lmfit_leastsquares_optimized.c llist.c lmfit2toplevel.c lmfit_preprocessing_optimized.c selfclutter.c lmfit_determinations.h error_estimates.h lmfit_fitting.h lmfit_leastsquares.h llist.h lmfit2toplevel.h lmfit_preprocessing.h selfclutter.h lmfit_optimized.h
OBJS = lmfit_determinations_optimized.o error_estimates.o lmfit_fitting_optimized.o lmfit_leastsquares_optimized.o llist.o lmfit2toplevel.o lmfit_preprocessing_optimized.o selfclutter.o
INC=$(IPATH)/superdarn
DSTPATH=$(LIBPATH)
OUTPUT=lmfit
LINK="2.0_optimized.4.0"



include $(MAKELIB).$(SYSTEM)


