# Makefile for fitacf_v3.0_optimized2 library
# ============================================
# Optimized SuperDARN FitACF processing library
# by GitHub Copilot Assistant
#
include $(MAKECFG).$(SYSTEM)


INCLUDE= -I$(IPATH)/base -I$(IPATH)/general  -I$(IPATH)/superdarn/fit -I$(IPATH)/superdarn  -I../include
SRC = fit_structures_optimized.c fitacftoplevel_optimized.c fitting_optimized.c fitacf_v3_optimized2.c  \
	preprocessing_optimized.c fit_structures_optimized.h fitacftoplevel_optimized.h fitting_optimized.h fitacf_v3_optimized2.h  \
	preprocessing_optimized.h
OBJS = fit_structures_optimized.o fitacftoplevel_optimized.o fitting_optimized.o fitacf_v3_optimized2.o \
	preprocessing_optimized.o


include $(MAKELIB).$(SYSTEM)

INC=$(IPATH)/superdarn
LINK="1"
DSTPATH=$(LIBPATH)
OUTPUT = fitacf_v3.0_optimized2

# Performance optimizations (conditional)
ifdef USE_OPENMP
    CFLAGS += -fopenmp -DUSE_OPENMP
endif

ifdef USE_SIMD  
    CFLAGS += -mavx2 -mfma -DUSE_SIMD
endif

ifdef USE_CUDA
    CFLAGS += -DUSE_CUDA
endif

# Default performance optimizations
CFLAGS += -O3 -ffast-math -march=native -finline-functions

