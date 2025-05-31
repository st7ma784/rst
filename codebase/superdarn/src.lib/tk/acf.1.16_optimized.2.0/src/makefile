# Makefile for ACF Library (Optimized Version 2.0)
# Author: R.J.Barnes (Original), SuperDARN Optimization Framework (Optimized)

include $(MAKECFG).$(SYSTEM)

# Use same include structure as working original
INCLUDE=-I$(IPATH)/base -I$(IPATH)/superdarn

SRC=acfcalculate.c average.c badlag.c normalize.c sumpower.c sumproduct.c

# Use same object structure as working original
OBJS = acfcalculate.o average.o badlag.o normalize.o sumpower.o sumproduct.o
INC=$(IPATH)/superdarn
LINK="1"
DSTPATH=$(LIBPATH)
OUTPUT = acf

# Optimization flags - enabled by default
CFLAGS += -O3 -march=native -mtune=native

# OpenMP support
ifdef OPENMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

# AVX2 support
ifdef AVX2
CFLAGS += -mavx2 -mfma
endif

# Windows-specific optimizations
ifeq ($(SYSTEM),win32)
ifdef MSVC
CFLAGS += /O2 /arch:AVX2
ifdef OPENMP
CFLAGS += /openmp
endif
endif
endif

include $(MAKELIB).$(SYSTEM)
