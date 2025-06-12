# Makefile for SuperDARN Grid Library (Optimized Version 1.24)
# Uses standard RST build system with optimization enhancements

include $(MAKECFG).$(SYSTEM)

# Standard RST include paths
INCLUDE=-I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn

INC=$(IPATH)/superdarn
LINK="1"
DSTPATH=$(LIBPATH)
OUTPUT = grd

# Optimization flags - compatible with RST build system
CFLAGS += -O2

# OpenMP support (optional)
ifdef OPENMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

# AVX2 support (optional)
ifdef AVX2
CFLAGS += -mavx2 -mfma -DAVX2_ENABLED
endif

include $(MAKELIB).$(SYSTEM)
