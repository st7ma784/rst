# Makefile for Optimized Fit Speck Removal Tool
# =============================================

include $(MAKECFG).$(SYSTEM)

INCLUDE=-I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn

OBJS = fit_speck_removal_optimized.o
SRC=hlpstr.h errstr.h fit_speck_removal_optimized.c
DSTPATH = $(BINPATH)
OUTPUT = fit_speck_removal_optimized
LIBS=-lradar.1 -lfit.1 -ldmap.1 -lopt.1 -lrcnv.1 
SLIB=-lm -lz

# Optimization flags
CFLAGS += -O3 -march=native -mtune=native -funroll-loops -ffast-math
CFLAGS += -flto -fomit-frame-pointer

# OpenMP support
ifdef USE_OPENMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

# AVX2 SIMD support (auto-detect)
SIMD_FLAGS = $(shell gcc -march=native -dM -E - < /dev/null | grep -q __AVX2__ && echo "-mavx2 -mfma")
CFLAGS += $(SIMD_FLAGS)

# Memory optimization
CFLAGS += -DCACHE_LINE_SIZE=64 -falign-functions=64 -falign-loops=64

# Debugging (optional)
ifdef DEBUG
CFLAGS += -g -O0 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address
else
CFLAGS += -DNDEBUG
endif

# Profile-guided optimization (optional)
ifdef PGO
CFLAGS += -fprofile-generate
LDFLAGS += -fprofile-generate
endif

include $(MAKEBIN).$(SYSTEM)


