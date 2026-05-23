# Makefile for SuperDARN Grid Library (Optimized Version 1.24)
# Uses standard RST build system with optimization enhancements.
# Produces libgrdopt.* (separate from libgrd.*) so the original
# grid.1.24 library keeps providing the file-I/O symbols
# (GridFseek, GridIndexFload, GridFread) that this optimized
# variant does not implement.

include $(MAKECFG).$(SYSTEM)

INCLUDE=-I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn

# gridio_parallel.c, gridmerge_parallel.c, mergegrid_parallel.c, and
# gridseek_optimized.c are excluded -- they depend on Intel SVML intrinsics
# (_mm256_sin_pd/cos_pd) not available in GCC, undefined GridFilterParams,
# and a DataMapFindScalar signature that doesn't match the real RST API.
# They need real refactoring before they can compile.
SRC = addgrid_parallel.c avggrid_parallel.c copygrid_parallel.c \
      filtergrid_parallel.c grid_parallel_utils.c gridseek_parallel.c \
      integrategrid_parallel.c sortgrid_parallel.c

OBJS = addgrid_parallel.o avggrid_parallel.o copygrid_parallel.o \
       filtergrid_parallel.o grid_parallel_utils.o gridseek_parallel.o \
       integrategrid_parallel.o sortgrid_parallel.o

INC=$(IPATH)/superdarn
LINK="1"
DSTPATH=$(LIBPATH)
OUTPUT = grdopt

CFLAGS += -O2 -fopenmp

ifdef AVX2
CFLAGS += -mavx2 -mfma -DAVX2_ENABLED
endif

include $(MAKELIB).$(SYSTEM)
