# Makefile for SuperDARN Grid Library (Optimized Version 1.24)
# Uses standard RST build system with optimization enhancements.
# Produces libgrdopt.* (separate from libgrd.*) so the original
# grid.1.24 library keeps providing the file-I/O symbols
# (GridFseek, GridIndexFload, GridFread) that this optimized
# variant does not implement.

include $(MAKECFG).$(SYSTEM)

INCLUDE=-I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn

# All 12 grid_parallel/grid_optimized source files now compile.
# Per AUDIT.md Phase A:
#  A1: gridseek_optimized.c -- include order + AVX-512 store buffer width.
#  A2: mergegrid_parallel.c -- SVML sin/cos -> scalar loop.
#  A3: gridmerge_parallel.c -- GridFilterParams defined, anonymous-time
#      compare via GP_TIME_KEY, missing helpers stubbed as static gp_*.
#  A4: gridio_parallel.c    -- DataMapFindScalar/Array via local helpers,
#      anonymous time-struct populated directly, verr.{min,max}, GridSVecOpt.
SRC = addgrid_parallel.c avggrid_parallel.c copygrid_parallel.c \
      filtergrid_parallel.c grid_parallel_utils.c gridio_parallel.c \
      gridmerge_parallel.c gridopt_wrappers.c gridseek_optimized.c \
      gridseek_parallel.c integrategrid_parallel.c mergegrid_parallel.c \
      sortgrid_parallel.c

OBJS = addgrid_parallel.o avggrid_parallel.o copygrid_parallel.o \
       filtergrid_parallel.o grid_parallel_utils.o gridio_parallel.o \
       gridmerge_parallel.o gridopt_wrappers.o gridseek_optimized.o \
       gridseek_parallel.o integrategrid_parallel.o mergegrid_parallel.o \
       sortgrid_parallel.o

INC=$(IPATH)/superdarn
LINK="1"
DSTPATH=$(LIBPATH)
OUTPUT = grdopt

CFLAGS += -O2 -fopenmp

ifdef AVX2
CFLAGS += -mavx2 -mfma -DAVX2_ENABLED
endif

include $(MAKELIB).$(SYSTEM)
