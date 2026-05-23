# Makefile for SuperDARN Grid Library (Optimized Version 1.24)
# Uses standard RST build system with optimization enhancements.
# Produces libgrdopt.* (separate from libgrd.*) so the original
# grid.1.24 library keeps providing the file-I/O symbols
# (GridFseek, GridIndexFload, GridFread) that this optimized
# variant does not implement.

include $(MAKECFG).$(SYSTEM)

INCLUDE=-I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn

SRC = addgrid_parallel.c avggrid_parallel.c copygrid_parallel.c \
      filtergrid_parallel.c gridio_parallel.c gridmerge_parallel.c \
      grid_parallel_utils.c gridseek_optimized.c gridseek_parallel.c \
      integrategrid_parallel.c mergegrid_parallel.c sortgrid_parallel.c

OBJS = addgrid_parallel.o avggrid_parallel.o copygrid_parallel.o \
       filtergrid_parallel.o gridio_parallel.o gridmerge_parallel.o \
       grid_parallel_utils.o gridseek_optimized.o gridseek_parallel.o \
       integrategrid_parallel.o mergegrid_parallel.o sortgrid_parallel.o

INC=$(IPATH)/superdarn
LINK="1"
DSTPATH=$(LIBPATH)
OUTPUT = grdopt

CFLAGS += -O2 -fopenmp

ifdef AVX2
CFLAGS += -mavx2 -mfma -DAVX2_ENABLED
endif

include $(MAKELIB).$(SYSTEM)
