# Makefile for fitacf_v3.0_optimized2 library
# ============================================
# Optimized SuperDARN FitACF processing library
# by GitHub Copilot Assistant
#
include $(MAKECFG).$(SYSTEM)

INCLUDE=-I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn
SRC = src/fit_structures_optimized.c src/preprocessing_optimized.c \
      src/fitting_optimized.c src/fitacf_toplevel_optimized.c
OBJS = fit_structures_optimized.o preprocessing_optimized.o \
       fitting_optimized.o fitacf_toplevel_optimized.o
INC=$(IPATH)/superdarn
LINK="1"
DSTPATH=$(LIBPATH)
OUTPUT = fitacf_v3.0_optimized2

include $(MAKELIB).$(SYSTEM)
