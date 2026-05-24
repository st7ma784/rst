# fitacf_opt_compare -- equivalence + benchmark harness for libfitacf.3.0
include $(MAKECFG).$(SYSTEM)

INCLUDE = -I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn \
          -I$(IPATH)/superdarn/fit
SRC  = fitacf_opt_compare.c
OBJS = fitacf_opt_compare.o
OUTPUT  = fitacf_opt_compare
DSTPATH = $(BINPATH)

LIBS = -lfitacf.3.0 -lfit.1 -lraw.1 -lradar.1 -lelevation.1 -ldmap.1 \
       -lrtime.1 -lrcnv.1 -lrmath.1 -lopt.1 -lmpfit.1
SLIB = -lz -lm -lpthread

include $(MAKEBIN).$(SYSTEM)
