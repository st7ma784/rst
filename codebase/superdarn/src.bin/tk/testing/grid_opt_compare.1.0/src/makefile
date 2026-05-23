# grid_opt_compare -- equivalence + benchmark of libgrd vs libgrdopt
include $(MAKECFG).$(SYSTEM)

INCLUDE=-I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn
SRC = grid_opt_compare.c adapter.c stubs.c
OBJS = grid_opt_compare.o adapter.o stubs.o
OUTPUT = grid_opt_compare
DSTPATH = $(BINPATH)

LIBS = -lgrdopt.1 -lgrd.1 -ldmap.1 -lrtime.1 -lrcnv.1
SLIB = -lz -lm

include $(MAKEBIN).$(SYSTEM)
