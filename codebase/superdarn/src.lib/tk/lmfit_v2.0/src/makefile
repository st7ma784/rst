# makefile for lmfit 2.0
include $(MAKECFG).$(SYSTEM)

INCLUDE= -I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn -I$(IPATH)/analysis

SRC = lmfit_determinations.c error_estimates.c lmfit_fitting.c lmfit_leastsquares.c llist.c lmfit2toplevel.c lmfit_preprocessing.c selfclutter.c lmfit_determinations.h error_estimates.h lmfit_fitting.h lmfit_leastsquares.h llist.h lmfit2toplevel.h lmfit_preprocessing.h selfclutter.h
OBJS = lmfit_determinations.o error_estimates.o lmfit_fitting.o lmfit_leastsquares.o llist.o lmfit2toplevel.o lmfit_preprocessing.o selfclutter.o
INC=$(IPATH)/superdarn
DSTPATH=$(LIBPATH)
OUTPUT=lmfit
LINK="2.0"

# Build targets
.PHONY: all clean test benchmark profile

all: $(OUTPUT)

# Profile-guided optimization build
profile: clean
	$(MAKE) CFLAGS="$(CFLAGS) -fprofile-generate" all
	@echo "Run your typical workload now to generate profile data..."
	@echo "Then run 'make optimized' to build with profile data"

optimized:
	$(MAKE) CFLAGS="$(CFLAGS) -fprofile-use" all

# Benchmark target
benchmark: all
	@echo "Building performance benchmark..."
	$(CC) $(CFLAGS) -DBENCHMARK_MODE -o grid_benchmark benchmark/grid_benchmark.c $(OBJS) $(LDFLAGS)

# Test target
test: all
	@echo "Building unit tests..."
	$(CC) $(CFLAGS) -DTEST_MODE -o grid_test tests/grid_test.c $(OBJS) $(LDFLAGS)

# Debugging build
debug:
	$(MAKE) CFLAGS="$(CFLAGS) -DDEBUG -g -O0" all

# Clean target
clean:
	rm -f *.o *.gcda *.gcno $(OUTPUT) grid_benchmark grid_test

# Documentation
docs:
	doxygen ../docs/Doxyfile

# Static analysis
analyze:
	cppcheck --enable=all --suppress=missingIncludeSystem $(SRC)

# Performance analysis
perf-analysis: benchmark
	@echo "Run: perf record -g ./grid_benchmark"
	@echo "Then: perf report"


include $(MAKELIB).$(SYSTEM)
