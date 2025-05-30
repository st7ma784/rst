# Makefile for Optimized Fit Speck Removal Tool
# =============================================

include $(MAKECFG).$(SYSTEM)

INCLUDE_PATH = -I$(IPATH)/general -I$(IPATH)/base -I$(IPATH)/superdarn
LIBRARY_PATH = -L$(LIBPATH) -lfit.1 -ldmap.1 -lopt.1 -lrtime.1 -lrtypes.1

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

OBJS = fit_speck_removal_optimized.o
TARGETS = fit_speck_removal_optimized

all: $(TARGETS)

fit_speck_removal_optimized: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBRARY_PATH)

fit_speck_removal_optimized.o: fit_speck_removal_optimized.c
	$(CC) $(CFLAGS) $(INCLUDE_PATH) -c $<

# Performance benchmarking
benchmark: fit_speck_removal_optimized
	@echo "Running performance benchmark..."
	@echo "Single-threaded:"
	@time ./fit_speck_removal_optimized --no-parallel test_data.fit > /dev/null
	@echo "Multi-threaded (auto):"
	@time ./fit_speck_removal_optimized test_data.fit > /dev/null
	@echo "Multi-threaded (4 threads):"
	@time ./fit_speck_removal_optimized --threads=4 test_data.fit > /dev/null

# Memory profiling with valgrind (if available)
memcheck: fit_speck_removal_optimized
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "Running memory check..."; \
		valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
		./fit_speck_removal_optimized test_data.fit > /dev/null; \
	else \
		echo "Valgrind not available for memory checking"; \
	fi

# Performance profiling with gprof
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: fit_speck_removal_optimized
	@echo "Running with profiling enabled..."
	@./fit_speck_removal_optimized test_data.fit > /dev/null
	@gprof fit_speck_removal_optimized gmon.out > profile_report.txt
	@echo "Profile report generated: profile_report.txt"

# Generate optimized build with profile feedback
pgo-build: 
	@echo "Building with Profile-Guided Optimization..."
	@$(MAKE) clean
	@$(MAKE) PGO=1 fit_speck_removal_optimized
	@echo "Training with sample data..."
	@./fit_speck_removal_optimized test_data.fit > /dev/null || true
	@$(MAKE) clean
	@$(MAKE) CFLAGS="$(CFLAGS) -fprofile-use" fit_speck_removal_optimized
	@echo "PGO build complete"

# Thread scaling test
thread-test: fit_speck_removal_optimized
	@echo "Testing thread scaling performance..."
	@for threads in 1 2 4 8; do \
		echo "Testing with $$threads threads:"; \
		time ./fit_speck_removal_optimized --threads=$$threads test_data.fit > /dev/null; \
		echo ""; \
	done

install: fit_speck_removal_optimized
	$(INSTALL_BIN) $< $(BINPATH)

clean:
	rm -f *.o $(TARGETS) gmon.out profile_report.txt *.gcda *.gcno

.PHONY: all benchmark memcheck profile pgo-build thread-test install clean
