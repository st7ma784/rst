# Makefile for SuperDARN Grid Parallel Library v1.24
# Enhanced build system with OpenMP, CUDA, and optimization support

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -ffast-math -march=native -mtune=native
OMPFLAGS = -fopenmp
AVXFLAGS = -mavx2 -mfma
CUDAFLAGS = -O3 -arch=sm_60 -Xcompiler -fPIC
INCLUDES = -Iinclude -I../../../base/src.lib/misc/include
LDFLAGS = -lm -fopenmp

# CUDA settings (optional)
CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)
ifdef CUDA_AVAILABLE
    CUDA_ENABLED = 1
    CFLAGS += -DCUDA_ENABLED
    CUDALIBS = -lcudart -lcublas
else
    CUDA_ENABLED = 0
    CUDALIBS =
endif

# AVX2 support detection
AVX2_SUPPORT := $(shell echo | $(CC) -mavx2 -dM -E - 2>/dev/null | grep -q __AVX2__ && echo 1 || echo 0)
ifeq ($(AVX2_SUPPORT),1)
    CFLAGS += -DAVX2_ENABLED $(AVXFLAGS)
endif

# Directory structure
SRCDIR = src
INCDIR = include
TESTDIR = test
BUILDDIR = build
OBJDIR = $(BUILDDIR)/obj
LIBDIR = $(BUILDDIR)/lib
BINDIR = $(BUILDDIR)/bin

# Source files
CSRCS = $(wildcard $(SRCDIR)/*.c)
CUSRCS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(CSRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
CUOBJS = $(CUSRCS:$(SRCDIR)/%.cu=$(OBJDIR)/%.cu.o)

# Test files
TESTSRCS = $(wildcard $(TESTDIR)/*.c)
TESTOBJS = $(TESTSRCS:$(TESTDIR)/%.c=$(OBJDIR)/test_%.o)
TESTBINS = $(TESTSRCS:$(TESTDIR)/%.c=$(BINDIR)/test_%)

# Library name
LIBNAME = libgrid_parallel
STATIC_LIB = $(LIBDIR)/$(LIBNAME).a
SHARED_LIB = $(LIBDIR)/$(LIBNAME).so

# Default target
all: directories $(STATIC_LIB) $(SHARED_LIB) tests

# Create directory structure
directories:
	@mkdir -p $(OBJDIR) $(LIBDIR) $(BINDIR)

# Compile C source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCLUDES) -fPIC -c $< -o $@

# Compile CUDA source files
$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
ifeq ($(CUDA_ENABLED),1)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@
else
	@echo "CUDA not available, skipping $<"
	@touch $@
endif

# Create static library
$(STATIC_LIB): $(OBJS) $(CUOBJS)
	@echo "Creating static library..."
	ar rcs $@ $^

# Create shared library
$(SHARED_LIB): $(OBJS) $(CUOBJS)
	@echo "Creating shared library..."
ifeq ($(CUDA_ENABLED),1)
	$(CC) -shared -fPIC $^ -o $@ $(LDFLAGS) $(CUDALIBS)
else
	$(CC) -shared -fPIC $^ -o $@ $(LDFLAGS)
endif

# Compile test files
$(OBJDIR)/test_%.o: $(TESTDIR)/%.c
	@echo "Compiling test $<..."
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

# Link test executables
$(BINDIR)/test_%: $(OBJDIR)/test_%.o $(STATIC_LIB)
	@echo "Linking test $@..."
ifeq ($(CUDA_ENABLED),1)
	$(CC) $< -L$(LIBDIR) -lgrid_parallel $(LDFLAGS) $(CUDALIBS) -o $@
else
	$(CC) $< -L$(LIBDIR) -lgrid_parallel $(LDFLAGS) -o $@
endif

# Build all tests
tests: directories $(TESTBINS)

# Performance benchmarks
benchmark: tests
	@echo "Running performance benchmarks..."
	@for test in $(TESTBINS); do \
		echo "Running $$test..."; \
		$$test --benchmark; \
	done

# Memory leak detection
memcheck: tests
	@echo "Running memory leak detection..."
	@for test in $(TESTBINS); do \
		echo "Checking $$test with valgrind..."; \
		valgrind --leak-check=full --show-leak-kinds=all $$test; \
	done

# Profile guided optimization
pgo-generate: CFLAGS += -fprofile-generate
pgo-generate: clean all
	@echo "Running tests for profile generation..."
	@for test in $(TESTBINS); do $$test; done

pgo-use: CFLAGS += -fprofile-use -fprofile-correction
pgo-use: clean all

# Installation targets
install: all
	@echo "Installing grid parallel library..."
	install -d $(DESTDIR)/usr/local/include/grid_parallel
	install -m 644 $(INCDIR)/*.h $(DESTDIR)/usr/local/include/grid_parallel/
	install -d $(DESTDIR)/usr/local/lib
	install -m 644 $(STATIC_LIB) $(DESTDIR)/usr/local/lib/
	install -m 755 $(SHARED_LIB) $(DESTDIR)/usr/local/lib/
	ldconfig

uninstall:
	@echo "Uninstalling grid parallel library..."
	rm -rf $(DESTDIR)/usr/local/include/grid_parallel
	rm -f $(DESTDIR)/usr/local/lib/$(LIBNAME).*
	ldconfig

# Documentation
docs:
	@echo "Generating documentation..."
	doxygen docs/Doxyfile

# Cleanup targets
clean:
	rm -rf $(BUILDDIR)
	rm -f $(SRCDIR)/*.gcda $(SRCDIR)/*.gcno

distclean: clean
	rm -rf docs/html docs/latex

# Debug build
debug: CFLAGS += -g -DDEBUG -O0
debug: CFLAGS := $(filter-out -O3 -ffast-math,$(CFLAGS))
debug: all

# Release build with maximum optimization
release: CFLAGS += -DNDEBUG -flto -fuse-linker-plugin
release: LDFLAGS += -flto -fuse-linker-plugin
release: all strip

strip:
	strip $(SHARED_LIB) $(TESTBINS)

# Static analysis
analyze:
	@echo "Running static analysis..."
	cppcheck --enable=all --std=c99 $(SRCDIR)/ $(INCDIR)/
	scan-build make clean all

# Code formatting
format:
	@echo "Formatting code..."
	clang-format -i $(SRCDIR)/*.c $(INCDIR)/*.h $(TESTDIR)/*.c

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build library and tests"
	@echo "  tests      - Build test suite"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  memcheck   - Run memory leak detection"
	@echo "  pgo-*      - Profile guided optimization"
	@echo "  install    - Install library system-wide"
	@echo "  docs       - Generate documentation"
	@echo "  debug      - Build with debug symbols"
	@echo "  release    - Build optimized release"
	@echo "  analyze    - Run static analysis"
	@echo "  format     - Format source code"
	@echo "  clean      - Remove build files"
	@echo "  help       - Show this help"

# Phony targets
.PHONY: all directories tests benchmark memcheck pgo-generate pgo-use install uninstall docs clean distclean debug release strip analyze format help

# Print build configuration
info:
	@echo "Build Configuration:"
	@echo "  CC: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  CUDA_ENABLED: $(CUDA_ENABLED)"
	@echo "  AVX2_SUPPORT: $(AVX2_SUPPORT)"
	@echo "  INCLUDES: $(INCLUDES)"
	@echo "  LDFLAGS: $(LDFLAGS)"
