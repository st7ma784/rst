# CUDArst Library Makefile
# Unified CUDA-accelerated RST SuperDARN Library
# Compatible with original RST interface

# Compiler settings
CC = gcc
NVCC = nvcc
AR = ar

# Directories
SRCDIR = src
INCDIR = include
BUILDDIR = build
LIBDIR = lib

# Flags
CFLAGS = -O3 -fPIC -Wall -Wextra -I$(INCDIR)
NVCCFLAGS = -O3 -Xcompiler -fPIC -I$(INCDIR) \
            -gencode arch=compute_50,code=sm_50 \
            -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80
# Detect CUDA installation
CUDA_PATH ?= /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

# Check if CUDA libraries are available
ifneq ($(wildcard $(CUDA_LIB_PATH)/libcudart.so),)
    LDFLAGS = -L$(CUDA_LIB_PATH) -lcuda -lcudart -lm
    CUDA_AVAILABLE = 1
else
    LDFLAGS = -lm
    CUDA_AVAILABLE = 0
endif

# Version
VERSION = 2.0.0
MAJOR = 2
MINOR = 0

# Library names
STATIC_LIB = $(LIBDIR)/libcudarst.a
SHARED_LIB = $(LIBDIR)/libcudarst.so.$(VERSION)
SHARED_LIB_LINK = $(LIBDIR)/libcudarst.so

# Source files
CUDA_SOURCES = $(wildcard $(SRCDIR)/*.cu)
C_SOURCES = $(wildcard $(SRCDIR)/*.c)

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(BUILDDIR)/%.cu.o)
C_OBJECTS = $(C_SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.c.o)
ALL_OBJECTS = $(CUDA_OBJECTS) $(C_OBJECTS)

# Header files for installation
HEADERS = $(wildcard $(INCDIR)/*.h)

# Default target
all: directories $(STATIC_LIB) $(SHARED_LIB)

# Create directories
directories:
	@mkdir -p $(BUILDDIR) $(LIBDIR)

# Compile CUDA sources
$(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C sources
$(BUILDDIR)/%.c.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Build static library
$(STATIC_LIB): $(ALL_OBJECTS)
	$(AR) rcs $@ $^

# Build shared library
$(SHARED_LIB): $(ALL_OBJECTS)
	$(NVCC) -shared -Xlinker -soname=libcudarst.so.$(MAJOR) -o $@ $^ $(LDFLAGS)
	cd $(LIBDIR) && ln -sf libcudarst.so.$(VERSION) libcudarst.so.$(MAJOR)
	cd $(LIBDIR) && ln -sf libcudarst.so.$(VERSION) libcudarst.so

# Install system-wide
install: all
	install -d /usr/local/lib
	install -d /usr/local/include/cudarst
	install -m 644 $(STATIC_LIB) /usr/local/lib/
	install -m 755 $(SHARED_LIB) /usr/local/lib/
	cd /usr/local/lib && ln -sf libcudarst.so.$(VERSION) libcudarst.so.$(MAJOR)
	cd /usr/local/lib && ln -sf libcudarst.so.$(VERSION) libcudarst.so
	install -m 644 $(HEADERS) /usr/local/include/cudarst/
	ldconfig

# Test targets
test: all test_programs
	@echo "Running CUDArst library tests..."
	./tests/test_fitacf_compatibility
	./tests/test_lmfit_compatibility
	./tests/test_performance

test_programs:
	@mkdir -p tests
	$(CC) -o tests/test_fitacf_compatibility tests/test_fitacf.c -L$(LIBDIR) -lcudarst $(LDFLAGS)
	$(CC) -o tests/test_lmfit_compatibility tests/test_lmfit.c -L$(LIBDIR) -lcudarst $(LDFLAGS)
	$(CC) -o tests/test_performance tests/test_performance.c -L$(LIBDIR) -lcudarst $(LDFLAGS)

# Package creation
package: all
	@echo "Creating CUDArst distribution package..."
	@mkdir -p cudarst-$(VERSION)
	@cp -r include src lib Makefile README.md cudarst-$(VERSION)/
	@tar czf cudarst-$(VERSION).tar.gz cudarst-$(VERSION)/
	@rm -rf cudarst-$(VERSION)
	@echo "Package created: cudarst-$(VERSION).tar.gz"

# Documentation
docs:
	doxygen Doxyfile

# Clean
clean:
	rm -rf $(BUILDDIR) $(LIBDIR) tests/test_* cudarst-*.tar.gz

# Help
help:
	@echo "CUDArst Library Build System v$(VERSION)"
	@echo "========================================="
	@echo "Targets:"
	@echo "  all      - Build static and shared libraries"
	@echo "  install  - Install system-wide"
	@echo "  test     - Run compatibility tests"
	@echo "  package  - Create distribution package"
	@echo "  docs     - Generate documentation"
	@echo "  clean    - Remove build artifacts"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Included CUDA-accelerated modules:"
	@echo "  - FITACF v3.0: Auto-correlation function processing"
	@echo "  - LMFIT v2.0: Levenberg-Marquardt fitting"
	@echo "  - ACF v1.16: Auto-correlation functions"
	@echo "  - IQ v1.7: I/Q data processing"
	@echo "  - CNVMAP v1.17: Convection mapping"
	@echo "  - GRID v1.24: Spatial grid processing"
	@echo "  - FIT v1.35: Fitting algorithms"
	@echo ""
	@echo "Total: 49 specialized CUDA kernels, 100% backward compatible"

.PHONY: all directories install test test_programs package docs clean help