# Makefile for SuperDARN Grid Library (Optimized Version 1.24)
# Uses the new unified build system

# Include common build configuration
TOPDIR ?= ../../../..
include $(TOPDIR)/build/common.mk
include $(TOPDIR)/build/module.mk

# Module name and version
MODULE_NAME = grid.1.24_optimized.1
MODULE_VERSION = 1.24.1

# Source files
C_SRCS = $(wildcard src/*.c)

# Include paths
INCLUDES += -I$(TOPDIR)/include -I./include

# Library name
LIBRARY = libgrid_optimized.a

# Build flags
CFLAGS += -DHAVE_OPENMP=1
CXXFLAGS += -DHAVE_OPENMP=1

# Enable vectorization reports in debug builds
ifneq ($(BUILD_TYPE),release)
    CFLAGS += -fopt-info-vec-optimized -fopt-info-vec-missed
endif

# Debug symbols in debug mode
ifdef DEBUG
    CFLAGS += -g -DDEBUG
else
    CFLAGS += -DNDEBUG
endif

# Include the standard RST library build rules
include $(MAKELIB).$(SYSTEM)

# Test programs
test: test_gridseek_optimized

# Build test programs with the same optimization flags
test_gridseek_optimized: test/test_gridseek_optimized.c $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm

.PHONY: clean
clean:
	$(RM) *.o *.a test_gridseek_optimized