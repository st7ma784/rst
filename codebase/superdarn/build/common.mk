# Common build configuration for SuperDARN RST
# This file contains common build settings and rules

# Default compiler settings
CC ?= gcc
CXX ?= g++
NVCC ?= nvcc
AR ?= ar
RANLIB ?= ranlib
MKDIR ?= mkdir -p
RM ?= rm -f

# Build type (debug, release, relwithdebinfo, minsizerel)
BUILD_TYPE ?= release

# Installation directories
PREFIX ?= /usr/local
INCLUDEDIR ?= $(PREFIX)/include
LIBDIR ?= $(PREFIX)/lib
BINDIR ?= $(PREFIX)/bin

# Common compiler flags
COMMON_CFLAGS := -Wall -Wextra -Wpedantic -Werror=implicit-function-declaration
COMMON_CXXFLAGS := -std=c++17
COMMON_NVCCFLAGS := -std=c++17 --expt-relaxed-constexpr

# Build type specific flags
ifeq ($(BUILD_TYPE),debug)
    OPT_CFLAGS := -g -O0 -DDEBUG
else ifeq ($(BUILD_TYPE),release)
    OPT_CFLAGS := -O3 -DNDEBUG
else ifeq ($(BUILD_TYPE),relwithdebinfo)
    OPT_CFLAGS := -O2 -g -DNDEBUG
else ifeq ($(BUILD_TYPE),minsizerel)
    OPT_CFLAGS := -Os -DNDEBUG
else
    $(error Unknown build type: $(BUILD_TYPE))
endif

# CPU architecture flags
ifeq ($(shell uname -m),x86_64)
    # Enable architecture-specific optimizations
    ARCH_CFLAGS := -march=native -mtune=native
    
    # Check for AVX-512 support
    ifeq ($(shell grep -c avx512f /proc/cpuinfo),1)
        ARCH_CFLAGS += -mavx512f -mavx512cd
    # Fall back to AVX2
    else ifeq ($(shell grep -c avx2 /proc/cpuinfo),1)
        ARCH_CFLAGS += -mavx2 -mfma
    # Fall back to SSE4.2
    else ifeq ($(shell grep -c sse4_2 /proc/cpuinfo),1)
        ARCH_CFLAGS += -msse4.2
    endif
endif

# OpenMP support
ifdef OPENMP
    OMP_CFLAGS := -fopenmp
    OMP_LDFLAGS := -fopenmp
endif

# Final compiler flags
CFLAGS := $(COMMON_CFLAGS) $(OPT_CFLAGS) $(ARCH_CFLAGS) $(OMP_CFLAGS) $(CFLAGS)
CXXFLAGS := $(COMMON_CXXFLAGS) $(CFLAGS) $(CXXFLAGS)
NVCCFLAGS := $(COMMON_NVCCFLAGS) $(NVCCFLAGS)
LDFLAGS := $(OMP_LDFLAGS) $(LDFLAGS)

# Common linker flags
LDLIBS := -lm -lpthread $(LDLIBS)

# Build rules
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Create directories if they don't exist
$(shell mkdir -p $(BUILDDIR) $(BINDIR) $(LIBDIR) $(INCLUDEDIR))

# Include dependency files if they exist
-include $(wildcard $(BUILDDIR)/*.d)

# Generate dependencies for C files
$(BUILDDIR)/%.d: %.c
	@$(MKDIR) $(@D)
	@$(CC) $(CFLAGS) -MM -MP -MT '$(@:.d=.o)' $< > $@

# Generate dependencies for C++ files
$(BUILDDIR)/%.d: %.cpp
	@$(MKDIR) $(@D)
	@$(CXX) $(CXXFLAGS) -MM -MP -MT '$(@:.d=.o)' $< > $@

# Generate dependencies for CUDA files
$(BUILDDIR)/%.d: %.cu
	@$(MKDIR) $(@D)
	@$(NVCC) $(NVCCFLAGS) -M -MT '$(@:.d=.o)' $< > $@

# Common clean target
.PHONY: clean
clean:
	$(RM) -r $(BUILDDIR) $(TARGETS)

# Common help target
.PHONY: help
help:
	@echo "SuperDARN RST Build System"
	@echo "--------------------------"
	@echo "Available targets:"
	@echo "  all          - Build all targets (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Build options:"
	@echo "  BUILD_TYPE   - Set build type (debug, release, relwithdebinfo, minsizerel)"
	@echo "  OPENMP=1     - Enable OpenMP support"
	@echo "  PREFIX       - Installation prefix (default: /usr/local)"
	@echo "  CC, CXX      - C/C++ compiler commands"
	@echo "  NVCC         - CUDA compiler command"
	@echo "  CFLAGS       - Additional C compiler flags"
	@echo "  CXXFLAGS     - Additional C++ compiler flags"
	@echo "  NVCCFLAGS    - Additional CUDA compiler flags"
	@echo "  LDFLAGS      - Additional linker flags"
	@echo "  LDLIBS       - Additional libraries to link against"
