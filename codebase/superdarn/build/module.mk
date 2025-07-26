# Module build template for SuperDARN RST
# Include this file in module Makefiles to use the common build system

# Include common build configuration
include $(TOPDIR)/build/common.mk

# Module name (should be set by including Makefile)
MODULE_NAME ?= $(notdir $(CURDIR))
MODULE_VERSION ?= $(shell echo $(MODULE_NAME) | grep -oE '[0-9]+\.[0-9]+' || echo "0.0")

# Default source directories
SRC_DIR ?= src
INCLUDE_DIR ?= include
TEST_DIR ?= tests

# Default build directories
BUILDDIR ?= $(TOPDIR)/build/$(MODULE_NAME)
LIBDIR ?= $(TOPDIR)/lib
BINDIR ?= $(TOPDIR)/bin
INCLUDEDIR ?= $(TOPDIR)/include

# Default source file patterns
C_SRCS ?= $(wildcard $(SRC_DIR)/*.c)
CPP_SRCS ?= $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRCS ?= $(wildcard $(SRC_DIR)/*.cu)

# Default include directories
INCLUDES := -I$(INCLUDE_DIR) -I$(TOPDIR)/include $(INCLUDES)

# Default library name
LIBRARY ?= lib$(MODULE_NAME).a
LIBRARY_CUDA ?= lib$(MODULE_NAME).cuda.a
LIBRARY_SHARED ?= lib$(MODULE_NAME).so

# Object files
C_OBJS := $(addprefix $(BUILDDIR)/, $(C_SRCS:.c=.o))
CPP_OBJS := $(addprefix $(BUILDDIR)/, $(CPP_SRCS:.cpp=.o))
CUDA_OBJS := $(addprefix $(BUILDDIR)/, $(CUDA_SRCS:.cu=.o))

# Dependency files
DEPS := $(C_OBJS:.o=.d) $(CPP_OBJS:.o=.d) $(CUDA_OBJS:.o=.d)

# Default target
all: $(LIBDIR)/$(LIBRARY) $(if $(CUDA_SRCS),$(LIBDIR)/$(LIBRARY_CUDA))

# Create static library
$(LIBDIR)/$(LIBRARY): $(C_OBJS) $(CPP_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^
	$(RANLIB) $@

# Create CUDA static library
$(LIBDIR)/$(LIBRARY_CUDA): $(CUDA_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^
	$(RANLIB) $@

# Create shared library
$(LIBDIR)/$(LIBRARY_SHARED): $(C_OBJS) $(CPP_OBJS)
	@mkdir -p $(@D)
	$(CC) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Install headers
install-headers:
	@mkdir -p $(DESTDIR)$(INCLUDEDIR)/$(MODULE_NAME)
	cp -r $(INCLUDE_DIR)/*.h $(DESTDIR)$(INCLUDEDIR)/$(MODULE_NAME)/

# Install libraries
install-libs: $(LIBDIR)/$(LIBRARY) $(if $(CUDA_SRCS),$(LIBDIR)/$(LIBRARY_CUDA))
	@mkdir -p $(DESTDIR)$(LIBDIR)
	cp $(LIBDIR)/$(LIBRARY) $(DESTDIR)$(LIBDIR)/
	if [ -f "$(LIBDIR)/$(LIBRARY_CUDA)" ]; then \
	    cp $(LIBDIR)/$(LIBRARY_CUDA) $(DESTDIR)$(LIBDIR)/; \
	fi

# Install everything
install: install-headers install-libs

# Clean build artifacts
clean:
	$(RM) -r $(BUILDDIR) $(LIBDIR)/$(LIBRARY) $(LIBDIR)/$(LIBRARY_CUDA) $(LIBDIR)/$(LIBRARY_SHARED)

# Include dependencies
-include $(DEPS)

# Print build info
info:
	@echo "Module: $(MODULE_NAME) (v$(MODULE_VERSION))"
	@echo "Build type: $(BUILD_TYPE)"
	@echo "Sources: $(C_SRCS) $(CPP_SRCS) $(CUDA_SRCS)"
	@echo "Objects: $(C_OBJS) $(CPP_OBJS) $(CUDA_OBJS)"
	@echo "Includes: $(INCLUDES)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "NVCCFLAGS: $(NVCCFLAGS)"

.PHONY: all clean install install-headers install-libs info
