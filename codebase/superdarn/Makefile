# Top-level Makefile for SuperDARN RST
# This file coordinates the build of all modules

# Set the top-level directory
TOPDIR := $(CURDIR)
export TOPDIR

# Default target
all: modules

# List of modules to build (alphabetical order)
MODULES := \
	tk/acf.1.16 \
	tk/acf.1.16_optimized.2.0 \
	tk/acfex.1.3 \
	tk/binplotlib.1.0 \
	tk/binplotlib.1.0_optimized.2.0 \
	tk/cfit.1.19 \
	tk/channel.1.0 \
	tk/cnvmap.1.17 \
	tk/cnvmodel.1.0 \
	tk/elevation.1.0 \
	tk/filter.1.8 \
	tk/fit.1.35 \
	tk/fitacf.2.5 \
	tk/fitacf_v3.0 \
	tk/fitacfex.1.3 \
	tk/fitacfex2.1.0 \
	tk/fitcnx.1.16 \
	tk/freqband.1.0 \
	tk/grid.1.24 \
	tk/grid.1.24_optimized.1 \
	tk/gtable.2.0 \
	tk/gtablewrite.1.9 \
	tk/hmb.1.0 \
	tk/iq.1.7 \
	tk/lmfit.1.0 \
	tk/lmfit_v2.0 \
	tk/oldcnvmap.1.2 \
	tk/oldfit.1.25 \
	tk/oldfitcnx.1.10 \
	tk/oldgrid.1.3 \
	tk/oldgtablewrite.1.4 \
	tk/oldraw.1.16 \
	tk/radar.1.22 \
	tk/raw.1.22 \
	tk/rpos.1.7 \
	tk/scan.1.7 \
	tk/shf.1.10 \
	tk/sim_data.1.0 \
	tk/smr.1.7 \
	tk/snd.1.0 \
	tk/tsg.1.13

# Build all modules
modules: $(addsuffix -build,$(MODULES))

# Build a specific module
%-build:
	@echo "\n=== Building $* ==="
	@if [ -f "src.lib/$*/Makefile" ]; then \
	    $(MAKE) -C src.lib/$* all; \
	elif [ -f "src.lib/$*/makefile.cuda" ]; then \
	    $(MAKE) -f makefile.cuda -C src.lib/$* all; \
	else \
	    echo "No build system found for $*"; \
	    exit 1; \
	fi

# Clean all modules
clean: $(addsuffix -clean,$(MODULES))
	@rm -rf build lib include

# Clean a specific module
%-clean:
	@if [ -f "src.lib/$*/Makefile" ]; then \
	    $(MAKE) -C src.lib/$* clean; \
	elif [ -f "src.lib/$*/makefile.cuda" ]; then \
	    $(MAKE) -f makefile.cuda -C src.lib/$* clean; \
	fi

# Install all modules
install: all
	@mkdir -p $(DESTDIR)$(PREFIX)/{lib,include}
	@cp -r include/* $(DESTDIR)$(PREFIX)/include/
	@cp lib/*.a $(DESTDIR)$(PREFIX)/lib/

# Help target
help:
	@echo "SuperDARN RST Build System"
	@echo "=========================="
	@echo "Available targets:"
	@echo "  all          - Build all modules (default)"
	@echo "  clean        - Remove all build artifacts"
	@echo "  install      - Install libraries and headers"
	@echo "  help         - Show this help message"
	@echo "  <module>     - Build a specific module"
	@echo "  <module>-clean - Clean a specific module"
	@echo ""
	@echo "Environment variables:"
	@echo "  BUILD_TYPE   - Set build type (debug, release, relwithdebinfo, minsizerel)"
	@echo "  PREFIX       - Installation prefix (default: /usr/local)"
	@echo "  CC, CXX      - C/C++ compiler commands"
	@echo "  NVCC         - CUDA compiler command"

.PHONY: all modules clean install help $(addsuffix -build,$(MODULES)) $(addsuffix -clean,$(MODULES))
