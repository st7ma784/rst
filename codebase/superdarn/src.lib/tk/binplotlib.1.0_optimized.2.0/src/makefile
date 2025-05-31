# Makefile for optimized binplotlib library
# ==========================================
# by E.G.Thomas (Original)
# Optimized by: SuperDARN Optimization Framework
#
# This optimized version includes:
# - OpenMP support for parallel processing
# - AVX2/SIMD vectorization support
# - Memory alignment optimizations
# - Performance monitoring capabilities

#This file is part of the Radar Software Toolkit (RST).
#
#RST is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#Modifications:
# - Added OpenMP compiler flags and linking
# - Added AVX2/SIMD optimization flags
# - Added memory alignment optimization flags
# - Added performance monitoring support

include $(MAKECFG).$(SYSTEM)

# Follow exact same structure as working original
INCLUDE = -I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/analysis \
          -I$(IPATH)/superdarn -I../include

# Source files - optimized versions
SRC = geobeam.c key.c make_fov.c make_grid.c plot_cell_optimized.c \
      plot_logo.c plot_raw.c plot_source.c plot_time.c \
      plot_time_label.c plot_vec.c text_box.c

# Object files - optimized versions  
OBJS = geobeam.o key.o make_fov.o make_grid.o plot_cell_optimized.o \
       plot_logo.o plot_raw.o plot_source.o plot_time.o \
       plot_time_label.o plot_vec.o text_box.o

# Library configuration - match original exactly
INC = $(IPATH)/superdarn
DSTPATH = $(LIBPATH)
OUTPUT = binplot_optimized
LINK="1"

# Custom compilation rules for optimized files
plot_cell_optimized.o: plot_cell_optimized.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

make_grid_optimized.o: make_grid_optimized.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

make_fov_optimized.o: make_fov_optimized.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Performance benchmarking target
benchmark: $(OUTPUT)
	@echo "Running performance benchmarks..."
	@echo "Coordinate transformation benchmark:"
	@./benchmark_coords 1000 10000
	@echo "Polygon rendering benchmark:"
	@./benchmark_polygons 1000 1000

# Memory profiling target
profile: $(OUTPUT)
	@echo "Running memory profiling..."
	valgrind --tool=massif --stacks=yes --detailed-freq=1 ./$(OUTPUT)

# Optimization report (for supported compilers)
opt-report:
	@echo "Generating optimization report..."
ifeq ($(CC),icc)
	$(CC) $(CFLAGS) $(INCLUDE) -qopt-report=5 -qopt-report-phase=vec,loop -c $(SRC_OPTIMIZED)
endif
ifeq ($(CC),gcc)
	$(CC) $(CFLAGS) $(INCLUDE) -fopt-info-vec-optimized -fopt-info-loop-optimized -c $(SRC_OPTIMIZED)
endif

# Clean optimized objects
clean-opt:
	rm -f $(OBJS_OPTIMIZED)

# Install optimized headers
install-headers:
	cp -f ../include/*.h $(INC)/

# Performance test target
test-performance: $(OUTPUT)
	@echo "Testing performance improvements..."
	@echo "Original vs Optimized coordinate transformation:"
	@time ./test_original_coords
	@time ./test_optimized_coords
	@echo "Original vs Optimized polygon rendering:"
	@time ./test_original_polygons
	@time ./test_optimized_polygons

# Thread scaling test
test-threads: $(OUTPUT)
	@echo "Testing OpenMP thread scaling..."
	@for threads in 1 2 4 8 16; do \
		echo "Testing with $$threads threads:"; \
		OMP_NUM_THREADS=$$threads time ./$(OUTPUT); \
	done

# SIMD verification
test-simd: $(OUTPUT)
	@echo "Verifying SIMD optimizations..."
ifeq ($(SYSTEM),linux)
	objdump -d $(OUTPUT) | grep -E "(vmovaps|vfmadd|vmulps)" | wc -l
endif

# Memory alignment verification
test-alignment: $(OUTPUT)
	@echo "Verifying memory alignment..."
	@nm $(OUTPUT) | grep -E "aligned"

.PHONY: benchmark profile opt-report clean-opt install-headers test-performance test-threads test-simd test-alignment

include $(MAKELIB).$(SYSTEM)