# Test Makefile for fitacf_v3.0 module
# Builds and runs test programs with CPU, OpenMP, and CUDA support

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3
NVCCFLAGS = -O3 -arch=sm_50
LDFLAGS = -lm

# OpenMP flags
OMP_FLAGS = -fopenmp
OMP_LIBS = -fopenmp

# Test program name
TARGET = phase_unwrap_test

# Source files
CPU_SOURCES = standalone_test.c
CUDA_SOURCES = phase_unwrap.cu

# Object files
CPU_OBJECTS = $(CPU_SOURCES:.c=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Default target
all: cpu openmp cuda

# CPU target (single-threaded)
cpu: CFLAGS += -DCPU_ONLY
cpu: $(CPU_OBJECTS)
	$(CC) -o $(TARGET)_cpu $^ $(LDFLAGS)

# OpenMP target
openmp: CFLAGS += $(OMP_FLAGS)
openmp: $(CPU_OBJECTS)
	$(CC) -o $(TARGET)_openmp $^ $(LDFLAGS) $(OMP_LIBS)

# CUDA target
cuda: NVCCFLAGS += -D__CUDACC__ -Xcompiler "$(OMP_FLAGS)"
cuda: $(CPU_OBJECTS) $(CUDA_OBJECTS)
	nvcc -o $(TARGET)_cuda $^ $(LDFLAGS) -Xcompiler "$(OMP_FLAGS)"

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile CUDA source files
%.o: %.cu
	nvcc $(NVCCFLAGS) -c -o $@ $<

# Run tests
test: all
	@echo "\n=== Running CPU (Single-threaded) Test ==="
	@./$(TARGET)_cpu 1000 256 10 2 || echo "CPU test failed"
	@echo "\n=== Running OpenMP Test ==="
	@./$(TARGET)_openmp 1000 256 10 2 || echo "OpenMP test failed"
	@if [ -f "$(TARGET)_cuda" ]; then \
		echo "\n=== Running CUDA Test ==="; \
		./$(TARGET)_cuda 1000 256 10 2 || echo "CUDA test failed"; \
	else \
		echo "\n=== CUDA Test SKIPPED (CUDA not available) ==="; \
	fi

# Performance test
perf: all
	@echo "\n=== Performance Test (1000 ranges, 256 lags, 100 iterations) ==="
	@echo "CPU (Single-threaded):"
	@./$(TARGET)_cpu 1000 256 100 10 || echo "CPU test failed"
	@echo "\nOpenMP (Multi-threaded):"
	@./$(TARGET)_openmp 1000 256 100 10 || echo "OpenMP test failed"
	@if [ -f "$(TARGET)_cuda" ]; then \
		echo "\nCUDA (GPU):"; \
		./$(TARGET)_cuda 1000 256 100 10 || echo "CUDA test failed"; \
	else \
		echo "\nCUDA (GPU): Not available"; \
	fi

# Clean up
clean:
	rm -f $(TARGET)_cpu $(TARGET)_openmp $(TARGET)_cuda *.o

.PHONY: all cpu openmp cuda test perf clean
