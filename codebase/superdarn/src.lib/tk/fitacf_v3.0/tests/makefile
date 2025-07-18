# Test makefile for CUDA linked list validation
#
include $(MAKECFG).$(SYSTEM)

# Test-specific includes
INCLUDE= -I$(IPATH)/base -I$(IPATH)/general -I$(IPATH)/superdarn/fit -I$(IPATH)/superdarn \
         -I../include

# CUDA configuration
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda

# Test source files
TEST_VALIDATION_SRC = test_cuda_validation.c
TEST_RAWACF_SRC = test_rawacf_processing.c

# Object files
TEST_VALIDATION_OBJ = test_cuda_validation.o
TEST_RAWACF_OBJ = test_rawacf_processing.o

# Executables
TEST_VALIDATION_EXE = test_cuda_validation
TEST_RAWACF_EXE = test_rawacf_processing

# Library dependencies
FITACF_LIB = ../src/libfitacf.a
FITACF_CUDA_LIB = ../src/libfitacf_cuda.a
FITACF_COMPAT_LIB = ../src/libfitacf_compat.a

# Common libraries
COMMON_LIBS = -lm -lpthread -lbz2

# Compiler flags
CFLAGS += -DCUDA_ENABLED -std=c99 -Wall -Wextra
CUDA_CFLAGS = -std=c++11 -arch=sm_35

# Default target
all: $(TEST_VALIDATION_EXE) $(TEST_RAWACF_EXE)

# Test validation executable
$(TEST_VALIDATION_EXE): $(TEST_VALIDATION_OBJ) $(FITACF_LIB) $(FITACF_CUDA_LIB) $(FITACF_COMPAT_LIB)
	$(CC) -o $@ $(TEST_VALIDATION_OBJ) $(FITACF_COMPAT_LIB) $(FITACF_CUDA_LIB) $(FITACF_LIB) \
	      $(CUDA_LIBS) $(COMMON_LIBS)

# RAWACF test executable
$(TEST_RAWACF_EXE): $(TEST_RAWACF_OBJ) $(FITACF_LIB) $(FITACF_CUDA_LIB) $(FITACF_COMPAT_LIB)
	$(CC) -o $@ $(TEST_RAWACF_OBJ) $(FITACF_COMPAT_LIB) $(FITACF_CUDA_LIB) $(FITACF_LIB) \
	      $(CUDA_LIBS) $(COMMON_LIBS)

# Object file compilation
$(TEST_VALIDATION_OBJ): $(TEST_VALIDATION_SRC)
	$(CC) $(CFLAGS) $(INCLUDE) $(CUDA_INCLUDE) -c $< -o $@

$(TEST_RAWACF_OBJ): $(TEST_RAWACF_SRC)
	$(CC) $(CFLAGS) $(INCLUDE) $(CUDA_INCLUDE) -c $< -o $@

# Build required libraries first
$(FITACF_LIB):
	$(MAKE) -C ../src -f makefile

$(FITACF_CUDA_LIB):
	$(MAKE) -C ../src -f makefile.cuda cuda

$(FITACF_COMPAT_LIB):
	$(MAKE) -C ../src -f makefile.cuda compat

# Test targets
test: test-validation test-rawacf

test-validation: $(TEST_VALIDATION_EXE)
	@echo "Running CUDA validation tests..."
	./$(TEST_VALIDATION_EXE)

test-rawacf: $(TEST_RAWACF_EXE)
	@echo "Running RAWACF processing tests..."
	./$(TEST_RAWACF_EXE)

# Performance benchmarking
benchmark: $(TEST_VALIDATION_EXE) $(TEST_RAWACF_EXE)
	@echo "Running performance benchmarks..."
	@echo "=== Validation Benchmark ==="
	./$(TEST_VALIDATION_EXE)
	@echo ""
	@echo "=== RAWACF Processing Benchmark ==="
	./$(TEST_RAWACF_EXE)

# Memory testing with valgrind (CPU only)
memtest: $(TEST_VALIDATION_EXE)
	@echo "Running memory tests (CPU implementation only)..."
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
	         --track-origins=yes ./$(TEST_VALIDATION_EXE)

# Generate test report
report: test
	@echo "Generating test report..."
	@echo "Test Report - $(shell date)" > test_report.txt
	@echo "=============================" >> test_report.txt
	@echo "" >> test_report.txt
	@echo "Validation Tests:" >> test_report.txt
	./$(TEST_VALIDATION_EXE) >> test_report.txt 2>&1
	@echo "" >> test_report.txt
	@echo "RAWACF Processing Tests:" >> test_report.txt
	./$(TEST_RAWACF_EXE) >> test_report.txt 2>&1
	@echo "Report saved to test_report.txt"

# Clean targets
clean:
	rm -f $(TEST_VALIDATION_OBJ) $(TEST_RAWACF_OBJ)
	rm -f $(TEST_VALIDATION_EXE) $(TEST_RAWACF_EXE)
	rm -f test_report.txt

distclean: clean
	$(MAKE) -C ../src -f makefile clean
	$(MAKE) -C ../src -f makefile.cuda clean

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all test executables"
	@echo "  test         - Run all tests"
	@echo "  test-validation - Run validation tests only"
	@echo "  test-rawacf  - Run RAWACF processing tests only"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  memtest      - Run memory leak tests (CPU only)"
	@echo "  report       - Generate comprehensive test report"
	@echo "  clean        - Clean test objects and executables"
	@echo "  distclean    - Clean everything including libraries"
	@echo "  help         - Show this help message"

.PHONY: all test test-validation test-rawacf benchmark memtest report clean distclean help
