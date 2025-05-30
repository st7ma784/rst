# Makefile for Optimized Fit Speck Removal Test Suite
# ===================================================

include $(MAKECFG).$(SYSTEM)

# Use RST paths if available, otherwise use relative paths
RST_INCLUDE = $(if $(IPATH),-I$(IPATH)/general -I$(IPATH)/base -I$(IPATH)/superdarn,-I../../../../../../../include)
RST_LIBS = $(if $(LIBPATH),-L$(LIBPATH),-L../../../../../../../lib)

LIBRARY_PATH = $(RST_LIBS) -lfit.1 -ldmap.1 -lopt.1 -lrtime.1 -lrtypes.1

# Test executable
TEST_TARGET = test_fit_speck_removal_optimized
TEST_SOURCE = test_fit_speck_removal_optimized.c

# Optimization flags for testing
CFLAGS += -O2 -Wall -Wextra
CFLAGS += -fopenmp

# Debug mode
ifdef DEBUG
CFLAGS += -g -O0 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address
else
CFLAGS += -DNDEBUG
endif

# Memory debugging
ifdef MEMCHECK
CFLAGS += -g -O1
endif

# Build targets
all: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_SOURCE)
	$(CC) $(CFLAGS) $(RST_INCLUDE) -o $@ $< $(LIBRARY_PATH) -lm

# Test execution targets
test: $(TEST_TARGET)
	@echo "Running comprehensive test suite..."
	./$(TEST_TARGET)

quick-test: $(TEST_TARGET)
	@echo "Running quick test suite..."
	./$(TEST_TARGET) --quick

benchmark: $(TEST_TARGET)
	@echo "Running performance benchmarks..."
	./$(TEST_TARGET) --benchmark

memory-test: $(TEST_TARGET)
	@echo "Running memory tests..."
	./$(TEST_TARGET) --memory

scaling-test: $(TEST_TARGET)
	@echo "Running thread scaling tests..."
	./$(TEST_TARGET) --scaling

# Memory leak detection with valgrind (if available)
memcheck: $(TEST_TARGET)
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "Running valgrind memory check..."; \
		valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
		         --track-origins=yes ./$(TEST_TARGET) --memory; \
	else \
		echo "Valgrind not available, running basic memory test"; \
		./$(TEST_TARGET) --memory; \
	fi

# Performance profiling with gprof
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: $(TEST_TARGET)
	@echo "Running with profiling..."
	./$(TEST_TARGET) --benchmark
	@if [ -f gmon.out ]; then \
		gprof $(TEST_TARGET) gmon.out > profile_report.txt; \
		echo "Profile report generated: profile_report.txt"; \
	fi

# Thread scaling analysis
thread-scaling: $(TEST_TARGET)
	@echo "Testing thread scaling performance..."
	@echo "Threads | Time (ms) | Speedup | Efficiency"
	@echo "--------|-----------|---------|----------"
	@for threads in 1 2 4 8 12 16; do \
		if [ $$threads -le $$(nproc) ]; then \
			export OMP_NUM_THREADS=$$threads; \
			echo -n "   $$threads    | "; \
			./$(TEST_TARGET) --scaling 2>/dev/null | grep "Avg Time" | awk '{print $$3}' || echo "Failed"; \
		fi; \
	done

# Generate test data for manual testing
test-data:
	@echo "Generating test data files..."
	@if [ -f ../fit_speck_removal_optimized ]; then \
		echo "Test data generation would require SuperDARN fitacf files"; \
		echo "Please provide sample fitacf files for testing"; \
	else \
		echo "Please build the optimized tool first: make -C .. all"; \
	fi

# Compare with original implementation
compare-original: $(TEST_TARGET)
	@echo "Comparing with original implementation..."
	@if [ -f ../../fit_speck_removal.1.0/fit_speck_removal ]; then \
		./$(TEST_TARGET) --compare; \
	else \
		echo "Original implementation not found. Please build it first."; \
		echo "cd ../../fit_speck_removal.1.0 && make all"; \
	fi

# Clean up
clean:
	rm -f $(TEST_TARGET)
	rm -f *.o
	rm -f gmon.out
	rm -f profile_report.txt
	rm -f test_*.fit
	rm -f temp_*

# Install test in bin directory (if RST environment is set up)
install: $(TEST_TARGET)
	@if [ -n "$(BINPATH)" ]; then \
		$(INSTALL_BIN) $(TEST_TARGET) $(BINPATH); \
		echo "Test installed to $(BINPATH)"; \
	else \
		echo "BINPATH not set, cannot install"; \
	fi

# Help target
help:
	@echo "Optimized Fit Speck Removal Test Suite Makefile"
	@echo "================================================"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build test executable"
	@echo "  test             Run comprehensive test suite"
	@echo "  quick-test       Run quick test suite"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  memory-test      Run memory usage tests"
	@echo "  scaling-test     Run thread scaling tests"
	@echo "  memcheck         Run valgrind memory leak detection"
	@echo "  profile          Run with performance profiling"
	@echo "  thread-scaling   Detailed thread scaling analysis"
	@echo "  test-data        Generate test data files"
	@echo "  compare-original Compare with original implementation"
	@echo "  clean            Clean build artifacts"
	@echo "  install          Install test executable"
	@echo "  help             Show this help message"
	@echo ""
	@echo "Build modes:"
	@echo "  make DEBUG=1     Build with debug information"
	@echo "  make MEMCHECK=1  Build for memory checking"
	@echo ""
	@echo "Examples:"
	@echo "  make test                    # Run all tests"
	@echo "  make benchmark               # Performance benchmarks"
	@echo "  make memcheck                # Memory leak detection"
	@echo "  make OMP_NUM_THREADS=8 test  # Test with 8 threads"

.PHONY: all test quick-test benchmark memory-test scaling-test memcheck profile thread-scaling test-data compare-original clean install help
