#!/bin/bash
# SuperDARN FitACF v3.0 Docker Performance Testing Script
# 
# This script runs comprehensive performance tests for the FitACF v3.0
# array-based implementation vs the original linked list implementation
# inside a Docker container for consistent testing environment.
#
# Author: SuperDARN Performance Optimization Team
# Date: May 30, 2025

set -e

echo "======================================================"
echo "SuperDARN FitACF v3.0 Docker Performance Testing"
echo "======================================================"
echo "Date: $(date)"
echo "Container: $(hostname)"
echo "Cores available: $(nproc)"
echo "Memory available: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo ""

# Change to the FitACF directory
cd /workspace/codebase/superdarn/src.lib/tk/fitacf_v3.0

# Set up environment variables
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_SCHEDULE=dynamic

# Create results directory
mkdir -p test_results
RESULTS_DIR="test_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run tests with different configurations
run_test_configuration() {
    local test_name="$1"
    local threads="$2"
    local description="$3"
    
    echo "----------------------------------------"
    echo "Running: $test_name ($description)"
    echo "Threads: $threads"
    echo "----------------------------------------"
    
    export OMP_NUM_THREADS=$threads
    
    # Run the test and capture output
    if make -f Makefile.performance test_perf 2>&1 | tee "$RESULTS_DIR/${test_name}_${threads}threads.log"; then
        echo "✓ $test_name with $threads threads: PASSED"
    else
        echo "✗ $test_name with $threads threads: FAILED"
        return 1
    fi
    
    echo ""
}

# Function to run benchmarks
run_benchmarks() {
    echo "=========================================="
    echo "Running Performance Benchmarks"
    echo "=========================================="
    
    # Build benchmark suite
    if ! make -f Makefile.performance benchmark_performance; then
        echo "Failed to build benchmark suite"
        return 1
    fi
    
    # Run benchmarks with different thread counts
    for threads in 1 2 4 8; do
        if [ $threads -le $(nproc) ]; then
            echo "Running benchmark with $threads threads..."
            export OMP_NUM_THREADS=$threads
            ./benchmark_performance > "$RESULTS_DIR/benchmark_${threads}threads.txt" 2>&1
        fi
    done
    
    echo "Benchmarks completed"
}

# Function to run memory analysis
run_memory_analysis() {
    echo "=========================================="
    echo "Running Memory Analysis"
    echo "=========================================="
    
    # Check if Valgrind is available
    if command -v valgrind &> /dev/null; then
        echo "Running Valgrind memory analysis..."
        make -f Makefile.performance valgrind > "$RESULTS_DIR/valgrind_analysis.log" 2>&1
        echo "✓ Valgrind analysis completed"
    else
        echo "Valgrind not available, skipping memory analysis"
    fi
}

# Function to run performance profiling
run_profiling() {
    echo "=========================================="
    echo "Running Performance Profiling"
    echo "=========================================="
    
    # Check if perf is available
    if command -v perf &> /dev/null; then
        echo "Running perf analysis..."
        make -f Makefile.performance perf_analysis > "$RESULTS_DIR/perf_analysis.log" 2>&1
        echo "✓ Performance profiling completed"
    else
        echo "perf not available, skipping performance profiling"
    fi
}

# Function to analyze results
analyze_results() {
    echo "=========================================="
    echo "Analyzing Results"
    echo "=========================================="
    
    # Create summary report
    local summary_file="$RESULTS_DIR/test_summary.txt"
    
    echo "SuperDARN FitACF v3.0 Performance Test Summary" > "$summary_file"
    echo "===============================================" >> "$summary_file"
    echo "Date: $(date)" >> "$summary_file"
    echo "Environment: Docker Container" >> "$summary_file"
    echo "Cores: $(nproc)" >> "$summary_file"
    echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')" >> "$summary_file"
    echo "" >> "$summary_file"
    
    # Extract key performance metrics
    echo "Performance Metrics:" >> "$summary_file"
    echo "===================" >> "$summary_file"
    
    for log_file in "$RESULTS_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            echo "File: $(basename "$log_file")" >> "$summary_file"
            
            # Extract speedup information
            if grep -q "Speedup:" "$log_file"; then
                grep "Speedup:" "$log_file" >> "$summary_file"
            fi
            
            # Extract test results
            if grep -q "PASS\|FAIL" "$log_file"; then
                grep -c "PASS" "$log_file" | sed 's/^/  Tests passed: /' >> "$summary_file"
                grep -c "FAIL" "$log_file" | sed 's/^/  Tests failed: /' >> "$summary_file"
            fi
            
            echo "" >> "$summary_file"
        fi
    done
    
    # Display summary
    echo "Test Summary:"
    cat "$summary_file"
}

# Main test execution
main() {
    echo "Starting comprehensive FitACF v3.0 performance testing..."
    echo ""
    
    # Build all targets
    echo "Building all implementations..."
    if ! make -f Makefile.performance all; then
        echo "Build failed!"
        exit 1
    fi
    echo "✓ Build completed successfully"
    echo ""
    
    # Run unit tests first
    echo "Running unit tests..."
    if make -f Makefile.performance test_unit > "$RESULTS_DIR/unit_tests.log" 2>&1; then
        echo "✓ Unit tests passed"
    else
        echo "✗ Unit tests failed"
        cat "$RESULTS_DIR/unit_tests.log"
        exit 1
    fi
    echo ""
    
    # Run performance tests with different thread counts
    local max_threads=$(nproc)
    
    # Test single-threaded performance
    run_test_configuration "single_thread" 1 "Baseline single-threaded performance"
    
    # Test multi-threaded performance
    if [ $max_threads -ge 2 ]; then
        run_test_configuration "dual_thread" 2 "Dual-threaded performance"
    fi
    
    if [ $max_threads -ge 4 ]; then
        run_test_configuration "quad_thread" 4 "Quad-threaded performance"
    fi
    
    if [ $max_threads -ge 8 ]; then
        run_test_configuration "octa_thread" 8 "Octa-threaded performance"
    fi
    
    # Run scaling test
    echo "Running scaling analysis..."
    make -f Makefile.performance test_scaling > "$RESULTS_DIR/scaling_analysis.log" 2>&1
    echo "✓ Scaling analysis completed"
    echo ""
    
    # Run comprehensive benchmarks
    run_benchmarks
    
    # Run memory analysis if tools are available
    run_memory_analysis
    
    # Run performance profiling if tools are available
    run_profiling
    
    # Analyze and summarize results
    analyze_results
    
    echo ""
    echo "======================================================"
    echo "All tests completed successfully!"
    echo "Results saved to: $RESULTS_DIR"
    echo "======================================================"
    
    # Copy results to mounted volume if available
    if [ -d "/results" ]; then
        cp -r "$RESULTS_DIR" /results/
        echo "Results also copied to /results/ for host access"
    fi
}

# Error handling
trap 'echo "Test failed at line $LINENO"' ERR

# Run main function
main "$@"
