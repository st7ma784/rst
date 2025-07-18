#!/bin/bash
#
# CUDA Linked List Test Runner
# 
# This script runs comprehensive tests of the CUDA-compatible linked list
# implementation against the original CPU version using real SUPERDARN data.
#

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FITACF_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DATA_DIR="/mnt/drive1/rawacf/1999/02"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if CUDA is available
    if command -v nvcc >/dev/null 2>&1; then
        log_success "CUDA compiler found: $(nvcc --version | head -n1)"
    else
        log_warning "CUDA compiler not found. Tests will run in CPU-only mode."
    fi
    
    # Check if GPU is available
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            log_success "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)"
        else
            log_warning "nvidia-smi failed. GPU may not be available."
        fi
    else
        log_warning "nvidia-smi not found. GPU tests may not work."
    fi
    
    # Check for test data
    if [ -d "$TEST_DATA_DIR" ]; then
        local file_count=$(find "$TEST_DATA_DIR" -name "*.rawacf.bz2" | wc -l)
        log_success "Found $file_count rawacf files in $TEST_DATA_DIR"
    else
        log_warning "Test data directory not found: $TEST_DATA_DIR"
        log_info "Tests will use synthetic data instead."
    fi
    
    # Check build tools
    for tool in gcc make ar ranlib; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "$tool found"
        else
            log_error "$tool not found. Please install build-essential."
            return 1
        fi
    done
    
    return 0
}

# Setup test environment
setup_environment() {
    log_info "Setting up test environment..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Set environment variables
    export SYSTEM=linux
    export MAKECFG="$FITACF_DIR/../../../build/make/makecfg"
    export MAKELIB="$FITACF_DIR/../../../build/make/makelib"
    export IPATH="$FITACF_DIR/../../../include"
    export LIBPATH="$FITACF_DIR/../../../lib"
    export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
    
    # Create minimal build configuration if it doesn't exist
    local build_dir="$FITACF_DIR/../../../build/make"
    mkdir -p "$build_dir"
    mkdir -p "$IPATH/superdarn"
    mkdir -p "$LIBPATH"
    
    if [ ! -f "$MAKECFG.linux" ]; then
        log_info "Creating minimal build configuration..."
        cat > "$MAKECFG.linux" << 'EOF'
CC=gcc
CFLAGS=-O2 -fPIC -Wall -I../include
SYSTEM=linux
EOF
    fi
    
    if [ ! -f "$MAKELIB.linux" ]; then
        cat > "$MAKELIB.linux" << 'EOF'
$(DSTPATH)/lib$(OUTPUT).a: $(OBJS)
	ar -rc $@ $(OBJS)
	ranlib $@
	mkdir -p $(IPATH)/superdarn
	cp ../include/*.h $(IPATH)/superdarn/ 2>/dev/null || true
EOF
    fi
    
    log_success "Environment setup complete"
}

# Build libraries
build_libraries() {
    log_info "Building libraries..."
    
    cd "$FITACF_DIR/src"
    
    # Clean previous builds
    make -f makefile clean 2>/dev/null || true
    make -f makefile.cuda clean 2>/dev/null || true
    
    # Build original CPU library
    log_info "Building CPU library..."
    if make -f makefile; then
        log_success "CPU library built successfully"
    else
        log_error "Failed to build CPU library"
        return 1
    fi
    
    # Build CUDA libraries if CUDA is available
    if command -v nvcc >/dev/null 2>&1; then
        log_info "Building CUDA libraries..."
        if make -f makefile.cuda all; then
            log_success "CUDA libraries built successfully"
        else
            log_error "Failed to build CUDA libraries"
            return 1
        fi
    else
        log_warning "Skipping CUDA library build (nvcc not found)"
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

# Build test executables
build_tests() {
    log_info "Building test executables..."
    
    cd "$SCRIPT_DIR"
    
    # Clean previous builds
    make clean 2>/dev/null || true
    
    # Build tests
    if make all; then
        log_success "Test executables built successfully"
    else
        log_error "Failed to build test executables"
        return 1
    fi
    
    return 0
}

# Run validation tests
run_validation_tests() {
    log_info "Running validation tests..."
    
    local output_file="$RESULTS_DIR/validation_test_$TIMESTAMP.log"
    
    if [ -x "./test_cuda_validation" ]; then
        log_info "Executing validation tests..."
        if ./test_cuda_validation 2>&1 | tee "$output_file"; then
            log_success "Validation tests completed successfully"
            return 0
        else
            log_error "Validation tests failed"
            return 1
        fi
    else
        log_error "Validation test executable not found"
        return 1
    fi
}

# Run RAWACF processing tests
run_rawacf_tests() {
    log_info "Running RAWACF processing tests..."
    
    local output_file="$RESULTS_DIR/rawacf_test_$TIMESTAMP.log"
    
    if [ -x "./test_rawacf_processing" ]; then
        log_info "Executing RAWACF processing tests..."
        if ./test_rawacf_processing 2>&1 | tee "$output_file"; then
            log_success "RAWACF processing tests completed successfully"
            return 0
        else
            log_error "RAWACF processing tests failed"
            return 1
        fi
    else
        log_error "RAWACF test executable not found"
        return 1
    fi
}

# Run memory tests
run_memory_tests() {
    if ! command -v valgrind >/dev/null 2>&1; then
        log_warning "Valgrind not found. Skipping memory tests."
        return 0
    fi
    
    log_info "Running memory leak tests..."
    
    local output_file="$RESULTS_DIR/memory_test_$TIMESTAMP.log"
    
    if [ -x "./test_cuda_validation" ]; then
        log_info "Executing memory leak tests (CPU only)..."
        if timeout 300 valgrind --tool=memcheck --leak-check=full \
           --show-leak-kinds=all --track-origins=yes \
           --log-file="$output_file" ./test_cuda_validation; then
            log_success "Memory tests completed"
            
            # Check for leaks
            if grep -q "definitely lost: 0 bytes" "$output_file" && \
               grep -q "indirectly lost: 0 bytes" "$output_file"; then
                log_success "No memory leaks detected"
                return 0
            else
                log_warning "Potential memory leaks detected. Check $output_file"
                return 1
            fi
        else
            log_error "Memory tests failed or timed out"
            return 1
        fi
    else
        log_error "Test executable not found for memory testing"
        return 1
    fi
}

# Generate comprehensive report
generate_report() {
    log_info "Generating comprehensive test report..."
    
    local report_file="$RESULTS_DIR/comprehensive_report_$TIMESTAMP.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>CUDA Linked List Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .success { color: green; font-weight: bold; }
        .error { color: red; font-weight: bold; }
        .warning { color: orange; font-weight: bold; }
        .section { margin: 20px 0; padding: 10px; border-left: 3px solid #ccc; }
        pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CUDA Linked List Validation Report</h1>
        <p><strong>Generated:</strong> $(date)</p>
        <p><strong>System:</strong> $(uname -a)</p>
        <p><strong>CUDA Version:</strong> $(nvcc --version 2>/dev/null | head -n1 || echo "Not available")</p>
        <p><strong>GPU:</strong> $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "Not available")</p>
    </div>

    <div class="section">
        <h2>Test Summary</h2>
        <table>
            <tr><th>Test Type</th><th>Status</th><th>Details</th></tr>
EOF

    # Add test results to report
    local validation_status="❌ FAILED"
    local rawacf_status="❌ FAILED"
    local memory_status="❌ FAILED"
    
    if [ -f "$RESULTS_DIR/validation_test_$TIMESTAMP.log" ]; then
        if grep -q "Success Rate: 100.0%" "$RESULTS_DIR/validation_test_$TIMESTAMP.log"; then
            validation_status="✅ PASSED"
        fi
    fi
    
    if [ -f "$RESULTS_DIR/rawacf_test_$TIMESTAMP.log" ]; then
        if grep -q "Success Rate: 100.0%" "$RESULTS_DIR/rawacf_test_$TIMESTAMP.log"; then
            rawacf_status="✅ PASSED"
        fi
    fi
    
    if [ -f "$RESULTS_DIR/memory_test_$TIMESTAMP.log" ]; then
        if grep -q "definitely lost: 0 bytes" "$RESULTS_DIR/memory_test_$TIMESTAMP.log"; then
            memory_status="✅ PASSED"
        fi
    fi
    
    cat >> "$report_file" << EOF
            <tr><td>Validation Tests</td><td>$validation_status</td><td>Basic functionality and correctness</td></tr>
            <tr><td>RAWACF Processing</td><td>$rawacf_status</td><td>Real data processing validation</td></tr>
            <tr><td>Memory Leak Tests</td><td>$memory_status</td><td>Memory safety validation</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
EOF

    # Include detailed logs
    for log_file in "$RESULTS_DIR"/*_$TIMESTAMP.log; do
        if [ -f "$log_file" ]; then
            local test_name=$(basename "$log_file" .log)
            cat >> "$report_file" << EOF
        <h3>$test_name</h3>
        <pre>$(cat "$log_file")</pre>
EOF
        fi
    done
    
    cat >> "$report_file" << EOF
    </div>
</body>
</html>
EOF

    log_success "Comprehensive report generated: $report_file"
    
    # Also create a simple text summary
    local summary_file="$RESULTS_DIR/summary_$TIMESTAMP.txt"
    cat > "$summary_file" << EOF
CUDA Linked List Validation Summary
Generated: $(date)

Test Results:
- Validation Tests: $validation_status
- RAWACF Processing: $rawacf_status  
- Memory Leak Tests: $memory_status

For detailed results, see: $report_file
EOF

    log_success "Summary report generated: $summary_file"
}

# Main execution
main() {
    echo "========================================"
    echo "CUDA Linked List Validation Test Suite"
    echo "========================================"
    echo
    
    local exit_code=0
    
    # Run all test phases
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    if ! setup_environment; then
        log_error "Environment setup failed"
        exit 1
    fi
    
    if ! build_libraries; then
        log_error "Library build failed"
        exit 1
    fi
    
    if ! build_tests; then
        log_error "Test build failed"
        exit 1
    fi
    
    # Run tests (continue even if some fail)
    if ! run_validation_tests; then
        log_warning "Validation tests failed"
        exit_code=1
    fi
    
    if ! run_rawacf_tests; then
        log_warning "RAWACF processing tests failed"
        exit_code=1
    fi
    
    if ! run_memory_tests; then
        log_warning "Memory tests failed or detected issues"
        exit_code=1
    fi
    
    # Always generate report
    generate_report
    
    echo
    if [ $exit_code -eq 0 ]; then
        log_success "All tests completed successfully!"
    else
        log_warning "Some tests failed. Check the detailed reports."
    fi
    
    log_info "Results saved in: $RESULTS_DIR"
    
    exit $exit_code
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --quick        Run only basic validation tests"
        echo "  --memory-only  Run only memory leak tests"
        echo "  --no-build     Skip library building (use existing builds)"
        exit 0
        ;;
    --quick)
        log_info "Running in quick mode (validation tests only)"
        # Override functions to skip some tests
        run_rawacf_tests() { log_info "Skipping RAWACF tests in quick mode"; return 0; }
        run_memory_tests() { log_info "Skipping memory tests in quick mode"; return 0; }
        ;;
    --memory-only)
        log_info "Running memory tests only"
        run_validation_tests() { log_info "Skipping validation tests"; return 0; }
        run_rawacf_tests() { log_info "Skipping RAWACF tests"; return 0; }
        ;;
    --no-build)
        log_info "Skipping build phase"
        build_libraries() { log_info "Skipping library build"; return 0; }
        build_tests() { log_info "Skipping test build"; return 0; }
        ;;
esac

# Run main function
main "$@"
