#!/bin/bash
# SuperDARN FitACF v3.0 Docker Testing Script
# 
# This script builds and runs comprehensive tests for the FitACF array implementation
# in a controlled Docker environment with full RST dependencies.

set -e  # Exit on any error

echo "=== SuperDARN FitACF v3.0 Docker Testing Suite ==="
echo "$(date): Starting Docker-based testing..."

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose not found, will use docker commands directly"
    USE_COMPOSE=false
else
    USE_COMPOSE=true
fi

# Create results directory
mkdir -p ./test-results
chmod 777 ./test-results

print_status "Building FitACF testing Docker image..."

# Build the Docker image
if ! docker build -f dockerfile.fitacf -t fitacf-test .; then
    print_error "Failed to build Docker image"
    exit 1
fi

print_success "Docker image built successfully"

# Function to run tests with different configurations
run_test_suite() {
    local test_name="$1"
    local threads="$2"
    local extra_env="$3"
    
    print_status "Running $test_name with $threads threads..."
    
    docker run --rm \
        -v "$(pwd)/codebase/superdarn/src.lib/tk/fitacf_v3.0:/workspace/fitacf_v3.0" \
        -v "$(pwd)/test-results:/workspace/results" \
        -e "OMP_NUM_THREADS=$threads" \
        -e "OMP_SCHEDULE=dynamic,1" \
        -e "OMP_PROC_BIND=true" \
        $extra_env \
        fitacf-test \
        /bin/bash -c "
            source /opt/rst/.profile.bash &&
            cd /workspace/fitacf_v3.0/src &&
            echo 'Environment check:' &&
            echo 'OMP_NUM_THREADS: '\$OMP_NUM_THREADS &&
            echo 'Available CPU cores:' \$(nproc) &&
            echo 'OpenMP support:' &&
            echo '#include <omp.h>' | gcc -fopenmp -E - > /dev/null 2>&1 && echo 'OpenMP: Available' || echo 'OpenMP: Not available' &&
            echo '' &&
            echo 'Building tests...' &&
            make -f makefile_standalone clean &&
            if make -f makefile_standalone tests; then
                echo 'Build successful, running tests...' &&
                echo '' &&
                echo '=== Test Results ===' &&
                ./test_baseline 2>&1 | tee /workspace/results/${test_name}_baseline_${threads}threads.txt &&
                ./test_comparison 2>&1 | tee /workspace/results/${test_name}_comparison_${threads}threads.txt &&
                ./test_performance 2>&1 | tee /workspace/results/${test_name}_performance_${threads}threads.txt &&
                echo 'Tests completed successfully'
            else
                echo 'Build failed!' >&2 &&
                exit 1
            fi
        "
}

# Run tests with different thread configurations
print_status "Starting comprehensive test suite..."

# Single-threaded test
run_test_suite "single_thread" "1" ""

# Multi-threaded tests
for threads in 2 4 8; do
    run_test_suite "multi_thread" "$threads" ""
done

# High-performance test with additional optimizations
run_test_suite "high_performance" "8" "-e CFLAGS='-O3 -march=native -mtune=native'"

print_status "Running interactive test environment..."

# Provide an interactive environment for manual testing
docker run -it --rm \
    -v "$(pwd)/codebase/superdarn/src.lib/tk/fitacf_v3.0:/workspace/fitacf_v3.0" \
    -v "$(pwd)/test-results:/workspace/results" \
    -e "OMP_NUM_THREADS=4" \
    fitacf-test \
    /bin/bash -c "
        source /opt/rst/.profile.bash &&
        cd /workspace/fitacf_v3.0 &&
        echo 'Interactive FitACF testing environment ready!' &&
        echo 'Available commands:' &&
        echo '  cd src && make -f makefile_standalone tests  # Build tests' &&
        echo '  ./test_baseline                               # Run baseline test' &&
        echo '  ./test_comparison                             # Run comparison test' &&
        echo '  ./test_performance                            # Run performance test' &&
        echo '  exit                                          # Exit container' &&
        echo '' &&
        /bin/bash
    "

print_success "All tests completed!"
print_status "Test results are available in: ./test-results/"

# Summary of results
echo ""
echo "=== Test Summary ==="
if [ -d "./test-results" ]; then
    echo "Generated test files:"
    ls -la ./test-results/
    
    echo ""
    echo "Performance summary (if available):"
    if ls ./test-results/*performance*.txt 1> /dev/null 2>&1; then
        grep -h "Speedup\|Performance\|threads" ./test-results/*performance*.txt | sort | uniq
    fi
fi

echo ""
print_success "FitACF Docker testing completed successfully!"
echo "Review the results in ./test-results/ for detailed analysis."
