#!/bin/bash
# docker_performance_tester.sh
# ============================
# Automated performance testing framework for SuperDARN RST Docker environments
# 
# This script runs comprehensive performance comparisons between standard and
# optimized RST builds within Docker containers, generating detailed reports.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/test-results/docker-performance"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_SESSION="docker_perf_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️ WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}

log_test() {
    echo -e "${PURPLE}🧪 TEST:${NC} $1"
}

log_performance() {
    echo -e "${CYAN}⚡ PERFORMANCE:${NC} $1"
}

# Setup
setup_test_environment() {
    log_info "Setting up Docker performance testing environment..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Create session directory
    SESSION_DIR="$RESULTS_DIR/$TEST_SESSION"
    mkdir -p "$SESSION_DIR"
    
    log_success "Test session directory: $SESSION_DIR"
    export SESSION_DIR
}

# Docker utilities
ensure_docker_available() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not available. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not available. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

build_docker_images() {
    log_info "Building Docker images for performance testing..."
    
    # Build all optimization targets
    log_info "Building standard RST image..."
    docker-compose -f docker-compose.optimized.yml build superdarn-standard
    
    log_info "Building optimized RST image..."
    docker-compose -f docker-compose.optimized.yml build superdarn-optimized
    
    log_info "Building development image..."
    docker-compose -f docker-compose.optimized.yml build superdarn-dev
    
    log_success "All Docker images built successfully"
}

# Performance test functions
run_container_performance_test() {
    local container_name="$1"
    local test_type="$2"
    local output_prefix="$3"
    
    log_test "Running performance test: $test_type in $container_name"
    
    # Run the container with performance testing
    docker run --rm \
        --name "${container_name}-perf-test" \
        -v "$PWD/scripts:/workspace/scripts:ro" \
        -v "$SESSION_DIR:/workspace/results" \
        -e "TEST_TYPE=$test_type" \
        -e "OUTPUT_PREFIX=$output_prefix" \
        "$container_name" \
        /bin/bash -c "
            cd /workspace
            echo 'Starting performance test: $test_type'
            
            # Record system information
            echo '=== System Information ===' > results/${output_prefix}_system_info.txt
            echo 'Container: $container_name' >> results/${output_prefix}_system_info.txt
            echo 'Test Type: $test_type' >> results/${output_prefix}_system_info.txt
            echo 'Timestamp: $(date)' >> results/${output_prefix}_system_info.txt
            echo 'CPU Info:' >> results/${output_prefix}_system_info.txt
            cat /proc/cpuinfo | grep 'model name' | head -1 >> results/${output_prefix}_system_info.txt
            echo 'CPU Cores: $(nproc)' >> results/${output_prefix}_system_info.txt
            echo 'Memory:' >> results/${output_prefix}_system_info.txt
            cat /proc/meminfo | head -3 >> results/${output_prefix}_system_info.txt
            echo 'AVX2 Support: $(grep -c avx2 /proc/cpuinfo || echo 0)' >> results/${output_prefix}_system_info.txt
            echo '' >> results/${output_prefix}_system_info.txt
            
            # Run RST validation if available
            if [ -f '/app/rst/validate_optimization_system.sh' ]; then
                echo '=== RST Optimization Validation ===' >> results/${output_prefix}_system_info.txt
                cd /app/rst && ./validate_optimization_system.sh >> /workspace/results/${output_prefix}_system_info.txt 2>&1
                cd /workspace
            fi
            
            # Run performance benchmarks
            if [ -f 'scripts/superdarn_test_suite.sh' ]; then
                echo 'Running SuperDARN test suite...'
                timeout 300s ./scripts/superdarn_test_suite.sh --output-prefix ${output_prefix}_ --benchmark 2>&1 | tee results/${output_prefix}_test_output.log
            else
                echo 'SuperDARN test suite not found, running basic tests...'
                
                # Basic FitACF test if available
                if command -v fitacf &> /dev/null; then
                    echo 'Testing FitACF performance...'
                    time fitacf --help > results/${output_prefix}_fitacf_test.txt 2>&1
                fi
                
                # Basic module loading test
                echo 'Testing module availability...'
                find /app/rst/build -name '*.so' 2>/dev/null > results/${output_prefix}_modules.txt
                
                # Record build information
                echo 'Build information:' > results/${output_prefix}_build_info.txt
                if [ -f '/app/rst/build/script/make.code.optimized' ]; then
                    echo 'Optimized build system available' >> results/${output_prefix}_build_info.txt
                    /app/rst/build/script/make.code.optimized --list-optimizations >> results/${output_prefix}_build_info.txt 2>&1 || true
                else
                    echo 'Standard build system' >> results/${output_prefix}_build_info.txt
                fi
            fi
            
            echo 'Performance test completed for $test_type'
        "
    
    if [ $? -eq 0 ]; then
        log_success "Performance test completed: $test_type"
    else
        log_warning "Performance test completed with warnings: $test_type"
    fi
}

run_memory_usage_test() {
    local container_name="$1"
    local output_prefix="$2"
    
    log_test "Running memory usage test for $container_name"
    
    # Start container in background and monitor memory
    CONTAINER_ID=$(docker run -d \
        --name "${container_name}-memory-test" \
        -v "$SESSION_DIR:/workspace/results" \
        "$container_name" \
        tail -f /dev/null)
    
    # Monitor memory usage for 30 seconds
    for i in {1..30}; do
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" "$CONTAINER_ID" >> "$SESSION_DIR/${output_prefix}_memory_usage.txt"
        sleep 1
    done
    
    # Clean up
    docker stop "$CONTAINER_ID" > /dev/null
    docker rm "$CONTAINER_ID" > /dev/null
    
    log_success "Memory usage test completed"
}

run_startup_time_test() {
    local container_name="$1"
    local output_prefix="$2"
    
    log_test "Running startup time test for $container_name"
    
    # Test container startup time
    for i in {1..5}; do
        start_time=$(date +%s.%N)
        docker run --rm "$container_name" echo "Container started" > /dev/null
        end_time=$(date +%s.%N)
        startup_time=$(echo "$end_time - $start_time" | bc -l)
        echo "Run $i: ${startup_time}s" >> "$SESSION_DIR/${output_prefix}_startup_times.txt"
    done
    
    log_success "Startup time test completed"
}

# Analysis functions
generate_performance_comparison() {
    log_info "Generating performance comparison report..."
    
    # Create comprehensive comparison report
    cat > "$SESSION_DIR/performance_comparison_report.md" << EOF
# SuperDARN RST Docker Performance Comparison Report

**Test Session:** $TEST_SESSION  
**Date:** $(date)  
**Location:** $SESSION_DIR

## Test Overview

This report compares the performance characteristics of different SuperDARN RST Docker builds:

- **Standard RST**: Baseline build with standard optimizations
- **Optimized RST**: Hardware-optimized build with dynamic optimization detection
- **Development Environment**: Both builds available for comparison

## Test Methodology

1. **Build Performance**: Container build times and resource usage
2. **Runtime Performance**: Application execution speed and efficiency  
3. **Memory Usage**: RAM consumption during operation
4. **Startup Performance**: Container initialization time

## Results Summary

EOF

    # Add system information
    echo "### System Information" >> "$SESSION_DIR/performance_comparison_report.md"
    echo "" >> "$SESSION_DIR/performance_comparison_report.md"
    echo "- **Host OS**: $(uname -s)" >> "$SESSION_DIR/performance_comparison_report.md"
    echo "- **Architecture**: $(uname -m)" >> "$SESSION_DIR/performance_comparison_report.md"
    echo "- **Docker Version**: $(docker --version)" >> "$SESSION_DIR/performance_comparison_report.md"
    echo "" >> "$SESSION_DIR/performance_comparison_report.md"

    # Add detailed results for each test
    for test_file in "$SESSION_DIR"/*_system_info.txt; do
        if [ -f "$test_file" ]; then
            test_name=$(basename "$test_file" _system_info.txt)
            echo "### $test_name Test Results" >> "$SESSION_DIR/performance_comparison_report.md"
            echo "" >> "$SESSION_DIR/performance_comparison_report.md"
            echo '```' >> "$SESSION_DIR/performance_comparison_report.md"
            cat "$test_file" >> "$SESSION_DIR/performance_comparison_report.md"
            echo '```' >> "$SESSION_DIR/performance_comparison_report.md"
            echo "" >> "$SESSION_DIR/performance_comparison_report.md"
        fi
    done

    # Add conclusions
    cat >> "$SESSION_DIR/performance_comparison_report.md" << EOF

## Conclusions

1. **Optimization Effectiveness**: Compare the performance gains achieved by the optimized build
2. **Resource Efficiency**: Analyze memory and CPU usage patterns
3. **Stability**: Evaluate the reliability of optimized vs standard builds
4. **Recommendations**: Suggest optimal configurations for different use cases

## Files Generated

$(ls -la "$SESSION_DIR" | grep -v "^total" | awk '{print "- " $9}')

## Next Steps

1. Review detailed log files for specific performance metrics
2. Run additional tests if needed
3. Implement optimizations based on findings
4. Document best practices for production deployment

---
*Report generated by SuperDARN RST Docker Performance Tester*
EOF

    log_success "Performance comparison report generated: $SESSION_DIR/performance_comparison_report.md"
}

# Main test execution
main() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}🐳 SuperDARN RST Docker Performance Tester${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""

    # Setup
    setup_test_environment
    ensure_docker_available
    
    # Build images
    build_docker_images
    
    log_info "Starting comprehensive performance testing..."
    echo ""

    # Test 1: Standard RST Performance
    log_performance "Testing Standard RST build performance..."
    run_container_performance_test "superdarn-rst-optimized_rst_standard" "standard" "standard"
    run_memory_usage_test "superdarn-rst-optimized_rst_standard" "standard"
    run_startup_time_test "superdarn-rst-optimized_rst_standard" "standard"
    echo ""

    # Test 2: Optimized RST Performance  
    log_performance "Testing Optimized RST build performance..."
    run_container_performance_test "superdarn-rst-optimized_rst_optimized" "optimized" "optimized"
    run_memory_usage_test "superdarn-rst-optimized_rst_optimized" "optimized"
    run_startup_time_test "superdarn-rst-optimized_rst_optimized" "optimized"
    echo ""

    # Test 3: Development Environment Performance
    log_performance "Testing Development environment performance..."
    run_container_performance_test "superdarn-rst-optimized_rst_development" "development" "development"
    run_memory_usage_test "superdarn-rst-optimized_rst_development" "development"
    run_startup_time_test "superdarn-rst-optimized_rst_development" "development"
    echo ""

    # Generate comparison report
    generate_performance_comparison
    
    # Summary
    echo ""
    echo -e "${GREEN}✅ Docker performance testing completed successfully!${NC}"
    echo ""
    echo -e "${CYAN}📊 Results Location:${NC} $SESSION_DIR"
    echo -e "${CYAN}📈 Main Report:${NC} $SESSION_DIR/performance_comparison_report.md"
    echo ""
    echo "Available result files:"
    ls -la "$SESSION_DIR" | grep -v "^total" | awk -v cyan="$CYAN" -v nc="$NC" '{print cyan "  - " $9 nc}'
    echo ""
    
    # Open report if possible
    if command -v open &> /dev/null; then
        log_info "Opening report in default application..."
        open "$SESSION_DIR/performance_comparison_report.md"
    elif command -v xdg-open &> /dev/null; then
        log_info "Opening report in default application..."
        xdg-open "$SESSION_DIR/performance_comparison_report.md"
    fi
    
    echo "Performance testing session complete: $TEST_SESSION"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "SuperDARN RST Docker Performance Tester"
        echo ""
        echo "USAGE:"
        echo "  $0                 # Run full performance test suite"
        echo "  $0 --help         # Show this help"
        echo "  $0 --quick        # Run quick performance tests only"
        echo "  $0 --memory       # Run memory usage tests only"
        echo "  $0 --startup      # Run startup time tests only"
        echo ""
        echo "This script builds and tests the performance of SuperDARN RST"
        echo "Docker containers, comparing standard vs optimized builds."
        exit 0
        ;;
    --quick)
        log_info "Running quick performance tests..."
        setup_test_environment
        ensure_docker_available
        run_container_performance_test "superdarn-rst-optimized_rst_standard" "quick" "quick_standard"
        run_container_performance_test "superdarn-rst-optimized_rst_optimized" "quick" "quick_optimized"
        generate_performance_comparison
        ;;
    --memory)
        log_info "Running memory usage tests..."
        setup_test_environment
        ensure_docker_available
        run_memory_usage_test "superdarn-rst-optimized_rst_standard" "memory_standard"
        run_memory_usage_test "superdarn-rst-optimized_rst_optimized" "memory_optimized"
        ;;
    --startup)
        log_info "Running startup time tests..."
        setup_test_environment
        ensure_docker_available
        run_startup_time_test "superdarn-rst-optimized_rst_standard" "startup_standard"
        run_startup_time_test "superdarn-rst-optimized_rst_optimized" "startup_optimized"
        ;;
    "")
        # Run full test suite
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
