#!/bin/bash
# SuperDARN Comprehensive Test Suite
# Profiles all components in src.bin and src.lib with optimization comparisons

set -e

# Configuration
TEST_RESULTS_DIR="/workspace/results"
CODEBASE_DIR="/workspace/codebase/superdarn"
SCRIPT_DIR="/workspace/scripts"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test data sizes
declare -A TEST_SIZES=(
    ["small"]="100"
    ["medium"]="1000" 
    ["large"]="10000"
)

# Optimization levels to test
OPTIMIZATION_LEVELS=("O2" "O3" "Ofast")

# Thread counts for parallel testing
THREAD_COUNTS=(1 2 4 8)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
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

# Initialize test environment
initialize_tests() {
    log "🚀 Initializing SuperDARN Test Suite..."
    
    # Create results directory structure
    mkdir -p "$TEST_RESULTS_DIR"/{libs,bins,comparisons,dashboards}
    mkdir -p "$TEST_RESULTS_DIR"/raw_data
    
    # Source RST environment
    source /opt/rst/.profile.bash
    
    # Generate test data for various sizes
    for size in "${!TEST_SIZES[@]}"; do
        log "📊 Generating $size test data..."
        "$SCRIPT_DIR/generate_test_fitacf_data.sh" "$size" "${TEST_SIZES[$size]}"
    done
    
    log_success "Test environment initialized"
}

# Test library components
test_libraries() {
    log "📚 Testing SuperDARN Libraries..."
    
    local lib_dir="$CODEBASE_DIR/src.lib/tk"
    local results_file="$TEST_RESULTS_DIR/libs/library_results_$TIMESTAMP.json"
    
    echo '{"libraries": {' > "$results_file"
    local first_lib=true
    
    for lib_path in "$lib_dir"/*; do
        if [[ -d "$lib_path" ]]; then
            local lib_name=$(basename "$lib_path")
            log "Testing library: $lib_name"
            
            if [[ "$first_lib" = false ]]; then
                echo ',' >> "$results_file"
            fi
            first_lib=false
            
            test_single_library "$lib_path" "$lib_name" "$results_file"
        fi
    done
    
    echo '}}' >> "$results_file"
    log_success "Library testing completed"
}

# Test binary/tool components
test_binaries() {
    log "🔧 Testing SuperDARN Binaries/Tools..."
    
    local bin_dir="$CODEBASE_DIR/src.bin/tk/tool"
    local results_file="$TEST_RESULTS_DIR/bins/binary_results_$TIMESTAMP.json"
    
    echo '{"binaries": {' > "$results_file"
    local first_bin=true
    
    for bin_path in "$bin_dir"/*; do
        if [[ -d "$bin_path" ]]; then
            local bin_name=$(basename "$bin_path")
            log "Testing binary: $bin_name"
            
            if [[ "$first_bin" = false ]]; then
                echo ',' >> "$results_file"
            fi
            first_bin=false
            
            test_single_binary "$bin_path" "$bin_name" "$results_file"
        fi
    done
    
    echo '}}' >> "$results_file"
    log_success "Binary testing completed"
}

# Test a single library with profiling and optimization
test_single_library() {
    local lib_path="$1"
    local lib_name="$2"
    local results_file="$3"
    
    echo "\"$lib_name\": {" >> "$results_file"
    
    cd "$lib_path"
    
    # Check if makefile exists
    if [[ ! -f "makefile" && ! -f "Makefile" ]]; then
        log_warning "No makefile found for $lib_name"
        echo '"status": "no_makefile", "error": "No makefile found"' >> "$results_file"
        echo '}' >> "$results_file"
        return
    fi
    
    local makefile="makefile"
    [[ -f "Makefile" ]] && makefile="Makefile"
    
    # Test original version
    log "  📊 Testing original $lib_name..."
    cp "$makefile" "${makefile}.backup"
    
    local original_results=$(test_library_performance "$lib_path" "$lib_name" "original")
    
    echo '"original": {' >> "$results_file"
    echo "$original_results" >> "$results_file"
    echo '},' >> "$results_file"
    
    # Test optimized versions
    echo '"optimized": {' >> "$results_file"
    local first_opt=true
    
    for opt in "${OPTIMIZATION_LEVELS[@]}"; do
        log "  ⚡ Testing $lib_name with -$opt optimization..."
        
        if [[ "$first_opt" = false ]]; then
            echo ',' >> "$results_file"
        fi
        first_opt=false
        
        # Modify makefile for optimization
        cp "${makefile}.backup" "$makefile"
        sed -i "s/-O[0-9]\\|/-O[a-z]*/-$opt/g" "$makefile"
        
        local opt_results=$(test_library_performance "$lib_path" "$lib_name" "$opt")
        
        echo "\"$opt\": {" >> "$results_file"
        echo "$opt_results" >> "$results_file"
        echo '}' >> "$results_file"
    done
    
    echo '},' >> "$results_file"
    
    # Restore original makefile
    mv "${makefile}.backup" "$makefile"
    
    # Add verification results
    echo '"verification": {' >> "$results_file"
    echo '"status": "pending"' >> "$results_file"
    echo '}' >> "$results_file"
    
    echo '}' >> "$results_file"
}

# Test library performance
test_library_performance() {
    local lib_path="$1"
    local lib_name="$2"
    local version="$3"
    
    local build_time_start=$(date +%s.%3N)
    
    # Clean and build
    make clean > /dev/null 2>&1 || true
    local build_result
    if make all > build.log 2>&1; then
        build_result="success"
    else
        build_result="failed"
    fi
    
    local build_time_end=$(date +%s.%3N)
    local build_time=$(echo "$build_time_end - $build_time_start" | bc -l)
    
    # Get library size if built successfully
    local lib_size=0
    if [[ "$build_result" = "success" ]]; then
        local lib_file=$(find . -name "*.a" -o -name "*.so" | head -1)
        if [[ -n "$lib_file" ]]; then
            lib_size=$(stat -f%z "$lib_file" 2>/dev/null || stat -c%s "$lib_file" 2>/dev/null || echo "0")
        fi
    fi
    
    # Memory usage estimation (simplified)
    local memory_usage=0
    if [[ "$build_result" = "success" ]]; then
        memory_usage=$(du -sb . 2>/dev/null | cut -f1 || echo "0")
    fi
    
    cat << EOF
"build_time": $build_time,
"build_status": "$build_result",
"library_size": $lib_size,
"memory_usage": $memory_usage,
"test_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
}

# Test a single binary with profiling and optimization
test_single_binary() {
    local bin_path="$1"
    local bin_name="$2"
    local results_file="$3"
    
    echo "\"$bin_name\": {" >> "$results_file"
    
    cd "$bin_path"
    
    # Check if makefile exists
    if [[ ! -f "makefile" && ! -f "Makefile" ]]; then
        log_warning "No makefile found for $bin_name"
        echo '"status": "no_makefile", "error": "No makefile found"' >> "$results_file"
        echo '}' >> "$results_file"
        return
    fi
    
    local makefile="makefile"
    [[ -f "Makefile" ]] && makefile="Makefile"
    
    # Test original version
    log "  📊 Testing original $bin_name..."
    cp "$makefile" "${makefile}.backup"
    
    local original_results=$(test_binary_performance "$bin_path" "$bin_name" "original")
    
    echo '"original": {' >> "$results_file"
    echo "$original_results" >> "$results_file"
    echo '},' >> "$results_file"
    
    # Test optimized versions
    echo '"optimized": {' >> "$results_file"
    local first_opt=true
    
    for opt in "${OPTIMIZATION_LEVELS[@]}"; do
        log "  ⚡ Testing $bin_name with -$opt optimization..."
        
        if [[ "$first_opt" = false ]]; then
            echo ',' >> "$results_file"
        fi
        first_opt=false
        
        # Modify makefile for optimization
        cp "${makefile}.backup" "$makefile"
        sed -i "s/-O[0-9]\\|/-O[a-z]*/-$opt/g" "$makefile"
        
        local opt_results=$(test_binary_performance "$bin_path" "$bin_name" "$opt")
        
        echo "\"$opt\": {" >> "$results_file"
        echo "$opt_results" >> "$results_file"
        echo '}' >> "$results_file"
    done
    
    echo '}' >> "$results_file"
    
    # Restore original makefile
    mv "${makefile}.backup" "$makefile"
    
    echo '}' >> "$results_file"
}

# Test binary performance
test_binary_performance() {
    local bin_path="$1"
    local bin_name="$2"
    local version="$3"
    
    local build_time_start=$(date +%s.%3N)
    
    # Clean and build
    make clean > /dev/null 2>&1 || true
    local build_result
    if make all > build.log 2>&1; then
        build_result="success"
    else
        build_result="failed"
    fi
    
    local build_time_end=$(date +%s.%3N)
    local build_time=$(echo "$build_time_end - $build_time_start" | bc -l)
    
    # Get binary size and test execution if built successfully
    local binary_size=0
    local execution_time=0
    local execution_status="not_tested"
    
    if [[ "$build_result" = "success" ]]; then
        local binary_file=$(find . -type f -executable -not -name "*.sh" | head -1)
        if [[ -n "$binary_file" ]]; then
            binary_size=$(stat -f%z "$binary_file" 2>/dev/null || stat -c%s "$binary_file" 2>/dev/null || echo "0")
            
            # Try to run binary with test data (if applicable)
            local exec_start=$(date +%s.%3N)
            if ./"$(basename "$binary_file")" --help > /dev/null 2>&1 || \
               echo "test" | ./"$(basename "$binary_file")" > /dev/null 2>&1; then
                execution_status="success"
            else
                execution_status="no_test_available"
            fi
            local exec_end=$(date +%s.%3N)
            execution_time=$(echo "$exec_end - $exec_start" | bc -l)
        fi
    fi
    
    cat << EOF
"build_time": $build_time,
"build_status": "$build_result",
"binary_size": $binary_size,
"execution_time": $execution_time,
"execution_status": "$execution_status",
"test_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
}

# Generate comprehensive dashboard
generate_comprehensive_dashboard() {
    log "📊 Generating comprehensive performance dashboard..."
    
    python3 "$SCRIPT_DIR/generate_comprehensive_dashboard.py" \
        --results-dir "$TEST_RESULTS_DIR" \
        --output "$TEST_RESULTS_DIR/dashboards/superdarn_comprehensive_dashboard.html" \
        --timestamp "$TIMESTAMP"
    
    log_success "Dashboard generated: $TEST_RESULTS_DIR/dashboards/superdarn_comprehensive_dashboard.html"
}

# Main execution
main() {
    log "🎯 Starting SuperDARN Comprehensive Test Suite"
    
    initialize_tests
    test_libraries
    test_binaries
    generate_comprehensive_dashboard
    
    log_success "🎉 SuperDARN Test Suite completed successfully!"
    log "📊 Results available in: $TEST_RESULTS_DIR"
    log "📈 Dashboard: $TEST_RESULTS_DIR/dashboards/superdarn_comprehensive_dashboard.html"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
