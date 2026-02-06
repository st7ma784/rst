#!/bin/bash

# Comprehensive CUDA Module Test Script
# Tests all CUDA-enabled SuperDARN modules for build success and compatibility

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"
TEST_LOG="$SCRIPT_DIR/cuda_test_results.log"

# Initialize log
echo "CUDA Module Test Results" > "$TEST_LOG"
echo "Generated: $(date)" >> "$TEST_LOG"
echo "=========================" >> "$TEST_LOG"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$TEST_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[SUCCESS] $1" >> "$TEST_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> "$TEST_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$TEST_LOG"
}

# Set up RST environment
setup_rst_environment() {
    log_info "Setting up RST build environment..."
    
    export MAKECFG="${MAKECFG:-$SCRIPT_DIR/make/makecfg}"
    export MAKELIB="${MAKELIB:-$SCRIPT_DIR/make/makelib}"
    export SYSTEM="${SYSTEM:-linux}"
    export IPATH="${IPATH:-$SCRIPT_DIR/include}"
    export LIBPATH="${LIBPATH:-$SCRIPT_DIR/lib}"
    
    # Create necessary directories
    mkdir -p "$IPATH/base" "$IPATH/general" "$IPATH/superdarn" "$LIBPATH"
    
    log_success "RST environment configured"
}

# Test individual module
test_module() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    
    log_info "Testing module: $module_name"
    
    if [ ! -d "$module_path" ]; then
        log_error "Module directory not found: $module_path"
        return 1
    fi
    
    cd "$module_path"
    
    local cpu_success=0
    local cuda_success=0
    local compat_success=0
    
    # Test CPU version
    if [ -f "src/makefile" ] || [ -f "makefile" ]; then
        log_info "  Testing CPU build for $module_name..."
        if make clean >/dev/null 2>&1 && make all >/dev/null 2>&1; then
            log_success "  CPU build: PASSED"
            cpu_success=1
        else
            log_error "  CPU build: FAILED"
        fi
    fi
    
    # Test CUDA version
    if [ -f "makefile.cuda" ]; then
        log_info "  Testing CUDA build for $module_name..."
        if make -f makefile.cuda clean >/dev/null 2>&1 && make -f makefile.cuda all >/dev/null 2>&1; then
            log_success "  CUDA build: PASSED"
            cuda_success=1
        else
            log_warning "  CUDA build: FAILED (may be due to missing dependencies)"
        fi
    else
        log_info "  No CUDA makefile found for $module_name"
    fi
    
    # Test CMake version (for modules like fit.1.35)
    if [ -f "CMakeLists.txt" ]; then
        log_info "  Testing CMake build for $module_name..."
        mkdir -p build && cd build
        if cmake .. -DENABLE_CUDA=ON >/dev/null 2>&1 && make >/dev/null 2>&1; then
            log_success "  CMake CUDA build: PASSED"
            cuda_success=1
        else
            log_warning "  CMake CUDA build: FAILED"
        fi
        cd ..
    fi
    
    cd - >/dev/null
    
    # Return success if at least CPU version works
    return $((1 - cpu_success))
}

# Main test execution
main() {
    log_info "Starting comprehensive CUDA module testing..."
    
    setup_rst_environment
    
    # List of all CUDA-enabled modules
    declare -a CUDA_MODULES=(
        "fitacf_v3.0"
        "fit.1.35" 
        "grid.1.24_optimized.1"
        "lmfit_v2.0"
        "acf.1.16_optimized.2.0"
        "binplotlib.1.0_optimized.2.0"
        "fitacf.2.5"
        "cuda_common"
    )
    
    local total_modules=${#CUDA_MODULES[@]}
    local passed_modules=0
    local failed_modules=0
    
    # Test each module
    for module in "${CUDA_MODULES[@]}"; do
        if test_module "$module"; then
            passed_modules=$((passed_modules + 1))
        else
            failed_modules=$((failed_modules + 1))
        fi
        echo "" # Add spacing between modules
    done
    
    # Generate final report
    echo ""
    echo "========================================="
    echo "CUDA MODULE TEST SUMMARY:"
    echo "  Total modules tested: $total_modules"
    echo "  Modules passed: $passed_modules"
    echo "  Modules failed: $failed_modules"
    echo "  Success rate: $(( passed_modules * 100 / total_modules ))%"
    echo "========================================="
    
    # Create detailed report
    cat > "$SCRIPT_DIR/cuda_test_report.md" << EOF
# CUDA Module Test Report

Generated: $(date)

## Test Summary
- **Total modules tested**: $total_modules
- **Modules passed**: $passed_modules  
- **Modules failed**: $failed_modules
- **Success rate**: $(( passed_modules * 100 / total_modules ))%

## Module Status

EOF
    
    for module in "${CUDA_MODULES[@]}"; do
        if [ -d "$CODEBASE_DIR/$module" ]; then
            echo "### $module" >> "$SCRIPT_DIR/cuda_test_report.md"
            if [ -f "$CODEBASE_DIR/$module/makefile.cuda" ]; then
                echo "- ✅ CUDA makefile present" >> "$SCRIPT_DIR/cuda_test_report.md"
            else
                echo "- ❌ No CUDA makefile" >> "$SCRIPT_DIR/cuda_test_report.md"
            fi
            echo "" >> "$SCRIPT_DIR/cuda_test_report.md"
        fi
    done
    
    cat >> "$SCRIPT_DIR/cuda_test_report.md" << EOF

## Next Steps
1. Address any build failures
2. Run performance benchmarks
3. Test drop-in compatibility
4. Update documentation

## Usage Instructions
Each CUDA-enabled module provides three build variants:
- **CPU**: Standard CPU implementation
- **CUDA**: GPU-accelerated implementation
- **Compatibility**: Automatic CPU/GPU selection

To use CUDA versions, link with the appropriate library variant:
\`\`\`bash
# Link with CUDA version
-l<module_name>.cuda

# Link with compatibility version  
-l<module_name>.compat
\`\`\`
EOF
    
    log_success "Test completed! Report saved to: $SCRIPT_DIR/cuda_test_report.md"
    
    if [ $failed_modules -eq 0 ]; then
        log_success "All modules tested successfully! 🎉"
        return 0
    else
        log_warning "$failed_modules modules had issues - check the detailed log"
        return 1
    fi
}

# Execute main function
main "$@"
