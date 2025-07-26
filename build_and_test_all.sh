#!/bin/bash

# Comprehensive SuperDARN Module Build and Test Script
# Tests all modules (standard and CUDA) and validates drop-in replacement capability
# Author: CUDA Conversion Project
# Date: 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"
BUILD_LOG="$SCRIPT_DIR/build_report.log"
TEST_LOG="$SCRIPT_DIR/test_report.log"
REPORT_FILE="$SCRIPT_DIR/module_compatibility_report.md"

# Build status tracking
declare -A BUILD_STATUS
declare -A TEST_STATUS
declare -A CUDA_AVAILABLE_STATUS

# Initialize logs
echo "SuperDARN Module Build and Test Report" > "$BUILD_LOG"
echo "Generated: $(date)" >> "$BUILD_LOG"
echo "========================================" >> "$BUILD_LOG"
echo "" >> "$BUILD_LOG"

echo "SuperDARN Module Test Report" > "$TEST_LOG"
echo "Generated: $(date)" >> "$TEST_LOG"
echo "============================" >> "$TEST_LOG"
echo "" >> "$TEST_LOG"

# Function to log with color
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$BUILD_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[SUCCESS] $1" >> "$BUILD_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> "$BUILD_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$BUILD_LOG"
}

# Function to check CUDA availability
check_cuda_availability() {
    log_info "Checking CUDA availability..."
    
    # Check for nvcc compiler
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_success "NVCC compiler found: version $NVCC_VERSION"
        
        # Check for CUDA runtime
        if command -v nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            if [ $? -eq 0 ]; then
                log_success "CUDA runtime available: $GPU_INFO"
                return 0
            else
                log_warning "NVCC found but no CUDA runtime detected"
                return 1
            fi
        else
            log_warning "NVCC found but nvidia-smi not available"
            return 1
        fi
    else
        log_warning "NVCC compiler not found - CUDA builds will be skipped"
        return 1
    fi
}

# Function to build a module with multiple variants
build_module() {
    local module_name="$1"
    local module_path="$2"
    local build_system="$3"  # "makefile", "cmake", or "both"
    
    log_info "Building module: $module_name"
    
    if [ ! -d "$module_path" ]; then
        log_error "Module directory not found: $module_path"
        BUILD_STATUS["$module_name"]="MISSING"
        return 1
    fi
    
    cd "$module_path"
    
    local success=true
    
    case "$build_system" in
        "makefile")
            build_with_makefile "$module_name" || success=false
            ;;
        "cmake")
            build_with_cmake "$module_name" || success=false
            ;;
        "both")
            build_with_makefile "$module_name" || success=false
            build_with_cmake "$module_name" || success=false
            ;;
        *)
            log_error "Unknown build system: $build_system"
            success=false
            ;;
    esac
    
    if $success; then
        BUILD_STATUS["$module_name"]="SUCCESS"
        log_success "Module $module_name built successfully"
    else
        BUILD_STATUS["$module_name"]="FAILED"
        log_error "Module $module_name build failed"
    fi
    
    cd "$SCRIPT_DIR"
    return $([ "$success" = "true" ])
}

# Function to build with makefile
build_with_makefile() {
    local module_name="$1"
    
    log_info "Building $module_name with Makefile..."
    
    # Set up RST environment
    export SYSTEM=linux
    export MAKECFG="$SCRIPT_DIR/build/make/makecfg"
    export MAKELIB="$SCRIPT_DIR/build/make/makelib"
    export MAKEBIN="$SCRIPT_DIR/build/make/makebin"
    export LIBPATH="$SCRIPT_DIR/build/lib"
    export IPATH="$SCRIPT_DIR/build/include"
    export BINPATH="$SCRIPT_DIR/build/bin"
    
    # Ensure build directories exist
    mkdir -p "$LIBPATH" "$IPATH" "$BINPATH"
    
    # Check for RST-style makefile in src directory
    if [ -f "src/makefile" ]; then
        log_info "Building RST-style module in src/..."
        cd src
        if make clean 2>/dev/null; then
            log_info "Cleaned previous build"
        fi
        if make all 2>&1 | tee -a "$BUILD_LOG"; then
            log_success "RST makefile build succeeded"
            cd ..
        else
            log_error "RST makefile build failed"
            cd ..
            return 1
        fi
    # Check for standard makefile in root
    elif [ -f "makefile" ] || [ -f "Makefile" ]; then
        log_info "Building standard makefile..."
        if make clean 2>/dev/null; then
            log_info "Cleaned previous build"
        fi
        if make all 2>&1 | tee -a "$BUILD_LOG"; then
            log_success "Standard makefile build succeeded"
        else
            log_error "Standard makefile build failed"
            return 1
        fi
    else
        log_warning "No makefile found for $module_name"
    fi
    
    # Check for CUDA makefile
    if [ -f "makefile.cuda" ]; then
        log_info "Building CUDA version..."
        if check_cuda_availability; then
            if make -f makefile.cuda clean 2>/dev/null; then
                log_info "Cleaned CUDA build"
            fi
            if make -f makefile.cuda all 2>&1 | tee -a "$BUILD_LOG"; then
                log_success "CUDA makefile build succeeded"
                CUDA_AVAILABLE_STATUS["$module_name"]="SUCCESS"
            else
                log_error "CUDA makefile build failed"
                CUDA_AVAILABLE_STATUS["$module_name"]="FAILED"
                return 1
            fi
        else
            log_warning "Skipping CUDA build - CUDA not available"
            CUDA_AVAILABLE_STATUS["$module_name"]="SKIPPED"
        fi
    fi
    
    return 0
}

# Function to build with CMake
build_with_cmake() {
    local module_name="$1"
    
    log_info "Building $module_name with CMake..."
    
    if [ ! -f "CMakeLists.txt" ]; then
        log_warning "No CMakeLists.txt found for $module_name"
        return 0
    fi
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure and build standard version
    log_info "Configuring CMake for standard build..."
    if cmake .. -DENABLE_CUDA=OFF -DENABLE_AVX=ON -DBUILD_TESTS=ON 2>&1 | tee -a "$BUILD_LOG"; then
        log_info "Building standard version..."
        if make -j$(nproc) 2>&1 | tee -a "$BUILD_LOG"; then
            log_success "CMake standard build succeeded"
        else
            log_error "CMake standard build failed"
            cd ..
            return 1
        fi
    else
        log_error "CMake configuration failed"
        cd ..
        return 1
    fi
    
    # Build CUDA version if available
    if check_cuda_availability; then
        log_info "Configuring CMake for CUDA build..."
        rm -rf *  # Clean build directory
        if cmake .. -DENABLE_CUDA=ON -DENABLE_AVX=ON -DBUILD_TESTS=ON 2>&1 | tee -a "$BUILD_LOG"; then
            log_info "Building CUDA version..."
            if make -j$(nproc) 2>&1 | tee -a "$BUILD_LOG"; then
                log_success "CMake CUDA build succeeded"
                CUDA_AVAILABLE_STATUS["$module_name"]="SUCCESS"
            else
                log_error "CMake CUDA build failed"
                CUDA_AVAILABLE_STATUS["$module_name"]="FAILED"
                cd ..
                return 1
            fi
        else
            log_error "CMake CUDA configuration failed"
            CUDA_AVAILABLE_STATUS["$module_name"]="FAILED"
            cd ..
            return 1
        fi
    else
        log_warning "Skipping CUDA build - CUDA not available"
        CUDA_AVAILABLE_STATUS["$module_name"]="SKIPPED"
    fi
    
    cd ..
    return 0
}

# Function to test drop-in replacement capability
test_drop_in_replacement() {
    local module_name="$1"
    local module_path="$2"
    
    log_info "Testing drop-in replacement for: $module_name"
    
    cd "$module_path"
    
    # Look for test programs
    local test_success=true
    
    if [ -d "tests" ]; then
        log_info "Running tests for $module_name..."
        
        # Try to run existing tests
        if [ -f "tests/makefile" ]; then
            if make -C tests clean && make -C tests all 2>&1 | tee -a "$TEST_LOG"; then
                log_info "Running test suite..."
                if make -C tests test 2>&1 | tee -a "$TEST_LOG"; then
                    log_success "Tests passed for $module_name"
                else
                    log_warning "Some tests failed for $module_name"
                    test_success=false
                fi
            else
                log_warning "Test compilation failed for $module_name"
                test_success=false
            fi
        fi
        
        # Look for specific test executables
        find tests -name "*test*" -executable -type f | while read test_exec; do
            log_info "Running test: $test_exec"
            if timeout 60 "$test_exec" 2>&1 | tee -a "$TEST_LOG"; then
                log_success "Test $test_exec passed"
            else
                log_warning "Test $test_exec failed or timed out"
                test_success=false
            fi
        done
    fi
    
    # Test API compatibility by checking headers
    if [ -d "include" ]; then
        log_info "Checking API compatibility for $module_name..."
        
        # Check for header consistency
        local header_count=$(find include -name "*.h" | wc -l)
        if [ $header_count -gt 0 ]; then
            log_success "Found $header_count header files"
            
            # Basic header syntax check
            find include -name "*.h" | while read header; do
                if gcc -fsyntax-only -I. -I../.. "$header" 2>/dev/null; then
                    log_success "Header $header syntax OK"
                else
                    log_warning "Header $header has syntax issues"
                    test_success=false
                fi
            done
        else
            log_warning "No header files found for $module_name"
        fi
    fi
    
    if $test_success; then
        TEST_STATUS["$module_name"]="SUCCESS"
    else
        TEST_STATUS["$module_name"]="FAILED"
    fi
    
    cd "$SCRIPT_DIR"
    return $([ "$test_success" = "true" ])
}

# Function to generate comprehensive report
generate_report() {
    log_info "Generating comprehensive report..."
    
    cat > "$REPORT_FILE" << EOF
# SuperDARN Module Compatibility Report

Generated: $(date)

## Executive Summary

This report details the build status and drop-in replacement capability of all SuperDARN modules, including CUDA variants.

## CUDA Environment

EOF
    
    if check_cuda_availability; then
        echo "✅ CUDA is available and functional" >> "$REPORT_FILE"
        nvcc --version | head -1 >> "$REPORT_FILE"
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/GPU: /' >> "$REPORT_FILE"
    else
        echo "❌ CUDA is not available - CUDA builds were skipped" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "## Module Build Status" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| Module | Standard Build | CUDA Build | Tests | Drop-in Ready |" >> "$REPORT_FILE"
    echo "|--------|---------------|------------|-------|---------------|" >> "$REPORT_FILE"
    
    for module in "${!BUILD_STATUS[@]}"; do
        local std_status="${BUILD_STATUS[$module]}"
        local cuda_status="${CUDA_AVAILABLE_STATUS[$module]:-N/A}"
        local test_status="${TEST_STATUS[$module]:-N/A}"
        
        # Determine drop-in readiness
        local drop_in_ready="❌"
        if [[ "$std_status" == "SUCCESS" && ("$cuda_status" == "SUCCESS" || "$cuda_status" == "SKIPPED") ]]; then
            if [[ "$test_status" == "SUCCESS" || "$test_status" == "N/A" ]]; then
                drop_in_ready="✅"
            fi
        fi
        
        # Convert status to emojis
        local std_emoji="❌"
        [[ "$std_status" == "SUCCESS" ]] && std_emoji="✅"
        
        local cuda_emoji="❌"
        [[ "$cuda_status" == "SUCCESS" ]] && cuda_emoji="✅"
        [[ "$cuda_status" == "SKIPPED" ]] && cuda_emoji="⏭️"
        [[ "$cuda_status" == "N/A" ]] && cuda_emoji="➖"
        
        local test_emoji="❌"
        [[ "$test_status" == "SUCCESS" ]] && test_emoji="✅"
        [[ "$test_status" == "N/A" ]] && test_emoji="➖"
        
        echo "| $module | $std_emoji $std_status | $cuda_emoji $cuda_status | $test_emoji $test_status | $drop_in_ready |" >> "$REPORT_FILE"
    done
    
    echo "" >> "$REPORT_FILE"
    echo "## Detailed Issues" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Report failed modules
    local failed_modules=()
    for module in "${!BUILD_STATUS[@]}"; do
        if [[ "${BUILD_STATUS[$module]}" != "SUCCESS" ]]; then
            failed_modules+=("$module")
        fi
    done
    
    if [ ${#failed_modules[@]} -gt 0 ]; then
        echo "### Modules requiring attention:" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        for module in "${failed_modules[@]}"; do
            echo "- **$module**: ${BUILD_STATUS[$module]}" >> "$REPORT_FILE"
        done
        echo "" >> "$REPORT_FILE"
    else
        echo "### All modules built successfully! 🎉" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    echo "## Build Logs" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "- Full build log: \`$BUILD_LOG\`" >> "$REPORT_FILE"
    echo "- Test log: \`$TEST_LOG\`" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "## Next Steps" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "1. Fix any failed builds identified above" >> "$REPORT_FILE"
    echo "2. Ensure all CUDA variants provide identical APIs" >> "$REPORT_FILE"
    echo "3. Run performance benchmarks comparing CPU vs CUDA implementations" >> "$REPORT_FILE"
    echo "4. Update documentation for drop-in replacement usage" >> "$REPORT_FILE"
    
    log_success "Report generated: $REPORT_FILE"
}

# Main execution
main() {
    log_info "Starting SuperDARN Module Build and Test Process"
    log_info "Script directory: $SCRIPT_DIR"
    log_info "Codebase directory: $CODEBASE_DIR"
    
    if [ ! -d "$CODEBASE_DIR" ]; then
        log_error "Codebase directory not found: $CODEBASE_DIR"
        exit 1
    fi
    
    # Check CUDA availability globally
    check_cuda_availability
    
    # Define modules to build and test
    declare -A MODULES=(
        ["fitacf_v3.0"]="$CODEBASE_DIR/fitacf_v3.0:makefile"
        ["fit.1.35"]="$CODEBASE_DIR/fit.1.35:cmake"
        ["grid.1.24_optimized.1"]="$CODEBASE_DIR/grid.1.24_optimized.1:both"
        ["lmfit_v2.0"]="$CODEBASE_DIR/lmfit_v2.0:makefile"
        ["acf.1.16_optimized.2.0"]="$CODEBASE_DIR/acf.1.16_optimized.2.0:makefile"
        ["binplotlib.1.0_optimized.2.0"]="$CODEBASE_DIR/binplotlib.1.0_optimized.2.0:makefile"
        ["fitacf.2.5"]="$CODEBASE_DIR/fitacf.2.5:makefile"
        ["grid.1.24"]="$CODEBASE_DIR/grid.1.24:makefile"
        ["cfit.1.19"]="$CODEBASE_DIR/cfit.1.19:makefile"
        ["raw.1.22"]="$CODEBASE_DIR/raw.1.22:makefile"
        ["radar.1.22"]="$CODEBASE_DIR/radar.1.22:makefile"
    )
    
    # Build all modules
    log_info "Building all modules..."
    for module_spec in "${!MODULES[@]}"; do
        IFS=':' read -r module_path build_system <<< "${MODULES[$module_spec]}"
        build_module "$module_spec" "$module_path" "$build_system"
    done
    
    # Test all modules
    log_info "Testing all modules..."
    for module_spec in "${!MODULES[@]}"; do
        IFS=':' read -r module_path build_system <<< "${MODULES[$module_spec]}"
        test_drop_in_replacement "$module_spec" "$module_path"
    done
    
    # Generate final report
    generate_report
    
    # Summary
    echo ""
    log_info "Build and test process complete!"
    log_info "Report available at: $REPORT_FILE"
    
    # Count successes and failures
    local total_modules=${#MODULES[@]}
    local successful_builds=0
    local failed_builds=0
    
    for status in "${BUILD_STATUS[@]}"; do
        if [[ "$status" == "SUCCESS" ]]; then
            ((successful_builds++))
        else
            ((failed_builds++))
        fi
    done
    
    echo ""
    echo "========================================="
    echo "BUILD SUMMARY:"
    echo "  Total modules: $total_modules"
    echo "  Successful builds: $successful_builds"
    echo "  Failed builds: $failed_builds"
    echo "========================================="
    
    if [ $failed_builds -eq 0 ]; then
        log_success "All modules built successfully! 🎉"
        exit 0
    else
        log_warning "$failed_builds modules require attention"
        exit 1
    fi
}

# Run main function
main "$@"
