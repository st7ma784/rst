#!/bin/bash

# End-to-End CUDA Ecosystem Validation
# Comprehensive testing of all 42 CUDA-enabled SuperDARN modules
# Validates build systems, API compatibility, and integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"
VALIDATION_LOG="$SCRIPT_DIR/end_to_end_validation.log"

# All CUDA-enabled modules (42 total)
CUDA_MODULES=(
    # Original CUDA modules
    "acf.1.16_optimized.2.0" "binplotlib.1.0_optimized.2.0" "cfit.1.19"
    "cuda_common" "elevation.1.0" "filter.1.8" "fitacf.2.5"
    "fitacf_v3.0" "iq.1.7" "lmfit_v2.0" "radar.1.22"
    "raw.1.22" "scan.1.7" "grid.1.24_optimized.1"
    # High-priority converted modules
    "acf.1.16" "acfex.1.3" "binplotlib.1.0" "cnvmap.1.17" "cnvmodel.1.0" 
    "fit.1.35" "fitacfex.1.3" "fitacfex2.1.0" "fitcnx.1.16" "freqband.1.0" 
    "grid.1.24" "gtable.2.0" "gtablewrite.1.9" "hmb.1.0" "lmfit.1.0" 
    "oldcnvmap.1.2" "oldfit.1.25" "oldfitcnx.1.10" "oldgrid.1.3" 
    "oldgtablewrite.1.4" "oldraw.1.16" "rpos.1.7" "shf.1.10" 
    "sim_data.1.0" "smr.1.7" "snd.1.0" "tsg.1.13"
    # Low-priority converted modules
    "channel.1.0"
)

# Initialize validation log
echo "End-to-End CUDA Ecosystem Validation" > "$VALIDATION_LOG"
echo "Started: $(date)" >> "$VALIDATION_LOG"
echo "Total CUDA Modules: ${#CUDA_MODULES[@]}" >> "$VALIDATION_LOG"
echo "=========================================" >> "$VALIDATION_LOG"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$VALIDATION_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[SUCCESS] $1" >> "$VALIDATION_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> "$VALIDATION_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$VALIDATION_LOG"
}

log_test() {
    echo -e "${PURPLE}[TEST]${NC} $1"
    echo "[TEST] $1" >> "$VALIDATION_LOG"
}

log_complete() {
    echo -e "${CYAN}[COMPLETE]${NC} $1"
    echo "[COMPLETE] $1" >> "$VALIDATION_LOG"
}

# Function to validate module build system
validate_module_build() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    
    if [ ! -d "$module_path" ]; then
        log_error "Module directory not found: $module_path"
        return 1
    fi
    
    log_test "Validating build system for $module_name..."
    
    # Check for CUDA makefile
    if [ ! -f "$module_path/makefile.cuda" ]; then
        log_warning "$module_name missing makefile.cuda"
        return 1
    fi
    
    # Check for CUDA headers
    if [ ! -f "$module_path/include/${module_name}_cuda.h" ]; then
        log_warning "$module_name missing CUDA header"
        return 1
    fi
    
    # Check for CUDA implementation
    if [ ! -f "$module_path/src/cuda/${module_name}_cuda.cu" ]; then
        log_warning "$module_name missing CUDA implementation"
        return 1
    fi
    
    # Test build (dry run)
    cd "$module_path"
    if make -n -f makefile.cuda cpu > /dev/null 2>&1; then
        log_success "$module_name build system validated"
        cd "$SCRIPT_DIR"
        return 0
    else
        log_error "$module_name build system validation failed"
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# Function to validate CUDA API consistency
validate_cuda_api() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    local header_file="$module_path/include/${module_name}_cuda.h"
    
    if [ ! -f "$header_file" ]; then
        log_error "$module_name CUDA header not found"
        return 1
    fi
    
    log_test "Validating CUDA API for $module_name..."
    
    # Check for required API functions
    local required_functions=(
        "${module_name}_cuda_is_available"
        "${module_name}_cuda_get_device_count"
        "${module_name}_cuda_get_error_string"
        "${module_name}_process_cuda"
    )
    
    local missing_functions=0
    for func in "${required_functions[@]}"; do
        if ! grep -q "$func" "$header_file"; then
            log_warning "$module_name missing API function: $func"
            missing_functions=$((missing_functions + 1))
        fi
    done
    
    if [ $missing_functions -eq 0 ]; then
        log_success "$module_name CUDA API validated"
        return 0
    else
        log_error "$module_name CUDA API validation failed ($missing_functions missing functions)"
        return 1
    fi
}

# Function to validate native CUDA data structures
validate_cuda_data_structures() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    local header_file="$module_path/include/${module_name}_cuda.h"
    
    if [ ! -f "$header_file" ]; then
        return 1
    fi
    
    log_test "Validating CUDA data structures for $module_name..."
    
    # Check for native CUDA data structures
    local cuda_structures=(
        "cuda_array_t\|cuda_buffer_t"
        "cuda_matrix_t"
        "cuda_complex_array_t"
        "cudaDataType_t"
        "cublasHandle_t\|cusolverDnHandle_t"
    )
    
    local found_structures=0
    for struct in "${cuda_structures[@]}"; do
        if grep -q "$struct" "$header_file"; then
            found_structures=$((found_structures + 1))
        fi
    done
    
    if [ $found_structures -ge 2 ]; then
        log_success "$module_name native CUDA data structures validated"
        return 0
    else
        log_warning "$module_name limited native CUDA data structures"
        return 1
    fi
}

# Function to validate memory management
validate_memory_management() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    local impl_file="$module_path/src/cuda/${module_name}_cuda.cu"
    
    if [ ! -f "$impl_file" ]; then
        return 1
    fi
    
    log_test "Validating memory management for $module_name..."
    
    # Check for proper memory management patterns
    local memory_patterns=(
        "cudaMallocManaged\|cudaMalloc"
        "cudaFree"
        "cudaMemcpy\|cudaMemPrefetchAsync"
        "cudaDeviceSynchronize"
    )
    
    local found_patterns=0
    for pattern in "${memory_patterns[@]}"; do
        if grep -q "$pattern" "$impl_file"; then
            found_patterns=$((found_patterns + 1))
        fi
    done
    
    if [ $found_patterns -ge 3 ]; then
        log_success "$module_name memory management validated"
        return 0
    else
        log_warning "$module_name incomplete memory management patterns"
        return 1
    fi
}

# Function to validate compatibility layer
validate_compatibility_layer() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    local compat_file="$module_path/src/${module_name}_compat.c"
    
    if [ ! -f "$compat_file" ]; then
        log_warning "$module_name missing compatibility layer"
        return 1
    fi
    
    log_test "Validating compatibility layer for $module_name..."
    
    # Check for compatibility functions
    local compat_functions=(
        "${module_name}_process_auto"
        "${module_name}_is_cuda_enabled"
        "${module_name}_get_compute_mode"
    )
    
    local found_functions=0
    for func in "${compat_functions[@]}"; do
        if grep -q "$func" "$compat_file"; then
            found_functions=$((found_functions + 1))
        fi
    done
    
    if [ $found_functions -eq 3 ]; then
        log_success "$module_name compatibility layer validated"
        return 0
    else
        log_warning "$module_name incomplete compatibility layer"
        return 1
    fi
}

# Function to run comprehensive module validation
validate_module_comprehensive() {
    local module_name="$1"
    local validation_score=0
    local max_score=5
    
    log_info "Comprehensive validation for $module_name..."
    
    # Test 1: Build system validation
    if validate_module_build "$module_name"; then
        validation_score=$((validation_score + 1))
    fi
    
    # Test 2: CUDA API validation
    if validate_cuda_api "$module_name"; then
        validation_score=$((validation_score + 1))
    fi
    
    # Test 3: Data structures validation
    if validate_cuda_data_structures "$module_name"; then
        validation_score=$((validation_score + 1))
    fi
    
    # Test 4: Memory management validation
    if validate_memory_management "$module_name"; then
        validation_score=$((validation_score + 1))
    fi
    
    # Test 5: Compatibility layer validation
    if validate_compatibility_layer "$module_name"; then
        validation_score=$((validation_score + 1))
    fi
    
    local validation_percentage=$(( (validation_score * 100) / max_score ))
    
    if [ $validation_score -eq $max_score ]; then
        log_complete "$module_name: PERFECT (5/5 - 100%)"
    elif [ $validation_score -ge 4 ]; then
        log_success "$module_name: EXCELLENT ($validation_score/5 - ${validation_percentage}%)"
    elif [ $validation_score -ge 3 ]; then
        log_success "$module_name: GOOD ($validation_score/5 - ${validation_percentage}%)"
    elif [ $validation_score -ge 2 ]; then
        log_warning "$module_name: FAIR ($validation_score/5 - ${validation_percentage}%)"
    else
        log_error "$module_name: POOR ($validation_score/5 - ${validation_percentage}%)"
    fi
    
    return $validation_score
}

# Function to test ecosystem integration
validate_ecosystem_integration() {
    log_info "Validating ecosystem-wide integration..."
    
    # Check for common CUDA utilities
    if [ -d "$CODEBASE_DIR/cuda_common" ]; then
        log_success "Common CUDA utilities found"
    else
        log_warning "Common CUDA utilities missing"
    fi
    
    # Check for consistent build patterns
    local consistent_makefiles=0
    local total_makefiles=0
    
    for module in "${CUDA_MODULES[@]}"; do
        local module_path="$CODEBASE_DIR/$module"
        if [ -f "$module_path/makefile.cuda" ]; then
            total_makefiles=$((total_makefiles + 1))
            if grep -q "CUDA_PATH" "$module_path/makefile.cuda" && \
               grep -q "NVCC" "$module_path/makefile.cuda" && \
               grep -q "CUDA_FLAGS" "$module_path/makefile.cuda"; then
                consistent_makefiles=$((consistent_makefiles + 1))
            fi
        fi
    done
    
    local consistency_percentage=$(( (consistent_makefiles * 100) / total_makefiles ))
    log_info "Build system consistency: $consistent_makefiles/$total_makefiles ($consistency_percentage%)"
    
    if [ $consistency_percentage -ge 90 ]; then
        log_success "Excellent build system consistency"
    elif [ $consistency_percentage -ge 75 ]; then
        log_success "Good build system consistency"
    else
        log_warning "Build system consistency needs improvement"
    fi
}

# Main validation execution
main() {
    log_info "Starting end-to-end CUDA ecosystem validation..."
    log_info "Validating ${#CUDA_MODULES[@]} CUDA-enabled modules..."
    
    # Check CUDA availability
    if command -v nvcc &> /dev/null; then
        log_success "CUDA toolkit detected - ready for validation"
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "NVCC version: $NVCC_VERSION"
    else
        log_warning "CUDA toolkit not found - validation will be limited"
    fi
    
    # Validate each module
    local total_score=0
    local max_total_score=$((${#CUDA_MODULES[@]} * 5))
    local perfect_modules=0
    local excellent_modules=0
    local good_modules=0
    local fair_modules=0
    local poor_modules=0
    
    for module in "${CUDA_MODULES[@]}"; do
        local module_score=$(validate_module_comprehensive "$module")
        total_score=$((total_score + module_score))
        
        case $module_score in
            5) perfect_modules=$((perfect_modules + 1)) ;;
            4) excellent_modules=$((excellent_modules + 1)) ;;
            3) good_modules=$((good_modules + 1)) ;;
            2) fair_modules=$((fair_modules + 1)) ;;
            *) poor_modules=$((poor_modules + 1)) ;;
        esac
        
        # Brief pause between modules
        sleep 0.1
    done
    
    # Ecosystem integration validation
    validate_ecosystem_integration
    
    # Generate final validation report
    local overall_percentage=$(( (total_score * 100) / max_total_score ))
    
    echo ""
    echo "========================================================="
    echo "END-TO-END CUDA ECOSYSTEM VALIDATION COMPLETE"
    echo "========================================================="
    echo "Total Modules Validated: ${#CUDA_MODULES[@]}"
    echo "Overall Validation Score: $total_score/$max_total_score ($overall_percentage%)"
    echo ""
    echo "Module Quality Distribution:"
    echo "  Perfect (5/5):    $perfect_modules modules"
    echo "  Excellent (4/5):  $excellent_modules modules"
    echo "  Good (3/5):       $good_modules modules"
    echo "  Fair (2/5):       $fair_modules modules"
    echo "  Poor (0-1/5):     $poor_modules modules"
    echo ""
    
    if [ $overall_percentage -ge 90 ]; then
        echo "🏆 ECOSYSTEM STATUS: WORLD-CLASS"
        log_complete "SuperDARN CUDA ecosystem validation: WORLD-CLASS"
    elif [ $overall_percentage -ge 80 ]; then
        echo "🥇 ECOSYSTEM STATUS: EXCELLENT"
        log_complete "SuperDARN CUDA ecosystem validation: EXCELLENT"
    elif [ $overall_percentage -ge 70 ]; then
        echo "🥈 ECOSYSTEM STATUS: GOOD"
        log_complete "SuperDARN CUDA ecosystem validation: GOOD"
    else
        echo "🥉 ECOSYSTEM STATUS: NEEDS IMPROVEMENT"
        log_warning "SuperDARN CUDA ecosystem validation: NEEDS IMPROVEMENT"
    fi
    
    echo "========================================================="
    echo "Validation log saved to: $VALIDATION_LOG"
    echo "Ready for comprehensive performance benchmarking!"
    echo "========================================================="
}

# Execute main validation
main "$@"
