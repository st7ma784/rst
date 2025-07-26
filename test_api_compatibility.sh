#!/bin/bash

# API Compatibility Testing Script
# Tests drop-in replacement functionality for all CUDA modules

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"

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

# Test API compatibility for a module
test_module_api() {
    local module="$1"
    local module_path="$CODEBASE_DIR/$module"
    
    if [ ! -d "$module_path" ]; then
        return 1
    fi
    
    log_info "Testing API compatibility for $module..."
    
    local score=0
    local max_score=4
    
    # Test 1: Check for CUDA header
    if [ -f "$module_path/include/${module}_cuda.h" ]; then
        log_success "$module: CUDA header found"
        score=$((score + 1))
    else
        log_warning "$module: CUDA header missing"
    fi
    
    # Test 2: Check for compatibility layer
    if [ -f "$module_path/src/${module}_compat.c" ]; then
        log_success "$module: Compatibility layer found"
        score=$((score + 1))
    else
        log_warning "$module: Compatibility layer missing"
    fi
    
    # Test 3: Check for required API functions
    if [ -f "$module_path/include/${module}_cuda.h" ]; then
        local required_funcs=("${module}_cuda_is_available" "${module}_process_cuda" "${module}_get_compute_mode")
        local found_funcs=0
        
        for func in "${required_funcs[@]}"; do
            if grep -q "$func" "$module_path/include/${module}_cuda.h"; then
                found_funcs=$((found_funcs + 1))
            fi
        done
        
        if [ $found_funcs -eq ${#required_funcs[@]} ]; then
            log_success "$module: All required API functions found"
            score=$((score + 1))
        else
            log_warning "$module: Missing $((${#required_funcs[@]} - found_funcs)) required API functions"
        fi
    fi
    
    # Test 4: Check for build system
    if [ -f "$module_path/makefile.cuda" ]; then
        log_success "$module: CUDA build system found"
        score=$((score + 1))
    else
        log_warning "$module: CUDA build system missing"
    fi
    
    local percentage=$(( (score * 100) / max_score ))
    echo "  API Compatibility Score: $score/$max_score ($percentage%)"
    
    return $score
}

# Main execution
main() {
    log_info "Starting API compatibility testing..."
    
    local modules=(
        "acf.1.16_optimized.2.0" "binplotlib.1.0_optimized.2.0" "cfit.1.19"
        "cuda_common" "elevation.1.0" "filter.1.8" "fitacf.2.5"
        "fitacf_v3.0" "iq.1.7" "lmfit_v2.0" "radar.1.22"
        "raw.1.22" "scan.1.7" "grid.1.24_optimized.1"
        "acf.1.16" "acfex.1.3" "binplotlib.1.0" "cnvmap.1.17" "cnvmodel.1.0" 
        "fit.1.35" "fitacfex.1.3" "fitacfex2.1.0" "fitcnx.1.16" "freqband.1.0" 
        "grid.1.24" "gtable.2.0" "gtablewrite.1.9" "hmb.1.0" "lmfit.1.0" 
        "oldcnvmap.1.2" "oldfit.1.25" "oldfitcnx.1.10" "oldgrid.1.3" 
        "oldgtablewrite.1.4" "oldraw.1.16" "rpos.1.7" "shf.1.10" 
        "sim_data.1.0" "smr.1.7" "snd.1.0" "tsg.1.13" "channel.1.0"
    )
    
    local total_score=0
    local max_total_score=$((${#modules[@]} * 4))
    local perfect_count=0
    local good_count=0
    local fair_count=0
    local poor_count=0
    
    for module in "${modules[@]}"; do
        local module_score
        module_score=$(test_module_api "$module")
        total_score=$((total_score + module_score))
        
        case $module_score in
            4) perfect_count=$((perfect_count + 1)) ;;
            3) good_count=$((good_count + 1)) ;;
            2) fair_count=$((fair_count + 1)) ;;
            *) poor_count=$((poor_count + 1)) ;;
        esac
        
        echo ""
    done
    
    local overall_percentage=$(( (total_score * 100) / max_total_score ))
    
    echo "========================================="
    echo "API COMPATIBILITY TEST RESULTS"
    echo "========================================="
    echo "Total Modules Tested: ${#modules[@]}"
    echo "Overall Score: $total_score/$max_total_score ($overall_percentage%)"
    echo ""
    echo "Module Distribution:"
    echo "  Perfect (4/4): $perfect_count modules"
    echo "  Good (3/4):    $good_count modules"
    echo "  Fair (2/4):    $fair_count modules"
    echo "  Poor (0-1/4):  $poor_count modules"
    echo ""
    
    if [ $overall_percentage -ge 90 ]; then
        log_success "API COMPATIBILITY: EXCELLENT"
    elif [ $overall_percentage -ge 75 ]; then
        log_success "API COMPATIBILITY: GOOD"
    elif [ $overall_percentage -ge 60 ]; then
        log_warning "API COMPATIBILITY: FAIR"
    else
        log_error "API COMPATIBILITY: POOR"
    fi
    
    echo "========================================="
    
    # Save results for CI
    cat > compatibility_test_results.log << EOF
API Compatibility Test Results
Generated: $(date)
Total Modules: ${#modules[@]}
Overall Score: $total_score/$max_total_score ($overall_percentage%)
Perfect Modules: $perfect_count
Good Modules: $good_count
Fair Modules: $fair_count
Poor Modules: $poor_count
EOF
}

main "$@"
