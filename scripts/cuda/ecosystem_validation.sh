#!/bin/bash

# Simplified CUDA Ecosystem Validation
# Tests all CUDA-enabled modules for completeness

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"

# All CUDA modules
CUDA_MODULES=(
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

# Validate single module
validate_module() {
    local module="$1"
    local module_path="$CODEBASE_DIR/$module"
    local score=0
    
    echo "Validating $module..."
    
    if [ ! -d "$module_path" ]; then
        echo "  ❌ Directory missing"
        return 0
    fi
    
    # Check for CUDA makefile
    if [ -f "$module_path/makefile.cuda" ]; then
        echo "  ✅ CUDA makefile found"
        score=$((score + 1))
    else
        echo "  ❌ CUDA makefile missing"
    fi
    
    # Check for CUDA header
    if [ -f "$module_path/include/${module}_cuda.h" ]; then
        echo "  ✅ CUDA header found"
        score=$((score + 1))
    else
        echo "  ❌ CUDA header missing"
    fi
    
    # Check for CUDA implementation
    if [ -f "$module_path/src/cuda/${module}_cuda.cu" ]; then
        echo "  ✅ CUDA implementation found"
        score=$((score + 1))
    else
        echo "  ❌ CUDA implementation missing"
    fi
    
    # Check for compatibility layer
    if [ -f "$module_path/src/${module}_compat.c" ]; then
        echo "  ✅ Compatibility layer found"
        score=$((score + 1))
    else
        echo "  ❌ Compatibility layer missing"
    fi
    
    echo "  Score: $score/4"
    echo ""
    
    return $score
}

# Main validation
main() {
    log_info "Starting CUDA ecosystem validation..."
    log_info "Total modules to validate: ${#CUDA_MODULES[@]}"
    
    local total_score=0
    local max_score=$((${#CUDA_MODULES[@]} * 4))
    local perfect_count=0
    local good_count=0
    local partial_count=0
    local missing_count=0
    
    for module in "${CUDA_MODULES[@]}"; do
        local module_score
        module_score=$(validate_module "$module")
        total_score=$((total_score + module_score))
        
        case $module_score in
            4) perfect_count=$((perfect_count + 1)) ;;
            3) good_count=$((good_count + 1)) ;;
            2|1) partial_count=$((partial_count + 1)) ;;
            0) missing_count=$((missing_count + 1)) ;;
        esac
    done
    
    local percentage=$(( (total_score * 100) / max_score ))
    
    echo "========================================="
    echo "CUDA ECOSYSTEM VALIDATION RESULTS"
    echo "========================================="
    echo "Total Modules: ${#CUDA_MODULES[@]}"
    echo "Overall Score: $total_score/$max_score ($percentage%)"
    echo ""
    echo "Module Status:"
    echo "  Perfect (4/4): $perfect_count modules"
    echo "  Good (3/4):    $good_count modules"
    echo "  Partial (1-2/4): $partial_count modules"
    echo "  Missing (0/4): $missing_count modules"
    echo ""
    
    if [ $percentage -ge 80 ]; then
        log_success "ECOSYSTEM STATUS: EXCELLENT"
    elif [ $percentage -ge 60 ]; then
        log_success "ECOSYSTEM STATUS: GOOD"
    else
        log_warning "ECOSYSTEM STATUS: NEEDS WORK"
    fi
    
    echo "========================================="
}

main "$@"
