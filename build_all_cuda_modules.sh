#!/bin/bash

# Build All CUDA Modules Script
# Automated building for CI/CD and local development

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"
BUILD_TYPE="${1:-compat}"

# All CUDA-enabled modules
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

# Build single module
build_module() {
    local module="$1"
    local build_type="$2"
    local module_path="$CODEBASE_DIR/$module"
    
    if [ ! -d "$module_path" ]; then
        log_warning "Module directory not found: $module"
        return 1
    fi
    
    if [ ! -f "$module_path/makefile.cuda" ]; then
        log_warning "No CUDA makefile found for $module"
        return 1
    fi
    
    log_info "Building $module ($build_type)..."
    
    cd "$module_path"
    
    # Clean previous builds
    make -f makefile.cuda clean > /dev/null 2>&1 || true
    
    # Build with specified type
    if make -f makefile.cuda "$build_type" > build.log 2>&1; then
        log_success "$module built successfully"
        cd "$SCRIPT_DIR"
        return 0
    else
        log_error "$module build failed"
        cat build.log
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# Main build process
main() {
    log_info "Building all CUDA modules with type: $BUILD_TYPE"
    log_info "Total modules to build: ${#CUDA_MODULES[@]}"
    
    # Check CUDA availability
    if command -v nvcc &> /dev/null; then
        log_success "CUDA toolkit detected"
        nvcc --version | head -1
    else
        log_warning "CUDA toolkit not found - building CPU/compatibility only"
        if [ "$BUILD_TYPE" = "cuda" ]; then
            log_error "Cannot build CUDA without CUDA toolkit"
            exit 1
        fi
    fi
    
    local success_count=0
    local fail_count=0
    local failed_modules=()
    
    # Build each module
    for module in "${CUDA_MODULES[@]}"; do
        if build_module "$module" "$BUILD_TYPE"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
            failed_modules+=("$module")
        fi
    done
    
    # Summary
    echo ""
    echo "========================================="
    echo "BUILD SUMMARY"
    echo "========================================="
    echo "Build Type: $BUILD_TYPE"
    echo "Total Modules: ${#CUDA_MODULES[@]}"
    echo "Successful: $success_count"
    echo "Failed: $fail_count"
    echo "Success Rate: $(( (success_count * 100) / ${#CUDA_MODULES[@]} ))%"
    
    if [ $fail_count -gt 0 ]; then
        echo ""
        echo "Failed Modules:"
        for module in "${failed_modules[@]}"; do
            echo "  - $module"
        done
        echo "========================================="
        exit 1
    else
        log_success "All modules built successfully!"
        echo "========================================="
    fi
}

# Validate arguments
case "$BUILD_TYPE" in
    cpu|cuda|compat)
        ;;
    *)
        echo "Usage: $0 [cpu|cuda|compat]"
        echo "  cpu    - Build CPU-only versions"
        echo "  cuda   - Build CUDA-accelerated versions"
        echo "  compat - Build compatibility versions (default)"
        exit 1
        ;;
esac

main "$@"
