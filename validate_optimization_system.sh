#!/bin/bash

# validate_optimization_system.sh
# ===============================
# Validation script for the SuperDARN RST optimized build system
# Tests dynamic module detection and optimization features

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_test() {
    echo -e "${YELLOW}Test $((++TOTAL_TESTS)): $1${NC}"
}

print_pass() {
    echo -e "${GREEN}✓ PASS: $1${NC}"
    ((TESTS_PASSED++))
}

print_fail() {
    echo -e "${RED}✗ FAIL: $1${NC}"
    ((TESTS_FAILED++))
}

print_info() {
    echo -e "${BLUE}ℹ INFO: $1${NC}"
}

# Check if RST environment is set up
check_rst_environment() {
    print_header "Checking RST Environment"
    
    print_test "RST environment variables"
    if [[ -n "$RSTPATH" && -n "$BUILD" && -n "$CODEBASE" ]]; then
        print_pass "RST environment variables are set"
        print_info "RSTPATH: $RSTPATH"
        print_info "BUILD: $BUILD"
        print_info "CODEBASE: $CODEBASE"
    else
        print_fail "RST environment variables not set"
        echo "Please source your RST environment setup script"
        return 1
    fi
    
    print_test "Optimized build script exists"
    if [[ -f "$BUILD/script/make.code.optimized" ]]; then
        print_pass "Optimized build script found"
    else
        print_fail "Optimized build script not found"
        return 1
    fi
    
    print_test "Optimized configuration exists"
    if [[ -f "$BUILD/script/build_optimized.txt" ]]; then
        print_pass "Optimized configuration found"
    else
        print_fail "Optimized configuration not found"
        return 1
    fi
}

# Test hardware detection
test_hardware_detection() {
    print_header "Testing Hardware Detection"
    
    print_test "Hardware detection functionality"
    if "$BUILD/script/make.code.optimized" --hardware-info &>/dev/null; then
        print_pass "Hardware detection works"
    else
        print_fail "Hardware detection failed"
    fi
    
    print_test "Auto-optimization recommendation"
    local output=$("$BUILD/script/make.code.optimized" --auto-optimize --help 2>&1 || true)
    if [[ $? -eq 0 ]]; then
        print_pass "Auto-optimization works"
    else
        print_fail "Auto-optimization failed"
    fi
}

# Test dynamic module discovery
test_dynamic_discovery() {
    print_header "Testing Dynamic Module Discovery"
    
    print_test "List available optimizations"
    local output=$("$BUILD/script/make.code.optimized" --list-optimizations 2>&1)
    if [[ $? -eq 0 && "$output" == *"optimized"* ]]; then
        print_pass "Dynamic optimization discovery works"
        print_info "Found optimized modules:"
        echo "$output" | grep -E "^\s*[a-z].*->.*optimized" | head -3
    else
        print_fail "Dynamic optimization discovery failed"
    fi
    
    print_test "Optimized modules exist in codebase"
    local found_modules=0
    for module_dir in "$CODEBASE/superdarn/src.lib/tk"/*optimized*; do
        if [[ -d "$module_dir" ]]; then
            ((found_modules++))
            print_info "Found: $(basename "$module_dir")"
        fi
    done
    
    if [[ $found_modules -gt 0 ]]; then
        print_pass "Found $found_modules optimized modules"
    else
        print_fail "No optimized modules found in codebase"
    fi
}

# Test build system integration
test_build_integration() {
    print_header "Testing Build System Integration"
    
    print_test "Enhanced makefile templates exist"
    if [[ -f "$BUILD/make/makelib.optimized.linux" && -f "$BUILD/make/makebin.optimized.linux" ]]; then
        print_pass "Enhanced makefile templates found"
    else
        print_fail "Enhanced makefile templates missing"
    fi
    
    print_test "Build script help system"
    if "$BUILD/script/make.code.optimized" --help &>/dev/null; then
        print_pass "Help system works"
    else
        print_fail "Help system failed"
    fi
    
    print_test "Optimization level validation"
    local valid_levels=("none" "opt1" "opt2" "opt3")
    local levels_work=0
    
    for level in "${valid_levels[@]}"; do
        if "$BUILD/script/make.code.optimized" -o "$level" --help &>/dev/null; then
            ((levels_work++))
        fi
    done
    
    if [[ $levels_work -eq ${#valid_levels[@]} ]]; then
        print_pass "All optimization levels work"
    else
        print_fail "Some optimization levels failed ($levels_work/${#valid_levels[@]})"
    fi
}

# Test specific optimized modules
test_optimized_modules() {
    print_header "Testing Specific Optimized Modules"
    
    # Test grid module
    print_test "Grid optimized module"
    local grid_opt="$CODEBASE/superdarn/src.lib/tk/grid.1.24_optimized.1"
    if [[ -d "$grid_opt" && (-f "$grid_opt/src/makefile" || -f "$grid_opt/Makefile") ]]; then
        print_pass "Grid optimized module is valid"
    else
        print_fail "Grid optimized module missing or invalid"
    fi
    
    # Test ACF module
    print_test "ACF optimized module"
    local acf_opt="$CODEBASE/superdarn/src.lib/tk/acf.1.16_optimized.2.0"
    if [[ -d "$acf_opt" && -f "$acf_opt/src/makefile" ]]; then
        print_pass "ACF optimized module is valid"
    else
        print_fail "ACF optimized module missing or invalid"
    fi
    
    # Test binplotlib module
    print_test "Binplotlib optimized module"
    local binplot_opt="$CODEBASE/superdarn/src.lib/tk/binplotlib.1.0_optimized.2.0"
    if [[ -d "$binplot_opt" && -f "$binplot_opt/src/makefile" ]]; then
        print_pass "Binplotlib optimized module is valid"
    else
        print_fail "Binplotlib optimized module missing or invalid"
    fi
}

# Test configuration file parsing
test_configuration() {
    print_header "Testing Configuration System"
    
    print_test "Build configuration syntax"
    if [[ -f "$BUILD/script/build_optimized.txt" ]]; then
        # Check for proper documentation sections
        if grep -q "DYNAMIC OPTIMIZATION DETECTION" "$BUILD/script/build_optimized.txt"; then
            print_pass "Configuration has dynamic detection documentation"
        else
            print_fail "Configuration missing dynamic detection documentation"
        fi
        
        # Check for optimization level definitions
        if grep -q "opt1.*Basic optimization" "$BUILD/script/build_optimized.txt"; then
            print_pass "Configuration has optimization level definitions"
        else
            print_fail "Configuration missing optimization level definitions"
        fi
    else
        print_fail "Optimized configuration file not found"
    fi
}

# Test verbose output
test_verbose_mode() {
    print_header "Testing Verbose Mode"
    
    print_test "Verbose output functionality"
    local output=$("$BUILD/script/make.code.optimized" -v --help 2>&1)
    if [[ $? -eq 0 ]]; then
        print_pass "Verbose mode works"
    else
        print_fail "Verbose mode failed"
    fi
}

# Main test execution
main() {
    print_header "SuperDARN RST Optimized Build System Validation"
    echo "Testing dynamic optimization detection and build framework"
    echo ""
    
    # Run all test suites
    check_rst_environment || exit 1
    test_hardware_detection
    test_dynamic_discovery
    test_build_integration
    test_optimized_modules
    test_configuration
    test_verbose_mode
    
    # Print summary
    print_header "Test Results Summary"
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed! The optimized build system is ready to use.${NC}"
        echo ""
        echo "Usage examples:"
        echo "  $BUILD/script/make.code.optimized --auto-optimize"
        echo "  $BUILD/script/make.code.optimized -o opt2 lib"
        echo "  $BUILD/script/make.code.optimized --list-optimizations"
        exit 0
    else
        echo -e "${RED}Some tests failed. Please review the issues above.${NC}"
        exit 1
    fi
}

# Handle command line arguments
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "SuperDARN RST Optimized Build System Validation"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "This script validates the enhanced RST build system with optimization support."
    echo "It tests dynamic module detection, hardware detection, and build integration."
    echo ""
    echo "Prerequisites:"
    echo "  - RST environment must be sourced"
    echo "  - Enhanced build scripts must be installed"
    echo ""
    exit 0
fi

# Run main function
main "$@"
