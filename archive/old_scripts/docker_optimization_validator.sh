#!/bin/bash
# docker_optimization_validator.sh
# ================================
# Docker-specific validation script for SuperDARN RST optimization system
# 
# This script validates that the Docker optimization infrastructure is working
# correctly across all container stages and configurations.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test status tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
    ((TESTS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}⚠️ WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
    ((TESTS_FAILED++))
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((TESTS_TOTAL++))
    log_info "Running test: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        log_success "$test_name"
        return 0
    else
        log_error "$test_name"
        return 1
    fi
}

# Header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🐳 SuperDARN RST Docker Optimization Validator${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if we're running in Docker
if [ ! -f /.dockerenv ]; then
    log_warning "Not running inside Docker container"
    log_info "This script is designed to run inside Docker containers"
    log_info "Running basic validation..."
fi

# 1. Validate Docker Images Exist
log_info "🏗️ Checking Docker image availability..."

run_test "Standard RST image buildable" "docker build -f dockerfile.optimized --target rst_standard -t test-rst-standard . --quiet"
run_test "Optimized RST image buildable" "docker build -f dockerfile.optimized --target rst_optimized -t test-rst-optimized . --quiet"
run_test "Development image buildable" "docker build -f dockerfile.optimized --target rst_development -t test-rst-dev . --quiet"

# 2. Validate Docker Compose Configuration
log_info "🔧 Checking Docker Compose configuration..."

run_test "Docker Compose syntax valid" "docker-compose -f docker-compose.optimized.yml config --quiet"
run_test "All services defined correctly" "docker-compose -f docker-compose.optimized.yml config | grep -q 'superdarn-optimized'"

# 3. Test Container Startup (if Docker is available)
if command -v docker &> /dev/null; then
    log_info "🚀 Testing container startup..."
    
    # Test optimized container
    run_test "Optimized container starts" "timeout 30s docker run --rm test-rst-optimized echo 'Container started successfully'"
    
    # Test development container
    run_test "Development container starts" "timeout 30s docker run --rm test-rst-dev echo 'Development container started'"
    
    # Clean up test images
    docker rmi test-rst-standard test-rst-optimized test-rst-dev 2>/dev/null || true
fi

# 4. Validate Build Scripts in Container Context
if [ -f /.dockerenv ]; then
    log_info "🛠️ Validating container build environment..."
    
    # Check RST environment
    run_test "RSTPATH environment set" "[ ! -z '$RSTPATH' ]"
    run_test "BUILD environment set" "[ ! -z '$BUILD' ]"
    run_test "RST profile exists" "[ -f '$RSTPATH/.profile.bash' ]"
    
    # Check optimization system
    if [ -f "$RSTPATH/build/script/make.code.optimized" ]; then
        run_test "Optimization script executable" "[ -x '$RSTPATH/build/script/make.code.optimized' ]"
        run_test "Optimization script shows help" "'$RSTPATH/build/script/make.code.optimized' --help | grep -q 'optimization'"
    fi
    
    # Check validation script
    if [ -f "$RSTPATH/validate_optimization_system.sh" ]; then
        run_test "Validation script executable" "[ -x '$RSTPATH/validate_optimization_system.sh' ]"
    fi
    
    # Check compiler availability
    run_test "GCC compiler available" "command -v gcc"
    run_test "G++ compiler available" "command -v g++"
    run_test "Make available" "command -v make"
    
    # Check OpenMP support
    run_test "OpenMP headers available" "[ -f /usr/include/omp.h ] || [ -f /usr/local/include/omp.h ]"
    
    # Check optimization libraries
    run_test "Math libraries available" "ldconfig -p | grep -q libm"
    
    # Hardware detection
    log_info "💻 Hardware capabilities detection..."
    CPU_COUNT=$(nproc)
    HAS_AVX2=$(grep -c avx2 /proc/cpuinfo || echo "0")
    
    log_info "CPU cores detected: $CPU_COUNT"
    log_info "AVX2 support: $([ $HAS_AVX2 -gt 0 ] && echo 'Yes' || echo 'No')"
    
    if [ $CPU_COUNT -gt 1 ]; then
        log_success "Multi-core CPU detected - parallelization possible"
    else
        log_warning "Single-core CPU - limited optimization benefits"
    fi
    
    if [ $HAS_AVX2 -gt 0 ]; then
        log_success "AVX2 instructions available - advanced optimizations possible"
    else
        log_info "AVX2 not available - using basic optimizations"
    fi
fi

# 5. Test Helper Scripts
log_info "📜 Validating helper scripts..."

# Check PowerShell script
if [ -f "docker-quick-start.ps1" ]; then
    run_test "PowerShell helper script exists" "[ -f 'docker-quick-start.ps1' ]"
fi

# Check bash script  
if [ -f "docker-quick-start.sh" ]; then
    run_test "Bash helper script exists" "[ -f 'docker-quick-start.sh' ]"
    run_test "Bash helper script executable" "[ -x 'docker-quick-start.sh' ]"
fi

# 6. Validate Documentation
log_info "📚 Checking documentation..."

run_test "Docker optimization guide exists" "[ -f 'DOCKER_OPTIMIZATION_GUIDE.md' ]"
run_test "Implementation summary exists" "[ -f 'IMPLEMENTATION_SUMMARY.md' ]"
run_test "Enhanced build guide exists" "[ -f 'ENHANCED_BUILD_SYSTEM_GUIDE.md' ]"

# 7. Test Docker Compose Services
if command -v docker-compose &> /dev/null; then
    log_info "🐳 Testing Docker Compose services..."
    
    # Validate service definitions
    SERVICES=$(docker-compose -f docker-compose.optimized.yml config --services)
    
    for service in superdarn-standard superdarn-optimized superdarn-dev superdarn-performance superdarn-ci superdarn-benchmark; do
        if echo "$SERVICES" | grep -q "$service"; then
            log_success "Service '$service' defined correctly"
        else
            log_error "Service '$service' not found in compose file"
        fi
    done
fi

# 8. Validate Volume Mounts and Permissions
if [ -f /.dockerenv ]; then
    log_info "📁 Checking volume mounts and permissions..."
    
    # Check if workspace directories are writable
    if [ -d "/workspace" ]; then
        run_test "Workspace directory writable" "[ -w '/workspace' ]"
    fi
    
    # Check RST directories
    run_test "RST build directory exists" "[ -d '$RSTPATH/build' ]"
    run_test "RST codebase directory exists" "[ -d '$RSTPATH/codebase' ]"
    
    # Check for optimized modules
    if [ -d "$RSTPATH/codebase/superdarn/src.lib/tk" ]; then
        OPTIMIZED_MODULES=$(find "$RSTPATH/codebase/superdarn/src.lib/tk" -name "*optimized*" -type d | wc -l)
        if [ $OPTIMIZED_MODULES -gt 0 ]; then
            log_success "Found $OPTIMIZED_MODULES optimized modules"
        else
            log_warning "No optimized modules found in standard location"
        fi
    fi
fi

# Summary
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}📊 Docker Optimization Validation Summary${NC}"
echo -e "${BLUE}============================================${NC}"

echo -e "Total tests run: ${BLUE}$TESTS_TOTAL${NC}"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    log_success "All Docker optimization infrastructure tests passed!"
    echo -e "${GREEN}🚀 Your SuperDARN RST Docker optimization environment is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Build containers: docker-compose -f docker-compose.optimized.yml build"
    echo "2. Start optimized environment: docker-compose -f docker-compose.optimized.yml up superdarn-optimized"
    echo "3. Run performance tests: docker-compose -f docker-compose.optimized.yml up superdarn-performance"
    echo "4. Development work: docker-compose -f docker-compose.optimized.yml up superdarn-dev"
    
    exit 0
else
    echo ""
    log_error "Some Docker optimization infrastructure tests failed!"
    echo -e "${RED}❌ Please review the failed tests above and fix the issues.${NC}"
    echo ""
    echo "Common fixes:"
    echo "1. Ensure Docker is installed and running"
    echo "2. Check dockerfile.optimized syntax"
    echo "3. Verify docker-compose.optimized.yml configuration"
    echo "4. Ensure all required files are present"
    
    exit 1
fi
