#!/bin/bash
# deploy_docker_optimization.sh
# =============================
# Complete deployment script for SuperDARN RST Docker optimization infrastructure
# 
# This script sets up the complete Docker-based optimization environment,
# validates the setup, and provides guidance for using the system.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/deployment-logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOYMENT_LOG="$LOG_DIR/deployment_${TIMESTAMP}.log"

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
    local message="$1"
    echo -e "${BLUE}ℹ INFO:${NC} $message" | tee -a "$DEPLOYMENT_LOG"
}

log_success() {
    local message="$1"
    echo -e "${GREEN}✅ SUCCESS:${NC} $message" | tee -a "$DEPLOYMENT_LOG"
}

log_warning() {
    local message="$1"
    echo -e "${YELLOW}⚠️ WARNING:${NC} $message" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    local message="$1"
    echo -e "${RED}❌ ERROR:${NC} $message" | tee -a "$DEPLOYMENT_LOG"
}

log_step() {
    local message="$1"
    echo -e "${PURPLE}🚀 STEP:${NC} $message" | tee -a "$DEPLOYMENT_LOG"
}

log_deploy() {
    local message="$1"
    echo -e "${CYAN}📦 DEPLOY:${NC} $message" | tee -a "$DEPLOYMENT_LOG"
}

# Setup logging
setup_logging() {
    mkdir -p "$LOG_DIR"
    touch "$DEPLOYMENT_LOG"
    log_info "Deployment logging started: $DEPLOYMENT_LOG"
}

# Validation functions
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        echo "Installation guide: https://docs.docker.com/get-docker/"
        exit 1
    fi
    log_success "Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        echo "Installation guide: https://docs.docker.com/compose/install/"
        exit 1
    fi
    log_success "Docker Compose is available"
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    log_success "Docker daemon is running"
    
    # Check disk space (need at least 4GB free)
    available_space=$(df "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    required_space=4194304  # 4GB in KB
    if [ "$available_space" -lt "$required_space" ]; then
        log_warning "Low disk space detected. Need at least 4GB free for Docker images."
        log_warning "Available: $(echo "scale=1; $available_space/1024/1024" | bc)GB"
    else
        log_success "Sufficient disk space available"
    fi
}

check_required_files() {
    log_step "Checking required files..."
    
    local required_files=(
        "dockerfile.optimized"
        "docker-compose.optimized.yml"
        "build/script/make.code.optimized"
        "build/script/build_optimized.txt"
        "validate_optimization_system.sh"
        "docker_optimization_validator.sh"
        "docker_performance_tester.sh"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "Found: $file"
        else
            log_error "Missing: $file"
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        log_error "Missing required files. Please ensure all optimization framework files are present."
        echo "Missing files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
    
    log_success "All required files are present"
}

# Deployment functions
build_docker_images() {
    log_step "Building Docker images..."
    
    # Build with progress output
    log_deploy "Building base RST environment..."
    docker-compose -f docker-compose.optimized.yml build --no-cache rst_base 2>&1 | tee -a "$DEPLOYMENT_LOG"
    
    log_deploy "Building standard RST image..."
    docker-compose -f docker-compose.optimized.yml build superdarn-standard 2>&1 | tee -a "$DEPLOYMENT_LOG"
    
    log_deploy "Building optimized RST image..."
    docker-compose -f docker-compose.optimized.yml build superdarn-optimized 2>&1 | tee -a "$DEPLOYMENT_LOG"
    
    log_deploy "Building development environment..."
    docker-compose -f docker-compose.optimized.yml build superdarn-dev 2>&1 | tee -a "$DEPLOYMENT_LOG"
    
    log_success "All Docker images built successfully"
}

validate_deployment() {
    log_step "Validating deployment..."
    
    # Run the validation script
    if [ -f "docker_optimization_validator.sh" ]; then
        chmod +x docker_optimization_validator.sh
        log_info "Running Docker optimization validator..."
        ./docker_optimization_validator.sh 2>&1 | tee -a "$DEPLOYMENT_LOG"
    else
        log_warning "Docker optimization validator not found, skipping validation"
    fi
    
    # Test container startup
    log_info "Testing container startup..."
    
    # Test optimized container
    if docker run --rm --name test-startup superdarn-rst-optimized_rst_optimized echo "Optimized container test successful" &> /dev/null; then
        log_success "Optimized container starts correctly"
    else
        log_error "Optimized container failed to start"
    fi
    
    # Test development container
    if docker run --rm --name test-dev-startup superdarn-rst-optimized_rst_development echo "Development container test successful" &> /dev/null; then
        log_success "Development container starts correctly"
    else
        log_error "Development container failed to start"
    fi
}

setup_helper_scripts() {
    log_step "Setting up helper scripts..."
    
    # Make scripts executable
    local scripts=(
        "docker_optimization_validator.sh"
        "docker_performance_tester.sh"
        "validate_optimization_system.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            chmod +x "$script"
            log_success "Made executable: $script"
        fi
    done
    
    # Check PowerShell script on Windows
    if [ -f "docker-quick-start.ps1" ]; then
        log_success "PowerShell helper script available: docker-quick-start.ps1"
    fi
    
    # Check bash helper script
    if [ -f "docker-quick-start.sh" ]; then
        chmod +x "docker-quick-start.sh"
        log_success "Bash helper script available: docker-quick-start.sh"
    fi
}

generate_deployment_summary() {
    log_step "Generating deployment summary..."
    
    local summary_file="$LOG_DIR/deployment_summary_${TIMESTAMP}.md"
    
    cat > "$summary_file" << EOF
# SuperDARN RST Docker Optimization Deployment Summary

**Deployment Date:** $(date)  
**Deployment ID:** $TIMESTAMP  
**Location:** $SCRIPT_DIR

## Deployment Status

✅ **SUCCESS** - SuperDARN RST Docker optimization infrastructure deployed successfully

## Available Docker Images

- **superdarn-rst-optimized_rst_base**: Base environment with dependencies
- **superdarn-rst-optimized_rst_standard**: Standard RST build (baseline)
- **superdarn-rst-optimized_rst_optimized**: Hardware-optimized RST build
- **superdarn-rst-optimized_rst_development**: Development environment with both builds

## Available Services (Docker Compose)

- **superdarn-standard**: Standard RST environment
- **superdarn-optimized**: Optimized RST environment  
- **superdarn-dev**: Development environment
- **superdarn-performance**: Automated performance testing
- **superdarn-ci**: Continuous integration testing
- **superdarn-benchmark**: Intensive benchmark testing

## Quick Start Commands

### Start Optimized Environment
\`\`\`bash
docker-compose -f docker-compose.optimized.yml up superdarn-optimized
\`\`\`

### Start Development Environment
\`\`\`bash
docker-compose -f docker-compose.optimized.yml up superdarn-dev
\`\`\`

### Run Performance Tests
\`\`\`bash
./docker_performance_tester.sh
\`\`\`

### Validate Installation
\`\`\`bash
./docker_optimization_validator.sh
\`\`\`

## Available Helper Scripts

- **docker_optimization_validator.sh**: Validate Docker optimization infrastructure
- **docker_performance_tester.sh**: Run comprehensive performance comparisons
- **docker-quick-start.sh**: Quick start helper (Unix/Linux)
- **docker-quick-start.ps1**: Quick start helper (Windows PowerShell)

## Documentation

- **DOCKER_OPTIMIZATION_GUIDE.md**: Complete Docker usage guide
- **ENHANCED_BUILD_SYSTEM_GUIDE.md**: Build system documentation
- **IMPLEMENTATION_SUMMARY.md**: Project overview and completion status

## System Information

- **Host OS**: $(uname -s)
- **Architecture**: $(uname -m)
- **Docker Version**: $(docker --version)
- **Docker Compose Version**: $(docker-compose --version)

## Next Steps

1. **Test the installation**: Run \`./docker_optimization_validator.sh\`
2. **Start using**: Launch \`docker-compose -f docker-compose.optimized.yml up superdarn-optimized\`
3. **Run benchmarks**: Execute \`./docker_performance_tester.sh\`
4. **Read documentation**: Review the guides in the main directory

## Support

For issues or questions:
1. Check the validation output from \`docker_optimization_validator.sh\`
2. Review the deployment log: \`$DEPLOYMENT_LOG\`
3. Consult the documentation guides
4. Check Docker and system requirements

---
*Deployment completed by SuperDARN RST Docker Optimization Deployment Script*
EOF

    log_success "Deployment summary generated: $summary_file"
    export SUMMARY_FILE="$summary_file"
}

display_completion_message() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}🎉 SuperDARN RST Docker Optimization Deployed!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "${CYAN}📊 Deployment Summary:${NC} $SUMMARY_FILE"
    echo -e "${CYAN}📝 Deployment Log:${NC} $DEPLOYMENT_LOG"
    echo ""
    echo -e "${BLUE}🚀 Quick Start:${NC}"
    echo "  1. Validate installation:  ./docker_optimization_validator.sh"
    echo "  2. Start optimized RST:    docker-compose -f docker-compose.optimized.yml up superdarn-optimized"
    echo "  3. Run performance tests:  ./docker_performance_tester.sh"
    echo "  4. Development work:       docker-compose -f docker-compose.optimized.yml up superdarn-dev"
    echo ""
    echo -e "${PURPLE}📚 Documentation:${NC}"
    echo "  - DOCKER_OPTIMIZATION_GUIDE.md    (Complete Docker usage guide)"
    echo "  - ENHANCED_BUILD_SYSTEM_GUIDE.md  (Build system documentation)"
    echo "  - IMPLEMENTATION_SUMMARY.md       (Project overview)"
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo "  - Use 'docker-compose -f docker-compose.optimized.yml ps' to see running services"
    echo "  - Access containers with 'docker exec -it <container-name> bash'"
    echo "  - View logs with 'docker-compose -f docker-compose.optimized.yml logs <service-name>'"
    echo ""
    echo "Happy optimizing! 🔬⚡"
}

# Main deployment function
main() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}🐳 SuperDARN RST Docker Optimization Deployer${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    
    setup_logging
    
    log_info "Starting SuperDARN RST Docker optimization deployment..."
    echo ""
    
    # Validation phase
    check_prerequisites
    check_required_files
    echo ""
    
    # Deployment phase
    build_docker_images
    echo ""
    
    # Setup phase
    setup_helper_scripts
    echo ""
    
    # Validation phase
    validate_deployment
    echo ""
    
    # Completion
    generate_deployment_summary
    display_completion_message
    
    log_success "SuperDARN RST Docker optimization deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "SuperDARN RST Docker Optimization Deployer"
        echo ""
        echo "USAGE:"
        echo "  $0                    # Run complete deployment"
        echo "  $0 --help            # Show this help"
        echo "  $0 --validate-only   # Only validate, don't build"
        echo "  $0 --build-only      # Only build images"
        echo "  $0 --quick           # Skip validation, just build and test"
        echo ""
        echo "This script deploys the complete SuperDARN RST Docker optimization"
        echo "infrastructure, including all images, services, and helper tools."
        exit 0
        ;;
    --validate-only)
        setup_logging
        check_prerequisites
        check_required_files
        echo ""
        log_success "Validation completed successfully!"
        ;;
    --build-only)
        setup_logging
        check_prerequisites
        build_docker_images
        echo ""
        log_success "Build completed successfully!"
        ;;
    --quick)
        setup_logging
        check_prerequisites
        build_docker_images
        setup_helper_scripts
        generate_deployment_summary
        display_completion_message
        ;;
    "")
        # Run full deployment
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
