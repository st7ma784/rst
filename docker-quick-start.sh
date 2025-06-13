#!/bin/bash

# SuperDARN RST Optimized Docker Quick Start
# ==========================================
# Helper script for using the optimized Docker environment

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ INFO: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
}

# Show help
show_help() {
    cat << EOF
SuperDARN RST Optimized Docker Quick Start

USAGE:
    $0 [command] [options]

COMMANDS:
    build           Build all optimized Docker images
    dev             Start development environment (both builds)
    optimized       Start optimized RST environment
    standard        Start standard RST environment  
    performance     Run automated performance comparison
    benchmark       Run intensive benchmark tests
    ci              Run continuous integration tests
    validate        Validate optimization system
    clean           Clean up Docker resources
    status          Show container status
    logs            Show container logs
    help            Show this help

EXAMPLES:
    $0 build                    # Build all images
    $0 dev                      # Start development environment
    $0 optimized               # Start optimized environment
    $0 performance             # Run performance comparison
    $0 validate                # Validate optimization system
    $0 clean                   # Clean up containers and images

ADVANCED OPTIONS:
    --no-cache                 # Build without Docker cache
    --verbose                  # Enable verbose output
    --follow-logs              # Follow container logs

EOF
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    if [ ! -f "docker-compose.optimized.yml" ]; then
        print_error "docker-compose.optimized.yml not found in current directory"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    print_header "Building SuperDARN RST Optimized Docker Images"
    
    local build_args=""
    if [[ "$*" == *"--no-cache"* ]]; then
        build_args="--no-cache"
        print_info "Building without cache..."
    fi
    
    print_info "Building all stages of the optimized Docker environment..."
    docker-compose -f docker-compose.optimized.yml build $build_args
    
    print_success "Docker images built successfully"
    
    # Show built images
    print_info "Available images:"
    docker images | grep rst
}

# Start development environment
start_dev() {
    print_header "Starting SuperDARN RST Development Environment"
    print_info "This environment includes both standard and optimized builds"
    print_info "Use 'switch-to-standard' and 'switch-to-optimized' to switch between builds"
    
    docker-compose -f docker-compose.optimized.yml run --rm superdarn-dev
}

# Start optimized environment
start_optimized() {
    print_header "Starting SuperDARN RST Optimized Environment"
    print_info "This environment uses hardware-optimized RST build"
    
    docker-compose -f docker-compose.optimized.yml run --rm superdarn-optimized
}

# Start standard environment
start_standard() {
    print_header "Starting SuperDARN RST Standard Environment"
    print_info "This environment uses standard RST build for comparison"
    
    docker-compose -f docker-compose.optimized.yml run --rm superdarn-standard
}

# Run performance comparison
run_performance() {
    print_header "Running SuperDARN RST Performance Comparison"
    print_info "This will compare standard vs optimized build performance"
    print_info "Results will be saved to ./test-results/"
    
    # Ensure results directory exists
    mkdir -p ./test-results
    
    print_info "Starting automated performance testing..."
    docker-compose -f docker-compose.optimized.yml up --abort-on-container-exit superdarn-performance
    
    print_success "Performance comparison completed"
    print_info "Results available in ./test-results/"
    
    # Show results summary
    if [ -f "./test-results/optimization_comparison_dashboard.html" ]; then
        print_success "Performance dashboard: ./test-results/optimization_comparison_dashboard.html"
    fi
}

# Run benchmark tests
run_benchmark() {
    print_header "Running SuperDARN RST Benchmark Tests"
    print_info "This will run intensive performance benchmarks"
    
    mkdir -p ./test-results
    
    docker-compose -f docker-compose.optimized.yml up --abort-on-container-exit superdarn-benchmark
    
    print_success "Benchmark testing completed"
    
    if [ -f "./test-results/benchmark_report.html" ]; then
        print_success "Benchmark report: ./test-results/benchmark_report.html"
    fi
}

# Run CI tests
run_ci() {
    print_header "Running SuperDARN RST CI Tests"
    print_info "Running validation tests suitable for CI/CD"
    
    docker-compose -f docker-compose.optimized.yml run --rm superdarn-ci
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "All CI tests passed"
    else
        print_error "CI tests failed with exit code $exit_code"
        exit $exit_code
    fi
}

# Validate optimization system
validate_system() {
    print_header "Validating SuperDARN RST Optimization System"
    print_info "Running comprehensive validation of the optimization framework"
    
    docker-compose -f docker-compose.optimized.yml run --rm superdarn-optimized check-optimization
    
    print_success "Optimization system validation completed"
}

# Clean up Docker resources
clean_up() {
    print_header "Cleaning Up SuperDARN RST Docker Resources"
    
    print_info "Stopping and removing containers..."
    docker-compose -f docker-compose.optimized.yml down
    
    print_info "Removing unused images and volumes..."
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Show container status
show_status() {
    print_header "SuperDARN RST Container Status"
    
    print_info "Running containers:"
    docker-compose -f docker-compose.optimized.yml ps
    
    print_info "Docker images:"
    docker images | grep -E "(rst|superdarn)" || echo "No RST images found"
    
    print_info "Volume usage:"
    docker system df
}

# Show container logs
show_logs() {
    print_header "SuperDARN RST Container Logs"
    
    local service="$1"
    if [ -z "$service" ]; then
        print_info "Available services:"
        docker-compose -f docker-compose.optimized.yml config --services
        echo ""
        read -p "Enter service name (or 'all' for all services): " service
    fi
    
    if [ "$service" = "all" ]; then
        docker-compose -f docker-compose.optimized.yml logs
    else
        docker-compose -f docker-compose.optimized.yml logs "$service"
    fi
}

# Main script logic
main() {
    local command="$1"
    shift || true
    
    # Check prerequisites for most commands
    if [[ "$command" != "help" && "$command" != "--help" && "$command" != "-h" ]]; then
        check_prerequisites
    fi
    
    case "$command" in
        "build")
            build_images "$@"
            ;;
        "dev"|"development")
            start_dev
            ;;
        "optimized"|"opt")
            start_optimized
            ;;
        "standard"|"std")
            start_standard
            ;;
        "performance"|"perf")
            run_performance
            ;;
        "benchmark"|"bench")
            run_benchmark
            ;;
        "ci")
            run_ci
            ;;
        "validate"|"val")
            validate_system
            ;;
        "clean"|"cleanup")
            clean_up
            ;;
        "status"|"ps")
            show_status
            ;;
        "logs")
            show_logs "$1"
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'print_warning "Script interrupted"; exit 130' INT

# Run main function
main "$@"
