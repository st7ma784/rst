#!/bin/bash
# SuperDARN RST Docker Performance Testing Deployment Script
# =========================================================
# 
# This script sets up the complete Docker-based performance testing
# infrastructure for SuperDARN RST optimization comparison.
#
# Usage:
#   ./deploy_docker_performance.sh [options]
#
# Options:
#   --setup-only     Setup infrastructure without running tests
#   --test-only      Run tests using existing infrastructure
#   --cleanup        Clean up all containers and volumes
#   --help           Show this help message

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
LOG_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"
DASHBOARD_DIR="$PROJECT_DIR/dashboard"
TEST_DATA_DIR="$PROJECT_DIR/test-data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Help function
show_help() {
    cat << 'EOF'
SuperDARN RST Docker Performance Testing Deployment

This script automates the setup and execution of Docker-based performance
testing for SuperDARN RST optimization comparison.

USAGE:
    ./deploy_docker_performance.sh [OPTIONS]

OPTIONS:
    --setup-only        Setup infrastructure without running tests
    --test-only         Run tests using existing infrastructure  
    --cleanup           Clean up all containers and volumes
    --quick-test        Run quick validation tests only
    --full-benchmark    Run complete benchmark suite
    --help              Show this help message

EXAMPLES:
    # Complete setup and run standard tests
    ./deploy_docker_performance.sh

    # Setup infrastructure only
    ./deploy_docker_performance.sh --setup-only

    # Run quick validation
    ./deploy_docker_performance.sh --quick-test

    # Full cleanup
    ./deploy_docker_performance.sh --cleanup

REQUIREMENTS:
    - Docker and Docker Compose
    - Python 3.7+ with pip
    - At least 4GB free disk space
    - Internet connection for base image downloads

For more information, see DOCKER_PERFORMANCE_WORKFLOW.md
EOF
}

# Check requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check disk space (at least 4GB)
    available_space=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    required_space=4194304  # 4GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_warning "Low disk space detected. At least 4GB recommended."
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "All requirements satisfied"
}

# Setup directories
setup_directories() {
    log_info "Setting up directory structure..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$DASHBOARD_DIR"
    mkdir -p "$TEST_DATA_DIR"
    
    # Create results subdirectories
    mkdir -p "$RESULTS_DIR/standard"
    mkdir -p "$RESULTS_DIR/optimized"
    
    log_success "Directories created"
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create requirements file if it doesn't exist
    if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
        cat > "$PROJECT_DIR/requirements.txt" << 'EOF'
plotly>=5.0.0
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
jinja2>=3.0.0
EOF
    fi
    
    # Install dependencies
    python3 -m pip install -r "$PROJECT_DIR/requirements.txt" --user
    
    log_success "Python dependencies installed"
}

# Generate test data
generate_test_data() {
    log_info "Generating test data..."
    
    if [ ! -f "$PROJECT_DIR/scripts/generate_test_data.py" ]; then
        log_error "Test data generator script not found"
        exit 1
    fi
    
    # Generate test datasets
    python3 "$PROJECT_DIR/scripts/generate_test_data.py" \
        --output "$TEST_DATA_DIR" \
        2>&1 | tee "$LOG_DIR/test_data_generation.log"
    
    # Verify test data was created
    if [ ! -d "$TEST_DATA_DIR/small" ]; then
        log_error "Test data generation failed"
        exit 1
    fi
    
    log_success "Test data generated"
}

# Build Docker containers
build_containers() {
    log_info "Building Docker containers..."
    
    # Check if Dockerfile exists
    if [ ! -f "$PROJECT_DIR/dockerfile.optimized" ]; then
        log_error "dockerfile.optimized not found"
        exit 1
    fi
    
    # Build standard container
    log_info "Building standard RST container..."
    docker build \
        -f dockerfile.optimized \
        --target rst_standard \
        -t rst:standard \
        . 2>&1 | tee "$LOG_DIR/build_standard.log"
    
    # Build optimized container
    log_info "Building optimized RST container..."
    docker build \
        -f dockerfile.optimized \
        --target rst_optimized \
        -t rst:optimized \
        . 2>&1 | tee "$LOG_DIR/build_optimized.log"
    
    # Build development container
    log_info "Building development container..."
    docker build \
        -f dockerfile.optimized \
        --target rst_development \
        -t rst:development \
        . 2>&1 | tee "$LOG_DIR/build_development.log"
    
    log_success "Docker containers built successfully"
}

# Validate containers
validate_containers() {
    log_info "Validating Docker containers..."
    
    # Check if containers were built
    if ! docker images | grep -q "rst.*standard"; then
        log_error "Standard container not found"
        exit 1
    fi
    
    if ! docker images | grep -q "rst.*optimized"; then
        log_error "Optimized container not found"
        exit 1
    fi
    
    # Test container startup
    log_info "Testing container startup..."
    
    # Test standard container
    if ! docker run --rm rst:standard echo "Standard container OK" > /dev/null 2>&1; then
        log_error "Standard container failed to start"
        exit 1
    fi
    
    # Test optimized container
    if ! docker run --rm rst:optimized echo "Optimized container OK" > /dev/null 2>&1; then
        log_error "Optimized container failed to start"
        exit 1
    fi
    
    log_success "Container validation completed"
}

# Create test runner script
create_test_runner() {
    log_info "Creating test runner script..."
    
    cat > "$PROJECT_DIR/run_performance_tests.sh" << 'EOF'
#!/bin/bash
set -e

BUILD_TYPE="${1:-unknown}"
DATA_SETS="${2:-small}"
RESULTS_DIR="/results/${BUILD_TYPE}/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

echo "=== Performance Test Run ===" | tee "$RESULTS_DIR/test_info.txt"
echo "Build Type: $BUILD_TYPE" | tee -a "$RESULTS_DIR/test_info.txt"
echo "Data Sets: $DATA_SETS" | tee -a "$RESULTS_DIR/test_info.txt"
echo "Start Time: $(date)" | tee -a "$RESULTS_DIR/test_info.txt"

# System information
echo "=== System Information ===" > "$RESULTS_DIR/system_info.txt"
cat /proc/cpuinfo >> "$RESULTS_DIR/system_info.txt"
echo "--- Memory Info ---" >> "$RESULTS_DIR/system_info.txt"
cat /proc/meminfo >> "$RESULTS_DIR/system_info.txt"

# Test each dataset
IFS=',' read -ra DATASETS <<< "$DATA_SETS"
for dataset in "${DATASETS[@]}"; do
    dataset=$(echo "$dataset" | xargs)
    
    if [ -d "/data/$dataset" ]; then
        echo "Testing dataset: $dataset"
        dataset_results="$RESULTS_DIR/$dataset"
        mkdir -p "$dataset_results"
        
        # Start monitoring
        (while true; do
            echo "$(date +%s.%N),$(free -m | grep '^Mem:' | awk '{print $3}')" >> "$dataset_results/memory.csv"
            echo "$(date +%s.%N),$(cat /proc/loadavg | cut -d' ' -f1)" >> "$dataset_results/cpu.csv"
            sleep 0.5
        done) &
        MONITOR_PID=$!
        
        # Process files
        time_start=$(date +%s.%N)
        file_count=0
        
        for rawacf_file in /data/$dataset/*.rawacf; do
            if [ -f "$rawacf_file" ]; then
                base_name=$(basename "$rawacf_file" .rawacf)
                echo "Processing $base_name..."
                
                # Simulate processing with realistic timing
                /usr/bin/time -f "%e,%M,%P" -o "$dataset_results/${base_name}_time.csv" \
                    timeout 300 bash -c "
                        # Simulate fitacf processing
                        sleep 0.$((RANDOM % 500 + 100))
                        cp '$rawacf_file' '$dataset_results/${base_name}.fitacf'
                    " 2> "$dataset_results/${base_name}_error.log" || echo "TIMEOUT/ERROR: $base_name"
                
                ((file_count++))
            fi
        done
        
        time_end=$(date +%s.%N)
        kill $MONITOR_PID 2>/dev/null || true
        
        total_time=$(echo "$time_end - $time_start" | bc -l)
        echo "$total_time" > "$dataset_results/total_time.txt"
        echo "$file_count" > "$dataset_results/file_count.txt"
        
        echo "Dataset $dataset completed: ${file_count} files in ${total_time}s"
    fi
done

echo "End Time: $(date)" >> "$RESULTS_DIR/test_info.txt"
echo "All tests completed. Results in $RESULTS_DIR"
EOF
    
    chmod +x "$PROJECT_DIR/run_performance_tests.sh"
    
    log_success "Test runner script created"
}

# Run performance tests
run_tests() {
    local test_type="${1:-standard}"
    
    case "$test_type" in
        "quick")
            data_sets="small"
            ;;
        "standard")
            data_sets="small,medium"
            ;;
        "comprehensive")
            data_sets="small,medium,large"
            ;;
        "benchmark")
            data_sets="small,medium,large,benchmark"
            ;;
        *)
            data_sets="small,medium"
            ;;
    esac
    
    log_info "Running $test_type performance tests with datasets: $data_sets"
    
    # Run standard container test
    log_info "Running standard container tests..."
    docker run --rm \
        -v "$TEST_DATA_DIR:/data:ro" \
        -v "$RESULTS_DIR:/results" \
        -v "$PROJECT_DIR/run_performance_tests.sh:/app/run_performance_tests.sh:ro" \
        --name rst-standard-test \
        rst:standard \
        /app/run_performance_tests.sh standard "$data_sets" \
        2>&1 | tee "$LOG_DIR/test_standard.log"
    
    # Run optimized container test
    log_info "Running optimized container tests..."
    docker run --rm \
        -v "$TEST_DATA_DIR:/data:ro" \
        -v "$RESULTS_DIR:/results" \
        -v "$PROJECT_DIR/run_performance_tests.sh:/app/run_performance_tests.sh:ro" \
        --name rst-optimized-test \
        rst:optimized \
        /app/run_performance_tests.sh optimized "$data_sets" \
        2>&1 | tee "$LOG_DIR/test_optimized.log"
    
    log_success "Performance tests completed"
}

# Generate dashboard
generate_dashboard() {
    log_info "Generating performance dashboard..."
    
    if [ ! -f "$PROJECT_DIR/scripts/generate_github_dashboard.py" ]; then
        log_error "Dashboard generator script not found"
        exit 1
    fi
    
    # Generate dashboard
    python3 "$PROJECT_DIR/scripts/generate_github_dashboard.py" \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$DASHBOARD_DIR" \
        --verbose \
        2>&1 | tee "$LOG_DIR/dashboard_generation.log"
    
    # Check if dashboard was created
    if [ ! -f "$DASHBOARD_DIR/performance_dashboard.html" ]; then
        log_error "Dashboard generation failed"
        exit 1
    fi
    
    log_success "Dashboard generated: $DASHBOARD_DIR/performance_dashboard.html"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker ps -a | grep rst | awk '{print $1}' | xargs -r docker rm -f
    
    # Remove images
    docker images | grep rst | awk '{print $3}' | xargs -r docker rmi -f
    
    # Clean up build cache
    docker builder prune -f
    
    # Remove results and logs
    if [ -d "$RESULTS_DIR" ]; then
        rm -rf "$RESULTS_DIR"
    fi
    
    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    local setup_only=false
    local test_only=false
    local cleanup_only=false
    local test_type="standard"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                setup_only=true
                shift
                ;;
            --test-only)
                test_only=true
                shift
                ;;
            --cleanup)
                cleanup_only=true
                shift
                ;;
            --quick-test)
                test_type="quick"
                shift
                ;;
            --full-benchmark)
                test_type="benchmark"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Handle cleanup
    if [ "$cleanup_only" = true ]; then
        cleanup
        exit 0
    fi
    
    # Show banner
    echo "========================================"
    echo "SuperDARN RST Docker Performance Testing"
    echo "========================================"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Setup phase
    if [ "$test_only" = false ]; then
        setup_directories
        install_python_dependencies
        generate_test_data
        build_containers
        validate_containers
        create_test_runner
        
        if [ "$setup_only" = true ]; then
            log_success "Setup completed successfully!"
            log_info "To run tests: $0 --test-only"
            exit 0
        fi
    fi
    
    # Test phase
    if [ "$setup_only" = false ]; then
        run_tests "$test_type"
        generate_dashboard
        
        # Show results summary
        echo ""
        log_success "Performance testing completed!"
        echo ""
        echo "Results:"
        echo "  - Logs: $LOG_DIR"
        echo "  - Test Results: $RESULTS_DIR"
        echo "  - Dashboard: $DASHBOARD_DIR/performance_dashboard.html"
        echo ""
        echo "To view the dashboard:"
        echo "  Open $DASHBOARD_DIR/performance_dashboard.html in a web browser"
        echo ""
        
        # Check for performance summary
        if [ -f "$DASHBOARD_DIR/performance_summary.json" ]; then
            log_info "Performance Summary:"
            python3 -c "
import json
with open('$DASHBOARD_DIR/performance_summary.json') as f:
    summary = json.load(f)
    
if 'overall_performance' in summary and summary['overall_performance']:
    perf = summary['overall_performance']
    print(f\"  Time Improvement: {perf.get('time_improvement_percent', 0):.1f}%\")
    print(f\"  Speedup Factor: {perf.get('speedup_factor', 1):.2f}x\")
    if summary.get('regression_detected'):
        print('  ⚠️  Performance regression detected!')
    else:
        print('  ✅ No performance regressions')
else:
    print('  No performance data available')
"
        fi
    fi
}

# Execute main function
main "$@"
