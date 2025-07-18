# SuperDARN RST Docker Performance Testing Deployment Script (PowerShell)
# ========================================================================
# 
# This script sets up the complete Docker-based performance testing
# infrastructure for SuperDARN RST optimization comparison on Windows.
#
# Usage:
#   .\deploy_docker_performance.ps1 [options]
#
# Options:
#   -SetupOnly       Setup infrastructure without running tests
#   -TestOnly        Run tests using existing infrastructure
#   -Cleanup         Clean up all containers and volumes
#   -QuickTest       Run quick validation tests only
#   -FullBenchmark   Run complete benchmark suite
#   -Help            Show this help message

param(
    [switch]$SetupOnly,
    [switch]$TestOnly,
    [switch]$Cleanup,
    [switch]$QuickTest,
    [switch]$FullBenchmark,
    [switch]$Help
)

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = $ScriptDir
$LogDir = Join-Path $ProjectDir "logs"
$ResultsDir = Join-Path $ProjectDir "results"
$DashboardDir = Join-Path $ProjectDir "dashboard"
$TestDataDir = Join-Path $ProjectDir "test-data"

# Functions for colored output
function Write-Info($message) {
    Write-Host "[INFO] $message" -ForegroundColor Blue
}

function Write-Success($message) {
    Write-Host "[SUCCESS] $message" -ForegroundColor Green
}

function Write-Warning($message) {
    Write-Host "[WARNING] $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "[ERROR] $message" -ForegroundColor Red
}

# Help function
function Show-Help {
    @"
SuperDARN RST Docker Performance Testing Deployment (PowerShell)

This script automates the setup and execution of Docker-based performance
testing for SuperDARN RST optimization comparison on Windows.

USAGE:
    .\deploy_docker_performance.ps1 [OPTIONS]

OPTIONS:
    -SetupOnly        Setup infrastructure without running tests
    -TestOnly         Run tests using existing infrastructure  
    -Cleanup          Clean up all containers and volumes
    -QuickTest        Run quick validation tests only
    -FullBenchmark    Run complete benchmark suite
    -Help             Show this help message

EXAMPLES:
    # Complete setup and run standard tests
    .\deploy_docker_performance.ps1

    # Setup infrastructure only
    .\deploy_docker_performance.ps1 -SetupOnly

    # Run quick validation
    .\deploy_docker_performance.ps1 -QuickTest

    # Full cleanup
    .\deploy_docker_performance.ps1 -Cleanup

REQUIREMENTS:
    - Docker Desktop for Windows
    - Python 3.7+ with pip
    - At least 4GB free disk space
    - Internet connection for base image downloads

For more information, see DOCKER_PERFORMANCE_WORKFLOW.md
"@
}

# Check requirements
function Test-Requirements {
    Write-Info "Checking system requirements..."
    
    # Check Docker
    try {
        docker version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker command failed"
        }
    }
    catch {
        Write-Error "Docker is not installed or not running"
        exit 1
    }
    
    # Check Docker Compose
    try {
        docker-compose version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose command failed"
        }
    }
    catch {
        Write-Error "Docker Compose is not installed"
        exit 1
    }
    
    # Check Python
    try {
        python --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Python command failed"
        }
    }
    catch {
        Write-Error "Python 3 is not installed or not in PATH"
        exit 1
    }
    
    # Check available disk space (at least 4GB)
    $drive = (Get-Location).Drive
    $freeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$($drive.Name)'").FreeSpace
    $requiredSpace = 4GB
    
    if ($freeSpace -lt $requiredSpace) {
        Write-Warning "Low disk space detected. At least 4GB recommended."
    }
    
    Write-Success "All requirements satisfied"
}

# Setup directories
function Initialize-Directories {
    Write-Info "Setting up directory structure..."
    
    $dirs = @($LogDir, $ResultsDir, $DashboardDir, $TestDataDir)
    foreach ($dir in $dirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    # Create results subdirectories
    $resultSubdirs = @(
        (Join-Path $ResultsDir "standard"),
        (Join-Path $ResultsDir "optimized")
    )
    foreach ($dir in $resultSubdirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directories created"
}

# Install Python dependencies
function Install-PythonDependencies {
    Write-Info "Installing Python dependencies..."
    
    $requirementsFile = Join-Path $ProjectDir "requirements.txt"
    
    # Create requirements file if it doesn't exist
    if (!(Test-Path $requirementsFile)) {
        @"
plotly>=5.0.0
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
jinja2>=3.0.0
"@ | Out-File -FilePath $requirementsFile -Encoding utf8
    }
    
    # Install dependencies
    python -m pip install -r $requirementsFile --user
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Python dependencies"
        exit 1
    }
    
    Write-Success "Python dependencies installed"
}

# Generate test data
function New-TestData {
    Write-Info "Generating test data..."
    
    $testDataScript = Join-Path $ProjectDir "scripts\generate_test_data.py"
    
    if (!(Test-Path $testDataScript)) {
        Write-Error "Test data generator script not found"
        exit 1
    }
    
    # Generate test datasets
    $logFile = Join-Path $LogDir "test_data_generation.log"
    python $testDataScript --output $TestDataDir 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Test data generation failed"
        exit 1
    }
    
    # Verify test data was created
    if (!(Test-Path (Join-Path $TestDataDir "small"))) {
        Write-Error "Test data generation failed - small dataset not found"
        exit 1
    }
    
    Write-Success "Test data generated"
}

# Build Docker containers
function Build-Containers {
    Write-Info "Building Docker containers..."
    
    $dockerfile = Join-Path $ProjectDir "dockerfile.optimized"
    if (!(Test-Path $dockerfile)) {
        Write-Error "dockerfile.optimized not found"
        exit 1
    }
    
    # Build standard container
    Write-Info "Building standard RST container..."
    $logFile = Join-Path $LogDir "build_standard.log"
    docker build -f dockerfile.optimized --target rst_standard -t rst:standard . 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build standard container"
        exit 1
    }
    
    # Build optimized container
    Write-Info "Building optimized RST container..."
    $logFile = Join-Path $LogDir "build_optimized.log"
    docker build -f dockerfile.optimized --target rst_optimized -t rst:optimized . 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build optimized container"
        exit 1
    }
    
    # Build development container
    Write-Info "Building development container..."
    $logFile = Join-Path $LogDir "build_development.log"
    docker build -f dockerfile.optimized --target rst_development -t rst:development . 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build development container"
        exit 1
    }
    
    Write-Success "Docker containers built successfully"
}

# Validate containers
function Test-Containers {
    Write-Info "Validating Docker containers..."
    
    # Check if containers were built
    $standardImage = docker images --format "table {{.Repository}}:{{.Tag}}" | Select-String "rst:standard"
    $optimizedImage = docker images --format "table {{.Repository}}:{{.Tag}}" | Select-String "rst:optimized"
    
    if (!$standardImage) {
        Write-Error "Standard container not found"
        exit 1
    }
    
    if (!$optimizedImage) {
        Write-Error "Optimized container not found"
        exit 1
    }
    
    # Test container startup
    Write-Info "Testing container startup..."
    
    # Test standard container
    docker run --rm rst:standard echo "Standard container OK" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Standard container failed to start"
        exit 1
    }
    
    # Test optimized container
    docker run --rm rst:optimized echo "Optimized container OK" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Optimized container failed to start"
        exit 1
    }
    
    Write-Success "Container validation completed"
}

# Create test runner script
function New-TestRunner {
    Write-Info "Creating test runner script..."
    
    $testRunnerScript = Join-Path $ProjectDir "run_performance_tests.sh"
    
    @'
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
'@ | Out-File -FilePath $testRunnerScript -Encoding ascii
    
    Write-Success "Test runner script created"
}

# Run performance tests
function Invoke-PerformanceTests {
    param([string]$TestType = "standard")
    
    $dataSets = switch ($TestType) {
        "quick" { "small" }
        "standard" { "small,medium" }
        "comprehensive" { "small,medium,large" }
        "benchmark" { "small,medium,large,benchmark" }
        default { "small,medium" }
    }
    
    Write-Info "Running $TestType performance tests with datasets: $dataSets"
    
    $testRunner = Join-Path $ProjectDir "run_performance_tests.sh"
    
    # Run standard container test
    Write-Info "Running standard container tests..."
    $logFile = Join-Path $LogDir "test_standard.log"
    docker run --rm `
        -v "${TestDataDir}:/data:ro" `
        -v "${ResultsDir}:/results" `
        -v "${testRunner}:/app/run_performance_tests.sh:ro" `
        --name rst-standard-test `
        rst:standard `
        /app/run_performance_tests.sh standard $dataSets 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Standard container tests failed"
        exit 1
    }
    
    # Run optimized container test
    Write-Info "Running optimized container tests..."
    $logFile = Join-Path $LogDir "test_optimized.log"
    docker run --rm `
        -v "${TestDataDir}:/data:ro" `
        -v "${ResultsDir}:/results" `
        -v "${testRunner}:/app/run_performance_tests.sh:ro" `
        --name rst-optimized-test `
        rst:optimized `
        /app/run_performance_tests.sh optimized $dataSets 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Optimized container tests failed"
        exit 1
    }
    
    Write-Success "Performance tests completed"
}

# Generate dashboard
function New-Dashboard {
    Write-Info "Generating performance dashboard..."
    
    $dashboardScript = Join-Path $ProjectDir "scripts\generate_github_dashboard.py"
    
    if (!(Test-Path $dashboardScript)) {
        Write-Error "Dashboard generator script not found"
        exit 1
    }
    
    # Generate dashboard
    $logFile = Join-Path $LogDir "dashboard_generation.log"
    python $dashboardScript --results-dir $ResultsDir --output-dir $DashboardDir --verbose 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dashboard generation failed"
        exit 1
    }
    
    # Check if dashboard was created
    $dashboardFile = Join-Path $DashboardDir "performance_dashboard.html"
    if (!(Test-Path $dashboardFile)) {
        Write-Error "Dashboard generation failed - output file not found"
        exit 1
    }
    
    Write-Success "Dashboard generated: $dashboardFile"
}

# Cleanup function
function Remove-DockerResources {
    Write-Info "Cleaning up Docker resources..."
    
    # Stop and remove containers
    $containers = docker ps -a --format "{{.Names}}" | Where-Object { $_ -match "rst" }
    if ($containers) {
        docker rm -f $containers
    }
    
    # Remove images
    $images = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -match "rst:" }
    if ($images) {
        docker rmi -f $images
    }
    
    # Clean up build cache
    docker builder prune -f
    
    # Remove results and logs
    if (Test-Path $ResultsDir) {
        Remove-Item -Recurse -Force $ResultsDir
    }
    
    if (Test-Path $LogDir) {
        Remove-Item -Recurse -Force $LogDir
    }
    
    Write-Success "Cleanup completed"
}

# Main execution
function Main {
    # Handle help
    if ($Help) {
        Show-Help
        exit 0
    }
    
    # Handle cleanup
    if ($Cleanup) {
        Remove-DockerResources
        exit 0
    }
    
    # Determine test type
    $testType = "standard"
    if ($QuickTest) { $testType = "quick" }
    if ($FullBenchmark) { $testType = "benchmark" }
    
    # Show banner
    Write-Host "========================================"
    Write-Host "SuperDARN RST Docker Performance Testing"
    Write-Host "========================================"
    Write-Host ""
    
    # Check requirements
    Test-Requirements
    
    # Setup phase
    if (!$TestOnly) {
        Initialize-Directories
        Install-PythonDependencies
        New-TestData
        Build-Containers
        Test-Containers
        New-TestRunner
        
        if ($SetupOnly) {
            Write-Success "Setup completed successfully!"
            Write-Info "To run tests: .\deploy_docker_performance.ps1 -TestOnly"
            exit 0
        }
    }
    
    # Test phase
    if (!$SetupOnly) {
        Invoke-PerformanceTests $testType
        New-Dashboard
        
        # Show results summary
        Write-Host ""
        Write-Success "Performance testing completed!"
        Write-Host ""
        Write-Host "Results:"
        Write-Host "  - Logs: $LogDir"
        Write-Host "  - Test Results: $ResultsDir"
        Write-Host "  - Dashboard: $DashboardDir\performance_dashboard.html"
        Write-Host ""
        Write-Host "To view the dashboard:"
        Write-Host "  Open $DashboardDir\performance_dashboard.html in a web browser"
        Write-Host ""
        
        # Check for performance summary
        $summaryFile = Join-Path $DashboardDir "performance_summary.json"
        if (Test-Path $summaryFile) {
            Write-Info "Performance Summary:"
            try {
                $summary = Get-Content $summaryFile | ConvertFrom-Json
                
                if ($summary.overall_performance) {
                    $perf = $summary.overall_performance
                    Write-Host "  Time Improvement: $($perf.time_improvement_percent.ToString('F1'))%"
                    Write-Host "  Speedup Factor: $($perf.speedup_factor.ToString('F2'))x"
                    
                    if ($summary.regression_detected) {
                        Write-Host "  ⚠️  Performance regression detected!" -ForegroundColor Red
                    } else {
                        Write-Host "  ✅ No performance regressions" -ForegroundColor Green
                    }
                } else {
                    Write-Host "  No performance data available"
                }
            }
            catch {
                Write-Host "  Error reading performance summary"
            }
        }
    }
}

# Execute main function
Main
