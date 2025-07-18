# SuperDARN RST Optimized Docker Quick Start (PowerShell)
# ========================================================
# Helper script for using the optimized Docker environment on Windows

param(
    [Parameter(Position=0)]
    [string]$Command = "",
    
    [switch]$NoCache,
    [switch]$Verbose,
    [switch]$FollowLogs,
    [switch]$Help
)

# Color output functions
function Write-Header {
    param([string]$Message)
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ INFO: $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️ WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ ERROR: $Message" -ForegroundColor Red
}

# Show help
function Show-Help {
    @"
SuperDARN RST Optimized Docker Quick Start (PowerShell)

USAGE:
    .\docker-quick-start.ps1 [command] [options]

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
    .\docker-quick-start.ps1 build                    # Build all images
    .\docker-quick-start.ps1 dev                      # Start development environment
    .\docker-quick-start.ps1 optimized               # Start optimized environment
    .\docker-quick-start.ps1 performance             # Run performance comparison
    .\docker-quick-start.ps1 validate                # Validate optimization system
    .\docker-quick-start.ps1 clean                   # Clean up containers and images

OPTIONS:
    -NoCache                   # Build without Docker cache
    -Verbose                   # Enable verbose output
    -FollowLogs               # Follow container logs
    -Help                     # Show this help

EXAMPLES WITH OPTIONS:
    .\docker-quick-start.ps1 build -NoCache
    .\docker-quick-start.ps1 logs -FollowLogs
"@
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed or not in PATH"
        exit 1
    }
    
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Compose is not installed or not in PATH"
        exit 1
    }
    
    if (-not (Test-Path "docker-compose.optimized.yml")) {
        Write-Error "docker-compose.optimized.yml not found in current directory"
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Build Docker images
function Build-Images {
    Write-Header "Building SuperDARN RST Optimized Docker Images"
    
    $buildArgs = @()
    if ($NoCache) {
        $buildArgs += "--no-cache"
        Write-Info "Building without cache..."
    }
    
    Write-Info "Building all stages of the optimized Docker environment..."
    
    try {
        & docker-compose -f docker-compose.optimized.yml build @buildArgs
        Write-Success "Docker images built successfully"
        
        # Show built images
        Write-Info "Available images:"
        & docker images | Where-Object { $_ -match "rst" }
    }
    catch {
        Write-Error "Failed to build Docker images: $_"
        exit 1
    }
}

# Start development environment
function Start-Development {
    Write-Header "Starting SuperDARN RST Development Environment"
    Write-Info "This environment includes both standard and optimized builds"
    Write-Info "Use 'switch-to-standard' and 'switch-to-optimized' to switch between builds"
    
    & docker-compose -f docker-compose.optimized.yml run --rm superdarn-dev
}

# Start optimized environment
function Start-Optimized {
    Write-Header "Starting SuperDARN RST Optimized Environment"
    Write-Info "This environment uses hardware-optimized RST build"
    
    & docker-compose -f docker-compose.optimized.yml run --rm superdarn-optimized
}

# Start standard environment
function Start-Standard {
    Write-Header "Starting SuperDARN RST Standard Environment"
    Write-Info "This environment uses standard RST build for comparison"
    
    & docker-compose -f docker-compose.optimized.yml run --rm superdarn-standard
}

# Run performance comparison
function Start-Performance {
    Write-Header "Running SuperDARN RST Performance Comparison"
    Write-Info "This will compare standard vs optimized build performance"
    Write-Info "Results will be saved to ./test-results/"
    
    # Ensure results directory exists
    if (-not (Test-Path "./test-results")) {
        New-Item -ItemType Directory -Path "./test-results" | Out-Null
    }
    
    Write-Info "Starting automated performance testing..."
    & docker-compose -f docker-compose.optimized.yml up --abort-on-container-exit superdarn-performance
    
    Write-Success "Performance comparison completed"
    Write-Info "Results available in ./test-results/"
    
    # Show results summary
    if (Test-Path "./test-results/optimization_comparison_dashboard.html") {
        Write-Success "Performance dashboard: ./test-results/optimization_comparison_dashboard.html"
    }
}

# Run benchmark tests
function Start-Benchmark {
    Write-Header "Running SuperDARN RST Benchmark Tests"
    Write-Info "This will run intensive performance benchmarks"
    
    if (-not (Test-Path "./test-results")) {
        New-Item -ItemType Directory -Path "./test-results" | Out-Null
    }
    
    & docker-compose -f docker-compose.optimized.yml up --abort-on-container-exit superdarn-benchmark
    
    Write-Success "Benchmark testing completed"
    
    if (Test-Path "./test-results/benchmark_report.html") {
        Write-Success "Benchmark report: ./test-results/benchmark_report.html"
    }
}

# Run CI tests
function Start-CI {
    Write-Header "Running SuperDARN RST CI Tests"
    Write-Info "Running validation tests suitable for CI/CD"
    
    & docker-compose -f docker-compose.optimized.yml run --rm superdarn-ci
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Success "All CI tests passed"
    } else {
        Write-Error "CI tests failed with exit code $exitCode"
        exit $exitCode
    }
}

# Validate optimization system
function Test-OptimizationSystem {
    Write-Header "Validating SuperDARN RST Optimization System"
    Write-Info "Running comprehensive validation of the optimization framework"
    
    & docker-compose -f docker-compose.optimized.yml run --rm superdarn-optimized check-optimization
    
    Write-Success "Optimization system validation completed"
}

# Clean up Docker resources
function Remove-DockerResources {
    Write-Header "Cleaning Up SuperDARN RST Docker Resources"
    
    Write-Info "Stopping and removing containers..."
    & docker-compose -f docker-compose.optimized.yml down
    
    Write-Info "Removing unused images and volumes..."
    & docker system prune -f
    
    Write-Success "Cleanup completed"
}

# Show container status
function Show-Status {
    Write-Header "SuperDARN RST Container Status"
    
    Write-Info "Running containers:"
    & docker-compose -f docker-compose.optimized.yml ps
    
    Write-Info "Docker images:"
    $images = & docker images | Where-Object { $_ -match "(rst|superdarn)" }
    if ($images) {
        $images
    } else {
        Write-Host "No RST images found"
    }
    
    Write-Info "Volume usage:"
    & docker system df
}

# Show container logs
function Show-Logs {
    param([string]$ServiceName)
    
    Write-Header "SuperDARN RST Container Logs"
    
    if (-not $ServiceName) {
        Write-Info "Available services:"
        & docker-compose -f docker-compose.optimized.yml config --services
        Write-Host ""
        $ServiceName = Read-Host "Enter service name (or 'all' for all services)"
    }
    
    if ($ServiceName -eq "all") {
        if ($FollowLogs) {
            & docker-compose -f docker-compose.optimized.yml logs -f
        } else {
            & docker-compose -f docker-compose.optimized.yml logs
        }
    } else {
        if ($FollowLogs) {
            & docker-compose -f docker-compose.optimized.yml logs -f $ServiceName
        } else {
            & docker-compose -f docker-compose.optimized.yml logs $ServiceName
        }
    }
}

# Main script logic
function Main {
    if ($Help -or $Command -eq "help" -or $Command -eq "" -or $Command -eq "--help") {
        Show-Help
        return
    }
    
    # Check prerequisites for most commands
    if ($Command -ne "help") {
        Test-Prerequisites
    }
    
    switch ($Command.ToLower()) {
        "build" {
            Build-Images
        }
        "dev" -or "development" {
            Start-Development
        }
        "optimized" -or "opt" {
            Start-Optimized
        }
        "standard" -or "std" {
            Start-Standard
        }
        "performance" -or "perf" {
            Start-Performance
        }
        "benchmark" -or "bench" {
            Start-Benchmark
        }
        "ci" {
            Start-CI
        }
        "validate" -or "val" {
            Test-OptimizationSystem
        }
        "clean" -or "cleanup" {
            Remove-DockerResources
        }
        "status" -or "ps" {
            Show-Status
        }
        "logs" {
            Show-Logs
        }
        default {
            Write-Error "Unknown command: $Command"
            Write-Host ""
            Show-Help
            exit 1
        }
    }
}

# Handle script interruption
trap {
    Write-Warning "Script interrupted"
    exit 130
}

# Run main function
Main
