# SuperDARN RST Optimized Docker Environment Guide

## Overview

The SuperDARN RST Optimized Docker Environment provides multiple containerized environments for building, testing, and comparing standard and optimized RST builds. This system enables comprehensive performance analysis and validation of optimization improvements.

## Available Docker Images

### 1. Standard RST (`rst_standard`)
- **Purpose**: Baseline RST build for comparison
- **Optimization**: None (standard -O2 compiler flags)
- **Use Case**: Reference implementation and compatibility testing

### 2. Optimized RST (`rst_optimized`) 
- **Purpose**: Hardware-optimized RST build
- **Optimization**: Auto-detected based on container hardware (opt1/opt2/opt3)
- **Features**: 
  - Dynamic module detection
  - OpenMP parallelization
  - SIMD/AVX2 instructions (where available)
  - Advanced compiler optimizations
- **Use Case**: Production performance and research workloads

### 3. Development Environment (`rst_development`)
- **Purpose**: Complete development environment with both builds
- **Features**: 
  - Both standard and optimized builds available
  - Build switching utilities
  - Development tools (gdb, valgrind, etc.)
  - Performance analysis tools
- **Use Case**: Development, debugging, and performance comparison

## Quick Start

### Build and Run Optimized RST
```bash
# Build and start optimized environment
docker-compose -f docker-compose.optimized.yml up --build superdarn-optimized

# Interactive session
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash
```

### Development Environment (Both Builds)
```bash
# Start development environment
docker-compose -f docker-compose.optimized.yml up --build superdarn-dev

# Access container
docker exec -it superdarn-dev bash
```

### Performance Testing and Validation
```bash
# Validate Docker optimization infrastructure
./docker_optimization_validator.sh

# Run comprehensive performance comparison
./docker_performance_tester.sh

# Quick performance test
./docker_performance_tester.sh --quick

# Memory usage analysis only
./docker_performance_tester.sh --memory
```

### Automated Performance Testing
```bash
# Run complete performance comparison
docker-compose -f docker-compose.optimized.yml up --build superdarn-performance

# View results
docker-compose -f docker-compose.optimized.yml logs superdarn-performance
```

## Available Services

### `superdarn-standard`
- **Description**: Standard RST build environment
- **Usage**: Baseline testing and compatibility validation
- **Command**: Interactive bash shell

### `superdarn-optimized`  
- **Description**: Optimized RST with auto-detected optimization level
- **Usage**: High-performance computing and research
- **Features**:
  - Hardware detection on startup
  - Optimization validation
  - Enhanced build tools

### `superdarn-dev`
- **Description**: Development environment with both builds
- **Usage**: Development, debugging, and build comparison
- **Special Commands**:
  - `switch-to-standard` - Switch to standard build
  - `switch-to-optimized` - Switch to optimized build  
  - `compare-builds` - Show build information
  - `check-optimization` - Validate optimization system

### `superdarn-performance`
- **Description**: Automated performance comparison testing
- **Usage**: CI/CD performance validation
- **Outputs**: 
  - Performance comparison dashboard
  - Detailed timing analysis
  - Memory usage reports

### `superdarn-ci`
- **Description**: Continuous integration testing
- **Usage**: Automated validation in CI/CD pipelines
- **Features**:
  - Quick validation tests
  - Build system verification
  - Exit with status codes

### `superdarn-benchmark`
- **Description**: Intensive benchmark testing
- **Usage**: Performance characterization and optimization validation
- **Features**:
  - Hardware optimization detection
  - Memory usage analysis
  - Comprehensive performance reports

## Environment Variables

### Optimization Control
```bash
# OpenMP Configuration
OMP_NUM_THREADS=4           # Number of parallel threads
OMP_SCHEDULE=dynamic        # Thread scheduling strategy
OMP_PROC_BIND=spread        # Processor binding strategy
OMP_PLACES=cores            # Thread placement strategy

# Build Type Identification
BUILD_TYPE=optimized        # Container type identifier
```

### RST Environment
```bash
# Core RST paths (automatically configured)
RSTPATH=/app/rst
BUILD=/app/rst/build
CODEBASE=/app/rst/codebase

# Data and table paths
MAPDATA=/app/rst/tables/general/map_data
SD_HDWPATH=/app/rst/tables/superdarn/hdw/
# ... (all standard RST environment variables)
```

## Usage Examples

### 1. Interactive Development Session
```bash
# Start development environment
docker-compose -f docker-compose.optimized.yml run superdarn-dev bash

# Inside container:
compare-builds                    # Show available builds
switch-to-optimized              # Use optimized build
make.code.optimized --help       # Show optimization options
make.code.optimized --auto-optimize  # Build with auto-detected optimization
```

### 2. Performance Testing Workflow
```bash
# Run automated performance comparison
docker-compose -f docker-compose.optimized.yml up superdarn-performance

# Check results
docker cp superdarn-performance:/workspace/results ./local-results
```

### 3. Build System Validation
```bash
# Validate optimization system
docker-compose -f docker-compose.optimized.yml run superdarn-optimized check-optimization

# Manual validation
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  cd /app/rst && ./validate_optimization_system.sh
"
```

### 4. Custom Optimization Testing
```bash
# Test specific optimization level
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  make.code.optimized -o opt1 -v lib
"

# Test with specific module pattern
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  make.code.optimized -o opt2 -p grid -v
"
```

## Volume Mounts

### Source Code (`./codebase`)
- **Mount**: `/workspace/codebase`
- **Purpose**: Live code editing and development
- **Usage**: Modify source code on host, rebuild in container

### Scripts (`./scripts`)
- **Mount**: `/workspace/scripts`
- **Purpose**: Testing and analysis scripts
- **Usage**: Performance testing and validation scripts

### Results (`./test-results`)
- **Mount**: `/workspace/results`
- **Purpose**: Test results and performance data
- **Usage**: Access test results from host system

## Performance Analysis

### Available Tools in Containers

#### Standard Development Tools
- `gdb` - GNU debugger
- `valgrind` - Memory error detection
- `perf` - Linux performance analysis
- `htop` - Process monitoring

#### RST-Specific Tools
- `make.code.optimized` - Enhanced build system
- `check-optimization` - Optimization validation
- `compare-builds` - Build comparison utility

#### Performance Scripts
- `superdarn_test_suite.sh` - Comprehensive test suite
- `test_fitacf_comprehensive.sh` - FitACF performance analysis
- `generate_optimization_dashboard.py` - Performance visualization

### Reading Performance Results

Performance test results are saved to `/workspace/results` and include:

- **Timing Analysis**: Execution time comparisons
- **Memory Usage**: Memory consumption analysis  
- **Optimization Reports**: Hardware utilization analysis
- **Dashboard Files**: HTML visualization reports

## Troubleshooting

### Common Issues

#### Container Build Failures
```bash
# Clean build (removes cached layers)
docker-compose -f docker-compose.optimized.yml build --no-cache superdarn-optimized

# Check build logs
docker-compose -f docker-compose.optimized.yml logs superdarn-optimized
```

#### Optimization Detection Problems
```bash
# Validate optimization system
docker-compose -f docker-compose.optimized.yml run superdarn-optimized check-optimization

# Manual hardware check
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  make.code.optimized --hardware-info
"
```

#### Environment Variable Issues
```bash
# Check RST environment
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  echo 'RSTPATH: $RSTPATH'
  echo 'BUILD: $BUILD'
  echo 'CODEBASE: $CODEBASE'
"
```

#### Performance Test Failures
```bash
# Run with verbose output
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  ./scripts/superdarn_test_suite.sh --verbose
"

# Check specific module
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  make.code.optimized --list-optimizations
"
```

### Debug Mode
```bash
# Start container with debug shell
docker-compose -f docker-compose.optimized.yml run --entrypoint bash superdarn-optimized

# Enable verbose optimization logging
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  make.code.optimized -o opt2 -v lib
"
```

## Best Practices

### Development Workflow
1. **Start with Development Environment**: Use `superdarn-dev` for interactive work
2. **Test Optimizations**: Use `make.code.optimized --auto-optimize` for initial testing
3. **Compare Performance**: Run `superdarn-performance` for comprehensive comparison
4. **Validate Changes**: Use `superdarn-ci` for automated validation

### Performance Testing
1. **Baseline First**: Always test standard build before optimized
2. **Multiple Runs**: Run tests multiple times for statistical accuracy
3. **Document Environment**: Record hardware specifications and optimization levels
4. **Monitor Resources**: Use container resource limits to ensure fair comparison

### Production Deployment
1. **Validate Optimization**: Run full validation suite before deployment
2. **Test Compatibility**: Verify optimized builds work with existing data
3. **Monitor Performance**: Track performance improvements in production
4. **Document Configuration**: Record optimization settings and hardware requirements

## Advanced Usage

### Custom Optimization Levels
```bash
# Build with specific optimization flags
docker-compose -f docker-compose.optimized.yml run superdarn-optimized bash -c "
  export OPTIMIZATION_FLAGS='-O3 -march=native -fopenmp -mavx2'
  make.code.optimized -v lib
"
```

### Multi-Container Performance Testing
```bash
# Run multiple optimization levels simultaneously
docker-compose -f docker-compose.optimized.yml up -d superdarn-standard superdarn-optimized

# Compare results
docker exec superdarn-standard bash -c "time make_grid test_data.dat"
docker exec superdarn-optimized bash -c "time make_grid test_data.dat"
```

### Integration with CI/CD
```bash
# Use in GitHub Actions or similar
docker-compose -f docker-compose.optimized.yml run superdarn-ci
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ All tests passed"
else
  echo "❌ Tests failed"
  exit $EXIT_CODE
fi
```

---

## Support and Resources

- **Docker Compose File**: `docker-compose.optimized.yml`
- **Dockerfile**: `dockerfile.optimized`
- **Validation Scripts**: `validate_optimization_system.sh`, `validate_optimization_system.ps1`
- **Documentation**: `ENHANCED_BUILD_SYSTEM_GUIDE.md`

For additional support, refer to the comprehensive build system documentation and validation scripts included in the repository.
