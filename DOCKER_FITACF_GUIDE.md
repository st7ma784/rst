# SuperDARN FitACF v3.0 Docker Testing Guide

This guide explains how to use the Docker environment for testing the FitACF v3.0 array implementation.

## Prerequisites

- Docker installed and running
- Docker Compose (optional, but recommended)
- Windows PowerShell or Linux/macOS terminal

## Quick Start

### Option 1: Automated Comprehensive Testing

Run the complete test suite with a single command:

```bash
./test_fitacf_docker_comprehensive.sh
```

This will:
- Build the Docker image with full RST environment
- Run tests with 1, 2, 4, and 8 threads
- Generate performance reports
- Provide an interactive testing environment

### Option 2: Using Docker Compose

```bash
# Interactive testing environment
docker-compose -f docker-compose.fitacf.yml run fitacf-test

# Automated test suite
docker-compose -f docker-compose.fitacf.yml run fitacf-autotest

# Performance benchmarking
docker-compose -f docker-compose.fitacf.yml run fitacf-benchmark
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build -f dockerfile.fitacf -t fitacf-test .

# Run interactive environment
docker run -it --rm \
  -v "$(pwd)/codebase/superdarn/src.lib/tk/fitacf_v3.0:/workspace/fitacf_v3.0" \
  -v "$(pwd)/test-results:/workspace/results" \
  -e OMP_NUM_THREADS=4 \
  fitacf-test

# Inside the container:
source /opt/rst/.profile.bash
cd /workspace/fitacf_v3.0/src
make -f makefile_standalone tests
./test_baseline
./test_comparison
./test_performance
```

## Docker Environment Features

### Base Image: Ubuntu 22.04 LTS
- Full development environment with GCC, Make, CMake
- OpenMP support for parallel processing
- Scientific libraries (GSL, FFTW, NetCDF, HDF5)
- Debugging tools (GDB, Valgrind)
- Performance analysis tools

### RST Environment
- Minimal RST environment variables set up
- Compatible makefile system
- Standard SuperDARN directory structure
- Proper include paths and library locations

### Environment Variables
- `RSTPATH=/opt/rst` - RST installation path
- `MAKECFG=/opt/rst/build/make/makecfg` - Build configuration
- `SYSTEM=linux` - Target system type
- `OMP_NUM_THREADS=4` - OpenMP thread count (configurable)

## Available Test Programs

### 1. test_baseline
**Purpose**: Validates basic FitACF functionality
**Usage**: `./test_baseline`
**Output**: Pass/fail status with timing information

### 2. test_comparison
**Purpose**: Compares array vs linked list implementations
**Usage**: `./test_comparison`
**Output**: Accuracy metrics and performance comparison

### 3. test_performance
**Purpose**: Comprehensive performance benchmarking
**Usage**: `./test_performance`
**Output**: Detailed timing, speedup ratios, and resource usage

## Testing Configurations

### Thread Scaling Tests
```bash
# Test with different thread counts
for threads in 1 2 4 8; do
  OMP_NUM_THREADS=$threads ./test_performance
done
```

### Memory Analysis
```bash
# Run with memory debugging
valgrind --tool=memcheck --leak-check=full ./test_comparison
```

### Performance Profiling
```bash
# Profile with gprof
gcc -pg [compile flags] ...
./test_performance
gprof test_performance gmon.out > profile.txt
```

## Results and Output

### Result Files Location
- Host directory: `./test-results/`
- Container directory: `/workspace/results/`

### Expected Output Files
- `*_baseline_*threads.txt` - Baseline test results
- `*_comparison_*threads.txt` - Implementation comparison
- `*_performance_*threads.txt` - Performance benchmarks

### Performance Metrics
- **Execution Time**: Processing time for different data sizes
- **Speedup Ratio**: Array implementation vs baseline
- **Thread Scaling**: Efficiency with increasing thread count
- **Memory Usage**: Peak memory consumption and efficiency

## Troubleshooting

### Common Issues

1. **Docker Build Fails**
   ```bash
   # Clean and rebuild
   docker system prune -f
   docker build --no-cache -f dockerfile.fitacf -t fitacf-test .
   ```

2. **OpenMP Not Working**
   ```bash
   # Check OpenMP in container
   echo '#include <omp.h>' | gcc -fopenmp -E - && echo "OpenMP available"
   ```

3. **Permission Errors**
   ```bash
   # Fix file permissions
   chmod -R 777 ./test-results/
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop -> Settings -> Resources -> Memory
   ```

### Debug Mode

Run tests with verbose output:
```bash
docker run -it --rm \
  -e DEBUG_ARRAY=1 \
  -e OMP_NUM_THREADS=1 \
  [other options] \
  fitacf-test
```

## Performance Expectations

### Typical Results
- **Single-threaded**: 1.2-1.5x speedup over linked lists
- **Multi-threaded (4 cores)**: 3-5x speedup
- **Multi-threaded (8 cores)**: 5-8x speedup
- **Memory efficiency**: 20-30% reduction in peak usage

### Success Criteria
- All tests pass without errors
- Array implementation produces results within tolerance of linked list version
- Significant speedup demonstrated with multiple threads
- No memory leaks detected

## Contributing

To add new tests or modify the environment:

1. Update test files in `codebase/superdarn/src.lib/tk/fitacf_v3.0/test/`
2. Modify `makefile_standalone` if needed
3. Test in Docker environment
4. Update this documentation

## Support

For issues with the Docker testing environment:
1. Check the troubleshooting section above
2. Review Docker logs: `docker logs [container_id]`
3. Verify host system requirements
4. Consult SuperDARN FitACF documentation
