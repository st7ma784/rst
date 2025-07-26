#!/bin/bash

# Default build type
BUILD_TYPE=Release
ENABLE_AVX=ON
ENABLE_CUDA=OFF
BUILD_TESTS=ON
BUILD_BENCHMARKS=OFF
INSTALL_PREFIX=/usr/local
NUM_JOBS=$(nproc)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --release)
            BUILD_TYPE=Release
            shift
            ;;
        --no-avx)
            ENABLE_AVX=OFF
            shift
            ;;
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --benchmarks)
            BUILD_BENCHMARKS=ON
            shift
            ;;
        --prefix=*)
            INSTALL_PREFIX="${1#*=}"
            shift
            ;;
        -j*)
            NUM_JOBS="${1#-j}"
            shift
            ;;
        --help)
            echo "Build script for fit.1.35 module"
            echo ""
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug           Build with debug symbols"
            echo "  --release         Build with optimizations (default)"
            echo "  --no-avx          Disable AVX/AVX2/AVX-512 optimizations"
            echo "  --cuda            Enable CUDA acceleration"
            echo "  --no-tests        Disable building tests"
            echo "  --benchmarks      Enable building benchmarks"
            echo "  --prefix=PATH     Installation prefix (default: /usr/local)"
            echo "  -jN               Number of parallel jobs (default: nproc)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Building fit.1.35 module ==="
echo "Build type: ${BUILD_TYPE}"
echo "AVX optimizations: ${ENABLE_AVX}"
echo "CUDA support: ${ENABLE_CUDA}"
echo "Build tests: ${BUILD_TESTS}"
echo "Build benchmarks: ${BUILD_BENCHMARKS}"
echo "Install prefix: ${INSTALL_PREFIX}"
echo "Parallel jobs: ${NUM_JOBS}"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_AVX=${ENABLE_AVX} \
    -DENABLE_CUDA=${ENABLE_CUDA} \
    -DBUILD_TESTS=${BUILD_TESTS} \
    -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}

# Build the project
cmake --build . -- -j${NUM_JOBS}

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "\n=== Build successful! ==="
    echo "To install: cmake --build . --target install"
    echo "To run tests: cmake --build . --target test"
    echo "To run benchmarks: cmake --build . --target run_benchmarks"
else
    echo "\n=== Build failed ==="
    exit 1
fi
